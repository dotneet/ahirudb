//! Row key normalization and the hash index.
//!
//! Aggregation (GROUP BY) and joins (equi-joins) both "look things up by a key made of several
//! columns' values". **Implementing them separately would let the details of equality (NULL
//! handling, -0.0, NaN) drift apart and break silently**, so they are unified in one place.
//! There is no reason to carry two, code size included.
//!
//! Keys are **for equality only** and carry no ordering. Sorting compares values directly.

use crate::prelude::*;
use crate::rt::hash::hash_u64;
use crate::vector::{Data, Ty, Vector};

/// Writes one row's key into `out`.
///
/// Each column is "a 1-byte presence flag + the value". NULL is just flag 0 with no value.
/// Under SQL's `=`, NULL does not equal NULL, but **under GROUP BY and DISTINCT they land in
/// the same group**. This function encodes with the latter semantics, so the join side must
/// reject NULL-key rows itself before feeding them in.
pub fn encode_key(cols: &[&Vector], row: usize, out: &mut Vec<u8>) {
    out.clear();
    for c in cols {
        if !c.is_valid(row) {
            out.push(0);
            continue;
        }
        out.push(1);
        match c.data() {
            Data::Bool(b) => out.push(b.get(row) as u8),
            Data::I32(v) => out.extend_from_slice(&v[row].to_le_bytes()),
            Data::I64(v) => out.extend_from_slice(&v[row].to_le_bytes()),
            // INTERVAL's three packed components are flattened first: `1 day` and `24 hours`
            // are the same value and must land in the same group / join bucket.
            Data::I128(v) if c.ty() == Ty::Interval => {
                out.extend_from_slice(&interval_key(v[row]).to_le_bytes())
            }
            Data::I128(v) => out.extend_from_slice(&v[row].to_le_bytes()),
            Data::F64(v) => out.extend_from_slice(&canonical_f64(v[row]).to_le_bytes()),
            Data::Bytes(b) => {
                let s = b.get(row);
                out.extend_from_slice(&(s.len() as u32).to_le_bytes());
                out.extend_from_slice(s);
            }
        }
    }
}

/// Normalizes a floating-point bit representation.
///
/// `-0.0` and `0.0` are the same value and belong in the same group, and all NaNs collapse into
/// one (the same behavior as DuckDB). Using the raw bits as the key would split both of those
/// into separate groups.
///
/// This is the same equality `expr::kernels::cmp_f64` implements for `=` — floating point
/// compares under a total order there, so an equi-join and the `=` predicate the planner might
/// use instead can never disagree about a NaN key.
#[inline]
pub fn canonical_f64(v: f64) -> u64 {
    if v.is_nan() {
        0x7ff8_0000_0000_0000
    } else if v == 0.0 {
        0
    } else {
        v.to_bits()
    }
}

/// Normalizes an INTERVAL into a single comparable/hashable microsecond count.
///
/// INTERVAL is stored as three independent components packed into one i128
/// (`vector::pack_interval`: months, days, microseconds). Comparing that bit pattern directly
/// makes `INTERVAL 1 DAY` and `INTERVAL 24 HOUR` different keys even though they denote the same
/// span, so an equi-join on them found nothing, `UNION` kept both, and `ORDER BY` ranked
/// `1 day` above `25 hours` (the months field dominates the high bits).
///
/// DuckDB (and PostgreSQL) compare intervals by flattening them with the fixed conversions
/// **1 month = 30 days** and **1 day = 24 hours**, and that is what this reproduces. The
/// conversions are deliberately calendar-independent: interval comparison has no anchor date, so
/// there is nothing to ask how long "one month" really is. Adding an interval to a timestamp
/// still uses real calendar arithmetic (`expr::funcs::add_interval_to_ts`) -- only comparison
/// normalizes.
///
/// The result cannot overflow: the widest input (`i32::MAX` months + `i32::MAX` days +
/// `i64::MIN` microseconds) stays below 6e24, far inside i128.
#[inline]
pub fn interval_key(v: i128) -> i128 {
    const US_PER_DAY: i128 = 86_400_000_000;
    let (months, days, micros) = crate::vector::unpack_interval(v);
    (months as i128) * 30 * US_PER_DAY + (days as i128) * US_PER_DAY + micros as i128
}

/// Whether the key contains even one NULL. In an equi-join a NULL key never matches, so this is
/// used to reject rows before insertion and probing.
pub fn key_has_null(cols: &[&Vector], row: usize) -> bool {
    cols.iter().any(|c| !c.is_valid(row))
}

/// 10^scale. `f64::powi` is not in core, so it is built by multiplication.
/// Shared by the DECIMAL <-> f64 conversions in `exec::agg`/`exec::window`.
pub fn pow10(scale: u8) -> f64 {
    let mut d = 1.0f64;
    for _ in 0..scale {
        d *= 10.0;
    }
    d
}

/// A total order treating NaN as "greater than everything".
///
/// That way MAX returns NaN only when there is no other value, and MIN prefers anything but NaN
/// (only an all-NaN group gives NaN). It does not contradict `encode_key`/`canonical_f64`
/// collapsing NaN into one group.
/// Shared by the MIN/MAX-family aggregates in `exec::agg`/`exec::window`, and by
/// `expr::kernels::cmp_f64`, which is what makes `=` agree with an equi-join on a NaN key.
#[inline]
pub fn ord_f64(a: f64, b: f64) -> core::cmp::Ordering {
    use core::cmp::Ordering::*;
    if a < b {
        Less
    } else if a > b {
        Greater
    } else if a == b {
        Equal
    } else {
        match (a.is_nan(), b.is_nan()) {
            (true, true) => Equal,
            (true, false) => Greater,
            _ => Less,
        }
    }
}

/// An open-addressing table mapping byte-sequence keys to `u32` values.
///
/// The keys themselves are concatenated into a single arena. A `Vec` per entry would make
/// allocations grow in proportion to the row count.
pub struct HashIndex {
    /// The arena of concatenated keys.
    keys: Vec<u8>,
    /// `(key start, key length, value)`
    entries: Vec<(u32, u32, u32)>,
    /// Each entry's hash value. Saves rereading keys when rehashing.
    hashes: Vec<u64>,
    /// The buckets. 0 is empty; anything else is an index into `entries` plus 1.
    buckets: Vec<u32>,
    mask: usize,
}

/// The buckets' initial capacity (a power of two).
const INITIAL_BUCKETS: usize = 1024;
/// Doubles once this load factor is exceeded.
const LOAD_NUM: usize = 7;
const LOAD_DEN: usize = 10;

impl HashIndex {
    pub fn new() -> Self {
        HashIndex {
            keys: Vec::new(),
            entries: Vec::new(),
            hashes: Vec::new(),
            buckets: vec![0; INITIAL_BUCKETS],
            mask: INITIAL_BUCKETS - 1,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The total bytes of keys held. Used for the memory cap check.
    pub fn key_bytes(&self) -> usize {
        self.keys.len()
    }

    /// The rough byte usage for the memory cap check. Adds a fixed per-entry overhead (the
    /// bucket, the hash, and the entry tuple, estimated at 32 bytes) to the keys themselves
    /// (`key_bytes`). The shared estimation formula used by every operator's `Oom` check.
    pub fn approx_bytes(&self) -> usize {
        const ENTRY_OVERHEAD: usize = 32;
        self.key_bytes() + self.len() * ENTRY_OVERHEAD
    }

    #[inline]
    fn hash(key: &[u8]) -> u64 {
        crate::rt::hash::hash_bytes(key)
    }

    #[inline]
    fn entry_key(&self, i: usize) -> &[u8] {
        let (off, len, _) = self.entries[i];
        &self.keys[off as usize..(off + len) as usize]
    }

    /// Finds, by linear probing, an empty slot or the slot of a matching entry.
    #[inline]
    fn probe(&self, key: &[u8], h: u64) -> (usize, Option<usize>) {
        let mut slot = (hash_u64(h) as usize) & self.mask;
        loop {
            let b = self.buckets[slot];
            if b == 0 {
                return (slot, None);
            }
            let i = (b - 1) as usize;
            if self.hashes[i] == h && self.entry_key(i) == key {
                return (slot, Some(i));
            }
            slot = (slot + 1) & self.mask;
        }
    }

    /// For aggregation. If the key is absent, inserts "the next slot number" as the value.
    /// Returns `(value, whether it was newly inserted)`.
    pub fn get_or_insert(&mut self, key: &[u8]) -> (u32, bool) {
        let h = Self::hash(key);
        let (slot, hit) = self.probe(key, h);
        if let Some(i) = hit {
            return (self.entries[i].2, false);
        }
        let value = self.entries.len() as u32;
        self.insert_at(slot, key, h, value);
        (value, true)
    }

    /// Lookup only. `None` if absent.
    pub fn lookup(&self, key: &[u8]) -> Option<u32> {
        let h = Self::hash(key);
        let (_, hit) = self.probe(key, h);
        hit.map(|i| self.entries[i].2)
    }

    /// For a join's build side. The same key may appear several times.
    ///
    /// It returns the existing value while replacing it with the new one, so the caller can build
    /// the chain as `next[value] = the returned value`.
    pub fn insert_chained(&mut self, key: &[u8], value: u32) -> Option<u32> {
        let h = Self::hash(key);
        let (slot, hit) = self.probe(key, h);
        match hit {
            Some(i) => {
                let prev = self.entries[i].2;
                self.entries[i].2 = value;
                Some(prev)
            }
            None => {
                self.insert_at(slot, key, h, value);
                None
            }
        }
    }

    fn insert_at(&mut self, slot: usize, key: &[u8], h: u64, value: u32) {
        let off = self.keys.len() as u32;
        self.keys.extend_from_slice(key);
        self.entries.push((off, key.len() as u32, value));
        self.hashes.push(h);
        self.buckets[slot] = self.entries.len() as u32;
        if self.entries.len() * LOAD_DEN > self.buckets.len() * LOAD_NUM {
            self.grow();
        }
    }

    fn grow(&mut self) {
        let n = self.buckets.len() * 2;
        self.buckets.clear();
        self.buckets.resize(n, 0);
        self.mask = n - 1;
        for i in 0..self.entries.len() {
            let mut slot = (hash_u64(self.hashes[i]) as usize) & self.mask;
            while self.buckets[slot] != 0 {
                slot = (slot + 1) & self.mask;
            }
            self.buckets[slot] = (i + 1) as u32;
        }
    }
}

impl Default for HashIndex {
    fn default() -> Self {
        HashIndex::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::{Ty, Value, Vector};

    fn vec_i32(vals: &[Option<i32>]) -> Vector {
        let mut v = Vector::new(Ty::Int);
        for x in vals {
            match x {
                Some(x) => v.push_value(&Value::I32(*x)),
                None => v.push_null(),
            }
        }
        v
    }

    fn vec_str(vals: &[Option<&str>]) -> Vector {
        let mut v = Vector::new(Ty::Varchar);
        for x in vals {
            match x {
                Some(x) => v.push_value(&Value::Bytes(x.as_bytes().to_vec())),
                None => v.push_null(),
            }
        }
        v
    }

    fn key(cols: &[&Vector], row: usize) -> Vec<u8> {
        let mut out = Vec::new();
        encode_key(cols, row, &mut out);
        out
    }

    #[test]
    fn equal_values_encode_equally() {
        let a = vec_i32(&[Some(1), Some(1), Some(2)]);
        assert_eq!(key(&[&a], 0), key(&[&a], 1));
        assert_ne!(key(&[&a], 0), key(&[&a], 2));
    }

    #[test]
    fn nulls_group_together_and_differ_from_values() {
        let a = vec_i32(&[None, None, Some(0)]);
        assert_eq!(key(&[&a], 0), key(&[&a], 1));
        assert_ne!(key(&[&a], 0), key(&[&a], 2));
    }

    #[test]
    fn multi_column_keys_are_not_confusable() {
        // ("a", "bc") and ("ab", "c") must not give the same key.
        // Without a length prefix they would collide.
        let a1 = vec_str(&[Some("a")]);
        let b1 = vec_str(&[Some("bc")]);
        let a2 = vec_str(&[Some("ab")]);
        let b2 = vec_str(&[Some("c")]);
        assert_ne!(key(&[&a1, &b1], 0), key(&[&a2, &b2], 0));
    }

    #[test]
    fn negative_zero_and_nan_are_canonicalised() {
        let mut v = Vector::new(Ty::Double);
        v.push_value(&Value::F64(0.0));
        v.push_value(&Value::F64(-0.0));
        v.push_value(&Value::F64(f64::NAN));
        v.push_value(&Value::F64(-f64::NAN));
        assert_eq!(key(&[&v], 0), key(&[&v], 1), "-0.0 groups with 0.0");
        assert_eq!(key(&[&v], 2), key(&[&v], 3), "NaN forms a single group");
        assert_ne!(key(&[&v], 0), key(&[&v], 2));
    }

    // DECIMAL(precision>18)/HUGEINT/UBIGINT/INTERVAL all share the physical representation
    // `Data::I128` (`Ty::phys`). Encoding itself takes the same code path for every logical
    // type, so checking at the physical-type level is enough.
    fn vec_i128(ty: Ty, vals: &[Option<i128>]) -> Vector {
        let mut v = Vector::new(ty);
        for x in vals {
            match x {
                Some(x) => v.push_value(&Value::I128(*x)),
                None => v.push_null(),
            }
        }
        v
    }

    #[test]
    fn i128_keys_distinguish_sign_and_magnitude() {
        let a = vec_i128(
            Ty::HugeInt,
            &[Some(1), Some(1), Some(-1), Some(i128::MAX), Some(i128::MIN), Some(0)],
        );
        // Equal values give equal keys.
        assert_eq!(key(&[&a], 0), key(&[&a], 1));
        // Differing signs and boundary values all give different keys (checked exhaustively).
        let n = 6;
        for i in 0..n {
            for j in 0..n {
                if (i == 0 && j == 1) || (i == 1 && j == 0) {
                    continue; // excluded, since vals[0] == vals[1] == 1 (both directions).
                }
                if i != j {
                    assert_ne!(
                        key(&[&a], i),
                        key(&[&a], j),
                        "rows {i} and {j} have different values yet their keys collided"
                    );
                }
            }
        }
    }

    #[test]
    fn i128_null_is_distinguished_from_zero() {
        let a = vec_i128(Ty::HugeInt, &[None, None, Some(0)]);
        assert_eq!(key(&[&a], 0), key(&[&a], 1), "NULLs group together");
        assert_ne!(key(&[&a], 0), key(&[&a], 2), "NULL and 0 are separate groups");
    }

    #[test]
    fn i128_composite_key_with_null_in_another_column() {
        // With a composite key of a DECIMAL(38,0) column and an INTEGER column, confirms that
        // equal DECIMAL values still form separate groups by the other column's NULL/non-NULL.
        let dec = vec_i128(Ty::decimal(38, 0), &[Some(100), Some(100)]);
        let other = vec_i32(&[None, Some(0)]);
        assert_ne!(key(&[&dec, &other], 0), key(&[&dec, &other], 1));
    }

    #[test]
    fn key_null_detection() {
        let a = vec_i32(&[Some(1), None]);
        let b = vec_i32(&[Some(1), Some(1)]);
        assert!(!key_has_null(&[&a, &b], 0));
        assert!(key_has_null(&[&a, &b], 1));
    }

    #[test]
    fn get_or_insert_assigns_sequential_slots() {
        let mut h = HashIndex::new();
        assert_eq!(h.get_or_insert(b"a"), (0, true));
        assert_eq!(h.get_or_insert(b"b"), (1, true));
        assert_eq!(h.get_or_insert(b"a"), (0, false));
        assert_eq!(h.len(), 2);
        assert_eq!(h.lookup(b"b"), Some(1));
        assert_eq!(h.lookup(b"zzz"), None);
    }

    #[test]
    fn growth_preserves_all_entries() {
        let mut h = HashIndex::new();
        // Feed in far more than the initial capacity to trigger a rehash.
        let n = 5000u32;
        for i in 0..n {
            let k = i.to_le_bytes();
            assert_eq!(h.get_or_insert(&k), (i, true));
        }
        assert_eq!(h.len(), n as usize);
        for i in 0..n {
            let k = i.to_le_bytes();
            assert_eq!(h.lookup(&k), Some(i), "{i} cannot be looked up after rehashing");
        }
    }

    #[test]
    fn chained_insert_returns_previous_head() {
        let mut h = HashIndex::new();
        assert_eq!(h.insert_chained(b"k", 0), None);
        assert_eq!(h.insert_chained(b"k", 1), Some(0));
        assert_eq!(h.insert_chained(b"k", 2), Some(1));
        // The last one inserted comes first.
        assert_eq!(h.lookup(b"k"), Some(2));
        assert_eq!(h.len(), 1);
    }

    #[test]
    fn empty_key_is_a_valid_key() {
        // An aggregate without GROUP BY forms one group under the empty key.
        let mut h = HashIndex::new();
        assert_eq!(h.get_or_insert(b""), (0, true));
        assert_eq!(h.get_or_insert(b""), (0, false));
    }

    #[test]
    fn intervals_of_equal_span_share_a_key() {
        // The packed representation differs (days vs microseconds), the normalized span does
        // not: duckdb treats 1 day and 24 hours -- and 1 month and 30 days -- as equal.
        let day = crate::vector::pack_interval(0, 1, 0);
        let hours24 = crate::vector::pack_interval(0, 0, 24 * 3_600_000_000);
        assert_ne!(day, hours24, "the packed forms are genuinely different");
        assert_eq!(interval_key(day), interval_key(hours24));
        assert_eq!(
            interval_key(crate::vector::pack_interval(1, 0, 0)),
            interval_key(crate::vector::pack_interval(0, 30, 0))
        );
        // 23 hours < 1 day < 25 hours, which the raw bit pattern gets wrong.
        let h23 = interval_key(crate::vector::pack_interval(0, 0, 23 * 3_600_000_000));
        let h25 = interval_key(crate::vector::pack_interval(0, 0, 25 * 3_600_000_000));
        assert!(h23 < interval_key(day) && interval_key(day) < h25);
        // Negative components stay ordered, and the extremes do not overflow i128.
        assert!(interval_key(crate::vector::pack_interval(0, -1, 0)) < 0);
        let _ = interval_key(crate::vector::pack_interval(i32::MAX, i32::MAX, i64::MAX));
        let _ = interval_key(crate::vector::pack_interval(i32::MIN, i32::MIN, i64::MIN));
    }

    #[test]
    fn interval_columns_encode_the_normalized_span() {
        let mut v = Vector::new(Ty::Interval);
        v.push_value(&Value::I128(crate::vector::pack_interval(0, 1, 0)));
        v.push_value(&Value::I128(crate::vector::pack_interval(0, 0, 24 * 3_600_000_000)));
        v.push_value(&Value::I128(crate::vector::pack_interval(0, 0, 25 * 3_600_000_000)));
        assert_eq!(key(&[&v], 0), key(&[&v], 1));
        assert_ne!(key(&[&v], 0), key(&[&v], 2));
    }
}
