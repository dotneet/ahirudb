//! Hash aggregation (GROUP BY / HAVING).
//!
//! Group key encoding and the hash table come from `exec::rowkey`. Divergent equality between
//! aggregation and joins would break things silently, and there is no reason to keep two tables.
//!
//! ## Blocking and resumption
//!
//! Aggregation cannot emit a single row until the input is fully read. Meanwhile a scan can
//! interrupt by returning `Step::NeedIo` / `Step::NeedCodec`, so **the partially built state is
//! kept, the interruption is passed straight up, and the next `next()` continues from there**.
//! The unit of ingestion is one batch, which is either "fully taken in" or "not taken in at
//! all" (`consume` never bails out midway). That structurally eliminates dropped and
//! double-counted rows on resumption.
//!
//! ## Overflow
//!
//! There is no spilling (external aggregation). Exceeding the hash table's cap returns `Oom`.
//! Carrying a sorted-run merge as well within a 1 MiB binary budget does not pay, so this is
//! accepted as a known limitation.

use crate::exec::rowkey::{encode_key, ord_f64, pow10, HashIndex};
use crate::exec::{ExecContext, Operator, Step};
use crate::expr::Program;
use crate::plan::{Agg, AggKind};
use crate::prelude::*;
use crate::vector::{Batch, Data, PhysType, Ty, Value, Vector, BATCH_SIZE};

/// The approximate byte budget allowed for the hash table and states. Exceeding it gives `Oom`.
///
/// The WASM linear memory the host provides is a 32-bit space, and during execution the scan
/// buffers and output batch share it. 64 MiB is a value that keeps aggregation from eating it all.
const MAX_STATE_BYTES: usize = 64 << 20;

/// The update rule chosen at runtime. Decided once from `AggKind` and the input's physical type.
///
/// No dedicated variant exists for `ApproxCountDistinct`. An exact count is fine for v1, and
/// that is exactly the same operation as "a `Count` with DISTINCT deduplication forced on its
/// argument", so it rides `Op::Count` directly
/// (see how the `distinct` array is built in `HashAggregate::new`).
#[derive(Clone, Copy, PartialEq)]
enum Op {
    CountStar,
    Count,
    /// SUM for integers and DECIMAL. Accumulated in i128.
    SumInt,
    SumF64,
    /// AVG for integers and DECIMAL. Accumulated in i128 and divided into f64 on output.
    AvgInt,
    AvgF64,
    Min,
    Max,
    /// Sample standard deviation and sample variance. Computed with Welford's online algorithm
    /// (updating the mean and the sum of squared deviations M2 in one pass). The naive
    /// "sum of squares minus square of sum" formula loses too many digits to trust on real data, so it is not used.
    StdDev,
    Variance,
    /// The continuous-distribution median (the same as `quantile_cont(x, 0.5)`, linear
    /// interpolation). An exact value cannot be computed while streaming, so every non-NULL
    /// value is retained and sorted exactly once on output.
    Median,
    /// The mode. Counted with a frequency table keyed by (group, value); on a tie the one found
    /// first wins (a `>` test for updating makes that happen naturally).
    Mode,
    /// String aggregation concatenating with a separator. Concatenated in arrival order
    /// (neither the SQL standard nor DuckDB guarantees an order without `ORDER BY`).
    StringAgg,
    /// Collects values into JSON-like text. A substitute representation, since there is no LIST
    /// type (the same judgment as DESIGN.md's handling of nested types). It differs from the
    /// other aggregates only in including NULL as an element (the same as DuckDB's `array_agg`/`list`).
    ArrayAgg,
}

/// The state of one group x one aggregate.
struct State {
    /// The count of non-NULL inputs. For `COUNT(*)`, all rows. For ArrayAgg, NULLs are counted
    /// as elements too, so it means "how many elements were pushed".
    n: i64,
    /// The accumulated value. `Value::Null` means "no non-NULL input yet".
    /// For StringAgg/ArrayAgg it doubles as the text being accumulated (`Value::Bytes`).
    acc: Value,
    /// The Welford accumulator for StdDev/Variance (the online mean). Left at 0.0 for other operations.
    mean: f64,
    /// The Welford accumulator for StdDev/Variance (the sum of squared deviations, M2).
    m2: f64,
    /// The non-NULL values Median holds until output.
    median_vals: Vec<f64>,
    /// Mode's current highest vote count.
    mode_best: i64,
}

impl State {
    fn empty() -> Self {
        State { n: 0, acc: Value::Null, mean: 0.0, m2: 0.0, median_vals: Vec::new(), mode_best: 0 }
    }
}

enum Phase {
    Building,
    Emitting,
    Done,
}

pub struct HashAggregate {
    input: Box<dyn Operator>,
    groups: Vec<Program>,
    aggs: Vec<Agg>,
    having: Option<Program>,

    /// The update rule per aggregate. In the same order as `aggs`.
    ops: Vec<Op>,
    /// The output type per aggregate. **Always derived from `Agg::result_ty()`** (the same function as the binder's).
    out_tys: Vec<Ty>,
    /// 10^scale for AVG(DECIMAL). 1.0 otherwise.
    avg_div: Vec<f64>,

    /// Group key -> group number. Numbers are assigned consecutively from 0.
    index: HashIndex,
    /// The group keys themselves, retained for output. In the same order as `groups`, one row per group.
    key_cols: Vec<Vector>,
    /// The state column per aggregate. `states[agg][group]`.
    states: Vec<Vec<State>>,
    /// The deduplication set for DISTINCT aggregates. The key is "group number ++ value".
    /// `ApproxCountDistinct` always keeps a deduplication table here regardless of whether
    /// DISTINCT was specified (the v1 implementation is an exact count, so this is the substance itself).
    distinct: Vec<Option<HashIndex>>,
    /// Mode's (group, value) -> frequency table index. The same "prefix the group number" scheme
    /// as the DISTINCT deduplication table, with one table per aggregate shared across all
    /// groups (like `distinct`, in the same order as `aggs`).
    mode_freq: Vec<Option<HashIndex>>,
    /// The occurrence count per `mode_freq` index.
    mode_counts: Vec<Vec<i64>>,

    /// The total bytes of variable-length growing state: MIN/MAX's winning byte sequence,
    /// Mode's winning byte sequence, StringAgg/ArrayAgg's accumulated text, Median's temporary
    /// values, and so on. Included in the memory check.
    acc_bytes: usize,
    /// The approximate byte count of the group key columns themselves.
    key_bytes: usize,

    phase: Phase,
    /// The next group number to output.
    emit_pos: usize,
}

impl HashAggregate {
    pub fn new(
        input: Box<dyn Operator>,
        groups: Vec<Program>,
        aggs: Vec<Agg>,
        having: Option<Program>,
    ) -> Result<Self> {
        let mut ops = Vec::with_capacity(aggs.len());
        let mut out_tys = Vec::with_capacity(aggs.len());
        let mut avg_div = Vec::with_capacity(aggs.len());
        let mut states = Vec::with_capacity(aggs.len());
        let mut distinct = Vec::with_capacity(aggs.len());
        let mut mode_freq = Vec::with_capacity(aggs.len());
        let mut mode_counts = Vec::with_capacity(aggs.len());
        for a in &aggs {
            let ity = a.input_ty();
            // Type checking is finished here. The output type always comes from the same function as the binder's.
            out_tys.push(a.result_ty()?);
            let float = ity.phys() == PhysType::F64;
            ops.push(match a.kind {
                AggKind::CountStar => Op::CountStar,
                AggKind::Count => Op::Count,
                AggKind::Sum => {
                    if float {
                        Op::SumF64
                    } else {
                        Op::SumInt
                    }
                }
                AggKind::Avg => {
                    if float {
                        Op::AvgF64
                    } else {
                        Op::AvgInt
                    }
                }
                AggKind::Min => Op::Min,
                AggKind::Max => Op::Max,
                AggKind::StdDev => Op::StdDev,
                AggKind::Variance => Op::Variance,
                AggKind::Median => Op::Median,
                AggKind::Mode => Op::Mode,
                // The `distinct` construction below forcibly gives it a deduplication table, so
                // the operation itself can be the same `Op::Count` as COUNT(DISTINCT x).
                AggKind::ApproxCountDistinct => Op::Count,
                AggKind::StringAgg => Op::StringAgg,
                AggKind::ArrayAgg => Op::ArrayAgg,
            });
            // DECIMAL is internally an integer, so AVG/StdDev/Variance/Median divide back by
            // 10^scale.
            avg_div.push(match ity {
                Ty::Decimal { scale, .. } => pow10(scale),
                _ => 1.0,
            });
            states.push(Vec::new());
            // `COUNT(*)` never carries DISTINCT (it has no argument).
            // ApproxCountDistinct always deduplicates whether or not the author wrote DISTINCT
            // (that is its definition).
            distinct.push(
                if (a.distinct || a.kind == AggKind::ApproxCountDistinct) && a.arg.is_some() {
                    Some(HashIndex::new())
                } else {
                    None
                },
            );
            mode_freq.push(if a.kind == AggKind::Mode { Some(HashIndex::new()) } else { None });
            mode_counts.push(Vec::new());
        }
        let key_cols = groups.iter().map(|g| Vector::new(g.result_ty)).collect();
        Ok(HashAggregate {
            input,
            groups,
            aggs,
            having,
            ops,
            out_tys,
            avg_div,
            index: HashIndex::new(),
            key_cols,
            states,
            distinct,
            mode_freq,
            mode_counts,
            acc_bytes: 0,
            key_bytes: 0,
            phase: Phase::Building,
            emit_pos: 0,
        })
    }

    fn num_groups(&self) -> usize {
        self.index.len()
    }

    /// The approximate memory usage. An exact value is unnecessary (it is used only for the cap check).
    fn mem_used(&self) -> usize {
        let mut n = self.index.approx_bytes();
        n += self.key_bytes + self.acc_bytes;
        n += self.num_groups() * self.aggs.len() * core::mem::size_of::<State>();
        for d in self.distinct.iter().flatten() {
            n += d.approx_bytes();
        }
        for d in self.mode_freq.iter().flatten() {
            n += d.approx_bytes();
        }
        n
    }

    /// Takes in one whole batch. **It never bails out midway** (the unit of resumption is a batch).
    fn consume(&mut self, ctx: &mut ExecContext, batch: &Batch) -> Result<()> {
        let rows = batch.card();
        if rows == 0 {
            return Ok(());
        }
        // The VM's result is a dense vector with selection already resolved, so row numbers are 0..rows.
        let mut gvs = Vec::with_capacity(self.groups.len());
        for (i, p) in self.groups.iter().enumerate() {
            let v = ctx.vm.eval(p, batch)?;
            // A mismatch between the declared type and the real data would corrupt the output column. Detect the contract violation.
            ensure!(v.data().phys() == self.key_cols[i].ty().phys(), Internal);
            gvs.push(v);
        }
        let mut avs = Vec::with_capacity(self.aggs.len());
        for a in &self.aggs {
            avs.push(match &a.arg {
                Some(p) => Some(ctx.vm.eval(p, batch)?),
                None => None,
            });
        }
        // `FILTER (WHERE cond)`. An independent BOOLEAN column is evaluated per aggregate up
        // front, so the row loop only has to look up a truth value.
        let mut fvs = Vec::with_capacity(self.aggs.len());
        for a in &self.aggs {
            fvs.push(match &a.filter {
                Some(p) => Some(ctx.vm.eval(p, batch)?),
                None => None,
            });
        }

        let refs: Vec<&Vector> = gvs.iter().collect();
        let mut key = Vec::new();
        let mut dkey = Vec::new();
        let mut vkey = Vec::new();
        for row in 0..rows {
            encode_key(&refs, row, &mut key);
            let (slot, is_new) = self.index.get_or_insert(&key);
            if is_new {
                self.key_bytes += key.len();
                for (i, c) in self.key_cols.iter_mut().enumerate() {
                    c.push_value(&gvs[i].value_at(row));
                }
                for s in self.states.iter_mut() {
                    s.push(State::empty());
                }
            }
            let g = slot as usize;

            for (ai, av) in avs.iter().enumerate() {
                // Rows where FILTER is false or NULL are treated as absent for this aggregate
                // (SQL three-valued logic: UNKNOWN is excluded too). Other aggregates are unaffected.
                if let Some(fv) = &fvs[ai] {
                    // `compile_predicate` only guarantees the result type is BOOLEAN or NULL
                    // (the `WHERE NULL` case). A NULL type's physical representation is not
                    // necessarily Bool, so the physical type is checked before calling `.bools()`
                    // (the same defense as `vm::eval_filter`).
                    let pass = fv.data().phys() == PhysType::Bool
                        && fv.is_valid(row)
                        && fv.bools().get(row);
                    if !pass {
                        continue;
                    }
                }
                if self.ops[ai] == Op::CountStar {
                    // COUNT(*) counts rows that are all NULL too.
                    self.states[ai][g].n += 1;
                    continue;
                }
                let col = match av {
                    Some(c) => c,
                    // Everything but COUNT(*) always has an argument (the `Agg` contract).
                    None => err!(Internal),
                };
                let valid = col.is_valid(row);
                // SUM/MIN/MAX/AVG/COUNT(x) ignore NULLs. Only ArrayAgg counts NULL as an element
                // (the same as DuckDB's array_agg/list).
                if !valid && self.ops[ai] != Op::ArrayAgg {
                    continue;
                }
                if let Some(seen) = &mut self.distinct[ai] {
                    // Prefixing the group number lets one table serve every group. It avoids a
                    // nested table, at the cost of as many keys as there are (group, value)
                    // pairs, so memory grows accordingly.
                    // `encode_key` encodes invalid values with flag=0 as well, so ArrayAgg(DISTINCT x)'s
                    // NULL deduplication passes through unchanged.
                    encode_key(&[col], row, &mut vkey);
                    dkey.clear();
                    dkey.extend_from_slice(&slot.to_le_bytes());
                    dkey.extend_from_slice(&vkey);
                    if !seen.get_or_insert(&dkey).1 {
                        continue;
                    }
                }
                if valid {
                    self.update(ai, g, col, row)?;
                } else {
                    // Only ArrayAgg's NULL rows reach here.
                    self.push_array_null(ai, g);
                }
            }
        }

        ensure!(self.mem_used() <= MAX_STATE_BYTES, Oom);
        Ok(())
    }

    /// Folds one non-NULL, already-deduplicated value into the state.
    fn update(&mut self, ai: usize, g: usize, col: &Vector, row: usize) -> Result<()> {
        let op = self.ops[ai];
        let st = &mut self.states[ai][g];
        st.n += 1;
        match op {
            Op::CountStar | Op::Count => {}
            Op::SumInt | Op::AvgInt => {
                let x = as_i128(col, row)?;
                let sum = match &st.acc {
                    Value::I128(s) => match s.checked_add(x) {
                        Some(v) => v,
                        // A sum that overflows even i128 is an error rather than a silent wraparound.
                        None => err!(ValueOutOfRange),
                    },
                    _ => x,
                };
                st.acc = Value::I128(sum);
            }
            Op::SumF64 | Op::AvgF64 => {
                let x = as_f64(col, row)?;
                let sum = match &st.acc {
                    Value::F64(s) => s + x,
                    _ => x,
                };
                st.acc = Value::F64(sum);
            }
            Op::Min | Op::Max => {
                let take = match &st.acc {
                    Value::Null => true,
                    acc => {
                        let c = cmp_at(col, row, acc);
                        if op == Op::Min {
                            c.is_lt()
                        } else {
                            c.is_gt()
                        }
                    }
                };
                if take {
                    if let Value::Bytes(b) = &st.acc {
                        self.acc_bytes = self.acc_bytes.saturating_sub(b.len());
                    }
                    let v = col.value_at(row);
                    if let Value::Bytes(b) = &v {
                        self.acc_bytes += b.len();
                    }
                    self.states[ai][g].acc = v;
                }
            }
            Op::StdDev | Op::Variance => {
                // Welford's online update. `st.n` was already incremented at the top of this
                // function, so it serves directly as "the total so far".
                let x = as_f64_generic(col, row, self.avg_div[ai])?;
                let delta = x - st.mean;
                st.mean += delta / st.n as f64;
                let delta2 = x - st.mean;
                st.m2 += delta * delta2;
            }
            Op::Median => {
                // An exact median cannot be found while streaming, so every non-NULL value is
                // held until output. Sorting happens exactly once, in push_result.
                let x = as_f64_generic(col, row, self.avg_div[ai])?;
                st.median_vals.push(x);
                self.acc_bytes += 8;
            }
            Op::Mode => {
                // The same "prefix the group number" scheme as the DISTINCT deduplication table,
                // with one frequency table shared across every group.
                let mut vkey = Vec::new();
                encode_key(&[col], row, &mut vkey);
                let mut fkey = Vec::with_capacity(4 + vkey.len());
                fkey.extend_from_slice(&(g as u32).to_le_bytes());
                fkey.extend_from_slice(&vkey);
                let freq = match &mut self.mode_freq[ai] {
                    Some(f) => f,
                    // Mode's Op always has a matching frequency table (a contract violation otherwise).
                    None => err!(Internal),
                };
                let (slot, is_new) = freq.get_or_insert(&fkey);
                if is_new {
                    self.mode_counts[ai].push(1);
                    self.acc_bytes += 8;
                } else {
                    self.mode_counts[ai][slot as usize] += 1;
                }
                let cnt = self.mode_counts[ai][slot as usize];
                // Being `>`, on a tie the value found first stays the winner
                // (the "first arrival wins" behavior observed in DuckDB).
                if cnt > st.mode_best {
                    st.mode_best = cnt;
                    if let Value::Bytes(b) = &st.acc {
                        self.acc_bytes = self.acc_bytes.saturating_sub(b.len());
                    }
                    let v = col.value_at(row);
                    if let Value::Bytes(b) = &v {
                        self.acc_bytes += b.len();
                    }
                    st.acc = v;
                }
            }
            Op::StringAgg => {
                let bytes = match col.data() {
                    Data::Bytes(b) => b.get(row),
                    // A defense in case the binder hands over something other than VARCHAR.
                    _ => err!(TypeMismatch),
                };
                let sep = &self.aggs[ai].separator;
                let old_len = match &st.acc {
                    Value::Bytes(b) => b.len(),
                    _ => 0,
                };
                let mut buf = match core::mem::replace(&mut st.acc, Value::Null) {
                    Value::Bytes(b) => b,
                    _ => Vec::new(),
                };
                if old_len > 0 {
                    buf.extend_from_slice(sep);
                }
                buf.extend_from_slice(bytes);
                self.acc_bytes += buf.len() - old_len;
                st.acc = Value::Bytes(buf);
            }
            Op::ArrayAgg => {
                let mut text = Vec::new();
                push_json_scalar(col, row, &mut text);
                let old_len = match &st.acc {
                    Value::Bytes(b) => b.len(),
                    _ => 0,
                };
                let buf = match core::mem::replace(&mut st.acc, Value::Null) {
                    Value::Bytes(b) => b,
                    _ => Vec::new(),
                };
                let new_buf = append_array_text(buf, &text);
                self.acc_bytes += new_buf.len() - old_len;
                st.acc = Value::Bytes(new_buf);
            }
        }
        Ok(())
    }

    /// Pushes a NULL element for ArrayAgg. Unlike other aggregates it does not skip NULLs
    /// (DuckDB's `array_agg`/`list` include NULL as an element in the result).
    fn push_array_null(&mut self, ai: usize, g: usize) {
        let st = &mut self.states[ai][g];
        st.n += 1;
        let old_len = match &st.acc {
            Value::Bytes(b) => b.len(),
            _ => 0,
        };
        let buf = match core::mem::replace(&mut st.acc, Value::Null) {
            Value::Bytes(b) => b,
            _ => Vec::new(),
        };
        let new_buf = append_array_text(buf, b"null");
        self.acc_bytes += new_buf.len() - old_len;
        st.acc = Value::Bytes(new_buf);
    }

    /// Pushes one group's aggregate results into the output vectors.
    ///
    /// Only Median needs `&mut self`, because it sorts `median_vals` once just before output
    /// (the other operations do not change state here).
    fn push_result(&mut self, ai: usize, g: usize, out: &mut Vector) {
        match self.ops[ai] {
            Op::CountStar | Op::Count => out.push_value(&Value::I64(self.states[ai][g].n)),
            // A group with not a single non-NULL input is NULL. StringAgg follows the same rule
            // (with no non-NULL value ever added, acc stays Null).
            // Mode's winner is in acc directly too.
            Op::SumInt | Op::SumF64 | Op::Min | Op::Max | Op::StringAgg | Op::Mode => {
                out.push_value(&self.states[ai][g].acc)
            }
            Op::AvgInt => {
                // Integers are summed exactly in i128 and divided exactly once. Summing in f64
                // would accumulate rounding error.
                let st = &self.states[ai][g];
                match &st.acc {
                    Value::I128(s) if st.n > 0 => {
                        out.push_value(&Value::F64(*s as f64 / self.avg_div[ai] / st.n as f64))
                    }
                    _ => out.push_null(),
                }
            }
            Op::AvgF64 => {
                let st = &self.states[ai][g];
                match &st.acc {
                    Value::F64(s) if st.n > 0 => out.push_value(&Value::F64(s / st.n as f64)),
                    _ => out.push_null(),
                }
            }
            Op::StdDev | Op::Variance => {
                let st = &self.states[ai][g];
                // Sample variance and sample standard deviation are undefined for n < 2 (NULL in DuckDB too).
                if st.n < 2 {
                    out.push_null();
                } else {
                    // Floating-point rounding can make M2 slightly negative, so it is clamped at
                    // 0 (preventing a group of identical values from producing a variance that is
                    // not exactly 0 and yielding the square root of NaN).
                    let var = (st.m2 / (st.n as f64 - 1.0)).max(0.0);
                    let v = if self.ops[ai] == Op::StdDev { f_sqrt(var) } else { var };
                    out.push_value(&Value::F64(v));
                }
            }
            Op::Median => {
                let st = &mut self.states[ai][g];
                if st.median_vals.is_empty() {
                    out.push_null();
                } else {
                    // Aligned with the same total order as MIN/MAX, "NaN is the maximum"
                    // (NaN in the median's input is unlikely in the first place, but this avoids
                    // carrying two ordering functions).
                    st.median_vals.sort_by(|a, b| ord_f64(*a, *b));
                    let m = st.median_vals.len();
                    // The same linear interpolation as quantile_cont(0.5): h = (m-1)*0.5.
                    // `core` has no `f64::floor`/`ceil` (they are in libm), so floor(h)/ceil(h)
                    // are derived with integer arithmetic alone, using the fact that `m-1` is a
                    // non-negative integer (truncation happens automatically).
                    let lo = (m - 1) / 2;
                    let hi = m / 2;
                    let v = if lo == hi {
                        st.median_vals[lo]
                    } else {
                        let h = (m - 1) as f64 * 0.5;
                        let frac = h - lo as f64;
                        st.median_vals[lo] + (st.median_vals[hi] - st.median_vals[lo]) * frac
                    };
                    out.push_value(&Value::F64(v));
                }
            }
            Op::ArrayAgg => match &self.states[ai][g].acc {
                // While accumulating there is no closing bracket (see `append_array_text`), so
                // `]` is added here for the first time.
                Value::Bytes(b) => {
                    let mut v = b.clone();
                    v.push(b']');
                    out.push_value(&Value::Bytes(v));
                }
                _ => out.push_null(),
            },
        }
    }

    fn emit(&mut self, ctx: &mut ExecContext) -> Result<Step> {
        let total = self.num_groups();
        while self.emit_pos < total {
            let start = self.emit_pos;
            let end = (start + BATCH_SIZE).min(total);
            let idx: Vec<u32> = (start as u32..end as u32).collect();
            // The output puts group columns first and aggregate columns after (the same order as the binder's schema).
            let mut cols = Vec::with_capacity(self.key_cols.len() + self.ops.len());
            for c in &self.key_cols {
                cols.push(c.gather(&idx));
            }
            for ai in 0..self.ops.len() {
                let mut v = Vector::with_capacity(self.out_tys[ai], end - start);
                for g in start..end {
                    self.push_result(ai, g, &mut v);
                }
                v.compact_validity();
                cols.push(v);
            }
            self.emit_pos = end;

            let mut batch =
                if cols.is_empty() { Batch::rows_only(end - start) } else { Batch::new(cols) };
            if let Some(h) = &self.having {
                // HAVING is evaluated against the post-aggregation schema.
                let mut sel = Vec::new();
                ctx.vm.eval_filter(h, &batch, &mut sel)?;
                if sel.is_empty() {
                    continue;
                }
                batch.sel = Some(sel);
            }
            return Ok(Step::Ready(batch));
        }
        self.phase = Phase::Done;
        Ok(Step::Done)
    }
}

impl Operator for HashAggregate {
    fn next(&mut self, ctx: &mut ExecContext) -> Result<Step> {
        loop {
            match self.phase {
                Phase::Building => match self.input.next(ctx)? {
                    Step::Ready(b) => self.consume(ctx, &b)?,
                    // The partial state is left as is and passed straight through. Next time it
                    // resumes from here. Ingestion is per batch, so an interruption drops no rows
                    // and double-counts none.
                    other @ (Step::NeedIo | Step::NeedCodec) => return Ok(other),
                    Step::Done => {
                        // An aggregate without GROUP BY returns one row even for empty input
                        // (COUNT(*) is 0, everything else NULL). With GROUP BY it is 0 rows.
                        // This asymmetry is exactly as SQL specifies.
                        if self.groups.is_empty() && self.index.is_empty() {
                            self.index.get_or_insert(&[]);
                            for s in self.states.iter_mut() {
                                s.push(State::empty());
                            }
                        }
                        self.phase = Phase::Emitting;
                    }
                },
                Phase::Emitting => return self.emit(ctx),
                Phase::Done => return Ok(Step::Done),
            }
        }
    }
}

/// Extracts one integer-family value as i128. Always widened, to avoid SUM overflow.
fn as_i128(col: &Vector, row: usize) -> Result<i128> {
    Ok(match col.data() {
        Data::I32(v) => v[row] as i128,
        Data::I64(v) => v[row] as i128,
        Data::I128(v) => v[row],
        _ => err!(TypeMismatch),
    })
}

fn as_f64(col: &Vector, row: usize) -> Result<f64> {
    Ok(match col.data() {
        Data::F64(v) => v[row],
        _ => err!(TypeMismatch),
    })
}

/// Compares row `row` of a column with the accumulated value. A physical type mismatch is a caller bug.
fn cmp_at(col: &Vector, row: usize, acc: &Value) -> core::cmp::Ordering {
    use core::cmp::Ordering;
    match (col.data(), acc) {
        (Data::Bool(b), Value::Bool(x)) => b.get(row).cmp(x),
        (Data::I32(v), Value::I32(x)) => v[row].cmp(x),
        (Data::I64(v), Value::I64(x)) => v[row].cmp(x),
        (Data::I128(v), Value::I128(x)) => v[row].cmp(x),
        (Data::F64(v), Value::F64(x)) => ord_f64(v[row], *x),
        (Data::Bytes(b), Value::Bytes(x)) => b.get(row).cmp(x.as_slice()),
        _ => Ordering::Equal,
    }
}

/// Extracts one value of a numeric column as f64. DECIMAL is divided back by `div` (= 10^scale).
/// StdDev/Variance/Median always produce DOUBLE, so unlike SUM/AVG there is no point
/// accumulating as integers and dividing later; they move to f64 from the start.
fn as_f64_generic(col: &Vector, row: usize, div: f64) -> Result<f64> {
    Ok(match col.data() {
        Data::I32(v) => v[row] as f64 / div,
        Data::I64(v) => v[row] as f64 / div,
        Data::I128(v) => v[row] as f64 / div,
        Data::F64(v) => v[row],
        _ => err!(TypeMismatch),
    })
}

/// Square root. `core` has no `f64::sqrt` (it is in libm). `expr::funcs::f_sqrt` is not exposed
/// for this use (it stays private), so it is rebuilt here on the same idea (Newton's method
/// from an initial value with a halved exponent, converging to double precision in five
/// iterations).
fn f_sqrt(x: f64) -> f64 {
    if x <= 0.0 || !x.is_finite() {
        return if x < 0.0 { f64::NAN } else { x };
    }
    let mut y = f64::from_bits((x.to_bits() + (1023u64 << 52)) >> 1);
    for _ in 0..5 {
        y = 0.5 * (y + x / y);
    }
    y
}

/// Appends one element to ArrayAgg's accumulation buffer. When empty it starts writing from
/// `[`, and the closing bracket is not added (`push_result` adds it exactly once just before
/// output). A NULL element takes the same path; the caller just passes `b"null"`.
fn append_array_text(mut buf: Vec<u8>, text: &[u8]) -> Vec<u8> {
    if buf.is_empty() {
        buf.push(b'[');
    } else {
        buf.extend_from_slice(b", ");
    }
    buf.extend_from_slice(text);
    buf
}

/// Turns one ArrayAgg element into JSON-like text and appends it to `out`. This system has no
/// LIST type, so following DESIGN.md's handling of nested types (the same idea as
/// `format::jsonl` keeping a nested value as VARCHAR text), the value is moved to a JSON-like
/// string. Values reaching here are always non-NULL (for NULL the caller passes `b"null"`
/// directly to `append_array_text`).
fn push_json_scalar(col: &Vector, row: usize, out: &mut Vec<u8>) {
    match col.data() {
        Data::Bool(b) => out.extend_from_slice(if b.get(row) { b"true" } else { b"false" }),
        Data::I32(v) => push_int_text(out, v[row] as i128, 0),
        Data::I64(v) => push_int_text(out, v[row] as i128, 0),
        Data::I128(v) => {
            let scale = match col.ty() {
                Ty::Decimal { scale, .. } => scale,
                _ => 0,
            };
            push_int_text(out, v[row], scale);
        }
        Data::F64(v) => push_f64_text(out, v[row]),
        Data::Bytes(b) => push_json_string(out, b.get(row)),
    }
}

/// Renders `v` as decimal text (with a decimal point at `scale` digits for DECIMAL).
/// `expr::kernels::fmt_int` is private and cannot be called from outside, so just as much as
/// ArrayAgg needs is rewritten here on the same idea.
fn push_int_text(out: &mut Vec<u8>, v: i128, scale: u8) {
    let neg = v < 0;
    let mut u = v.unsigned_abs();
    let mut buf = [0u8; 48];
    let mut k = 0usize;
    if u == 0 {
        buf[0] = b'0';
        k = 1;
    }
    while u > 0 {
        buf[k] = b'0' + (u % 10) as u8;
        u /= 10;
        k += 1;
    }
    // Supplies the leading 0 when there is no integer part, as with 0.05.
    while k <= scale as usize {
        buf[k] = b'0';
        k += 1;
    }
    if neg {
        out.push(b'-');
    }
    for i in (0..k).rev() {
        out.push(buf[i]);
        if scale > 0 && i == scale as usize {
            out.push(b'.');
        }
    }
}

/// A simple decimal rendering of f64. Rounded to 15 significant digits with trailing zeros
/// dropped. A full CAST-grade round-trippable representation is unnecessary (this is only for
/// ArrayAgg's display), so it is not built as rigorously as `expr::kernels::fmt_f64` (which is private and uncallable).
fn push_f64_text(out: &mut Vec<u8>, x: f64) {
    if x.is_nan() {
        out.extend_from_slice(b"NaN");
        return;
    }
    if x.is_infinite() {
        out.extend_from_slice(if x < 0.0 { b"-Infinity" } else { b"Infinity" });
        return;
    }
    if x == 0.0 {
        out.extend_from_slice(if x.is_sign_negative() { b"-0" } else { b"0" });
        return;
    }
    if x < 0.0 {
        out.push(b'-');
    }
    let v = if x < 0.0 { -x } else { x };
    // Normalizes m into [1e14, 1e15) while preserving v = m * 10^e10.
    let mut m = v;
    let mut e10: i32 = 0;
    while m >= 1e15 {
        m /= 10.0;
        e10 += 1;
    }
    while m < 1e14 {
        m *= 10.0;
        e10 -= 1;
    }
    let mut mant = (m + 0.5) as u64;
    if mant >= 1_000_000_000_000_000 {
        mant /= 10;
        e10 += 1;
    }
    let mut digits = [0u8; 15];
    let mut t = mant;
    for i in (0..15).rev() {
        digits[i] = b'0' + (t % 10) as u8;
        t /= 10;
    }
    let mut k = 15usize;
    while k > 1 && digits[k - 1] == b'0' {
        k -= 1;
    }
    // v = 0.d1..dk × 10^p
    let p = e10 + 15;
    if !(1..=17).contains(&p) {
        // Exponential notation.
        out.push(digits[0]);
        if k > 1 {
            out.push(b'.');
            out.extend_from_slice(&digits[1..k]);
        }
        out.push(b'e');
        let e = p - 1;
        if e < 0 {
            out.push(b'-');
        }
        push_int_text(out, if e < 0 { -(e as i128) } else { e as i128 }, 0);
    } else if (p as usize) >= k {
        out.extend_from_slice(&digits[..k]);
        for _ in 0..(p as usize - k) {
            out.push(b'0');
        }
    } else {
        out.extend_from_slice(&digits[..p as usize]);
        out.push(b'.');
        out.extend_from_slice(&digits[p as usize..k]);
    }
}

/// Writes a JSON-like string literal. A full JSON character codec is unnecessary, so only the
/// characters that would break the round trip (quote, backslash, control characters) are
/// escaped.
fn push_json_string(out: &mut Vec<u8>, s: &[u8]) {
    out.push(b'"');
    for &b in s {
        match b {
            b'"' => out.extend_from_slice(b"\\\""),
            b'\\' => out.extend_from_slice(b"\\\\"),
            b'\n' => out.extend_from_slice(b"\\n"),
            b'\r' => out.extend_from_slice(b"\\r"),
            b'\t' => out.extend_from_slice(b"\\t"),
            0x00..=0x1f => {
                out.extend_from_slice(b"\\u00");
                out.push(hex_digit(b >> 4));
                out.push(hex_digit(b & 0xf));
            }
            _ => out.push(b),
        }
    }
    out.push(b'"');
}

fn hex_digit(n: u8) -> u8 {
    if n < 10 {
        b'0' + n
    } else {
        b'a' + (n - 10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::Catalog;
    use crate::error::code_of;
    use crate::expr::vm::Vm;
    use crate::expr::{Instr, OpCode};

    // --- Construction helpers -----------------------------------------------

    fn col(ty: Ty, vals: &[Option<Value>]) -> Vector {
        let mut v = Vector::new(ty);
        for x in vals {
            match x {
                Some(x) => v.push_value(x),
                None => v.push_null(),
            }
        }
        v
    }

    fn i32s(vals: &[Option<i32>]) -> Vector {
        col(Ty::Int, &vals.iter().map(|v| v.map(Value::I32)).collect::<Vec<_>>())
    }

    fn i64s(vals: &[Option<i64>]) -> Vector {
        col(Ty::BigInt, &vals.iter().map(|v| v.map(Value::I64)).collect::<Vec<_>>())
    }

    fn f64s(vals: &[Option<f64>]) -> Vector {
        col(Ty::Double, &vals.iter().map(|v| v.map(Value::F64)).collect::<Vec<_>>())
    }

    fn strs(vals: &[Option<&str>]) -> Vector {
        col(
            Ty::Varchar,
            &vals
                .iter()
                .map(|v| v.map(|s| Value::Bytes(s.as_bytes().to_vec())))
                .collect::<Vec<_>>(),
        )
    }

    /// A program that returns column `idx` of the input batch unchanged.
    fn load(ty: Ty, idx: u16) -> Program {
        let mut p = Program::new();
        let r = p.alloc_reg();
        p.push(Instr::with_aux(OpCode::LoadCol, ty.phys(), r, 0, 0, idx));
        p.result = r;
        p.result_ty = ty;
        p
    }

    /// The predicate `col[idx] <op> const`. For HAVING.
    fn cmp_const(op: OpCode, ty: Ty, idx: u16, v: Value) -> Program {
        let mut p = Program::new();
        let r0 = p.alloc_reg();
        let r1 = p.alloc_reg();
        let r2 = p.alloc_reg();
        p.push(Instr::with_aux(OpCode::LoadCol, ty.phys(), r0, 0, 0, idx));
        let c = p.add_const(ty, v);
        p.push(Instr::with_aux(OpCode::LoadConst, ty.phys(), r1, 0, 0, c));
        p.push(Instr::new(op, ty.phys(), r2, r0, r1));
        p.result = r2;
        p.result_ty = Ty::Boolean;
        p
    }

    fn agg(kind: AggKind, arg: Option<Program>) -> Agg {
        Agg {
            kind,
            arg,
            distinct: false,
            name: String::from("a"),
            separator: Vec::new(),
            filter: None,
        }
    }

    fn agg_distinct(kind: AggKind, arg: Program) -> Agg {
        Agg {
            kind,
            arg: Some(arg),
            distinct: true,
            name: String::from("a"),
            separator: Vec::new(),
            filter: None,
        }
    }

    /// StringAgg with a separator.
    fn agg_sep(kind: AggKind, arg: Program, sep: &[u8]) -> Agg {
        Agg {
            kind,
            arg: Some(arg),
            distinct: false,
            name: String::from("a"),
            separator: sep.to_vec(),
            filter: None,
        }
    }

    /// An aggregate with `FILTER (WHERE cond)`.
    fn agg_filter(kind: AggKind, arg: Option<Program>, filter: Program) -> Agg {
        Agg {
            kind,
            arg,
            distinct: false,
            name: String::from("a"),
            separator: Vec::new(),
            filter: Some(filter),
        }
    }

    /// Floating-point comparison with a tolerance. Welford produces some rounding error.
    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    // --- Mock operators -----------------------------------------------------

    enum MockStep {
        Rows(Batch),
        /// Reproduces waiting on I/O in the middle of the input.
        NeedIo,
        /// Waiting on the host to decompress a codec.
        NeedCodec,
    }

    struct Mock {
        steps: Vec<MockStep>,
        pos: usize,
        /// How many times it was called after returning Done. For detecting double consumption.
        after_done: usize,
    }

    impl Mock {
        fn new(steps: Vec<MockStep>) -> Box<Mock> {
            Box::new(Mock { steps, pos: 0, after_done: 0 })
        }
    }

    impl Operator for Mock {
        fn next(&mut self, _ctx: &mut ExecContext) -> Result<Step> {
            if self.pos >= self.steps.len() {
                self.after_done += 1;
                return Ok(Step::Done);
            }
            let i = self.pos;
            self.pos += 1;
            Ok(match &mut self.steps[i] {
                MockStep::NeedIo => Step::NeedIo,
                MockStep::NeedCodec => Step::NeedCodec,
                MockStep::Rows(b) => Step::Ready(core::mem::replace(b, Batch::rows_only(0))),
            })
        }
    }

    fn batches(cols: Vec<Vec<Vector>>) -> Vec<MockStep> {
        cols.into_iter().map(|c| MockStep::Rows(Batch::new(c))).collect()
    }

    // --- Execution helpers --------------------------------------------------

    /// Flattens the output into a sequence of rows. `NeedIo` simply resumes, as if the host had filled it.
    fn run(mut op: HashAggregate) -> Result<Vec<Vec<Value>>> {
        let mut catalog = Catalog::new();
        let mut vm = Vm::new();
        let mut rows = Vec::new();
        let mut guard = 0;
        loop {
            guard += 1;
            assert!(guard < 100_000, "does not terminate");
            let mut ctx = ExecContext {
                catalog: &mut catalog,
                vm: &mut vm,
                io: Vec::new(),
                codec: Vec::new(),
            };
            match op.next(&mut ctx)? {
                Step::Ready(b) => {
                    let n = b.card();
                    for i in 0..n {
                        let r = match &b.sel {
                            Some(s) => s[i] as usize,
                            None => i,
                        };
                        rows.push(b.cols.iter().map(|c| c.value_at(r)).collect::<Vec<_>>());
                    }
                }
                Step::NeedIo | Step::NeedCodec => {}
                Step::Done => break,
            }
        }
        Ok(rows)
    }

    fn build(
        steps: Vec<MockStep>,
        groups: Vec<Program>,
        aggs: Vec<Agg>,
        having: Option<Program>,
    ) -> HashAggregate {
        HashAggregate::new(Mock::new(steps), groups, aggs, having).unwrap()
    }

    /// Sorts output rows by the string representation of the key column, for easier comparison.
    fn sorted(mut rows: Vec<Vec<Value>>) -> Vec<Vec<Value>> {
        rows.sort_by_key(|r| fmt_row(r));
        rows
    }

    fn fmt_row(r: &[Value]) -> String {
        let mut s = String::new();
        for v in r {
            s.push_str(&format!("{v:?}|"));
        }
        s
    }

    // --- Types and basic shapes ---------------------------------------------

    #[test]
    fn result_types_come_from_agg_result_ty() {
        // SUM(INT) is HUGEINT, AVG is DOUBLE, COUNT is BIGINT.
        let a = HashAggregate::new(
            Mock::new(vec![]),
            vec![],
            vec![
                agg(AggKind::CountStar, None),
                agg(AggKind::Sum, Some(load(Ty::Int, 0))),
                agg(AggKind::Avg, Some(load(Ty::Int, 0))),
                agg(AggKind::Min, Some(load(Ty::Varchar, 0))),
            ],
            None,
        )
        .unwrap();
        assert_eq!(a.out_tys, vec![Ty::BigInt, Ty::HugeInt, Ty::Double, Ty::Varchar]);
    }

    #[test]
    fn invalid_input_type_is_rejected_at_construction() {
        let r = HashAggregate::new(
            Mock::new(vec![]),
            vec![],
            vec![agg(AggKind::Sum, Some(load(Ty::Varchar, 0)))],
            None,
        );
        assert_eq!(code_of(r.map(|_| ())), Some(Code::TypeMismatch));
    }

    #[test]
    fn ungrouped_count_sum_min_max_avg() {
        let steps = batches(vec![
            vec![i32s(&[Some(1), Some(2), None]), strs(&[Some("b"), Some("a"), Some("c")])],
            vec![i32s(&[Some(4)]), strs(&[Some("z")])],
        ]);
        let op = build(
            steps,
            vec![],
            vec![
                agg(AggKind::CountStar, None),
                agg(AggKind::Count, Some(load(Ty::Int, 0))),
                agg(AggKind::Sum, Some(load(Ty::Int, 0))),
                agg(AggKind::Min, Some(load(Ty::Int, 0))),
                agg(AggKind::Max, Some(load(Ty::Int, 0))),
                agg(AggKind::Avg, Some(load(Ty::Int, 0))),
                agg(AggKind::Min, Some(load(Ty::Varchar, 1))),
                agg(AggKind::Max, Some(load(Ty::Varchar, 1))),
            ],
            None,
        );
        let rows = run(op).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], Value::I64(4), "COUNT(*) counts NULL rows too");
        assert_eq!(rows[0][1], Value::I64(3), "COUNT(x) does not count NULLs");
        assert_eq!(rows[0][2], Value::I128(7));
        assert_eq!(rows[0][3], Value::I32(1));
        assert_eq!(rows[0][4], Value::I32(4));
        assert_eq!(rows[0][5], Value::F64(7.0 / 3.0));
        assert_eq!(rows[0][6], Value::Bytes(b"a".to_vec()));
        assert_eq!(rows[0][7], Value::Bytes(b"z".to_vec()));
    }

    #[test]
    fn grouped_aggregates() {
        let steps = batches(vec![vec![
            strs(&[Some("a"), Some("b"), Some("a"), Some("b"), Some("a")]),
            i32s(&[Some(1), Some(10), Some(2), None, Some(3)]),
        ]]);
        let op = build(
            steps,
            vec![load(Ty::Varchar, 0)],
            vec![
                agg(AggKind::CountStar, None),
                agg(AggKind::Count, Some(load(Ty::Int, 1))),
                agg(AggKind::Sum, Some(load(Ty::Int, 1))),
                agg(AggKind::Min, Some(load(Ty::Int, 1))),
                agg(AggKind::Max, Some(load(Ty::Int, 1))),
                agg(AggKind::Avg, Some(load(Ty::Int, 1))),
            ],
            None,
        );
        let rows = sorted(run(op).unwrap());
        assert_eq!(rows.len(), 2);
        // Group columns first, aggregate columns after.
        assert_eq!(rows[0][0], Value::Bytes(b"a".to_vec()));
        assert_eq!(rows[0][1], Value::I64(3));
        assert_eq!(rows[0][2], Value::I64(3));
        assert_eq!(rows[0][3], Value::I128(6));
        assert_eq!(rows[0][4], Value::I32(1));
        assert_eq!(rows[0][5], Value::I32(3));
        assert_eq!(rows[0][6], Value::F64(2.0));
        assert_eq!(rows[1][0], Value::Bytes(b"b".to_vec()));
        assert_eq!(rows[1][1], Value::I64(2));
        assert_eq!(rows[1][2], Value::I64(1));
        assert_eq!(rows[1][3], Value::I128(10));
    }

    // --- FILTER (WHERE cond) -------------------------------------------------

    #[test]
    fn filter_restricts_which_rows_feed_the_aggregate() {
        // col0 = value, col1 = the condition column. Emulates FILTER (WHERE col1 > 5).
        let steps = batches(vec![vec![
            i32s(&[Some(1), Some(2), Some(3), Some(4)]),
            i32s(&[Some(10), Some(1), Some(10), Some(1)]),
        ]]);
        let op = build(
            steps,
            vec![],
            vec![
                // Without FILTER: counts/sums every row.
                agg(AggKind::CountStar, None),
                agg(AggKind::Sum, Some(load(Ty::Int, 0))),
                // With FILTER: counts/sums only the rows where col1 > 5 (the 1st and 3rd).
                agg_filter(
                    AggKind::CountStar,
                    None,
                    cmp_const(OpCode::Gt, Ty::Int, 1, Value::I32(5)),
                ),
                agg_filter(
                    AggKind::Sum,
                    Some(load(Ty::Int, 0)),
                    cmp_const(OpCode::Gt, Ty::Int, 1, Value::I32(5)),
                ),
            ],
            None,
        );
        let rows = run(op).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], Value::I64(4), "COUNT(*) without FILTER covers every row");
        assert_eq!(rows[0][1], Value::I128(10), "SUM without FILTER covers every row (1+2+3+4)");
        assert_eq!(
            rows[0][2],
            Value::I64(2),
            "COUNT(*) with FILTER covers only the qualifying rows"
        );
        assert_eq!(rows[0][3], Value::I128(4), "SUM with FILTER covers the 1st and 3rd (1+3)");
    }

    #[test]
    fn filter_is_independent_per_aggregate_and_group() {
        let steps = batches(vec![vec![
            strs(&[Some("a"), Some("a"), Some("b"), Some("b")]),
            i32s(&[Some(1), Some(2), Some(3), Some(4)]),
            // The condition column: only row 1 is true for group a, only row 2 for group b.
            i32s(&[Some(1), Some(0), Some(0), Some(1)]),
        ]]);
        let cond = cmp_const(OpCode::Eq, Ty::Int, 2, Value::I32(1));
        let op = build(
            steps,
            vec![load(Ty::Varchar, 0)],
            vec![agg_filter(AggKind::Sum, Some(load(Ty::Int, 1)), cond)],
            None,
        );
        let rows = sorted(run(op).unwrap());
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0][0], Value::Bytes(b"a".to_vec()));
        assert_eq!(rows[0][1], Value::I128(1), "only row 1 qualifies in group a");
        assert_eq!(rows[1][0], Value::Bytes(b"b".to_vec()));
        assert_eq!(rows[1][1], Value::I128(4), "only row 2 qualifies in group b");
    }

    #[test]
    fn filter_that_matches_nothing_yields_count_zero_and_sum_null() {
        let steps = batches(vec![vec![i32s(&[Some(1), Some(2)])]]);
        let op = build(
            steps,
            vec![],
            vec![
                agg_filter(
                    AggKind::CountStar,
                    None,
                    cmp_const(OpCode::Gt, Ty::Int, 0, Value::I32(100)),
                ),
                agg_filter(
                    AggKind::Sum,
                    Some(load(Ty::Int, 0)),
                    cmp_const(OpCode::Gt, Ty::Int, 0, Value::I32(100)),
                ),
            ],
            None,
        );
        let rows = run(op).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], Value::I64(0), "COUNT(*) FILTER is 0 when nothing qualifies");
        assert!(rows[0][1].is_null(), "SUM FILTER is NULL when nothing qualifies");
    }

    #[test]
    fn filter_applies_to_statistical_aggregates() {
        // Of all 8 values (2,4,4,4,5,5,7,9), FILTER (WHERE col1 > 5) selects the 3rd, 6th, and
        // 8th, where col1 is 10 (values 4, 5, 9).
        // StdDev/Median without FILTER see all 8 values; with FILTER, only the narrowed 3.
        let steps = batches(vec![vec![
            i32s(&[Some(2), Some(4), Some(4), Some(4), Some(5), Some(5), Some(7), Some(9)]),
            i32s(&[Some(1), Some(1), Some(10), Some(1), Some(1), Some(10), Some(1), Some(10)]),
            // StringAgg accepts only VARCHAR, so the same values as col0 are carried in parallel as text.
            strs(&[
                Some("2"),
                Some("4"),
                Some("4"),
                Some("4"),
                Some("5"),
                Some("5"),
                Some("7"),
                Some("9"),
            ]),
        ]]);
        let cond = cmp_const(OpCode::Gt, Ty::Int, 1, Value::I32(5));
        let op = build(
            steps,
            vec![],
            vec![
                agg(AggKind::Median, Some(load(Ty::Int, 0))),
                agg_filter(AggKind::Median, Some(load(Ty::Int, 0)), cond.clone()),
                agg(AggKind::StringAgg, Some(load(Ty::Varchar, 2))),
                agg_filter(AggKind::StringAgg, Some(load(Ty::Varchar, 2)), cond),
            ],
            None,
        );
        let rows = run(op).unwrap();
        assert_eq!(
            rows[0][0],
            Value::F64(4.5),
            "the median without FILTER comes from all 8 values"
        );
        assert_eq!(
            rows[0][1],
            Value::F64(5.0),
            "the median with FILTER comes from the 3 values 4,5,9 (the middle one)"
        );
        // The separator is agg()'s default (the empty string). The separator itself is tested separately.
        assert_eq!(rows[0][2], Value::Bytes(b"24445579".to_vec()));
        assert_eq!(rows[0][3], Value::Bytes(b"459".to_vec()), "StringAgg with FILTER");
    }

    // --- State separation across several groups ---------------------------------

    #[test]
    fn statistical_aggregates_keep_separate_state_per_group_across_interleaved_batches() {
        // Rows of groups a and b are fed alternately across several batches, confirming that
        // Welford's mean/m2, the median buffer, Mode's frequency table, and StringAgg/ArrayAgg's
        // accumulation buffers do not get crossed between groups.
        let steps = batches(vec![
            vec![
                strs(&[Some("a"), Some("b"), Some("a")]),
                i32s(&[Some(1), Some(10), Some(2)]),
                strs(&[Some("1"), Some("10"), Some("2")]),
            ],
            vec![
                strs(&[Some("b"), Some("a"), Some("b")]),
                i32s(&[Some(20), Some(3), Some(10)]),
                strs(&[Some("20"), Some("3"), Some("10")]),
            ],
        ]);
        let op = build(
            steps,
            vec![load(Ty::Varchar, 0)],
            vec![
                agg(AggKind::StdDev, Some(load(Ty::Int, 1))),
                agg(AggKind::Median, Some(load(Ty::Int, 1))),
                agg(AggKind::Mode, Some(load(Ty::Int, 1))),
                agg(AggKind::StringAgg, Some(load(Ty::Varchar, 2))),
            ],
            None,
        );
        let rows = sorted(run(op).unwrap());
        assert_eq!(rows.len(), 2);
        // Group a: 1, 2, 3.
        assert_eq!(rows[0][0], Value::Bytes(b"a".to_vec()));
        assert!(close(rows[0][1].as_f64().unwrap(), 1.0), "a's stddev = {:?}", rows[0][1]);
        assert_eq!(rows[0][2], Value::F64(2.0), "a's median");
        assert_eq!(rows[0][3], Value::I32(1), "a's mode (all appear once, so the first, 1)");
        // The separator is agg()'s default (the empty string). The separator itself is tested separately.
        assert_eq!(rows[0][4], Value::Bytes(b"123".to_vec()), "a's StringAgg (arrival order)");
        // Group b: 10, 20, 10.
        assert_eq!(rows[1][0], Value::Bytes(b"b".to_vec()));
        assert!(
            close(rows[1][1].as_f64().unwrap(), 5.773502691896258),
            "b's stddev = {:?}",
            rows[1][1]
        );
        assert_eq!(rows[1][2], Value::F64(10.0), "b's median");
        assert_eq!(rows[1][3], Value::I32(10), "b's mode (10 appears twice, the most)");
        assert_eq!(rows[1][4], Value::Bytes(b"102010".to_vec()), "b's StringAgg (arrival order)");
    }

    // --- Resumption (the main event) ----------------------------------------

    #[test]
    fn need_io_in_the_middle_resumes_with_identical_result() {
        let data: Vec<Vec<Option<i32>>> =
            vec![vec![Some(1), Some(2), None], vec![Some(3), Some(1)], vec![Some(2), Some(9)]];
        let keys: Vec<Vec<Option<i32>>> =
            vec![vec![Some(0), Some(1), Some(0)], vec![Some(1), Some(0)], vec![Some(1), Some(0)]];

        let make = |interrupt: bool| {
            let mut steps = Vec::new();
            for (i, (k, v)) in keys.iter().zip(data.iter()).enumerate() {
                if interrupt && i == 1 {
                    // Waits on I/O between batches, in the middle of the input.
                    steps.push(MockStep::NeedIo);
                }
                steps.push(MockStep::Rows(Batch::new(vec![i32s(k), i32s(v)])));
                if interrupt && i == 1 {
                    steps.push(MockStep::NeedIo);
                }
            }
            build(
                steps,
                vec![load(Ty::Int, 0)],
                vec![
                    agg(AggKind::CountStar, None),
                    agg(AggKind::Count, Some(load(Ty::Int, 1))),
                    agg(AggKind::Sum, Some(load(Ty::Int, 1))),
                    agg(AggKind::Max, Some(load(Ty::Int, 1))),
                ],
                None,
            )
        };
        let plain = sorted(run(make(false)).unwrap());
        let interrupted = sorted(run(make(true)).unwrap());
        assert_eq!(plain.len(), 2);
        assert_eq!(plain, interrupted, "the result must not change across a NeedIo");
        // The contents are checked directly too (so a lost or extra row is noticed).
        // key=0's values are 1, NULL, 1, 9; key=1's are 2, 3, 2.
        assert_eq!(plain[0][1], Value::I64(4)); // COUNT(*)
        assert_eq!(plain[0][2], Value::I64(3)); // COUNT(x) excludes NULL
        assert_eq!(plain[0][3], Value::I128(1 + 1 + 9));
        assert_eq!(plain[0][4], Value::I32(9));
        assert_eq!(plain[1][1], Value::I64(3));
        assert_eq!(plain[1][3], Value::I128(2 + 3 + 2));
        assert_eq!(plain[1][4], Value::I32(3));
    }

    #[test]
    fn need_codec_is_passed_through_and_resumes() {
        // An interruption asking the host to decompress a codec is treated like NeedIo.
        let mut steps = vec![MockStep::Rows(Batch::new(vec![i32s(&[Some(1), Some(2)])]))];
        steps.push(MockStep::NeedCodec);
        steps.push(MockStep::Rows(Batch::new(vec![i32s(&[Some(3)])])));
        let op = build(
            steps,
            vec![],
            vec![agg(AggKind::CountStar, None), agg(AggKind::Sum, Some(load(Ty::Int, 0)))],
            None,
        );
        let rows = run(op).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], Value::I64(3));
        assert_eq!(rows[0][1], Value::I128(6));
    }

    #[test]
    fn need_io_before_any_input() {
        let mut steps = vec![MockStep::NeedIo, MockStep::NeedIo];
        steps.extend(batches(vec![vec![i32s(&[Some(5), Some(6)])]]));
        let op = build(steps, vec![], vec![agg(AggKind::Sum, Some(load(Ty::Int, 0)))], None);
        let rows = run(op).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], Value::I128(11));
    }

    #[test]
    fn done_is_sticky_and_input_is_not_reconsumed() {
        let steps = batches(vec![vec![i32s(&[Some(1), Some(2)])]]);
        let mut op = build(steps, vec![], vec![agg(AggKind::CountStar, None)], None);
        let mut catalog = Catalog::new();
        let mut vm = Vm::new();
        let mut first = None;
        for _ in 0..5 {
            let mut ctx = ExecContext {
                catalog: &mut catalog,
                vm: &mut vm,
                io: Vec::new(),
                codec: Vec::new(),
            };
            match op.next(&mut ctx).unwrap() {
                Step::Ready(b) => {
                    assert!(first.is_none(), "there should be exactly one output batch");
                    first = Some(b.cols[0].value_at(0));
                }
                Step::NeedIo | Step::NeedCodec => panic!("no interruption should occur"),
                Step::Done => {}
            }
        }
        assert_eq!(first, Some(Value::I64(2)));
    }

    // --- Empty input and all-NULL -------------------------------------------

    #[test]
    fn empty_input_ungrouped_emits_one_row() {
        let op = build(
            vec![],
            vec![],
            vec![
                agg(AggKind::CountStar, None),
                agg(AggKind::Count, Some(load(Ty::Int, 0))),
                agg(AggKind::Sum, Some(load(Ty::Int, 0))),
                agg(AggKind::Min, Some(load(Ty::Int, 0))),
                agg(AggKind::Max, Some(load(Ty::Int, 0))),
                agg(AggKind::Avg, Some(load(Ty::Int, 0))),
            ],
            None,
        );
        let rows = run(op).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], Value::I64(0));
        assert_eq!(rows[0][1], Value::I64(0));
        for v in &rows[0][2..] {
            assert_eq!(*v, Value::Null);
        }
    }

    #[test]
    fn empty_input_grouped_emits_nothing() {
        let op = build(vec![], vec![load(Ty::Int, 0)], vec![agg(AggKind::CountStar, None)], None);
        assert!(run(op).unwrap().is_empty());
    }

    #[test]
    fn empty_batches_are_ignored() {
        // A batch with card()==0 mixed in does not change the one-row result.
        let steps = vec![
            MockStep::Rows(Batch::new(vec![i32s(&[])])),
            MockStep::Rows(Batch::new(vec![i32s(&[Some(4)])])),
        ];
        let op = build(steps, vec![], vec![agg(AggKind::CountStar, None)], None);
        let rows = run(op).unwrap();
        assert_eq!(rows[0][0], Value::I64(1));
    }

    #[test]
    fn all_null_group_counts_zero_and_sums_null() {
        let steps = batches(vec![vec![
            strs(&[Some("a"), Some("a"), Some("b")]),
            i32s(&[None, None, Some(7)]),
        ]]);
        let op = build(
            steps,
            vec![load(Ty::Varchar, 0)],
            vec![
                agg(AggKind::CountStar, None),
                agg(AggKind::Count, Some(load(Ty::Int, 1))),
                agg(AggKind::Sum, Some(load(Ty::Int, 1))),
                agg(AggKind::Avg, Some(load(Ty::Int, 1))),
                agg(AggKind::Min, Some(load(Ty::Int, 1))),
            ],
            None,
        );
        let rows = sorted(run(op).unwrap());
        assert_eq!(rows[0][1], Value::I64(2), "COUNT(*) is 2");
        assert_eq!(rows[0][2], Value::I64(0), "COUNT(x) is 0");
        assert_eq!(rows[0][3], Value::Null);
        assert_eq!(rows[0][4], Value::Null);
        assert_eq!(rows[0][5], Value::Null);
    }

    // --- Group keys ---------------------------------------------------------

    #[test]
    fn null_keys_form_their_own_group() {
        let steps = batches(vec![vec![
            i32s(&[None, Some(1), None, Some(1)]),
            i32s(&[Some(1), Some(2), Some(3), Some(4)]),
        ]]);
        let op = build(
            steps,
            vec![load(Ty::Int, 0)],
            vec![agg(AggKind::CountStar, None), agg(AggKind::Sum, Some(load(Ty::Int, 1)))],
            None,
        );
        let rows = sorted(run(op).unwrap());
        assert_eq!(rows.len(), 2);
        // Two rows each for the NULL group and the 1 group.
        let null_row = rows.iter().find(|r| r[0] == Value::Null).unwrap();
        assert_eq!(null_row[1], Value::I64(2));
        assert_eq!(null_row[2], Value::I128(4));
    }

    #[test]
    fn multiple_group_columns() {
        let steps = batches(vec![vec![
            strs(&[Some("a"), Some("a"), Some("b"), Some("a")]),
            i32s(&[Some(1), Some(2), Some(1), Some(1)]),
        ]]);
        let op = build(
            steps,
            vec![load(Ty::Varchar, 0), load(Ty::Int, 1)],
            vec![agg(AggKind::CountStar, None)],
            None,
        );
        let rows = sorted(run(op).unwrap());
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0][0], Value::Bytes(b"a".to_vec()));
        assert_eq!(rows[0][1], Value::I32(1));
        assert_eq!(rows[0][2], Value::I64(2));
    }

    #[test]
    fn float_group_key_merges_zeros_and_nans() {
        let steps = batches(vec![vec![
            f64s(&[Some(0.0), Some(-0.0), Some(f64::NAN), Some(-f64::NAN), Some(1.5)]),
            i32s(&[Some(1), Some(1), Some(1), Some(1), Some(1)]),
        ]]);
        let op = build(steps, vec![load(Ty::Double, 0)], vec![agg(AggKind::CountStar, None)], None);
        let rows = run(op).unwrap();
        assert_eq!(rows.len(), 3, "0.0/-0.0 and NaN/-NaN each form one group");
        for r in &rows {
            let n = r[1].as_i64().unwrap();
            assert!(n == 2 || n == 1);
        }
        let total: i64 = rows.iter().map(|r| r[1].as_i64().unwrap()).sum();
        assert_eq!(total, 5);
    }

    #[test]
    fn min_max_treat_nan_as_largest() {
        let steps = batches(vec![vec![
            i32s(&[Some(0), Some(0), Some(0), Some(1)]),
            f64s(&[Some(1.0), Some(f64::NAN), Some(-2.0), Some(f64::NAN)]),
        ]]);
        let op = build(
            steps,
            vec![load(Ty::Int, 0)],
            vec![
                agg(AggKind::Min, Some(load(Ty::Double, 1))),
                agg(AggKind::Max, Some(load(Ty::Double, 1))),
            ],
            None,
        );
        let rows = sorted(run(op).unwrap());
        assert_eq!(rows[0][1], Value::F64(-2.0));
        assert!(rows[0][2].as_f64().unwrap().is_nan(), "MAX is NaN");
        // In an all-NaN group MIN is NaN too.
        assert!(rows[1][1].as_f64().unwrap().is_nan());
    }

    #[test]
    fn min_max_over_bools() {
        let steps = batches(vec![vec![col(
            Ty::Boolean,
            &[Some(Value::Bool(true)), Some(Value::Bool(false)), None],
        )]]);
        let op = build(
            steps,
            vec![],
            vec![
                agg(AggKind::Min, Some(load(Ty::Boolean, 0))),
                agg(AggKind::Max, Some(load(Ty::Boolean, 0))),
            ],
            None,
        );
        let rows = run(op).unwrap();
        assert_eq!(rows[0][0], Value::Bool(false));
        assert_eq!(rows[0][1], Value::Bool(true));
    }

    // --- Overflow -----------------------------------------------------------

    #[test]
    fn sum_of_i64_accumulates_in_i128() {
        let big = i64::MAX;
        let steps = batches(vec![vec![i64s(&[Some(big), Some(big), Some(big)])]]);
        let op = build(steps, vec![], vec![agg(AggKind::Sum, Some(load(Ty::BigInt, 0)))], None);
        let rows = run(op).unwrap();
        assert_eq!(rows[0][0], Value::I128(big as i128 * 3), "a sum that overflows i64");
    }

    #[test]
    fn sum_overflowing_i128_errors() {
        let huge = col(Ty::HugeInt, &[Some(Value::I128(i128::MAX)), Some(Value::I128(i128::MAX))]);
        let steps = vec![MockStep::Rows(Batch::new(vec![huge]))];
        let op = build(steps, vec![], vec![agg(AggKind::Sum, Some(load(Ty::HugeInt, 0)))], None);
        assert_eq!(code_of(run(op)), Some(Code::ValueOutOfRange));
    }

    // --- DISTINCT -----------------------------------------------------------

    #[test]
    fn distinct_aggregates() {
        let steps = batches(vec![
            vec![
                strs(&[Some("a"), Some("a"), Some("a"), Some("b")]),
                i32s(&[Some(1), Some(1), Some(2), Some(5)]),
            ],
            vec![strs(&[Some("a"), Some("b")]), i32s(&[Some(2), Some(5)])],
        ]);
        let op = build(
            steps,
            vec![load(Ty::Varchar, 0)],
            vec![
                agg_distinct(AggKind::Count, load(Ty::Int, 1)),
                agg_distinct(AggKind::Sum, load(Ty::Int, 1)),
                agg(AggKind::Count, Some(load(Ty::Int, 1))),
                agg(AggKind::Sum, Some(load(Ty::Int, 1))),
                agg_distinct(AggKind::Avg, load(Ty::Int, 1)),
            ],
            None,
        );
        let rows = sorted(run(op).unwrap());
        // a: values are 1,1,2,2 -> distinct is {1,2}
        assert_eq!(rows[0][1], Value::I64(2));
        assert_eq!(rows[0][2], Value::I128(3));
        assert_eq!(rows[0][3], Value::I64(4), "non-DISTINCT counts duplicates");
        assert_eq!(rows[0][4], Value::I128(6));
        assert_eq!(rows[0][5], Value::F64(1.5));
        // b: values are 5,5 -> distinct is {5}
        assert_eq!(rows[1][1], Value::I64(1));
        assert_eq!(rows[1][2], Value::I128(5));
    }

    #[test]
    fn distinct_dedup_is_per_group() {
        let steps = batches(vec![vec![
            i32s(&[Some(0), Some(1), Some(0), Some(1)]),
            i32s(&[Some(7), Some(7), Some(7), Some(7)]),
        ]]);
        let op = build(
            steps,
            vec![load(Ty::Int, 0)],
            vec![agg_distinct(AggKind::Count, load(Ty::Int, 1))],
            None,
        );
        let rows = sorted(run(op).unwrap());
        assert_eq!(rows.len(), 2);
        // The same value 7 is counted separately when the groups differ.
        assert_eq!(rows[0][1], Value::I64(1));
        assert_eq!(rows[1][1], Value::I64(1));
    }

    #[test]
    fn distinct_ignores_nulls() {
        let steps = batches(vec![vec![i32s(&[None, None, Some(3), Some(3)])]]);
        let op = build(steps, vec![], vec![agg_distinct(AggKind::Count, load(Ty::Int, 0))], None);
        let rows = run(op).unwrap();
        assert_eq!(rows[0][0], Value::I64(1));
    }

    // --- HAVING -------------------------------------------------------------

    #[test]
    fn having_filters_groups() {
        let steps =
            batches(vec![vec![i32s(&[Some(1), Some(1), Some(2), Some(3), Some(3), Some(3)])]]);
        // HAVING COUNT(*) > 1. The output schema is [key, count].
        let op = build(
            steps,
            vec![load(Ty::Int, 0)],
            vec![agg(AggKind::CountStar, None)],
            Some(cmp_const(OpCode::Gt, Ty::BigInt, 1, Value::I64(1))),
        );
        let rows = sorted(run(op).unwrap());
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0][0], Value::I32(1));
        assert_eq!(rows[1][0], Value::I32(3));
    }

    #[test]
    fn having_can_remove_everything() {
        let steps = batches(vec![vec![i32s(&[Some(1), Some(2), Some(3)])]]);
        let op = build(
            steps,
            vec![load(Ty::Int, 0)],
            vec![agg(AggKind::CountStar, None)],
            Some(cmp_const(OpCode::Gt, Ty::BigInt, 1, Value::I64(100))),
        );
        assert!(run(op).unwrap().is_empty());
    }

    #[test]
    fn having_on_ungrouped_row() {
        let steps = batches(vec![vec![i32s(&[Some(1), Some(2)])]]);
        let op = build(
            steps,
            vec![],
            vec![agg(AggKind::CountStar, None)],
            Some(cmp_const(OpCode::Gt, Ty::BigInt, 0, Value::I64(5))),
        );
        assert!(run(op).unwrap().is_empty());
    }

    // --- Size ---------------------------------------------------------------

    #[test]
    fn more_groups_than_batch_size_emits_multiple_batches() {
        // More groups than BATCH_SIZE. The hash table rehashes several times too.
        let n = BATCH_SIZE * 2 + 37;
        let mut steps = Vec::new();
        let mut i = 0usize;
        while i < n {
            let end = (i + 500).min(n);
            let keys: Vec<Option<i32>> = (i..end).map(|k| Some(k as i32)).collect();
            let vals: Vec<Option<i32>> = (i..end).map(|_| Some(2)).collect();
            steps.push(MockStep::Rows(Batch::new(vec![i32s(&keys), i32s(&vals)])));
            // Even with an I/O wait interposed every time, the groups must not change in number.
            steps.push(MockStep::NeedIo);
            i = end;
        }
        // The same keys are fed once more, so each group ends up with two rows.
        let keys: Vec<Option<i32>> = (0..n).map(|k| Some(k as i32)).collect();
        let vals: Vec<Option<i32>> = (0..n).map(|_| Some(3)).collect();
        steps.push(MockStep::Rows(Batch::new(vec![i32s(&keys), i32s(&vals)])));

        let op = build(
            steps,
            vec![load(Ty::Int, 0)],
            vec![agg(AggKind::CountStar, None), agg(AggKind::Sum, Some(load(Ty::Int, 1)))],
            None,
        );
        let rows = run(op).unwrap();
        assert_eq!(rows.len(), n, "the group count is unchanged");
        let mut seen = vec![false; n];
        for r in &rows {
            let k = r[0].as_i64().unwrap() as usize;
            assert!(!seen[k], "group {k} is duplicated");
            seen[k] = true;
            assert_eq!(r[1], Value::I64(2));
            assert_eq!(r[2], Value::I128(5));
        }
        assert!(seen.iter().all(|s| *s));
    }

    #[test]
    fn output_is_chunked_to_batch_size() {
        let n = BATCH_SIZE + 10;
        let keys: Vec<Option<i32>> = (0..n).map(|k| Some(k as i32)).collect();
        let steps = vec![MockStep::Rows(Batch::new(vec![i32s(&keys)]))];
        let mut op =
            build(steps, vec![load(Ty::Int, 0)], vec![agg(AggKind::CountStar, None)], None);
        let mut catalog = Catalog::new();
        let mut vm = Vm::new();
        let mut sizes = Vec::new();
        loop {
            let mut ctx = ExecContext {
                catalog: &mut catalog,
                vm: &mut vm,
                io: Vec::new(),
                codec: Vec::new(),
            };
            match op.next(&mut ctx).unwrap() {
                Step::Ready(b) => sizes.push(b.card()),
                Step::NeedIo | Step::NeedCodec => {}
                Step::Done => break,
            }
        }
        assert_eq!(sizes, vec![BATCH_SIZE, 10]);
    }

    #[test]
    fn memory_limit_is_enforced() {
        // Long keys are fed in bulk to hit the cap. There is no spilling, so it returns Oom.
        let mut steps = Vec::new();
        let mut i = 0u32;
        // Just over 1 KiB of key per group -> enough rows to reach 64 MiB.
        let filler = "x".repeat(1024);
        for _ in 0..80 {
            let keys: Vec<Option<String>> = (0..1000)
                .map(|k| {
                    i += 1;
                    Some(format!("{filler}{k}-{i}"))
                })
                .collect();
            let refs: Vec<Option<&str>> = keys.iter().map(|s| s.as_deref()).collect();
            steps.push(MockStep::Rows(Batch::new(vec![strs(&refs)])));
        }
        let op =
            build(steps, vec![load(Ty::Varchar, 0)], vec![agg(AggKind::CountStar, None)], None);
        assert_eq!(code_of(run(op)), Some(Code::Oom));
    }

    #[test]
    fn phys_type_mismatch_between_program_and_data_is_detected() {
        // Declared INT but the real data is VARCHAR. Internal, as a contract violation.
        let steps = vec![MockStep::Rows(Batch::new(vec![strs(&[Some("a")])]))];
        let op = build(steps, vec![load(Ty::Int, 0)], vec![agg(AggKind::CountStar, None)], None);
        assert_eq!(code_of(run(op)), Some(Code::Internal));
    }

    // --- StdDev / Variance ---------------------------------------------------

    #[test]
    fn stddev_and_variance_match_hand_verified_welford_result() {
        // 2,4,4,4,5,5,7,9 -> variance 32/7 = 4.571428571428571,
        // standard deviation = 2.138089935299395 (cross-checked with duckdb's stddev_samp/var_samp).
        let steps = batches(vec![vec![i32s(&[
            Some(2),
            Some(4),
            Some(4),
            Some(4),
            Some(5),
            Some(5),
            Some(7),
            Some(9),
        ])]]);
        let op = build(
            steps,
            vec![],
            vec![
                agg(AggKind::Variance, Some(load(Ty::Int, 0))),
                agg(AggKind::StdDev, Some(load(Ty::Int, 0))),
            ],
            None,
        );
        let rows = run(op).unwrap();
        let var = rows[0][0].as_f64().unwrap();
        let sd = rows[0][1].as_f64().unwrap();
        assert!(close(var, 32.0 / 7.0), "variance = {var}");
        assert!(close(sd, 2.138089935299395), "stddev = {sd}");
    }

    #[test]
    fn stddev_variance_below_two_values_is_null() {
        // n == 0 (empty input).
        let op = build(vec![], vec![], vec![agg(AggKind::StdDev, Some(load(Ty::Int, 0)))], None);
        assert_eq!(run(op).unwrap()[0][0], Value::Null);

        // n == 1. Sample variance and sample standard deviation are undefined (NULL in duckdb too).
        let steps = batches(vec![vec![i32s(&[Some(5)])]]);
        let op = build(steps, vec![], vec![agg(AggKind::Variance, Some(load(Ty::Int, 0)))], None);
        assert_eq!(run(op).unwrap()[0][0], Value::Null);
    }

    #[test]
    fn variance_of_constant_values_is_exactly_zero() {
        // Confirms that an M2 made slightly negative by rounding is clamped to 0.
        let steps = batches(vec![vec![f64s(&[Some(5.0), Some(5.0), Some(5.0)])]]);
        let op = build(
            steps,
            vec![],
            vec![
                agg(AggKind::Variance, Some(load(Ty::Double, 0))),
                agg(AggKind::StdDev, Some(load(Ty::Double, 0))),
            ],
            None,
        );
        let rows = run(op).unwrap();
        assert_eq!(rows[0][0], Value::F64(0.0));
        assert_eq!(rows[0][1], Value::F64(0.0));
    }

    // --- Median ----------------------------------------------------------------

    #[test]
    fn median_even_and_odd_counts() {
        let steps = batches(vec![vec![i32s(&[Some(1), Some(2), Some(3), Some(4)])]]);
        let op = build(steps, vec![], vec![agg(AggKind::Median, Some(load(Ty::Int, 0)))], None);
        assert_eq!(
            run(op).unwrap()[0][0],
            Value::F64(2.5),
            "an even count interpolates between the middle two"
        );

        let steps = batches(vec![vec![i32s(&[Some(1), Some(2), Some(3), Some(4), Some(5)])]]);
        let op = build(steps, vec![], vec![agg(AggKind::Median, Some(load(Ty::Int, 0)))], None);
        assert_eq!(
            run(op).unwrap()[0][0],
            Value::F64(3.0),
            "an odd count gives exactly the middle value"
        );
    }

    #[test]
    fn median_of_decimal_matches_duckdb_quantile_cont() {
        // DECIMAL(10,2) 1.00,2.00,3.00,4.00 -> duckdb gives median = 2.50 (staying DECIMAL).
        // This engine always fixes Median's output type to DOUBLE (`AggKind::result_ty`), so the
        // value is 2.5 (DOUBLE).
        let ty = Ty::Decimal { precision: 10, scale: 2 };
        let steps = batches(vec![vec![col(
            ty,
            &[
                Some(Value::I64(100)),
                Some(Value::I64(200)),
                Some(Value::I64(300)),
                Some(Value::I64(400)),
            ],
        )]]);
        let op = build(steps, vec![], vec![agg(AggKind::Median, Some(load(ty, 0)))], None);
        assert_eq!(run(op).unwrap()[0][0], Value::F64(2.5));
    }

    #[test]
    fn median_empty_group_is_null() {
        let op = build(vec![], vec![], vec![agg(AggKind::Median, Some(load(Ty::Int, 0)))], None);
        assert_eq!(run(op).unwrap()[0][0], Value::Null);
    }

    // --- Mode --------------------------------------------------------------

    #[test]
    fn mode_has_a_clear_winner() {
        let steps = batches(vec![vec![i32s(&[Some(1), Some(2), Some(2), Some(2), Some(3)])]]);
        let op = build(steps, vec![], vec![agg(AggKind::Mode, Some(load(Ty::Int, 0)))], None);
        assert_eq!(run(op).unwrap()[0][0], Value::I32(2));
    }

    #[test]
    fn mode_tie_breaks_to_first_encountered_value() {
        // The behavior observed in duckdb: on a tie the value appearing first wins.
        // (SELECT mode(x) FROM (SELECT unnest([3,2,2,1,1]) AS x) -> 2)
        let steps = batches(vec![vec![i32s(&[Some(3), Some(2), Some(2), Some(1), Some(1)])]]);
        let op = build(steps, vec![], vec![agg(AggKind::Mode, Some(load(Ty::Int, 0)))], None);
        assert_eq!(run(op).unwrap()[0][0], Value::I32(2), "2 appears before 1");
    }

    #[test]
    fn mode_empty_group_is_null() {
        let op = build(vec![], vec![], vec![agg(AggKind::Mode, Some(load(Ty::Int, 0)))], None);
        assert_eq!(run(op).unwrap()[0][0], Value::Null);
    }

    #[test]
    fn mode_distinct_dedups_before_counting() {
        // With DISTINCT, deduplication takes effect before Mode's frequency table, so every value
        // looks like it appears "exactly once". As in the behavior observed in duckdb, the first
        // distinct value to appear wins (SELECT mode(DISTINCT x) FROM
        // (SELECT unnest([1,1,2,3]) AS x) -> 1).
        let steps = batches(vec![vec![i32s(&[Some(1), Some(1), Some(2), Some(3)])]]);
        let op = build(steps, vec![], vec![agg_distinct(AggKind::Mode, load(Ty::Int, 0))], None);
        assert_eq!(run(op).unwrap()[0][0], Value::I32(1));
    }

    // --- Resumption for Median / Mode ---------------------------------------

    #[test]
    fn need_io_mid_input_does_not_change_median_or_mode_result() {
        let data: Vec<Vec<Option<i32>>> = vec![
            vec![Some(5), Some(1), Some(9)],
            vec![Some(3), Some(7)],
            vec![Some(1), Some(1), Some(8)],
        ];
        let make = |interrupt: bool| {
            let mut steps = Vec::new();
            for (i, v) in data.iter().enumerate() {
                if interrupt && i == 1 {
                    steps.push(MockStep::NeedIo);
                }
                steps.push(MockStep::Rows(Batch::new(vec![i32s(v)])));
                if interrupt && i == 1 {
                    steps.push(MockStep::NeedCodec);
                }
            }
            build(
                steps,
                vec![],
                vec![
                    agg(AggKind::Median, Some(load(Ty::Int, 0))),
                    agg(AggKind::Mode, Some(load(Ty::Int, 0))),
                ],
                None,
            )
        };
        let plain = run(make(false)).unwrap();
        let interrupted = run(make(true)).unwrap();
        assert_eq!(plain, interrupted, "the result must not change across a NeedIo/NeedCodec");
        // The values themselves too: 1,1,1,3,5,7,8,9 -> median = (3+5)/2 = 4.0, mode = 1.
        assert_eq!(plain[0][0], Value::F64(4.0));
        assert_eq!(plain[0][1], Value::I32(1));
    }

    // --- ApproxCountDistinct -------------------------------------------------

    #[test]
    fn approx_count_distinct_matches_count_distinct_exactly() {
        // v1 is an exact count. It should give the same result as COUNT(DISTINCT x).
        let steps = batches(vec![vec![i32s(&[Some(1), Some(1), Some(2), Some(3), None, Some(3)])]]);
        let op = build(
            steps,
            vec![],
            vec![
                agg(AggKind::ApproxCountDistinct, Some(load(Ty::Int, 0))),
                agg_distinct(AggKind::Count, load(Ty::Int, 0)),
            ],
            None,
        );
        let rows = run(op).unwrap();
        assert_eq!(rows[0][0], rows[0][1]);
        assert_eq!(rows[0][0], Value::I64(3));
    }

    #[test]
    fn approx_count_distinct_of_empty_input_is_zero() {
        let op = build(
            vec![],
            vec![],
            vec![agg(AggKind::ApproxCountDistinct, Some(load(Ty::Int, 0)))],
            None,
        );
        assert_eq!(run(op).unwrap()[0][0], Value::I64(0));
    }

    // --- StringAgg -----------------------------------------------------------

    #[test]
    fn string_agg_default_separator_is_empty_and_explicit_separator_is_used() {
        // This file merely reads `a.separator` as given. What fills it in when omitted is the
        // binder's contract (`AggKind::optional_arg_default`), currently the empty string.
        // duckdb's default is a comma, so the value differs, but that is the binder's decision
        // and outside this file's remit.
        let steps = batches(vec![vec![strs(&[Some("a"), Some("b"), Some("c")])]]);
        let op =
            build(steps, vec![], vec![agg(AggKind::StringAgg, Some(load(Ty::Varchar, 0)))], None);
        assert_eq!(run(op).unwrap()[0][0], Value::Bytes(b"abc".to_vec()));

        let steps = batches(vec![vec![strs(&[Some("a"), Some("b"), Some("c")])]]);
        let op = build(
            steps,
            vec![],
            vec![agg_sep(AggKind::StringAgg, load(Ty::Varchar, 0), b",")],
            None,
        );
        assert_eq!(run(op).unwrap()[0][0], Value::Bytes(b"a,b,c".to_vec()));
    }

    #[test]
    fn string_agg_skips_nulls_and_all_null_group_is_null() {
        let steps = batches(vec![vec![strs(&[Some("a"), None, Some("b")])]]);
        let op = build(
            steps,
            vec![],
            vec![agg_sep(AggKind::StringAgg, load(Ty::Varchar, 0), b",")],
            None,
        );
        assert_eq!(run(op).unwrap()[0][0], Value::Bytes(b"a,b".to_vec()));

        let steps = batches(vec![vec![strs(&[None, None])]]);
        let op = build(
            steps,
            vec![],
            vec![agg_sep(AggKind::StringAgg, load(Ty::Varchar, 0), b",")],
            None,
        );
        assert_eq!(run(op).unwrap()[0][0], Value::Null, "NULL when there is not a single non-NULL");
    }

    #[test]
    fn string_agg_distinct_dedups_values() {
        let steps = batches(vec![vec![strs(&[Some("a"), Some("a"), Some("b")])]]);
        let op = build(
            steps,
            vec![],
            vec![Agg {
                kind: AggKind::StringAgg,
                arg: Some(load(Ty::Varchar, 0)),
                distinct: true,
                name: String::from("a"),
                separator: b",".to_vec(),
                filter: None,
            }],
            None,
        );
        let rows = run(op).unwrap();
        let s = match &rows[0][0] {
            Value::Bytes(b) => b.clone(),
            other => panic!("expected bytes but got {other:?}"),
        };
        // DISTINCT's arrival order depends on the hash table's implementation and is not strictly
        // promised, so only membership of {a, b} is checked.
        let mut parts: Vec<&[u8]> = s.split(|&c| c == b',').collect();
        parts.sort();
        assert_eq!(parts, vec![b"a".as_slice(), b"b".as_slice()]);
    }

    // --- ArrayAgg --------------------------------------------------------------

    #[test]
    fn array_agg_integers_and_varchars() {
        let steps = batches(vec![vec![i32s(&[Some(1), Some(2), Some(3)])]]);
        let op = build(steps, vec![], vec![agg(AggKind::ArrayAgg, Some(load(Ty::Int, 0)))], None);
        assert_eq!(run(op).unwrap()[0][0], Value::Bytes(b"[1, 2, 3]".to_vec()));

        let steps = batches(vec![vec![strs(&[Some("a"), Some("b")])]]);
        let op =
            build(steps, vec![], vec![agg(AggKind::ArrayAgg, Some(load(Ty::Varchar, 0)))], None);
        assert_eq!(run(op).unwrap()[0][0], Value::Bytes(b"[\"a\", \"b\"]".to_vec()));
    }

    #[test]
    fn array_agg_includes_null_elements_as_literal_null() {
        // Unlike other aggregates, ArrayAgg counts NULL as an element
        // (duckdb's array_agg/list do the same: [1, NULL, 3]).
        let steps = batches(vec![vec![i32s(&[Some(1), None, Some(3)])]]);
        let op = build(steps, vec![], vec![agg(AggKind::ArrayAgg, Some(load(Ty::Int, 0)))], None);
        assert_eq!(run(op).unwrap()[0][0], Value::Bytes(b"[1, null, 3]".to_vec()));
    }

    #[test]
    fn array_agg_of_f64_uses_shortest_decimal_or_exponential_form() {
        // push_f64_text is never exercised by the INT/VARCHAR array_agg tests. The normal range,
        // the boundary where it switches to exponential notation (p outside 1..=17), and
        // NaN/Infinity are all checked together here. The format matches duckdb's to_json(x).
        let steps = batches(vec![vec![f64s(&[
            Some(1.5),
            Some(1e20),  // p = 21 > 17 -> exponential notation
            Some(1e-10), // p = -9 < 1 -> exponential notation
            Some(f64::NAN),
            Some(f64::INFINITY),
        ])]]);
        let op =
            build(steps, vec![], vec![agg(AggKind::ArrayAgg, Some(load(Ty::Double, 0)))], None);
        let s = match &run(op).unwrap()[0][0] {
            Value::Bytes(b) => b.clone(),
            other => panic!("expected bytes but got {other:?}"),
        };
        assert_eq!(s, b"[1.5, 1e20, 1e-10, NaN, Infinity]".to_vec());
    }

    #[test]
    fn array_agg_of_zero_rows_is_null_not_empty_array() {
        // In duckdb too, array_agg(x) over "0 rows" is NULL (not `[]`).
        // That differs from having rows that are all NULL (which gives `[NULL]`; see
        // `array_agg_includes_null_elements_as_literal_null` above).
        let op = build(vec![], vec![], vec![agg(AggKind::ArrayAgg, Some(load(Ty::Int, 0)))], None);
        assert_eq!(run(op).unwrap()[0][0], Value::Null);
    }

    #[test]
    fn array_agg_distinct_dedups_nulls_too() {
        let steps = batches(vec![vec![i32s(&[Some(1), None, Some(1), None, Some(2)])]]);
        let op =
            build(steps, vec![], vec![agg_distinct(AggKind::ArrayAgg, load(Ty::Int, 0))], None);
        let s = match &run(op).unwrap()[0][0] {
            Value::Bytes(b) => b.clone(),
            other => panic!("expected bytes but got {other:?}"),
        };
        // The order is not promised, so only the set of elements is checked.
        let inner = core::str::from_utf8(&s[1..s.len() - 1]).unwrap();
        let mut parts: Vec<&str> = inner.split(", ").collect();
        parts.sort();
        assert_eq!(parts, vec!["1", "2", "null"]);
    }

    // --- Memory cap (the new operations) ------------------------------------

    #[test]
    fn memory_limit_is_enforced_for_array_agg() {
        // The same idea as MIN/MAX: pile up large byte sequences to hit the cap.
        // This is a single group, so the accumulated JSON-like text itself keeps growing until it
        // exceeds the cap.
        let filler = "x".repeat(1024);
        let mut steps = Vec::new();
        for _ in 0..80 {
            let vals: Vec<Option<String>> = (0..1000).map(|_| Some(filler.clone())).collect();
            let refs: Vec<Option<&str>> = vals.iter().map(|s| s.as_deref()).collect();
            steps.push(MockStep::Rows(Batch::new(vec![strs(&refs)])));
        }
        let op =
            build(steps, vec![], vec![agg(AggKind::ArrayAgg, Some(load(Ty::Varchar, 0)))], None);
        assert_eq!(code_of(run(op)), Some(Code::Oom));
    }

    #[test]
    fn memory_limit_is_enforced_for_median() {
        // median_vals counts 8 bytes per value. Exceeding 64 MiB takes just over 8.38 million
        // values, so 9 million rows are fed in with room to spare.
        let mut steps = Vec::new();
        for _ in 0..90 {
            let vals: Vec<Option<i32>> = (0..100_000).map(Some).collect();
            steps.push(MockStep::Rows(Batch::new(vec![i32s(&vals)])));
        }
        let op = build(steps, vec![], vec![agg(AggKind::Median, Some(load(Ty::Int, 0)))], None);
        assert_eq!(code_of(run(op)), Some(Code::Oom));
    }
}
