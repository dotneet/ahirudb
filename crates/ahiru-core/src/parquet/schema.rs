//! Resolution from Parquet schema to ahirudb types.
//!
//! The footer's `schema` is a flat, depth-first-ordered array of `SchemaElement`s.
//! Each element declares only its own number of children via `num_children` (the
//! actual tree structure only emerges by recursively consuming the next
//! `num_children` elements).
//! This is because the Thrift Compact footer records a preorder traversal, not the
//! tree itself.
//!
//! A STRUCT that contains no REPEATED (i.e. a REQUIRED/OPTIONAL group with
//! children) physically just has an independent column chunk per leaf, so it can be
//! read simply by walking the tree and collecting leaf columns with dot-separated
//! names.
//!
//! A subtree containing REPEATED (LIST/MAP, or a bare REPEATED field) requires
//! decoding the repetition level to map rows to array elements, so it is handled
//! separately. Here we adopt the design of turning a whole subtree that contains
//! REPEATED into a single `Ty::Json` column: `reader::nested` builds the per-row
//! nested structure from the repetition/definition levels using the Dremel method,
//! and serializes it into JSON text (the physical type stays at 6 kinds without
//! adding more. DESIGN.md §8, §11).
//! Therefore only a STRUCT that contains no REPEATED at all continues to be
//! flattened into dot-separated columns (e.g. `address.city`). If a STRUCT contains
//! a LIST/MAP, the whole STRUCT becomes a single JSON column.
//!
//! The SQL-side dot accessor (`SELECT s.field`) is not handled here. What this
//! module provides is only a set of columns with dot-separated names like
//! `address.city`, as if they were flat; how those get bound to SQL is the
//! caller's responsibility.

use crate::parquet::meta::{ColumnMetaData, FileMetaData, SchemaElement};
use crate::parquet::*;
use crate::prelude::*;
use crate::vector::Ty;

/// Depth limit when recursively walking the schema tree. A defense against a
/// corrupted/malicious file exhausting the stack (the same idea as `MAX_DEPTH` in
/// thrift.rs, but kept as a separate constant since the target here is the schema
/// tree, not a Thrift value).
const MAX_SCHEMA_DEPTH: usize = 32;

/// Upper bound on the number of physical leaf columns that can be resolved. Kept
/// aligned with `meta.rs::MAX_SCHEMA_ELEMENTS`. For a real file, the footer's Thrift
/// decoding already stays within that limit, but `resolve_schema` can be called
/// independently of that (from tests, or a future entry point), so we enforce the
/// same limit here as well and never allow unbounded allocation. The number of
/// output (logical) columns is always at most the number of physical leaves, so
/// this single limit protects both.
const MAX_LEAF_COLUMNS: usize = 16_384;

/// Decoding information for one leaf of a nested column (used by `reader::nested`).
/// `ColumnDesc::leaves` is arranged in the same order and length as
/// `ColumnDesc::phys_cols`.
#[derive(Clone, Copy)]
pub struct LeafDecodeInfo {
    pub ptype: PType,
    pub type_length: usize,
    pub time_unit: Option<TimeUnit>,
    pub ty: Ty,
    /// Number of OPTIONAL+REPEATED elements on the path to this leaf (including
    /// itself, counted from the root of the nested column). Used to determine
    /// whether a value is present.
    pub max_def_level: u16,
    /// Number of REPEATED elements on the path to this leaf (including itself).
    /// Used to determine the bit width of the repetition level stream in the page.
    pub max_rep_level: u16,
}

/// A node of the nested schema tree (a subtree that contains REPEATED).
pub struct NestedNode {
    /// Used to render as a STRUCT field name (not used for the array element itself).
    pub name: String,
    pub repetition: Repetition,
    /// Cumulative count of OPTIONAL+REPEATED elements on the path to this node
    /// (including itself, counted from the root of the nested column).
    pub def_depth: u16,
    /// Cumulative count of REPEATED elements on the path to this node (including itself).
    pub rep_depth: u16,
    pub content: NestedContent,
    /// Index into the `leaves` array of the representative leaf used for presence
    /// checks and repetition-boundary determination. Points to the first leaf under
    /// this node (any leaf under this node gives the same answer for this node's
    /// presence and repeat count).
    pub rep_leaf: usize,
}

pub enum NestedContent {
    /// A physical leaf column. The value is fetched from `leaves[index]`.
    Leaf(usize),
    /// A child node array. If non-REPEATED, represents named fields (STRUCT); if
    /// REPEATED, represents "the contents of one array element" (the intermediate
    /// group of the 3-level/2-level encoding, or a MAP's key/value).
    Group(Vec<NestedNode>),
}

/// Information needed to read a single leaf column.
pub struct ColumnDesc {
    /// A leaf under a STRUCT uses a dot-separated name (`address.city`).
    pub name: String,
    pub ty: Ty,
    pub nullable: bool,
    /// Number of OPTIONAL elements on the path to this leaf (including the leaf
    /// itself). In a STRUCT chain with no REPEATED, this can be used directly as
    /// the definition-level upper bound for "level matches => value present"
    /// (whichever intermediate group is NULL, it gets cut off before reaching the
    /// leaf and collapses to the same single validity bit). 0 or 1 for a flat
    /// column. Unused (always 0) for a column where `nested` is `Some`.
    pub max_def_level: u16,
    pub ptype: PType,
    /// Byte length for FIXED_LEN_BYTE_ARRAY. 0 otherwise.
    /// Unused for a column where `nested` is `Some`.
    pub type_length: usize,
    /// On-file resolution of TIME / TIMESTAMP. Normalized to microseconds when read.
    /// Unused for a column where `nested` is `Some`.
    pub time_unit: Option<TimeUnit>,
    /// Physical column chunk number(s) (index into `row_group.columns`). A flat
    /// column always has exactly one. A nested column has multiple, one for each
    /// leaf of the subtree (read order matches the order of `leaves`).
    pub phys_cols: Vec<usize>,
    /// Per-leaf decoding information for a nested column, arranged in the same
    /// order and length as `phys_cols`. Empty for a flat column.
    pub leaves: Vec<LeafDecodeInfo>,
    /// Structure of a subtree containing REPEATED. If `Some`, `ty == Ty::Json`,
    /// and reading uses `reader::nested`'s Dremel-assembly path. If `None`, this is
    /// a plain flat read as before.
    pub nested: Option<Box<NestedNode>>,
}

pub struct ParquetSchema {
    pub columns: Vec<ColumnDesc>,
}

impl ParquetSchema {
    /// Look up a column by name (case-insensitive). A column under a STRUCT is
    /// looked up by its full dot-separated name, like `address.city`.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.columns
            .iter()
            .position(|c| crate::rt::hash::eq_ascii_ci(c.name.as_bytes(), name.as_bytes()))
    }

    /// Whether `prefix` exists as a dot-separated-name prefix of a STRUCT column.
    ///
    /// `index_of("address")` returns `None` because `address` itself is not found
    /// (it is not a leaf). This check is provided separately so that DESCRIBE and
    /// error-message generation can distinguish "that name is a STRUCT, not a
    /// column". The SQL dot-accessor syntax itself is not handled here.
    pub fn is_struct_prefix(&self, prefix: &str) -> bool {
        self.columns.iter().any(|c| {
            let name = c.name.as_bytes();
            let p = prefix.as_bytes();
            name.len() > p.len()
                && name[p.len()] == b'.'
                && crate::rt::hash::eq_ascii_ci(&name[..p.len()], p)
        })
    }
}

/// Resolves leaf columns from the footer's schema element array (depth-first order).
///
/// Consumes the `nchildren` elements directly under the root, in order. If an
/// element is a STRUCT group, its children are consumed too, recursing through the
/// whole tree.
pub fn resolve_schema(md: &FileMetaData) -> Result<ParquetSchema> {
    ensure!(!md.schema.is_empty(), BadThrift);
    let root = &md.schema[0];
    let nchildren = root.num_children.unwrap_or(0);
    ensure!(nchildren >= 0, BadThrift);

    let mut columns = Vec::new();
    let mut pos = 1usize;
    let mut phys = 0usize;
    for _ in 0..nchildren {
        pos = resolve_node(md, pos, "", 0, 0, &mut phys, &mut columns)?;
    }
    // A mismatch between the declared top-level child count and the number of
    // elements actually consumed (including subtrees) indicates a corrupted or
    // deliberately crafted schema.
    ensure!(pos == md.schema.len(), UnsupportedNested);
    Ok(ParquetSchema { columns })
}

/// Consumes one node's worth of the schema tree (just itself if a leaf, the whole
/// subtree if a group), and returns the next position to consume.
///
/// - `prefix` is the ancestor group names joined with `.` so far (an empty string
///   at the top level).
/// - `parent_def_level` is the number of OPTIONAL elements among the ancestors
///   (not including itself).
/// - `phys` is the number of physical leaf columns resolved so far (the next
///   index into `row_group.columns`). It is advanced every time a leaf is
///   consumed, whether flat or nested.
///
fn resolve_node(
    md: &FileMetaData,
    pos: usize,
    prefix: &str,
    parent_def_level: u16,
    depth: usize,
    phys: &mut usize,
    out: &mut Vec<ColumnDesc>,
) -> Result<usize> {
    ensure!(depth < MAX_SCHEMA_DEPTH, NestingTooDeep);
    // Running out of elements before consuming the declared number of children means the schema is corrupted.
    ensure!(pos < md.schema.len(), UnsupportedNested);
    let e = &md.schema[pos];

    let mut name = String::with_capacity(prefix.len() + 1 + e.name.len());
    if !prefix.is_empty() {
        name.push_str(prefix);
        name.push('.');
    }
    name.push_str(&e.name);

    // If REPEATED appears anywhere in the subtree (including the node itself being
    // REPEATED), don't flatten as a STRUCT -- turn this whole subtree into a single
    // JSON column instead.
    let (has_repeated, _) = scan_repeated(md, pos, depth)?;
    if has_repeated {
        let mut leaves = Vec::new();
        let (node, next_pos) = build_nested_node(
            md,
            pos,
            name.clone(),
            parent_def_level,
            0,
            depth,
            phys,
            &mut leaves,
        )?;
        let nullable = node.repetition == Repetition::Optional;
        let phys_cols: Vec<usize> = (0..leaves.len()).map(|i| leaves[i].0).collect();
        let leaf_info: Vec<LeafDecodeInfo> = leaves.into_iter().map(|(_, info)| info).collect();
        out.push(ColumnDesc {
            name,
            ty: Ty::Json,
            nullable,
            max_def_level: 0,
            ptype: PType::Boolean,
            type_length: 0,
            time_unit: None,
            phys_cols,
            leaves: leaf_info,
            nested: Some(Box::new(node)),
        });
        return Ok(next_pos);
    }

    let is_optional = e.repetition != Some(Repetition::Required);
    let def_level = parent_def_level + u16::from(is_optional);

    let nchildren = e.num_children.unwrap_or(0);
    ensure!(nchildren >= 0, BadThrift);

    if nchildren > 0 {
        // Having children means this is a STRUCT group. Prefix child column names with this group's name.
        let mut child_pos = pos + 1;
        for _ in 0..nchildren {
            child_pos = resolve_node(md, child_pos, &name, def_level, depth + 1, phys, out)?;
        }
        return Ok(child_pos);
    }

    out.push(resolve_leaf(e, name, def_level, phys)?);
    Ok(pos + 1)
}

/// Whether the subtree at `pos` (including itself) contains any REPEATED element.
/// Returns `(found, next position consumed)`. The caller uses this function's
/// returned position only when `has_repeated` is false; when `has_repeated` is
/// true, `build_nested_node` walks the same subtree again to build the body
/// (the footer is small, so the cost of the double traversal is negligible).
fn scan_repeated(md: &FileMetaData, pos: usize, depth: usize) -> Result<(bool, usize)> {
    ensure!(depth < MAX_SCHEMA_DEPTH, NestingTooDeep);
    ensure!(pos < md.schema.len(), UnsupportedNested);
    let e = &md.schema[pos];
    let mut found = e.repetition == Some(Repetition::Repeated);
    let nchildren = e.num_children.unwrap_or(0);
    ensure!(nchildren >= 0, BadThrift);
    let mut p = pos + 1;
    for _ in 0..nchildren {
        let (f, next) = scan_repeated(md, p, depth + 1)?;
        found |= f;
        p = next;
    }
    Ok((found, p))
}

/// Builds a subtree containing REPEATED into a `NestedNode`. Every time a leaf is
/// encountered, `phys` is advanced and `(physical column number, decode info)` is
/// appended to `leaves` (append order matches the index order that
/// `NestedContent::Leaf` points to).
#[allow(clippy::too_many_arguments)]
fn build_nested_node(
    md: &FileMetaData,
    pos: usize,
    name: String,
    parent_def: u16,
    parent_rep: u16,
    depth: usize,
    phys: &mut usize,
    leaves: &mut Vec<(usize, LeafDecodeInfo)>,
) -> Result<(NestedNode, usize)> {
    ensure!(depth < MAX_SCHEMA_DEPTH, NestingTooDeep);
    ensure!(pos < md.schema.len(), UnsupportedNested);
    let e = &md.schema[pos];
    let repetition = e.repetition.unwrap_or(Repetition::Required);
    let is_optional_ish = repetition != Repetition::Required;
    let def_depth = parent_def + u16::from(is_optional_ish);
    let is_repeated = repetition == Repetition::Repeated;
    let rep_depth = parent_rep + u16::from(is_repeated);

    let nchildren = e.num_children.unwrap_or(0);
    ensure!(nchildren >= 0, BadThrift);

    if nchildren > 0 {
        let mut children = Vec::with_capacity(nchildren as usize);
        let mut p = pos + 1;
        for _ in 0..nchildren {
            ensure!(p < md.schema.len(), UnsupportedNested);
            let child_name = md.schema[p].name.clone();
            let (child, next) = build_nested_node(
                md,
                p,
                child_name,
                def_depth,
                rep_depth,
                depth + 1,
                phys,
                leaves,
            )?;
            children.push(child);
            p = next;
        }
        // A group with zero children (nchildren == 0 never reaches here) can't
        // happen, but the situation where `first()` is `None` (a corrupted schema
        // where nchildren > 0 yet no children were consumed) is already rejected
        // earlier by the `ensure!` in the loop above.
        let rep_leaf = children.first().map(|c| c.rep_leaf).unwrap_or(0);
        Ok((
            NestedNode {
                name,
                repetition,
                def_depth,
                rep_depth,
                content: NestedContent::Group(children),
                rep_leaf,
            },
            p,
        ))
    } else {
        ensure!(*phys < MAX_LEAF_COLUMNS, UnsupportedNested);
        let ptype = match e.ptype {
            Some(p) => p,
            None => err!(UnsupportedNested),
        };
        let (ty, time_unit) = map_type(e, ptype)?;
        let leaf_index = leaves.len();
        let info = LeafDecodeInfo {
            ptype,
            type_length: e.type_length.unwrap_or(0).max(0) as usize,
            time_unit,
            ty,
            max_def_level: def_depth,
            max_rep_level: rep_depth,
        };
        leaves.push((*phys, info));
        *phys += 1;
        Ok((
            NestedNode {
                name,
                repetition,
                def_depth,
                rep_depth,
                content: NestedContent::Leaf(leaf_index),
                rep_leaf: leaf_index,
            },
            pos + 1,
        ))
    }
}

/// Turns a single leaf element into a `ColumnDesc`. `max_def_level` is passed in
/// already computed from the whole path by the caller (either `resolve_node`'s
/// recursion, or `resolve_column`, which handles a standalone flat column). `phys`
/// is advanced by one after assigning this leaf's physical column number.
fn resolve_leaf(
    e: &SchemaElement,
    name: String,
    max_def_level: u16,
    phys: &mut usize,
) -> Result<ColumnDesc> {
    let ptype = match e.ptype {
        Some(p) => p,
        // No physical type = a group element. The caller (`resolve_node`) already
        // branches as a group when `num_children > 0`, so reaching here means a
        // corrupted element with `num_children == 0` and no physical type either.
        None => err!(UnsupportedNested),
    };
    ensure!(*phys < MAX_LEAF_COLUMNS, UnsupportedNested);
    let nullable = e.repetition != Some(Repetition::Required);
    let (ty, time_unit) = map_type(e, ptype)?;
    let col_index = *phys;
    *phys += 1;
    Ok(ColumnDesc {
        name,
        ty,
        nullable,
        max_def_level,
        ptype,
        type_length: e.type_length.unwrap_or(0).max(0) as usize,
        time_unit,
        phys_cols: vec![col_index],
        leaves: Vec::new(),
        nested: None,
    })
}

/// Resolves a standalone flat element as a top-level column. Since it has no
/// ancestors, `max_def_level` is determined solely by its own OPTIONAL/REQUIRED
/// (the same computation as `resolve_node` with `prefix == ""`,
/// `parent_def_level == 0`). `resolve_schema` uses `resolve_node`/`resolve_leaf`
/// directly to walk the tree, so currently only unit tests call this (useful when
/// you want to verify a single element in isolation without building a tree).
#[cfg(test)]
fn resolve_column(e: &SchemaElement) -> Result<ColumnDesc> {
    let max_def_level = u16::from(e.repetition != Some(Repetition::Required));
    resolve_leaf(e, e.name.clone(), max_def_level, &mut 0)
}

/// Decides the SQL type in priority order: logical type -> converted type -> physical type.
fn map_type(e: &SchemaElement, ptype: PType) -> Result<(Ty, Option<TimeUnit>)> {
    // We don't validate consistency between logical type and physical type (writer
    // behavior varies too much). Whether conversion is actually possible is decided
    // by the reader based on the (ptype, ty) pair.
    if let Some(l) = e.logical {
        return map_logical(l);
    }
    if let Some(c) = e.converted_type {
        return map_converted(c, e);
    }
    Ok((map_physical(ptype)?, None))
}

fn map_logical(l: LogicalType) -> Result<(Ty, Option<TimeUnit>)> {
    use LogicalType as L;
    Ok(match l {
        L::String | L::Enum | L::Json => (Ty::Varchar, None),
        L::Bson => (Ty::Blob, None),
        // UUID's physical representation is the raw bytes of FLBA(16) -- the same
        // `Bytes` family as `Ty::Blob` -- but differs only in that text
        // display/parsing uses the hyphenated hex format (see the `Ty::Uuid` doc).
        L::Uuid => (Ty::Uuid, None),
        L::Decimal { scale, precision } => {
            ensure!((1..=38).contains(&precision), UnsupportedType);
            ensure!((0..=precision).contains(&scale), UnsupportedType);
            (Ty::Decimal { precision: precision as u8, scale: scale as u8 }, None)
        }
        L::Date => (Ty::Date, None),
        L::Time { unit, .. } => (Ty::Time, Some(unit)),
        // If `utc` (`isAdjustedToUTC`) is true, this column already represents a UTC
        // instant timestamp (`Ty::Timestamptz`). If false, it's a plain date/time
        // with no timezone (`Ty::Timestamp`). The legacy `ConvertedType` path
        // (`map_converted`) has no such bit, so that path always stays
        // `Ty::Timestamp`.
        L::Timestamp { unit, utc } => {
            (if utc { Ty::Timestamptz } else { Ty::Timestamp }, Some(unit))
        }
        L::Integer { bit_width, signed } => (int_ty(bit_width, signed)?, None),
        L::Unknown => (Ty::Null, None),
        // LIST / MAP are handled by build_nested_node as the entry point for "a
        // subtree containing REPEATED", so reaching here means the annotation is
        // attached directly to a leaf (a corrupted schema).
        L::List | L::Map => err!(UnsupportedNested),
        // FLOAT16 is FLBA(2) holding an IEEE binary16. Every binary16 value is
        // representable in binary32, so it widens to `Ty::Float` losslessly
        // (`reader::push_byte_values` does the conversion).
        L::Float16 => (Ty::Float, None),
    })
}

fn map_converted(c: ConvertedType, e: &SchemaElement) -> Result<(Ty, Option<TimeUnit>)> {
    use ConvertedType as C;
    Ok(match c {
        C::Utf8 | C::Json | C::Enum => (Ty::Varchar, None),
        C::Bson => (Ty::Blob, None),
        C::Decimal => {
            let precision = e.precision.unwrap_or(18);
            let scale = e.scale.unwrap_or(0);
            ensure!((1..=38).contains(&precision), UnsupportedType);
            ensure!((0..=precision).contains(&scale), UnsupportedType);
            (Ty::Decimal { precision: precision as u8, scale: scale as u8 }, None)
        }
        C::Date => (Ty::Date, None),
        C::TimeMillis => (Ty::Time, Some(TimeUnit::Millis)),
        C::TimeMicros => (Ty::Time, Some(TimeUnit::Micros)),
        C::TimestampMillis => (Ty::Timestamp, Some(TimeUnit::Millis)),
        C::TimestampMicros => (Ty::Timestamp, Some(TimeUnit::Micros)),
        C::Uint8 => (Ty::UTinyInt, None),
        C::Uint16 => (Ty::USmallInt, None),
        C::Uint32 => (Ty::UInt, None),
        C::Uint64 => (Ty::UBigInt, None),
        C::Int8 => (Ty::TinyInt, None),
        C::Int16 => (Ty::SmallInt, None),
        C::Int32 => (Ty::Int, None),
        C::Int64 => (Ty::BigInt, None),
        // LIST/MAP are handled at the build_nested_node entry point (see the comment above).
        C::List | C::Map | C::MapKeyValue => err!(UnsupportedNested),
        // INTERVAL is FLBA(12): months, days and milliseconds, each an unsigned
        // 32-bit little-endian integer (`reader::push_byte_values` repacks it into
        // the engine's months/days/micros layout).
        C::Interval => (Ty::Interval, None),
    })
}

fn map_physical(ptype: PType) -> Result<Ty> {
    Ok(match ptype {
        PType::Boolean => Ty::Boolean,
        PType::Int32 => Ty::Int,
        PType::Int64 => Ty::BigInt,
        // INT96 is deprecated but still shows up in files originating from Hive/Spark.
        // Interpreted as a nanosecond-precision timestamp.
        PType::Int96 => Ty::Timestamp,
        PType::Float => Ty::Float,
        PType::Double => Ty::Double,
        PType::ByteArray | PType::FixedLenByteArray => Ty::Blob,
    })
}

fn int_ty(bit_width: u8, signed: bool) -> Result<Ty> {
    Ok(match (bit_width, signed) {
        (8, true) => Ty::TinyInt,
        (16, true) => Ty::SmallInt,
        (32, true) => Ty::Int,
        (64, true) => Ty::BigInt,
        (8, false) => Ty::UTinyInt,
        (16, false) => Ty::USmallInt,
        (32, false) => Ty::UInt,
        (64, false) => Ty::UBigInt,
        _ => err!(UnsupportedType),
    })
}

/// Confirms that the column chunk is neither encrypted nor an external file reference.
pub fn check_chunk_supported(meta: &ColumnMetaData) -> Result<()> {
    ensure!(meta.codec != Compression::Lzo, UnsupportedCodec);
    ensure!(meta.codec != Compression::Lz4, UnsupportedCodec);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::code_of;
    use crate::parquet::meta::SchemaElement;

    fn elem(name: &str, ptype: Option<PType>, rep: Repetition) -> SchemaElement {
        SchemaElement {
            ptype,
            type_length: None,
            repetition: Some(rep),
            name: name.into(),
            num_children: None,
            converted_type: None,
            scale: None,
            precision: None,
            logical: None,
        }
    }

    fn root(n: i32) -> SchemaElement {
        let mut e = elem("root", None, Repetition::Required);
        e.num_children = Some(n);
        e
    }

    #[test]
    fn flat_schema_resolves() {
        let md = FileMetaData {
            version: 2,
            schema: vec![
                root(2),
                elem("a", Some(PType::Int32), Repetition::Required),
                elem("b", Some(PType::ByteArray), Repetition::Optional),
            ],
            num_rows: 0,
            row_groups: Vec::new(),
            created_by: None,
        };
        let s = resolve_schema(&md).unwrap();
        assert_eq!(s.columns.len(), 2);
        assert_eq!(s.columns[0].ty, Ty::Int);
        assert!(!s.columns[0].nullable);
        assert_eq!(s.columns[0].max_def_level, 0);
        assert_eq!(s.columns[1].ty, Ty::Blob);
        assert!(s.columns[1].nullable);
        assert_eq!(s.columns[1].max_def_level, 1);
        assert_eq!(s.index_of("A"), Some(0));
        assert_eq!(s.columns[0].phys_cols, vec![0]);
        assert_eq!(s.columns[1].phys_cols, vec![1]);
    }

    #[test]
    fn logical_types_take_precedence() {
        let mut e = elem("t", Some(PType::Int64), Repetition::Optional);
        e.converted_type = Some(ConvertedType::TimestampMillis);
        e.logical = Some(LogicalType::Timestamp { utc: true, unit: TimeUnit::Nanos });
        let c = resolve_column(&e).unwrap();
        // `logical` wins over `converted_type` (which would have produced
        // plain `Ty::Timestamp` with no way to signal `isAdjustedToUTC`).
        // `logical`'s `utc: true` additionally means this resolves to
        // `Ty::Timestamptz`, not `Ty::Timestamp`.
        assert_eq!(c.ty, Ty::Timestamptz);
        assert_eq!(c.time_unit, Some(TimeUnit::Nanos));
    }

    #[test]
    fn timestamp_without_utc_flag_stays_plain_timestamp() {
        let mut e = elem("t", Some(PType::Int64), Repetition::Optional);
        e.logical = Some(LogicalType::Timestamp { utc: false, unit: TimeUnit::Micros });
        let c = resolve_column(&e).unwrap();
        assert_eq!(c.ty, Ty::Timestamp);
    }

    #[test]
    fn uuid_logical_type_maps_to_ty_uuid() {
        let mut e = elem("u", Some(PType::FixedLenByteArray), Repetition::Optional);
        e.type_length = Some(16);
        e.logical = Some(LogicalType::Uuid);
        let c = resolve_column(&e).unwrap();
        assert_eq!(c.ty, Ty::Uuid);
    }

    #[test]
    fn utf8_converted_type_maps_to_varchar() {
        let mut e = elem("s", Some(PType::ByteArray), Repetition::Optional);
        e.converted_type = Some(ConvertedType::Utf8);
        assert_eq!(resolve_column(&e).unwrap().ty, Ty::Varchar);
    }

    #[test]
    fn unsigned_ints_widen() {
        let mut e = elem("u", Some(PType::Int32), Repetition::Required);
        e.logical = Some(LogicalType::Integer { bit_width: 32, signed: false });
        let c = resolve_column(&e).unwrap();
        assert_eq!(c.ty, Ty::UInt);
        assert_eq!(c.ty.phys(), crate::vector::PhysType::I64);
    }

    #[test]
    fn malformed_schema_is_rejected_explicitly() {
        let mut group = elem("g", None, Repetition::Optional);
        group.num_children = Some(1);
        let md = FileMetaData {
            version: 2,
            schema: vec![root(1), group],
            num_rows: 0,
            row_groups: Vec::new(),
            created_by: None,
        };
        assert_eq!(code_of(resolve_schema(&md)), Some(Code::UnsupportedNested));
    }

    #[test]
    fn decimal_precision_is_validated() {
        let mut e = elem("d", Some(PType::FixedLenByteArray), Repetition::Required);
        e.logical = Some(LogicalType::Decimal { scale: 2, precision: 40 });
        assert_eq!(code_of(resolve_column(&e)), Some(Code::UnsupportedType));

        e.logical = Some(LogicalType::Decimal { scale: 2, precision: 10 });
        assert_eq!(resolve_column(&e).unwrap().ty, Ty::Decimal { precision: 10, scale: 2 });
    }

    // --- STRUCT / depth and width limits with synthetic schemas ------------------------

    /// A single-level OPTIONAL group with one child, plus one non-STRUCT leaf.
    /// Confirms, with a minimal setup and no real file involved, that the name
    /// becomes dot-separated and that `max_def_level` accumulates across ancestors.
    #[test]
    fn single_level_struct_resolves_to_dotted_names() {
        let mut group = elem("address", None, Repetition::Optional);
        group.num_children = Some(2);
        let md = FileMetaData {
            version: 2,
            schema: vec![
                root(2),
                elem("id", Some(PType::Int32), Repetition::Required),
                group,
                elem("city", Some(PType::ByteArray), Repetition::Optional),
                elem("zip", Some(PType::Int32), Repetition::Optional),
            ],
            num_rows: 0,
            row_groups: Vec::new(),
            created_by: None,
        };
        let s = resolve_schema(&md).unwrap();
        let names: Vec<&str> = s.columns.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["id", "address.city", "address.zip"]);
        assert_eq!(s.columns[0].max_def_level, 0); // id is REQUIRED.
                                                   // 2 levels: address (OPTIONAL) + city (OPTIONAL).
        assert_eq!(s.columns[1].max_def_level, 2);
        assert_eq!(s.columns[2].max_def_level, 2);
        assert_eq!(s.index_of("address.city"), Some(1));
        assert_eq!(s.index_of("address"), None); // Not a leaf, so it can't be looked up.
        assert!(s.is_struct_prefix("address"));
        assert!(s.is_struct_prefix("ADDRESS")); // Case-insensitive.
        assert!(!s.is_struct_prefix("id"));
    }

    /// A REQUIRED ancestor does not contribute to the definition level.
    #[test]
    fn required_ancestor_does_not_add_to_def_level() {
        let mut group = elem("g", None, Repetition::Required);
        group.num_children = Some(1);
        let md = FileMetaData {
            version: 2,
            schema: vec![root(1), group, elem("leaf", Some(PType::Int32), Repetition::Optional)],
            num_rows: 0,
            row_groups: Vec::new(),
            created_by: None,
        };
        let s = resolve_schema(&md).unwrap();
        assert_eq!(s.columns[0].name, "g.leaf");
        // g is REQUIRED so it contributes nothing; only the 1 from leaf's own OPTIONAL.
        assert_eq!(s.columns[0].max_def_level, 1);
    }

    /// A group nested more than 32 levels deep is rejected as `NestingTooDeep`,
    /// not a stack overflow.
    #[test]
    fn pathologically_deep_schema_is_rejected_not_overflowed() {
        let mut schema = vec![root(1)];
        // 40 levels of OPTIONAL groups plus a terminal leaf. Exceeds MAX_SCHEMA_DEPTH (32).
        for _ in 0..40 {
            let mut g = elem("g", None, Repetition::Optional);
            g.num_children = Some(1);
            schema.push(g);
        }
        schema.push(elem("leaf", Some(PType::Int32), Repetition::Optional));
        let md = FileMetaData {
            version: 2,
            schema,
            num_rows: 0,
            row_groups: Vec::new(),
            created_by: None,
        };
        assert_eq!(code_of(resolve_schema(&md)), Some(Code::NestingTooDeep));
    }

    /// A wide schema whose leaf count exceeds the limit is rejected before allocation becomes unbounded.
    #[test]
    fn pathologically_wide_schema_is_rejected_via_leaf_cap() {
        let n = MAX_LEAF_COLUMNS + 10;
        let mut schema = vec![root(n as i32)];
        for _ in 0..n {
            schema.push(elem("c", Some(PType::Int32), Repetition::Required));
        }
        let md = FileMetaData {
            version: 2,
            schema,
            num_rows: 0,
            row_groups: Vec::new(),
            created_by: None,
        };
        assert_eq!(code_of(resolve_schema(&md)), Some(Code::UnsupportedNested));
    }

    /// A bare REPEATED leaf (no LIST/MAP annotation, no STRUCT wrapper -- the
    /// oldest form of a repeated field) can also be resolved as a single JSON column.
    #[test]
    fn bare_repeated_leaf_becomes_json_column() {
        let md = FileMetaData {
            version: 2,
            schema: vec![root(1), elem("xs", Some(PType::Int32), Repetition::Repeated)],
            num_rows: 0,
            row_groups: Vec::new(),
            created_by: None,
        };
        let s = resolve_schema(&md).unwrap();
        assert_eq!(s.columns.len(), 1);
        assert_eq!(s.columns[0].name, "xs");
        assert_eq!(s.columns[0].ty, Ty::Json);
        // A bare REPEATED can never be SQL NULL (it can only be an empty array).
        assert!(!s.columns[0].nullable);
        assert_eq!(s.columns[0].phys_cols, vec![0]);
        let node = s.columns[0].nested.as_ref().unwrap();
        assert_eq!(node.repetition, Repetition::Repeated);
    }

    // --- End-to-end verification through real files (DuckDB output) ------------
    //
    // Everything below this point actually reads tests/data/*.parquet via
    // `file.rs::open_bytes` / `reader.rs::read_column_chunk`, and cross-checks
    // against DuckDB's output (`duckdb -c "SELECT ..."` /
    // `duckdb -c "SELECT ... FROM parquet_schema(...)"`). Whether the definition
    // level computation lines up with the reader side's
    // `validity = (def_level == max_def_level)` cannot be confirmed from the unit
    // tests' synthetic schemas alone (it only becomes clear once the actual
    // definition-level bit stream written into a real page is decoded), so real
    // byte data is read here.
    //

    use crate::parquet::file::open_bytes;
    use crate::parquet::reader::{read_column_chunk, NoPageCache};
    use crate::vector::{Value, Vector};

    fn data_path(name: &str) -> String {
        let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/");
        std::format!("{p}{name}")
    }

    fn read_bytes(name: &str) -> Vec<u8> {
        std::fs::read(data_path(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
    }

    /// Reads the whole file column by column and concatenates vertically across
    /// RowGroups (the same procedure as
    /// `crates/ahiru-core/tests/parquet_files.rs`). Nested columns
    /// (`nested.is_some()`) are not included (the caller uses
    /// `crate::parquet::reader::nested::read_nested_column` separately for those).
    fn read_all(bytes: &[u8]) -> (Vec<String>, Vec<Vector>) {
        let f = open_bytes(bytes).expect("open");
        let names: Vec<String> = f.schema.columns.iter().map(|c| c.name.clone()).collect();
        let mut cols: Vec<Option<Vector>> = (0..f.schema.columns.len()).map(|_| None).collect();

        for rg in &f.meta.row_groups {
            for (i, desc) in f.schema.columns.iter().enumerate() {
                if desc.nested.is_some() {
                    continue;
                }
                let meta = rg.columns[desc.phys_cols[0]].meta.as_ref().expect("column metadata");
                let (start, end) = meta.byte_range();
                let v = read_column_chunk(
                    desc,
                    meta,
                    &bytes[start as usize..end as usize],
                    start,
                    rg.num_rows as usize,
                    &NoPageCache,
                )
                .unwrap_or_else(|e| panic!("column {}: {e}", desc.name));
                match &mut cols[i] {
                    Some(acc) => {
                        let mut merged = Vector::with_capacity(desc.ty, acc.len() + v.len());
                        for k in 0..acc.len() {
                            merged.push_value(&acc.value_at(k));
                        }
                        for k in 0..v.len() {
                            merged.push_value(&v.value_at(k));
                        }
                        *acc = merged;
                    }
                    None => cols[i] = Some(v),
                }
            }
        }
        (names, cols.into_iter().flatten().collect())
    }

    #[test]
    fn struct_of_scalars_matches_duckdb() {
        // duckdb schema: id INTEGER, address STRUCT(city VARCHAR, zip INTEGER)
        // Both OPTIONAL, city/zip also OPTIONAL (confirmed via parquet_schema()).
        let bytes = read_bytes("struct1.parquet");
        let f = open_bytes(&bytes).expect("open struct1.parquet");
        let names: Vec<&str> = f.schema.columns.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["id", "address.city", "address.zip"]);
        assert_eq!(f.schema.columns[1].ty, Ty::Varchar);
        assert_eq!(f.schema.columns[2].ty, Ty::Int);
        assert!(f.schema.columns[1].nullable);
        assert_eq!(f.schema.columns[1].max_def_level, 2);
        assert_eq!(f.schema.index_of("address.city"), Some(1));
        assert!(f.schema.is_struct_prefix("address"));

        let (_, cols) = read_all(&bytes);
        assert_eq!(cols[0].len(), 100);
        // duckdb -c "SELECT id, address.city, address.zip FROM 'struct1.parquet' ORDER BY id LIMIT 5;"
        assert_eq!(cols[0].value_at(0), Value::I32(0));
        assert_eq!(cols[1].value_at(0), Value::Bytes(b"Tokyo".to_vec()));
        assert_eq!(cols[2].value_at(0), Value::I32(10000));
        assert_eq!(cols[0].value_at(99), Value::I32(99));
        assert_eq!(cols[1].value_at(99), Value::Bytes(b"Tokyo".to_vec()));
        assert_eq!(cols[2].value_at(99), Value::I32(10099));
    }

    #[test]
    fn deeply_nested_struct_matches_duckdb() {
        // duckdb schema: id INTEGER, nested STRUCT(a STRUCT(b STRUCT(c INTEGER)))
        // nested/a/b/c are all OPTIONAL (confirmed via parquet_schema()).
        let bytes = read_bytes("struct_deep.parquet");
        let f = open_bytes(&bytes).expect("open struct_deep.parquet");
        let names: Vec<&str> = f.schema.columns.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["id", "nested.a.b.c"]);
        assert_eq!(f.schema.columns[1].max_def_level, 4);

        let (_, cols) = read_all(&bytes);
        assert_eq!(cols[0].len(), 20);
        // duckdb -c "SELECT id, nested.a.b.c FROM 'struct_deep.parquet' ORDER BY id;"
        for i in 0..20usize {
            assert_eq!(cols[0].value_at(i), Value::I32(i as i32), "id[{i}]");
            assert_eq!(cols[1].value_at(i), Value::I32(i as i32), "nested.a.b.c[{i}]");
        }
    }

    #[test]
    fn null_struct_nulls_every_leaf_for_that_row() {
        // duckdb: CASE WHEN i % 3 = 0 THEN NULL ELSE {...} END AS address
        // In other words, there are 10 out of 30 rows where the address column itself
        // is NULL (`duckdb -c "SELECT count(*) FROM 'struct_null.parquet' WHERE address
        // IS NULL"` = 10). If the definition-level insight is correct, city/zip should
        // both be NULL on the same rows, and both should have values on every other row.
        let bytes = read_bytes("struct_null.parquet");
        let (_, cols) = read_all(&bytes);
        assert_eq!(cols[0].len(), 30);

        let mut null_rows = 0usize;
        for i in 0..30usize {
            let expect_null = i % 3 == 0;
            if expect_null {
                null_rows += 1;
            }
            assert_eq!(cols[1].value_at(i) == Value::Null, expect_null, "city[{i}]");
            assert_eq!(cols[2].value_at(i) == Value::Null, expect_null, "zip[{i}]");
            if !expect_null {
                assert_eq!(cols[1].value_at(i), Value::Bytes(b"Tokyo".to_vec()), "city[{i}]");
                assert_eq!(cols[2].value_at(i), Value::I32(10000 + i as i32), "zip[{i}]");
            }
        }
        assert_eq!(null_rows, 10);
    }

    #[test]
    fn list_column_resolves_as_one_json_column() {
        // duckdb schema: id INTEGER, xs INTEGER[]
        // xs is an OPTIONAL group(LIST) -> REPEATED group(list) -> OPTIONAL
        // leaf(element). This entire thing becomes a single JSON column.
        let bytes = read_bytes("list1.parquet");
        let f = open_bytes(&bytes).expect("open list1.parquet");
        let names: Vec<&str> = f.schema.columns.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["id", "xs"]);
        assert_eq!(f.schema.columns[1].ty, Ty::Json);
        assert!(f.schema.columns[1].nullable);
        assert!(f.schema.columns[1].nested.is_some());
        // Consumes only the single leaf xs.list.element.
        assert_eq!(f.schema.columns[1].phys_cols, vec![1]);
    }
}
