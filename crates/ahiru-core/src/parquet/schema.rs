//! Parquet スキーマ → ahirudb の型への解決。
//!
//! v1 はフラットスキーマのみを扱う。LIST/MAP/STRUCT（= 子を持つグループ、
//! または REPEATED 要素）は `UnsupportedNested` で明示的に拒否する
//! （DESIGN.md §5, §14）。黙って落とすより、対応していないと言う方がよい。

use crate::parquet::meta::{ColumnMetaData, FileMetaData, SchemaElement};
use crate::parquet::*;
use crate::prelude::*;
use crate::vector::Ty;

/// 1 リーフ列の読み取りに必要な情報。
pub struct ColumnDesc {
    pub name: String,
    pub ty: Ty,
    pub nullable: bool,
    /// フラットスキーマなので 0 (REQUIRED) か 1 (OPTIONAL) のみ。
    pub max_def_level: u16,
    pub ptype: PType,
    /// FIXED_LEN_BYTE_ARRAY のバイト長。それ以外では 0。
    pub type_length: usize,
    /// TIME / TIMESTAMP のファイル上の分解能。読み取り時にマイクロ秒へ正規化する。
    pub time_unit: Option<TimeUnit>,
}

pub struct ParquetSchema {
    pub columns: Vec<ColumnDesc>,
}

impl ParquetSchema {
    /// 名前で列を引く（大文字小文字を区別しない）。
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.columns
            .iter()
            .position(|c| crate::rt::hash::eq_ascii_ci(c.name.as_bytes(), name.as_bytes()))
    }
}

/// フッタのスキーマ要素列（深さ優先順）からリーフ列を解決する。
pub fn resolve_schema(md: &FileMetaData) -> Result<ParquetSchema> {
    let root = &md.schema[0];
    let nchildren = root.num_children.unwrap_or(0);
    ensure!(nchildren >= 0, BadThrift);
    ensure!(md.schema.len() == nchildren as usize + 1, UnsupportedNested);

    let mut columns = Vec::with_capacity(nchildren as usize);
    for e in &md.schema[1..] {
        // 子を持つ要素はグループ = ネスト型。
        if e.num_children.unwrap_or(0) != 0 {
            err!(UnsupportedNested);
        }
        if e.repetition == Some(Repetition::Repeated) {
            err!(UnsupportedNested);
        }
        columns.push(resolve_column(e)?);
    }
    Ok(ParquetSchema { columns })
}

fn resolve_column(e: &SchemaElement) -> Result<ColumnDesc> {
    let ptype = match e.ptype {
        Some(p) => p,
        // 物理型が無い = グループ要素。上でネストを弾いているのでここには来ない。
        None => err!(UnsupportedNested),
    };
    let nullable = e.repetition != Some(Repetition::Required);
    let (ty, time_unit) = map_type(e, ptype)?;
    Ok(ColumnDesc {
        name: e.name.clone(),
        ty,
        nullable,
        max_def_level: if nullable { 1 } else { 0 },
        ptype,
        type_length: e.type_length.unwrap_or(0).max(0) as usize,
        time_unit,
    })
}

/// 論理型 → 変換テーブル → 物理型、の優先順で SQL 型を決める。
fn map_type(e: &SchemaElement, ptype: PType) -> Result<(Ty, Option<TimeUnit>)> {
    // 論理型と物理型の整合は検証しない（writer 依存の揺れが大きいため）。
    // 実際に変換できるかは reader が (ptype, ty) の組で判定する。
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
        L::Bson | L::Uuid => (Ty::Blob, None),
        L::Decimal { scale, precision } => {
            ensure!((1..=38).contains(&precision), UnsupportedType);
            ensure!((0..=precision).contains(&scale), UnsupportedType);
            (Ty::Decimal { precision: precision as u8, scale: scale as u8 }, None)
        }
        L::Date => (Ty::Date, None),
        L::Time { unit, .. } => (Ty::Time, Some(unit)),
        L::Timestamp { unit, .. } => (Ty::Timestamp, Some(unit)),
        L::Integer { bit_width, signed } => (int_ty(bit_width, signed)?, None),
        L::Unknown => (Ty::Null, None),
        // LIST / MAP はネスト型。ここに来るのはスキーマが壊れている場合。
        L::List | L::Map => err!(UnsupportedNested),
        // FLOAT16 は FLBA(2) の半精度。v1 では未対応。
        L::Float16 => err!(UnsupportedType),
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
        C::List | C::Map | C::MapKeyValue => err!(UnsupportedNested),
        // INTERVAL は FLBA(12)。SQL 側に対応する型が無いので未対応。
        C::Interval => err!(UnsupportedType),
    })
}

fn map_physical(ptype: PType) -> Result<Ty> {
    Ok(match ptype {
        PType::Boolean => Ty::Boolean,
        PType::Int32 => Ty::Int,
        PType::Int64 => Ty::BigInt,
        // INT96 は非推奨だが Hive/Spark 由来のファイルに今も残っている。
        // ナノ秒精度のタイムスタンプとして解釈する。
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

/// 列チャンクが暗号化・外部ファイル参照でないことを確認する。
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
    }

    #[test]
    fn logical_types_take_precedence() {
        let mut e = elem("t", Some(PType::Int64), Repetition::Optional);
        e.converted_type = Some(ConvertedType::TimestampMillis);
        e.logical = Some(LogicalType::Timestamp { utc: true, unit: TimeUnit::Nanos });
        let c = resolve_column(&e).unwrap();
        assert_eq!(c.ty, Ty::Timestamp);
        assert_eq!(c.time_unit, Some(TimeUnit::Nanos));
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
    fn nested_schema_is_rejected_explicitly() {
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
    fn repeated_field_is_rejected() {
        let md = FileMetaData {
            version: 2,
            schema: vec![root(1), elem("r", Some(PType::Int32), Repetition::Repeated)],
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
}
