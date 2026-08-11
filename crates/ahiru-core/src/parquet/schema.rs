//! Parquet スキーマ → ahirudb の型への解決。
//!
//! フッタの `schema` は深さ優先順に並んだ `SchemaElement` の平坦な列で、
//! 各要素は `num_children` で自分の子の数だけを申告する（実際の木構造は
//! 「次の `num_children` 個ぶんを再帰的に消費する」ことで初めて分かる）。
//! これは Thrift Compact のフッタが木そのものではなく前順走査の記録だから。
//!
//! REPEATED を含まない STRUCT（= 子を持つ REQUIRED/OPTIONAL グループ）は
//! 物理的にはリーフごとに独立した列チャンクを持つだけなので、
//! 「木を辿ってドット区切り名のリーフ列を集める」だけで読める。
//!
//! REPEATED を含む部分木（LIST/MAP、または素の REPEATED フィールド）は
//! 行と配列要素の対応付けに repetition level の解読が要るので別に扱う。
//! ここでは「REPEATED を含む部分木をまるごと 1 本の `Ty::Json` 列にする」
//! という設計を採る： `reader::nested` が repetition/definition level から
//! Dremel 方式で行ごとの入れ子構造を組み立て、JSON テキストへ直列化する
//! （物理型は 6 種のまま増やさない。DESIGN.md §8, §11）。
//! したがって「REPEATED を全く含まない STRUCT」だけが引き続きドット区切り
//! 列へフラット化される（例: `address.city`）。STRUCT の中に LIST/MAP が
//! あれば、その STRUCT ごと 1 本の JSON 列になる。
//!
//! SQL 側のドットアクセサ（`SELECT s.field`）はここでは扱わない。ここが
//! 提供するのは `address.city` のようなドット区切り名を持つ、あたかも
//! フラットであるかのような列の集合だけで、それをどう SQL にバインドする
//! かは呼び出し側の仕事。

use crate::parquet::meta::{ColumnMetaData, FileMetaData, SchemaElement};
use crate::parquet::*;
use crate::prelude::*;
use crate::vector::Ty;

/// スキーマ木を再帰的に辿る際の深さ上限。壊れた/悪意あるファイルが
/// スタックを使い切らないための防御（thrift.rs の `MAX_DEPTH` と同じ考え方
/// だが、対象が Thrift 値ではなくスキーマ木なので別定数として持つ）。
const MAX_SCHEMA_DEPTH: usize = 32;

/// 解決できる物理リーフ列数の上限。`meta.rs::MAX_SCHEMA_ELEMENTS` と揃えて
/// ある。実ファイル由来ならフッタの Thrift デコード時点でその上限に既に
/// 収まっているが、`resolve_schema` はそれとは独立に呼べる（テストや将来の
/// 別入口）ので、ここでも同じ上限を自前で守り、青天井の確保を許さない。
/// 出力列（論理列）数は物理リーフ数以下に必ず収まるので、この 1 つの上限で
/// 両方を守れる。
const MAX_LEAF_COLUMNS: usize = 16_384;

/// 入れ子列 1 リーフぶんのデコード情報（`reader::nested` が使う）。
/// `ColumnDesc::leaves` は `ColumnDesc::phys_cols` と同じ順序・長さで並ぶ。
#[derive(Clone, Copy)]
pub struct LeafDecodeInfo {
    pub ptype: PType,
    pub type_length: usize,
    pub time_unit: Option<TimeUnit>,
    pub ty: Ty,
    /// このリーフに至る経路上の OPTIONAL+REPEATED 数（自分含む、入れ子列の
    /// 根を基準に数える）。値が存在する判定に使う。
    pub max_def_level: u16,
    /// このリーフに至る経路上の REPEATED 数（自分含む）。ページ内の
    /// repetition level ストリームのビット幅を決めるのに使う。
    pub max_rep_level: u16,
}

/// 入れ子スキーマ木（REPEATED を含む部分木）のノード。
pub struct NestedNode {
    /// STRUCT フィールド名としての描画に使う（配列要素そのものには使わない）。
    pub name: String,
    pub repetition: Repetition,
    /// このノードに至る経路上の OPTIONAL+REPEATED 数の累計（自分を含む、
    /// 入れ子列の根を基準に数える）。
    pub def_depth: u16,
    /// このノードに至る経路上の REPEATED 数の累計（自分を含む）。
    pub rep_depth: u16,
    pub content: NestedContent,
    /// 存在確認・くり返し境界判定に使う代表リーフの `leaves` 配列中の添字。
    /// このノード配下の最初のリーフを指す（配下のどのリーフも、このノードの
    /// 存在・くり返し回数について同じ答えを持つ）。
    pub rep_leaf: usize,
}

pub enum NestedContent {
    /// 物理リーフ列。値は `leaves[index]` から取り出す。
    Leaf(usize),
    /// 子ノード列。非 REPEATED なら名前つきフィールド（STRUCT）、REPEATED
    /// なら「配列 1 要素ぶんの中身」（3 段/2 段エンコーディングの中間
    /// group、または MAP の key/value）を表す。
    Group(Vec<NestedNode>),
}

/// 1 リーフ列の読み取りに必要な情報。
pub struct ColumnDesc {
    /// STRUCT の下にあるリーフはドット区切り（`address.city`）。
    pub name: String,
    pub ty: Ty,
    pub nullable: bool,
    /// このリーフに至る経路上の OPTIONAL 要素数（リーフ自身を含む）。
    /// REPEATED が無い STRUCT チェーンでは、これがそのまま
    /// 「レベル一致 = 値あり」の definition level 上限として使える
    /// （途中のどのグループが NULL でも、リーフに届く前に打ち切られて
    /// 同じ 1 ビットの validity に潰れるため）。フラット列なら 0 か 1。
    /// `nested` が `Some` の列では未使用（常に 0）。
    pub max_def_level: u16,
    pub ptype: PType,
    /// FIXED_LEN_BYTE_ARRAY のバイト長。それ以外では 0。
    /// `nested` が `Some` の列では未使用。
    pub type_length: usize,
    /// TIME / TIMESTAMP のファイル上の分解能。読み取り時にマイクロ秒へ正規化する。
    /// `nested` が `Some` の列では未使用。
    pub time_unit: Option<TimeUnit>,
    /// 物理列チャンク番号（`row_group.columns` の添字）。フラット列は必ず
    /// 1 個。入れ子列は部分木の全リーフぶん複数持つ（読み取り順 = `leaves`
    /// と同じ順序）。
    pub phys_cols: Vec<usize>,
    /// `phys_cols` と同じ順序・長さで並ぶ、入れ子列のリーフごとのデコード
    /// 情報。フラット列では空。
    pub leaves: Vec<LeafDecodeInfo>,
    /// REPEATED を含む部分木の構造。`Some` なら `ty == Ty::Json` で、
    /// 読み取りは `reader::nested` の Dremel 組み立て経路を使う。
    /// `None` なら今までどおりのフラット読み取り。
    pub nested: Option<Box<NestedNode>>,
}

pub struct ParquetSchema {
    pub columns: Vec<ColumnDesc>,
}

impl ParquetSchema {
    /// 名前で列を引く（大文字小文字を区別しない）。STRUCT 配下の列は
    /// `address.city` のようなドット区切りの完全名で引く。
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.columns
            .iter()
            .position(|c| crate::rt::hash::eq_ascii_ci(c.name.as_bytes(), name.as_bytes()))
    }

    /// `prefix` が STRUCT 列のドット区切り名の接頭辞として存在するか。
    ///
    /// `index_of("address")` は（`address` 自体はリーフではないので）
    /// 見つからずに `None` を返す。DESCRIBE やエラーメッセージ生成側が
    /// 「その名前は列ではなく STRUCT だ」と言い分けられるよう、この判定
    /// だけ別に提供する。SQL のドットアクセサ構文自体はここでは扱わない。
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

/// フッタのスキーマ要素列（深さ優先順）からリーフ列を解決する。
///
/// ルート直下の `nchildren` 個を順に消費する。各要素が STRUCT グループなら
/// さらにその子を消費する、という再帰で木全体を辿る。
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
    // 申告されたトップレベル子数と、実際に消費した要素数（部分木含む）が
    // 食い違うのは壊れた/意図的に細工されたスキーマ。
    ensure!(pos == md.schema.len(), UnsupportedNested);
    Ok(ParquetSchema { columns })
}

/// スキーマ木を 1 ノード分（リーフなら自分だけ、グループなら部分木全体）
/// 消費し、次に消費すべき位置を返す。
///
/// - `prefix` はここまでの祖先グループ名を `.` で連結したもの（トップレベル
///   なら空文字列）。
/// - `parent_def_level` は祖先（自分を含まない）の OPTIONAL 要素数。
/// - `phys` はここまでに解決した物理リーフ列数（`row_group.columns` の
///   次の添字）。フラット列・入れ子列を問わず、リーフを 1 つ消費するたびに
///   進める。
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
    // 子の数を申告どおりに消費し切る前に列が尽きた = 壊れたスキーマ。
    ensure!(pos < md.schema.len(), UnsupportedNested);
    let e = &md.schema[pos];

    let mut name = String::with_capacity(prefix.len() + 1 + e.name.len());
    if !prefix.is_empty() {
        name.push_str(prefix);
        name.push('.');
    }
    name.push_str(&e.name);

    // REPEATED を部分木のどこかに含む（自分自身が REPEATED な場合も含む）
    // なら、STRUCT フラット化はせずこの部分木をまるごと 1 本の JSON 列にする。
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
        // 子を持つ = STRUCT グループ。子の列名にはこのグループ名を前置する。
        let mut child_pos = pos + 1;
        for _ in 0..nchildren {
            child_pos = resolve_node(md, child_pos, &name, def_level, depth + 1, phys, out)?;
        }
        return Ok(child_pos);
    }

    out.push(resolve_leaf(e, name, def_level, phys)?);
    Ok(pos + 1)
}

/// `pos` の部分木（自分自身を含む）に REPEATED な要素が 1 つでもあるか。
/// 返り値は `(見つかったか, 消費した次の位置)`。呼び出し側が `has_repeated`
/// でない場合だけこの関数の返す位置を使い、`has_repeated` の場合は
/// `build_nested_node` が改めて同じ部分木を辿って本体を組み立てる
/// （フッタは小さいので二重走査のコストは無視できる）。
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

/// REPEATED を含む部分木を `NestedNode` に組み立てる。リーフに出会うたびに
/// `phys` を進め、`leaves` に `(物理列番号, デコード情報)` を追記する
/// （追記順 = `NestedContent::Leaf` が指す添字の順）。
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
        // 子が 1 つも無いグループ（nchildren == 0 はここに来ない）はあり
        // 得ないが、`first()` が `None` になる状況（壊れたスキーマで
        // nchildren > 0 なのに子が消費できていない）は上のループの
        // `ensure!` で先に弾かれている。
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

/// リーフ要素 1 つを `ColumnDesc` にする。`max_def_level` は呼び出し側
/// （`resolve_node` の再帰、またはフラット列単体を扱う `resolve_column`）
/// が経路全体から計算済みのものを渡す。`phys` はこのリーフの物理列番号を
/// 割り当てたあと 1 つ進める。
fn resolve_leaf(
    e: &SchemaElement,
    name: String,
    max_def_level: u16,
    phys: &mut usize,
) -> Result<ColumnDesc> {
    let ptype = match e.ptype {
        Some(p) => p,
        // 物理型が無い = グループ要素。呼び出し元（`resolve_node`）は
        // `num_children > 0` のときグループとして先に分岐しているので、
        // ここに来るのは `num_children == 0` なのに物理型も無い壊れた要素。
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

/// フラットな単体要素をトップレベル列として解決する。祖先を持たないので
/// `max_def_level` は自分の OPTIONAL/REQUIRED だけで決まる
/// （`resolve_node` の `prefix == ""`, `parent_def_level == 0` と同じ計算）。
/// `resolve_schema` は木を辿る `resolve_node`/`resolve_leaf` を直接使うので、
/// 現在これを呼ぶのは単体テストだけ（1 要素だけを単独で検証したい場合に、
/// 木を組み立てずに済む）。
#[cfg(test)]
fn resolve_column(e: &SchemaElement) -> Result<ColumnDesc> {
    let max_def_level = u16::from(e.repetition != Some(Repetition::Required));
    resolve_leaf(e, e.name.clone(), max_def_level, &mut 0)
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
        L::Bson => (Ty::Blob, None),
        // UUID の物理表現は FLBA(16) の生バイト列で、`Ty::Blob` と同じ
        // `Bytes` 系だが、テキスト表示・パースがハイフン付き 16 進形式に
        // なる点だけ異なる（`Ty::Uuid` の doc 参照）。
        L::Uuid => (Ty::Uuid, None),
        L::Decimal { scale, precision } => {
            ensure!((1..=38).contains(&precision), UnsupportedType);
            ensure!((0..=precision).contains(&scale), UnsupportedType);
            (Ty::Decimal { precision: precision as u8, scale: scale as u8 }, None)
        }
        L::Date => (Ty::Date, None),
        L::Time { unit, .. } => (Ty::Time, Some(unit)),
        // `utc` (`isAdjustedToUTC`) が真なら、この列は既に UTC の瞬間を表す
        // タイムスタンプ（`Ty::Timestamptz`）。偽ならタイムゾーン無しの素の
        // 日時（`Ty::Timestamp`）。レガシーな `ConvertedType` 経路
        // （`map_converted`）にはこのビットが無いので、そちらは常に
        // `Ty::Timestamp` のままにしてある。
        L::Timestamp { unit, utc } => {
            (if utc { Ty::Timestamptz } else { Ty::Timestamp }, Some(unit))
        }
        L::Integer { bit_width, signed } => (int_ty(bit_width, signed)?, None),
        L::Unknown => (Ty::Null, None),
        // LIST / MAP は「REPEATED を含む部分木」の入口として build_nested_node
        // 側で処理されるので、ここに来るのはリーフに直接付いている場合
        // （スキーマが壊れている）だけ。
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
        // LIST/MAP は build_nested_node 側の入口で処理される（上記コメント参照）。
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

    // --- 合成スキーマでの STRUCT / 深さ・幅の上限 ------------------------

    /// 単一段の OPTIONAL グループの子 1 つと STRUCT でないリーフ 1 つ。
    /// ドット区切り名になること、`max_def_level` が祖先ぶん積み上がること
    /// を、実ファイルを介さず最小構成で確かめる。
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
        assert_eq!(s.columns[0].max_def_level, 0); // id は REQUIRED。
                                                   // address (OPTIONAL) + city (OPTIONAL) の 2 段ぶん。
        assert_eq!(s.columns[1].max_def_level, 2);
        assert_eq!(s.columns[2].max_def_level, 2);
        assert_eq!(s.index_of("address.city"), Some(1));
        assert_eq!(s.index_of("address"), None); // リーフではないので引けない。
        assert!(s.is_struct_prefix("address"));
        assert!(s.is_struct_prefix("ADDRESS")); // 大文字小文字を無視。
        assert!(!s.is_struct_prefix("id"));
    }

    /// REQUIRED な祖先は definition level に寄与しない。
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
        // g は REQUIRED なので寄与せず、leaf 自身の OPTIONAL ぶんの 1 だけ。
        assert_eq!(s.columns[0].max_def_level, 1);
    }

    /// 32 段を超えて入れ子になったグループはスタックオーバーフローではなく
    /// `NestingTooDeep` として拒否される。
    #[test]
    fn pathologically_deep_schema_is_rejected_not_overflowed() {
        let mut schema = vec![root(1)];
        // 40 段の OPTIONAL グループ + 末端のリーフ。MAX_SCHEMA_DEPTH (32) 超え。
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

    /// リーフ数が上限を超える幅広スキーマは、確保が青天井になる前に拒否される。
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

    /// 素の REPEATED リーフ（LIST/MAP 注釈も STRUCT ラッパーも無い、
    /// 最も古い形の繰り返しフィールド）も 1 本の JSON 列として解決できる。
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
        // 素の REPEATED は SQL NULL になり得ない（空配列にしかならない）。
        assert!(!s.columns[0].nullable);
        assert_eq!(s.columns[0].phys_cols, vec![0]);
        let node = s.columns[0].nested.as_ref().unwrap();
        assert_eq!(node.repetition, Repetition::Repeated);
    }

    // --- 実ファイル（DuckDB 書き出し）を介した end-to-end 検証 ------------
    //
    // ここから下は tests/data/*.parquet を `file.rs::open_bytes` /
    // `reader.rs::read_column_chunk` で実際に読み、DuckDB の出力
    // （`duckdb -c "SELECT ..."` / `duckdb -c "SELECT ... FROM parquet_schema(...)"`）
    // と突き合わせる。definition level の計算がリーダ側の
    // `validity = (def_level == max_def_level)` とかみ合っているかは、
    // 単体テストの合成スキーマだけでは確認できない
    // （実際のページに書かれた definition level のビット列そのものを
    // デコードして初めて分かる）ため、ここで実バイト列を読む。

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

    /// ファイル全体を列ごとに読み、RowGroup をまたいで縦に連結する
    /// （`crates/ahiru-core/tests/parquet_files.rs` と同じ手順）。
    /// 入れ子列 (`nested.is_some()`) は含めない（呼び出し側が
    /// `crate::parquet::reader::nested::read_nested_column` を別途使う）。
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
        // 両方 OPTIONAL、city/zip も OPTIONAL（parquet_schema() で確認済み）。
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
        // nested/a/b/c すべて OPTIONAL（parquet_schema() で確認済み）。
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
        // つまり address 列そのものが NULL の行が 10/30 行ある
        // （`duckdb -c "SELECT count(*) FROM 'struct_null.parquet' WHERE address IS NULL"` = 10）。
        // definition level の insight が正しければ、city/zip 両方が同じ行で
        // NULL になり、それ以外の行では両方とも値を持つ。
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
        // xs は OPTIONAL グループ(LIST) → REPEATED グループ(list) → OPTIONAL
        // リーフ(element)。これ全体が 1 本の JSON 列になる。
        let bytes = read_bytes("list1.parquet");
        let f = open_bytes(&bytes).expect("open list1.parquet");
        let names: Vec<&str> = f.schema.columns.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["id", "xs"]);
        assert_eq!(f.schema.columns[1].ty, Ty::Json);
        assert!(f.schema.columns[1].nullable);
        assert!(f.schema.columns[1].nested.is_some());
        // xs.list.element の 1 リーフだけを消費する。
        assert_eq!(f.schema.columns[1].phys_cols, vec![1]);
    }
}
