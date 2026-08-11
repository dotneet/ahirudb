//! LIST/MAP など REPEATED を含む列を Dremel 方式で組み立て、`Ty::Json`
//! （物理表現は UTF-8 の JSON テキスト、`PhysType::Bytes`）の 1 列にする。
//!
//! ## Dremel 組み立ての要点
//!
//! Parquet の REPEATED フィールドは、値ごとに repetition level（どの階層で
//! 新しいくり返しが始まったか）と definition level（NULL ならどこで
//! 経路が途切れたか）が付く。あるノードの「存在するか」は
//! `def_level >= node.def_depth` で判定でき、REPEATED ノードの「まだ同じ
//! 配列の続きか」は `next.rep_level >= node.rep_depth` で判定できる
//! （`node.def_depth`/`node.rep_depth` はスキーマ解決時に
//! `schema::build_nested_node` が経路全体から積算済み）。
//!
//! これを使うと、1 つのリーフの中身は「代表リーフを 1 つ覗き見て、存在
//! しなければ配下の全リーフから境界エントリを 1 つずつ捨てて NULL、
//! 存在すれば中身を組み立てる（REPEATED ならこれを配列としてくり返す）」
//! という単純な再帰に落ちる。複数のリーフ（STRUCT のフィールド、MAP の
//! key/value）は互いのカーソルを一切参照せず、それぞれが自分の
//! def_level/rep_level だけを見て独立に消費個数を決める
//! （Parquet のシュレッディング規約がそれを保証する）。
//!
//! ## I/O バリアとの関係
//!
//! ページ読み取り自体は既存の仕組み（ページヘッダは非圧縮なので復号前に
//! 走査できる、内蔵していないコーデックはホストに委譲する）に従う。
//! 入れ子列は複数の物理列チャンクにまたがるので、`format::parquet` 側が
//! 分割の開始時点で全リーフぶんのバイト範囲を確定させ、ここには常に
//! 「そのリーフの列チャンク全体」を渡す（ページ単位の絞り込みはしない。
//! REPEATED 列は 1 ページの値数と行数が一致しないため、既存のページ選択
//! ロジック（`first_row_index` 前提）をそのまま使い回せない）。

use alloc::borrow::Cow;

use crate::expr::{funcs, kernels};
use crate::parquet::encoding::{self, RleDecoder};
use crate::parquet::meta::{decode_page_header, ColumnMetaData, PageHeader};
use crate::parquet::reader::{self, PageCache};
use crate::parquet::schema::{ColumnDesc, LeafDecodeInfo, NestedContent, NestedNode};
use crate::parquet::*;
use crate::prelude::*;
use crate::vector::{Ty, Value, Vector};

/// 動的に組み立てる JSON 値。数値は事前に整形したトークンのバイト列を
/// そのまま埋め込む（自前の 10 進変換は行わず `expr::kernels::fmt_int`/
/// `fmt_f64`、日時は `expr::funcs::fmt_*` を再利用する。どちらもこの
/// クレート内 `pub(crate)` の既存実装で、`format!`/`core::fmt` は使わない）。
enum JsonValue {
    Null,
    Bool(bool),
    /// 妥当な JSON 数値トークン（符号・数字・小数点・指数部のみ）。
    Num(Vec<u8>),
    /// 生バイト列。直列化時にエスケープする。
    Str(Vec<u8>),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

// --- 直列化 ------------------------------------------------------------

fn write_json(v: &JsonValue, out: &mut Vec<u8>) {
    match v {
        JsonValue::Null => out.extend_from_slice(b"null"),
        JsonValue::Bool(b) => out.extend_from_slice(if *b { b"true" } else { b"false" }),
        JsonValue::Num(tok) => out.extend_from_slice(tok),
        JsonValue::Str(bytes) => write_json_string(bytes, out),
        JsonValue::Array(items) => {
            out.push(b'[');
            for (i, it) in items.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_json(it, out);
            }
            out.push(b']');
        }
        JsonValue::Object(fields) => {
            out.push(b'{');
            for (i, (k, fv)) in fields.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_json_string(k.as_bytes(), out);
                out.push(b':');
                write_json(fv, out);
            }
            out.push(b'}');
        }
    }
}

fn write_json_string(bytes: &[u8], out: &mut Vec<u8>) {
    out.push(b'"');
    for &b in bytes {
        match b {
            b'"' => out.extend_from_slice(b"\\\""),
            b'\\' => out.extend_from_slice(b"\\\\"),
            0x08 => out.extend_from_slice(b"\\b"),
            0x0C => out.extend_from_slice(b"\\f"),
            b'\n' => out.extend_from_slice(b"\\n"),
            b'\r' => out.extend_from_slice(b"\\r"),
            b'\t' => out.extend_from_slice(b"\\t"),
            0x00..=0x1F => {
                out.extend_from_slice(b"\\u00");
                out.push(hex_digit(b >> 4));
                out.push(hex_digit(b & 0xF));
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

fn hex_encode(bytes: &[u8], out: &mut Vec<u8>) {
    for &b in bytes {
        out.push(hex_digit(b >> 4));
        out.push(hex_digit(b & 0xF));
    }
}

/// リーフの `Value` を、その論理型に応じて JSON 表現へ変換する。
/// 数値の整形は `expr::kernels`、日時は `expr::funcs` の既存実装に委ねる
/// （このクレートの CAST や CSV/JSONL 書き出しと同じ土台を使うことで、
/// 独自の丸め・書式の食い違いを避ける）。
fn leaf_value_to_json(ty: Ty, v: Value) -> JsonValue {
    match v {
        Value::Null => JsonValue::Null,
        Value::Bool(b) => JsonValue::Bool(b),
        Value::I32(x) => int_like_to_json(ty, x as i128),
        Value::I64(x) => match ty {
            Ty::Time => {
                let mut b = Vec::new();
                funcs::fmt_time(x, &mut b);
                JsonValue::Str(b)
            }
            Ty::Timestamp => {
                let mut b = Vec::new();
                funcs::fmt_timestamp(x, &mut b);
                JsonValue::Str(b)
            }
            Ty::Timestamptz => {
                let mut b = Vec::new();
                funcs::fmt_timestamptz(x, &mut b);
                JsonValue::Str(b)
            }
            _ => int_like_to_json(ty, x as i128),
        },
        Value::I128(x) => int_like_to_json(ty, x),
        Value::F64(x) => {
            let mut b = Vec::new();
            if x.is_finite() {
                kernels::fmt_f64(x, &mut b);
            } else {
                // JSON に NaN/Infinity は無い。DuckDB の to_json も NULL に潰す。
                b.extend_from_slice(b"null");
            }
            JsonValue::Num(b)
        }
        Value::Bytes(b) => match ty {
            Ty::Varchar => JsonValue::Str(b),
            Ty::Uuid => {
                let mut h = Vec::new();
                if let Ok(raw) = <[u8; 16]>::try_from(b.as_slice()) {
                    funcs::fmt_uuid(&raw, &mut h);
                }
                JsonValue::Str(h)
            }
            // BLOB は JSON に直接の対応が無いので 16 進文字列にする。
            _ => {
                let mut h = Vec::new();
                hex_encode(&b, &mut h);
                JsonValue::Str(h)
            }
        },
    }
}

fn int_like_to_json(ty: Ty, x: i128) -> JsonValue {
    match ty {
        Ty::Decimal { scale, .. } => {
            let mut b = Vec::new();
            kernels::fmt_int(x.unsigned_abs(), x < 0, scale, &mut b);
            JsonValue::Num(b)
        }
        Ty::Date => {
            let mut b = Vec::new();
            funcs::fmt_date(x as i64, &mut b);
            JsonValue::Str(b)
        }
        _ => {
            let mut b = Vec::new();
            kernels::fmt_int(x.unsigned_abs(), x < 0, 0, &mut b);
            JsonValue::Num(b)
        }
    }
}

// --- リーフごとのレベル + 値の読み取り -------------------------------------

/// 1 リーフ列ぶんの、全ページを通した repetition/definition level と、
/// 値が存在するエントリだけを詰めた密な値ベクタ。
struct LeafRuns {
    rep: Vec<u16>,
    def: Vec<u16>,
    values: Vector,
}

/// v1: 4 バイト長前置の RLE レベルストリームを読む。`max_level == 0` なら
/// ストリーム自体が省略される（全エントリ 0 と決め打つ）。
fn read_levels_v1(page: &[u8], n: usize, max_level: u16) -> Result<(Vec<u16>, usize)> {
    if max_level == 0 {
        return Ok((vec![0u16; n], 0));
    }
    ensure!(page.len() >= 4, UnexpectedEof, 0);
    let len = u32::from_le_bytes([page[0], page[1], page[2], page[3]]) as usize;
    ensure!(len <= page.len() - 4, UnexpectedEof, 4);
    let bw = encoding::bit_width(max_level as u32);
    let mut d = RleDecoder::new(&page[4..4 + len], bw);
    let mut raw = Vec::with_capacity(n);
    d.read_u32(n, &mut raw)?;
    Ok((raw.into_iter().map(|v| v as u16).collect(), 4 + len))
}

/// v2: 長さは `DataPageHeaderV2` のフィールドで既知なので前置は無い。
fn read_levels_v2(data: &[u8], n: usize, max_level: u16) -> Result<Vec<u16>> {
    if max_level == 0 {
        return Ok(vec![0u16; n]);
    }
    let bw = encoding::bit_width(max_level as u32);
    let mut d = RleDecoder::new(data, bw);
    let mut raw = Vec::with_capacity(n);
    d.read_u32(n, &mut raw)?;
    Ok(raw.into_iter().map(|v| v as u16).collect())
}

/// データページ v1 を 1 枚読み、repetition level → definition level → 値
/// の順で消費する（v1 のページ内レイアウト）。`reader::read_data_page_v1`
/// と違い repetition level を読む点、値を NULL 込みで散らばせず密なまま
/// 追記する点が異なる。
#[allow(clippy::too_many_arguments)]
fn read_nested_page_v1(
    desc: &ColumnDesc,
    meta: &ColumnMetaData,
    hdr: &PageHeader,
    raw: &[u8],
    raw_off: u64,
    dict: Option<&Vector>,
    max_def_level: u16,
    max_rep_level: u16,
    rep_out: &mut Vec<u16>,
    def_out: &mut Vec<u16>,
    values_out: &mut Vector,
    cache: &dyn PageCache,
) -> Result<()> {
    let dp = hdr.data_page.as_ref().ok_or(reader::err_at(Code::BadPageHeader, 0))?;
    let n = reader::check_count(dp.num_values)?;
    let page = reader::decompress(meta.codec, raw, hdr.uncompressed_page_size, raw_off, cache)?;
    let mut off = 0usize;

    ensure!(dp.repetition_level_encoding == Encoding::Rle, UnsupportedEncoding);
    ensure!(off <= page.len(), UnexpectedEof, off);
    let (rep_levels, used) = read_levels_v1(&page[off..], n, max_rep_level)?;
    off += used;

    ensure!(dp.definition_level_encoding == Encoding::Rle, UnsupportedEncoding);
    ensure!(off <= page.len(), UnexpectedEof, off);
    let (def_levels, used2) = read_levels_v1(&page[off..], n, max_def_level)?;
    off += used2;

    let present = def_levels.iter().filter(|&&d| d == max_def_level).count();
    ensure!(off <= page.len(), UnexpectedEof, off);
    let dense = reader::decode_dense(desc, dp.encoding, &page[off..], present, dict)?;
    ensure!(dense.len() == present, BadCompressedData);
    reader::append_all(values_out, &dense, present)?;
    rep_out.extend_from_slice(&rep_levels);
    def_out.extend_from_slice(&def_levels);
    Ok(())
}

/// データページ v2 を 1 枚読む。レベルは非圧縮でページ先頭に置かれ、値
/// 部分だけが（あれば）圧縮される。
#[allow(clippy::too_many_arguments)]
fn read_nested_page_v2(
    desc: &ColumnDesc,
    meta: &ColumnMetaData,
    hdr: &PageHeader,
    raw: &[u8],
    raw_off: u64,
    dict: Option<&Vector>,
    max_def_level: u16,
    max_rep_level: u16,
    rep_out: &mut Vec<u16>,
    def_out: &mut Vec<u16>,
    values_out: &mut Vector,
    cache: &dyn PageCache,
) -> Result<()> {
    let dp = hdr.data_page_v2.as_ref().ok_or(reader::err_at(Code::BadPageHeader, 0))?;
    let n = reader::check_count(dp.num_values)?;
    let rep_len = reader::check_len(dp.repetition_levels_byte_length)?;
    let def_len = reader::check_len(dp.definition_levels_byte_length)?;
    ensure!(rep_len <= raw.len(), UnexpectedEof, 0);
    ensure!(def_len <= raw.len() - rep_len, UnexpectedEof, rep_len);
    let rep_levels = read_levels_v2(&raw[..rep_len], n, max_rep_level)?;
    let def_levels = read_levels_v2(&raw[rep_len..rep_len + def_len], n, max_def_level)?;

    let present = def_levels.iter().filter(|&&d| d == max_def_level).count();
    let skip = rep_len + def_len;
    let values_raw = &raw[skip..];
    let values = if dp.is_compressed {
        let want = (hdr.uncompressed_page_size as i64) - skip as i64;
        ensure!(want >= 0, BadPageHeader);
        reader::decompress(meta.codec, values_raw, want as i32, raw_off + skip as u64, cache)?
    } else {
        Cow::Borrowed(values_raw)
    };
    let dense = reader::decode_dense(desc, dp.encoding, &values, present, dict)?;
    ensure!(dense.len() == present, BadCompressedData);
    reader::append_all(values_out, &dense, present)?;
    rep_out.extend_from_slice(&rep_levels);
    def_out.extend_from_slice(&def_levels);
    Ok(())
}

/// 1 リーフの列チャンク全体（辞書ページ + データページ群）を読み、
/// repetition/definition level と密な値ベクタにする。フラット列の
/// `reader::read_column_chunk` と違い、行数ではなくバッファを使い切る
/// ことをループの終了条件にする（REPEATED 列は 1 ページの値数がそのまま
/// 行数にならないため）。
fn read_nested_leaf_chunk(
    info: &LeafDecodeInfo,
    meta: &ColumnMetaData,
    buf: &[u8],
    chunk_start: u64,
    cache: &dyn PageCache,
) -> Result<LeafRuns> {
    // decode_dense 等の再利用のためだけの一時 ColumnDesc。物理デコードに
    // 必要な情報 (ty/ptype/type_length/time_unit) だけを埋める。
    let desc = ColumnDesc {
        name: String::new(),
        ty: info.ty,
        nullable: true,
        max_def_level: info.max_def_level,
        ptype: info.ptype,
        type_length: info.type_length,
        time_unit: info.time_unit,
        phys_cols: Vec::new(),
        leaves: Vec::new(),
        nested: None,
    };
    let mut rep = Vec::new();
    let mut def = Vec::new();
    let mut values = Vector::with_capacity(info.ty, 0);
    let mut dict: Option<Vector> = None;
    let mut pos = 0usize;

    while pos < buf.len() {
        let (hdr, hlen) = decode_page_header(&buf[pos..])?;
        pos += hlen;
        let clen = hdr.compressed_page_size as usize;
        ensure!(clen <= buf.len().saturating_sub(pos), UnexpectedEof, pos);
        let raw_off = chunk_start + pos as u64;
        let raw = &buf[pos..pos + clen];
        pos += clen;

        match hdr.ptype {
            PageType::DictionaryPage => {
                dict =
                    Some(reader::decode_dictionary_page(&desc, meta, &hdr, raw, raw_off, cache)?);
            }
            PageType::DataPage => read_nested_page_v1(
                &desc,
                meta,
                &hdr,
                raw,
                raw_off,
                dict.as_ref(),
                info.max_def_level,
                info.max_rep_level,
                &mut rep,
                &mut def,
                &mut values,
                cache,
            )?,
            PageType::DataPageV2 => read_nested_page_v2(
                &desc,
                meta,
                &hdr,
                raw,
                raw_off,
                dict.as_ref(),
                info.max_def_level,
                info.max_rep_level,
                &mut rep,
                &mut def,
                &mut values,
                cache,
            )?,
            // 実際には書かれないページ種別。読み飛ばす。
            PageType::IndexPage => {}
        }
    }
    Ok(LeafRuns { rep, def, values })
}

// --- Dremel 組み立て ---------------------------------------------------

/// 1 リーフを読み進めるためのカーソル。`pos` は生エントリ（NULL 含む）の
/// 添字、`val_idx` は値が存在するエントリだけを数える別添字。
struct LeafCursor<'a> {
    ty: Ty,
    rep: &'a [u16],
    def: &'a [u16],
    values: &'a Vector,
    pos: usize,
    val_idx: usize,
}

impl<'a> LeafCursor<'a> {
    /// 今の位置の `(repetition level, definition level)`。行の途中で尽きる
    /// のは壊れたファイル（宣言された行数と実際のレベル列が食い違う）。
    #[inline]
    fn peek(&self) -> Result<(u16, u16)> {
        match (self.rep.get(self.pos), self.def.get(self.pos)) {
            (Some(&r), Some(&d)) => Ok((r, d)),
            _ => err!(BadCompressedData),
        }
    }

    /// くり返し継続判定用。列の終端（次のリーフ/行が無い）は正常系なので
    /// `None` を返す。
    #[inline]
    fn peek_opt(&self) -> Option<(u16, u16)> {
        match (self.rep.get(self.pos), self.def.get(self.pos)) {
            (Some(&r), Some(&d)) => Some((r, d)),
            _ => None,
        }
    }

    #[inline]
    fn advance_raw(&mut self) {
        self.pos += 1;
    }

    #[inline]
    fn take_value(&mut self) -> Value {
        let v = self.values.value_at(self.val_idx);
        self.val_idx += 1;
        v
    }
}

/// ノード 1 個ぶんを組み立てる。REPEATED なら配列、それ以外なら
/// 単発の値（NULL もあり得る）。
fn assemble(node: &NestedNode, cursors: &mut [LeafCursor]) -> Result<JsonValue> {
    if node.repetition == Repetition::Repeated {
        assemble_repeated(node, cursors)
    } else {
        assemble_single(node, cursors)
    }
}

/// 非 REPEATED ノード。代表リーフの definition level がこのノードの
/// `def_depth` に届いていなければ、配下の全リーフから境界エントリを 1 つ
/// ずつ消費して NULL を返す。
fn assemble_single(node: &NestedNode, cursors: &mut [LeafCursor]) -> Result<JsonValue> {
    let (_, def) = cursors[node.rep_leaf].peek()?;
    if def < node.def_depth {
        consume_boundary(node, cursors);
        return Ok(JsonValue::Null);
    }
    render_present(node, cursors)
}

/// REPEATED ノード。0 要素なら空配列（境界エントリを 1 つ消費するだけ）。
/// 1 要素以上なら、代表リーフの repetition level がこのノードの
/// `rep_depth` 以上である限り「同じ配列の続き」として要素を読み続ける。
fn assemble_repeated(node: &NestedNode, cursors: &mut [LeafCursor]) -> Result<JsonValue> {
    let mut items = Vec::new();
    loop {
        let (_, def) = cursors[node.rep_leaf].peek()?;
        if def < node.def_depth {
            consume_boundary(node, cursors);
            break;
        }
        items.push(render_element(node, cursors)?);
        match cursors[node.rep_leaf].peek_opt() {
            Some((next_rep, _)) if next_rep >= node.rep_depth => continue,
            _ => break,
        }
    }
    Ok(JsonValue::Array(items))
}

/// ノードが「存在しない」と確定したとき、配下の全リーフから境界エントリを
/// ちょうど 1 つずつ消費する（Parquet のシュレッディング規約により、
/// 経路のどこで途切れても配下のリーフは必ずこの 1 エントリを持つ）。
fn consume_boundary(node: &NestedNode, cursors: &mut [LeafCursor]) {
    match &node.content {
        NestedContent::Leaf(idx) => cursors[*idx].advance_raw(),
        NestedContent::Group(children) => {
            for c in children {
                consume_boundary(c, cursors);
            }
        }
    }
}

/// 非 REPEATED ノードの「存在する」中身を描画する。子が 1 個かつそれ自身が
/// REPEATED なら（LIST/MAP ラッパー group）名前を作らずそのまま委譲する
/// （でなければ `{"list": [...]}` のような余計な階層ができてしまう）。
fn render_present(node: &NestedNode, cursors: &mut [LeafCursor]) -> Result<JsonValue> {
    match &node.content {
        NestedContent::Leaf(idx) => Ok(take_leaf_value(*idx, cursors)),
        NestedContent::Group(children) => {
            if children.len() == 1 && children[0].repetition == Repetition::Repeated {
                return assemble(&children[0], cursors);
            }
            let mut obj = Vec::with_capacity(children.len());
            for c in children {
                obj.push((c.name.clone(), assemble(c, cursors)?));
            }
            Ok(JsonValue::Object(obj))
        }
    }
}

/// REPEATED ノードの「1 要素ぶん」を描画する。子が 1 個なら（3 段/2 段
/// エンコーディングの中間 group、または LIST<STRUCT> の element）その子を
/// そのまま要素として使う（`render_present` と違い、REPEATED かどうかは
/// 問わない ―― repeated ノード自身の唯一の子は常にラップせず透過する）。
/// 子が 2 個以上なら（MAP の key/value など）名前つきオブジェクトにする。
fn render_element(node: &NestedNode, cursors: &mut [LeafCursor]) -> Result<JsonValue> {
    match &node.content {
        NestedContent::Leaf(idx) => Ok(take_leaf_value(*idx, cursors)),
        NestedContent::Group(children) => {
            if children.len() == 1 {
                return assemble(&children[0], cursors);
            }
            let mut obj = Vec::with_capacity(children.len());
            for c in children {
                obj.push((c.name.clone(), assemble(c, cursors)?));
            }
            Ok(JsonValue::Object(obj))
        }
    }
}

fn take_leaf_value(idx: usize, cursors: &mut [LeafCursor]) -> JsonValue {
    let ty = cursors[idx].ty;
    let v = cursors[idx].take_value();
    cursors[idx].advance_raw();
    leaf_value_to_json(ty, v)
}

// --- 入口 ----------------------------------------------------------------

/// 入れ子列 (LIST/MAP など REPEATED を含む部分木) を組み立てて 1 本の
/// `Ty::Json` ベクタにする。
///
/// `chunks` は `desc.leaves`/`desc.phys_cols` と同じ順序で、各リーフの
/// `(列メタデータ, 列チャンク全体のバイト列, ファイル上の開始オフセット)`。
/// ページ選択はしない（呼び出し側は常に列チャンク全体を渡す。理由は
/// モジュール冒頭のコメントを参照）。
pub fn read_nested_column(
    desc: &ColumnDesc,
    chunks: &[(&ColumnMetaData, &[u8], u64)],
    num_rows: usize,
    cache: &dyn PageCache,
) -> Result<Vector> {
    let root = match &desc.nested {
        Some(n) => n.as_ref(),
        None => err!(Internal),
    };
    ensure!(chunks.len() == desc.leaves.len(), Internal);
    ensure!(chunks.len() == desc.phys_cols.len(), Internal);

    let mut runs: Vec<LeafRuns> = Vec::with_capacity(chunks.len());
    for (i, &(meta, buf, start)) in chunks.iter().enumerate() {
        runs.push(read_nested_leaf_chunk(&desc.leaves[i], meta, buf, start, cache)?);
    }

    let mut cursors: Vec<LeafCursor> = runs
        .iter()
        .zip(desc.leaves.iter())
        .map(|(r, info)| LeafCursor {
            ty: info.ty,
            rep: &r.rep,
            def: &r.def,
            values: &r.values,
            pos: 0,
            val_idx: 0,
        })
        .collect();

    let mut out = Vector::with_capacity(Ty::Json, num_rows);
    for _ in 0..num_rows {
        match assemble(root, &mut cursors)? {
            JsonValue::Null => out.push_null(),
            other => {
                let mut buf = Vec::new();
                write_json(&other, &mut buf);
                out.push_value(&Value::Bytes(buf));
            }
        }
    }

    // 全リーフのカーソルが厳密に使い切られているか（宣言された行数ぶん
    // 過不足なく消費できたか）を確認する。壊れたファイルの検出に使う。
    for c in &cursors {
        ensure!(c.pos == c.rep.len() && c.pos == c.def.len(), BadCompressedData);
    }
    Ok(out)
}
