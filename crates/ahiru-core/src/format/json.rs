//! JSON（`read_json`/`read_json_auto` 相当）— ファイル全体が 1 個の JSON 値。
//!
//! `format::jsonl`（NDJSON、1 行 1 オブジェクト）とは違い、こちらはファイル
//! 全体が **1 つの JSON ドキュメント**（トップレベル配列、またはオブジェクト
//! 単体）になっている入力を読む。区切りが改行ではなく `[`/`,`/`]` なので、
//! 分割の途中でレコード境界が確定しない。
//!
//! ## トップレベルの規則（`duckdb` の `read_json_auto` で実測して確認した）
//!
//! - トップレベルが配列 `[...]` → 各要素を 1 行として読む。
//! - トップレベルが単一のオブジェクト `{...}` → それ 1 個を 1 行のテーブルと
//!   して読む（`duckdb -c "SELECT * FROM read_json_auto('single_obj.json')"`
//!   で確認: 1 行のテーブルになる）。
//! - 配列の要素がオブジェクトでない（スカラーや配列そのもの）場合、および
//!   トップレベルがオブジェクトでも配列でもない生のスカラーの場合は、
//!   `"json"` という名前の列 1 本に生の値をそのまま入れる（`duckdb` の
//!   `read_json_auto('[1,2,3]')` が `json BIGINT` という 1 列テーブルになる
//!   のと同じ考え方）。**この判定は行ごとに独立して行う**ので、オブジェクトの
//!   行とスカラーの行が混在していても（`duckdb` は基本的に想定しない入力だが）
//!   クラッシュはしない。オブジェクトでない行は `"json"` 列にだけ値が入り、
//!   他の列は NULL になる。
//! - 空配列 `[]` は `duckdb` にならい `"json"` という 1 列（型 JSON）・0 行の
//!   テーブルにする（列が 1 本も無い退化したテーブルより扱いやすいため）。
//! - 空ファイル（0 バイト）はトップレベル値が存在しないので `UnexpectedEof`
//!   にする。JSONL の「空ファイルは空テーブル」とは違い、こちらは 1 つの
//!   JSON ドキュメントを読む前提なので、空はそもそも構文違反になる。
//!
//! ## 非ストリーミング設計（v1 の制限）
//!
//! 他のフォーマット（Parquet/CSV/JSONL）は「分割境界でしか I/O を待たない」
//! 設計（`format` モジュール doc、DESIGN.md §6）に従うが、JSON 配列は途中の
//! バイトだけでは要素の区切りが構造的に確定しない（配列の深さ・文字列の
//! エスケープ次第でどこまでが 1 要素か変わる）。そこで `resolve` は
//! **ファイル全体のバイトが揃うまで `NeedIo` を返し続け、揃った時点で
//! 一括パースする**という割り切りにしてある。`write::export_all` が
//! 「クエリ実行中の `NEED_IO` を再開せず失敗させる非再開設計」であることを
//! 明記しているのと同じ考え方で、ここでは「分割 = ファイル全体 1 個」に
//! 単純化することで分割境界バリアの形だけは保っている（`num_splits` は常に
//! 高々 1）。読み取り自体は `read_split` でもう一度ファイル全体を走査する
//! （`resolve` 側のスキャンとは別実装。自己参照構造体を避けるための単純化）。
//!
//! ## メモリの安全弁
//!
//! ファイル全体を一度にメモリへ載せる都合上、`MAX_JSON_BYTES` を超える
//! ファイルは `resolve` の時点で `Oom` として拒否する（`exec::join` の
//! `MAX_BUILD_BYTES` などと同じ「静かに落とさず明示エラーにする」方針。
//! DESIGN.md の「メモリ上限: 既定 512 MB」と同系統だが、ファイル全体を
//! 保持する分さらに保守的な値にしてある）。
//!
//! ## スキーマ推論
//!
//! 列集合はオブジェクトのキーの和集合（初出順）、型は
//! NULL → BOOLEAN → BIGINT → DOUBLE → VARCHAR/JSON の束で広げる —
//! ここまでは `format::jsonl` と同じ考え方。ただし `jsonl` には `Ty::Json`
//! が無かった時代の設計なので、ネストした値や噛み合わない型の混在は
//! `Ty::Varchar`（生テキストのまま）に倒していた。こちらは `Ty::Json` が
//! 使えるので、`duckdb` の実測（後述）に合わせてそちらへ倒す:
//!
//! - ネストした値（配列・オブジェクト）は常に `Ty::Json`
//!   （`duckdb -c "SELECT * FROM read_json_auto('nested_mixed.json')"` で
//!   `[1,2,3]` と `5` が混ざった列が `JSON` 型になることを確認）。
//! - 文字列とみなせる系統（プレーン文字列 / DATE / TIMESTAMP と判定された
//!   文字列）同士の混在は、どれも decode 後のテキストとして表現できるので
//!   `Ty::Varchar` に寄せる（`format::jsonl` の `widen` と同じ判断）。
//! - それ以外の噛み合わない混在（数値と文字列、真偽値と文字列、
//!   ネストとスカラーなど）は物理表現が生 JSON テキストしか無いので
//!   `Ty::Json` に倒す（`duckdb -c "SELECT * FROM read_json_auto('widen.json')"`
//!   で int/double/bool/string が混ざった列が `JSON` 型になることを確認）。
//! - サンプル中すべて NULL だった列は `format::jsonl` と同じく安全側の
//!   `Ty::Varchar` にする（`duckdb` は `JSON` にするが、`jsonl` との一貫性を
//!   優先した）。
//!
//! 低レベルの JSON スキャナ（文字列・数値の走査、エスケープ復号、
//! 日付/タイムスタンプの判定）は `format::jsonl` と同じ考え方の実装だが、
//! `jsonl` 側の型は非公開でこのモジュールから再利用できず、
//! かつ `jsonl.rs` 自体の変更は本タスクの担当範囲外（分割境界 I/O バリアに
//! 最適化された既存実装に手を入れない）なので、独立した実装として
//! 書き起こしてある。構造はできるだけ揃えてあるので、将来どちらかに
//! 寄せる形で共通化することは可能なはず。
//!
//! ## 入力は信用しない
//!
//! `format::jsonl` と同じ方針。壊れた JSON は `Err` になる（NULL セルへの
//! 読み替えはスキーマ確定後、サンプル外で型が外れた値にのみ適用される）。
//! 走査は再帰せず、入れ子は `MAX_DEPTH` で打ち切る。

use crate::catalog::Source;
use crate::format::{ResolveStep, TableFormat};
use crate::prelude::*;
use crate::vector::{Bitmap, Data, Field, Ty, Vector};

/// スキーマ推論（列の型の広がり）に使う先頭からの最大行数。
/// `format::jsonl::SAMPLE_LINES` と同じ考え方。これを超える要素も構文検査
/// （`resolve` がファイル全体を読み切る設計のため）と行数カウントはするが、
/// 型推論の widen 計算には使わない。
pub const SAMPLE_ELEMENTS: usize = 1000;

/// 値の入れ子の上限。`format::jsonl::MAX_DEPTH` と同じ理由・同じ値。
const MAX_DEPTH: u32 = 32;

/// 推論で作る列数の上限。`format::jsonl::MAX_COLUMNS` と同じ。
const MAX_COLUMNS: usize = 1024;

/// ファイル全体を一度にメモリへ載せる設計上の安全弁。`exec::join` の
/// `MAX_BUILD_BYTES` (128 MiB) と同じ桁にしてある。
const MAX_JSON_BYTES: u64 = 128 * 1024 * 1024;

pub struct JsonFormat {
    schema: Vec<Field>,
    resolved: bool,
    total_len: u64,
    row_count: u64,
}

impl JsonFormat {
    pub fn new() -> Self {
        JsonFormat { schema: Vec::new(), resolved: false, total_len: 0, row_count: 0 }
    }
}

impl Default for JsonFormat {
    fn default() -> Self {
        JsonFormat::new()
    }
}

impl TableFormat for JsonFormat {
    fn resolve(&mut self, src: &Source) -> Result<ResolveStep> {
        if self.resolved {
            return Ok(Ok(()));
        }
        ensure!(src.total_len <= MAX_JSON_BYTES, Oom);
        // 空ファイルはトップレベル値が無いので、そもそも JSON として不正
        // （JSONL と違い「1 個の JSON ドキュメント」を読む前提のため）。
        ensure!(src.total_len > 0, UnexpectedEof);
        self.total_len = src.total_len;
        let buf = match src.get(0, src.total_len as usize) {
            Some(b) => b,
            // 分割境界バリアではなく「ファイル全体」を要求する（モジュール doc）。
            None => return Ok(Err((0, src.total_len))),
        };
        let (schema, row_count) = parse_schema(buf)?;
        self.schema = schema;
        self.row_count = row_count;
        self.resolved = true;
        Ok(Ok(()))
    }

    fn is_resolved(&self) -> bool {
        self.resolved
    }

    fn schema(&self) -> &[Field] {
        &self.schema
    }

    fn num_splits(&self) -> usize {
        // 非ストリーミング設計: 分割は常にファイル全体 1 個（モジュール doc）。
        if self.resolved {
            1
        } else {
            0
        }
    }

    fn split_rows(&self, split: usize) -> Option<u64> {
        if split == 0 {
            Some(self.row_count)
        } else {
            None
        }
    }

    fn split_ranges(
        &self,
        split: usize,
        _projection: &[usize],
        out: &mut Vec<(u64, u64)>,
    ) -> Result<()> {
        ensure!(split < self.num_splits(), Internal);
        // 構造が確定しないので射影があっても常にファイル全体が要る。
        out.push((0, self.total_len));
        Ok(())
    }

    fn read_split(&self, src: &Source, split: usize, projection: &[usize]) -> Result<Vec<Vector>> {
        ensure!(self.resolved, Internal);
        ensure!(split < self.num_splits(), Internal);
        let buf = match src.get(0, self.total_len as usize) {
            Some(b) => b,
            // split_ranges が示した範囲は呼び出し側が揃えている約束。
            None => err!(Internal),
        };

        let mut names: Vec<&[u8]> = Vec::with_capacity(projection.len());
        let mut builders: Vec<Builder> = Vec::with_capacity(projection.len());
        for &c in projection {
            let f = match self.schema.get(c) {
                Some(f) => f,
                None => err!(Internal),
            };
            names.push(f.name.as_bytes());
            builders.push(Builder::new(f.ty, 64));
        }

        let mut slots: Vec<Option<Member>> = vec![None; projection.len()];
        let mut key = Vec::new();
        let mut val = Vec::new();

        let i0 = skip_ws(buf, skip_bom(buf));
        if byte_at(buf, i0)? == b'[' {
            let mut it = Elements::new(buf, i0)?;
            while let Some((s, e)) = it.next()? {
                process_row(&buf[s..e], &names, &mut slots, &mut builders, &mut key, &mut val)?;
            }
            // 配列の閉じ括弧の後に残るのは空白だけ。
            ensure!(skip_ws(buf, it.i) == buf.len(), SyntaxError, it.i);
        } else {
            let end = skip_value(buf, i0)?;
            ensure!(skip_ws(buf, end) == buf.len(), SyntaxError, end);
            process_row(&buf[i0..end], &names, &mut slots, &mut builders, &mut key, &mut val)?;
        }

        Ok(builders.into_iter().map(|b| b.finish()).collect())
    }
}

// --- スキーマ推論 -------------------------------------------------------------

/// `buf` 全体を 1 個の JSON ドキュメントとして解決し、`(スキーマ, 行数)` を返す。
/// 行数は `SAMPLE_ELEMENTS` を超えても正確に数える（構文検査のためにどのみち
/// 全要素を走査するので、追加コストは無い）。widen 計算だけがサンプルに限る。
fn parse_schema(buf: &[u8]) -> Result<(Vec<Field>, u64)> {
    let i = skip_ws(buf, skip_bom(buf));
    let c = byte_at(buf, i)?;

    let mut names: Vec<String> = Vec::new();
    let mut infs: Vec<Inf> = Vec::new();
    let mut key = Vec::new();
    let mut row_count: u64 = 0;

    if c == b'[' {
        let mut it = Elements::new(buf, i)?;
        while let Some((s, e)) = it.next()? {
            row_count += 1;
            if row_count as usize <= SAMPLE_ELEMENTS {
                accumulate_row(&buf[s..e], &mut names, &mut infs, &mut key)?;
            }
        }
        ensure!(skip_ws(buf, it.i) == buf.len(), SyntaxError, it.i);
        if row_count == 0 {
            // 空配列。列が 1 本も無い退化したテーブルより duckdb に合わせる
            // 方が扱いやすいので、`"json"` という JSON 型の列にする。
            names.push(String::from("json"));
            infs.push(Inf::Json);
        }
    } else {
        // トップレベルが配列でない: オブジェクト単体、または生スカラー。
        // どちらも「1 行」として扱う（モジュール doc）。
        let end = skip_value(buf, i)?;
        ensure!(skip_ws(buf, end) == buf.len(), SyntaxError, end);
        row_count = 1;
        accumulate_row(&buf[i..end], &mut names, &mut infs, &mut key)?;
    }

    let schema = names.into_iter().zip(infs).map(|(n, i)| Field::new(n, i.ty(), true)).collect();
    Ok((schema, row_count))
}

/// 1 行分（配列の 1 要素、またはトップレベル単体の値）を列集合へ畳み込む。
/// `row` はちょうどその値のバイト列（前後に空白無し）。
fn accumulate_row(
    row: &[u8],
    names: &mut Vec<String>,
    infs: &mut Vec<Inf>,
    key: &mut Vec<u8>,
) -> Result<()> {
    let c = byte_at(row, 0)?;
    if c == b'{' {
        let mut it = Members::new(row)?;
        while let Some(m) = it.next()? {
            let name = member_key(&m, key)?;
            merge(names, infs, name, infer(&m))?;
        }
    } else {
        // オブジェクトでない行は `"json"` 列 1 本に生の値を入れる
        // （モジュール doc）。
        let m = Member { key: b"json", key_escaped: false, val: row, kind: kind_of(c) };
        merge(names, infs, b"json", infer(&m))?;
    }
    Ok(())
}

/// 列 `name` の推論型を `inf` で束ねる。列が無ければ新設する。
fn merge(names: &mut Vec<String>, infs: &mut Vec<Inf>, name: &[u8], inf: Inf) -> Result<()> {
    let idx = match names.iter().position(|s| s.as_bytes() == name) {
        Some(i) => i,
        None => {
            ensure!(names.len() < MAX_COLUMNS, LimitExceeded);
            names.push(String::from_utf8_lossy(name).into_owned());
            infs.push(Inf::Null);
            names.len() - 1
        }
    };
    infs[idx] = widen(infs[idx], inf);
    Ok(())
}

/// 推論中の列の型。`Ty` を直接使わないのは、DATE/TIMESTAMP/JSON を
/// 「文字列またはネストから昇格した特別扱い」として束の中で区別したいから。
#[derive(Clone, Copy, PartialEq, Eq)]
enum Inf {
    Null,
    Bool,
    Int,
    Double,
    Date,
    Timestamp,
    /// 日付/タイムスタンプとして解釈できなかったプレーン文字列。
    Str,
    /// ネストした値、または噛み合わない型どうしの混在。生 JSON テキストの
    /// まま `Ty::Json` 列に持たせる。
    Json,
}

impl Inf {
    fn ty(self) -> Ty {
        match self {
            // サンプル中すべて NULL。`format::jsonl` と同じく安全側の
            // VARCHAR にしておく（モジュール doc の duckdb との相違点）。
            Inf::Null => Ty::Varchar,
            Inf::Bool => Ty::Boolean,
            Inf::Int => Ty::BigInt,
            Inf::Double => Ty::Double,
            Inf::Date => Ty::Date,
            Inf::Timestamp => Ty::Timestamp,
            Inf::Str => Ty::Varchar,
            Inf::Json => Ty::Json,
        }
    }
}

/// NULL → BOOLEAN → BIGINT → DOUBLE の束、文字列系（Str/Date/Timestamp）は
/// 互いに VARCHAR へ、それ以外の噛み合わない組は JSON へ倒す
/// （モジュール doc の duckdb 実測を参照）。
fn widen(a: Inf, b: Inf) -> Inf {
    use Inf::*;
    if a == b {
        return a;
    }
    match (a, b) {
        (Null, x) | (x, Null) => x,
        (Bool, Int) | (Int, Bool) => Int,
        (Bool, Double) | (Double, Bool) => Double,
        (Int, Double) | (Double, Int) => Double,
        (Date, Timestamp) | (Timestamp, Date) => Timestamp,
        // 文字列系（Str/Date/Timestamp）どうしの残りの組は、どれも decode
        // 後のテキストとして表現できるので VARCHAR 側（Str）に寄せる。
        (Str, Date) | (Date, Str) => Str,
        (Str, Timestamp) | (Timestamp, Str) => Str,
        // 数値/真偽値と文字列の混在、ネストが絡む混在は生 JSON テキストでしか
        // 表現できないので JSON に倒す。
        _ => Json,
    }
}

fn infer(m: &Member<'_>) -> Inf {
    match m.kind {
        Kind::Null => Inf::Null,
        Kind::Bool => Inf::Bool,
        Kind::Num => {
            // 小数点・指数があれば無条件に DOUBLE。i64 に入らない整数も DOUBLE。
            if m.val.iter().any(|&c| c == b'.' || c == b'e' || c == b'E') {
                Inf::Double
            } else if parse_i64(m.val).is_some() {
                Inf::Int
            } else {
                Inf::Double
            }
        }
        Kind::Str => {
            // エスケープを含む文字列が日付になることはまず無いので復号しない。
            if m.value_escaped() {
                return Inf::Str;
            }
            let s = m.str_body();
            if parse_date(s).is_some() {
                Inf::Date
            } else if parse_timestamp(s).is_some() {
                Inf::Timestamp
            } else {
                Inf::Str
            }
        }
        // ネストは常に JSON（モジュール doc）。
        Kind::Nested => Inf::Json,
    }
}

// --- 列ビルダ ---------------------------------------------------------------

enum Cell<'a> {
    Bool(bool),
    I32(i32),
    I64(i64),
    F64(f64),
    Bytes(&'a [u8]),
}

/// 1 列分の組み立て。データ長と validity 長を必ず 1:1 で進めることだけを守る。
struct Builder {
    ty: Ty,
    data: Data,
    valid: Bitmap,
    any_null: bool,
}

impl Builder {
    fn new(ty: Ty, cap: usize) -> Self {
        Builder {
            ty,
            data: Data::with_capacity(ty.phys(), cap),
            valid: Bitmap::with_capacity(cap),
            any_null: false,
        }
    }

    /// `None` と物理型の不一致はどちらも NULL。型が合わない値を捨てるのは、
    /// 推論がサンプル由来で後続行が外れうるため（エラーにはしない）。
    fn push(&mut self, cell: Option<Cell<'_>>) {
        let ok = match (&mut self.data, &cell) {
            (Data::Bool(d), Some(Cell::Bool(v))) => {
                d.push(*v);
                true
            }
            (Data::I32(d), Some(Cell::I32(v))) => {
                d.push(*v);
                true
            }
            (Data::I64(d), Some(Cell::I64(v))) => {
                d.push(*v);
                true
            }
            (Data::F64(d), Some(Cell::F64(v))) => {
                d.push(*v);
                true
            }
            (Data::Bytes(d), Some(Cell::Bytes(v))) => {
                d.push(v);
                true
            }
            _ => false,
        };
        if !ok {
            match &mut self.data {
                Data::Bool(d) => d.push(false),
                Data::I32(d) => d.push(0),
                Data::I64(d) => d.push(0),
                Data::F64(d) => d.push(0.0),
                Data::I128(d) => d.push(0),
                Data::Bytes(d) => d.push_empty(),
            }
            self.any_null = true;
        }
        self.valid.push(ok);
    }

    fn finish(self) -> Vector {
        let v = if self.any_null { Some(self.valid) } else { None };
        Vector::from_data(self.ty, self.data, v)
    }
}

/// 1 行分の値を、射影された列 (`names`) の中の対応する位置へ振り分ける。
fn process_row<'a>(
    row: &'a [u8],
    names: &[&[u8]],
    slots: &mut [Option<Member<'a>>],
    builders: &mut [Builder],
    key: &mut Vec<u8>,
    val: &mut Vec<u8>,
) -> Result<()> {
    for s in slots.iter_mut() {
        *s = None;
    }
    let c = byte_at(row, 0)?;
    if c == b'{' {
        let mut it = Members::new(row)?;
        while let Some(m) = it.next()? {
            let name = member_key(&m, key)?;
            if let Some(j) = names.iter().position(|n| *n == name) {
                slots[j] = Some(m);
            }
        }
    } else if let Some(j) = names.iter().position(|n| *n == b"json") {
        slots[j] = Some(Member { key: b"json", key_escaped: false, val: row, kind: kind_of(c) });
    }
    for (j, b) in builders.iter_mut().enumerate() {
        push_member(b, slots[j].as_ref(), val)?;
    }
    Ok(())
}

/// メンバ 1 つを列の型に合わせて積む。キー欠落 (`None`) と `null` は NULL。
fn push_member(b: &mut Builder, m: Option<&Member<'_>>, scratch: &mut Vec<u8>) -> Result<()> {
    let m = match m {
        Some(m) if m.kind != Kind::Null => m,
        _ => {
            b.push(None);
            return Ok(());
        }
    };
    match b.ty {
        Ty::Boolean => b.push(match m.kind {
            Kind::Bool => Some(Cell::Bool(m.val.first() == Some(&b't'))),
            _ => None,
        }),
        Ty::BigInt => b.push(match m.kind {
            Kind::Num => parse_i64(m.val).map(Cell::I64),
            _ => None,
        }),
        Ty::Double => b.push(match m.kind {
            Kind::Num => parse_f64(m.val).map(Cell::F64),
            _ => None,
        }),
        Ty::Date => b.push(match m.kind {
            Kind::Str => parse_date(m.str_body()).map(Cell::I32),
            _ => None,
        }),
        Ty::Timestamp => b.push(match m.kind {
            Kind::Str => parse_timestamp(m.str_body()).map(Cell::I64),
            _ => None,
        }),
        // JSON は常に生の JSON テキストのまま（文字列でも引用符ごと保持する。
        // `Ty::Json` の物理表現は「UTF-8 の JSON テキスト」であって decode 後
        // の文字列ではないため — `vector::Ty::Json` の doc 参照）。
        Ty::Json => b.push(Some(Cell::Bytes(m.val))),
        // VARCHAR は何でも受ける。文字列は復号し、それ以外は生の JSON テキスト。
        _ => match m.kind {
            Kind::Str => {
                scratch.clear();
                decode_string(m.str_body(), scratch)?;
                b.push(Some(Cell::Bytes(scratch)));
            }
            _ => b.push(Some(Cell::Bytes(m.val))),
        },
    }
    Ok(())
}

// --- JSON 走査 --------------------------------------------------------------
//
// `format::jsonl` の私有スキャナと同じ考え方の独立実装（モジュール doc 参照）。

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Null,
    Bool,
    Num,
    Str,
    /// 配列またはオブジェクト。
    Nested,
}

#[derive(Clone, Copy)]
struct Member<'a> {
    /// キーの本体（引用符の内側、エスケープ未展開）。オブジェクトでない行を
    /// `"json"` 列として合成したときは `b"json"`（エスケープ無し）を使う。
    key: &'a [u8],
    key_escaped: bool,
    /// 値のスパン。文字列なら引用符を含む。
    val: &'a [u8],
    kind: Kind,
}

impl<'a> Member<'a> {
    /// 文字列値の本体（引用符を除く）。`kind == Str` のときだけ意味を持つ。
    fn str_body(&self) -> &'a [u8] {
        let n = self.val.len();
        if n >= 2 {
            &self.val[1..n - 1]
        } else {
            &[]
        }
    }

    /// 文字列値がエスケープを含むか。
    fn value_escaped(&self) -> bool {
        self.str_body().contains(&b'\\')
    }
}

/// トップレベルオブジェクトのメンバを順に返す。再帰しない。
/// `obj` はちょうど 1 個のオブジェクト（前後に余分なバイト無し）。
struct Members<'a> {
    b: &'a [u8],
    i: usize,
    /// 既に 1 つ以上返したか。カンマの要否判定に使う。
    started: bool,
    done: bool,
}

impl<'a> Members<'a> {
    fn new(obj: &'a [u8]) -> Result<Self> {
        let i = skip_ws(obj, 0);
        ensure!(byte_at(obj, i)? == b'{', SyntaxError, i);
        Ok(Members { b: obj, i: i + 1, started: false, done: false })
    }

    fn next(&mut self) -> Result<Option<Member<'a>>> {
        if self.done {
            return Ok(None);
        }
        let b = self.b;
        let mut i = skip_ws(b, self.i);
        let mut c = byte_at(b, i)?;
        if c == b'}' {
            self.done = true;
            // 閉じた後に残るのは空白だけ。`{}{}` のような入力は弾く。
            ensure!(skip_ws(b, i + 1) == b.len(), SyntaxError, i + 1);
            return Ok(None);
        }
        if self.started {
            ensure!(c == b',', SyntaxError, i);
            i = skip_ws(b, i + 1);
            c = byte_at(b, i)?;
        }
        ensure!(c == b'"', SyntaxError, i);
        let (key, key_escaped, ni) = scan_string(b, i)?;
        i = skip_ws(b, ni);
        ensure!(byte_at(b, i)? == b':', SyntaxError, i);
        i = skip_ws(b, i + 1);
        let kind = kind_of(byte_at(b, i)?);
        let v0 = i;
        i = skip_value(b, i)?;
        self.i = i;
        self.started = true;
        Ok(Some(Member { key, key_escaped, val: &b[v0..i], kind }))
    }
}

/// トップレベル配列の要素を順に返す。各要素の `(開始, 終了)` バイト位置
/// （前後の空白・カンマは含まない、値そのもののスパン）を返す。
struct Elements<'a> {
    b: &'a [u8],
    /// 次に読む位置。イテレーションが終わった後は `]` の直後を指す。
    i: usize,
    started: bool,
    done: bool,
}

impl<'a> Elements<'a> {
    /// `at` は `[` の位置。
    fn new(b: &'a [u8], at: usize) -> Result<Self> {
        ensure!(byte_at(b, at)? == b'[', SyntaxError, at);
        Ok(Elements { b, i: at + 1, started: false, done: false })
    }

    fn next(&mut self) -> Result<Option<(usize, usize)>> {
        if self.done {
            return Ok(None);
        }
        let b = self.b;
        let mut i = skip_ws(b, self.i);
        let mut c = byte_at(b, i)?;
        if c == b']' {
            self.done = true;
            self.i = i + 1;
            return Ok(None);
        }
        if self.started {
            ensure!(c == b',', SyntaxError, i);
            i = skip_ws(b, i + 1);
            c = byte_at(b, i)?;
        }
        let _ = c;
        let start = i;
        let end = skip_value(b, start)?;
        self.i = end;
        self.started = true;
        Ok(Some((start, end)))
    }
}

fn kind_of(c: u8) -> Kind {
    match c {
        b'{' | b'[' => Kind::Nested,
        b'"' => Kind::Str,
        b't' | b'f' => Kind::Bool,
        b'n' => Kind::Null,
        _ => Kind::Num,
    }
}

/// メンバのキーをバイト列として得る。エスケープ付きだけ `scratch` に復号する。
fn member_key<'k>(m: &Member<'k>, scratch: &'k mut Vec<u8>) -> Result<&'k [u8]> {
    if !m.key_escaped {
        return Ok(m.key);
    }
    scratch.clear();
    decode_string(m.key, scratch)?;
    Ok(scratch)
}

/// `b[i]` から始まる値 1 つを読み飛ばし、その直後の位置を返す。
///
/// 再帰しない。開いているコンテナの種別は `u32` のビットスタックで持ち、
/// 深さは `MAX_DEPTH` で頭打ちにする（`u32` のビット数と一致させてある）。
fn skip_value(b: &[u8], start: usize) -> Result<usize> {
    let mut i = start;
    // ビット 1 = オブジェクト、0 = 配列。最下位ビットが現在のコンテナ。
    let mut stack: u32 = 0;
    let mut depth: u32 = 0;

    'value: loop {
        i = skip_ws(b, i);
        let c = byte_at(b, i)?;
        if c == b'{' || c == b'[' {
            let obj = c == b'{';
            ensure!(depth < MAX_DEPTH, NestingTooDeep, i);
            stack = (stack << 1) | obj as u32;
            depth += 1;
            i = skip_ws(b, i + 1);
            let n = byte_at(b, i)?;
            if (obj && n == b'}') || (!obj && n == b']') {
                // 空コンテナ。値を 1 つ消費したものとして下の閉じ処理へ落ちる。
                i += 1;
                stack >>= 1;
                depth -= 1;
            } else {
                if obj {
                    i = skip_member_key(b, i)?;
                }
                continue 'value;
            }
        } else {
            i = skip_scalar(b, i)?;
        }

        // 値を 1 つ消費した。区切り／閉じ括弧を処理する。
        loop {
            if depth == 0 {
                return Ok(i);
            }
            i = skip_ws(b, i);
            let c = byte_at(b, i)?;
            let obj = stack & 1 == 1;
            match c {
                b',' => {
                    i += 1;
                    if obj {
                        i = skip_member_key(b, skip_ws(b, i))?;
                    }
                    continue 'value;
                }
                b'}' if obj => {
                    i += 1;
                    stack >>= 1;
                    depth -= 1;
                }
                b']' if !obj => {
                    i += 1;
                    stack >>= 1;
                    depth -= 1;
                }
                _ => err!(SyntaxError, i),
            }
        }
    }
}

/// `"key" :` を読み飛ばし、値の開始位置を返す。
fn skip_member_key(b: &[u8], i: usize) -> Result<usize> {
    let i = skip_ws(b, i);
    ensure!(byte_at(b, i)? == b'"', SyntaxError, i);
    let (_, _, ni) = scan_string(b, i)?;
    let ni = skip_ws(b, ni);
    ensure!(byte_at(b, ni)? == b':', SyntaxError, ni);
    Ok(ni + 1)
}

fn skip_scalar(b: &[u8], i: usize) -> Result<usize> {
    match byte_at(b, i)? {
        b'"' => Ok(scan_string(b, i)?.2),
        b't' => skip_lit(b, i, b"true"),
        b'f' => skip_lit(b, i, b"false"),
        b'n' => skip_lit(b, i, b"null"),
        _ => scan_number(b, i),
    }
}

fn skip_lit(b: &[u8], i: usize, w: &[u8]) -> Result<usize> {
    let e = i + w.len();
    ensure!(b.len() >= e, UnexpectedEof, i);
    ensure!(&b[i..e] == w, SyntaxError, i);
    Ok(e)
}

fn scan_number(b: &[u8], start: usize) -> Result<usize> {
    let mut i = start;
    if b.get(i) == Some(&b'-') {
        i += 1;
    }
    let d0 = i;
    while matches!(b.get(i), Some(c) if c.is_ascii_digit()) {
        i += 1;
    }
    ensure!(i > d0, SyntaxError, start);
    if b.get(i) == Some(&b'.') {
        i += 1;
        let f0 = i;
        while matches!(b.get(i), Some(c) if c.is_ascii_digit()) {
            i += 1;
        }
        ensure!(i > f0, SyntaxError, start);
    }
    if matches!(b.get(i), Some(b'e') | Some(b'E')) {
        i += 1;
        if matches!(b.get(i), Some(b'+') | Some(b'-')) {
            i += 1;
        }
        let e0 = i;
        while matches!(b.get(i), Some(c) if c.is_ascii_digit()) {
            i += 1;
        }
        ensure!(i > e0, SyntaxError, start);
    }
    Ok(i)
}

/// `b[i]` は `"`。`(本体, エスケープ有無, 閉じ引用符の次)` を返す。
fn scan_string(b: &[u8], i: usize) -> Result<(&[u8], bool, usize)> {
    let mut j = i + 1;
    let mut esc = false;
    loop {
        match byte_at(b, j)? {
            b'"' => return Ok((&b[i + 1..j], esc, j + 1)),
            b'\\' => {
                // 次の 1 バイトは必ず消費する。`\"` を終端と誤認しないため。
                byte_at(b, j + 1)?;
                esc = true;
                j += 2;
            }
            _ => j += 1,
        }
    }
}

/// 文字列本体のエスケープを展開して UTF-8 として `out` に書く。
fn decode_string(body: &[u8], out: &mut Vec<u8>) -> Result<()> {
    let mut i = 0;
    while i < body.len() {
        let c = body[i];
        if c != b'\\' {
            out.push(c);
            i += 1;
            continue;
        }
        i += 1;
        let e = byte_at(body, i)?;
        i += 1;
        match e {
            b'"' => out.push(b'"'),
            b'\\' => out.push(b'\\'),
            b'/' => out.push(b'/'),
            b'b' => out.push(0x08),
            b'f' => out.push(0x0c),
            b'n' => out.push(b'\n'),
            b'r' => out.push(b'\r'),
            b't' => out.push(b'\t'),
            b'u' => {
                let hi = hex4(body, i)?;
                i += 4;
                let cp = if (0xD800..0xDC00).contains(&hi) {
                    // 上位サロゲート。直後が下位サロゲートなら結合する。
                    match (body.get(i), body.get(i + 1)) {
                        (Some(b'\\'), Some(b'u')) => match hex4(body, i + 2) {
                            Ok(lo) if (0xDC00..0xE000).contains(&lo) => {
                                i += 6;
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            }
                            // 対になっていない上位サロゲートはエラーにせず
                            // U+FFFD に潰す（壊れた入力で行ごと失いたくない）。
                            _ => 0xFFFD,
                        },
                        _ => 0xFFFD,
                    }
                } else if (0xDC00..0xE000).contains(&hi) {
                    // 単独の下位サロゲート。
                    0xFFFD
                } else {
                    hi
                };
                let ch = char::from_u32(cp).unwrap_or(char::REPLACEMENT_CHARACTER);
                let mut buf = [0u8; 4];
                out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            }
            _ => err!(SyntaxError, i),
        }
    }
    Ok(())
}

fn hex4(b: &[u8], i: usize) -> Result<u32> {
    let mut v = 0u32;
    for k in 0..4 {
        let c = byte_at(b, i + k)?;
        let d = match c {
            b'0'..=b'9' => (c - b'0') as u32,
            b'a'..=b'f' => (c - b'a' + 10) as u32,
            b'A'..=b'F' => (c - b'A' + 10) as u32,
            _ => err!(SyntaxError, i + k),
        };
        v = (v << 4) | d;
    }
    Ok(v)
}

// --- 数値・日時 --------------------------------------------------------------

/// 整数リテラルを i64 に。小数点・指数付き、または範囲外なら `None`。
fn parse_i64(s: &[u8]) -> Option<i64> {
    let (neg, ds) = match s.first() {
        Some(b'-') => (true, &s[1..]),
        _ => (false, s),
    };
    if ds.is_empty() {
        return None;
    }
    // 負側で累算する。i64::MIN を特別扱いせずに済む。
    let mut acc: i64 = 0;
    for &c in ds {
        if !c.is_ascii_digit() {
            return None;
        }
        acc = acc.checked_mul(10)?.checked_sub((c - b'0') as i64)?;
    }
    if neg {
        Some(acc)
    } else {
        acc.checked_neg()
    }
}

/// 浮動小数は core の `str::parse::<f64>`（dec2flt）に任せる。
/// 自前で書くより小さく、丸めも正しい。
fn parse_f64(s: &[u8]) -> Option<f64> {
    core::str::from_utf8(s).ok()?.parse::<f64>().ok()
}

/// 先頭 `n` バイトを 10 進数として読む。数字以外が混じれば `None`。
fn digits(s: &[u8], n: usize) -> Option<u32> {
    if s.len() < n {
        return None;
    }
    let mut v = 0u32;
    for &c in &s[..n] {
        if !c.is_ascii_digit() {
            return None;
        }
        v = v * 10 + (c - b'0') as u32;
    }
    Some(v)
}

fn is_leap(y: i32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn days_in_month(y: i32, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap(y) => 29,
        2 => 28,
        _ => 0,
    }
}

/// days-from-civil (Howard Hinnant)。エポック 1970-01-01 からの日数。
fn days_from_civil(y: i32, m: u32, d: u32) -> i32 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u32; // [0, 399]
    let mp = if m > 2 { m - 3 } else { m + 9 }; // 3 月始まりに寄せる
    let doy = (153 * mp + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146097 + doe as i32 - 719468
}

/// `YYYY-MM-DD` をエポックからの日数に。
fn parse_date(s: &[u8]) -> Option<i32> {
    if s.len() != 10 || s[4] != b'-' || s[7] != b'-' {
        return None;
    }
    let y = digits(s, 4)? as i32;
    let m = digits(&s[5..], 2)?;
    let d = digits(&s[8..], 2)?;
    if !(1..=12).contains(&m) || d < 1 || d > days_in_month(y, m) {
        return None;
    }
    Some(days_from_civil(y, m, d))
}

/// `YYYY-MM-DD[ T]HH:MM:SS[.ffffff]` をエポックからのマイクロ秒に。
/// 日付だけの文字列は深夜 0 時として受ける（DATE → TIMESTAMP の昇格用）。
fn parse_timestamp(s: &[u8]) -> Option<i64> {
    let days = parse_date(s.get(..10)?)? as i64;
    if s.len() == 10 {
        return Some(days * 86_400_000_000);
    }
    if s.len() < 19 || (s[10] != b' ' && s[10] != b'T') || s[13] != b':' || s[16] != b':' {
        return None;
    }
    let hh = digits(&s[11..], 2)?;
    let mi = digits(&s[14..], 2)?;
    let ss = digits(&s[17..], 2)?;
    if hh > 23 || mi > 59 || ss > 59 {
        return None;
    }
    let mut micros = 0u32;
    if s.len() > 19 {
        if s[19] != b'.' {
            return None;
        }
        let frac = &s[20..];
        if frac.is_empty() || frac.len() > 9 {
            return None;
        }
        // 6 桁に切り詰め／ゼロ埋めしてマイクロ秒にする。
        for k in 0..6 {
            let d = match frac.get(k) {
                Some(&c) if c.is_ascii_digit() => (c - b'0') as u32,
                Some(_) => return None,
                None => 0,
            };
            micros = micros * 10 + d;
        }
        for &c in frac.iter().skip(6) {
            if !c.is_ascii_digit() {
                return None;
            }
        }
    }
    let secs = days * 86_400 + (hh as i64) * 3600 + (mi as i64) * 60 + ss as i64;
    Some(secs * 1_000_000 + micros as i64)
}

// --- バイト列ユーティリティ ---------------------------------------------------

fn byte_at(b: &[u8], i: usize) -> Result<u8> {
    match b.get(i) {
        Some(&c) => Ok(c),
        None => err!(UnexpectedEof, i),
    }
}

fn skip_ws(b: &[u8], mut i: usize) -> usize {
    while matches!(b.get(i), Some(b' ') | Some(b'\t') | Some(b'\r') | Some(b'\n')) {
        i += 1;
    }
    i
}

/// UTF-8 BOM があれば読み飛ばす。
fn skip_bom(b: &[u8]) -> usize {
    if b.starts_with(&[0xEF, 0xBB, 0xBF]) {
        3
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{code_of, Code};
    use crate::vector::Value;

    fn data_file(name: &str) -> Vec<u8> {
        let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/");
        std::fs::read(format!("{p}{name}")).unwrap_or_else(|e| panic!("{name}: {e}"))
    }

    fn resolve(bytes: &[u8]) -> (JsonFormat, Source) {
        let src = Source::from_bytes(bytes.to_vec());
        let mut f = JsonFormat::new();
        match f.resolve(&src).expect("resolve") {
            Ok(()) => {}
            Err(r) => panic!("メモリ上の Source なのに範囲要求 {r:?} が返った"),
        }
        (f, src)
    }

    fn schema_of(text: &str) -> Vec<(String, Ty)> {
        let (f, _) = resolve(text.as_bytes());
        f.schema().iter().map(|c| (c.name.clone(), c.ty)).collect()
    }

    fn read_all(text: &str) -> Vec<Vec<Value>> {
        let (f, src) = resolve(text.as_bytes());
        let projection: Vec<usize> = (0..f.schema().len()).collect();
        let mut ranges = Vec::new();
        f.split_ranges(0, &projection, &mut ranges).expect("split_ranges");
        let cols = f.read_split(&src, 0, &projection).expect("read_split");
        let mut out: Vec<Vec<Value>> = projection.iter().map(|_| Vec::new()).collect();
        for (j, c) in cols.iter().enumerate() {
            for i in 0..c.len() {
                out[j].push(c.value_at(i));
            }
        }
        out
    }

    fn s(v: &str) -> Value {
        Value::Bytes(v.as_bytes().to_vec())
    }

    fn infer_ty(values: &[&str]) -> Ty {
        let mut text = String::from("[");
        for (i, v) in values.iter().enumerate() {
            if i > 0 {
                text.push(',');
            }
            text.push_str("{\"a\":");
            text.push_str(v);
            text.push('}');
        }
        text.push(']');
        schema_of(&text)[0].1
    }

    // --- 実ファイル -----------------------------------------------------

    #[test]
    fn basic_json_array_schema_matches_duckdb() {
        // duckdb: DESCRIBE SELECT * FROM read_json_auto('tests/data/basic_array.json')
        let (f, _) = resolve(&data_file("basic_array.json"));
        let got: Vec<(&str, Ty)> = f.schema().iter().map(|c| (c.name.as_str(), c.ty)).collect();
        assert_eq!(
            got,
            [
                ("id", Ty::BigInt),
                ("name", Ty::Varchar),
                ("score", Ty::Double),
                ("flag", Ty::Boolean),
                ("big", Ty::BigInt),
                ("d", Ty::Timestamp),
            ]
        );
        assert!(f.schema().iter().all(|c| c.nullable));
    }

    #[test]
    fn basic_json_array_values_match_the_jsonl_reference() {
        let cols = read_all_file("basic_array.json");
        assert!(cols.iter().all(|c| c.len() == 1000));
        assert_eq!(cols[0][0], Value::I64(0));
        assert_eq!(cols[1][0], s("name_0"));
        assert_eq!(cols[2][0], Value::F64(0.0));
        assert_eq!(cols[3][0], Value::Bool(true));
        assert_eq!(cols[4][0], Value::Null);
        assert_eq!(cols[2][1], Value::F64(1.5));
        assert_eq!(cols[4][1], Value::I64(100));
        assert_eq!(cols[5][0], Value::I64(1_704_067_200_000_000));
        assert_eq!(cols[0][999], Value::I64(999));
        assert_eq!(cols[1][999], s("name_5"));
        assert_eq!(cols[4][999], Value::I64(99_900));
    }

    fn read_all_file(name: &str) -> Vec<Vec<Value>> {
        let bytes = data_file(name);
        let src = Source::from_bytes(bytes);
        let mut f = JsonFormat::new();
        f.resolve(&src).expect("resolve").expect("bytes");
        let projection: Vec<usize> = (0..f.schema().len()).collect();
        let mut ranges = Vec::new();
        f.split_ranges(0, &projection, &mut ranges).expect("split_ranges");
        let cols = f.read_split(&src, 0, &projection).expect("read_split");
        let mut out: Vec<Vec<Value>> = projection.iter().map(|_| Vec::new()).collect();
        for (j, c) in cols.iter().enumerate() {
            for i in 0..c.len() {
                out[j].push(c.value_at(i));
            }
        }
        out
    }

    // --- トップレベルの規則 -----------------------------------------------

    #[test]
    fn top_level_object_is_a_single_row_table() {
        // duckdb: SELECT * FROM read_json_auto('single_obj.json') -> 1 行。
        let cols = read_all("{\"a\":1,\"b\":\"hello\"}");
        assert_eq!(cols[0], [Value::I64(1)]);
        assert_eq!(cols[1], [s("hello")]);
    }

    #[test]
    fn array_of_scalars_becomes_a_single_json_named_column() {
        // duckdb: SELECT * FROM read_json_auto('[1,2,3]') -> 列名 "json"。
        let (f, _) = resolve(b"[1,2,3]");
        assert_eq!(f.schema().len(), 1);
        assert_eq!(f.schema()[0].name, "json");
        assert_eq!(f.schema()[0].ty, Ty::BigInt);
        let cols = read_all("[1,2,3]");
        assert_eq!(cols[0], [Value::I64(1), Value::I64(2), Value::I64(3)]);
    }

    #[test]
    fn empty_array_is_a_single_json_column_with_no_rows() {
        // duckdb: SELECT * FROM read_json_auto('[]') -> json JSON, 0 行。
        let (f, _) = resolve(b"[]");
        assert_eq!(f.schema().len(), 1);
        assert_eq!(f.schema()[0].name, "json");
        assert_eq!(f.schema()[0].ty, Ty::Json);
        let cols = read_all("[]");
        assert_eq!(cols[0].len(), 0);
    }

    #[test]
    fn mixed_object_and_scalar_rows_do_not_crash() {
        // duckdb は基本的に想定しない入力だが、壊れずに扱えることを確認する。
        // オブジェクトでない行は "json" 列にだけ入り、他の列は NULL。
        let cols = read_all("[{\"a\":1},5,{\"a\":2}]");
        let names: Vec<_> = {
            let (f, _) = resolve(b"[{\"a\":1},5,{\"a\":2}]");
            f.schema().iter().map(|c| c.name.clone()).collect()
        };
        let a = names.iter().position(|n| n == "a").unwrap();
        let j = names.iter().position(|n| n == "json").unwrap();
        assert_eq!(cols[a], [Value::I64(1), Value::Null, Value::I64(2)]);
        // "json" 列は行 1 の `5` しか寄与しないので BIGINT に推論される
        // （他の行は "json" というキーを持たないオブジェクトなので NULL）。
        assert_eq!(cols[j], [Value::Null, Value::I64(5), Value::Null]);
    }

    // --- 列集合の和・NULL -------------------------------------------------

    #[test]
    fn schema_is_the_union_of_keys_in_first_seen_order() {
        let got = schema_of("[{\"b\":1,\"a\":\"x\"},{\"c\":true},{\"a\":\"y\",\"b\":2}]");
        let names: Vec<&str> = got.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["b", "a", "c"]);
    }

    #[test]
    fn missing_key_and_explicit_null_are_both_null() {
        let cols = read_all("[{\"a\":1,\"b\":2},{\"a\":3},{\"a\":null,\"b\":4}]");
        assert_eq!(cols[0], [Value::I64(1), Value::I64(3), Value::Null]);
        assert_eq!(cols[1], [Value::I64(2), Value::Null, Value::I64(4)]);
    }

    // --- 型推論 -------------------------------------------------------------

    #[test]
    fn inference_lattice_covers_every_transition() {
        assert_eq!(infer_ty(&["null"]), Ty::Varchar);
        assert_eq!(infer_ty(&["true"]), Ty::Boolean);
        assert_eq!(infer_ty(&["null", "true"]), Ty::Boolean);
        assert_eq!(infer_ty(&["1"]), Ty::BigInt);
        assert_eq!(infer_ty(&["true", "1"]), Ty::BigInt);
        assert_eq!(infer_ty(&["1", "1.5"]), Ty::Double);
        assert_eq!(infer_ty(&["true", "1.5"]), Ty::Double);
        // 数値/真偽値と文字列の混在は duckdb と合わせて JSON に倒す
        // （`format::jsonl` は VARCHAR に倒すが、`Ty::Json` が無かった当時の
        // 制約による差 — モジュール doc 参照）。
        assert_eq!(infer_ty(&["1", "\"x\""]), Ty::Json);
        assert_eq!(infer_ty(&["1.5", "\"x\""]), Ty::Json);
        assert_eq!(infer_ty(&["true", "\"x\""]), Ty::Json);
    }

    #[test]
    fn date_and_timestamp_strings_get_temporal_types() {
        assert_eq!(infer_ty(&["\"2024-01-01\""]), Ty::Date);
        assert_eq!(infer_ty(&["\"2024-01-01 12:34:56\""]), Ty::Timestamp);
        assert_eq!(infer_ty(&["\"2024-01-01T12:34:56\""]), Ty::Timestamp);
        assert_eq!(infer_ty(&["\"2024-01-01T12:34:56.123456\""]), Ty::Timestamp);
        assert_eq!(infer_ty(&["\"2024-01-01\"", "\"2024-01-01 00:00:00\""]), Ty::Timestamp);
        // 文字列系どうしの混在は VARCHAR（`format::jsonl` と同じ判断）。
        assert_eq!(infer_ty(&["\"2024-01-01\"", "\"hello\""]), Ty::Varchar);
        assert_eq!(infer_ty(&["\"2024-01-01 00:00:00\"", "\"hello\""]), Ty::Varchar);
        assert_eq!(infer_ty(&["\"2024-02-30\""]), Ty::Varchar);
        assert_eq!(infer_ty(&["\"2024-13-01\""]), Ty::Varchar);
        // 数値と混ざれば JSON（文字列系どうしの混在とは違う扱い）。
        assert_eq!(infer_ty(&["\"2024-01-01\"", "1"]), Ty::Json);
    }

    #[test]
    fn temporal_values_use_the_internal_representation() {
        let cols = read_all("[{\"a\":\"1970-01-02\"},{\"a\":\"2024-01-01\"}]");
        assert_eq!(cols[0], [Value::I32(1), Value::I32(19723)]);

        let cols = read_all(
            "[{\"a\":\"1970-01-01 00:00:01\"},\
              {\"a\":\"1969-12-31 23:59:59\"},\
              {\"a\":\"2024-01-01T00:00:00.000123\"}]",
        );
        assert_eq!(
            cols[0],
            [Value::I64(1_000_000), Value::I64(-1_000_000), Value::I64(1_704_067_200_000_123),]
        );
    }

    #[test]
    fn numbers_cover_integer_negative_fraction_exponent_and_overflow() {
        assert_eq!(infer_ty(&["-3"]), Ty::BigInt);
        assert_eq!(infer_ty(&["1e3"]), Ty::Double);
        assert_eq!(infer_ty(&["-1.5E-2"]), Ty::Double);
        assert_eq!(infer_ty(&["9223372036854775808"]), Ty::Double);
        assert_eq!(infer_ty(&["-9223372036854775809"]), Ty::Double);
        assert_eq!(infer_ty(&["-9223372036854775808"]), Ty::BigInt);

        let cols = read_all("[{\"a\":-3},{\"a\":9223372036854775807}]");
        assert_eq!(cols[0], [Value::I64(-3), Value::I64(i64::MAX)]);

        let cols = read_all("[{\"a\":1e3},{\"a\":-1.5E-2},{\"a\":7}]");
        assert_eq!(cols[0], [Value::F64(1000.0), Value::F64(-0.015), Value::F64(7.0)]);

        // 推論はサンプル (SAMPLE_ELEMENTS 個) しか見ない。そこから外れた行に
        // 型の合わない値が出てきたら、エラーにせずその行だけ NULL にする。
        let mut text = String::from("[");
        for i in 0..SAMPLE_ELEMENTS {
            if i > 0 {
                text.push(',');
            }
            text.push_str("{\"a\":1}");
        }
        text.push_str(",{\"a\":1.5},{\"a\":\"x\"},{\"a\":2}]");
        let cols = read_all(&text);
        assert_eq!(cols[0].len(), SAMPLE_ELEMENTS + 3);
        assert_eq!(cols[0][SAMPLE_ELEMENTS], Value::Null);
        assert_eq!(cols[0][SAMPLE_ELEMENTS + 1], Value::Null);
        assert_eq!(cols[0][SAMPLE_ELEMENTS + 2], Value::I64(2));
    }

    // --- 文字列 ---------------------------------------------------------

    #[test]
    fn string_escapes_are_decoded() {
        let cols = read_all("[{\"a\":\"q\\\"b\\\\s\\/t\\bf\\f\\nn\\rr\\tt\"}]");
        assert_eq!(cols[0][0], s("q\"b\\s/t\u{8}f\u{c}\nn\rr\tt"));
    }

    #[test]
    fn unicode_escapes_and_surrogate_pairs() {
        let cols = read_all("[{\"a\":\"\\u00e9\\u65e5\\uD83D\\uDE00\"}]");
        assert_eq!(cols[0][0], s("é日😀"));
    }

    #[test]
    fn lone_surrogates_become_the_replacement_character() {
        let cols =
            read_all("[{\"a\":\"\\uD800\"},{\"a\":\"\\uDC00x\"},{\"a\":\"\\uD800\\u0041\"}]");
        assert_eq!(cols[0][0], s("\u{FFFD}"));
        assert_eq!(cols[0][1], s("\u{FFFD}x"));
        assert_eq!(cols[0][2], s("\u{FFFD}A"));
    }

    #[test]
    fn escaped_keys_match_their_decoded_name() {
        let (f, _) = resolve(b"[{\"a\\u0062\":1},{\"ab\":2}]");
        assert_eq!(f.schema().len(), 1);
        assert_eq!(f.schema()[0].name, "ab");
        let cols = read_all("[{\"a\\u0062\":1},{\"ab\":2}]");
        assert_eq!(cols[0], [Value::I64(1), Value::I64(2)]);
    }

    // --- 入れ子 -----------------------------------------------------------

    #[test]
    fn nested_values_become_json_typed_columns() {
        // duckdb は struct/list 型に推論するが、このエンジンは
        // LIST/STRUCT を持たないので Ty::Json（生 JSON テキスト）に倒す
        // （モジュール doc、`Ty::Json` の doc 参照）。
        let text = "[{\"a\":[1,2,{\"x\":\"y\"}],\"b\":{\"k\":[true,null]}},\
                     {\"a\":[],\"b\":{}}]";
        let sch = schema_of(text);
        assert_eq!(sch[0].1, Ty::Json);
        assert_eq!(sch[1].1, Ty::Json);
        let cols = read_all(text);
        // Ty::Json は decode せず、引用符を含む生テキストのまま保持する。
        assert_eq!(cols[0][0], s("[1,2,{\"x\":\"y\"}]"));
        assert_eq!(cols[1][0], s("{\"k\":[true,null]}"));
        assert_eq!(cols[0][1], s("[]"));
        assert_eq!(cols[1][1], s("{}"));
    }

    #[test]
    fn nested_mixed_with_scalar_is_json() {
        // duckdb: read_json_auto('nested_mixed.json') で確認した挙動。
        assert_eq!(schema_of("[{\"a\":[1,2,3]},{\"a\":5}]")[0].1, Ty::Json);
        let cols = read_all("[{\"a\":[1,2,3]},{\"a\":5}]");
        assert_eq!(cols[0][0], s("[1,2,3]"));
        assert_eq!(cols[0][1], s("5"));
    }

    #[test]
    fn json_column_preserves_raw_text_including_quotes_for_strings() {
        // duckdb: read_json_auto('str_nested.json') で "hello" が引用符付きの
        // まま JSON 型に入ることを確認した。
        let cols = read_all("[{\"a\":\"hello\"},{\"a\":[1,2,3]}]");
        assert_eq!(schema_of("[{\"a\":\"hello\"},{\"a\":[1,2,3]}]")[0].1, Ty::Json);
        assert_eq!(cols[0][0], s("\"hello\""));
        assert_eq!(cols[0][1], s("[1,2,3]"));
    }

    #[test]
    fn varchar_column_keeps_decoded_text_for_string_family_mix() {
        let cols =
            read_all("[{\"a\":\"2024-01-01\"},{\"a\":\"2024-01-01 00:00:00\"},{\"a\":\"hello\"}]");
        assert_eq!(cols[0], [s("2024-01-01"), s("2024-01-01 00:00:00"), s("hello")]);
    }

    // --- 射影 ---------------------------------------------------------------

    #[test]
    fn projection_returns_requested_columns_in_order() {
        let bytes = data_file("basic_array.json");
        let src = Source::from_bytes(bytes);
        let mut f = JsonFormat::new();
        f.resolve(&src).unwrap().unwrap();
        let proj = [4usize, 1usize];
        let mut ranges = Vec::new();
        f.split_ranges(0, &proj, &mut ranges).unwrap();
        let cols = f.read_split(&src, 0, &proj).unwrap();
        assert_eq!(cols.len(), 2);
        assert_eq!(cols[0].value_at(1), Value::I64(100)); // big
        assert_eq!(cols[1].value_at(1), s("name_1")); // name
        assert_eq!(cols[0].value_at(0), Value::Null);
        // 空射影でも壊れない。
        let cols = f.read_split(&src, 0, &[]).unwrap();
        assert!(cols.is_empty());
    }

    // --- 分割 ---------------------------------------------------------------

    #[test]
    fn split_count_is_always_one_when_resolved() {
        let bytes = data_file("basic_array.json");
        let src = Source::from_bytes(bytes.clone());
        let mut f = JsonFormat::new();
        f.resolve(&src).unwrap().unwrap();
        assert_eq!(f.num_splits(), 1);
        let mut out = Vec::new();
        f.split_ranges(0, &[], &mut out).unwrap();
        assert_eq!(out, [(0, bytes.len() as u64)]);
        assert_eq!(code_of(f.split_ranges(1, &[], &mut out)), Some(Code::Internal));
        assert_eq!(f.split_rows(0), Some(1000));
        assert_eq!(f.split_rows(1), None);
    }

    // --- resolve の I/O 要求 -------------------------------------------------

    #[test]
    fn resolve_requests_the_whole_file_when_bytes_are_missing() {
        let mut f = JsonFormat::new();
        let src = Source::remote(1_000_000);
        match f.resolve(&src).unwrap() {
            Err((off, len)) => {
                assert_eq!(off, 0);
                assert_eq!(len, 1_000_000);
            }
            Ok(()) => panic!("バイトが無いのに解決できるはずがない"),
        }
        assert!(!f.is_resolved());
        assert_eq!(f.num_splits(), 0);
    }

    #[test]
    fn oversized_file_is_rejected_before_reading_any_bytes() {
        let mut f = JsonFormat::new();
        let src = Source::remote(MAX_JSON_BYTES + 1);
        assert_eq!(code_of(f.resolve(&src)), Some(Code::Oom));
    }

    #[test]
    fn empty_file_is_rejected() {
        let src = Source::from_bytes(Vec::new());
        let mut f = JsonFormat::new();
        assert_eq!(code_of(f.resolve(&src)), Some(Code::UnexpectedEof));
    }

    #[test]
    fn unresolved_format_reports_no_splits() {
        let f = JsonFormat::new();
        assert!(!f.is_resolved());
        assert_eq!(f.num_splits(), 0);
        assert!(f.schema().is_empty());
    }

    // --- 壊れた入力 ---------------------------------------------------------

    fn resolve_err(text: &str) -> Option<Code> {
        let src = Source::from_bytes(text.as_bytes().to_vec());
        let mut f = JsonFormat::new();
        match f.resolve(&src) {
            Err(e) => Some(e.code),
            Ok(Err(_)) => None,
            Ok(Ok(())) => None,
        }
    }

    #[test]
    fn malformed_input_is_an_error_not_a_panic() {
        assert_eq!(resolve_err("[{\"a\":\"abc}]"), Some(Code::UnexpectedEof));
        assert_eq!(resolve_err("[{\"a\":1]"), Some(Code::SyntaxError));
        assert_eq!(resolve_err("[{\"a\":1}"), Some(Code::UnexpectedEof));
        assert_eq!(resolve_err("not json"), Some(Code::SyntaxError));
        assert_eq!(resolve_err("[1,2,3"), Some(Code::UnexpectedEof));
        assert_eq!(resolve_err("[1,2,3] garbage"), Some(Code::SyntaxError));
        assert_eq!(resolve_err("{\"a\":1,}"), Some(Code::SyntaxError));
        assert_eq!(resolve_err("[{\"a\":1},]"), Some(Code::SyntaxError));
    }

    #[test]
    fn deeply_nested_input_hits_the_depth_cap() {
        // 配列要素そのものを深くネストさせる。`Elements::next` が要素の
        // スパンを求めるのに `skip_value` を要素の先頭（自分自身の
        // `[`/`{`）から直接呼ぶので、depth の起点はこの要素の最外殻になる
        // （オブジェクトの 1 メンバとして埋め込んだ場合は、そのメンバの
        // 行 = 要素自身の `{` が 1 段消費するので、使える深さが 1 少なくなる。
        // `format::jsonl` の `Members` はメンバの値だけに `skip_value` を
        // 呼ぶので起点がずれない — この違いはモジュール doc の「独立実装」
        // 注記のとおり）。
        let mut text = String::from("[");
        let deep = MAX_DEPTH as usize + 1;
        for _ in 0..deep {
            text.push('[');
        }
        text.push('1');
        for _ in 0..deep {
            text.push(']');
        }
        text.push(']');
        assert_eq!(resolve_err(&text), Some(Code::NestingTooDeep));

        let mut ok = String::from("[");
        for _ in 0..MAX_DEPTH {
            ok.push('[');
        }
        ok.push('1');
        for _ in 0..MAX_DEPTH {
            ok.push(']');
        }
        ok.push(']');
        assert_eq!(resolve_err(&ok), None);
    }

    #[test]
    fn too_many_columns_is_rejected() {
        let mut text = String::from("[");
        for chunk in 0..2 {
            if chunk > 0 {
                text.push(',');
            }
            text.push('{');
            for i in 0..MAX_COLUMNS {
                if i > 0 {
                    text.push(',');
                }
                text.push_str(&format!("\"c{}\":1", chunk * MAX_COLUMNS + i));
            }
            text.push('}');
        }
        text.push(']');
        let src = Source::from_bytes(text.into_bytes());
        let mut f = JsonFormat::new();
        assert_eq!(code_of(f.resolve(&src)), Some(Code::LimitExceeded));
    }

    // --- 補助関数 -----------------------------------------------------------

    #[test]
    fn days_from_civil_matches_known_dates() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(1969, 12, 31), -1);
        assert_eq!(days_from_civil(2000, 3, 1), 11017);
        assert_eq!(days_from_civil(2024, 1, 1), 19723);
        assert_eq!(days_from_civil(2024, 2, 29), 19782);
    }
}
