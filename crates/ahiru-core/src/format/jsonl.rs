//! JSONL (NDJSON) — 1 行 1 JSON オブジェクト。
//!
//! ## 分割
//!
//! 行指向なので統計は無く、分割は固定長バイトチャンク（`TEXT_SPLIT_BYTES`）。
//! 射影を渡されても取得バイトは減らせないが、`read_split` で不要な列の復号を
//! 省ける（`format` モジュールの doc にある 2 段階の後半だけが効く）。
//!
//! ## 入れ子の値の扱い（設計判断）
//!
//! `crate::vector::Ty` に LIST / STRUCT は無い。Parquet 側は入れ子スキーマを
//! `UnsupportedNested` で拒否するが、JSONL で同じことをすると「1 列だけ配列が
//! 混じったファイル」が丸ごと読めなくなる。JSON では入れ子が普通に出てくるので、
//! ここでは**入れ子の値を VARCHAR 列として扱い、その生の JSON テキストを格納する**。
//!
//! つまり `SELECT tags FROM t` は `["a","b"]` という文字列を返す。
//! **入れ子の中への構造的なアクセス（要素取り出し・展開）は未対応。**
//! 必要になったら型システム側に LIST/STRUCT を入れる話になる。
//!
//! ## 型推論
//!
//! 先頭 `SAMPLE_BYTES` の中の最大 `SAMPLE_LINES` 行を見て、列ごとに
//! NULL → BOOLEAN → BIGINT → DOUBLE → VARCHAR の束で広げる。列集合は
//! 行ごとのキーの**和集合**（初出順）で、ある行に無いキーはその行だけ NULL。
//! CSV と違って行ごとに列が欠けうる点がこのフォーマットの本質。
//!
//! ## 入力は信用しない
//!
//! 壊れた JSON は `Err` か NULL セルになる。パニック・範囲外参照・無限再帰は
//! 起こさない。走査は再帰せず、入れ子は `MAX_DEPTH` で打ち切る。

use crate::catalog::Source;
use crate::format::{ResolveStep, TableFormat, TEXT_MAX_RECORD, TEXT_SPLIT_BYTES};
use crate::prelude::*;
use crate::vector::{Bitmap, Data, Field, Ty, Vector};

/// スキーマ推論のために先頭から読むバイト数。
pub const SAMPLE_BYTES: u64 = 256 * 1024;

/// スキーマ推論に使う最大行数。これ以上見ても型はまず変わらない。
pub const SAMPLE_LINES: usize = 1000;

/// 値の入れ子の上限。これを超える入力は `NestingTooDeep`。
/// 32 なのは、開いているコンテナ種別を `u32` のビットスタックで持てるから。
const MAX_DEPTH: u32 = 32;

/// 推論で作る列数の上限。キーが行ごとに違う病的な入力で
/// 無制限に確保しないための歯止め。
const MAX_COLUMNS: usize = 1024;

pub struct JsonlFormat {
    schema: Vec<Field>,
    resolved: bool,
    total_len: u64,
    split_bytes: u64,
}

impl JsonlFormat {
    pub fn new() -> Self {
        JsonlFormat {
            schema: Vec::new(),
            resolved: false,
            total_len: 0,
            split_bytes: TEXT_SPLIT_BYTES,
        }
    }

    /// 分割サイズ。既定は `TEXT_SPLIT_BYTES` (8 MiB)。
    pub fn split_bytes(&self) -> u64 {
        self.split_bytes
    }

    /// 分割サイズを上書きする。小さなファイルでも複数分割の経路を通せるように
    /// しておく（分割境界の扱いはテストしないと壊れていても気づけない）。
    /// 0 は 1 に丸める。
    pub fn set_split_bytes(&mut self, n: u64) {
        self.split_bytes = n.max(1);
    }

    /// 分割 `split` が担当するバイト範囲 `[start, end)`。
    fn chunk(&self, split: usize) -> (u64, u64) {
        let start = (split as u64).saturating_mul(self.split_bytes).min(self.total_len);
        let end = start.saturating_add(self.split_bytes).min(self.total_len);
        (start, end)
    }
}

impl Default for JsonlFormat {
    fn default() -> Self {
        JsonlFormat::new()
    }
}

impl TableFormat for JsonlFormat {
    fn resolve(&mut self, src: &Source) -> Result<ResolveStep> {
        if self.resolved {
            return Ok(Ok(()));
        }
        self.total_len = src.total_len;
        let n = SAMPLE_BYTES.min(src.total_len);
        if n == 0 {
            // 空ファイル。列も分割も無いが、解決自体は成功させる。
            self.resolved = true;
            return Ok(Ok(()));
        }
        let buf = match src.get(0, n as usize) {
            Some(b) => b,
            None => return Ok(Err((0, n))),
        };
        // サンプルがファイル全体でないなら、最後の行は途中で切れている恐れがある。
        let partial = n < src.total_len;

        let mut names: Vec<String> = Vec::new();
        let mut infs: Vec<Inf> = Vec::new();
        let mut key = Vec::new();

        let mut pos = skip_bom(buf);
        let mut lines = 0usize;
        let mut truncated = false;
        while pos < buf.len() && lines < SAMPLE_LINES {
            let (line, next, terminated) = next_line(buf, pos);
            pos = next;
            if !terminated && partial {
                truncated = true;
                break;
            }
            if is_blank(line) {
                continue;
            }
            lines += 1;
            let mut it = Members::new(line)?;
            while let Some(m) = it.next()? {
                let name = member_key(&m, &mut key)?;
                let idx = match names.iter().position(|s| s.as_bytes() == name) {
                    Some(i) => i,
                    None => {
                        ensure!(names.len() < MAX_COLUMNS, LimitExceeded);
                        names.push(String::from_utf8_lossy(name).into_owned());
                        infs.push(Inf::Null);
                        names.len() - 1
                    }
                };
                infs[idx] = widen(infs[idx], infer(&m));
            }
        }
        // サンプル内に 1 行も収まらなかった＝1 レコードが 256 KiB を超えている。
        ensure!(!(truncated && lines == 0), LimitExceeded);

        // JSON の値はいつでも欠けうるので、列はすべて nullable。
        self.schema =
            names.into_iter().zip(infs).map(|(n, i)| Field::new(n, i.ty(), true)).collect();
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
        if !self.resolved || self.total_len == 0 {
            return 0;
        }
        let q = self.total_len / self.split_bytes;
        let r = if self.total_len.is_multiple_of(self.split_bytes) { 0 } else { 1 };
        (q + r) as usize
    }

    fn split_rows(&self, _split: usize) -> Option<u64> {
        // 読むまで行数は分からない。
        None
    }

    fn split_ranges(
        &self,
        split: usize,
        _projection: &[usize],
        out: &mut Vec<(u64, u64)>,
    ) -> Result<()> {
        ensure!(split < self.num_splits(), Internal);
        let (start, end) = self.chunk(split);
        // 行指向なので射影ではバイトは減らない。分割のバイトは常に全部要る。
        // 境界をまたいだレコードを読み切れるよう TEXT_MAX_RECORD だけ余分に取る。
        let read_end = end.saturating_add(TEXT_MAX_RECORD).min(self.total_len);
        out.push((start, read_end));
        Ok(())
    }

    fn read_split(&self, src: &Source, split: usize, projection: &[usize]) -> Result<Vec<Vector>> {
        ensure!(self.resolved, Internal);
        ensure!(split < self.num_splits(), Internal);
        let (chunk_start, chunk_end) = self.chunk(split);
        let read_end = chunk_end.saturating_add(TEXT_MAX_RECORD).min(self.total_len);
        let buf = match src.get(chunk_start, (read_end - chunk_start) as usize) {
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
            // 1 行 64 バイト前後を見込んだ雑な予約。外れても正しさには効かない。
            builders.push(Builder::new(f.ty, (buf.len() / 64).min(1 << 16)));
        }

        // JSON の文字列には生の改行を書けない（`\n` にエスケープされる）ので、
        // `\n` を探すだけでレコード境界が一意に決まる。CSV の引用符と違って
        // 状態を持たずに分割できるのはこのため。
        let mut pos = if split == 0 {
            skip_bom(buf)
        } else {
            // 先頭の改行までは前の分割が読み切る行。
            match find_nl(buf, 0) {
                Some(i) => i + 1,
                // 窓の中に改行が無い＝この分割から始まる行は 1 つも無い。
                None => return Ok(builders.into_iter().map(|b| b.finish()).collect()),
            }
        };

        // 行の所有権: 分割 k(>0) の最初の行は「chunk_start 以降の最初の改行の直後」
        // から始まるので、開始位置は必ず chunk_start より大きい。よって前の分割は
        // 開始位置が chunk_end **以下**の行までを担当しないと、境界ちょうどから
        // 始まる行が誰にも読まれずに消える。
        let mut slots: Vec<Option<Member>> = vec![None; projection.len()];
        let mut key = Vec::new();
        let mut val = Vec::new();
        while pos < buf.len() {
            let abs = chunk_start + pos as u64;
            if abs > chunk_end {
                break;
            }
            let (line, next) = match find_nl(buf, pos) {
                Some(i) => (strip_cr(&buf[pos..i]), i + 1),
                None => {
                    // 改行が無い。ファイル末尾なら最終行、そうでなければ
                    // TEXT_MAX_RECORD を超える長すぎるレコード。
                    ensure!(read_end == self.total_len, LimitExceeded);
                    (strip_cr(&buf[pos..]), buf.len())
                }
            };
            pos = next;
            if is_blank(line) {
                continue;
            }

            for s in slots.iter_mut() {
                *s = None;
            }
            let mut it = Members::new(line)?;
            while let Some(m) = it.next()? {
                let name = member_key(&m, &mut key)?;
                // 射影された列だけ位置を覚える。それ以外の値は復号しない
                // （Members が値スパンを読み飛ばすだけで済ませている）。
                if let Some(j) = names.iter().position(|n| *n == name) {
                    slots[j] = Some(m);
                }
            }
            for (j, b) in builders.iter_mut().enumerate() {
                push_member(b, slots[j].as_ref(), &mut val)?;
            }
        }

        Ok(builders.into_iter().map(|b| b.finish()).collect())
    }
}

// --- 型推論 -----------------------------------------------------------------

/// 推論中の列の型。`Ty` を直接使わないのは、DATE/TIMESTAMP を
/// 「文字列から昇格した特別扱い」として束の中で区別したいから。
#[derive(Clone, Copy, PartialEq, Eq)]
enum Inf {
    Null,
    Bool,
    Int,
    Double,
    Date,
    Timestamp,
    Varchar,
}

impl Inf {
    fn ty(self) -> Ty {
        match self {
            // サンプル中すべて NULL。以降どんな値が来ても落とさずに済む
            // VARCHAR にしておく。
            Inf::Null | Inf::Varchar => Ty::Varchar,
            Inf::Bool => Ty::Boolean,
            Inf::Int => Ty::BigInt,
            Inf::Double => Ty::Double,
            Inf::Date => Ty::Date,
            Inf::Timestamp => Ty::Timestamp,
        }
    }
}

/// NULL → BOOLEAN → BIGINT → DOUBLE → VARCHAR の束。
/// 系統をまたぐ混在（数値と日付など）はテキストに落とすしかない。
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
        _ => Varchar,
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
                return Inf::Varchar;
            }
            let s = m.str_body();
            if parse_date(s).is_some() {
                Inf::Date
            } else if parse_timestamp(s).is_some() {
                Inf::Timestamp
            } else {
                Inf::Varchar
            }
        }
        // 入れ子は生 JSON テキストとして VARCHAR に載せる（モジュール doc 参照）。
        Kind::Nested => Inf::Varchar,
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
    /// キーの本体（引用符の内側、エスケープ未展開）。
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
struct Members<'a> {
    b: &'a [u8],
    i: usize,
    /// 既に 1 つ以上返したか。カンマの要否判定に使う。
    started: bool,
    done: bool,
}

impl<'a> Members<'a> {
    fn new(line: &'a [u8]) -> Result<Self> {
        let i = skip_ws(line, 0);
        // オブジェクト以外の行は受け付けない。
        ensure!(byte_at(line, i)? == b'{', SyntaxError, i);
        Ok(Members { b: line, i: i + 1, started: false, done: false })
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
            // 閉じた後に残るのは空白だけ。`{}{}` のような行は弾く。
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

fn find_nl(b: &[u8], from: usize) -> Option<usize> {
    b.get(from..)?.iter().position(|&c| c == b'\n').map(|i| i + from)
}

/// CRLF の `\r` を落とす。
fn strip_cr(line: &[u8]) -> &[u8] {
    match line.split_last() {
        Some((b'\r', rest)) => rest,
        _ => line,
    }
}

fn is_blank(line: &[u8]) -> bool {
    line.iter().all(|&c| matches!(c, b' ' | b'\t' | b'\r'))
}

/// `pos` から 1 行取り出す。`(行, 次の位置, 改行で終わっていたか)`。
fn next_line(b: &[u8], pos: usize) -> (&[u8], usize, bool) {
    match find_nl(b, pos) {
        Some(i) => (strip_cr(&b[pos..i]), i + 1, true),
        None => (strip_cr(&b[pos..]), b.len(), false),
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

    /// 解決してスキーマを返す。
    fn resolve(bytes: &[u8]) -> (JsonlFormat, Source) {
        let src = Source::from_bytes(bytes.to_vec());
        let mut f = JsonlFormat::new();
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

    /// 全分割を読み、列ごとの `Value` 列にして返す。
    fn read_cols(bytes: &[u8], split_bytes: u64, proj: Option<&[usize]>) -> Vec<Vec<Value>> {
        let src = Source::from_bytes(bytes.to_vec());
        let mut f = JsonlFormat::new();
        if split_bytes > 0 {
            f.set_split_bytes(split_bytes);
        }
        f.resolve(&src).expect("resolve").expect("bytes");
        let projection: Vec<usize> = match proj {
            Some(p) => p.to_vec(),
            None => (0..f.schema().len()).collect(),
        };
        let mut out: Vec<Vec<Value>> = projection.iter().map(|_| Vec::new()).collect();
        let mut ranges = Vec::new();
        for s in 0..f.num_splits() {
            ranges.clear();
            f.split_ranges(s, &projection, &mut ranges).expect("split_ranges");
            assert_eq!(ranges.len(), 1, "行指向なので範囲は 1 本のはず");
            let cols = f.read_split(&src, s, &projection).expect("read_split");
            assert_eq!(cols.len(), projection.len());
            let rows = cols.first().map_or(0, |c| c.len());
            assert!(cols.iter().all(|c| c.len() == rows), "列の長さが揃っていない");
            for (j, c) in cols.iter().enumerate() {
                for i in 0..c.len() {
                    out[j].push(c.value_at(i));
                }
            }
        }
        out
    }

    fn read_all(text: &str) -> Vec<Vec<Value>> {
        read_cols(text.as_bytes(), 0, None)
    }

    fn s(v: &str) -> Value {
        Value::Bytes(v.as_bytes().to_vec())
    }

    /// 1 列だけのファイルを推論して型を得る。
    fn infer_ty(values: &[&str]) -> Ty {
        let mut text = String::new();
        for v in values {
            text.push_str("{\"a\":");
            text.push_str(v);
            text.push_str("}\n");
        }
        schema_of(&text)[0].1
    }

    // --- 実ファイル ---------------------------------------------------------

    #[test]
    fn basic_jsonl_schema_matches_duckdb() {
        // duckdb: DESCRIBE SELECT * FROM read_json_auto('tests/data/basic.jsonl')
        let (f, _) = resolve(&data_file("basic.jsonl"));
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
    fn basic_jsonl_values_match_the_parquet_reference() {
        let cols = read_cols(&data_file("basic.jsonl"), 0, None);
        assert!(cols.iter().all(|c| c.len() == 1000));
        assert_eq!(cols[0][0], Value::I64(0));
        assert_eq!(cols[1][0], s("name_0"));
        assert_eq!(cols[2][0], Value::F64(0.0));
        assert_eq!(cols[3][0], Value::Bool(true));
        assert_eq!(cols[4][0], Value::Null);
        assert_eq!(cols[2][1], Value::F64(1.5));
        assert_eq!(cols[4][1], Value::I64(100));
        // duckdb: SELECT epoch_us(d) FROM ... LIMIT 1  → 1704067200000000
        assert_eq!(cols[5][0], Value::I64(1_704_067_200_000_000));
        // 末尾。
        assert_eq!(cols[0][999], Value::I64(999));
        assert_eq!(cols[1][999], s("name_5"));
        assert_eq!(cols[4][999], Value::I64(99_900));
    }

    #[test]
    fn null_positions_match_the_parquet_file() {
        let cols = read_cols(&data_file("basic.jsonl"), 0, None);
        #[allow(clippy::needless_range_loop)]
        for i in 0..1000 {
            assert_eq!(
                cols[4][i] == Value::Null,
                i % 5 == 0,
                "行 {i} の big の NULL 判定がずれている"
            );
        }
    }

    // --- 列集合の和 ---------------------------------------------------------

    #[test]
    fn schema_is_the_union_of_keys_in_first_seen_order() {
        let got = schema_of(
            "{\"b\":1,\"a\":\"x\"}\n\
             {\"c\":true}\n\
             {\"a\":\"y\",\"b\":2}\n",
        );
        let names: Vec<&str> = got.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["b", "a", "c"]);
    }

    #[test]
    fn missing_key_and_explicit_null_are_both_null() {
        let cols = read_all(
            "{\"a\":1,\"b\":2}\n\
             {\"a\":3}\n\
             {\"a\":null,\"b\":4}\n",
        );
        assert_eq!(cols[0], [Value::I64(1), Value::I64(3), Value::Null]);
        // 2 行目は b というキー自体が無い。
        assert_eq!(cols[1], [Value::I64(2), Value::Null, Value::I64(4)]);
    }

    // --- 型推論 -------------------------------------------------------------

    #[test]
    fn inference_lattice_covers_every_transition() {
        assert_eq!(infer_ty(&["null"]), Ty::Varchar); // 全 NULL は VARCHAR に倒す
        assert_eq!(infer_ty(&["true"]), Ty::Boolean);
        assert_eq!(infer_ty(&["null", "true"]), Ty::Boolean);
        assert_eq!(infer_ty(&["1"]), Ty::BigInt);
        assert_eq!(infer_ty(&["true", "1"]), Ty::BigInt);
        assert_eq!(infer_ty(&["1", "1.5"]), Ty::Double);
        assert_eq!(infer_ty(&["true", "1.5"]), Ty::Double);
        assert_eq!(infer_ty(&["1", "\"x\""]), Ty::Varchar);
        assert_eq!(infer_ty(&["1.5", "\"x\""]), Ty::Varchar);
        assert_eq!(infer_ty(&["true", "\"x\""]), Ty::Varchar);
    }

    #[test]
    fn date_and_timestamp_strings_get_temporal_types() {
        assert_eq!(infer_ty(&["\"2024-01-01\""]), Ty::Date);
        assert_eq!(infer_ty(&["\"2024-01-01 12:34:56\""]), Ty::Timestamp);
        assert_eq!(infer_ty(&["\"2024-01-01T12:34:56\""]), Ty::Timestamp);
        assert_eq!(infer_ty(&["\"2024-01-01T12:34:56.123456\""]), Ty::Timestamp);
        // DATE と TIMESTAMP の混在は TIMESTAMP に寄せる。
        assert_eq!(infer_ty(&["\"2024-01-01\"", "\"2024-01-01 00:00:00\""]), Ty::Timestamp);
        // 他の文字列が混ざれば VARCHAR に落ちる。
        assert_eq!(infer_ty(&["\"2024-01-01\"", "\"hello\""]), Ty::Varchar);
        assert_eq!(infer_ty(&["\"2024-01-01 00:00:00\"", "\"hello\""]), Ty::Varchar);
        // 日付として無効なものは日付にしない。
        assert_eq!(infer_ty(&["\"2024-02-30\""]), Ty::Varchar);
        assert_eq!(infer_ty(&["\"2024-13-01\""]), Ty::Varchar);
        // 数値と混ざれば VARCHAR。
        assert_eq!(infer_ty(&["\"2024-01-01\"", "1"]), Ty::Varchar);
    }

    #[test]
    fn temporal_values_use_the_internal_representation() {
        let cols = read_all("{\"a\":\"1970-01-02\"}\n{\"a\":\"2024-01-01\"}\n");
        assert_eq!(cols[0], [Value::I32(1), Value::I32(19723)]);

        let cols = read_all(
            "{\"a\":\"1970-01-01 00:00:01\"}\n\
             {\"a\":\"1969-12-31 23:59:59\"}\n\
             {\"a\":\"2024-01-01T00:00:00.000123\"}\n",
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
        // i64 に入らない整数は DOUBLE へ広げる。
        assert_eq!(infer_ty(&["9223372036854775808"]), Ty::Double);
        assert_eq!(infer_ty(&["-9223372036854775809"]), Ty::Double);
        assert_eq!(infer_ty(&["-9223372036854775808"]), Ty::BigInt);

        let cols = read_all("{\"a\":-3}\n{\"a\":9223372036854775807}\n");
        assert_eq!(cols[0], [Value::I64(-3), Value::I64(i64::MAX)]);

        let cols = read_all("{\"a\":1e3}\n{\"a\":-1.5E-2}\n{\"a\":7}\n");
        assert_eq!(cols[0], [Value::F64(1000.0), Value::F64(-0.015), Value::F64(7.0)]);

        // 推論はサンプル (SAMPLE_LINES 行) しか見ない。そこから外れた行に
        // 型の合わない値が出てきたら、エラーにせずその行だけ NULL にする。
        let mut text = String::new();
        for _ in 0..SAMPLE_LINES {
            text.push_str("{\"a\":1}\n");
        }
        text.push_str("{\"a\":1.5}\n{\"a\":\"x\"}\n{\"a\":[1]}\n{\"a\":2}\n");
        let cols = read_all(&text);
        assert_eq!(cols[0].len(), SAMPLE_LINES + 4);
        assert_eq!(cols[0][SAMPLE_LINES], Value::Null);
        assert_eq!(cols[0][SAMPLE_LINES + 1], Value::Null);
        assert_eq!(cols[0][SAMPLE_LINES + 2], Value::Null);
        assert_eq!(cols[0][SAMPLE_LINES + 3], Value::I64(2));
    }

    // --- 文字列 -------------------------------------------------------------

    #[test]
    fn string_escapes_are_decoded() {
        let cols = read_all("{\"a\":\"q\\\"b\\\\s\\/t\\bf\\f\\nn\\rr\\tt\"}\n");
        assert_eq!(cols[0][0], s("q\"b\\s/t\u{8}f\u{c}\nn\rr\tt"));
    }

    #[test]
    fn unicode_escapes_and_surrogate_pairs() {
        // \u00e9 = é, \u65e5 = 日, \uD83D\uDE00 = 😀
        let cols = read_all("{\"a\":\"\\u00e9\\u65e5\\uD83D\\uDE00\"}\n");
        assert_eq!(cols[0][0], s("é日😀"));
    }

    #[test]
    fn lone_surrogates_become_the_replacement_character() {
        // 対にならない上位／下位サロゲートはエラーにせず U+FFFD にする。
        let cols = read_all(
            "{\"a\":\"\\uD800\"}\n\
             {\"a\":\"\\uDC00x\"}\n\
             {\"a\":\"\\uD800\\u0041\"}\n",
        );
        assert_eq!(cols[0][0], s("\u{FFFD}"));
        assert_eq!(cols[0][1], s("\u{FFFD}x"));
        assert_eq!(cols[0][2], s("\u{FFFD}A"));
    }

    #[test]
    fn escaped_keys_match_their_decoded_name() {
        let (f, _) = resolve(b"{\"a\\u0062\":1}\n{\"ab\":2}\n");
        // 同じ名前に解決されるので列は 1 本。
        assert_eq!(f.schema().len(), 1);
        assert_eq!(f.schema()[0].name, "ab");
        let cols = read_all("{\"a\\u0062\":1}\n{\"ab\":2}\n");
        assert_eq!(cols[0], [Value::I64(1), Value::I64(2)]);
    }

    // --- 入れ子 -------------------------------------------------------------

    #[test]
    fn nested_values_are_kept_as_raw_json_text() {
        let text = "{\"a\":[1,2,{\"x\":\"y\"}],\"b\":{\"k\":[true,null]}}\n\
                    {\"a\":[],\"b\":{}}\n";
        let sch = schema_of(text);
        assert_eq!(sch[0].1, Ty::Varchar);
        assert_eq!(sch[1].1, Ty::Varchar);
        let cols = read_all(text);
        assert_eq!(cols[0][0], s("[1,2,{\"x\":\"y\"}]"));
        assert_eq!(cols[1][0], s("{\"k\":[true,null]}"));
        assert_eq!(cols[0][1], s("[]"));
        assert_eq!(cols[1][1], s("{}"));
    }

    #[test]
    fn varchar_column_keeps_raw_text_for_non_strings() {
        // 型が混ざって VARCHAR になった列は、数値も真偽値も生テキストで載る。
        let cols = read_all("{\"a\":1}\n{\"a\":\"x\"}\n{\"a\":true}\n{\"a\":2.5}\n{\"a\":null}\n");
        assert_eq!(cols[0], [s("1"), s("x"), s("true"), s("2.5"), Value::Null]);
    }

    // --- 行の取り扱い -------------------------------------------------------

    #[test]
    fn blank_lines_are_skipped() {
        let cols = read_all("\n{\"a\":1}\n\n   \n{\"a\":2}\n\n");
        assert_eq!(cols[0], [Value::I64(1), Value::I64(2)]);
    }

    #[test]
    fn bom_and_crlf_are_handled() {
        let mut bytes = vec![0xEF, 0xBB, 0xBF];
        bytes.extend_from_slice(b"{\"a\":1}\r\n\r\n{\"a\":2}\r\n");
        let (f, _) = resolve(&bytes);
        assert_eq!(f.schema()[0].name, "a");
        let cols = read_cols(&bytes, 0, None);
        assert_eq!(cols[0], [Value::I64(1), Value::I64(2)]);
    }

    #[test]
    fn last_line_without_a_newline_is_read() {
        let cols = read_all("{\"a\":1}\n{\"a\":2}");
        assert_eq!(cols[0], [Value::I64(1), Value::I64(2)]);
    }

    // --- 射影 ---------------------------------------------------------------

    #[test]
    fn projection_returns_requested_columns_in_order() {
        let bytes = data_file("basic.jsonl");
        let cols = read_cols(&bytes, 0, Some(&[4, 1]));
        assert_eq!(cols.len(), 2);
        assert_eq!(cols[0][1], Value::I64(100)); // big
        assert_eq!(cols[1][1], s("name_1")); // name
        assert_eq!(cols[0][0], Value::Null);
        // 空射影でも壊れない。
        let cols = read_cols(&bytes, 0, Some(&[]));
        assert!(cols.is_empty());
    }

    // --- 分割 ---------------------------------------------------------------

    #[test]
    fn split_count_and_ranges_cover_the_file() {
        let bytes = data_file("basic.jsonl");
        let src = Source::from_bytes(bytes.clone());
        let mut f = JsonlFormat::new();
        f.set_split_bytes(1024);
        f.resolve(&src).unwrap().unwrap();
        let total = bytes.len() as u64;
        assert_eq!(f.num_splits() as u64, total.div_euclid(1024) + 1);
        let mut out = Vec::new();
        for i in 0..f.num_splits() {
            out.clear();
            f.split_ranges(i, &[], &mut out).unwrap();
            let (a, b) = out[0];
            assert_eq!(a, i as u64 * 1024);
            // 末尾の読み越し分はファイル長で頭打ち。
            assert_eq!(b, (a + 1024 + TEXT_MAX_RECORD).min(total));
        }
        // 範囲外の分割は要求できない。
        assert_eq!(code_of(f.split_ranges(f.num_splits(), &[], &mut out)), Some(Code::Internal));
    }

    #[test]
    fn multi_split_reads_every_row_exactly_once() {
        let bytes = data_file("basic.jsonl");
        let reference = read_cols(&bytes, 0, None);
        // 分割サイズを変えても行数・値・順序が変わらないこと。
        // 84 バイト/行なので、下の値はどれも行の途中で切れる。
        for &sb in &[1u64, 7, 63, 84, 85, 128, 1000, 4096] {
            let cols = read_cols(&bytes, sb, None);
            assert_eq!(cols[0].len(), 1000, "split_bytes={sb} で行数がずれた");
            for (j, col) in cols.iter().enumerate() {
                assert_eq!(col, &reference[j], "split_bytes={sb} の列 {j} がずれた");
            }
        }
    }

    #[test]
    fn split_boundary_values_are_not_duplicated_or_dropped() {
        // 1 行 11 バイト固定 (`{"a":0000}\n`) にそろえ、分割サイズを行長の
        // 倍数（境界がちょうど行頭に来る＝取りこぼしが起きやすい）と
        // 非倍数の両方で試す。
        let mut text = String::new();
        for i in 0..500 {
            text.push_str(&format!("{{\"a\":{i:04}}}\n"));
        }
        let bytes = text.into_bytes();
        assert_eq!(bytes.len(), 500 * 11);
        let expect: Vec<Value> = (0..500).map(Value::I64).collect();
        for &sb in &[5u64, 10, 11, 12, 22, 23, 110, 121] {
            let cols = read_cols(&bytes, sb, None);
            assert_eq!(cols[0], expect, "split_bytes={sb} で境界の行がずれた");
        }
    }

    #[test]
    fn oversized_record_is_rejected_instead_of_being_silently_split() {
        // TEXT_MAX_RECORD を超える行は読み越せない。境界より前で始まり、
        // 読み窓の外まで続く行は LimitExceeded になる。
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"{\"a\":1}\n{\"a\":");
        bytes.resize(bytes.len() + (TEXT_MAX_RECORD as usize + 16), b'0');
        bytes.extend_from_slice(b"}\n");
        let src = Source::from_bytes(bytes);
        let mut f = JsonlFormat::new();
        f.set_split_bytes(8);
        f.resolve(&src).unwrap().unwrap();
        assert_eq!(code_of(f.read_split(&src, 0, &[0])), Some(Code::LimitExceeded));
    }

    // --- 壊れた入力 ---------------------------------------------------------

    fn read_err(text: &str) -> Option<Code> {
        let src = Source::from_bytes(text.as_bytes().to_vec());
        let mut f = JsonlFormat::new();
        match f.resolve(&src) {
            Err(e) => return Some(e.code),
            Ok(Err(_)) => return None,
            Ok(Ok(())) => {}
        }
        let p: Vec<usize> = (0..f.schema().len()).collect();
        code_of(f.read_split(&src, 0, &p))
    }

    #[test]
    fn malformed_input_is_an_error_not_a_panic() {
        // 閉じていない文字列。
        assert_eq!(read_err("{\"a\":\"abc}\n"), Some(Code::UnexpectedEof));
        // 閉じていないオブジェクト。
        assert_eq!(read_err("{\"a\":1\n"), Some(Code::UnexpectedEof));
        assert_eq!(read_err("{\"a\":{\"b\":1}\n"), Some(Code::UnexpectedEof));
        // オブジェクトでない行。
        assert_eq!(read_err("[1,2,3]\n"), Some(Code::SyntaxError));
        assert_eq!(read_err("42\n"), Some(Code::SyntaxError));
        assert_eq!(read_err("\"x\"\n"), Some(Code::SyntaxError));
        // 閉じた後にゴミが続く。
        assert_eq!(read_err("{\"a\":1}{\"a\":2}\n"), Some(Code::SyntaxError));
        // 区切り・コロンの欠落と末尾カンマ。
        assert_eq!(read_err("{\"a\":1 \"b\":2}\n"), Some(Code::SyntaxError));
        assert_eq!(read_err("{\"a\" 1}\n"), Some(Code::SyntaxError));
        assert_eq!(read_err("{\"a\":1,}\n"), Some(Code::SyntaxError));
        // 数値・リテラルの形が壊れている。
        assert_eq!(read_err("{\"a\":1.}\n"), Some(Code::SyntaxError));
        assert_eq!(read_err("{\"a\":-}\n"), Some(Code::SyntaxError));
        assert_eq!(read_err("{\"a\":tru}\n"), Some(Code::SyntaxError));
        // 括弧の対応が取れていない。
        assert_eq!(read_err("{\"a\":[1,2}}\n"), Some(Code::SyntaxError));
        // \u の桁が 16 進でない／足りない。
        assert_eq!(read_err("{\"a\":\"\\u12zz\"}\n"), Some(Code::SyntaxError));
        assert_eq!(read_err("{\"a\":\"\\u12\"}\n"), Some(Code::UnexpectedEof));
        // 未知のエスケープ。
        assert_eq!(read_err("{\"a\":\"\\x\"}\n"), Some(Code::SyntaxError));
    }

    #[test]
    fn deeply_nested_input_hits_the_depth_cap() {
        let mut text = String::from("{\"a\":");
        let deep = MAX_DEPTH as usize + 1;
        for _ in 0..deep {
            text.push('[');
        }
        text.push('1');
        for _ in 0..deep {
            text.push(']');
        }
        text.push_str("}\n");
        assert_eq!(read_err(&text), Some(Code::NestingTooDeep));

        // 上限ちょうどは通る（VARCHAR 列として生テキストが載る）。
        let mut ok = String::from("{\"a\":");
        for _ in 0..MAX_DEPTH {
            ok.push('[');
        }
        ok.push('1');
        for _ in 0..MAX_DEPTH {
            ok.push(']');
        }
        ok.push_str("}\n");
        assert_eq!(read_err(&ok), None);
    }

    #[test]
    fn too_many_columns_is_rejected() {
        // 1 行あたり大量のキーを持つ行を数行。SAMPLE_LINES で頭打ちになる
        // 前に列数の上限へ到達する形にする。
        let mut text = String::new();
        for chunk in 0..2 {
            text.push('{');
            for i in 0..MAX_COLUMNS {
                if i > 0 {
                    text.push(',');
                }
                text.push_str(&format!("\"c{}\":1", chunk * MAX_COLUMNS + i));
            }
            text.push_str("}\n");
        }
        let src = Source::from_bytes(text.into_bytes());
        let mut f = JsonlFormat::new();
        assert_eq!(code_of(f.resolve(&src)), Some(Code::LimitExceeded));
    }

    // --- resolve の I/O 要求 -------------------------------------------------

    #[test]
    fn resolve_requests_the_sample_range_when_bytes_are_missing() {
        let mut f = JsonlFormat::new();
        let src = Source::remote(1_000_000);
        match f.resolve(&src).unwrap() {
            Err((off, len)) => {
                assert_eq!(off, 0);
                assert_eq!(len, SAMPLE_BYTES);
            }
            Ok(()) => panic!("バイトが無いのに解決できるはずがない"),
        }
        assert!(!f.is_resolved());
        assert_eq!(f.num_splits(), 0);

        // ファイルより長い範囲は要求しない。
        let mut f = JsonlFormat::new();
        let src = Source::remote(100);
        match f.resolve(&src).unwrap() {
            Err((off, len)) => assert_eq!((off, len), (0, 100)),
            Ok(()) => panic!("バイトが無いのに解決できるはずがない"),
        }
    }

    #[test]
    fn empty_file_resolves_to_an_empty_schema() {
        let (f, _) = resolve(b"");
        assert!(f.is_resolved());
        assert!(f.schema().is_empty());
        assert_eq!(f.num_splits(), 0);
        assert_eq!(f.split_rows(0), None);
    }

    #[test]
    fn unresolved_format_reports_no_splits() {
        let f = JsonlFormat::new();
        assert!(!f.is_resolved());
        assert_eq!(f.num_splits(), 0);
        assert!(f.schema().is_empty());
        assert_eq!(f.split_bytes(), TEXT_SPLIT_BYTES);
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
