//! JSON の共通基盤: トークナイザ・パス演算子・シリアライズ。
//!
//! `Ty::Json` の物理表現は UTF-8 の JSON テキスト (`vector::types` の doc 参照)。
//! この値をパースし直さず「その場で読み飛ばして目的の部分だけ取り出す」
//! ことで、DOM を組み立てるための入れ子データ構造（LIST/STRUCT 相当の
//! 物理型）を増やさずに済ませる。
//!
//! ## `format::jsonl` との関係
//!
//! `format::jsonl` は NDJSON の 1 行 1 オブジェクトを読み、列指向データへ
//! 組み立てるためのモジュールで、キー/値の走査（`Members`/`Member`）は
//! そのユースケースに特化している。一方このモジュールが要るのは
//! 「値の途中まで読み飛ばして特定の部分木だけ取り出す」ナビゲーションで、
//! 上位の反復子より下の**トークナイザそのもの**（空白・文字列・数値・
//! スカラ・任意の値 1 個の読み飛ばし、エスケープ展開）だけが両者で
//! 完全に共通する。そこでトークナイザ部分のみをこのモジュールへ切り出し、
//! `format::jsonl` はここから `use` する（`Members`/`Member`/スキーマ推論用の
//! 型束・日付パーサなど、NDJSON 固有の部分は元のまま `format::jsonl` に残す）。
//!
//! ## 入力は信用しない
//!
//! 壊れた JSON は `Err` になる（no_std/panic=abort なのでパニックや範囲外
//! 参照は起こさない）。ただしパス走査は**見に行った部分だけ**を検証する
//! 設計上の簡略化がある（下記「既知の制限」参照）。
//!
//! ## パス構文（サポート範囲）
//!
//! DuckDB の `$.a.b[0]` 形式を素直に実装し、加えて先頭の `$` を省略できる
//! （`a.b[0]` も同じ意味。これは DuckDB 非互換の意図的な簡略化）。
//!
//! - `$` または省略: ルート全体
//! - `.key` / 裸の `key`: オブジェクトのメンバ
//! - `."quoted key"`: `.`/`[` を含むキー用。エスケープは `\"` `\\` のみ対応
//!   （JSON 文字列の `\uXXXX` 等の完全なエスケープ集合は非対応の簡略化）
//! - `[N]`: 配列の要素。0 始まり、負数は末尾から（DuckDB と同じ）
//! - 上記を任意個連結できる（`$.a[0].b` 等）
//!
//! **非対応**: JSON Pointer 記法 (`/a/b/0`)。
//!
//! ## 既知の制限
//!
//! - パス走査は「たどった経路上の兄弟要素」までしか構造検証しない。
//!   目的の値が見つかった後に続く兄弟や、開かなかった部分木の中身が
//!   壊れていても検出されないことがある（`whole`/`validate` は逆に
//!   ドキュメント全体を検証する。CAST 時の検証はこちらを使う）。
//! - オブジェクトの等価比較 (`=`)・キー順序の正規化は行わない
//!   （呼び出し元 `expr::kernels::compare` がバイト列比較するだけ）。

use crate::prelude::*;

/// 値の入れ子の上限。`format::jsonl` と同じ値・同じ理由
/// （開いているコンテナ種別を `u32` のビットスタックで持てる）。
pub(crate) const MAX_DEPTH: u32 = 32;

/// JSON 値の種別。
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(Debug))]
pub(crate) enum Kind {
    Null,
    Bool,
    Num,
    Str,
    Object,
    Array,
}

pub(crate) fn kind_of(c: u8) -> Kind {
    match c {
        b'{' => Kind::Object,
        b'[' => Kind::Array,
        b'"' => Kind::Str,
        b't' | b'f' => Kind::Bool,
        b'n' => Kind::Null,
        _ => Kind::Num,
    }
}

// =========================================================================
// トークナイザ（`format::jsonl` と共有する下回り）
// =========================================================================

pub(crate) fn byte_at(b: &[u8], i: usize) -> Result<u8> {
    match b.get(i) {
        Some(&c) => Ok(c),
        None => err!(UnexpectedEof, i),
    }
}

pub(crate) fn skip_ws(b: &[u8], mut i: usize) -> usize {
    while matches!(b.get(i), Some(b' ') | Some(b'\t') | Some(b'\r') | Some(b'\n')) {
        i += 1;
    }
    i
}

/// `b[i]` は `"`。`(本体, エスケープ有無, 閉じ引用符の次)` を返す。
pub(crate) fn scan_string(b: &[u8], i: usize) -> Result<(&[u8], bool, usize)> {
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

pub(crate) fn scan_number(b: &[u8], start: usize) -> Result<usize> {
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

pub(crate) fn skip_lit(b: &[u8], i: usize, w: &[u8]) -> Result<usize> {
    let e = i + w.len();
    ensure!(b.len() >= e, UnexpectedEof, i);
    ensure!(&b[i..e] == w, SyntaxError, i);
    Ok(e)
}

pub(crate) fn skip_scalar(b: &[u8], i: usize) -> Result<usize> {
    match byte_at(b, i)? {
        b'"' => Ok(scan_string(b, i)?.2),
        b't' => skip_lit(b, i, b"true"),
        b'f' => skip_lit(b, i, b"false"),
        b'n' => skip_lit(b, i, b"null"),
        _ => scan_number(b, i),
    }
}

/// `"key" :` を読み飛ばし、値の開始位置を返す。
pub(crate) fn skip_member_key(b: &[u8], i: usize) -> Result<usize> {
    let i = skip_ws(b, i);
    ensure!(byte_at(b, i)? == b'"', SyntaxError, i);
    let (_, _, ni) = scan_string(b, i)?;
    let ni = skip_ws(b, ni);
    ensure!(byte_at(b, ni)? == b':', SyntaxError, ni);
    Ok(ni + 1)
}

/// `b[i]` から始まる値 1 つを読み飛ばし、その直後の位置を返す。
///
/// 再帰しない。開いているコンテナの種別は `u32` のビットスタックで持ち、
/// 深さは `MAX_DEPTH` で頭打ちにする（`u32` のビット数と一致させてある）。
pub(crate) fn skip_value(b: &[u8], start: usize) -> Result<usize> {
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

pub(crate) fn hex4(b: &[u8], i: usize) -> Result<u32> {
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

/// 文字列本体のエスケープを展開して UTF-8 として `out` に書く。
pub(crate) fn decode_string(body: &[u8], out: &mut Vec<u8>) -> Result<()> {
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

// =========================================================================
// 全体検証（CAST 用）
// =========================================================================

/// 先頭・末尾の空白を許して、ちょうど 1 個の JSON 値からなることを検証し、
/// そのスパンと種別を返す。パス無しの `json_type`/`json_array_length` や
/// `VARCHAR → JSON` の CAST 検証で使う（**ドキュメント全体**を検証する
/// 経路。パス経由のナビゲーションより厳密）。
pub(crate) fn whole(doc: &[u8]) -> Result<(&[u8], Kind)> {
    let start = skip_ws(doc, 0);
    let end = skip_value(doc, start)?;
    ensure!(skip_ws(doc, end) == doc.len(), SyntaxError, end);
    let kind = kind_of(byte_at(doc, start)?);
    Ok((&doc[start..end], kind))
}

/// `doc` が妥当な JSON テキスト 1 個かどうかだけを見る。
pub(crate) fn validate(doc: &[u8]) -> Result<()> {
    whole(doc)?;
    Ok(())
}

// =========================================================================
// パス
// =========================================================================

enum Seg {
    Key(Vec<u8>),
    Index(i64),
}

/// パス文字列をセグメント列へ。構文はモジュール doc の「パス構文」参照。
fn parse_path(p: &[u8]) -> Result<Vec<Seg>> {
    let mut i = if p.first() == Some(&b'$') { 1 } else { 0 };
    let mut segs = Vec::new();
    while i < p.len() {
        match p[i] {
            b'.' => {
                i += 1;
                if p.get(i) == Some(&b'"') {
                    let (body, esc, ni) = scan_string(p, i)?;
                    let key = if esc {
                        let mut buf = Vec::new();
                        decode_string(body, &mut buf)?;
                        buf
                    } else {
                        body.to_vec()
                    };
                    segs.push(Seg::Key(key));
                    i = ni;
                } else {
                    let start = i;
                    while i < p.len() && !matches!(p[i], b'.' | b'[') {
                        i += 1;
                    }
                    ensure!(i > start, SyntaxError, start);
                    segs.push(Seg::Key(p[start..i].to_vec()));
                }
            }
            b'[' => {
                i += 1;
                let neg = p.get(i) == Some(&b'-');
                let dstart = if neg { i + 1 } else { i };
                let mut j = dstart;
                while j < p.len() && p[j].is_ascii_digit() {
                    j += 1;
                }
                ensure!(j > dstart, SyntaxError, i);
                ensure!(p.get(j) == Some(&b']'), SyntaxError, j);
                let mut n: i64 = 0;
                for &d in &p[dstart..j] {
                    n = n.saturating_mul(10).saturating_add((d - b'0') as i64);
                }
                if neg {
                    n = -n;
                }
                segs.push(Seg::Index(n));
                i = j + 1;
            }
            _ => {
                // 先頭セグメントが `.`/`[` 無しで始まる簡略化パス（`a.b[0]`）。
                let start = i;
                while i < p.len() && !matches!(p[i], b'.' | b'[') {
                    i += 1;
                }
                ensure!(i > start, SyntaxError, start);
                segs.push(Seg::Key(p[start..i].to_vec()));
            }
        }
    }
    Ok(segs)
}

/// オブジェクト `b[obj_start]`（`{`）の中から `want` に一致するメンバの
/// 値の開始位置を探す。見つからなければ `None`。
///
/// 目的のキーの手前にあるメンバは、読み飛ばすために構造を検証する
/// （壊れていれば `Err`）。手前に無ければ検証されない（モジュール doc の
/// 「既知の制限」参照）。
fn find_member(b: &[u8], obj_start: usize, want: &[u8]) -> Result<Option<usize>> {
    let mut i = skip_ws(b, obj_start + 1);
    if byte_at(b, i)? == b'}' {
        return Ok(None);
    }
    let mut scratch = Vec::new();
    loop {
        ensure!(byte_at(b, i)? == b'"', SyntaxError, i);
        let (key, esc, ni) = scan_string(b, i)?;
        i = skip_ws(b, ni);
        ensure!(byte_at(b, i)? == b':', SyntaxError, i);
        let vs = skip_ws(b, i + 1);
        let matched = if esc {
            scratch.clear();
            decode_string(key, &mut scratch)?;
            scratch == want
        } else {
            key == want
        };
        if matched {
            return Ok(Some(vs));
        }
        let ve = skip_value(b, vs)?;
        i = skip_ws(b, ve);
        match byte_at(b, i)? {
            b',' => i = skip_ws(b, i + 1),
            b'}' => return Ok(None),
            _ => err!(SyntaxError, i),
        }
    }
}

/// 配列 `b[arr_start]`（`[`）の要素数。
fn count_elements(b: &[u8], arr_start: usize) -> Result<i64> {
    let mut i = skip_ws(b, arr_start + 1);
    if byte_at(b, i)? == b']' {
        return Ok(0);
    }
    let mut n = 0i64;
    loop {
        let ve = skip_value(b, i)?;
        n += 1;
        i = skip_ws(b, ve);
        match byte_at(b, i)? {
            b',' => i = skip_ws(b, i + 1),
            b']' => return Ok(n),
            _ => err!(SyntaxError, i),
        }
    }
}

/// 配列 `b[arr_start]`（`[`）の `want` 番目（0 始まり、負数は末尾から）の
/// 要素の開始位置。範囲外は `None`。
fn nth_element(b: &[u8], arr_start: usize, want: i64) -> Result<Option<usize>> {
    let count = count_elements(b, arr_start)?;
    let real = if want < 0 { count + want } else { want };
    if real < 0 || real >= count {
        return Ok(None);
    }
    let mut i = skip_ws(b, arr_start + 1);
    let mut idx = 0i64;
    loop {
        if idx == real {
            return Ok(Some(i));
        }
        let ve = skip_value(b, i)?;
        idx += 1;
        i = skip_ws(b, ve);
        match byte_at(b, i)? {
            b',' => i = skip_ws(b, i + 1),
            b']' => return Ok(None),
            _ => err!(SyntaxError, i),
        }
    }
}

/// `json_extract`/`json_extract_string`/`json_type`/`json_array_length` の
/// パス付き形の本体。`path` が空（または `$` のみ）ならドキュメント全体を
/// 返す。セグメントの種別が実際のコンテナと合わない、またはキー/添字が
/// 見つからない場合は `Ok(None)`（呼び出し側は SQL NULL にする）。
pub(crate) fn extract<'a>(doc: &'a [u8], path: &[u8]) -> Result<Option<(&'a [u8], Kind)>> {
    let segs = parse_path(path)?;
    if segs.is_empty() {
        let (span, kind) = whole(doc)?;
        return Ok(Some((span, kind)));
    }
    let mut vs = skip_ws(doc, 0);
    for seg in &segs {
        let b0 = byte_at(doc, vs)?;
        match seg {
            Seg::Key(k) => {
                if b0 != b'{' {
                    return Ok(None);
                }
                match find_member(doc, vs, k)? {
                    Some(s) => vs = s,
                    None => return Ok(None),
                }
            }
            Seg::Index(idx) => {
                if b0 != b'[' {
                    return Ok(None);
                }
                match nth_element(doc, vs, *idx)? {
                    Some(s) => vs = s,
                    None => return Ok(None),
                }
            }
        }
    }
    let end = skip_value(doc, vs)?;
    let kind = kind_of(byte_at(doc, vs)?);
    Ok(Some((&doc[vs..end], kind)))
}

/// `list_extract` の本体。1 始まり、負数は末尾から。範囲外・非配列は
/// `None`。
pub(crate) fn list_index(doc: &[u8], idx: i64) -> Result<Option<&[u8]>> {
    let start = skip_ws(doc, 0);
    if idx == 0 || byte_at(doc, start)? != b'[' {
        return Ok(None);
    }
    let zero_based = if idx > 0 { idx - 1 } else { idx };
    match nth_element(doc, start, zero_based)? {
        Some(vs) => {
            let ve = skip_value(doc, vs)?;
            Ok(Some(&doc[vs..ve]))
        }
        None => Ok(None),
    }
}

/// `list_slice` の本体。1 始まり、両端含む、負数は末尾から。`list_index`
/// （`list_extract`）と違って範囲外は **NULL にせず切り詰める**（DuckDB の
/// `expr[i:j]` 構文と同じ規則。`duckdb -c "select [1,2,3,4,5][10:20],
/// [1,2,3,4,5][-10:3]"` が `[]`/`[1, 2, 3]` を返すことで確認済み ——
/// 完全に範囲外なら空配列、一部だけはみ出すなら残った部分だけ返す）。
/// 非配列は `None`（呼び出し元で SQL NULL）。
///
/// 戻り値は `doc` 内の半開区間 `(lo, hi)`。`doc[lo..hi]` を `[` `]` で
/// 包めば結果になる（`lo == hi` は空配列 `[]` を意味する）。配列要素は
/// 連続したバイト列なので、要素ごとに書き出し直さずこの区間をまるごと
/// コピーするだけで済む（`nth_element`/`skip_value` を先頭要素・末尾要素の
/// 位置決めにだけ使う）。
pub(crate) fn list_slice(doc: &[u8], start: i64, end: i64) -> Result<Option<(usize, usize)>> {
    let arr_start = skip_ws(doc, 0);
    if byte_at(doc, arr_start)? != b'[' {
        return Ok(None);
    }
    let count = count_elements(doc, arr_start)?;
    // 空配列の書き出し先（`[` の直後、`]` の手前の空白終わり）。
    let empty_at = skip_ws(doc, arr_start + 1);
    if count == 0 {
        return Ok(Some((empty_at, empty_at)));
    }
    // 1 始まり・両端含む区間へ正規化してからクランプする。0 は 1 として
    // 扱う（`duckdb -c "select [1,2,3,4,5][0:2]"` が `[1,2,3,4,5][1:2]` と
    // 同じ `[1, 2]` を返すことで確認済み）。負数は `saturating_add` で
    // オーバーフローを避ける（信用できない入力からの巨大な負の添字対策）。
    let norm = |v: i64| -> i64 {
        if v == 0 {
            1
        } else if v < 0 {
            count.saturating_add(v).saturating_add(1)
        } else {
            v
        }
    };
    let s = norm(start).max(1);
    let e = norm(end).min(count);
    if s > e {
        return Ok(Some((empty_at, empty_at)));
    }
    // クランプ後の s/e は必ず [1, count] 内に収まるので、`nth_element` は
    // 常に `Some` を返す（0 始まりへ変換してから渡す）。
    let (Some(lo), Some(hi_start)) =
        (nth_element(doc, arr_start, s - 1)?, nth_element(doc, arr_start, e - 1)?)
    else {
        err!(Internal)
    };
    let hi = skip_value(doc, hi_start)?;
    Ok(Some((lo, hi)))
}

/// `map_extract` の本体。直接のメンバ名検索（パス構文は使わない）。
/// 非オブジェクト・キー無しは `None`。
pub(crate) fn map_get<'a>(doc: &'a [u8], key: &[u8]) -> Result<Option<&'a [u8]>> {
    let start = skip_ws(doc, 0);
    if byte_at(doc, start)? != b'{' {
        return Ok(None);
    }
    match find_member(doc, start, key)? {
        Some(vs) => {
            let ve = skip_value(doc, vs)?;
            Ok(Some(&doc[vs..ve]))
        }
        None => Ok(None),
    }
}

/// `json_array_length` の本体。配列でなければ 0（DuckDB と同じ）。
pub(crate) fn array_length(span: &[u8], kind: Kind) -> Result<i64> {
    if kind != Kind::Array {
        return Ok(0);
    }
    let start = skip_ws(span, 0);
    count_elements(span, start)
}

/// 配列要素 1 個のスパンと種別。
pub(crate) type Elem<'a> = (&'a [u8], Kind);

/// `doc`（前後の空白は許容）が配列なら、その要素を先頭から順に返す。
/// 配列でなければ `None`（`exec::unnest::Unnest` はこれを「展開結果 0 行」
/// として扱う。`nth_element` の走査を要素数ぶん繰り返す（O(n^2)）のを避け、
/// 1 パスで全要素のスパンを集める）。
pub(crate) fn array_elements(doc: &[u8]) -> Result<Option<Vec<Elem<'_>>>> {
    let start = skip_ws(doc, 0);
    if byte_at(doc, start)? != b'[' {
        return Ok(None);
    }
    let mut out = Vec::new();
    let mut i = skip_ws(doc, start + 1);
    if byte_at(doc, i)? == b']' {
        return Ok(Some(out));
    }
    loop {
        let kind = kind_of(byte_at(doc, i)?);
        let ve = skip_value(doc, i)?;
        out.push((&doc[i..ve], kind));
        i = skip_ws(doc, ve);
        match byte_at(doc, i)? {
            b',' => i = skip_ws(doc, i + 1),
            b']' => return Ok(Some(out)),
            _ => err!(SyntaxError, i),
        }
    }
}

/// JSON 数値トークンを `i64` に。小数点・指数付き、または範囲外なら
/// `None`（呼び出し側は SQL NULL にする。`format::jsonl::parse_i64` と同じ
/// 判断だが、こちらは UNNEST のネイティブ型復元専用に `json` 側へ置く）。
pub(crate) fn parse_i64(s: &[u8]) -> Option<i64> {
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

/// 浮動小数は core の `str::parse::<f64>()`（dec2flt）に任せる。
pub(crate) fn parse_f64(s: &[u8]) -> Option<f64> {
    core::str::from_utf8(s).ok()?.parse::<f64>().ok()
}

/// `json_type` の型名。DuckDB の実際の文字列に合わせるが、数値は
/// 符号・桁あふれで UBIGINT/DOUBLE を出し分ける DuckDB の挙動までは
/// 再現せず、整数はすべて `"BIGINT"` に単純化する（モジュール doc 外の
/// 追加の簡略化。詳細は `expr::funcs` 側のテスト・ドキュメントを参照）。
pub(crate) fn type_name(kind: Kind, span: &[u8]) -> &'static str {
    match kind {
        Kind::Null => "NULL",
        Kind::Bool => "BOOLEAN",
        Kind::Str => "VARCHAR",
        Kind::Object => "OBJECT",
        Kind::Array => "ARRAY",
        Kind::Num => {
            if span.iter().any(|&c| c == b'.' || c == b'e' || c == b'E') {
                "DOUBLE"
            } else {
                "BIGINT"
            }
        }
    }
}

/// `json_extract_string` のテキスト化。JSON `null` は SQL NULL（`false`）。
/// 文字列はエスケープを展開し、それ以外はスパンをそのまま書く
/// （数値の正規化はしない: `1e3` はそのまま `1e3`。DuckDB は正規化した
/// 数値表記を返すが、ここでは行わない簡略化）。
pub(crate) fn write_extracted_text(span: &[u8], kind: Kind, out: &mut Vec<u8>) -> Result<bool> {
    match kind {
        Kind::Null => Ok(false),
        Kind::Str => {
            let body = if span.len() >= 2 { &span[1..span.len() - 1] } else { &[][..] };
            decode_string(body, out)?;
            Ok(true)
        }
        _ => {
            out.extend_from_slice(span);
            Ok(true)
        }
    }
}

// =========================================================================
// シリアライズ
// =========================================================================

fn hex_digit(n: u8) -> u8 {
    if n < 10 {
        b'0' + n
    } else {
        b'a' + (n - 10)
    }
}

/// 任意のバイト列（UTF-8 の VARCHAR 内容）を JSON 文字列リテラルとして書く。
/// 制御文字だけエスケープし、非 ASCII の UTF-8 バイトはそのまま埋め込む
/// （DuckDB の既定と同じで `\uXXXX` へは変換しない）。
pub(crate) fn write_json_string(s: &[u8], out: &mut Vec<u8>) {
    out.push(b'"');
    for &c in s {
        match c {
            b'"' => out.extend_from_slice(b"\\\""),
            b'\\' => out.extend_from_slice(b"\\\\"),
            0x08 => out.extend_from_slice(b"\\b"),
            0x0c => out.extend_from_slice(b"\\f"),
            b'\n' => out.extend_from_slice(b"\\n"),
            b'\r' => out.extend_from_slice(b"\\r"),
            b'\t' => out.extend_from_slice(b"\\t"),
            0x00..=0x1f => {
                out.extend_from_slice(b"\\u00");
                out.push(hex_digit(c >> 4));
                out.push(hex_digit(c & 0xf));
            }
            _ => out.push(c),
        }
    }
    out.push(b'"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{code_of, Code};

    fn ext<'a>(doc: &'a str, path: &str) -> Option<(&'a str, Kind)> {
        extract(doc.as_bytes(), path.as_bytes())
            .unwrap()
            .map(|(s, k)| (core::str::from_utf8(s).unwrap(), k))
    }

    #[test]
    fn extract_dollar_path_matches_duckdb() {
        // duckdb: SELECT json_extract('{"a":{"b":[1,2,3]}}', '$.a.b[1]') -> 2
        assert_eq!(ext(r#"{"a":{"b":[1,2,3]}}"#, "$.a.b[1]").unwrap().0, "2");
        assert_eq!(ext(r#"{"a":1}"#, "$.a").unwrap().0, "1");
        assert_eq!(ext(r#"{"a":1}"#, "$").unwrap().0, r#"{"a":1}"#);
        assert_eq!(ext("[1,2,3]", "$[0]").unwrap().0, "1");
        // duckdb: json_extract('[1,2,3]', '$[-1]') -> 3
        assert_eq!(ext("[1,2,3]", "$[-1]").unwrap().0, "3");
    }

    #[test]
    fn extract_bare_path_is_our_simplification() {
        // DuckDB では 'a.b[1]' は NULL になるが、ここでは $ 省略を許す。
        assert_eq!(ext(r#"{"a":{"b":[1,2,3]}}"#, "a.b[1]").unwrap().0, "2");
    }

    #[test]
    fn extract_quoted_key() {
        assert_eq!(ext(r#"{"a b":1}"#, r#"$."a b""#).unwrap().0, "1");
    }

    #[test]
    fn extract_missing_path_is_none() {
        assert!(ext(r#"{"a":1}"#, "$.b").is_none());
        assert!(ext("[1,2,3]", "$[10]").is_none());
        assert!(ext(r#"{"a":1}"#, "$.a.b").is_none());
        assert!(ext("[1,2,3]", "$.a").is_none());
    }

    #[test]
    fn extract_null_value_is_a_real_json_null_not_missing() {
        let (span, kind) = ext(r#"{"a":null}"#, "$.a").unwrap();
        assert_eq!(span, "null");
        assert!(kind == Kind::Null);
    }

    #[test]
    fn malformed_json_is_an_error() {
        assert_eq!(code_of(extract(b"{not json", b"$.a")), Some(Code::SyntaxError));
        assert_eq!(code_of(whole(b"{not json")), Some(Code::SyntaxError));
    }

    #[test]
    fn whole_rejects_trailing_garbage() {
        assert_eq!(code_of(whole(b"{}{}")), Some(Code::SyntaxError));
        assert!(whole(b"  {\"a\":1}  ").is_ok());
    }

    #[test]
    fn list_index_is_one_based_with_negative_from_end() {
        assert_eq!(list_index(b"[10,20,30]", 1).unwrap(), Some(&b"10"[..]));
        assert_eq!(list_index(b"[10,20,30]", -1).unwrap(), Some(&b"30"[..]));
        assert_eq!(list_index(b"[10,20,30]", 0).unwrap(), None);
        assert_eq!(list_index(b"[10,20,30]", 100).unwrap(), None);
        assert_eq!(list_index(b"{\"a\":1}", 1).unwrap(), None);
    }

    fn slice(doc: &str, start: i64, end: i64) -> Option<String> {
        list_slice(doc.as_bytes(), start, end).unwrap().map(|(lo, hi)| {
            let mut out = String::from("[");
            out.push_str(&doc[lo..hi]);
            out.push(']');
            out
        })
    }

    #[test]
    fn list_slice_clamps_out_of_range_instead_of_nulling() {
        // duckdb -c "select [1,2,3,4,5][2:3]" -> [2, 3]
        assert_eq!(slice("[1,2,3,4,5]", 2, 3).as_deref(), Some("[2,3]"));
        // duckdb -c "select [1,2,3,4,5][-2:-1]" -> [4, 5]
        assert_eq!(slice("[1,2,3,4,5]", -2, -1).as_deref(), Some("[4,5]"));
        // duckdb -c "select [1,2,3,4,5][10:20]" -> []  (fully out of range: empty, not NULL)
        assert_eq!(slice("[1,2,3,4,5]", 10, 20).as_deref(), Some("[]"));
        // duckdb -c "select [1,2,3,4,5][-10:3]" -> [1, 2, 3] (start clamps up to 1)
        assert_eq!(slice("[1,2,3,4,5]", -10, 3).as_deref(), Some("[1,2,3]"));
        // duckdb -c "select [1,2,3,4,5][3:1]" -> [] (start past end)
        assert_eq!(slice("[1,2,3,4,5]", 3, 1).as_deref(), Some("[]"));
        // duckdb -c "select [1,2,3,4,5][0:2]" -> [1, 2] (0 behaves like 1)
        assert_eq!(slice("[1,2,3,4,5]", 0, 2).as_deref(), Some("[1,2]"));
        // duckdb -c "select [][1:2]" -> []
        assert_eq!(slice("[]", 1, 2).as_deref(), Some("[]"));
        // Non-array base -> None (caller turns this into SQL NULL).
        assert_eq!(list_slice(b"{\"a\":1}", 1, 2).unwrap(), None);
        // Omitted-bound sentinels used by the parser desugaring.
        // duckdb -c "select [1,2,3,4,5][:3], [1,2,3,4,5][2:]" -> [1,2,3] / [2,3,4,5]
        assert_eq!(slice("[1,2,3,4,5]", 1, 3).as_deref(), Some("[1,2,3]"));
        assert_eq!(slice("[1,2,3,4,5]", 2, i64::MAX).as_deref(), Some("[2,3,4,5]"));
    }

    #[test]
    fn map_get_looks_up_a_direct_member() {
        assert_eq!(map_get(br#"{"a":1,"b":2}"#, b"a").unwrap(), Some(&b"1"[..]));
        assert_eq!(map_get(br#"{"a":1}"#, b"z").unwrap(), None);
        assert_eq!(map_get(b"[1,2]", b"a").unwrap(), None);
    }

    #[test]
    fn array_length_is_zero_for_non_arrays() {
        assert_eq!(array_length(b"[1,2,3]", Kind::Array).unwrap(), 3);
        assert_eq!(array_length(b"[]", Kind::Array).unwrap(), 0);
        assert_eq!(array_length(b"5", Kind::Num).unwrap(), 0);
        assert_eq!(array_length(br#"{"a":1}"#, Kind::Object).unwrap(), 0);
    }

    #[test]
    fn type_name_matches_duckdb_strings() {
        assert_eq!(type_name(Kind::Object, b"{}"), "OBJECT");
        assert_eq!(type_name(Kind::Array, b"[]"), "ARRAY");
        assert_eq!(type_name(Kind::Str, b"\"x\""), "VARCHAR");
        assert_eq!(type_name(Kind::Bool, b"true"), "BOOLEAN");
        assert_eq!(type_name(Kind::Null, b"null"), "NULL");
        assert_eq!(type_name(Kind::Num, b"1"), "BIGINT");
        assert_eq!(type_name(Kind::Num, b"-1"), "BIGINT");
        assert_eq!(type_name(Kind::Num, b"1.5"), "DOUBLE");
        assert_eq!(type_name(Kind::Num, b"1e3"), "DOUBLE");
    }

    #[test]
    fn write_extracted_text_unquotes_strings_and_nulls_json_null() {
        let mut out = Vec::new();
        assert!(write_extracted_text(b"\"hi\"", Kind::Str, &mut out).unwrap());
        assert_eq!(out, b"hi");
        out.clear();
        assert!(!write_extracted_text(b"null", Kind::Null, &mut out).unwrap());
        out.clear();
        assert!(write_extracted_text(b"1.5", Kind::Num, &mut out).unwrap());
        assert_eq!(out, b"1.5");
    }

    #[test]
    fn write_json_string_escapes_control_chars_and_quotes() {
        let mut out = Vec::new();
        write_json_string(b"a\"b\\c\nd", &mut out);
        assert_eq!(out, b"\"a\\\"b\\\\c\\nd\"");
    }

    #[test]
    fn array_elements_splits_top_level_values_only() {
        let got = array_elements(b"[1,[2,3],\"a,b\",null]").unwrap().unwrap();
        assert_eq!(got.len(), 4);
        assert_eq!(got[0], (&b"1"[..], Kind::Num));
        assert_eq!(got[1], (&b"[2,3]"[..], Kind::Array));
        assert_eq!(got[2], (&b"\"a,b\""[..], Kind::Str));
        assert_eq!(got[3], (&b"null"[..], Kind::Null));
    }

    #[test]
    fn array_elements_empty_array_is_some_empty_vec() {
        assert_eq!(array_elements(b"[]").unwrap(), Some(Vec::new()));
        assert_eq!(array_elements(b"  [ ]  ").unwrap(), Some(Vec::new()));
    }

    #[test]
    fn array_elements_non_array_is_none() {
        assert_eq!(array_elements(b"5").unwrap(), None);
        assert_eq!(array_elements(br#"{"a":1}"#).unwrap(), None);
        assert_eq!(array_elements(b"null").unwrap(), None);
    }

    #[test]
    fn parse_i64_rejects_non_integer_tokens() {
        assert_eq!(parse_i64(b"42"), Some(42));
        assert_eq!(parse_i64(b"-42"), Some(-42));
        assert_eq!(parse_i64(b"4.2"), None);
        assert_eq!(parse_i64(b""), None);
    }

    #[test]
    fn parse_f64_parses_plain_and_exponent_forms() {
        assert_eq!(parse_f64(b"1.5"), Some(1.5));
        assert_eq!(parse_f64(b"1e3"), Some(1000.0));
        assert_eq!(parse_f64(b"nope"), None);
    }

    // --- 境界値・壊れた入力 ----------------------------------------------------

    #[test]
    fn nesting_exactly_at_max_depth_succeeds_one_more_fails() {
        // `skip_value` は非再帰（`u32` ビットスタック）なので深いネストで
        // スタックオーバーフローはしないが、MAX_DEPTH（32）自体は明示的な
        // 上限として効くはず。
        let mut at_limit = vec![b'['; MAX_DEPTH as usize];
        at_limit.extend(vec![b']'; MAX_DEPTH as usize]);
        assert!(whole(&at_limit).is_ok(), "MAX_DEPTH ちょうどは通る");

        let mut over_limit = vec![b'['; MAX_DEPTH as usize + 1];
        over_limit.extend(vec![b']'; MAX_DEPTH as usize + 1]);
        assert_eq!(code_of(whole(&over_limit)), Some(Code::NestingTooDeep));
    }

    #[test]
    fn very_deeply_nested_input_errors_without_panicking_or_hanging() {
        // MAX_DEPTH の何倍も深い入力でも（非再帰実装なので）panic せず、
        // ただの SyntaxError/NestingTooDeep で終わることを確認する。
        let deep: Vec<u8> = vec![b'['; 10_000];
        assert!(whole(&deep).is_err());
    }

    #[test]
    fn trailing_comma_is_rejected() {
        assert_eq!(code_of(whole(b"[1,2,]")), Some(Code::SyntaxError));
        assert_eq!(code_of(whole(br#"{"a":1,}"#)), Some(Code::SyntaxError));
    }

    #[test]
    fn missing_closing_bracket_is_unexpected_eof_not_a_panic() {
        assert!(whole(b"[1,2,3").is_err());
        assert!(whole(br#"{"a":1"#).is_err());
        assert!(whole(br#"{"a""#).is_err());
    }

    #[test]
    fn decode_string_reassembles_surrogate_pairs_into_one_codepoint() {
        // U+1F600 (😀) は UTF-16 で高位・低位サロゲートのペアになる。
        let mut out = Vec::new();
        decode_string(b"\\ud83d\\ude00", &mut out).unwrap();
        assert_eq!(core::str::from_utf8(&out).unwrap(), "\u{1F600}");
    }

    #[test]
    fn decode_string_replaces_unpaired_surrogates_with_replacement_char() {
        // 対にならない上位/下位サロゲートはエラーにせず U+FFFD に潰す
        // （壊れた入力で他のフィールドの読み取りごと失いたくないため）。
        let mut out = Vec::new();
        decode_string(b"\\ud83d", &mut out).unwrap();
        assert_eq!(core::str::from_utf8(&out).unwrap(), "\u{FFFD}");
        out.clear();
        decode_string(b"\\udc00", &mut out).unwrap();
        assert_eq!(core::str::from_utf8(&out).unwrap(), "\u{FFFD}");
    }

    #[test]
    fn raw_control_bytes_in_a_string_are_accepted_without_escaping() {
        // RFC 8259 は文字列中の生の制御文字（0x00-0x1F）を許さないが、
        // `scan_string` は `"` と `\` の位置しか見ておらず、これを検査しない。
        // 誤って壊すよりゆるく通す方を選んだ既知の簡略化で、意図的な挙動
        // として固定しておく（誤って構文エラーになるよう「厳格化」しても
        // 気づけるように）。
        assert!(whole(b"\"a\x01b\"").is_ok());
    }
}
