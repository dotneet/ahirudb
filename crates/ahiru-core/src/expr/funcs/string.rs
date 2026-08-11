//! 文字列（Bytes 出力）
use super::datetime::strftime;
use super::json::{json_extract_or_whole, write_json_scalar};
use super::numeric::{pow10, round_half_up};
use super::*;

/// UTF-8 のコードポイント数。継続バイトを数えないだけ。
pub(super) fn cp_count(s: &[u8]) -> usize {
    s.iter().filter(|&&b| (b & 0xc0) != 0x80).count()
}

/// 先頭から `k` 個目のコードポイント境界のバイト位置。範囲外は `s.len()`。
fn cp_byte(s: &[u8], k: usize) -> usize {
    let mut seen = 0usize;
    for (i, &b) in s.iter().enumerate() {
        if (b & 0xc0) != 0x80 {
            if seen == k {
                return i;
            }
            seen += 1;
        }
    }
    s.len()
}

/// `i` 位置から始まるコードポイントのバイト数。
fn cp_width(s: &[u8], i: usize) -> usize {
    let mut w = 1;
    while i + w < s.len() && (s[i + w] & 0xc0) == 0x80 {
        w += 1;
    }
    w
}

/// `hi` で終わるコードポイントの開始位置。
fn cp_back(s: &[u8], hi: usize) -> usize {
    let mut lo = hi - 1;
    while lo > 0 && (s[lo] & 0xc0) == 0x80 {
        lo -= 1;
    }
    lo
}

/// `set` に含まれるコードポイントか（`trim` の文字集合）。
fn in_set(set: &[u8], c: &[u8]) -> bool {
    let mut i = 0;
    while i < set.len() {
        let w = cp_width(set, i);
        if &set[i..i + w] == c {
            return true;
        }
        i += w;
    }
    false
}

/// `hay` の中の `needle` の先頭バイト位置。無ければ `None`。空針は 0。
pub(super) fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    if needle.len() > hay.len() {
        return None;
    }
    (0..=hay.len() - needle.len()).find(|&i| &hay[i..i + needle.len()] == needle)
}

/// 1 行ぶんの文字列関数。`false` を返した行は NULL。
pub(super) fn eval_str(id: FuncId, a: &A, out: &mut Vec<u8>) -> Result<bool> {
    match id {
        F_UPPER | F_LOWER => {
            // ASCII のみ。Unicode の大小文字テーブルは 1MiB 予算に入らない。
            let s = a.bytes(0);
            out.extend_from_slice(s);
            for c in out.iter_mut() {
                *c = if id == F_UPPER { c.to_ascii_uppercase() } else { c.to_ascii_lowercase() };
            }
        }
        F_TRIM | F_LTRIM | F_RTRIM => {
            let s = a.bytes(0);
            // 1 引数版が落とすのは**空白のみ**（DuckDB と同じ。タブは残る）。
            let set: &[u8] = if a.n() >= 2 { a.bytes(1) } else { b" " };
            let (mut lo, mut hi) = (0usize, s.len());
            if id != F_RTRIM {
                while lo < hi {
                    let w = cp_width(s, lo);
                    if !in_set(set, &s[lo..lo + w]) {
                        break;
                    }
                    lo += w;
                }
            }
            if id != F_LTRIM {
                while hi > lo {
                    let st = cp_back(s, hi);
                    if !in_set(set, &s[st..hi]) {
                        break;
                    }
                    hi = st;
                }
            }
            out.extend_from_slice(&s[lo..hi]);
        }
        F_SUBSTR => {
            let s = a.bytes(0);
            let mut start = a.int(1);
            // 負の開始位置は末尾から数える（DuckDB）。
            if start < 0 {
                start += cp_count(s) as i64 + 1;
            }
            let (mut from, mut to) = if a.n() >= 3 {
                let e = start.saturating_add(a.int(2));
                // 負の長さは「開始位置から手前へ」（DuckDB は範囲を反転する）。
                if a.int(2) < 0 {
                    (e, start)
                } else {
                    (start, e)
                }
            } else {
                (start, i64::MAX)
            };
            from = from.max(1);
            to = to.max(from);
            let cap = s.len() as i64 + 1;
            let b0 = cp_byte(s, (from - 1).min(cap) as usize);
            let b1 = cp_byte(s, (to - 1).min(cap) as usize);
            out.extend_from_slice(&s[b0..b1.max(b0)]);
        }
        F_REPLACE => {
            let (s, from, to) = (a.bytes(0), a.bytes(1), a.bytes(2));
            if from.is_empty() {
                out.extend_from_slice(s);
            } else {
                let mut i = 0;
                while i < s.len() {
                    if s.len() - i >= from.len() && &s[i..i + from.len()] == from {
                        out.extend_from_slice(to);
                        i += from.len();
                    } else {
                        out.push(s[i]);
                        i += 1;
                    }
                }
            }
        }
        F_LPAD | F_RPAD => {
            let s = a.bytes(0);
            let want = a.int(1);
            ensure!(want <= MAX_STR, LimitExceeded);
            let cl = cp_count(s) as i64;
            if want <= cl {
                // 目標長より長ければ切り詰める（DuckDB）。
                let cut = cp_byte(s, want.max(0) as usize);
                out.extend_from_slice(&s[..cut]);
                return Ok(true);
            }
            let pad = a.bytes(2);
            let mut fill = Vec::new();
            if !pad.is_empty() {
                // 埋め草は繰り返し、最後は途中で切る。
                let pl = cp_count(pad);
                let mut need = (want - cl) as usize;
                while need > 0 {
                    let take = need.min(pl);
                    fill.extend_from_slice(&pad[..cp_byte(pad, take)]);
                    need -= take;
                }
            }
            if id == F_LPAD {
                out.extend_from_slice(&fill);
                out.extend_from_slice(s);
            } else {
                out.extend_from_slice(s);
                out.extend_from_slice(&fill);
            }
        }
        F_REVERSE => {
            let s = a.bytes(0);
            let mut hi = s.len();
            while hi > 0 {
                let lo = cp_back(s, hi);
                out.extend_from_slice(&s[lo..hi]);
                hi = lo;
            }
        }
        F_SPLIT_PART => {
            let (s, d, k) = (a.bytes(0), a.bytes(1), a.int(2));
            // 添字は 1 始まり。負なら末尾から。0 と範囲外は空文字列。
            if k == 0 {
                return Ok(true);
            }
            if d.is_empty() {
                if k == 1 || k == -1 {
                    out.extend_from_slice(s);
                }
                return Ok(true);
            }
            let mut cnt = 1i64;
            let mut i = 0usize;
            while i + d.len() <= s.len() {
                if &s[i..i + d.len()] == d {
                    cnt += 1;
                    i += d.len();
                } else {
                    i += 1;
                }
            }
            let idx = if k < 0 { cnt + k } else { k - 1 };
            if idx < 0 || idx >= cnt {
                return Ok(true);
            }
            let (mut cur, mut st) = (0i64, 0usize);
            i = 0;
            while i + d.len() <= s.len() {
                if &s[i..i + d.len()] == d {
                    if cur == idx {
                        out.extend_from_slice(&s[st..i]);
                        return Ok(true);
                    }
                    cur += 1;
                    i += d.len();
                    st = i;
                } else {
                    i += 1;
                }
            }
            out.extend_from_slice(&s[st..]);
        }
        F_REPEAT => {
            let s = a.bytes(0);
            let k = a.int(1);
            ensure!(k.saturating_mul(s.len() as i64) <= MAX_STR, LimitExceeded);
            for _ in 0..k.max(0) {
                out.extend_from_slice(s);
            }
        }
        F_STRFTIME => strftime(a.int(0), a.bytes(1), out),
        F_JSON_EXTRACT => {
            return match crate::json::extract(a.bytes(0), a.bytes(1))? {
                Some((span, _)) => {
                    out.extend_from_slice(span);
                    Ok(true)
                }
                None => Ok(false),
            };
        }
        F_JSON_EXTRACT_STRING => {
            return match crate::json::extract(a.bytes(0), a.bytes(1))? {
                Some((span, kind)) => crate::json::write_extracted_text(span, kind, out),
                None => Ok(false),
            };
        }
        F_JSON_TYPE => {
            let found = json_extract_or_whole(a)?;
            return match found {
                Some((span, kind)) => {
                    out.extend_from_slice(crate::json::type_name(kind, span).as_bytes());
                    Ok(true)
                }
                None => Ok(false),
            };
        }
        F_TO_JSON => {
            if let Some((v, j)) = a.at(0) {
                write_json_scalar(v, j, out);
            }
        }
        F_LIST_EXTRACT => {
            return match crate::json::list_index(a.bytes(0), a.int(1))? {
                Some(span) => {
                    out.extend_from_slice(span);
                    Ok(true)
                }
                None => Ok(false),
            };
        }
        F_LIST_SLICE => {
            let doc = a.bytes(0);
            return match crate::json::list_slice(doc, a.int(1), a.int(2))? {
                Some((lo, hi)) => {
                    out.push(b'[');
                    out.extend_from_slice(&doc[lo..hi]);
                    out.push(b']');
                    Ok(true)
                }
                None => Ok(false),
            };
        }
        F_MAP_EXTRACT => {
            return match crate::json::map_get(a.bytes(0), a.bytes(1))? {
                Some(span) => {
                    out.extend_from_slice(span);
                    Ok(true)
                }
                None => Ok(false),
            };
        }
        // NULL 伝播は呼び出し元の `call()`（`live(i)` 判定）に任せる既定どおり
        // でよいので、`json_array`/`concat` と違い `call()` 直下へバイパス
        // する必要が無い（可変長引数でも `A` は `args: &[&Vector]` 全体を
        // 持っているので、`eval_str` の中だけで完結する）。対応する書式指定子
        // は `printf_scan`/`format_scan` のコメント参照。
        F_PRINTF => printf_scan(a.bytes(0), a, out)?,
        F_FORMAT => format_scan(a.bytes(0), a, out)?,
        _ => err!(Internal),
    }
    Ok(true)
}

/// `printf` の最大精度（`%.<N>f` の `N`）。`kernels::fmt_int` の内部バッファ
/// は 48 バイトしかないので、これを大きく超える scale を渡すとバッファ
/// 添字が範囲外になる（=panic）。32 は安全側に十分な余裕を残した値。
const MAX_PRINTF_PREC: u32 = 32;

/// `printf`/`format` の幅指定の上限。悪意あるクエリが `%<巨大な数字>d` の
/// ような幅を書いてメモリ・時間を食い潰さないための上限（`repeat`/`lpad`
/// の `MAX_STR` と同じ動機）。
const MAX_PRINTF_WIDTH: usize = 1 << 16;

/// `printf(fmt, args...)`。C 由来の `%` 書式のうち、次だけに対応する:
///
/// - `%%`  リテラルの `%`
/// - `%[-][0][<幅>]d`  整数。対応する実引数は BOOLEAN（0/1）・整数型。
///   浮動小数は 0 方向へ切り捨てる。
/// - `%[-][0][<幅>][.<精度>]f`  浮動小数点数の固定小数点表記。精度既定 6 桁
///   （C の `printf` と同じ。`duckdb -c "select printf('%f', 3.5)"` が
///   `3.500000` になることを確認済み）、`MAX_PRINTF_PREC` まで指定できる。
/// - `%[-][<幅>]s`  `write_display` と同じ規則で文字列化する。
///
/// DuckDB との既知の違い: DuckDB は `%d` に FLOAT を、`%s` に INTEGER を
/// 渡すとエラーにする（`fmt` ライブラリの型厳格性）が、ここでは実用上の
/// 都合を優先してどちらも受け付ける（`%s` はどんな対応物理型でも文字列化
/// する）。`%x`/`%o` などの基数変換、`*` による幅の実引数指定、`%1$d`
/// 形式の引数番号付けは非対応（`UnsupportedFeature`）。指定子の数より
/// 実引数が少なければ `WrongArgCount`（DuckDB と同じ。余った実引数は無視
/// してよい: `duckdb -c "select printf('%d', 1, 2)"` は `1`）。
fn printf_scan(fmt: &[u8], a: &A, out: &mut Vec<u8>) -> Result<()> {
    let mut ai = 1usize; // args[0] は fmt 自身
    let mut i = 0usize;
    while i < fmt.len() {
        if fmt[i] != b'%' {
            out.push(fmt[i]);
            i += 1;
            continue;
        }
        i += 1;
        ensure!(i < fmt.len(), SyntaxError);
        if fmt[i] == b'%' {
            out.push(b'%');
            i += 1;
            continue;
        }
        let (mut left, mut zero) = (false, false);
        loop {
            match fmt.get(i) {
                Some(b'-') => {
                    left = true;
                    i += 1;
                }
                Some(b'0') if !zero => {
                    zero = true;
                    i += 1;
                }
                _ => break,
            }
        }
        let mut width = 0usize;
        while let Some(&b) = fmt.get(i) {
            if !b.is_ascii_digit() {
                break;
            }
            width = (width.saturating_mul(10) + (b - b'0') as usize).min(MAX_PRINTF_WIDTH);
            i += 1;
        }
        let mut prec: u32 = 6;
        if fmt.get(i) == Some(&b'.') {
            i += 1;
            prec = 0;
            while let Some(&b) = fmt.get(i) {
                if !b.is_ascii_digit() {
                    break;
                }
                prec = (prec.saturating_mul(10) + (b - b'0') as u32).min(MAX_PRINTF_PREC);
                i += 1;
            }
        }
        ensure!(i < fmt.len(), SyntaxError);
        let conv = fmt[i];
        i += 1;
        // 未対応の変換文字は、実引数が足りているかより先にチェックする
        // （`%x` はいくら実引数を積んでも `UnsupportedFeature` であるべき）。
        ensure!(matches!(conv, b'd' | b'f' | b's'), UnsupportedFeature);
        ensure!(ai < a.n(), WrongArgCount);
        let (v, row) = match a.at(ai) {
            Some(x) => x,
            None => err!(Internal),
        };
        ai += 1;
        let mut body = Vec::new();
        match conv {
            b'd' => {
                let x = numeric_i64(v, row)?;
                kernels::fmt_int(x.unsigned_abs() as u128, x < 0, 0, &mut body);
            }
            b'f' => {
                let x = numeric_f64(v, row)?;
                fmt_fixed(x, prec as u8, &mut body);
            }
            b's' => write_display(v, row, &mut body)?,
            _ => err!(UnsupportedFeature),
        }
        pad_field(out, &body, width, left, zero && conv != b's');
    }
    Ok(())
}

/// `format(fmt, args...)`。Python 風の `{}`/`{<n>}` プレースホルダのみ対応
/// する（`{:.2f}` のような書式ミニ言語は非対応: `UnsupportedFeature`）。
/// `{{`/`}}` はそれぞれリテラルの `{`/`}` になる
/// （`duckdb -c "select format('{{literal}}')"` が `{literal}` になることを
/// 確認済み）。値の文字列化は printf の `%s` と同じ `write_display` を使う
/// （`format` には型ごとの指定子が無いので常にこれ 1 種類）。
///
/// `{}` は出現順に実引数を消費し、`{<n>}` は 0 始まりの明示添字（自動採番
/// とは独立にカウントする）。指定子が指す実引数が無ければ `WrongArgCount`。
fn format_scan(fmt: &[u8], a: &A, out: &mut Vec<u8>) -> Result<()> {
    let mut auto_idx = 0usize;
    let mut i = 0usize;
    while i < fmt.len() {
        match fmt[i] {
            b'{' if fmt.get(i + 1) == Some(&b'{') => {
                out.push(b'{');
                i += 2;
            }
            b'{' => {
                i += 1;
                let start = i;
                while i < fmt.len() && fmt[i] != b'}' {
                    ensure!(fmt[i].is_ascii_digit(), UnsupportedFeature);
                    i += 1;
                }
                ensure!(i < fmt.len(), SyntaxError);
                let idx = if start == i {
                    let cur = auto_idx;
                    auto_idx += 1;
                    cur
                } else {
                    parse_format_index(&fmt[start..i])
                };
                i += 1; // '}'
                let ai = idx.saturating_add(1); // args[0] は fmt 自身
                ensure!(ai < a.n(), WrongArgCount);
                let (v, row) = match a.at(ai) {
                    Some(x) => x,
                    None => err!(Internal),
                };
                write_display(v, row, out)?;
            }
            b'}' if fmt.get(i + 1) == Some(&b'}') => {
                out.push(b'}');
                i += 2;
            }
            b'}' => err!(SyntaxError),
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    Ok(())
}

/// `{<digits>}` の添字部分を読む。桁あふれは飽和させる（後段の
/// `ai < a.n()` チェックでどのみち `WrongArgCount` になるので、パースの
/// 時点でオーバーフローを気にする必要は無い）。
fn parse_format_index(digits: &[u8]) -> usize {
    let mut v: usize = 0;
    for &b in digits {
        v = v.saturating_mul(10).saturating_add((b - b'0') as usize);
    }
    v
}

/// `%d`。BOOLEAN・整数型のみ対応。浮動小数は 0 方向へ切り捨てる
/// （Rust の `f64 as i64` は範囲外を飽和させるだけで panic しない）。
fn numeric_i64(v: &Vector, row: usize) -> Result<i64> {
    Ok(match v.data() {
        Data::Bool(b) => b.get(row) as i64,
        Data::I32(d) => d[row] as i64,
        Data::I64(d) => d[row],
        Data::F64(d) => d[row] as i64,
        Data::I128(_) | Data::Bytes(_) => err!(TypeMismatch),
    })
}

/// `%f`。BOOLEAN・整数型・浮動小数に対応。
fn numeric_f64(v: &Vector, row: usize) -> Result<f64> {
    Ok(match v.data() {
        Data::Bool(b) => b.get(row) as i64 as f64,
        Data::I32(d) => d[row] as f64,
        Data::I64(d) => d[row] as f64,
        Data::F64(d) => d[row],
        Data::I128(_) | Data::Bytes(_) => err!(TypeMismatch),
    })
}

/// `%s`（printf）・`{}`（format）が使う汎用の文字列化。対応する物理型は
/// BOOLEAN・整数（I32/I64）・浮動小数（F64）・バイト列（VARCHAR/JSON は
/// そのまま出す）。DATE/TIME/TIMESTAMP/DECIMAL/INTERVAL/HUGEINT（物理型
/// I128、または DECIMAL のスケール）は内部表現をそのまま数値として出す
/// だけで、暦・小数点の変換はしない（範囲を絞る設計判断。カレンダー等の
/// 表示が要る場合は呼び出し側で `CAST(.. AS VARCHAR)` してから渡すこと）。
fn write_display(v: &Vector, row: usize, out: &mut Vec<u8>) -> Result<()> {
    match v.data() {
        Data::Bool(b) => out.extend_from_slice(if b.get(row) { b"true" } else { b"false" }),
        Data::I32(d) => kernels::fmt_int(d[row].unsigned_abs() as u128, d[row] < 0, 0, out),
        Data::I64(d) => kernels::fmt_int(d[row].unsigned_abs() as u128, d[row] < 0, 0, out),
        Data::F64(d) => kernels::fmt_f64(d[row], out),
        Data::Bytes(b) => out.extend_from_slice(b.get(row)),
        Data::I128(_) => err!(UnsupportedFeature),
    }
    Ok(())
}

/// `%f` の固定小数点表記。`10^prec` 倍してから 0 から遠ざける丸めで整数化し
/// `kernels::fmt_int` に渡す（`fmt_int` が小数点の挿入まで面倒を見てくれる）。
/// `prec` は呼び出し元（`printf_scan`）が `MAX_PRINTF_PREC` 以下に丸め済み
/// なので、`fmt_int` 内部バッファ（48 バイト）を溢れさせない。
fn fmt_fixed(x: f64, prec: u8, out: &mut Vec<u8>) {
    if x.is_nan() {
        out.extend_from_slice(b"nan");
        return;
    }
    let neg = x.is_sign_negative();
    let ax = f_abs(x);
    if ax.is_infinite() {
        if neg {
            out.push(b'-');
        }
        out.extend_from_slice(b"inf");
        return;
    }
    let scaled = round_half_up(ax * pow10(prec as u32));
    // u128 に収まらないほど巨大な値は諦めて簡易表記にフォールバックする
    // （`fmt_int` へ渡す前に u128 のレンジ内であることを確認しておく）。
    if scaled >= 1.0e33 {
        if neg {
            out.push(b'-');
        }
        kernels::fmt_f64(ax, out);
        return;
    }
    kernels::fmt_int(scaled as u128, neg, prec, out);
}

/// 幅・0 埋め・左寄せを適用する（`%d`/`%f`/`%s` 共通）。`body` は変換済みの
/// 中身（符号込み）。`%s` は多バイト文字を渡せるので、幅はコードポイント
/// 単位で数える（`lpad`/`rpad` と同じ判断）。
fn pad_field(out: &mut Vec<u8>, body: &[u8], width: usize, left: bool, zero: bool) {
    let len = cp_count(body);
    if len >= width {
        out.extend_from_slice(body);
        return;
    }
    let pad_n = width - len;
    if left {
        out.extend_from_slice(body);
        for _ in 0..pad_n {
            out.push(b' ');
        }
    } else if zero {
        // 符号を残してから 0 を詰める（`-0003` のような形にする）。
        let (sign, rest): (&[u8], &[u8]) = match body.first() {
            Some(b'-') | Some(b'+') => (&body[..1], &body[1..]),
            _ => (&body[..0], body),
        };
        out.extend_from_slice(sign);
        for _ in 0..pad_n {
            out.push(b'0');
        }
        out.extend_from_slice(rest);
    } else {
        for _ in 0..pad_n {
            out.push(b' ');
        }
        out.extend_from_slice(body);
    }
}
