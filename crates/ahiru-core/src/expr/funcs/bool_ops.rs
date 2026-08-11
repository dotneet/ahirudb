//! Bool 出力
use super::string::find;
use super::*;

pub(super) fn eval_bool(id: FuncId, a: &A) -> Result<Option<bool>> {
    let (s, p) = (a.bytes(0), a.bytes(1));
    Ok(Some(match id {
        F_STARTS_WITH => s.len() >= p.len() && &s[..p.len()] == p,
        F_ENDS_WITH => s.len() >= p.len() && &s[s.len() - p.len()..] == p,
        F_CONTAINS => find(s, p).is_some(),
        // `glob_match` は `like_match` と同じ「直近の `*` を 1 個だけ覚える」
        // 2 ポインタ法なので、正規表現のようなコンパイル・キャッシュは不要
        // （呼び出しごとの `call()` 直下バイパスをしなくて済む理由）。
        F_GLOB => glob_match(s, p),
        _ => err!(Internal),
    }))
}

/// シェルグロブパターン照合。DuckDB の `GLOB` 演算子と同じ意味（`duckdb`
/// CLI で以下すべて確認済み）:
///
/// - `*` は 0 バイト以上、`?` はちょうど 1 バイトに一致する。
/// - `[...]` は文字クラス。`[!...]` が否定（`[^...]` は否定にならず、
///   「文字 `^` を含むクラス」になる。DuckDB もそう）。`a-z` のような範囲、
///   先頭に置いた `]` はリテラルの `]`（`[]]` は `]` 1 文字に一致）に対応。
/// - `\` は次の 1 バイトをそのままリテラル化する（`ESCAPE` 句という概念は
///   無く、常時有効）。
/// - 閉じ `]` の無い `[` は、それ以降のパターンをまるごと「絶対に一致しない
///   要素」として扱う（DuckDB も同じ: `'a[bc' GLOB 'a[bc'` ですら false に
///   なる＝ `[` はリテラルへフォールバックしない）。パニックはしない。
///
/// マルチバイト文字は他の文字列関数と違い**バイト単位**で扱う（`regexp`
/// 系と同じ判断。文字クラスをコードポイント単位にするとコード量が増える
/// わりに実利が薄い）。
///
/// `like_match`（`kernels.rs`）と同じ「直前の `*` の位置を 1 つだけ覚える」
/// 2 ポインタ法。バックトラックが 1 箇所しか無いので最悪 `O(|s| * |p|)` に
/// 収まり、`***...*` のような病的なパターンでも指数時間にならない。
fn glob_match(s: &[u8], p: &[u8]) -> bool {
    let (mut si, mut pi) = (0usize, 0usize);
    let (mut star_p, mut star_s) = (usize::MAX, 0usize);
    loop {
        if pi < p.len() && p[pi] == b'*' {
            star_p = pi;
            star_s = si;
            pi += 1;
            continue;
        }
        if si < s.len() && pi < p.len() && glob_atom(p, &mut pi, s[si]) {
            si += 1;
            continue;
        }
        // 直前の `*` に 1 バイト多く食わせて再試行する。`*` が一度も無ければ
        // 不一致で確定。
        if si < s.len() && star_p != usize::MAX {
            star_s += 1;
            si = star_s;
            pi = star_p + 1;
            continue;
        }
        break;
    }
    while pi < p.len() && p[pi] == b'*' {
        pi += 1;
    }
    si == s.len() && pi == p.len()
}

/// `p[*pi]` から始まる 1 要素（リテラル・`?`・`\x`・`[...]`）が `c` に
/// 一致するかを判定し、一致・不一致に関わらず `*pi` を要素の直後まで進める。
/// `*` は呼び出し側（`glob_match`）が先に処理するのでここには来ない。
fn glob_atom(p: &[u8], pi: &mut usize, c: u8) -> bool {
    match p[*pi] {
        b'?' => {
            *pi += 1;
            true
        }
        // 末尾の `\` はエスケープ対象が無いので、素直にリテラルの `\` として
        // 扱う（フォールバック。パニックはしない）。
        b'\\' if *pi + 1 < p.len() => {
            let want = p[*pi + 1];
            *pi += 2;
            want == c
        }
        b'[' => glob_class(p, pi, c),
        lit => {
            *pi += 1;
            lit == c
        }
    }
}

/// `p[*pi]`（== `[`）から始まる文字クラスを読み、`c` が属すかを判定して
/// `*pi` を閉じ `]` の直後まで進める。閉じ `]` が無ければ「常に不一致」を
/// 返し、`*pi` をパターン末尾まで進める（構造体コメントの通り、DuckDB も
/// 同じ挙動）。
fn glob_class(p: &[u8], pi: &mut usize, c: u8) -> bool {
    let mut i = *pi + 1;
    let negate = p.get(i) == Some(&b'!');
    if negate {
        i += 1;
    }
    let members_start = i;
    // 先頭の `]` はクラスの終端ではなくリテラルメンバとして扱う。
    if p.get(i) == Some(&b']') {
        i += 1;
    }
    while i < p.len() && p[i] != b']' {
        i += 1;
    }
    if i >= p.len() {
        *pi = p.len();
        return false;
    }
    let close = i;
    *pi = close + 1;
    let mut hit = false;
    let mut j = members_start;
    while j < close {
        // 先頭・末尾でない `-` は範囲指定（`a-z`）。
        if p[j] == b'-' && j > members_start && j + 1 < close {
            let (lo, hi) = (p[j - 1], p[j + 1]);
            if lo <= c && c <= hi {
                hit = true;
            }
            j += 2;
        } else {
            if p[j] == c {
                hit = true;
            }
            j += 1;
        }
    }
    hit != negate
}
