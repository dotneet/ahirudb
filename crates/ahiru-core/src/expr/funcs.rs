//! スカラ関数。
//!
//! `resolve` は計画時に 1 度だけ呼ばれ、名前と引数型から「関数 ID・各引数の
//! 変換先の型・結果型」を決める。暗黙変換は呼び出し側が `Cast` 命令として
//! 挿入するので、実行時 (`call`) は ID の `match` だけで済み、型検査も
//! 型ごとの分岐も行わない。
//!
//! サイズ方針は kernels.rs の 4 つのレバーを引き継ぎ、さらに 3 つ足す:
//!
//! 1. **出力物理型ごとに 1 本のループしか書かない。** 行あたりの処理は
//!    `eval_str` / `eval_int` / `eval_f64` / `eval_bool` の中の `match id` に
//!    閉じ込める。関数ごとにループを書くと関数の数だけコードが増える。
//! 2. **型の族ごとに ID を分ける。** `abs` は整数用と浮動小数用で ID が違う。
//!    実行時に論理型を見ないので、ループ内の判定が 1 段減る。
//! 3. **任意型の関数は kernels の合成で書く。** `coalesce` / `nullif` /
//!    `greatest` / `least` は物理型 6 種ぶんのコードを持たず、
//!    `kernels::pick` / `compare` / `logic` / `is_null` の組み合わせに落とす。
//!
//! ## 文字列はコードポイント単位
//!
//! `length` / `substr` / `strpos` / `lpad` / `rpad` / `reverse` は
//! **UTF-8 のコードポイント単位**で数える（DuckDB と同じ）。継続バイト
//! (`0b10xxxxxx`) を読み飛ばすだけなので、バイト単位実装との差は数十バイト
//! しかない。一方 `upper` / `lower` は **ASCII のみ**に限る。Unicode の
//! 大小文字テーブルは数十 KB あり、1MiB 予算に入らないため。
//!
//! ## NULL
//!
//! 既定は「引数のどれかが NULL ならその行の結果も NULL」。例外は
//! `coalesce` / `ifnull` / `nullif` / `greatest` / `least`（NULL を読み飛ばす）
//! と `concat`（NULL を空文字列として扱い、結果は決して NULL にならない）、
//! `json_array` / `json_object` / `list_value`（NULL 引数はその位置に
//! JSON `null` を埋め込む。`to_json(NULL)` 自体は既定どおり SQL NULL）。
//!
//! ## 現在時刻が無い
//!
//! `now()` / `current_timestamp` / `current_date` は **実装しない**
//! (`FunctionNotFound`)。no_std の wasm には時計が無く、ABI
//! (`crate::abi`) にもホストから時刻を受け取る入口が無い。値をでっち上げると
//! 「クエリごとに違う嘘の時刻」を返すことになるので、ホスト側から
//! エポックマイクロ秒を渡す ABI（例: `ahiru_set_now(i64)`）を足すまでは
//! 未対応のままにする。

use crate::expr::vm::Vm;
use crate::expr::{kernels, regex, OpCode, Program};
use crate::prelude::*;
use crate::vector::{Batch, Bitmap, BytesData, Data, PhysType, Ty, Value, Vector};

pub type FuncId = u16;

// --- 関数 ID ---------------------------------------------------------------
// 出力物理型ごとにまとめてある。`call` のループはこの区分で選ばれる。

// 文字列（Bytes 出力）
const F_UPPER: FuncId = 1;
const F_LOWER: FuncId = 2;
const F_TRIM: FuncId = 3;
const F_LTRIM: FuncId = 4;
const F_RTRIM: FuncId = 5;
const F_SUBSTR: FuncId = 6;
const F_REPLACE: FuncId = 7;
const F_LPAD: FuncId = 8;
const F_RPAD: FuncId = 9;
const F_REVERSE: FuncId = 10;
const F_SPLIT_PART: FuncId = 11;
const F_REPEAT: FuncId = 12;
const F_STRFTIME: FuncId = 13;
const F_CONCAT: FuncId = 14;
const F_REGEXP_EXTRACT: FuncId = 15;
const F_REGEXP_REPLACE: FuncId = 16;
const F_PRINTF: FuncId = 17;
const F_FORMAT: FuncId = 18;

// 整数出力（I32 = DATE、I64 = BIGINT / TIMESTAMP）
const F_LENGTH: FuncId = 20;
const F_STRPOS: FuncId = 21;
const F_DATE_PART: FuncId = 22;
const F_DATE_DIFF: FuncId = 23;
const F_ABS_I: FuncId = 24;
const F_SIGN_I: FuncId = 25;
const F_ROUND_I: FuncId = 26;
const F_MOD_I: FuncId = 27;
const F_DATE_TRUNC: FuncId = 28;
const F_DATE_ADD: FuncId = 29;
const F_TO_DATE: FuncId = 30;
const F_TO_TIMESTAMP: FuncId = 31;
const F_LAST_DAY: FuncId = 32;

// Bool 出力
const F_STARTS_WITH: FuncId = 40;
const F_ENDS_WITH: FuncId = 41;
const F_CONTAINS: FuncId = 42;
const F_REGEXP_MATCHES: FuncId = 43;
const F_GLOB: FuncId = 44;
/// `SIMILAR TO` の実体。`sql::parser` が `x SIMILAR TO y` をこの名前の
/// 関数呼び出しへ脱糖する（`Expr::Like` のような専用 AST ノードは増やさず、
/// 既存の関数呼び出し経路だけで完結させるため）。
const F_REGEXP_FULL_MATCH: FuncId = 45;

// 浮動小数出力
const F_ABS_F: FuncId = 50;
const F_SIGN_F: FuncId = 51;
const F_ROUND_F: FuncId = 52;
const F_CEIL_F: FuncId = 53;
const F_FLOOR_F: FuncId = 54;
const F_TRUNC_F: FuncId = 55;
const F_SQRT: FuncId = 56;
const F_EXP: FuncId = 57;
const F_LN: FuncId = 58;
const F_LOG10: FuncId = 59;
const F_POW: FuncId = 60;
const F_MOD_F: FuncId = 61;

// 任意型（kernels の合成で処理する）
const F_IDENT: FuncId = 70;
const F_COALESCE: FuncId = 71;
const F_NULLIF: FuncId = 72;
const F_GREATEST: FuncId = 73;
const F_LEAST: FuncId = 74;

// ビット演算（`&`/`|`/`<<`/`>>`/前置 `~` の糖衣構文先。`sql::parser` 参照）。
// BIGINT 固定（他の数値関数の「HUGEINT/DECIMAL も丸める」簡略化と同じ理由、
// `num_ty` の doc 参照）。
//
// `F_PART_BASE`（100）以上は `eval_int` が `date_part` 用の動的オフセットと
// して奪ってしまう（`id - F_PART_BASE` を part 番号として読む）ので、
// その手前の空き番号を使う。
const F_BIT_AND: FuncId = 62;
const F_BIT_OR: FuncId = 63;
const F_BIT_SHL: FuncId = 64;
const F_BIT_SHR: FuncId = 65;
const F_BIT_NOT: FuncId = 66;

// JSON（Bytes 出力）。`json_object`/`json_array` は引数の個数が可変で、
// かつ NULL 引数を読み飛ばさず JSON `null` として埋め込む必要があるため、
// `call` の行ループ（`eval_str`）の外、`F_CONCAT` などと同じ場所で処理する。
const F_JSON_EXTRACT: FuncId = 80;
const F_JSON_EXTRACT_STRING: FuncId = 81;
const F_JSON_TYPE: FuncId = 82;
const F_TO_JSON: FuncId = 83;
const F_LIST_EXTRACT: FuncId = 84;
const F_MAP_EXTRACT: FuncId = 85;
const F_JSON_OBJECT: FuncId = 86;
const F_JSON_ARRAY: FuncId = 87;
// `list_transform`/`list_filter`/`list_reduce`。通常のスカラ関数と違い
// `plan::compile::Compiler::lambda_call` が `resolve`/`call` を経由せず直接
// この ID を使ってコンパイルする（第 2 引数がラムダで通常の `Ty` を持たない
// ため）。実行は `call` ではなく `call_lambda`（`expr::vm::exec` の `Call`
// 命令、`CallSpec::lambda` 参照）。`pub(crate)` なのは `plan::compile` から
// 直接参照するため。
pub(crate) const F_LIST_TRANSFORM: FuncId = 88;
pub(crate) const F_LIST_FILTER: FuncId = 89;
pub(crate) const F_LIST_REDUCE: FuncId = 91;

// JSON（整数出力）
const F_JSON_ARRAY_LENGTH: FuncId = 90;

/// `list_slice`/`array_slice`、`expr[i:j]` の脱糖先。92 を選んだのは
/// 84-91 が既に埋まっていて（上記）、かつ `F_PART_BASE`（下記、100）以上は
/// 日付部分抽出専用に予約されているため — `eval_int`/`eval_str` などの
/// 冒頭で `id >= F_PART_BASE` を日付部分抽出とみなす特別扱いがあるので、
/// 100 以上の番号を汎用関数に使うと衝突する。
const F_LIST_SLICE: FuncId = 92;

/// `year()` などの略記。ID に part 番号を埋め込み、`date_part` と同じ
/// 抽出関数へ合流させる（関数ごとに ID を分けても中身は 1 本）。
const F_PART_BASE: FuncId = 100;

// --- 日時の part -----------------------------------------------------------

const P_YEAR: u8 = 0;
const P_QUARTER: u8 = 1;
const P_MONTH: u8 = 2;
const P_WEEK: u8 = 3;
const P_DAY: u8 = 4;
const P_HOUR: u8 = 5;
const P_MINUTE: u8 = 6;
const P_SECOND: u8 = 7;
const P_DOW: u8 = 8;
const P_DOY: u8 = 9;
const P_EPOCH: u8 = 10;

const PART_NAMES: [&[u8]; 11] = [
    b"year", b"quarter", b"month", b"week", b"day", b"hour", b"minute", b"second", b"dow", b"doy",
    b"epoch",
];

/// part 名（大文字小文字を問わない）を番号へ。複数形と `dayofweek` /
/// `dayofyear` も受ける。
fn part_id(s: &[u8]) -> Option<u8> {
    let eq = |w: &[u8], t: &[u8]| {
        w.len() == t.len() && w.iter().zip(t.iter()).all(|(a, b)| a.to_ascii_lowercase() == *b)
    };
    if eq(s, b"dayofweek") {
        return Some(P_DOW);
    }
    if eq(s, b"dayofyear") {
        return Some(P_DOY);
    }
    // 末尾の `s` は落とす（`years` = `year`）。
    let t = if s.len() > 1 && (s[s.len() - 1] | 0x20) == b's' { &s[..s.len() - 1] } else { s };
    PART_NAMES.iter().position(|nm| eq(t, nm)).map(|i| i as u8)
}

// --- 定数 ------------------------------------------------------------------

const US_PER_SEC: i64 = 1_000_000;
const US_PER_MIN: i64 = 60 * US_PER_SEC;
const US_PER_HOUR: i64 = 60 * US_PER_MIN;
const US_PER_DAY: i64 = 24 * US_PER_HOUR;

/// 文字列を生成する関数（`repeat` / `lpad` など）の 1 行あたりの上限。
/// 引数次第でいくらでも確保できてしまうのを防ぐ。
const MAX_STR: i64 = 1 << 24;

// =========================================================================
// resolve
// =========================================================================

/// 名前と引数型から関数を解決する。
///
/// 返り値は `(関数 ID, 各引数の変換先の型, 結果型)`。呼び出し側は引数を
/// 変換先の型へ `Cast` してから `call` に渡す。型検査をコンパイル時に
/// 済ませることで、実行時の分岐を減らす。
pub fn resolve(name: &str, args: &[Ty]) -> Result<(FuncId, Vec<Ty>, Ty)> {
    let lower = name.to_ascii_lowercase();
    let n = args.len();
    use Ty::*;
    match lower.as_str() {
        // --- 文字列 ---------------------------------------------------------
        "upper" | "ucase" => fixed(F_UPPER, &[Varchar], n, 1, Varchar),
        "lower" | "lcase" => fixed(F_LOWER, &[Varchar], n, 1, Varchar),
        "trim" => fixed(F_TRIM, &[Varchar, Varchar], n, 1, Varchar),
        "ltrim" => fixed(F_LTRIM, &[Varchar, Varchar], n, 1, Varchar),
        "rtrim" => fixed(F_RTRIM, &[Varchar, Varchar], n, 1, Varchar),
        "substr" | "substring" => fixed(F_SUBSTR, &[Varchar, BigInt, BigInt], n, 2, Varchar),
        "replace" => fixed(F_REPLACE, &[Varchar, Varchar, Varchar], n, 3, Varchar),
        "lpad" => fixed(F_LPAD, &[Varchar, BigInt, Varchar], n, 3, Varchar),
        "rpad" => fixed(F_RPAD, &[Varchar, BigInt, Varchar], n, 3, Varchar),
        "reverse" => fixed(F_REVERSE, &[Varchar], n, 1, Varchar),
        "split_part" => fixed(F_SPLIT_PART, &[Varchar, Varchar, BigInt], n, 3, Varchar),
        "repeat" => fixed(F_REPEAT, &[Varchar, BigInt], n, 2, Varchar),
        "concat" => {
            ensure!(n >= 1, WrongArgCount);
            Ok((F_CONCAT, vec![Varchar; n], Varchar))
        }
        "length" | "len" | "char_length" | "character_length" => {
            fixed(F_LENGTH, &[Varchar], n, 1, BigInt)
        }
        "strpos" | "position" | "instr" => fixed(F_STRPOS, &[Varchar, Varchar], n, 2, BigInt),
        "starts_with" | "prefix" => fixed(F_STARTS_WITH, &[Varchar, Varchar], n, 2, Boolean),
        "ends_with" | "suffix" => fixed(F_ENDS_WITH, &[Varchar, Varchar], n, 2, Boolean),
        "contains" => fixed(F_CONTAINS, &[Varchar, Varchar], n, 2, Boolean),
        // 正規表現エンジンの実体は expr::regex（対応構文・上限はそちらの
        // モジュール冒頭コメント参照）。ここは型解決だけ。
        "regexp_matches" => fixed(F_REGEXP_MATCHES, &[Varchar, Varchar], n, 2, Boolean),
        "regexp_extract" => fixed(F_REGEXP_EXTRACT, &[Varchar, Varchar, BigInt], n, 2, Varchar),
        "regexp_replace" => {
            fixed(F_REGEXP_REPLACE, &[Varchar, Varchar, Varchar, Varchar], n, 3, Varchar)
        }
        // `SIMILAR TO` の脱糖先（`sql::parser::Parser::similar_to`）。パターンは
        // `^(?:...)$` に包んで `expr::regex` の既存エンジンへそのまま渡す
        // （対応構文・ステップ数上限はそちらのモジュール doc 参照。新しい
        // 正規表現エンジンは書かない）。DuckDB もパターン構文としては同じ
        // ものを使う（`_`/`%` は POSIX 正規表現では特別扱いされない）。
        "regexp_full_match" => fixed(F_REGEXP_FULL_MATCH, &[Varchar, Varchar], n, 2, Boolean),
        // `GLOB` 演算子（`sql::parser::expr_body` が脱糖する）。シェルの
        // グロブパターン（`*` `?` `[...]`）で完全一致するかどうか。対応構文は
        // `glob_match` のコメント参照。
        "glob" => fixed(F_GLOB, &[Varchar, Varchar], n, 2, Boolean),
        // `printf`/`format` は対応する書式指定子の範囲が違う（printf: `%`
        // 書式、format: `{}` プレースホルダ）ので ID を分ける。1 個目の
        // 引数（書式文字列）だけ VARCHAR に固定し、残りは元の型のまま渡す
        // （`json_array`/`greatest` と同じ「型ごとの分岐は実行時に」方針。
        // 書式文字列は定数とは限らないので、`resolve` の時点では指定子の
        // 個数も種類も分からない）。対応範囲は `printf_scan`/`format_scan`
        // のコメント参照。
        "printf" | "format" => {
            ensure!(n >= 1, WrongArgCount);
            let mut want = vec![Varchar];
            want.extend_from_slice(&args[1..]);
            let id = if lower == "printf" { F_PRINTF } else { F_FORMAT };
            Ok((id, want, Varchar))
        }

        // --- 数値 -----------------------------------------------------------
        // 整数と浮動小数で ID を分ける。実行時に論理型を見ないため。
        "abs" => num1(args, n, F_ABS_I, F_ABS_F),
        "sign" => num1(args, n, F_SIGN_I, F_SIGN_F),
        // 整数の ceil / floor / trunc は恒等。専用カーネルを持たない。
        "ceil" | "ceiling" => num1(args, n, F_IDENT, F_CEIL_F),
        "floor" => num1(args, n, F_IDENT, F_FLOOR_F),
        "trunc" => num1(args, n, F_IDENT, F_TRUNC_F),
        "round" => {
            ensure!((1..=2).contains(&n), WrongArgCount);
            let t = num_ty(&args[..1])?;
            let id = if t == Double { F_ROUND_F } else { F_ROUND_I };
            let mut a = vec![t];
            if n == 2 {
                a.push(BigInt);
            }
            Ok((id, a, t))
        }
        "mod" => {
            ensure!(n == 2, WrongArgCount);
            let t = num_ty(args)?;
            Ok((if t == Double { F_MOD_F } else { F_MOD_I }, vec![t; 2], t))
        }
        "sqrt" => fixed(F_SQRT, &[Double], n, 1, Double),
        "exp" => fixed(F_EXP, &[Double], n, 1, Double),
        "ln" => fixed(F_LN, &[Double], n, 1, Double),
        // DuckDB の `log(x)` は常用対数。2 引数の `log(b, x)` は未対応。
        "log" | "log10" => fixed(F_LOG10, &[Double], n, 1, Double),
        "pow" | "power" => fixed(F_POW, &[Double, Double], n, 2, Double),
        // `&`/`|`/`<<`/`>>`/前置 `~` の糖衣構文先（`sql::parser` 参照）。
        "bit_and" => fixed(F_BIT_AND, &[BigInt, BigInt], n, 2, BigInt),
        "bit_or" => fixed(F_BIT_OR, &[BigInt, BigInt], n, 2, BigInt),
        "bit_shift_left" => fixed(F_BIT_SHL, &[BigInt, BigInt], n, 2, BigInt),
        "bit_shift_right" => fixed(F_BIT_SHR, &[BigInt, BigInt], n, 2, BigInt),
        "bit_not" => fixed(F_BIT_NOT, &[BigInt], n, 1, BigInt),

        // --- JSON -------------------------------------------------------
        // パス構文・既知の制限は `crate::json` のモジュール doc を参照。
        // `->`/`->>` 演算子は `json_extract`/`json_extract_string` への
        // 糖衣構文として `sql::parser` が展開する。
        "json_extract" => fixed(F_JSON_EXTRACT, &[Json, Varchar], n, 2, Json),
        "json_extract_string" => fixed(F_JSON_EXTRACT_STRING, &[Json, Varchar], n, 2, Varchar),
        "json_type" => json_path_opt(F_JSON_TYPE, n, Varchar),
        "json_array_length" => json_path_opt(F_JSON_ARRAY_LENGTH, n, BigInt),
        // DuckDB の慣習で LIST/MAP 系の名前を持つが、実体は Ty::Json の
        // 配列/オブジェクト形状を読むだけ（モジュール冒頭 doc の設計判断）。
        "list_extract" | "array_extract" => fixed(F_LIST_EXTRACT, &[Json, BigInt], n, 2, Json),
        // `expr[i:j]`（`sql::parser::Parser::subscript`）の脱糖先。両端含む、
        // 1 始まり、負数は末尾から、範囲外は `list_extract` と違い NULL では
        // なく切り詰める（`crate::json::list_slice` のモジュール doc 参照）。
        "list_slice" | "array_slice" => fixed(F_LIST_SLICE, &[Json, BigInt, BigInt], n, 3, Json),
        "map_extract" => fixed(F_MAP_EXTRACT, &[Json, Varchar], n, 2, Json),
        "to_json" => {
            ensure!(n == 1, WrongArgCount);
            ensure!(json_encodable(args[0]), TypeMismatch);
            Ok((F_TO_JSON, vec![args[0]], Json))
        }
        "json_object" => {
            ensure!(n >= 2 && n.is_multiple_of(2), WrongArgCount);
            let mut want = Vec::with_capacity(n);
            for (i, &a) in args.iter().enumerate() {
                if i % 2 == 0 {
                    want.push(Varchar);
                } else {
                    ensure!(json_encodable(a), TypeMismatch);
                    want.push(a);
                }
            }
            Ok((F_JSON_OBJECT, want, Json))
        }
        // `list_value` は DuckDB の慣習で `json_array` の別名。
        "json_array" | "list_value" => {
            ensure!(n >= 1, WrongArgCount);
            for &a in args {
                ensure!(json_encodable(a), TypeMismatch);
            }
            Ok((F_JSON_ARRAY, args.to_vec(), Json))
        }

        // --- 任意型 ---------------------------------------------------------
        "greatest" => anyn(F_GREATEST, args, 1),
        "least" => anyn(F_LEAST, args, 1),
        "coalesce" => anyn(F_COALESCE, args, 1),
        "ifnull" => {
            ensure!(n == 2, WrongArgCount);
            anyn(F_COALESCE, args, 2)
        }
        "nullif" => {
            ensure!(n == 2, WrongArgCount);
            anyn(F_NULLIF, args, 2)
        }

        // --- 日時 -----------------------------------------------------------
        // 内部表現は DATE=日数(I32) / TIMESTAMP=マイクロ秒(I64) / TIME=マイクロ秒(I64)。
        // 引数を TIMESTAMP に寄せるので、DATE 列も VARCHAR リテラルも
        // 呼び出し側の `Cast` 経由でそのまま渡せる。
        //
        // 注: DuckDB 1.4 の `date_trunc` は part によって DATE / TIMESTAMP を
        // 返し分けるが、`resolve` は part の**値**を見られない（型しか来ない）
        // ので常に TIMESTAMP を返す。同じ理由で `date_part('epoch', ..)` も
        // DOUBLE ではなく BIGINT（秒、切り捨て）を返す。
        "date_trunc" | "datetrunc" => fixed(F_DATE_TRUNC, &[Varchar, Timestamp], n, 2, Timestamp),
        "date_part" | "datepart" | "extract" => {
            fixed(F_DATE_PART, &[Varchar, Timestamp], n, 2, BigInt)
        }
        "date_diff" | "datediff" => {
            fixed(F_DATE_DIFF, &[Varchar, Timestamp, Timestamp], n, 3, BigInt)
        }
        // DuckDB の `date_add` は INTERVAL を取るが、この実装に INTERVAL 型は
        // 無い。意図的な非互換として `date_add(part, n, ts)` の 3 引数にする。
        "date_add" => fixed(F_DATE_ADD, &[Varchar, BigInt, Timestamp], n, 3, Timestamp),
        "last_day" => fixed(F_LAST_DAY, &[Timestamp], n, 1, Date),
        "strftime" => fixed(F_STRFTIME, &[Timestamp, Varchar], n, 2, Varchar),
        // DuckDB の `to_timestamp` はエポック秒(DOUBLE)を取るが、ここは
        // 文字列パーサとして定義する（`strptime` 相当）。意図的な非互換。
        "to_date" => fixed(F_TO_DATE, &[Varchar], n, 1, Date),
        "to_timestamp" => fixed(F_TO_TIMESTAMP, &[Varchar], n, 1, Timestamp),
        "year" => shorthand(P_YEAR, n),
        "quarter" => shorthand(P_QUARTER, n),
        "month" => shorthand(P_MONTH, n),
        "week" => shorthand(P_WEEK, n),
        "day" | "dayofmonth" => shorthand(P_DAY, n),
        "hour" => shorthand(P_HOUR, n),
        "minute" => shorthand(P_MINUTE, n),
        "second" => shorthand(P_SECOND, n),
        "dayofweek" => shorthand(P_DOW, n),
        "dayofyear" => shorthand(P_DOY, n),
        "epoch" => shorthand(P_EPOCH, n),

        // 時計が無いので実装しない（モジュール冒頭のコメント参照）。
        _ => err!(FunctionNotFound),
    }
}

/// 固定シグネチャ。`pat` は引数型のひな形で、`n < pat.len()` なら前から使う
/// （省略可能引数）。
fn fixed(id: FuncId, pat: &[Ty], n: usize, lo: usize, ret: Ty) -> Result<(FuncId, Vec<Ty>, Ty)> {
    ensure!(n >= lo && n <= pat.len(), WrongArgCount);
    Ok((id, pat[..n].to_vec(), ret))
}

/// `f(json)` / `f(json, path)` の 1-2 引数シグネチャ。`json_type`/
/// `json_array_length` など「対象が JSON 全体かパス指定か」を第 2 引数の
/// 有無だけで切り替える JSON 関数の共通形。
fn json_path_opt(id: FuncId, n: usize, ret: Ty) -> Result<(FuncId, Vec<Ty>, Ty)> {
    ensure!((1..=2).contains(&n), WrongArgCount);
    let mut want = vec![Ty::Json];
    if n == 2 {
        want.push(Ty::Varchar);
    }
    Ok((id, want, ret))
}

/// 数値 1 引数。整数なら `int_id`、それ以外は `flt_id`。
fn num1(args: &[Ty], n: usize, int_id: FuncId, flt_id: FuncId) -> Result<(FuncId, Vec<Ty>, Ty)> {
    ensure!(n == 1, WrongArgCount);
    let t = num_ty(args)?;
    Ok((if t == Ty::Double { flt_id } else { int_id }, vec![t], t))
}

/// 数値引数の共通型。整数だけなら BIGINT、それ以外は DOUBLE。
///
/// HUGEINT / DECIMAL も DOUBLE に寄せる。i128 と 10 進スケール用の分岐を
/// 全関数に持たせるとコード量がおよそ倍になるため（既知の精度上の制限）。
fn num_ty(args: &[Ty]) -> Result<Ty> {
    let mut int = true;
    for &t in args {
        if t == Ty::Null {
            continue;
        }
        ensure!(t.is_numeric(), TypeMismatch);
        if !t.is_integer() || matches!(t, Ty::HugeInt | Ty::UBigInt) {
            int = false;
        }
    }
    Ok(if int { Ty::BigInt } else { Ty::Double })
}

/// `to_json`/`json_array`/`json_object` の値として書ける型か。
/// 対応するのは NULL/BOOLEAN/整数/浮動小数/DECIMAL/VARCHAR/DATE/TIME/
/// TIMESTAMP/JSON（そのまま埋め込む）。BLOB・INTERVAL は非対応
/// （JSON に自然な表現が無いため、CAST と同様 `TypeMismatch` で拒否する）。
fn json_encodable(t: Ty) -> bool {
    use Ty::*;
    t.is_numeric() || matches!(t, Null | Boolean | Varchar | Date | Time | Timestamp | Json)
}

/// 任意型の可変長関数。全引数を共通型へ寄せる。
fn anyn(id: FuncId, args: &[Ty], lo: usize) -> Result<(FuncId, Vec<Ty>, Ty)> {
    ensure!(args.len() >= lo, WrongArgCount);
    let mut t = Ty::Null;
    for &a in args {
        t = match Ty::unify(t, a) {
            Some(u) => u,
            None => err!(TypeMismatch),
        };
    }
    // 全部が型未定の NULL なら適当な型に決めておく（DuckDB も INTEGER）。
    if t == Ty::Null {
        t = Ty::Int;
    }
    Ok((id, vec![t; args.len()], t))
}

/// `year(ts)` などの略記。part を ID に埋め込む。
fn shorthand(part: u8, n: usize) -> Result<(FuncId, Vec<Ty>, Ty)> {
    ensure!(n == 1, WrongArgCount);
    Ok((F_PART_BASE + part as FuncId, vec![Ty::Timestamp], Ty::BigInt))
}

// =========================================================================
// call
// =========================================================================

/// 実行する。`args` は `resolve` が返した型に揃っている。
///
/// 引数はすべて同じ長さか、長さ 1（定数）。定数は stride 0 として扱う。
pub fn call(id: FuncId, result_ty: Ty, args: &[&Vector]) -> Result<Vector> {
    // 任意型の関数は kernels の合成で片づける（物理型ごとのコードを持たない）。
    match id {
        F_IDENT => {
            ensure!(args.len() == 1, WrongArgCount);
            return Ok(args[0].clone());
        }
        F_COALESCE => return fold_null(args, result_ty),
        F_NULLIF => return nullif(args, result_ty),
        F_GREATEST | F_LEAST => return extremum(id == F_GREATEST, args, result_ty),
        F_CONCAT => return concat_all(args, result_ty),
        // 正規表現 3 関数は行ごとの eval_* ループに乗せない。パターン列が
        // 定数（stride 0）ならバッチ内で 1 回だけコンパイルしたいが、その
        // キャッシュを eval_bool/eval_str の共通シグネチャに持ち込むのは
        // 不自然なので、F_CONCAT 等と同じく call() 直下で専用関数へ渡す。
        F_REGEXP_MATCHES => return regex::eval_matches(args),
        F_REGEXP_EXTRACT => return regex::eval_extract(args),
        F_REGEXP_REPLACE => return regex::eval_replace(args),
        // `SIMILAR TO`（`regexp_full_match`）も同じ理由でパターン列のコンパイル
        // をキャッシュしたいので、専用関数へ渡す（`regexp_full_match_build`
        // 参照。実体は `regex::eval_matches` とほぼ同じで、パターンをアンカー
        // で包む点だけが違う）。
        F_REGEXP_FULL_MATCH => return regexp_full_match_build(args),
        // `json_array`/`json_object` も NULL 引数を読み飛ばさず JSON `null`
        // として埋め込む（`concat` と同じ理由で既定の NULL 伝播から外す）ので
        // 専用関数へ渡す。
        F_JSON_ARRAY => return json_array_build(args),
        F_JSON_OBJECT => return json_object_build(args),
        _ => {}
    }
    let (n, s) = strides(args)?;
    // 入力 validity の AND。NULL 行は評価自体を飛ばす（プレースホルダ値を
    // 読ませない）。
    let valid = combine(args, &s, n);
    let live = |i: usize| valid.as_ref().is_none_or(|b| b.get(i));
    let mut data = Data::with_capacity(result_ty.phys(), n);
    let mut bad: Option<Bitmap> = None;
    match result_ty.phys() {
        PhysType::Bytes => {
            let mut buf = Vec::new();
            if let Data::Bytes(d) = &mut data {
                for i in 0..n {
                    if !live(i) {
                        d.push_empty();
                        continue;
                    }
                    buf.clear();
                    if eval_str(id, &A { v: args, s: &s, i }, &mut buf)? {
                        // offsets は u32。これを超える結果は扱えない。
                        ensure!(d.data.len() + buf.len() <= u32::MAX as usize, LimitExceeded);
                        d.push(&buf);
                    } else {
                        d.push_empty();
                        set_null(&mut bad, i, n);
                    }
                }
            }
        }
        PhysType::Bool => {
            if let Data::Bool(d) = &mut data {
                for i in 0..n {
                    let v = if live(i) { eval_bool(id, &A { v: args, s: &s, i })? } else { None };
                    match v {
                        Some(x) => d.push(x),
                        None => {
                            d.push(false);
                            if live(i) {
                                set_null(&mut bad, i, n);
                            }
                        }
                    }
                }
            }
        }
        PhysType::F64 => {
            if let Data::F64(d) = &mut data {
                for i in 0..n {
                    let v = if live(i) { eval_f64(id, &A { v: args, s: &s, i })? } else { None };
                    match v {
                        Some(x) => d.push(x),
                        None => {
                            d.push(0.0);
                            if live(i) {
                                set_null(&mut bad, i, n);
                            }
                        }
                    }
                }
            }
        }
        PhysType::I32 | PhysType::I64 => {
            for i in 0..n {
                let v = if live(i) { eval_int(id, &A { v: args, s: &s, i })? } else { Some(0) };
                // I32 出力（DATE）は範囲外もその行だけ NULL にする。
                let ok = match v {
                    Some(x) => push_int(&mut data, x),
                    None => {
                        push_int(&mut data, 0);
                        false
                    }
                };
                if !ok && live(i) {
                    set_null(&mut bad, i, n);
                }
            }
        }
        PhysType::I128 => err!(TypeMismatch),
    }
    let mut out = Vector::from_data(result_ty, data, merge(valid, bad));
    out.compact_validity();
    Ok(out)
}

// --- 行ループの下回り -------------------------------------------------------

/// 行数と各引数の stride。長さ 1 の引数は stride 0（＝定数）。
/// `expr::regex` からも使う共通ヘルパー（同じ「定数 or 全部同じ長さ」規則）。
pub(crate) fn strides(args: &[&Vector]) -> Result<(usize, Vec<usize>)> {
    ensure!(!args.is_empty(), WrongArgCount);
    let mut n = 1usize;
    let mut empty = false;
    for a in args {
        let l = a.len();
        if l == 0 {
            empty = true;
        } else if l > n {
            n = l;
        }
    }
    if empty {
        n = 0;
    }
    let mut s = Vec::with_capacity(args.len());
    for a in args {
        s.push(if a.len() == n {
            1
        } else if a.len() == 1 {
            0
        } else {
            err!(Internal)
        });
    }
    Ok((n, s))
}

/// 全引数の validity の AND。NULL 無しなら `None`。`expr::regex` と共有。
pub(crate) fn combine(args: &[&Vector], s: &[usize], n: usize) -> Option<Bitmap> {
    if !args.iter().any(|a| a.has_nulls()) {
        return None;
    }
    let mut m = Bitmap::with_capacity(n);
    for i in 0..n {
        let mut ok = true;
        for (k, a) in args.iter().enumerate() {
            ok &= a.is_valid(i * s[k]);
        }
        m.push(ok);
    }
    Some(m)
}

/// 行 `i` を NULL にする（遅延確保）。`expr::kernels`/`expr::regex` にも
/// 同じ役割の実装がある（それぞれの呼び出しパターンに合わせて独立に持つ）。
pub(crate) fn set_null(m: &mut Option<Bitmap>, i: usize, n: usize) {
    if m.is_none() {
        *m = Some(Bitmap::ones(n));
    }
    if let Some(b) = m {
        b.set(i, false);
    }
}

fn merge(a: Option<Bitmap>, b: Option<Bitmap>) -> Option<Bitmap> {
    match (a, b) {
        (Some(mut x), Some(y)) => {
            x.and_assign(&y);
            Some(x)
        }
        (Some(x), None) => Some(x),
        (None, y) => y,
    }
}

/// i64 を整数出力へ書く。DATE (I32) の範囲外は `false`（＝ NULL）。
fn push_int(d: &mut Data, x: i64) -> bool {
    match d {
        Data::I32(v) => match i32::try_from(x) {
            Ok(z) => {
                v.push(z);
                true
            }
            Err(_) => {
                v.push(0);
                false
            }
        },
        Data::I64(v) => {
            v.push(x);
            true
        }
        _ => false,
    }
}

/// 行ごとの引数アクセサ。範囲外・物理型不一致では既定値を返す。
/// 正しい引数は `resolve` が保証するが、`call` を直接呼ばれても
/// パニックしないようにしておく。
struct A<'a, 'b> {
    v: &'a [&'b Vector],
    s: &'a [usize],
    i: usize,
}

impl<'b> A<'_, 'b> {
    #[inline]
    fn n(&self) -> usize {
        self.v.len()
    }

    #[inline]
    fn at(&self, k: usize) -> Option<(&'b Vector, usize)> {
        match (self.v.get(k), self.s.get(k)) {
            (Some(v), Some(st)) => Some((v, self.i * st)),
            _ => None,
        }
    }

    fn bytes(&self, k: usize) -> &'b [u8] {
        match self.at(k) {
            Some((v, j)) if matches!(v.data(), Data::Bytes(_)) && j < v.len() => v.bytes().get(j),
            _ => &[],
        }
    }

    fn int(&self, k: usize) -> i64 {
        match self.at(k) {
            Some((v, j)) => match v.data() {
                Data::I64(d) => d.get(j).copied().unwrap_or(0),
                Data::I32(d) => d.get(j).copied().unwrap_or(0) as i64,
                _ => 0,
            },
            None => 0,
        }
    }

    fn flt(&self, k: usize) -> f64 {
        match self.at(k) {
            Some((v, j)) => match v.data() {
                Data::F64(d) => d.get(j).copied().unwrap_or(0.0),
                _ => 0.0,
            },
            None => 0.0,
        }
    }
}

// --- Submodules --------------------------------------------------------
//
// This file stays a thin entry point: constants, `resolve`, `call`, and the
// `A` argument bundle. The per-physical-type eval bodies (`eval_str`/
// `eval_bool`/`eval_int`/`eval_f64`) and their supporting helpers live in
// `expr::funcs::*` submodules instead (same split strategy as
// `sql::parser` -> `sql::parser/`).

mod bool_ops;
mod datetime;
mod json;
mod lambda;
mod numeric;
mod string;

#[cfg(test)]
mod tests;

// `call()` above dispatches into these submodule bodies.
use bool_ops::eval_bool;
use json::{
    concat_all, extremum, fold_null, json_array_build, json_object_build, nullif,
    regexp_full_match_build,
};
use numeric::{eval_f64, eval_int};
use string::eval_str;

// Other modules (`expr::kernels`/`expr::regex`/`write::csv`/`write::jsonl`/
// `parquet::nested`/`plan::compile`/`expr::vm`) reach these items through
// the `funcs::X` path that existed before this split. Re-export them here
// so that path keeps resolving — only the implementation moved, not the
// public name/path.
pub(crate) use datetime::{
    add_interval_to_ts, days_from_civil, fmt_date, fmt_time, fmt_timestamp, fmt_timestamptz,
    fmt_uuid, parse_date, parse_time, parse_timestamp, parse_timestamptz, parse_uuid,
};
// Only `write::csv`/`write::jsonl` (both gated on `export`) reach this one;
// every other caller of `datetime::*` above is unconditional.
#[cfg(feature = "export")]
pub(crate) use datetime::civil_from_days;
pub use lambda::call_lambda;
pub(crate) use numeric::{f_abs, f_trunc};
