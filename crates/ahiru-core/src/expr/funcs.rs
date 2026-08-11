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

use crate::expr::{kernels, regex, OpCode};
use crate::prelude::*;
use crate::vector::{Bitmap, BytesData, Data, PhysType, Ty, Vector};

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

// JSON（整数出力）
const F_JSON_ARRAY_LENGTH: FuncId = 90;

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

        // --- JSON -------------------------------------------------------
        // パス構文・既知の制限は `crate::json` のモジュール doc を参照。
        // `->`/`->>` 演算子は `json_extract`/`json_extract_string` への
        // 糖衣構文として `sql::parser` が展開する。
        "json_extract" => fixed(F_JSON_EXTRACT, &[Json, Varchar], n, 2, Json),
        "json_extract_string" => fixed(F_JSON_EXTRACT_STRING, &[Json, Varchar], n, 2, Varchar),
        "json_type" => {
            ensure!((1..=2).contains(&n), WrongArgCount);
            let mut want = vec![Json];
            if n == 2 {
                want.push(Varchar);
            }
            Ok((F_JSON_TYPE, want, Varchar))
        }
        "json_array_length" => {
            ensure!((1..=2).contains(&n), WrongArgCount);
            let mut want = vec![Json];
            if n == 2 {
                want.push(Varchar);
            }
            Ok((F_JSON_ARRAY_LENGTH, want, BigInt))
        }
        // DuckDB の慣習で LIST/MAP 系の名前を持つが、実体は Ty::Json の
        // 配列/オブジェクト形状を読むだけ（モジュール冒頭 doc の設計判断）。
        "list_extract" | "array_extract" => fixed(F_LIST_EXTRACT, &[Json, BigInt], n, 2, Json),
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
fn strides(args: &[&Vector]) -> Result<(usize, Vec<usize>)> {
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

/// 全引数の validity の AND。NULL 無しなら `None`。
fn combine(args: &[&Vector], s: &[usize], n: usize) -> Option<Bitmap> {
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

/// 行 `i` を NULL にする（遅延確保）。
fn set_null(m: &mut Option<Bitmap>, i: usize, n: usize) {
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

// =========================================================================
// 任意型の関数（kernels の合成）
// =========================================================================

/// `COALESCE` / `IFNULL`。先頭から最初の非 NULL を採る。
fn fold_null(args: &[&Vector], ty: Ty) -> Result<Vector> {
    ensure!(!args.is_empty(), WrongArgCount);
    let mut acc = args[0].clone();
    for a in &args[1..] {
        acc = kernels::pick(None, &acc, a, ty)?;
    }
    Ok(acc)
}

/// `NULLIF(a, b)`。`a = b` が **TRUE のときだけ** NULL。
/// どちらかが NULL で比較結果が NULL の行は `a` をそのまま返す
/// （`nullif(1, NULL)` は 1）。
fn nullif(args: &[&Vector], ty: Ty) -> Result<Vector> {
    ensure!(args.len() == 2, WrongArgCount);
    let eq = kernels::compare(OpCode::Eq, ty.phys(), args[0], args[1])?;
    // 長さ 1 の NULL は stride 0 で全行に効く。行数ぶん作る必要は無い。
    let mut nul = Vector::new(ty);
    nul.push_null();
    kernels::pick(Some(&eq), &nul, args[0], ty)
}

/// `GREATEST` / `LEAST`。DuckDB と同じく NULL を読み飛ばし、全部 NULL の
/// ときだけ NULL を返す。
///
/// 「acc を残す条件」= `acc IS NOT NULL AND (b IS NULL OR acc > b)` を
/// 三値論理でそのまま組み立てる。物理型ごとの比較は `kernels::compare` が
/// 既に持っているので、ここに型分岐は現れない。
fn extremum(want_max: bool, args: &[&Vector], ty: Ty) -> Result<Vector> {
    ensure!(!args.is_empty(), WrongArgCount);
    let op = if want_max { OpCode::Gt } else { OpCode::Lt };
    let mut acc = args[0].clone();
    for b in &args[1..] {
        let c = kernels::compare(op, ty.phys(), &acc, b)?;
        let c = kernels::logic(OpCode::Or, &c, &kernels::is_null(b, true))?;
        let c = kernels::logic(OpCode::And, &kernels::is_null(&acc, false), &c)?;
        acc = kernels::pick(Some(&c), &acc, b, ty)?;
    }
    Ok(acc)
}

/// `concat`。DuckDB と同じく NULL 引数は空文字列として無視するので、
/// 結果は決して NULL にならない。
fn concat_all(args: &[&Vector], ty: Ty) -> Result<Vector> {
    let (n, s) = strides(args)?;
    let mut out = BytesData::with_capacity(n, n * 8);
    for i in 0..n {
        for (k, a) in args.iter().enumerate() {
            let j = i * s[k];
            if a.is_valid(j) {
                out.data.extend_from_slice(a.bytes().get(j));
            }
        }
        ensure!(out.data.len() <= u32::MAX as usize, LimitExceeded);
        out.offsets.push(out.data.len() as u32);
    }
    Ok(Vector::from_data(ty, Data::Bytes(out), None))
}

// =========================================================================
// JSON 構築（`to_json` の値直列化を共有する）
// =========================================================================

/// スカラ値 1 個を JSON テキストとして `out` に書く。`to_json`・
/// `json_array`・`json_object` の値部分がこれを共有する。NULL は JSON の
/// `null` になる（コンテナの要素としての規約。`to_json(NULL)` 自体が
/// SQL NULL になるのは、この関数を呼ぶ前の既定の NULL 伝播で処理される）。
///
/// 対応する型は `resolve` の `json_encodable` が絞っているので、想定外の
/// 論理型が来ても（バグでもパニックはしたくないので）安全側に `null` を書く。
fn write_json_scalar(v: &Vector, row: usize, out: &mut Vec<u8>) {
    if !v.is_valid(row) {
        out.extend_from_slice(b"null");
        return;
    }
    match v.ty() {
        Ty::Boolean => {
            out.extend_from_slice(if v.bools().get(row) { b"true" } else { b"false" });
        }
        // すでに妥当な JSON テキストのはずなので、そのまま埋め込む。
        Ty::Json => out.extend_from_slice(v.bytes().get(row)),
        Ty::Varchar => crate::json::write_json_string(v.bytes().get(row), out),
        Ty::Date => {
            out.push(b'"');
            fmt_date(v.i32s()[row] as i64, out);
            out.push(b'"');
        }
        Ty::Time => {
            out.push(b'"');
            fmt_time(v.i64s()[row], out);
            out.push(b'"');
        }
        Ty::Timestamp => {
            out.push(b'"');
            fmt_timestamp(v.i64s()[row], out);
            out.push(b'"');
        }
        Ty::Float | Ty::Double => {
            let x = v.f64s()[row];
            if x.is_finite() {
                kernels::fmt_f64(x, out);
            } else {
                // NaN/Infinity は JSON に表現が無いので null にする。
                out.extend_from_slice(b"null");
            }
        }
        t => {
            // 整数系・DECIMAL。`fmt_int` は符号なし絶対値 + scale で書く。
            let scale = match t {
                Ty::Decimal { scale, .. } => scale,
                _ => 0,
            };
            match v.data() {
                Data::I32(d) => {
                    kernels::fmt_int(d[row].unsigned_abs() as u128, d[row] < 0, scale, out)
                }
                Data::I64(d) => {
                    kernels::fmt_int(d[row].unsigned_abs() as u128, d[row] < 0, scale, out)
                }
                Data::I128(d) => kernels::fmt_int(d[row].unsigned_abs(), d[row] < 0, scale, out),
                _ => out.extend_from_slice(b"null"),
            }
        }
    }
}

/// `json_array`/`list_value`。各引数を要素として直列化する。引数どうしの
/// 型を揃える必要はない（`json_array(1, 'x', true)` のような混在を許す）。
fn json_array_build(args: &[&Vector]) -> Result<Vector> {
    let (n, s) = strides(args)?;
    let mut out = BytesData::with_capacity(n, n * 8);
    let mut buf = Vec::new();
    for i in 0..n {
        buf.clear();
        buf.push(b'[');
        for (k, a) in args.iter().enumerate() {
            if k > 0 {
                buf.push(b',');
            }
            write_json_scalar(a, i * s[k], &mut buf);
        }
        buf.push(b']');
        ensure!(out.data.len() + buf.len() <= u32::MAX as usize, LimitExceeded);
        out.push(&buf);
    }
    Ok(Vector::from_data(Ty::Json, Data::Bytes(out), None))
}

/// `json_object(key1, val1, key2, val2, ...)`。キーは `resolve` が
/// VARCHAR へ変換済み。NULL キーは意味のある既定が無いので空文字列キーに
/// 落とす（他の値と同じく `null` にはしない: JSON のキーは文字列でなければ
/// ならないため）。
fn json_object_build(args: &[&Vector]) -> Result<Vector> {
    let (n, s) = strides(args)?;
    let mut out = BytesData::with_capacity(n, n * 16);
    let mut buf = Vec::new();
    for i in 0..n {
        buf.clear();
        buf.push(b'{');
        for (pi, pair) in args.chunks(2).enumerate() {
            if pi > 0 {
                buf.push(b',');
            }
            let (key_v, val_v) = (pair[0], pair[1]);
            let kj = i * s[pi * 2];
            let vj = i * s[pi * 2 + 1];
            if key_v.is_valid(kj) {
                crate::json::write_json_string(key_v.bytes().get(kj), &mut buf);
            } else {
                buf.extend_from_slice(b"\"\"");
            }
            buf.push(b':');
            write_json_scalar(val_v, vj, &mut buf);
        }
        buf.push(b'}');
        ensure!(out.data.len() + buf.len() <= u32::MAX as usize, LimitExceeded);
        out.push(&buf);
    }
    Ok(Vector::from_data(Ty::Json, Data::Bytes(out), None))
}

// =========================================================================
// 文字列（Bytes 出力）
// =========================================================================

/// UTF-8 のコードポイント数。継続バイトを数えないだけ。
fn cp_count(s: &[u8]) -> usize {
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
fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    if needle.len() > hay.len() {
        return None;
    }
    (0..=hay.len() - needle.len()).find(|&i| &hay[i..i + needle.len()] == needle)
}

/// 1 行ぶんの文字列関数。`false` を返した行は NULL。
fn eval_str(id: FuncId, a: &A, out: &mut Vec<u8>) -> Result<bool> {
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
            let found = if a.n() >= 2 {
                crate::json::extract(a.bytes(0), a.bytes(1))?
            } else {
                Some(crate::json::whole(a.bytes(0))?)
            };
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
        F_MAP_EXTRACT => {
            return match crate::json::map_get(a.bytes(0), a.bytes(1))? {
                Some(span) => {
                    out.extend_from_slice(span);
                    Ok(true)
                }
                None => Ok(false),
            };
        }
        _ => err!(Internal),
    }
    Ok(true)
}

// =========================================================================
// Bool 出力
// =========================================================================

fn eval_bool(id: FuncId, a: &A) -> Result<Option<bool>> {
    let (s, p) = (a.bytes(0), a.bytes(1));
    Ok(Some(match id {
        F_STARTS_WITH => s.len() >= p.len() && &s[..p.len()] == p,
        F_ENDS_WITH => s.len() >= p.len() && &s[s.len() - p.len()..] == p,
        F_CONTAINS => find(s, p).is_some(),
        _ => err!(Internal),
    }))
}

// =========================================================================
// 整数出力
// =========================================================================

fn eval_int(id: FuncId, a: &A) -> Result<Option<i64>> {
    // year() などの略記。part 番号は ID に埋まっている。
    if id >= F_PART_BASE {
        return Ok(date_part((id - F_PART_BASE) as u8, a.int(0)));
    }
    Ok(match id {
        F_LENGTH => Some(cp_count(a.bytes(0)) as i64),
        F_STRPOS => {
            // 1 始まり、無ければ 0。位置はコードポイント単位。
            let (s, p) = (a.bytes(0), a.bytes(1));
            Some(match find(s, p) {
                Some(b) => cp_count(&s[..b]) as i64 + 1,
                None => 0,
            })
        }
        F_ABS_I => a.int(0).checked_abs(),
        F_SIGN_I => Some(a.int(0).signum()),
        F_ROUND_I => {
            let x = a.int(0);
            let d = if a.n() >= 2 { a.int(1) } else { 0 };
            if d >= 0 {
                Some(x)
            } else {
                round_int(x, -d)
            }
        }
        F_MOD_I => {
            let (x, y) = (a.int(0), a.int(1));
            // 0 除算と MIN % -1 は NULL（kernels の Mod と同じ判断）。
            if y == 0 || (y == -1 && x == i64::MIN) {
                None
            } else {
                Some(x % y)
            }
        }
        F_DATE_PART => match part_id(a.bytes(0)) {
            Some(p) => date_part(p, a.int(1)),
            None => err!(TypeMismatch),
        },
        F_DATE_DIFF => match part_id(a.bytes(0)) {
            Some(p) => date_diff(p, a.int(1), a.int(2))?,
            None => err!(TypeMismatch),
        },
        F_DATE_TRUNC => match part_id(a.bytes(0)) {
            Some(p) => date_trunc(p, a.int(1))?,
            None => err!(TypeMismatch),
        },
        F_DATE_ADD => match part_id(a.bytes(0)) {
            Some(p) => date_add(p, a.int(1), a.int(2))?,
            None => err!(TypeMismatch),
        },
        F_TO_DATE => parse_date(a.bytes(0)),
        F_TO_TIMESTAMP => parse_timestamp(a.bytes(0)),
        F_LAST_DAY => {
            let c = civil(a.int(0));
            Some(days_from_civil(c.y, c.mo, days_in_month(c.y, c.mo)))
        }
        F_JSON_ARRAY_LENGTH => {
            let found = if a.n() >= 2 {
                crate::json::extract(a.bytes(0), a.bytes(1))?
            } else {
                Some(crate::json::whole(a.bytes(0))?)
            };
            match found {
                Some((span, kind)) => Some(crate::json::array_length(span, kind)?),
                None => None,
            }
        }
        _ => err!(Internal),
    })
}

/// 10 の冪へ 0 から遠ざかる向きに丸める（`round(12345, -2)` → 12300）。
fn round_int(x: i64, k: i64) -> Option<i64> {
    if k > 18 {
        return Some(0);
    }
    let mut p: i64 = 1;
    for _ in 0..k {
        p = p.checked_mul(10)?;
    }
    let (q, r) = (x / p, x % p);
    let half = p / 2;
    let q = if r >= half {
        q.checked_add(1)?
    } else if r <= -half {
        q.checked_sub(1)?
    } else {
        q
    };
    q.checked_mul(p)
}

// =========================================================================
// 浮動小数出力
// =========================================================================

fn eval_f64(id: FuncId, a: &A) -> Result<Option<f64>> {
    let x = a.flt(0);
    Ok(match id {
        F_ABS_F => Some(f_abs(x)),
        F_SIGN_F => Some(if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else {
            x
        }),
        F_ROUND_F => {
            let d = if a.n() >= 2 { a.int(1) } else { 0 };
            Some(round_f64(x, d))
        }
        F_CEIL_F => Some(f_ceil(x)),
        F_FLOOR_F => Some(f_floor(x)),
        F_TRUNC_F => Some(f_trunc(x)),
        // DuckDB は定義域外をエラーにするが、ここは 0 除算と同じく NULL に
        // する（式の途中で失敗してもクエリ全体を落とさない）。
        F_SQRT => {
            if x < 0.0 {
                None
            } else {
                Some(f_sqrt(x))
            }
        }
        F_EXP => Some(f_exp(x)),
        F_LN => {
            if x <= 0.0 {
                None
            } else {
                Some(f_ln(x))
            }
        }
        F_LOG10 => {
            if x <= 0.0 {
                None
            } else {
                Some(f_ln(x) / core::f64::consts::LN_10)
            }
        }
        F_POW => Some(f_pow(x, a.flt(1))),
        F_MOD_F => Some(x % a.flt(1)),
        _ => err!(Internal),
    })
}

fn f_abs(x: f64) -> f64 {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

/// 0 方向への切り捨て。`f64::trunc` は std 専用なので自前で持つ。
fn f_trunc(x: f64) -> f64 {
    if f_abs(x) < 9_223_372_036_854_775_808.0 {
        (x as i64) as f64
    } else {
        x
    }
}

fn f_floor(x: f64) -> f64 {
    let t = f_trunc(x);
    if t > x {
        t - 1.0
    } else {
        t
    }
}

fn f_ceil(x: f64) -> f64 {
    let t = f_trunc(x);
    if t < x {
        t + 1.0
    } else {
        t
    }
}

/// 0 から遠ざかる丸め。DuckDB の `round` はこちらで、キャストの銀行丸め
/// (`kernels::f_round`) とは規則が違う（`round(2.5)` = 3、`round(3.5)` = 4）。
fn round_half_up(x: f64) -> f64 {
    let t = f_trunc(x);
    let frac = x - t;
    if frac >= 0.5 {
        t + 1.0
    } else if frac <= -0.5 {
        t - 1.0
    } else {
        t
    }
}

/// `round(x, d)`。DuckDB と同じく 10^d を掛けて丸め、割り戻す。
fn round_f64(x: f64, d: i64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    if d == 0 {
        return round_half_up(x);
    }
    let m = pow10(d.unsigned_abs().min(308) as u32);
    let r = if d > 0 { round_half_up(x * m) / m } else { round_half_up(x / m) * m };
    if r.is_finite() {
        r
    } else {
        x
    }
}

fn pow10(k: u32) -> f64 {
    let mut r = 1.0f64;
    for _ in 0..k {
        r *= 10.0;
    }
    r
}

/// 平方根。`core` に `f64::sqrt` は無い（libm 側）。指数部を半分にした
/// 初期値から Newton 法で 5 回反復すれば倍精度で厳密になる。
fn f_sqrt(x: f64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let mut y = f64::from_bits((x.to_bits() + (1023u64 << 52)) >> 1);
    for _ in 0..5 {
        y = 0.5 * (y + x / y);
    }
    y
}

/// 自然対数。`x = m * 2^e`（`m ∈ [√2/2, √2)`）に分解し、
/// `ln(m) = 2*atanh((m-1)/(m+1))` の級数で求める。|z| <= 0.1716、
/// z² <= 0.0295 なので、16 項で打ち切り誤差が 1e-22 まで落ちる。
fn f_ln(x: f64) -> f64 {
    let mut bits = x.to_bits();
    let mut e = 0i32;
    // 非正規化数は 2^64 倍してから扱う。
    if (bits >> 52) & 0x7ff == 0 {
        bits = (x * 18_446_744_073_709_551_616.0).to_bits();
        e -= 64;
    }
    e += ((bits >> 52) & 0x7ff) as i32 - 1023;
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | (1023u64 << 52));
    if m > core::f64::consts::SQRT_2 {
        m *= 0.5;
        e += 1;
    }
    let z = (m - 1.0) / (m + 1.0);
    let z2 = z * z;
    let mut s = 0.0f64;
    for k in (0..16).rev() {
        s = s * z2 + 1.0 / (2 * k + 1) as f64;
    }
    2.0 * z * s + e as f64 * core::f64::consts::LN_2
}

/// `2^k` を掛ける。指数部が一度に振り切れないよう 2 回に分ける。
fn scale2(m: f64, k: i32) -> f64 {
    let p = |e: i32| f64::from_bits(((e + 1023) as u64) << 52);
    let k1 = k.clamp(-700, 700);
    m * p(k1) * p(k - k1)
}

/// 指数関数。`x = k*ln2 + r` に分け、`exp(r)` を Taylor 展開（14 項）で
/// 求めて `2^k` を掛ける。|r| <= ln2/2 なので 14 項で倍精度に届く。
fn f_exp(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x > 709.8 {
        return f64::INFINITY;
    }
    if x < -745.2 {
        return 0.0;
    }
    let k = round_half_up(x / core::f64::consts::LN_2);
    // ln2 を上位・下位に分けて引くと、大きな x でも桁落ちしない。
    const LN2_HI: f64 = 0.693_147_180_369_123_8;
    const LN2_LO: f64 = 1.908_214_929_270_587_7e-10;
    let r = (x - k * LN2_HI) - k * LN2_LO;
    let mut s = 1.0f64;
    for i in (1..=14).rev() {
        s = 1.0 + r * s / i as f64;
    }
    scale2(s, k as i32)
}

/// べき乗。整数指数は繰り返し二乗で厳密に出す（`pow(2,3)` を
/// 7.999… にしないため）。それ以外は `exp(y * ln x)`。
fn f_pow(x: f64, y: f64) -> f64 {
    if y == 0.0 {
        return 1.0;
    }
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    let n = f_trunc(y);
    if n == y && f_abs(y) <= 1024.0 {
        let mut r = 1.0f64;
        let mut b = if n < 0.0 { 1.0 / x } else { x };
        let mut k = f_abs(n) as u32;
        while k > 0 {
            if k & 1 == 1 {
                r *= b;
            }
            b *= b;
            k >>= 1;
        }
        return r;
    }
    if x < 0.0 {
        // 負の底に非整数の指数は実数解を持たない。
        return f64::NAN;
    }
    if x == 0.0 {
        return if y < 0.0 { f64::INFINITY } else { 0.0 };
    }
    f_exp(y * f_ln(x))
}

// =========================================================================
// 暦
// =========================================================================

/// 暦日。`civil_from_days` の結果と時刻部分。
struct Civil {
    y: i64,
    mo: u32,
    d: u32,
    /// 深夜からのマイクロ秒。
    tod: i64,
    /// エポックからの日数。
    days: i64,
}

/// エポックからの日数 → `(年, 月, 日)`。Howard Hinnant の civil_from_days。
/// 表を持たないので、うるう年もうるう世紀も式だけで処理できる。
pub(crate) fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// `(年, 月, 日)` → エポックからの日数。civil_from_days の逆。
pub(crate) fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = y - if m <= 2 { 1 } else { 0 };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = if m > 2 { m as i64 - 3 } else { m as i64 + 9 };
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// TIMESTAMP（マイクロ秒）を暦に分解する。エポック以前でも 1 日ずれない
/// よう、除算はすべて床除算 (`div_euclid`) を使う。
fn civil(us: i64) -> Civil {
    let days = us.div_euclid(US_PER_DAY);
    let (y, mo, d) = civil_from_days(days);
    Civil { y, mo, d, tod: us.rem_euclid(US_PER_DAY), days }
}

fn is_leap(y: i64) -> bool {
    (y.rem_euclid(4) == 0 && y.rem_euclid(100) != 0) || y.rem_euclid(400) == 0
}

fn days_in_month(y: i64, m: u32) -> u32 {
    const N: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let i = (m.clamp(1, 12) - 1) as usize;
    if i == 1 && is_leap(y) {
        29
    } else {
        N[i]
    }
}

/// 曜日。0 = 日曜（DuckDB の `dow` と同じ）。1970-01-01 は木曜。
fn weekday(days: i64) -> i64 {
    (days + 4).rem_euclid(7)
}

/// ISO 8601 の週番号。その日を含む週の木曜が属する年の第何週か。
fn iso_week(days: i64) -> i64 {
    let iso_dow = match weekday(days) {
        0 => 7,
        w => w,
    };
    let thursday = days + (4 - iso_dow);
    let (y, _, _) = civil_from_days(thursday);
    (thursday - days_from_civil(y, 1, 1)) / 7 + 1
}

/// `date_part` / `year()` などの本体。
fn date_part(p: u8, us: i64) -> Option<i64> {
    let c = civil(us);
    Some(match p {
        P_YEAR => c.y,
        P_QUARTER => (c.mo as i64 + 2) / 3,
        P_MONTH => c.mo as i64,
        P_WEEK => iso_week(c.days),
        P_DAY => c.d as i64,
        P_HOUR => c.tod / US_PER_HOUR,
        P_MINUTE => c.tod / US_PER_MIN % 60,
        P_SECOND => c.tod / US_PER_SEC % 60,
        P_DOW => weekday(c.days),
        P_DOY => c.days - days_from_civil(c.y, 1, 1) + 1,
        // DuckDB は小数秒込みの DOUBLE を返すが、ここは BIGINT 秒（床）。
        _ => us.div_euclid(US_PER_SEC),
    })
}

/// `date_trunc`。切り下げは常に床方向なので、エポック以前でも
/// 「1 単位ずれる」ことが無い。
fn date_trunc(p: u8, us: i64) -> Result<Option<i64>> {
    let c = civil(us);
    let day = |d: i64| d.checked_mul(US_PER_DAY);
    let unit = |u: i64| Some(us.div_euclid(u) * u);
    Ok(match p {
        P_YEAR => day(days_from_civil(c.y, 1, 1)),
        P_QUARTER => day(days_from_civil(c.y, (c.mo - 1) / 3 * 3 + 1, 1)),
        P_MONTH => day(days_from_civil(c.y, c.mo, 1)),
        // 週の始まりは月曜（DuckDB と同じ）。
        P_WEEK => day(c.days - (weekday(c.days) + 6).rem_euclid(7)),
        P_DAY => day(c.days),
        P_HOUR => unit(US_PER_HOUR),
        P_MINUTE => unit(US_PER_MIN),
        P_SECOND => unit(US_PER_SEC),
        _ => err!(TypeMismatch),
    })
}

/// `date_diff`。DuckDB と同じく「境界をいくつ跨いだか」を数える
/// （`date_diff('day', '..23:00', '..01:00')` は 1）。
fn date_diff(p: u8, a: i64, b: i64) -> Result<Option<i64>> {
    let (ca, cb) = (civil(a), civil(b));
    let unit = |u: i64| b.div_euclid(u) - a.div_euclid(u);
    Ok(Some(match p {
        P_YEAR => cb.y - ca.y,
        P_QUARTER => (cb.y * 4 + (cb.mo as i64 - 1) / 3) - (ca.y * 4 + (ca.mo as i64 - 1) / 3),
        P_MONTH => (cb.y * 12 + cb.mo as i64) - (ca.y * 12 + ca.mo as i64),
        // 週差は日差の 7 分割（DuckDB もこの定義）。
        P_WEEK => (cb.days - ca.days) / 7,
        P_DAY => cb.days - ca.days,
        P_HOUR => unit(US_PER_HOUR),
        P_MINUTE => unit(US_PER_MIN),
        P_SECOND => unit(US_PER_SEC),
        P_EPOCH => unit(US_PER_SEC),
        _ => err!(TypeMismatch),
    }))
}

/// `date_add(part, n, ts)`。DuckDB の `date_add` は INTERVAL を取るが、
/// この実装に INTERVAL 型が無いための意図的な非互換（resolve のコメント参照）。
///
/// 年・四半期・月の加算は日を月末で切り詰める（`2024-01-31` + 1 か月 =
/// `2024-02-29`）。DuckDB と同じ規則。
fn date_add(p: u8, k: i64, us: i64) -> Result<Option<i64>> {
    let c = civil(us);
    let months = |m: i64| -> Option<i64> {
        let t = (c.y * 12 + c.mo as i64 - 1).checked_add(m)?;
        let (y, mo) = (t.div_euclid(12), t.rem_euclid(12) as u32 + 1);
        let d = c.d.min(days_in_month(y, mo));
        days_from_civil(y, mo, d).checked_mul(US_PER_DAY)?.checked_add(c.tod)
    };
    let unit = |u: i64| k.checked_mul(u).and_then(|d| us.checked_add(d));
    Ok(match p {
        P_YEAR => k.checked_mul(12).and_then(months),
        P_QUARTER => k.checked_mul(3).and_then(months),
        P_MONTH => months(k),
        P_WEEK => unit(7 * US_PER_DAY),
        P_DAY => unit(US_PER_DAY),
        P_HOUR => unit(US_PER_HOUR),
        P_MINUTE => unit(US_PER_MIN),
        P_SECOND => unit(US_PER_SEC),
        _ => err!(TypeMismatch),
    })
}

/// TIMESTAMP（マイクロ秒）に INTERVAL の 3 成分を足す。`expr::kernels::
/// ts_add_interval` から呼ぶ。月は `date_add` と同じ規則（月末を超えない
/// 日に切り詰める）、日とマイクロ秒は単純加算でよい（繰り上がりは
/// タイムスタンプの桁に自然に現れる）。
pub(crate) fn add_interval_to_ts(us: i64, months: i32, days: i32, micros: i64) -> Option<i64> {
    let c = civil(us);
    let t = (c.y * 12 + c.mo as i64 - 1).checked_add(months as i64)?;
    let (y, mo) = (t.div_euclid(12), t.rem_euclid(12) as u32 + 1);
    let d = c.d.min(days_in_month(y, mo));
    let total_days = days_from_civil(y, mo, d).checked_add(days as i64)?;
    let ts = total_days.checked_mul(US_PER_DAY)?.checked_add(c.tod)?;
    ts.checked_add(micros)
}

// =========================================================================
// 日時の入出力（kernels のキャストからも使う）
// =========================================================================

/// 右詰めゼロ埋めの 10 進表記。負なら先頭に `-`。
fn pad(v: i64, w: usize, out: &mut Vec<u8>) {
    let mut buf = [0u8; 24];
    let mut u = v.unsigned_abs();
    let mut k = 0usize;
    loop {
        buf[k] = b'0' + (u % 10) as u8;
        u /= 10;
        k += 1;
        if u == 0 || k == buf.len() {
            break;
        }
    }
    if v < 0 {
        out.push(b'-');
    }
    for _ in k..w {
        out.push(b'0');
    }
    for i in (0..k).rev() {
        out.push(buf[i]);
    }
}

/// `YYYY-MM-DD`。
pub(crate) fn fmt_date(days: i64, out: &mut Vec<u8>) {
    let (y, m, d) = civil_from_days(days);
    pad(y, 4, out);
    out.push(b'-');
    pad(m as i64, 2, out);
    out.push(b'-');
    pad(d as i64, 2, out);
}

/// `HH:MM:SS[.ffffff]`。小数部は末尾の 0 を落とす（DuckDB と同じ）。
pub(crate) fn fmt_time(us: i64, out: &mut Vec<u8>) {
    let t = us.rem_euclid(US_PER_DAY);
    pad(t / US_PER_HOUR, 2, out);
    out.push(b':');
    pad(t / US_PER_MIN % 60, 2, out);
    out.push(b':');
    pad(t / US_PER_SEC % 60, 2, out);
    let frac = t % US_PER_SEC;
    if frac != 0 {
        let mut buf = Vec::new();
        pad(frac, 6, &mut buf);
        while buf.last() == Some(&b'0') {
            buf.pop();
        }
        out.push(b'.');
        out.extend_from_slice(&buf);
    }
}

/// `YYYY-MM-DD HH:MM:SS[.ffffff]`。
pub(crate) fn fmt_timestamp(us: i64, out: &mut Vec<u8>) {
    fmt_date(us.div_euclid(US_PER_DAY), out);
    out.push(b' ');
    fmt_time(us.rem_euclid(US_PER_DAY), out);
}

/// `%Y %m %d %H %M %S %%` のみ解釈する。未知の指定子は `%` ごとそのまま出す。
fn strftime(us: i64, f: &[u8], out: &mut Vec<u8>) {
    let c = civil(us);
    let mut i = 0usize;
    while i < f.len() {
        if f[i] != b'%' || i + 1 >= f.len() {
            out.push(f[i]);
            i += 1;
            continue;
        }
        i += 1;
        match f[i] {
            b'Y' => pad(c.y, 4, out),
            b'm' => pad(c.mo as i64, 2, out),
            b'd' => pad(c.d as i64, 2, out),
            b'H' => pad(c.tod / US_PER_HOUR, 2, out),
            b'M' => pad(c.tod / US_PER_MIN % 60, 2, out),
            b'S' => pad(c.tod / US_PER_SEC % 60, 2, out),
            b'%' => out.push(b'%'),
            x => {
                out.push(b'%');
                out.push(x);
            }
        }
        i += 1;
    }
}

/// 読み取り位置つきの数値スキャン。`lo..=hi` 桁を読み、値と次の位置を返す。
fn scan(s: &[u8], i: usize, lo: usize, hi: usize) -> Option<(i64, usize)> {
    let mut v = 0i64;
    let mut k = 0usize;
    let mut j = i;
    while j < s.len() && k < hi && s[j].is_ascii_digit() {
        v = v * 10 + (s[j] - b'0') as i64;
        j += 1;
        k += 1;
    }
    if k < lo {
        None
    } else {
        Some((v, j))
    }
}

fn trim_ws(s: &[u8]) -> &[u8] {
    let mut lo = 0;
    let mut hi = s.len();
    while lo < hi && (s[lo] == b' ' || s[lo] == b'\t') {
        lo += 1;
    }
    while hi > lo && (s[hi - 1] == b' ' || s[hi - 1] == b'\t') {
        hi -= 1;
    }
    &s[lo..hi]
}

/// `YYYY-MM-DD` を読む。読めた位置も返す（TIMESTAMP から使うため）。
fn scan_date(s: &[u8]) -> Option<(i64, usize)> {
    let neg = s.first() == Some(&b'-');
    let i = usize::from(neg);
    let (y, i) = scan(s, i, 1, 6)?;
    if s.get(i) != Some(&b'-') {
        return None;
    }
    let (m, i) = scan(s, i + 1, 1, 2)?;
    if s.get(i) != Some(&b'-') {
        return None;
    }
    let (d, i) = scan(s, i + 1, 1, 2)?;
    let y = if neg { -y } else { y };
    // 範囲外の月日は読み取り失敗（＝その行は NULL）。
    if !(1..=12).contains(&m) || d < 1 || d > days_in_month(y, m as u32) as i64 {
        return None;
    }
    Some((days_from_civil(y, m as u32, d as u32), i))
}

/// `HH:MM[:SS[.ffffff]]` を読む。
fn scan_time(s: &[u8], i: usize) -> Option<(i64, usize)> {
    let (h, i) = scan(s, i, 1, 2)?;
    if s.get(i) != Some(&b':') {
        return None;
    }
    let (m, i) = scan(s, i + 1, 1, 2)?;
    let (sec, mut i) = if s.get(i) == Some(&b':') { scan(s, i + 1, 1, 2)? } else { (0, i) };
    let mut frac = 0i64;
    if s.get(i) == Some(&b'.') {
        let mut k = 0usize;
        i += 1;
        // 7 桁目以降は切り捨てる（内部表現がマイクロ秒のため）。
        while i < s.len() && s[i].is_ascii_digit() {
            if k < 6 {
                frac = frac * 10 + (s[i] - b'0') as i64;
                k += 1;
            }
            i += 1;
        }
        if k == 0 {
            return None;
        }
        while k < 6 {
            frac *= 10;
            k += 1;
        }
    }
    if h > 24 || m > 59 || sec > 59 {
        return None;
    }
    Some((h * US_PER_HOUR + m * US_PER_MIN + sec * US_PER_SEC + frac, i))
}

/// VARCHAR → DATE。読めなければ `None`（呼び出し側でその行を NULL にする）。
pub(crate) fn parse_date(s: &[u8]) -> Option<i64> {
    let s = trim_ws(s);
    let (d, i) = scan_date(s)?;
    if i == s.len() {
        Some(d)
    } else {
        None
    }
}

/// VARCHAR → TIMESTAMP。`YYYY-MM-DD[ T]HH:MM[:SS[.ffffff]]`。
/// 日付だけなら深夜 0 時とみなす。
pub(crate) fn parse_timestamp(s: &[u8]) -> Option<i64> {
    let s = trim_ws(s);
    let (d, i) = scan_date(s)?;
    let base = d.checked_mul(US_PER_DAY)?;
    if i == s.len() {
        return Some(base);
    }
    if s[i] != b' ' && s[i] != b'T' && s[i] != b't' {
        return None;
    }
    let (t, j) = scan_time(s, i + 1)?;
    if j != s.len() {
        return None;
    }
    base.checked_add(t)
}

/// VARCHAR → TIME。
pub(crate) fn parse_time(s: &[u8]) -> Option<i64> {
    let s = trim_ws(s);
    let (t, i) = scan_time(s, 0)?;
    if i == s.len() {
        Some(t)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::code_of;
    use crate::vector::Value;

    // --- 組み立てヘルパ -----------------------------------------------------

    fn vs(vals: &[Option<&str>]) -> Vector {
        let mut v = Vector::new(Ty::Varchar);
        for x in vals {
            match x {
                Some(s) => v.push_value(&Value::Bytes(s.as_bytes().to_vec())),
                None => v.push_null(),
            }
        }
        v
    }

    fn vi(ty: Ty, vals: &[Option<i64>]) -> Vector {
        let mut v = Vector::new(ty);
        for x in vals {
            match x {
                Some(n) if ty.phys() == PhysType::I32 => v.push_value(&Value::I32(*n as i32)),
                Some(n) => v.push_value(&Value::I64(*n)),
                None => v.push_null(),
            }
        }
        v
    }

    fn vf(vals: &[Option<f64>]) -> Vector {
        let mut v = Vector::new(Ty::Double);
        for x in vals {
            match x {
                Some(n) => v.push_value(&Value::F64(*n)),
                None => v.push_null(),
            }
        }
        v
    }

    /// `resolve` して `call` する。引数はすでに解決後の型で渡す。
    fn run(name: &str, args: &[&Vector]) -> Result<Vector> {
        let tys: Vec<Ty> = args.iter().map(|a| a.ty()).collect();
        let (id, want, ret) = resolve(name, &tys)?;
        assert_eq!(want.len(), args.len());
        for (i, w) in want.iter().enumerate() {
            assert_eq!(*w, tys[i], "{name}: 引数 {i} の型が違う");
        }
        call(id, ret, args)
    }

    fn str_at(v: &Vector, i: usize) -> Option<String> {
        if !v.is_valid(i) {
            return None;
        }
        Some(String::from_utf8(v.bytes().get(i).to_vec()).unwrap())
    }

    fn int_at(v: &Vector, i: usize) -> Option<i64> {
        if !v.is_valid(i) {
            return None;
        }
        match v.data() {
            Data::I32(d) => Some(d[i] as i64),
            Data::I64(d) => Some(d[i]),
            _ => None,
        }
    }

    fn flt_at(v: &Vector, i: usize) -> Option<f64> {
        if v.is_valid(i) {
            Some(v.f64s()[i])
        } else {
            None
        }
    }

    /// 文字列 1 引数（+ 追加引数）の結果を 1 行ぶん取り出す近道。
    fn s1(name: &str, s: Option<&str>) -> Option<String> {
        str_at(&run(name, &[&vs(&[s])]).unwrap(), 0)
    }

    fn ts(lit: &str) -> Vector {
        vi(Ty::Timestamp, &[Some(parse_timestamp(lit.as_bytes()).unwrap())])
    }

    fn part(name: &str, p: &str, lit: &str) -> Option<i64> {
        int_at(&run(name, &[&vs(&[Some(p)]), &ts(lit)]).unwrap(), 0)
    }

    // --- 文字列 -------------------------------------------------------------

    #[test]
    fn string_basics() {
        assert_eq!(s1("upper", Some("aBc")).as_deref(), Some("ABC"));
        assert_eq!(s1("lower", Some("aBc")).as_deref(), Some("abc"));
        assert_eq!(s1("upper", Some("")).as_deref(), Some(""));
        assert_eq!(s1("upper", None), None);
        assert_eq!(s1("reverse", Some("abc")).as_deref(), Some("cba"));
        // reverse はコードポイント単位（duckdb: reverse('あいう') = 'ういあ'）。
        assert_eq!(s1("reverse", Some("あいう")).as_deref(), Some("ういあ"));
        assert_eq!(s1("reverse", Some("")).as_deref(), Some(""));
        // duckdb: trim('  ab  ') = 'ab'、既定はスペースのみ（タブは残る）。
        assert_eq!(s1("trim", Some("  ab  ")).as_deref(), Some("ab"));
        assert_eq!(s1("trim", Some("\tab\t")).as_deref(), Some("\tab\t"));
        assert_eq!(s1("ltrim", Some("  ab  ")).as_deref(), Some("ab  "));
        assert_eq!(s1("rtrim", Some("  ab  ")).as_deref(), Some("  ab"));
    }

    #[test]
    fn trim_with_charset() {
        let f = |s: &str, c: &str, name: &str| {
            str_at(&run(name, &[&vs(&[Some(s)]), &vs(&[Some(c)])]).unwrap(), 0)
        };
        // duckdb: ltrim('xxabxx','x')='abxx', rtrim(..)='xxab', trim('xyabyx','xy')='ab'
        assert_eq!(f("xxabxx", "x", "ltrim").as_deref(), Some("abxx"));
        assert_eq!(f("xxabxx", "x", "rtrim").as_deref(), Some("xxab"));
        assert_eq!(f("xyabyx", "xy", "trim").as_deref(), Some("ab"));
        assert_eq!(f("abc", "", "trim").as_deref(), Some("abc"));
        assert_eq!(f("", "x", "trim").as_deref(), Some(""));
    }

    #[test]
    fn length_and_strpos_are_codepoint_based() {
        // duckdb: length('あいう') = 3, strpos('あいう','い') = 2
        assert_eq!(int_at(&run("length", &[&vs(&[Some("あいう")])]).unwrap(), 0), Some(3));
        assert_eq!(int_at(&run("length", &[&vs(&[Some("")])]).unwrap(), 0), Some(0));
        assert_eq!(int_at(&run("length", &[&vs(&[None])]).unwrap(), 0), None);
        let p = |s: &str, n: &str| {
            int_at(&run("strpos", &[&vs(&[Some(s)]), &vs(&[Some(n)])]).unwrap(), 0)
        };
        assert_eq!(p("あいう", "い"), Some(2));
        assert_eq!(p("abc", "b"), Some(2));
        assert_eq!(p("abc", "z"), Some(0));
        // duckdb: strpos('abc','') = 1
        assert_eq!(p("abc", ""), Some(1));
        assert_eq!(p("", "a"), Some(0));
    }

    #[test]
    fn substr_edges_match_duckdb() {
        let f = |s: &str, a: i64, b: Option<i64>| {
            let sv = vs(&[Some(s)]);
            let av = vi(Ty::BigInt, &[Some(a)]);
            let out = match b {
                Some(b) => run("substr", &[&sv, &av, &vi(Ty::BigInt, &[Some(b)])]),
                None => run("substr", &[&sv, &av]),
            };
            str_at(&out.unwrap(), 0).unwrap()
        };
        // すべて duckdb v1.4 の実測値。
        assert_eq!(f("hello", 2, None), "ello");
        assert_eq!(f("hello", 2, Some(3)), "ell");
        assert_eq!(f("hello", -2, None), "lo");
        assert_eq!(f("hello", -2, Some(3)), "lo");
        assert_eq!(f("hello", 0, None), "hello");
        assert_eq!(f("hello", 0, Some(3)), "he");
        assert_eq!(f("hello", 10, None), "");
        assert_eq!(f("hello", 2, Some(0)), "");
        assert_eq!(f("hello", 2, Some(100)), "ello");
        assert_eq!(f("hello", -10, Some(3)), "");
        assert_eq!(f("hello", 2, Some(-1)), "h");
        assert_eq!(f("", 1, Some(3)), "");
        // duckdb: substr('あいうえお',2,2) = 'いう'（コードポイント単位）
        assert_eq!(f("あいうえお", 2, Some(2)), "いう");
        // NULL 引数は NULL。
        let n = run("substr", &[&vs(&[None]), &vi(Ty::BigInt, &[Some(1)])]).unwrap();
        assert_eq!(str_at(&n, 0), None);
    }

    #[test]
    fn replace_repeat_split_pad() {
        let s3 = |name: &str, a: &str, b: &str, c: &str| {
            str_at(&run(name, &[&vs(&[Some(a)]), &vs(&[Some(b)]), &vs(&[Some(c)])]).unwrap(), 0)
                .unwrap()
        };
        assert_eq!(s3("replace", "aaa", "a", "bb"), "bbbbbb");
        assert_eq!(s3("replace", "abc", "", "x"), "abc");
        assert_eq!(s3("replace", "", "a", "b"), "");

        let pad = |name: &str, s: &str, n: i64, p: &str| {
            str_at(
                &run(name, &[&vs(&[Some(s)]), &vi(Ty::BigInt, &[Some(n)]), &vs(&[Some(p)])])
                    .unwrap(),
                0,
            )
            .unwrap()
        };
        assert_eq!(pad("lpad", "ab", 5, "xy"), "xyxab");
        assert_eq!(pad("rpad", "ab", 5, "xy"), "abxyx");
        assert_eq!(pad("lpad", "abcdef", 3, "x"), "abc");
        assert_eq!(pad("lpad", "ab", 0, "x"), "");
        // duckdb: lpad('あ',3,'x') = 'xxあ'（コードポイント単位）
        assert_eq!(pad("lpad", "あ", 3, "x"), "xxあ");

        let sp = |s: &str, d: &str, k: i64| {
            str_at(
                &run(
                    "split_part",
                    &[&vs(&[Some(s)]), &vs(&[Some(d)]), &vi(Ty::BigInt, &[Some(k)])],
                )
                .unwrap(),
                0,
            )
            .unwrap()
        };
        assert_eq!(sp("a,b,c", ",", 2), "b");
        assert_eq!(sp("a,b,c", ",", 0), "");
        assert_eq!(sp("a,b,c", ",", 5), "");
        assert_eq!(sp("a,b,c", ",", -1), "c");
        assert_eq!(sp("a,b,c", ",", 1), "a");
        assert_eq!(sp("", ",", 1), "");

        let rp = |s: &str, k: i64| {
            str_at(&run("repeat", &[&vs(&[Some(s)]), &vi(Ty::BigInt, &[Some(k)])]).unwrap(), 0)
                .unwrap()
        };
        assert_eq!(rp("ab", 3), "ababab");
        assert_eq!(rp("ab", 0), "");
        assert_eq!(rp("ab", -1), "");
    }

    #[test]
    fn concat_ignores_nulls() {
        // duckdb: concat('a', NULL, 'b') = 'ab'
        let out = run("concat", &[&vs(&[Some("a")]), &vs(&[None]), &vs(&[Some("b")])]).unwrap();
        assert_eq!(str_at(&out, 0).as_deref(), Some("ab"));
        let out = run("concat", &[&vs(&[None]), &vs(&[None])]).unwrap();
        assert_eq!(str_at(&out, 0).as_deref(), Some(""));
    }

    #[test]
    fn prefix_suffix_contains() {
        let f = |name: &str, s: &str, p: &str| {
            let v = run(name, &[&vs(&[Some(s)]), &vs(&[Some(p)])]).unwrap();
            v.is_valid(0).then(|| v.bools().get(0))
        };
        assert_eq!(f("starts_with", "abc", "ab"), Some(true));
        assert_eq!(f("starts_with", "abc", "b"), Some(false));
        // duckdb: starts_with('abc','') = true, contains('abc','') = true
        assert_eq!(f("starts_with", "abc", ""), Some(true));
        assert_eq!(f("ends_with", "abc", "bc"), Some(true));
        assert_eq!(f("ends_with", "abc", ""), Some(true));
        assert_eq!(f("contains", "abc", "b"), Some(true));
        assert_eq!(f("contains", "abc", ""), Some(true));
        assert_eq!(f("contains", "", "a"), Some(false));
        let n = run("contains", &[&vs(&[None]), &vs(&[Some("a")])]).unwrap();
        assert!(!n.is_valid(0));
    }

    // --- 数値 ---------------------------------------------------------------

    #[test]
    fn numeric_integer_domain() {
        let f = |name: &str, x: i64| int_at(&run(name, &[&vi(Ty::BigInt, &[Some(x)])]).unwrap(), 0);
        assert_eq!(f("abs", -3), Some(3));
        assert_eq!(f("abs", 0), Some(0));
        assert_eq!(f("sign", -5), Some(-1));
        assert_eq!(f("sign", 0), Some(0));
        assert_eq!(f("ceil", 3), Some(3));
        assert_eq!(f("floor", 3), Some(3));
        assert_eq!(f("trunc", 3), Some(3));
        assert_eq!(f("round", 3), Some(3));
        assert_eq!(int_at(&run("abs", &[&vi(Ty::BigInt, &[None])]).unwrap(), 0), None);
        // duckdb: round(12345,-2)=12300, round(-12345,-2)=-12300, round(15,-1)=20, round(25,-1)=30
        let r = |x: i64, d: i64| {
            int_at(
                &run("round", &[&vi(Ty::BigInt, &[Some(x)]), &vi(Ty::BigInt, &[Some(d)])]).unwrap(),
                0,
            )
        };
        assert_eq!(r(12345, -2), Some(12300));
        assert_eq!(r(-12345, -2), Some(-12300));
        assert_eq!(r(15, -1), Some(20));
        assert_eq!(r(25, -1), Some(30));
        assert_eq!(r(12345, 2), Some(12345));
        // duckdb: mod(-7,3) = -1, mod(7,-3) = 1、0 除算は NULL。
        let m = |x: i64, y: i64| {
            int_at(
                &run("mod", &[&vi(Ty::BigInt, &[Some(x)]), &vi(Ty::BigInt, &[Some(y)])]).unwrap(),
                0,
            )
        };
        assert_eq!(m(-7, 3), Some(-1));
        assert_eq!(m(7, -3), Some(1));
        assert_eq!(m(7, 0), None);
    }

    #[test]
    fn numeric_float_domain() {
        let f = |name: &str, x: f64| flt_at(&run(name, &[&vf(&[Some(x)])]).unwrap(), 0);
        // duckdb: round(2.5)=3, round(3.5)=4, round(-2.5)=-3（0 から遠ざかる）
        assert_eq!(f("round", 2.5), Some(3.0));
        assert_eq!(f("round", 3.5), Some(4.0));
        assert_eq!(f("round", -2.5), Some(-3.0));
        assert_eq!(f("round", 0.5), Some(1.0));
        assert_eq!(f("round", 1.5), Some(2.0));
        assert_eq!(f("abs", -2.5), Some(2.5));
        assert_eq!(f("sign", -5.0), Some(-1.0));
        assert_eq!(f("ceil", 3.2), Some(4.0));
        assert_eq!(f("ceil", -3.2), Some(-3.0));
        assert_eq!(f("floor", 3.8), Some(3.0));
        assert_eq!(f("floor", -3.2), Some(-4.0));
        assert_eq!(f("trunc", -2.7), Some(-2.0));
        assert_eq!(f("sqrt", 4.0), Some(2.0));
        assert_eq!(f("sqrt", 0.0), Some(0.0));
        // duckdb はエラーにするが、ここは NULL（式の途中で落とさない）。
        assert_eq!(f("sqrt", -1.0), None);
        assert_eq!(f("ln", 0.0), None);
        assert_eq!(f("log", 0.0), None);
        assert!((f("log", 100.0).unwrap() - 2.0).abs() < 1e-15);
        assert_eq!(f("exp", 0.0), Some(1.0));
        assert_eq!(f("abs", 0.0), Some(0.0));
        assert_eq!(flt_at(&run("abs", &[&vf(&[None])]).unwrap(), 0), None);

        // duckdb: round(2.675,2)=2.68, round(1.005,2)=1.0, round(2.345,2)=2.35
        let r = |x: f64, d: i64| {
            flt_at(&run("round", &[&vf(&[Some(x)]), &vi(Ty::BigInt, &[Some(d)])]).unwrap(), 0)
        };
        assert_eq!(r(2.675, 2), Some(2.68));
        assert_eq!(r(1.005, 2), Some(1.0));
        assert_eq!(r(2.345, 2), Some(2.35));
        assert_eq!(r(1234.5678, -2), Some(1200.0));

        let p =
            |x: f64, y: f64| flt_at(&run("pow", &[&vf(&[Some(x)]), &vf(&[Some(y)])]).unwrap(), 0);
        // 整数指数は繰り返し二乗なので厳密に一致する。
        assert_eq!(p(2.0, 3.0), Some(8.0));
        assert_eq!(p(2.0, -2.0), Some(0.25));
        assert_eq!(p(5.0, 0.0), Some(1.0));
        // 非整数指数は exp(y ln x) 経由なので数 ulp ずれる。
        assert!((p(9.0, 0.5).unwrap() - 3.0).abs() < 1e-14);
    }

    #[test]
    fn transcendental_accuracy() {
        // 自前の exp / ln / sqrt が倍精度でどれだけ合うか（相対誤差 1e-15 以内）。
        let close = |a: f64, b: f64| ((a - b) / b).abs() < 1e-15;
        assert!(close(f_exp(1.0), core::f64::consts::E));
        assert!(close(f_exp(10.0), 22026.465794806718));
        assert!(close(f_exp(-5.0), 0.006737946999085467));
        assert!(close(f_ln(core::f64::consts::E), 1.0));
        assert!(close(f_ln(10.0), core::f64::consts::LN_10));
        assert!(close(f_ln(1e-300), -690.7755278982137));
        assert!(close(f_sqrt(2.0), core::f64::consts::SQRT_2));
        assert!(close(f_sqrt(1e300), 1e150));
        assert_eq!(f_sqrt(0.0), 0.0);
        assert_eq!(f_sqrt(4.0), 2.0);
        // 非正規化数（指数部 0）も 2^64 倍して扱えている。
        assert!(f_ln(1e-310) < -713.0 && f_ln(1e-310) > -714.0);
        assert!((f_pow(2.0, 0.5) - core::f64::consts::SQRT_2).abs() < 1e-15);
    }

    // --- NULL 系 ------------------------------------------------------------

    #[test]
    fn null_handling_functions() {
        let n = || vi(Ty::Int, &[None]);
        let k = |x: i64| vi(Ty::Int, &[Some(x)]);
        // duckdb: coalesce(NULL,NULL,3)=3, ifnull(NULL,5)=5
        assert_eq!(int_at(&run("coalesce", &[&n(), &n(), &k(3)]).unwrap(), 0), Some(3));
        assert_eq!(int_at(&run("coalesce", &[&n(), &n()]).unwrap(), 0), None);
        assert_eq!(int_at(&run("ifnull", &[&n(), &k(5)]).unwrap(), 0), Some(5));
        assert_eq!(int_at(&run("ifnull", &[&k(1), &k(5)]).unwrap(), 0), Some(1));
        // duckdb: nullif(1,1)=NULL, nullif(1,2)=1, nullif(1,NULL)=1, nullif(NULL,1)=NULL
        assert_eq!(int_at(&run("nullif", &[&k(1), &k(1)]).unwrap(), 0), None);
        assert_eq!(int_at(&run("nullif", &[&k(1), &k(2)]).unwrap(), 0), Some(1));
        assert_eq!(int_at(&run("nullif", &[&k(1), &n()]).unwrap(), 0), Some(1));
        assert_eq!(int_at(&run("nullif", &[&n(), &k(1)]).unwrap(), 0), None);
    }

    #[test]
    fn greatest_least_skip_nulls() {
        let n = || vi(Ty::Int, &[None]);
        let k = |x: i64| vi(Ty::Int, &[Some(x)]);
        // duckdb: greatest(1,NULL,3)=3, least(NULL,2,1)=1, greatest(NULL,NULL)=NULL
        assert_eq!(int_at(&run("greatest", &[&k(1), &n(), &k(3)]).unwrap(), 0), Some(3));
        assert_eq!(int_at(&run("least", &[&n(), &k(2), &k(1)]).unwrap(), 0), Some(1));
        assert_eq!(int_at(&run("greatest", &[&n(), &n()]).unwrap(), 0), None);
        assert_eq!(int_at(&run("least", &[&k(5), &n()]).unwrap(), 0), Some(5));
        assert_eq!(int_at(&run("greatest", &[&n(), &k(7)]).unwrap(), 0), Some(7));
        // duckdb: greatest('a','b') = 'b'
        let g = run("greatest", &[&vs(&[Some("a")]), &vs(&[Some("b")])]).unwrap();
        assert_eq!(str_at(&g, 0).as_deref(), Some("b"));
    }

    // --- 日時 ---------------------------------------------------------------

    #[test]
    fn civil_roundtrip() {
        for &d in &[-719_468i64, -1, 0, 1, 19_000, 100_000, -50_000] {
            let (y, m, dd) = civil_from_days(d);
            assert_eq!(days_from_civil(y, m, dd), d);
        }
    }

    #[test]
    fn date_trunc_matches_duckdb() {
        let f = |p: &str, lit: &str| {
            let v = run("date_trunc", &[&vs(&[Some(p)]), &ts(lit)]).unwrap();
            let mut o = Vec::new();
            fmt_timestamp(int_at(&v, 0).unwrap(), &mut o);
            String::from_utf8(o).unwrap()
        };
        // エポック前（負のマイクロ秒）。切り捨てだと 1 単位ずれる境目。
        assert_eq!(f("year", "1969-07-20 20:17:40"), "1969-01-01 00:00:00");
        assert_eq!(f("quarter", "1969-07-20 20:17:40"), "1969-07-01 00:00:00");
        assert_eq!(f("month", "1969-07-20 20:17:40"), "1969-07-01 00:00:00");
        assert_eq!(f("week", "1969-07-20 20:17:40"), "1969-07-14 00:00:00");
        assert_eq!(f("day", "1969-07-20 20:17:40"), "1969-07-20 00:00:00");
        assert_eq!(f("hour", "1969-07-20 20:17:40"), "1969-07-20 20:00:00");
        assert_eq!(f("minute", "1969-07-20 20:17:40"), "1969-07-20 20:17:00");
        assert_eq!(f("second", "1969-07-20 20:17:40.123456"), "1969-07-20 20:17:40");
        // うるう日と年境界。
        assert_eq!(f("week", "2024-02-29 12:34:56"), "2024-02-26 00:00:00");
        assert_eq!(f("YEAR", "2024-12-31 23:59:59"), "2024-01-01 00:00:00");
        assert_eq!(f("month", "2024-01-01 00:00:00"), "2024-01-01 00:00:00");
    }

    #[test]
    fn date_part_matches_duckdb() {
        assert_eq!(part("date_part", "dow", "2024-02-29 12:00:00"), Some(4));
        assert_eq!(part("date_part", "doy", "2024-02-29 12:00:00"), Some(60));
        assert_eq!(part("date_part", "quarter", "2024-02-29 00:00:00"), Some(1));
        assert_eq!(part("date_part", "epoch", "1969-07-20 20:17:40"), Some(-14_182_940));
        assert_eq!(part("date_part", "dow", "1969-07-20 00:00:00"), Some(0));
        // ISO 週番号。年をまたぐ週が前年の 52/53 週になる境目を確認する。
        assert_eq!(part("date_part", "week", "2024-01-01 00:00:00"), Some(1));
        assert_eq!(part("date_part", "week", "2023-01-01 00:00:00"), Some(52));
        assert_eq!(part("date_part", "week", "2021-01-03 00:00:00"), Some(53));
        assert_eq!(part("date_part", "week", "2024-12-30 00:00:00"), Some(1));
        assert_eq!(part("date_part", "week", "1969-07-20 00:00:00"), Some(29));
        assert_eq!(part("date_part", "minute", "1969-07-20 20:17:40"), Some(17));
        assert_eq!(part("date_part", "second", "2024-01-01 10:20:30.5"), Some(30));
        // 大文字・複数形・extract の別名も受ける。
        assert_eq!(part("extract", "HOUR", "1969-07-20 20:17:40"), Some(20));
        assert_eq!(part("date_part", "years", "1900-01-01 00:00:00"), Some(1900));
        // 略記。
        let sh = |name: &str, lit: &str| int_at(&run(name, &[&ts(lit)]).unwrap(), 0);
        assert_eq!(sh("year", "1900-01-01 00:00:00"), Some(1900));
        assert_eq!(sh("month", "2024-02-29 00:00:00"), Some(2));
        assert_eq!(sh("day", "2024-02-29 00:00:00"), Some(29));
        assert_eq!(sh("hour", "1969-07-20 20:17:40"), Some(20));
        assert_eq!(sh("minute", "1969-07-20 20:17:40"), Some(17));
        assert_eq!(sh("second", "2024-01-01 10:20:30.5"), Some(30));
        // NULL 伝播。
        let n = run("year", &[&vi(Ty::Timestamp, &[None])]).unwrap();
        assert_eq!(int_at(&n, 0), None);
    }

    #[test]
    fn date_diff_and_add() {
        let d = |p: &str, a: &str, b: &str| {
            int_at(&run("date_diff", &[&vs(&[Some(p)]), &ts(a), &ts(b)]).unwrap(), 0)
        };
        // すべて duckdb 実測値（境界を跨いだ回数）。
        assert_eq!(d("day", "2024-01-01 00:00:00", "2024-03-01 00:00:00"), Some(60));
        assert_eq!(d("month", "2024-01-31 00:00:00", "2024-03-01 00:00:00"), Some(2));
        assert_eq!(d("year", "2020-12-31 00:00:00", "2021-01-01 00:00:00"), Some(1));
        assert_eq!(d("hour", "2024-01-01 01:59:00", "2024-01-01 03:00:00"), Some(2));
        assert_eq!(d("week", "2024-01-01 00:00:00", "2024-01-15 00:00:00"), Some(2));
        assert_eq!(d("week", "2024-01-02 00:00:00", "2024-01-08 00:00:00"), Some(0));
        assert_eq!(d("quarter", "2024-01-01 00:00:00", "2024-07-01 00:00:00"), Some(2));
        assert_eq!(d("day", "2024-01-01 23:00:00", "2024-01-02 01:00:00"), Some(1));
        assert_eq!(d("second", "2024-01-01 00:00:00.9", "2024-01-01 00:00:01.1"), Some(1));
        // エポック前をまたぐ。
        assert_eq!(d("day", "1969-12-31 00:00:00", "1970-01-01 00:00:00"), Some(1));

        let a = |p: &str, k: i64, lit: &str| {
            let v =
                run("date_add", &[&vs(&[Some(p)]), &vi(Ty::BigInt, &[Some(k)]), &ts(lit)]).unwrap();
            let mut o = Vec::new();
            fmt_timestamp(int_at(&v, 0).unwrap(), &mut o);
            String::from_utf8(o).unwrap()
        };
        // duckdb の INTERVAL 加算と同じく、日は月末で切り詰める。
        assert_eq!(a("month", 1, "2024-01-31 00:00:00"), "2024-02-29 00:00:00");
        assert_eq!(a("month", -1, "2024-03-31 00:00:00"), "2024-02-29 00:00:00");
        assert_eq!(a("year", 1, "2024-02-29 00:00:00"), "2025-02-28 00:00:00");
        assert_eq!(a("day", 1, "1969-12-31 23:00:00"), "1970-01-01 23:00:00");
        assert_eq!(a("hour", -1, "1970-01-01 00:00:00"), "1969-12-31 23:00:00");
    }

    #[test]
    fn add_interval_to_ts_matches_duckdb() {
        let a = |lit: &str, months: i32, days: i32, micros: i64| {
            let us = int_at(&ts(lit), 0).unwrap();
            let v = add_interval_to_ts(us, months, days, micros).unwrap();
            let mut o = Vec::new();
            fmt_timestamp(v, &mut o);
            String::from_utf8(o).unwrap()
        };
        // すべて duckdb 実測値。
        assert_eq!(a("2024-01-01 00:00:00", 0, 3, 0), "2024-01-04 00:00:00");
        assert_eq!(a("2024-01-01 00:00:00", 0, 0, 3_600_000_000), "2024-01-01 01:00:00");
        // 月末クランプ: 1 月が 31 日で 2 月は 29 日しかないので繰り上がらず 29 日止まり。
        assert_eq!(a("2024-01-31 00:00:00", 1, 0, 0), "2024-02-29 00:00:00");
        assert_eq!(a("2024-01-31 00:00:00", 2, 0, 0), "2024-03-31 00:00:00");
        assert_eq!(a("2024-01-31 00:00:00", 1, 1, 0), "2024-03-01 00:00:00");
        assert_eq!(a("2024-01-31 00:00:00", 1, -1, 0), "2024-02-28 00:00:00");
        assert_eq!(a("2024-03-31 00:00:00", -1, 0, 0), "2024-02-29 00:00:00");
        assert_eq!(a("2024-01-31 00:00:00", -1, -1, 0), "2023-12-30 00:00:00");
        assert_eq!(a("1970-01-01 00:00:00", 0, -1, 0), "1969-12-31 00:00:00");
        // 月加算のあと時刻はそのまま保たれ、マイクロ秒の加算で日を跨ぐ。
        assert_eq!(a("2024-01-31 23:30:00", 1, 0, 45 * 60 * 1_000_000), "2024-03-01 00:15:00");
    }

    #[test]
    fn last_day_and_strftime() {
        let l = int_at(&run("last_day", &[&ts("2024-02-10 00:00:00")]).unwrap(), 0).unwrap();
        let mut o = Vec::new();
        fmt_date(l, &mut o);
        assert_eq!(String::from_utf8(o).unwrap(), "2024-02-29");
        let f = run("strftime", &[&ts("2024-02-29 01:02:03"), &vs(&[Some("%Y-%m-%d %H:%M:%S")])])
            .unwrap();
        assert_eq!(str_at(&f, 0).as_deref(), Some("2024-02-29 01:02:03"));
        let f = run("strftime", &[&ts("2024-02-29 01:02:03"), &vs(&[Some("%Y/%%/%q")])]).unwrap();
        assert_eq!(str_at(&f, 0).as_deref(), Some("2024/%/%q"));
    }

    #[test]
    fn to_date_and_to_timestamp() {
        let d = |s: &str| int_at(&run("to_date", &[&vs(&[Some(s)])]).unwrap(), 0);
        assert_eq!(d("1970-01-01"), Some(0));
        assert_eq!(d("2024-1-5"), Some(days_from_civil(2024, 1, 5)));
        assert_eq!(d("nope"), None);
        assert_eq!(d("2024-13-01"), None);
        let t = |s: &str| int_at(&run("to_timestamp", &[&vs(&[Some(s)])]).unwrap(), 0);
        assert_eq!(t("1970-01-01 00:00:01"), Some(1_000_000));
        assert_eq!(t("1970-01-01"), Some(0));
        assert_eq!(t("bad"), None);
    }

    // --- パーサ / フォーマッタ（kernels のキャストからも使われる） ------------

    #[test]
    fn parse_and_format_roundtrip() {
        let fmt = |us: i64| {
            let mut o = Vec::new();
            fmt_timestamp(us, &mut o);
            String::from_utf8(o).unwrap()
        };
        for lit in [
            "2024-01-01 00:00:00",
            "1969-07-20 20:17:40",
            "0001-01-01 00:00:00",
            "2024-02-29 23:59:59.123456",
            "1900-12-31 12:00:00.1",
        ] {
            let us = parse_timestamp(lit.as_bytes()).unwrap();
            assert_eq!(fmt(us), lit, "往復に失敗: {lit}");
        }
        // duckdb の VARCHAR 表現に合わせる。
        assert_eq!(fmt(parse_timestamp(b"2024-01-01").unwrap()), "2024-01-01 00:00:00");
        assert_eq!(fmt(parse_timestamp(b"2024-01-05T10:20:30").unwrap()), "2024-01-05 10:20:30");
        assert_eq!(fmt(parse_timestamp(b" 2024-01-05 ").unwrap()), "2024-01-05 00:00:00");
    }

    #[test]
    fn parse_rejects_bad_input() {
        assert_eq!(parse_date(b"2024-13-01"), None);
        assert_eq!(parse_date(b"2024-02-30"), None);
        assert_eq!(parse_date(b"2024-00-01"), None);
        assert_eq!(parse_date(b"2024-01-00"), None);
        assert_eq!(parse_date(b""), None);
        assert_eq!(parse_date(b"abc"), None);
        assert_eq!(parse_date(b"2024-01-01x"), None);
        // うるう年は通す。
        assert!(parse_date(b"2024-02-29").is_some());
        assert_eq!(parse_date(b"2023-02-29"), None);
        assert_eq!(parse_timestamp(b"2024-01-01 25:00:00"), None);
        assert_eq!(parse_timestamp(b"2024-01-01 10:61:00"), None);
        assert_eq!(parse_timestamp(b"2024-01-01 10"), None);
        assert_eq!(parse_time(b"01:02:03"), Some(3_723_000_000));
        assert_eq!(parse_time(b"01:02"), Some(3_720_000_000));
        assert_eq!(parse_time(b"01:02:03.5"), Some(3_723_500_000));
        assert_eq!(parse_time(b"1:2:3"), Some(3_723_000_000));
        assert_eq!(parse_time(b"xx"), None);
        let mut o = Vec::new();
        fmt_time(3_723_500_000, &mut o);
        assert_eq!(String::from_utf8(o).unwrap(), "01:02:03.5");
    }

    // --- 文字列 ↔ 日時のキャスト（kernels 側の実装をここから検証する） ------

    #[test]
    fn string_to_temporal_cast() {
        let src = vs(&[
            Some("2024-02-29"),
            Some("not a date"),
            Some("2024-13-01"),
            Some("2024-02-30"),
            None,
        ]);
        let out = kernels::cast(Ty::Varchar, Ty::Date, &src).unwrap();
        assert_eq!(int_at(&out, 0), Some(days_from_civil(2024, 2, 29)));
        // 読めない・範囲外はエラーではなくその行だけ NULL。
        assert_eq!(int_at(&out, 1), None);
        assert_eq!(int_at(&out, 2), None);
        assert_eq!(int_at(&out, 3), None);
        assert_eq!(int_at(&out, 4), None);

        let src = vs(&[Some("2024-01-05 10:20:30"), Some("2024-01-05"), Some("x")]);
        let out = kernels::cast(Ty::Varchar, Ty::Timestamp, &src).unwrap();
        assert_eq!(int_at(&out, 0), Some(parse_timestamp(b"2024-01-05 10:20:30").unwrap()));
        // 日付だけの文字列は深夜 0 時として読む。
        assert_eq!(int_at(&out, 1), Some(days_from_civil(2024, 1, 5) * US_PER_DAY));
        assert_eq!(int_at(&out, 2), None);

        let src = vs(&[Some("01:02:03.5"), Some("nope")]);
        let out = kernels::cast(Ty::Varchar, Ty::Time, &src).unwrap();
        assert_eq!(int_at(&out, 0), Some(3_723_500_000));
        assert_eq!(int_at(&out, 1), None);
    }

    #[test]
    fn temporal_to_string_cast() {
        // duckdb の VARCHAR 表現と一致させる。
        let d = vi(Ty::Date, &[Some(days_from_civil(2024, 1, 1)), Some(0), None]);
        let out = kernels::cast(Ty::Date, Ty::Varchar, &d).unwrap();
        assert_eq!(str_at(&out, 0).as_deref(), Some("2024-01-01"));
        assert_eq!(str_at(&out, 1).as_deref(), Some("1970-01-01"));
        assert_eq!(str_at(&out, 2), None);

        let t = vi(
            Ty::Timestamp,
            &[
                Some(parse_timestamp(b"1969-07-20 20:17:40").unwrap()),
                Some(parse_timestamp(b"2024-01-01").unwrap()),
                Some(parse_timestamp(b"2024-01-01 00:00:00.123456").unwrap()),
                Some(parse_timestamp(b"2024-01-01 00:00:00.1").unwrap()),
            ],
        );
        let out = kernels::cast(Ty::Timestamp, Ty::Varchar, &t).unwrap();
        assert_eq!(str_at(&out, 0).as_deref(), Some("1969-07-20 20:17:40"));
        assert_eq!(str_at(&out, 1).as_deref(), Some("2024-01-01 00:00:00"));
        assert_eq!(str_at(&out, 2).as_deref(), Some("2024-01-01 00:00:00.123456"));
        assert_eq!(str_at(&out, 3).as_deref(), Some("2024-01-01 00:00:00.1"));

        let tm = vi(Ty::Time, &[Some(3_723_000_000), Some(3_723_500_000)]);
        let out = kernels::cast(Ty::Time, Ty::Varchar, &tm).unwrap();
        assert_eq!(str_at(&out, 0).as_deref(), Some("01:02:03"));
        assert_eq!(str_at(&out, 1).as_deref(), Some("01:02:03.5"));
    }

    #[test]
    fn timestamp_string_roundtrip_through_cast() {
        let lits = ["2024-02-29 23:59:59.123456", "1969-07-20 20:17:40", "0001-01-01 00:00:00"];
        let v = vi(
            Ty::Timestamp,
            &lits.iter().map(|l| parse_timestamp(l.as_bytes())).collect::<Vec<_>>(),
        );
        let s = kernels::cast(Ty::Timestamp, Ty::Varchar, &v).unwrap();
        let back = kernels::cast(Ty::Varchar, Ty::Timestamp, &s).unwrap();
        for (i, lit) in lits.iter().enumerate() {
            assert_eq!(int_at(&back, i), int_at(&v, i), "{lit} で往復に失敗");
        }
    }

    // --- ベクタ／定数の扱い -------------------------------------------------

    #[test]
    fn constant_folding_keeps_length_one() {
        // 全引数が長さ 1 なら結果も長さ 1（stride 0 に潰さない）。
        let out = run("upper", &[&vs(&[Some("ab")])]).unwrap();
        assert_eq!(out.len(), 1);
        let out = run("substr", &[&vs(&[Some("hello")]), &vi(Ty::BigInt, &[Some(2)])]).unwrap();
        assert_eq!(out.len(), 1);
        let out = run("coalesce", &[&vi(Ty::Int, &[None]), &vi(Ty::Int, &[Some(1)])]).unwrap();
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn constant_operand_broadcasts() {
        // 長さ 3 の列と長さ 1 の定数を混ぜると、結果は 3 行。
        let col = vs(&[Some("aa"), None, Some("cc")]);
        let k = vi(Ty::BigInt, &[Some(2)]);
        let out = run("substr", &[&col, &k]).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(str_at(&out, 0).as_deref(), Some("a"));
        assert_eq!(str_at(&out, 1), None);
        assert_eq!(str_at(&out, 2).as_deref(), Some("c"));
        // 空ベクタは結果も空。
        let empty = Vector::new(Ty::Varchar);
        let out = run("upper", &[&empty]).unwrap();
        assert_eq!(out.len(), 0);
    }

    #[test]
    fn multi_row_dates() {
        let col = vi(
            Ty::Timestamp,
            &[
                Some(parse_timestamp(b"2024-02-29 00:00:00").unwrap()),
                None,
                Some(parse_timestamp(b"1969-07-20 20:17:40").unwrap()),
            ],
        );
        let out = run("year", &[&col]).unwrap();
        assert_eq!(int_at(&out, 0), Some(2024));
        assert_eq!(int_at(&out, 1), None);
        assert_eq!(int_at(&out, 2), Some(1969));
    }

    // --- JSON -----------------------------------------------------------

    fn vj(vals: &[Option<&str>]) -> Vector {
        let mut v = Vector::new(Ty::Json);
        for x in vals {
            match x {
                Some(s) => v.push_value(&Value::Bytes(s.as_bytes().to_vec())),
                None => v.push_null(),
            }
        }
        v
    }

    #[test]
    fn json_extract_follows_dollar_paths_like_duckdb() {
        // duckdb: json_extract('{"a":{"b":[1,2,3]}}', '$.a.b[1]') -> 2
        let doc = vj(&[Some(r#"{"a":{"b":[1,2,3]}}"#)]);
        let path = vs(&[Some("$.a.b[1]")]);
        let out = run("json_extract", &[&doc, &path]).unwrap();
        assert_eq!(out.ty(), Ty::Json);
        assert_eq!(str_at(&out, 0).as_deref(), Some("2"));

        // duckdb: json_extract('{"a":1}', '$.b') -> NULL（キー無し）
        let doc = vj(&[Some(r#"{"a":1}"#)]);
        let path = vs(&[Some("$.b")]);
        assert_eq!(str_at(&run("json_extract", &[&doc, &path]).unwrap(), 0), None);
    }

    #[test]
    fn json_extract_operator_desugars_at_parse_time_not_tested_here() {
        // `->`/`->>` は sql::parser が json_extract(_string) へ展開するだけで、
        // この関数自体には演算子固有のロジックが無い（parser 側のテスト
        // `sql::parser::tests` で precedence 込みに確認する）。
        let doc = vj(&[Some(r#"{"a":1}"#)]);
        let path = vs(&[Some("$.a")]);
        assert_eq!(str_at(&run("json_extract", &[&doc, &path]).unwrap(), 0).as_deref(), Some("1"));
    }

    #[test]
    fn json_extract_string_unquotes_and_nulls_json_null() {
        // duckdb: json_extract_string('{"a":"hello"}', '$.a') -> 'hello'
        let doc = vj(&[Some(r#"{"a":"hello"}"#)]);
        let path = vs(&[Some("$.a")]);
        let out = run("json_extract_string", &[&doc, &path]).unwrap();
        assert_eq!(out.ty(), Ty::Varchar);
        assert_eq!(str_at(&out, 0).as_deref(), Some("hello"));

        // duckdb: json_extract_string('{"a":1}', '$.a') -> '1'
        let doc = vj(&[Some(r#"{"a":1}"#)]);
        assert_eq!(
            str_at(&run("json_extract_string", &[&doc, &path]).unwrap(), 0).as_deref(),
            Some("1")
        );

        // duckdb: json_extract_string('{"a":null}', '$.a') -> NULL
        let doc = vj(&[Some(r#"{"a":null}"#)]);
        assert_eq!(str_at(&run("json_extract_string", &[&doc, &path]).unwrap(), 0), None);
    }

    #[test]
    fn json_type_matches_duckdb_strings() {
        // duckdb: json_type('{"a":1}')='OBJECT', json_type('[1,2]')='ARRAY',
        // json_type('"x"')='VARCHAR', json_type('true')='BOOLEAN',
        // json_type('null')='NULL', json_type('1.5')='DOUBLE'
        let cases: &[(&str, &str)] = &[
            (r#"{"a":1}"#, "OBJECT"),
            ("[1,2]", "ARRAY"),
            ("\"x\"", "VARCHAR"),
            ("true", "BOOLEAN"),
            ("null", "NULL"),
            ("1.5", "DOUBLE"),
            ("1", "BIGINT"),
            ("-1", "BIGINT"),
        ];
        for (doc, want) in cases {
            let d = vj(&[Some(doc)]);
            assert_eq!(str_at(&run("json_type", &[&d]).unwrap(), 0).as_deref(), Some(*want));
        }
        // パス付き 2 引数形。
        let doc = vj(&[Some(r#"{"a":{"b":1}}"#)]);
        let path = vs(&[Some("$.a")]);
        assert_eq!(
            str_at(&run("json_type", &[&doc, &path]).unwrap(), 0).as_deref(),
            Some("OBJECT")
        );
    }

    #[test]
    fn json_array_length_is_zero_for_non_arrays_and_null_for_missing_path() {
        // duckdb: json_array_length('[1,2,3]')=3, json_array_length('{"a":1}')=0
        let a = vj(&[Some("[1,2,3]")]);
        assert_eq!(int_at(&run("json_array_length", &[&a]).unwrap(), 0), Some(3));
        let o = vj(&[Some(r#"{"a":1}"#)]);
        assert_eq!(int_at(&run("json_array_length", &[&o]).unwrap(), 0), Some(0));
        // duckdb: json_array_length('{"a":1}', '$.b') -> NULL（パス無し）
        let path = vs(&[Some("$.b")]);
        assert_eq!(int_at(&run("json_array_length", &[&o, &path]).unwrap(), 0), None);
    }

    #[test]
    fn list_extract_is_one_based_with_negative_from_end() {
        // duckdb: list_extract([10,20,30], 1) -> 10, list_extract(.., -1) -> 30,
        // list_extract(.., 0) -> NULL, list_extract(.., 100) -> NULL
        let a = vj(&[Some("[10,20,30]")]);
        let assert_idx = |idx: i64, want: Option<&str>| {
            let i = vi(Ty::BigInt, &[Some(idx)]);
            assert_eq!(str_at(&run("list_extract", &[&a, &i]).unwrap(), 0).as_deref(), want);
        };
        assert_idx(1, Some("10"));
        assert_idx(-1, Some("30"));
        assert_idx(0, None);
        assert_idx(100, None);
    }

    #[test]
    fn map_extract_looks_up_a_key_or_returns_null() {
        let m = vj(&[Some(r#"{"a":1,"b":2}"#)]);
        assert_eq!(
            str_at(&run("map_extract", &[&m, &vs(&[Some("a")])]).unwrap(), 0).as_deref(),
            Some("1")
        );
        assert_eq!(str_at(&run("map_extract", &[&m, &vs(&[Some("z")])]).unwrap(), 0), None);
    }

    #[test]
    fn to_json_covers_the_documented_scalar_types() {
        // duckdb: to_json(1)='1', to_json(1.5)='1.5', to_json('hello')='"hello"',
        // to_json(true)='true', to_json(DATE '2024-01-01')='"2024-01-01"'
        assert_eq!(
            str_at(&run("to_json", &[&vi(Ty::BigInt, &[Some(1)])]).unwrap(), 0).as_deref(),
            Some("1")
        );
        assert_eq!(
            str_at(&run("to_json", &[&vf(&[Some(1.5)])]).unwrap(), 0).as_deref(),
            Some("1.5")
        );
        assert_eq!(
            str_at(&run("to_json", &[&vs(&[Some("hello")])]).unwrap(), 0).as_deref(),
            Some("\"hello\"")
        );
        let mut b = Vector::new(Ty::Boolean);
        b.push_value(&Value::Bool(true));
        assert_eq!(str_at(&run("to_json", &[&b]).unwrap(), 0).as_deref(), Some("true"));
        let mut d = Vector::new(Ty::Date);
        d.push_value(&Value::I32(19723)); // 2024-01-01
        assert_eq!(str_at(&run("to_json", &[&d]).unwrap(), 0).as_deref(), Some("\"2024-01-01\""));
        // duckdb: to_json(NULL) は SQL NULL のまま（既定の NULL 伝播）。
        let n = vi(Ty::BigInt, &[None]);
        assert_eq!(str_at(&run("to_json", &[&n]).unwrap(), 0), None);
    }

    #[test]
    fn to_json_escapes_quotes_in_strings() {
        let out = run("to_json", &[&vs(&[Some("a\"b\\c")])]).unwrap();
        assert_eq!(str_at(&out, 0).as_deref(), Some(r#""a\"b\\c""#));
    }

    #[test]
    fn json_array_and_json_object_match_duckdb_construction() {
        // duckdb: json_array(1, 'x', true, NULL) -> [1,"x",true,null]
        let out = run(
            "json_array",
            &[
                &vi(Ty::BigInt, &[Some(1)]),
                &vs(&[Some("x")]),
                &{
                    let mut b = Vector::new(Ty::Boolean);
                    b.push_value(&Value::Bool(true));
                    b
                },
                &vi(Ty::BigInt, &[None]),
            ],
        )
        .unwrap();
        assert_eq!(out.ty(), Ty::Json);
        assert_eq!(str_at(&out, 0).as_deref(), Some(r#"[1,"x",true,null]"#));

        // duckdb: json_object('a', 1, 'b', 'x') -> {"a":1,"b":"x"}
        let out = run(
            "json_object",
            &[&vs(&[Some("a")]), &vi(Ty::BigInt, &[Some(1)]), &vs(&[Some("b")]), &vs(&[Some("x")])],
        )
        .unwrap();
        assert_eq!(str_at(&out, 0).as_deref(), Some(r#"{"a":1,"b":"x"}"#));

        // duckdb: json_object('a', NULL) -> {"a":null}（NULL 値は埋め込む）
        let out = run("json_object", &[&vs(&[Some("a")]), &vi(Ty::BigInt, &[None])]).unwrap();
        assert_eq!(str_at(&out, 0).as_deref(), Some(r#"{"a":null}"#));
    }

    #[test]
    fn list_value_is_an_alias_for_json_array() {
        let out =
            run("list_value", &[&vi(Ty::BigInt, &[Some(1)]), &vi(Ty::BigInt, &[Some(2)])]).unwrap();
        assert_eq!(str_at(&out, 0).as_deref(), Some("[1,2]"));
    }

    #[test]
    fn malformed_json_source_is_an_error_not_a_silent_null() {
        // duckdb: json_extract('{not valid json', '$.a') -> エラー
        let doc = vj(&[Some("{not valid json")]);
        let path = vs(&[Some("$.a")]);
        assert_eq!(code_of(run("json_extract", &[&doc, &path])), Some(Code::SyntaxError));
    }

    #[test]
    fn json_resolve_errors() {
        assert_eq!(code_of(resolve("json_extract", &[Ty::Json])), Some(Code::WrongArgCount));
        assert_eq!(code_of(resolve("to_json", &[])), Some(Code::WrongArgCount));
        // BLOB/INTERVAL は to_json/json_array の対象外。
        assert_eq!(code_of(resolve("to_json", &[Ty::Blob])), Some(Code::TypeMismatch));
        assert_eq!(code_of(resolve("to_json", &[Ty::Interval])), Some(Code::TypeMismatch));
        assert_eq!(
            code_of(resolve("json_object", &[Ty::Varchar])),
            Some(Code::WrongArgCount),
            "奇数個の引数は拒否"
        );
        assert_eq!(code_of(resolve("json_array", &[])), Some(Code::WrongArgCount));
    }

    // --- エラー -------------------------------------------------------------

    #[test]
    fn resolve_errors() {
        assert_eq!(code_of(resolve("upper", &[])), Some(Code::WrongArgCount));
        assert_eq!(
            code_of(resolve("upper", &[Ty::Varchar, Ty::Varchar])),
            Some(Code::WrongArgCount)
        );
        assert_eq!(code_of(resolve("substr", &[Ty::Varchar])), Some(Code::WrongArgCount));
        assert_eq!(code_of(resolve("pow", &[Ty::Double])), Some(Code::WrongArgCount));
        assert_eq!(code_of(resolve("abs", &[Ty::Varchar])), Some(Code::TypeMismatch));
        assert_eq!(code_of(resolve("mod", &[Ty::Int, Ty::Varchar])), Some(Code::TypeMismatch));
        assert_eq!(code_of(resolve("greatest", &[Ty::Int, Ty::Varchar])), Some(Code::TypeMismatch));
        assert_eq!(code_of(resolve("nosuchfunc", &[])), Some(Code::FunctionNotFound));
        // 時計が無いので未実装（モジュール冒頭のコメント参照）。
        assert_eq!(code_of(resolve("now", &[])), Some(Code::FunctionNotFound));
        assert_eq!(code_of(resolve("current_timestamp", &[])), Some(Code::FunctionNotFound));
        assert_eq!(code_of(resolve("current_date", &[])), Some(Code::FunctionNotFound));
    }

    #[test]
    fn resolve_coerces_argument_types() {
        // INT 引数は BIGINT へ、FLOAT 混在は DOUBLE へ寄せる。
        assert_eq!(resolve("abs", &[Ty::Int]).unwrap().1, vec![Ty::BigInt]);
        assert_eq!(resolve("abs", &[Ty::Double]).unwrap().1, vec![Ty::Double]);
        assert_eq!(
            resolve("abs", &[Ty::Decimal { precision: 9, scale: 2 }]).unwrap().2,
            Ty::Double
        );
        assert_eq!(resolve("length", &[Ty::Varchar]).unwrap().2, Ty::BigInt);
        // 日時関数は TIMESTAMP に寄せる（DATE 列も VARCHAR も Cast 経由で入る）。
        assert_eq!(resolve("year", &[Ty::Date]).unwrap().1, vec![Ty::Timestamp]);
        assert_eq!(
            resolve("date_trunc", &[Ty::Varchar, Ty::Date]).unwrap().1,
            vec![Ty::Varchar, Ty::Timestamp]
        );
        assert_eq!(resolve("coalesce", &[Ty::Int, Ty::BigInt]).unwrap().2, Ty::BigInt);
    }

    #[test]
    fn call_errors_on_bad_part() {
        let (id, _, ret) = resolve("date_part", &[Ty::Varchar, Ty::Timestamp]).unwrap();
        let bad = vs(&[Some("fortnight")]);
        let t = ts("2024-01-01 00:00:00");
        assert_eq!(code_of(call(id, ret, &[&bad, &t])), Some(Code::TypeMismatch));
        // 引数無しで呼ばれてもパニックしない。
        assert_eq!(code_of(call(id, ret, &[])), Some(Code::WrongArgCount));
    }

    #[test]
    fn oversized_output_is_rejected() {
        let (id, _, ret) = resolve("repeat", &[Ty::Varchar, Ty::BigInt]).unwrap();
        let s = vs(&[Some("abcdefgh")]);
        let k = vi(Ty::BigInt, &[Some(1 << 40)]);
        assert_eq!(code_of(call(id, ret, &[&s, &k])), Some(Code::LimitExceeded));
    }
}
