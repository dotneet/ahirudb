//! ネイティブ CLI。
//!
//! 開発とテストはここで回す。wasm 越しのデバッグは効率が悪いので、
//! 「開発とテストはネイティブ、サイズ計測だけ wasm」という分担にしている
//! （DESIGN.md §12）。

use std::collections::HashMap;
use std::process::ExitCode;

use ahiru_core::parquet::file::open_bytes;
use ahiru_core::vector::{Ty, Value};
use ahiru_core::{FormatKind, Prepared, QueryStep, Session};

fn usage() -> ExitCode {
    eprintln!(
        "usage:
  ahiru schema <file>          スキーマを表示 (Parquet は RowGroup も)
  ahiru dump   <file> [n]      先頭 n 行を表示 (既定 20)
  ahiru query  <file>... <sql>  SQL を実行 (テーブル名は t, t2, t3, ... とファイル名)

対応フォーマット: .parquet / .csv / .tsv / .jsonl / .ndjson
拡張子から自動判別する。

複数ファイルを 1 テーブルとして束ねたいときは、`+` で連結して 1 引数にする
(手元確認用の簡易記法。実際のパスに `+` を含むファイルは指定できない):
  ahiru query a.parquet+b.parquet+c.parquet 'SELECT count(*) FROM t'"
    );
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        return usage();
    }
    let r = match args[0].as_str() {
        "schema" if args.len() == 2 => cmd_schema(&args[1]),
        "dump" if args.len() >= 2 => {
            let n = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
            cmd_dump(&args[1], n)
        }
        // 最後の引数が SQL、その手前がすべてファイル。結合を試せるように
        // 複数テーブルを受け付ける。
        "query" if args.len() >= 3 => {
            let n = args.len();
            cmd_query(&args[1..n - 1], &args[n - 1])
        }
        _ => return usage(),
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

type R = Result<(), Box<dyn std::error::Error>>;

fn cmd_schema(path: &str) -> R {
    let bytes = std::fs::read(path)?;
    let kind = FormatKind::detect(path);

    // まずフォーマット非依存のスキーマを出す。
    let mut s = Session::new();
    s.register_bytes_as("t", bytes.clone(), kind)?;
    let table = s.catalog.index_of("t").ok_or("table not registered")?;
    match s.catalog.get_mut(table).ok_or("table not registered")?.resolve()? {
        Ok(()) => {}
        Err(_) => return Err("メモリ上にファイル全体があるのに解決できなかった".into()),
    }
    let t = s.catalog.get(table).ok_or("table not registered")?;
    println!("format: {kind:?}");
    println!("\ncolumns:");
    for (i, f) in t.schema().iter().enumerate() {
        println!(
            "  [{i}] {:<24} {:<12} {}",
            f.name,
            f.ty.name(),
            if f.nullable { "NULL" } else { "NOT NULL" }
        );
    }
    println!("splits: {}", t.num_splits());

    // Parquet だけは物理配置まで見たいので、追加で内訳を出す。
    if kind == FormatKind::Parquet {
        print_parquet_details(&bytes)?;
    }
    Ok(())
}

fn print_parquet_details(bytes: &[u8]) -> R {
    let f = open_bytes(bytes)?;
    println!("\nrows: {}", f.num_rows());
    if let Some(c) = &f.meta.created_by {
        println!("created by: {c}");
    }
    println!("\nrow groups:");
    for (i, rg) in f.meta.row_groups.iter().enumerate() {
        println!("  [{i}] rows={} bytes={}", rg.num_rows, rg.total_byte_size);
        for (j, cc) in rg.columns.iter().enumerate() {
            if let Some(m) = &cc.meta {
                let (start, end) = m.byte_range();
                println!(
                    "        col {j}: codec={:?} enc={:?} range=[{start},{end}) values={}",
                    m.codec, m.encodings, m.num_values
                );
            }
        }
    }
    Ok(())
}

/// `dump` は `SELECT * FROM t LIMIT n` と等価。フォーマット固有の経路を
/// 持たないので、対応フォーマットが増えてもそのまま動く。
fn cmd_dump(path: &str, limit: usize) -> R {
    let mut sql = String::from("SELECT * FROM t LIMIT ");
    sql.push_str(&limit.to_string());
    cmd_query(&[path.to_string()], &sql)
}

fn cmd_query(groups: &[String], sql: &str) -> R {
    let mut s = Session::new();
    // コーデック委譲で元バイトを引くため、(テーブル添字, パート添字) をキーに
    // 控えておく。登録は同名なら置き換わるので、添字は register の返り値から
    // 取る。単一ファイルのテーブルは常にパート 0。
    let mut sources: HashMap<(usize, usize), Vec<u8>> = HashMap::new();

    for (i, group) in groups.iter().enumerate() {
        // 1 つ目が `t`、2 つ目以降が `t2`, `t3`, ...。
        let name = if i == 0 { String::from("t") } else { format!("t{}", i + 1) };
        // `+` で連結された引数は 1 論理テーブルとして束ねる（usage 参照）。
        let paths: Vec<&str> = group.split('+').collect();
        if paths.len() == 1 {
            let path = paths[0];
            let bytes = std::fs::read(path)?;
            let a = s.register_bytes_as(&name, bytes.clone(), FormatKind::detect(path))?;
            sources.insert((a, 0), bytes.clone());
            // ファイル名でも参照できるようにしておく（`parquet('...')` 用）。
            let b = s.register_bytes(path, bytes.clone())?;
            sources.insert((b, 0), bytes);
        } else {
            let mut files = Vec::with_capacity(paths.len());
            for path in &paths {
                files.push((path.to_string(), std::fs::read(path)?));
            }
            let kind = FormatKind::detect(paths[0]);
            let a = s.register_multi_bytes(&name, files.clone(), kind)?;
            for (p, (_, bytes)) in files.into_iter().enumerate() {
                sources.insert((a, p), bytes);
            }
        }
    }

    // コアは時計を持たないので、CURRENT_DATE/CURRENT_TIMESTAMP/now() 用に
    // クエリ開始時刻をここで渡す（DESIGN.md §2、JS ホストの `ahiru_set_now`
    // 呼び出しと同じ役目）。
    let now_micros = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0);
    s.set_now(now_micros);
    let mut q = match s.prepare(sql, &[])? {
        Prepared::Ready(q) => q,
        // メモリ上にファイル全体があるので、ここには来ない。
        Prepared::NeedIo(_) => return Err("unexpected io request".into()),
    };
    // `COPY (SELECT ...) TO 'path'`: `ahiru-core` は no_std でファイルシステムに
    // 触れられないので、実ファイルへの書き込みはここ（ネイティブの CLI）が
    // 引き受ける。`ahiru-core` 側はバイト列と書き込み先パスを `Query` に
    // 載せて返すところまでを担う（`ahiru_core::write` モジュール doc、
    // `ahiru_core::session::Query::copy_result` 参照）。
    if let Some(c) = q.copy.take() {
        let n = c.data.len();
        std::fs::write(&c.path, &c.data)?;
        eprintln!("(copied {n} bytes to {})", c.path);
        return Ok(());
    }
    let types: Vec<Ty> = q.schema.iter().map(|f| f.ty).collect();
    let names: Vec<&str> = q.schema.iter().map(|f| f.name.as_str()).collect();
    println!("{}", names.join("\t"));

    let mut rows = 0usize;
    loop {
        match s.step(&mut q)? {
            QueryStep::Batch(mut b) => {
                b.materialize();
                for r in 0..b.num_rows() {
                    let cells: Vec<String> = b
                        .cols
                        .iter()
                        .zip(&types)
                        .map(|(c, t)| render(&c.value_at(r), *t))
                        .collect();
                    println!("{}", cells.join("\t"));
                    rows += 1;
                }
            }
            QueryStep::NeedIo(_) => return Err("unexpected io request".into()),
            // コーデック委譲。ZSTD は `ahiru-core` の既定フィーチャに含まれる
            // ようになったので、通常はここまで来ない（コア側で内蔵展開される）。
            // 残るのは GZIP（コアが意図的に内蔵していない。DESIGN.md §6）と、
            // `zstd` フィーチャを外してビルドした場合の ZSTD だけ。
            QueryStep::NeedCodec(reqs) => {
                for r in reqs {
                    let src = source_bytes(&sources, r.table, r.part, r.offset, r.len)?;
                    let out = decompress_host(r.codec, src, r.out_len as usize)?;
                    s.provide_decoded(r.table, r.part, r.offset, r.len, out)?;
                }
            }
            QueryStep::Done => break,
        }
    }
    eprintln!("({rows} rows)");
    Ok(())
}

fn source_bytes(
    sources: &HashMap<(usize, usize), Vec<u8>>,
    table: usize,
    part: usize,
    offset: u64,
    len: u32,
) -> Result<&[u8], String> {
    let b = sources
        .get(&(table, part))
        .ok_or_else(|| String::from("unknown table/part for codec request"))?;
    let s = offset as usize;
    let e = s + len as usize;
    if e > b.len() {
        return Err(String::from("codec request out of range"));
    }
    Ok(&b[s..e])
}

/// ホスト側のコーデック。
///
/// - ZSTD は `ahiru-zstd` を直接リンクする。`ahiru-core` の既定ビルドは
///   ZSTD を内蔵している（`zstd` フィーチャ）ので、通常はここまで来ない。
///   `--no-default-features` 相当で `zstd` を外した場合のフォールバック。
/// - GZIP はシステムの `gzip` に投げる。CLI は開発用なので外部プロセスで十分。
///   wasm では `DecompressionStream('gzip')` が同じ役目を果たす。
fn decompress_host(
    codec: ahiru_core::parquet::Compression,
    src: &[u8],
    out_len: usize,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    use ahiru_core::parquet::Compression;
    match codec {
        Compression::Zstd => ahiru_zstd::decompress(src, out_len)
            .map_err(|e| format!("zstd の展開に失敗: {e:?}").into()),
        Compression::Gzip => gunzip(src),
        other => Err(format!("{other:?} はホスト側でも未対応").into()),
    }
}

fn gunzip(src: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let mut c = Command::new("gzip")
        .arg("-dc")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    c.stdin.take().ok_or("no stdin")?.write_all(src)?;
    let out = c.wait_with_output()?;
    if !out.status.success() {
        return Err("gzip failed".into());
    }
    Ok(out.stdout)
}

fn render(v: &Value, ty: Ty) -> String {
    match v {
        Value::Null => "NULL".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::I32(x) if ty == Ty::Date => fmt_date(*x as i64),
        Value::I32(x) => x.to_string(),
        Value::I64(x) if ty == Ty::Timestamp => fmt_timestamp(*x),
        Value::I64(x) if ty == Ty::Timestamptz => format!("{}+00", fmt_timestamp(*x)),
        Value::I64(x) if ty == Ty::Time => fmt_time(*x),
        Value::I64(x) => fmt_scaled(*x as i128, ty),
        Value::I128(x) if ty == Ty::Interval => fmt_interval_value(*x),
        Value::I128(x) => fmt_scaled(*x, ty),
        Value::F64(x) => x.to_string(),
        Value::Bytes(b) if ty == Ty::Uuid => fmt_uuid(b),
        Value::Bytes(b) => match std::str::from_utf8(b) {
            Ok(s) => s.to_string(),
            Err(_) => format!("<{} bytes>", b.len()),
        },
    }
}

fn fmt_interval_value(packed: i128) -> String {
    let (months, days, micros) = ahiru_core::vector::unpack_interval(packed);
    let mut out = Vec::new();
    ahiru_core::vector::fmt_interval(months, days, micros, &mut out);
    String::from_utf8(out).unwrap_or_default()
}

/// DECIMAL はスケール付きの整数で保持しているので、表示時に小数点を入れる。
fn fmt_scaled(v: i128, ty: Ty) -> String {
    let scale = match ty {
        Ty::Decimal { scale, .. } => scale as usize,
        _ => return v.to_string(),
    };
    if scale == 0 {
        return v.to_string();
    }
    let neg = v < 0;
    let digits = v.unsigned_abs().to_string();
    // 整数部が無い場合（0.05 など）は先頭に 0 を補う。
    let padded = if digits.len() <= scale {
        format!("{}{}", "0".repeat(scale - digits.len() + 1), digits)
    } else {
        digits
    };
    let split = padded.len() - scale;
    format!("{}{}.{}", if neg { "-" } else { "" }, &padded[..split], &padded[split..])
}

/// エポックからの日数を `YYYY-MM-DD` にする。civil_from_days アルゴリズム。
fn fmt_date(days: i64) -> String {
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}")
}

/// TIME（深夜からのマイクロ秒）を `HH:MM:SS` で表示する。CLI 表示専用の
/// 簡略化で、マイクロ秒未満は落とす（`fmt_timestamp` の時刻部分と同じ規約）。
fn fmt_time(micros: i64) -> String {
    let rem = micros.rem_euclid(86_400_000_000);
    let (h, mi, s) = (rem / 3_600_000_000, rem / 60_000_000 % 60, rem / 1_000_000 % 60);
    format!("{h:02}:{mi:02}:{s:02}")
}

fn fmt_timestamp(micros: i64) -> String {
    let days = micros.div_euclid(86_400_000_000);
    let rem = micros.rem_euclid(86_400_000_000);
    let (y, m, d) = civil_from_days(days);
    let (h, mi, s) = (rem / 3_600_000_000, rem / 60_000_000 % 60, rem / 1_000_000 % 60);
    format!("{y:04}-{m:02}-{d:02} {h:02}:{mi:02}:{s:02}")
}

/// 16 バイトの UUID を `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` にする。
/// 長さが 16 でない値（本来起き得ない）は 16 進のまま表示する。
fn fmt_uuid(b: &[u8]) -> String {
    let Ok(b): Result<[u8; 16], _> = b.try_into() else {
        return b.iter().map(|x| format!("{x:02x}")).collect();
    };
    let mut s = String::with_capacity(36);
    for (i, byte) in b.iter().enumerate() {
        if matches!(i, 4 | 6 | 8 | 10) {
            s.push('-');
        }
        s.push_str(&format!("{byte:02x}"));
    }
    s
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
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
