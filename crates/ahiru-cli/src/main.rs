//! ネイティブ CLI。
//!
//! 開発とテストはここで回す。wasm 越しのデバッグは効率が悪いので、
//! 「開発とテストはネイティブ、サイズ計測だけ wasm」という分担にしている
//! （DESIGN.md §12）。

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
拡張子から自動判別する。"
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
    println!("splits: {}", t.format.num_splits());

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

fn cmd_query(paths: &[String], sql: &str) -> R {
    let mut s = Session::new();
    // コーデック委譲で元バイトを引くため、テーブル添字をキーに控えておく。
    // 登録は同名なら置き換わるので、添字は register の返り値から取る。
    let mut sources: Vec<Option<Vec<u8>>> = Vec::new();
    let remember = |sources: &mut Vec<Option<Vec<u8>>>, idx: usize, b: &[u8]| {
        if sources.len() <= idx {
            sources.resize(idx + 1, None);
        }
        sources[idx] = Some(b.to_vec());
    };
    for (i, path) in paths.iter().enumerate() {
        let bytes = std::fs::read(path)?;
        // 1 つ目が `t`、2 つ目以降が `t2`, `t3`, ...。
        let name = if i == 0 { String::from("t") } else { format!("t{}", i + 1) };
        let a = s.register_bytes_as(&name, bytes.clone(), FormatKind::detect(path))?;
        remember(&mut sources, a, &bytes);
        // ファイル名でも参照できるようにしておく（`parquet('...')` 用）。
        let b = s.register_bytes(path, bytes.clone())?;
        remember(&mut sources, b, &bytes);
    }

    let mut q = match s.prepare(sql, &[])? {
        Prepared::Ready(q) => q,
        // メモリ上にファイル全体があるので、ここには来ない。
        Prepared::NeedIo(_) => return Err("unexpected io request".into()),
    };
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
            // コーデック委譲。wasm では JS が `DecompressionStream` と
            // `ahiru-zstd.wasm` で処理する部分を、CLI ではここで肩代わりする。
            QueryStep::NeedCodec(reqs) => {
                for r in reqs {
                    let src = source_bytes(&sources, r.table, r.offset, r.len)?;
                    let out = decompress_host(r.codec, src, r.out_len as usize)?;
                    s.provide_decoded(r.table, r.offset, r.len, out)?;
                }
            }
            QueryStep::Done => break,
        }
    }
    eprintln!("({rows} rows)");
    Ok(())
}

fn source_bytes(
    sources: &[Option<Vec<u8>>],
    table: usize,
    offset: u64,
    len: u32,
) -> Result<&[u8], String> {
    let b = sources
        .get(table)
        .and_then(|o| o.as_ref())
        .ok_or_else(|| String::from("unknown table for codec request"))?;
    let s = offset as usize;
    let e = s + len as usize;
    if e > b.len() {
        return Err(String::from("codec request out of range"));
    }
    Ok(&b[s..e])
}

/// ホスト側のコーデック。
///
/// - ZSTD は `ahiru-zstd` を直接リンクする（wasm では別モジュール）。
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
        Value::I64(x) => x.to_string(),
        Value::I128(x) => x.to_string(),
        Value::F64(x) => x.to_string(),
        Value::Bytes(b) => match std::str::from_utf8(b) {
            Ok(s) => s.to_string(),
            Err(_) => format!("<{} bytes>", b.len()),
        },
    }
}

/// エポックからの日数を `YYYY-MM-DD` にする。civil_from_days アルゴリズム。
fn fmt_date(days: i64) -> String {
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}")
}

fn fmt_timestamp(micros: i64) -> String {
    let days = micros.div_euclid(86_400_000_000);
    let rem = micros.rem_euclid(86_400_000_000);
    let (y, m, d) = civil_from_days(days);
    let (h, mi, s) = (rem / 3_600_000_000, rem / 60_000_000 % 60, rem / 1_000_000 % 60);
    format!("{y:04}-{m:02}-{d:02} {h:02}:{mi:02}:{s:02}")
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
