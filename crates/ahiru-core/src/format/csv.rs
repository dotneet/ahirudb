//! CSV / TSV を `TableFormat` に適合させる読み取り器。
//!
//! 行指向フォーマットなので、Parquet アダプタと違って「統計で分割を落とす」も
//! 「射影で取得バイトを減らす」もできない。できるのは次の 2 つだけ:
//!
//! 1. 固定長バイトチャンクで分割し、I/O を分割境界に刻む
//! 2. 射影されていない列の**変換**を省く（走査はどのみち必要）
//!
//! スキーマは先頭 `SAMPLE_BYTES` からの推定なので後続行で外れることがある。
//! 外れた値はエラーにせず NULL にする。1 行の異常でクエリ全体を落とすより、
//! その 1 セルを NULL にする方が実用上ましだから。

use crate::catalog::Source;
use crate::format::{get_or_internal, ResolveStep, TableFormat, TEXT_MAX_RECORD, TEXT_SPLIT_BYTES};
use crate::prelude::*;
use crate::rt::hash::eq_ascii_ci;
use crate::vector::{Bitmap, Data, Field, Ty, Vector};

/// スキーマ推定のために先頭から要求するバイト数。
///
/// 大きいほど推定は当たるが、最初のクエリの待ち時間がそのまま伸びる。
const SAMPLE_BYTES: u64 = 256 * 1024;

/// 型推定に使う最大行数。サンプルバイトの方が先に尽きることもある。
const SAMPLE_ROWS: usize = 1000;

/// 列バッファの初期確保行数。入力は信用できないので、行数の見積りから
/// 確保量を決めることはしない（巨大な確保を仕込まれないため）。
const ROW_CAP: usize = 256;

pub struct CsvFormat {
    delimiter: u8,
    schema: Vec<Field>,
    /// ヘッダ行の直後（= 最初のデータレコードの先頭）の絶対オフセット。
    data_start: u64,
    total_len: u64,
    resolved: bool,
    /// 1 分割のバイト数。既定は [`TEXT_SPLIT_BYTES`]。
    ///
    /// 小さくすると I/O の往復が増える代わりに 1 分割のメモリが減る。
    /// テストで複数分割を作るのにも使う。
    pub split_bytes: u64,
    /// 1 レコードの上限バイト数。既定は [`TEXT_MAX_RECORD`]。
    ///
    /// 分割境界をまたぐレコードを読み切るための「読み越し量」でもあるので、
    /// これを超えるレコードがあるファイルは分割して読めない（`LimitExceeded`）。
    pub max_record: u64,
}

impl CsvFormat {
    pub fn new(delimiter: u8) -> Self {
        CsvFormat {
            delimiter,
            schema: Vec::new(),
            data_start: 0,
            total_len: 0,
            resolved: false,
            split_bytes: TEXT_SPLIT_BYTES,
            max_record: TEXT_MAX_RECORD,
        }
    }

    pub fn delimiter(&self) -> u8 {
        self.delimiter
    }

    /// データ領域（ヘッダを除いた部分）のバイト数。
    fn data_len(&self) -> u64 {
        self.total_len.saturating_sub(self.data_start)
    }

    /// 0 は分割数の計算を壊すので、常に 1 以上に丸めて使う。
    fn chunk_size(&self) -> u64 {
        self.split_bytes.max(1)
    }

    /// 分割 `split` の境界。
    ///
    /// 返すのは `(所有区間の開始, 所有区間の終了, 取得開始, 取得終了)`。
    /// 所有区間は半開区間で、**そこで始まる**レコードがこの分割のもの。
    ///
    /// 取得範囲が所有区間より広いのには 2 つ理由がある:
    ///
    /// - 後ろ側: 所有区間の末尾で始まったレコードを読み切るための読み越し。
    /// - 前側 1 バイト: 「直前のバイトが行終端か」を知るため。これが無いと、
    ///   レコードがちょうど所有区間の先頭から始まるとき、その 1 行が
    ///   どの分割にも属さずに消える（先頭の行終端まで読み飛ばす規則と、
    ///   半開区間の所有規則が食い違うため）。
    fn chunk(&self, split: usize) -> Result<(u64, u64, u64, u64)> {
        ensure!(split < self.num_splits(), Internal);
        let size = self.chunk_size();
        let off = match (split as u64).checked_mul(size) {
            Some(v) => v,
            None => err!(Internal),
        };
        let start = match self.data_start.checked_add(off) {
            Some(v) => v,
            None => err!(Internal),
        };
        let end = start.saturating_add(size).min(self.total_len);
        let fetch_start = if split == 0 { start } else { start - 1 };
        let fetch_end = end.saturating_add(self.max_record).min(self.total_len);
        Ok((start, end, fetch_start, fetch_end))
    }
}

impl TableFormat for CsvFormat {
    fn resolve(&mut self, src: &Source) -> Result<ResolveStep> {
        if self.resolved {
            return Ok(Ok(()));
        }
        self.total_len = src.total_len;
        if src.total_len == 0 {
            // 空ファイル。列 0・分割 0 として解決済みにする。エラーにはしない。
            self.data_start = 0;
            self.resolved = true;
            return Ok(Ok(()));
        }

        let n = SAMPLE_BYTES.min(src.total_len);
        let sample = match src.get(0, n as usize) {
            Some(b) => b,
            None => return Ok(Err((0, n))),
        };

        // UTF-8 BOM は落とす。残すと先頭列の名前に混ざって列参照が引けなくなる。
        let bom = if sample.starts_with(&[0xEF, 0xBB, 0xBF]) { 3 } else { 0 };
        let body = &sample[bom..];

        // --- ヘッダ行 -------------------------------------------------------
        let mut sc = Scanner::new(body, self.delimiter);
        let mut raw: Vec<Vec<u8>> = Vec::new();
        let mut scratch = Vec::new();
        let terminated = loop {
            let f = sc.field()?;
            raw.push(field_value(body, &f, &mut scratch).to_vec());
            if f.term != Term::Field {
                break f.term == Term::Record;
            }
        };
        // ヘッダがサンプルに収まらないのは異常入力。要求範囲を広げ続けるより
        // 上限として弾く方が、進まないループを作らずに済む。
        ensure!(terminated || n == src.total_len, LimitExceeded);
        self.data_start = bom as u64 + sc.pos() as u64;

        let names = column_names(&raw);

        // --- 型推定 ---------------------------------------------------------
        // 途中で切れた末尾レコードを推定に混ぜないよう、最後の行終端で切る。
        // （引用符内の改行で切ってしまうこともあるが、その場合に起きるのは
        //   「その列が VARCHAR に広がる」だけで、安全側に倒れる。）
        let rest = &body[sc.pos()..];
        let rest = if n < src.total_len {
            match rest.iter().rposition(|&c| c == b'\n') {
                Some(i) => &rest[..i + 1],
                None => &rest[..0],
            }
        } else {
            rest
        };

        let mut cands = vec![Cand::Empty; names.len()];
        let mut sc = Scanner::new(rest, self.delimiter);
        let mut rows = 0;
        'sample: while rows < SAMPLE_ROWS && !sc.at_end() {
            if sc.skip_blank_line() {
                continue;
            }
            let mut fi = 0;
            loop {
                // 壊れたレコードで推定を止めるだけにする。実際に読むときに
                // 同じ場所でエラーになるので、ここで落とす必要はない。
                let f = match sc.field() {
                    Ok(f) => f,
                    Err(_) => break 'sample,
                };
                if let Some(slot) = cands.get_mut(fi) {
                    let v = field_value(rest, &f, &mut scratch);
                    // 空セルは NULL 扱いなので型の証拠にしない。
                    if !v.is_empty() {
                        *slot = widen(*slot, classify(v));
                    }
                }
                fi += 1;
                if f.term != Term::Field {
                    break;
                }
            }
            rows += 1;
        }

        // CSV には NOT NULL を表す手段が無いので全列 nullable。
        self.schema =
            names.into_iter().zip(cands).map(|(name, c)| Field::new(name, c.ty(), true)).collect();
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
        if !self.resolved {
            return 0;
        }
        let len = self.data_len();
        if len == 0 {
            return 0;
        }
        let size = self.chunk_size();
        len.div_ceil(size) as usize
    }

    /// 行数は読むまで分からない。見積りにしか使われないので `None` でよい。
    fn split_rows(&self, _split: usize) -> Option<u64> {
        None
    }

    fn split_ranges(
        &self,
        split: usize,
        _projection: &[usize],
        out: &mut Vec<(u64, u64)>,
    ) -> Result<()> {
        // 射影は無視する。行指向なので、どの列が欲しくても分割の全バイトを
        // 走査しないとフィールド境界すら決まらない。
        let (_, _, fetch_start, fetch_end) = self.chunk(split)?;
        out.push((fetch_start, fetch_end));
        Ok(())
    }

    fn read_split(&self, src: &Source, split: usize, projection: &[usize]) -> Result<Vec<Vector>> {
        ensure!(self.resolved, Internal);
        let (own_start, own_end, fetch_start, fetch_end) = self.chunk(split)?;
        let buf = get_or_internal(src, fetch_start, fetch_end)?;
        let ncols = self.schema.len();

        // 射影に同じ列が 2 度現れても長さの揃った列を返せるよう、まず重複を
        // 除いた集合で組み立てて、最後に並べ直す。
        let mut uniq: Vec<usize> = Vec::with_capacity(projection.len());
        for &c in projection {
            ensure!(c < ncols, Internal);
            if !uniq.contains(&c) {
                uniq.push(c);
            }
        }
        let mut slot_of: Vec<Option<usize>> = vec![None; ncols];
        let mut cols: Vec<ColBuf> = Vec::with_capacity(uniq.len());
        for (slot, &c) in uniq.iter().enumerate() {
            match self.schema.get(c) {
                Some(f) => cols.push(ColBuf::new(f.ty, ROW_CAP)),
                None => err!(Internal),
            }
            if let Some(s) = slot_of.get_mut(c) {
                *s = Some(slot);
            }
        }

        let lead = (own_start - fetch_start) as usize;
        let mut sc = Scanner::new(buf, self.delimiter);
        if lead > 0 {
            // 所有区間の直前が行終端なら、先頭からレコードが始まっている。
            // そうでなければ最初の行終端の直後まで読み飛ばす（そのレコードは
            // 前の分割が読み切っている）。
            //
            // ここが引用符の唯一の弱点で、この走査は引用状態を知り得ない。
            // 引用符の内側の改行を行終端と誤認する可能性があり、
            // 「レコード長 <= max_record」だけでは防げない。分割してよいのは
            // 「引用符内の改行が分割境界の直後に来ない」場合に限られる、
            // というのが残る制約（RFC 4180 の CSV を並列に読む実装が共通して
            // 抱えるもので、ここでも解決していない）。
            match buf.first() {
                Some(&b'\n') => sc.seek(1),
                _ => match buf.iter().skip(1).position(|&c| c == b'\n') {
                    Some(i) => sc.seek(i + 2),
                    // 読み越し範囲内に行終端が無い＝この分割で始まるレコードは
                    // 無い（あるいはレコードが長すぎる）。空を返す。
                    None => return finish(cols, &uniq, projection),
                },
            }
        }

        let limit = (own_end - fetch_start) as usize;
        let at_file_end = fetch_end >= self.total_len;
        let mut scratch = Vec::new();
        while sc.pos() < limit && !sc.at_end() {
            // 空行は行として数えない（末尾の改行が 1 行増やさないのと同じ扱い）。
            if sc.skip_blank_line() {
                continue;
            }
            let mut fi = 0;
            loop {
                let f = match sc.field() {
                    Ok(f) => f,
                    Err(e) => {
                        // バッファ末尾で切れたのなら構文誤りではなく長すぎるレコード。
                        ensure!(at_file_end, LimitExceeded);
                        return Err(e);
                    }
                };
                if let Some(Some(slot)) = slot_of.get(fi) {
                    let v = field_value(buf, &f, &mut scratch);
                    if let Some(col) = cols.get_mut(*slot) {
                        col.push(v, f.quoted);
                    }
                } // 射影外の列は変換しない。走査だけは済んでいる。
                fi += 1;
                if f.term != Term::Field {
                    // 行終端に出会わずバッファが尽きた場合、ファイル末尾なら
                    // それが最終レコード、そうでなければ読み越し不足。
                    ensure!(f.term != Term::Eof || at_file_end, LimitExceeded);
                    break;
                }
            }
            // フィールドが足りない行は残りを NULL にする（多い分は捨てる）。
            for c in fi..ncols {
                if let Some(Some(slot)) = slot_of.get(c) {
                    if let Some(col) = cols.get_mut(*slot) {
                        col.push_null();
                    }
                }
            }
        }

        finish(cols, &uniq, projection)
    }
}

/// 列バッファを射影の並びに戻す。
fn finish(cols: Vec<ColBuf>, uniq: &[usize], projection: &[usize]) -> Result<Vec<Vector>> {
    let vecs: Vec<Vector> = cols.into_iter().map(|c| c.finish()).collect();
    if uniq.len() == projection.len() {
        // 重複が無ければ uniq は projection そのもの。複製は要らない。
        return Ok(vecs);
    }
    let mut out = Vec::with_capacity(projection.len());
    for &c in projection {
        match uniq.iter().position(|&u| u == c).and_then(|i| vecs.get(i)) {
            Some(v) => out.push(v.clone()),
            None => err!(Internal),
        }
    }
    Ok(out)
}

// --- 列名 --------------------------------------------------------------------

/// ヘッダから列名を決める。
///
/// 空・非 UTF-8・重複は `columnN` に置き換える。重複を放置すると列参照が
/// どちらを指すか決められなくなるので、必ず一意にしてから返す。
fn column_names(raw: &[Vec<u8>]) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(raw.len());
    for (i, r) in raw.iter().enumerate() {
        let mut name = match core::str::from_utf8(r) {
            Ok(s) if !s.is_empty() => s.to_owned(),
            _ => generated_name(i),
        };
        if out.iter().any(|p| eq_ascii_ci(p.as_bytes(), name.as_bytes())) {
            name = generated_name(i);
        }
        // 生成名がさらに衝突する（先の列が literally "column3" だった等）
        // 場合に備える。長さが毎回伸びるので必ず有限回で止まる。
        while out.iter().any(|p| eq_ascii_ci(p.as_bytes(), name.as_bytes())) {
            name.push('_');
        }
        out.push(name);
    }
    out
}

/// `column0`, `column1` … を作る。`format!` は使えないので桁を自前で書く。
fn generated_name(i: usize) -> String {
    let mut s = String::from("column");
    let mut digits = [0u8; 20];
    let mut n = 0;
    let mut v = i;
    loop {
        if n >= digits.len() {
            break;
        }
        digits[n] = b'0' + (v % 10) as u8;
        n += 1;
        v /= 10;
        if v == 0 {
            break;
        }
    }
    while n > 0 {
        n -= 1;
        s.push(digits[n] as char);
    }
    s
}

// --- 走査 --------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Term {
    /// 区切り文字。まだフィールドが続く。
    Field,
    /// 行終端。
    Record,
    /// バッファ末尾。
    Eof,
}

/// 1 フィールドの位置。`start..end` は引用符を外した値本体を指す。
struct Span {
    start: usize,
    end: usize,
    quoted: bool,
    /// `""` を含むので取り出すときに畳む必要がある。
    escaped: bool,
    term: Term,
}

struct Scanner<'a> {
    buf: &'a [u8],
    p: usize,
    delim: u8,
}

impl<'a> Scanner<'a> {
    fn new(buf: &'a [u8], delim: u8) -> Self {
        Scanner { buf, p: 0, delim }
    }

    fn pos(&self) -> usize {
        self.p
    }

    fn seek(&mut self, p: usize) {
        self.p = p.min(self.buf.len());
    }

    fn at_end(&self) -> bool {
        self.p >= self.buf.len()
    }

    /// 現在位置が空行なら読み飛ばして真を返す。
    fn skip_blank_line(&mut self) -> bool {
        match self.buf.get(self.p) {
            Some(&b'\n') => {
                self.p += 1;
                true
            }
            Some(&b'\r') if self.buf.get(self.p + 1) == Some(&b'\n') => {
                self.p += 2;
                true
            }
            _ => false,
        }
    }

    /// 1 フィールドを走査する。
    ///
    /// RFC 4180: `"` で始まるフィールドは引用され、内側の `""` は 1 個の `"`、
    /// 区切り文字・`\n`・`\r` は値の一部になる。**引用符の内側の改行は行終端
    /// ではない**。
    fn field(&mut self) -> Result<Span> {
        if self.buf.get(self.p) == Some(&b'"') {
            self.p += 1;
            let start = self.p;
            let mut escaped = false;
            loop {
                match self.buf.get(self.p) {
                    // 閉じない引用符。ここで打ち切らないと残り全部を 1 セルとして
                    // 抱え込むので、構文エラーにする。
                    None => err!(SyntaxError, self.p),
                    Some(&b'"') => {
                        if self.buf.get(self.p + 1) == Some(&b'"') {
                            escaped = true;
                            self.p += 2;
                        } else {
                            let end = self.p;
                            self.p += 1;
                            let term = self.skip_to_term();
                            return Ok(Span { start, end, quoted: true, escaped, term });
                        }
                    }
                    Some(_) => self.p += 1,
                }
            }
        }

        let start = self.p;
        while let Some(&c) = self.buf.get(self.p) {
            if c == self.delim {
                let end = self.p;
                self.p += 1;
                return Ok(Span { start, end, quoted: false, escaped: false, term: Term::Field });
            }
            if c == b'\n' {
                let end = self.p;
                self.p += 1;
                return Ok(Span { start, end, quoted: false, escaped: false, term: Term::Record });
            }
            // CRLF。単独の `\r` は行終端にしない（分割境界の走査が `\n` しか
            // 見ないので、両者の解釈を一致させておく）。
            if c == b'\r' && self.buf.get(self.p + 1) == Some(&b'\n') {
                let end = self.p;
                self.p += 2;
                return Ok(Span { start, end, quoted: false, escaped: false, term: Term::Record });
            }
            self.p += 1;
        }
        Ok(Span { start, end: self.buf.len(), quoted: false, escaped: false, term: Term::Eof })
    }

    /// 閉じ引用符の後ろから区切り／行終端まで進む。
    /// RFC 4180 では引用符の後ろにバイトが続くのは不正だが、捨てて先へ進む。
    fn skip_to_term(&mut self) -> Term {
        while let Some(&c) = self.buf.get(self.p) {
            if c == self.delim {
                self.p += 1;
                return Term::Field;
            }
            if c == b'\n' {
                self.p += 1;
                return Term::Record;
            }
            if c == b'\r' && self.buf.get(self.p + 1) == Some(&b'\n') {
                self.p += 2;
                return Term::Record;
            }
            self.p += 1;
        }
        Term::Eof
    }
}

/// フィールドの値バイト列。`""` を含むときだけ `scratch` に畳んで返す。
fn field_value<'b>(buf: &'b [u8], f: &Span, scratch: &'b mut Vec<u8>) -> &'b [u8] {
    if !f.escaped {
        return buf.get(f.start..f.end).unwrap_or(&[]);
    }
    scratch.clear();
    let mut i = f.start;
    while i < f.end {
        match buf.get(i) {
            Some(&b'"') => {
                scratch.push(b'"');
                i += 2; // `""` を 1 個に畳む
            }
            Some(&c) => {
                scratch.push(c);
                i += 1;
            }
            None => break,
        }
    }
    scratch
}

// --- 型推定 ------------------------------------------------------------------

/// 推定中の候補型。
///
/// 広がる順は BOOLEAN → BIGINT → DOUBLE → VARCHAR。DATE と TIMESTAMP は
/// 別の枝で、混ざれば TIMESTAMP に寄る。VARCHAR は吸収状態。
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(Debug))]
enum Cand {
    /// まだ非空の値を 1 つも見ていない。
    Empty,
    Bool,
    Int,
    Double,
    Date,
    Ts,
    Text,
}

impl Cand {
    fn ty(self) -> Ty {
        match self {
            // 全部空の列は VARCHAR にしておく。数値だと決める根拠が無い。
            Cand::Empty | Cand::Text => Ty::Varchar,
            Cand::Bool => Ty::Boolean,
            Cand::Int => Ty::BigInt,
            Cand::Double => Ty::Double,
            Cand::Date => Ty::Date,
            Cand::Ts => Ty::Timestamp,
        }
    }
}

fn classify(v: &[u8]) -> Cand {
    if parse_bool(v).is_some() {
        Cand::Bool
    } else if parse_i64(v).is_some() {
        Cand::Int
    } else if parse_f64(v).is_some() {
        Cand::Double
    } else if parse_date(v).is_some() {
        Cand::Date
    } else if parse_ts(v).is_some() {
        Cand::Ts
    } else {
        Cand::Text
    }
}

/// 候補型を広げる。格子の外に出る組み合わせはすべて VARCHAR。
///
/// BOOLEAN + BIGINT が BIGINT になるのは格子の定義どおり。`true` は BIGINT に
/// 収まらないので、その行は読み出し時に NULL になる。
fn widen(a: Cand, b: Cand) -> Cand {
    use Cand::*;
    if a == b {
        return a;
    }
    match (a, b) {
        (Empty, x) | (x, Empty) => x,
        (Bool, Int) | (Int, Bool) => Int,
        (Bool, Double) | (Double, Bool) => Double,
        (Int, Double) | (Double, Int) => Double,
        (Date, Ts) | (Ts, Date) => Ts,
        _ => Text,
    }
}

// --- 値の変換 ----------------------------------------------------------------

fn parse_bool(v: &[u8]) -> Option<bool> {
    if eq_ascii_ci(v, b"true") {
        Some(true)
    } else if eq_ascii_ci(v, b"false") {
        Some(false)
    } else {
        None
    }
}

fn parse_i64(v: &[u8]) -> Option<i64> {
    let (neg, ds) = match v.first()? {
        b'-' => (true, v.get(1..)?),
        b'+' => (false, v.get(1..)?),
        _ => (false, v),
    };
    if ds.is_empty() {
        return None;
    }
    // 負側で累算する。そうしないと i64::MIN が表現できない。
    let mut acc: i64 = 0;
    for &c in ds {
        let d = c.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        acc = acc.checked_mul(10)?.checked_sub(d as i64)?;
    }
    if neg {
        Some(acc)
    } else {
        acc.checked_neg()
    }
}

fn parse_f64(v: &[u8]) -> Option<f64> {
    // 形は自前で検査する。`inf` / `NaN` / 16 進を弾き、CSV の数値だけ通したい。
    let mut i = if matches!(v.first(), Some(b'+') | Some(b'-')) { 1 } else { 0 };
    let mut digits = 0usize;
    while let Some(&c) = v.get(i) {
        if !c.is_ascii_digit() {
            break;
        }
        digits += 1;
        i += 1;
    }
    if v.get(i) == Some(&b'.') {
        i += 1;
        while let Some(&c) = v.get(i) {
            if !c.is_ascii_digit() {
                break;
            }
            digits += 1;
            i += 1;
        }
    }
    if digits == 0 {
        return None;
    }
    if matches!(v.get(i), Some(b'e') | Some(b'E')) {
        i += 1;
        if matches!(v.get(i), Some(b'+') | Some(b'-')) {
            i += 1;
        }
        let mut exp = 0usize;
        while let Some(&c) = v.get(i) {
            if !c.is_ascii_digit() {
                break;
            }
            exp += 1;
            i += 1;
        }
        if exp == 0 {
            return None;
        }
    }
    if i != v.len() {
        return None;
    }
    // 値そのものの変換は core の `f64::from_str` に任せる。正しく丸める
    // 10 進→2 進変換を自作すると、誤差かコードサイズのどちらかを損なう。
    core::str::from_utf8(v).ok()?.parse::<f64>().ok()
}

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn days_in_month(y: i64, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap(y) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

/// エポック (1970-01-01) からの日数。Howard Hinnant の days_from_civil。
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    // 3 月始まりに寄せると閏日が年の末尾に来て、場合分けが消える。
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400; // [0, 399]
    let mp = ((m + 9) % 12) as i64; // 3 月 = 0
    let doy = (153 * mp + 2) / 5 + d as i64 - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146097 + doe - 719468
}

/// 10 進数字列を読む。数字以外が混ざれば `None`。
fn dec(b: &[u8]) -> Option<u32> {
    if b.is_empty() {
        return None;
    }
    let mut a = 0u32;
    for &c in b {
        let d = c.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        a = a * 10 + d as u32;
    }
    Some(a)
}

/// 先頭 10 バイトの `YYYY-MM-DD` を読む。
fn parse_ymd(v: &[u8]) -> Option<(i64, u32, u32)> {
    if v.len() < 10 || v.get(4) != Some(&b'-') || v.get(7) != Some(&b'-') {
        return None;
    }
    let y = dec(v.get(0..4)?)? as i64;
    let m = dec(v.get(5..7)?)?;
    let d = dec(v.get(8..10)?)?;
    if !(1..=12).contains(&m) || d < 1 || d > days_in_month(y, m) {
        return None;
    }
    Some((y, m, d))
}

/// DATE はエポックからの日数 (I32)。
fn parse_date(v: &[u8]) -> Option<i32> {
    if v.len() != 10 {
        return None;
    }
    let (y, m, d) = parse_ymd(v)?;
    let days = days_from_civil(y, m, d);
    if days < i32::MIN as i64 || days > i32::MAX as i64 {
        return None;
    }
    Some(days as i32)
}

/// TIMESTAMP はエポックからのマイクロ秒 (I64)。
/// 受け付ける形は `YYYY-MM-DD[ T]HH:MM:SS[.ffffff]`。タイムゾーンは非対応。
fn parse_ts(v: &[u8]) -> Option<i64> {
    if v.len() < 19 {
        return None;
    }
    let (y, m, d) = parse_ymd(v)?;
    if !matches!(v.get(10), Some(b' ') | Some(b'T')) {
        return None;
    }
    if v.get(13) != Some(&b':') || v.get(16) != Some(&b':') {
        return None;
    }
    let h = dec(v.get(11..13)?)?;
    let mi = dec(v.get(14..16)?)?;
    let s = dec(v.get(17..19)?)?;
    if h > 23 || mi > 59 || s > 59 {
        return None;
    }
    let mut us = 0i64;
    let mut i = 19;
    if v.get(19) == Some(&b'.') {
        i = 20;
        let mut scale = 100_000i64;
        while let Some(&c) = v.get(i) {
            if !c.is_ascii_digit() {
                break;
            }
            // 7 桁目以降は捨てる（マイクロ秒精度に正規化する）。
            if scale > 0 {
                us += (c - b'0') as i64 * scale;
                scale /= 10;
            }
            i += 1;
        }
        if i == 20 {
            return None;
        }
    }
    if i != v.len() {
        return None;
    }
    let secs = days_from_civil(y, m, d)
        .checked_mul(86_400)?
        .checked_add((h * 3600 + mi * 60 + s) as i64)?;
    // 負の時刻でも `秒 * 1e6 + 端数` で正しい向きになる（-1 秒 + 999999us = -1us）。
    secs.checked_mul(1_000_000)?.checked_add(us)
}

// --- 列バッファ --------------------------------------------------------------

/// 1 列分の組み立て中バッファ。`Value` を経由しないので行あたりの確保が無い。
struct ColBuf {
    ty: Ty,
    data: Data,
    validity: Bitmap,
}

impl ColBuf {
    fn new(ty: Ty, cap: usize) -> Self {
        ColBuf {
            ty,
            data: Data::with_capacity(ty.phys(), cap),
            validity: Bitmap::with_capacity(cap),
        }
    }

    fn push_null(&mut self) {
        match &mut self.data {
            Data::Bool(b) => b.push(false),
            Data::I32(v) => v.push(0),
            Data::I64(v) => v.push(0),
            Data::F64(v) => v.push(0.0),
            Data::I128(v) => v.push(0),
            Data::Bytes(b) => b.push_empty(),
        }
        self.validity.push(false);
    }

    /// 1 セルを積む。
    ///
    /// 引用符の無い空セルは NULL、`""` は空文字列。両者を区別できるのが
    /// CSV で NULL を表せる唯一の手段なので、ここは崩さない。
    /// 推定した型に合わない値はエラーにせず NULL にする。
    fn push(&mut self, v: &[u8], quoted: bool) {
        if v.is_empty() && !quoted {
            self.push_null();
            return;
        }
        let ok = match &mut self.data {
            Data::Bool(b) => match parse_bool(v) {
                Some(x) => {
                    b.push(x);
                    true
                }
                None => {
                    b.push(false);
                    false
                }
            },
            Data::I32(d) => match if self.ty == Ty::Date { parse_date(v) } else { None } {
                Some(x) => {
                    d.push(x);
                    true
                }
                None => {
                    d.push(0);
                    false
                }
            },
            Data::I64(d) => {
                let x = if self.ty == Ty::Timestamp { parse_ts(v) } else { parse_i64(v) };
                match x {
                    Some(x) => {
                        d.push(x);
                        true
                    }
                    None => {
                        d.push(0);
                        false
                    }
                }
            }
            Data::F64(d) => match parse_f64(v) {
                Some(x) => {
                    d.push(x);
                    true
                }
                None => {
                    d.push(0.0);
                    false
                }
            },
            Data::I128(d) => {
                // 推定が I128 を選ぶことは無い。念のため NULL で埋める。
                d.push(0);
                false
            }
            Data::Bytes(b) => {
                b.push(v);
                true
            }
        };
        self.validity.push(ok);
    }

    fn finish(self) -> Vector {
        let mut v = Vector::from_data(self.ty, self.data, Some(self.validity));
        // NULL が無ければビットマップを捨てて後段の演算を軽くする。
        v.compact_validity();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{code_of, Code};
    use crate::vector::Value;

    fn data_path(name: &str) -> String {
        let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/");
        std::format!("{p}{name}")
    }

    fn read_data(name: &str) -> Vec<u8> {
        std::fs::read(data_path(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
    }

    /// 全分割を読んで縦に連結する。列は `projection` の並び。
    fn read_all(fmt: &CsvFormat, src: &Source, projection: &[usize]) -> Vec<Vec<Value>> {
        let mut out: Vec<Vec<Value>> = projection.iter().map(|_| Vec::new()).collect();
        for s in 0..fmt.num_splits() {
            // 取得範囲の契約どおりのバイトが揃っていることも確かめる。
            let mut ranges = Vec::new();
            fmt.split_ranges(s, projection, &mut ranges).expect("split_ranges");
            for (a, b) in &ranges {
                assert!(src.get(*a, (b - a) as usize).is_some(), "split {s} range missing");
            }
            let cols = fmt.read_split(src, s, projection).expect("read_split");
            assert_eq!(cols.len(), projection.len());
            let n = cols.first().map_or(0, |c| c.len());
            for c in &cols {
                assert_eq!(c.len(), n, "split {s}: 列長が揃っていない");
            }
            for (i, c) in cols.iter().enumerate() {
                for r in 0..n {
                    out[i].push(c.value_at(r));
                }
            }
        }
        out
    }

    /// メモリ上の CSV を解決する。
    fn open(bytes: &[u8], delim: u8) -> (CsvFormat, Source) {
        let src = Source::from_bytes(bytes.to_vec());
        let mut f = CsvFormat::new(delim);
        match f.resolve(&src).expect("resolve") {
            Ok(()) => {}
            Err(r) => panic!("メモリ上のソースで I/O 要求 {r:?}"),
        }
        (f, src)
    }

    fn open_split(bytes: &[u8], delim: u8, split_bytes: u64) -> (CsvFormat, Source) {
        let (mut f, src) = open(bytes, delim);
        f.split_bytes = split_bytes;
        (f, src)
    }

    fn names(f: &CsvFormat) -> Vec<&str> {
        f.schema().iter().map(|c| c.name.as_str()).collect()
    }

    fn types(f: &CsvFormat) -> Vec<Ty> {
        f.schema().iter().map(|c| c.ty).collect()
    }

    fn text(v: &Value) -> String {
        match v {
            Value::Bytes(b) => String::from_utf8_lossy(b).into_owned(),
            _ => panic!("bytes ではない"),
        }
    }

    // --- ヘッダ -------------------------------------------------------------

    #[test]
    fn header_names_are_made_unique() {
        let (f, _) = open(b"a,,a,B,column2\n1,2,3,4,5\n", b',');
        // 空 → column1、重複（大小無視）→ column2。5 列目は自分より前に
        // 生成された "column2" と衝突するので、こちらが column4 になる。
        assert_eq!(names(&f), vec!["a", "column1", "column2", "B", "column4"]);
        // 生成名どうしがさらに衝突する場合は `_` を足して一意にする。
        let (g, _) = open(b"x,column1,x\n1,2,3\n", b',');
        assert_eq!(names(&g), vec!["x", "column1", "column2"]);
        let (h, _) = open(b"column2,a,a\n1,2,3\n", b',');
        assert_eq!(names(&h), vec!["column2", "a", "column2_"]);
    }

    #[test]
    fn header_only_file_has_schema_but_no_splits() {
        let (f, _) = open(b"a,b,c\n", b',');
        assert_eq!(names(&f), vec!["a", "b", "c"]);
        assert_eq!(f.num_splits(), 0);
        assert!(f.is_resolved());
    }

    #[test]
    fn header_without_trailing_newline() {
        let (f, _) = open(b"a,b", b',');
        assert_eq!(names(&f), vec!["a", "b"]);
        assert_eq!(f.num_splits(), 0);
    }

    #[test]
    fn empty_file_resolves_to_empty_schema() {
        let (f, _) = open(b"", b',');
        assert!(f.is_resolved());
        assert!(f.schema().is_empty());
        assert_eq!(f.num_splits(), 0);
    }

    #[test]
    fn quoted_header_names() {
        let (f, _) = open(b"\"a,1\",\"b\"\"c\"\n1,2\n", b',');
        assert_eq!(names(&f), vec!["a,1", "b\"c"]);
    }

    #[test]
    fn tab_delimiter() {
        let (f, src) = open(b"a\tb\n1\tx\n", b'\t');
        assert_eq!(names(&f), vec!["a", "b"]);
        assert_eq!(types(&f), vec![Ty::BigInt, Ty::Varchar]);
        let got = read_all(&f, &src, &[0, 1]);
        assert_eq!(got[0], vec![Value::I64(1)]);
    }

    // --- BOM / CRLF ---------------------------------------------------------

    #[test]
    fn bom_is_stripped_from_first_column_name() {
        let (f, src) = open("\u{feff}id,name\n1,x\n".as_bytes(), b',');
        assert_eq!(names(&f), vec!["id", "name"]);
        let got = read_all(&f, &src, &[0, 1]);
        assert_eq!(got[0], vec![Value::I64(1)]);
    }

    #[test]
    fn crlf_line_endings() {
        let (f, src) = open(b"a,b\r\n1,x\r\n2,y\r\n", b',');
        assert_eq!(names(&f), vec!["a", "b"]);
        assert_eq!(types(&f), vec![Ty::BigInt, Ty::Varchar]);
        let got = read_all(&f, &src, &[0, 1]);
        assert_eq!(got[0], vec![Value::I64(1), Value::I64(2)]);
        // `\r` が値に残っていないこと。
        assert_eq!(text(&got[1][0]), "x");
        assert_eq!(text(&got[1][1]), "y");
    }

    #[test]
    fn bom_with_crlf_and_split_boundaries() {
        let mut s = String::from("\u{feff}a,b\r\n");
        for i in 0..200 {
            s.push_str(&std::format!("{i},v{i}\r\n"));
        }
        let (f, src) = open_split(s.as_bytes(), b',', 16);
        assert!(f.num_splits() > 5);
        let got = read_all(&f, &src, &[0]);
        assert_eq!(got[0].len(), 200);
        assert_eq!(got[0][199], Value::I64(199));
    }

    #[test]
    fn blank_lines_are_skipped() {
        let (f, src) = open(b"a,b\n1,x\n\n2,y\n\n", b',');
        let got = read_all(&f, &src, &[0]);
        assert_eq!(got[0], vec![Value::I64(1), Value::I64(2)]);
    }

    // --- 型推定 -------------------------------------------------------------

    #[test]
    fn infers_each_type() {
        let csv = b"b,i,f,s,d,t\ntrue,1,1.5,x,2024-01-02,2024-01-02 03:04:05\n\
                    false,-2,-0.5e1,y,2000-02-29,1999-12-31T23:59:59.5\n";
        let (f, _) = open(csv, b',');
        assert_eq!(
            types(&f),
            vec![Ty::Boolean, Ty::BigInt, Ty::Double, Ty::Varchar, Ty::Date, Ty::Timestamp]
        );
    }

    #[test]
    fn widening_transitions() {
        // BOOLEAN → BIGINT → DOUBLE → VARCHAR、および DATE → TIMESTAMP。
        let cases: &[(&[u8], Ty)] = &[
            (b"c\ntrue\nfalse\n", Ty::Boolean),
            (b"c\ntrue\n1\n", Ty::BigInt),
            (b"c\ntrue\n1.5\n", Ty::Double),
            (b"c\ntrue\nx\n", Ty::Varchar),
            (b"c\n1\n2\n", Ty::BigInt),
            (b"c\n1\n2.5\n", Ty::Double),
            (b"c\n1\nx\n", Ty::Varchar),
            (b"c\n1.5\nx\n", Ty::Varchar),
            (b"c\n2024-01-01\n2024-01-02\n", Ty::Date),
            (b"c\n2024-01-01\n2024-01-02 00:00:00\n", Ty::Timestamp),
            (b"c\n2024-01-01\n1\n", Ty::Varchar),
            (b"c\n2024-01-01 00:00:00\nx\n", Ty::Varchar),
            // i64 に収まらない整数は DOUBLE に落ちる。
            (b"c\n99999999999999999999\n", Ty::Double),
            // 全部空なら VARCHAR のまま。
            (b"c\n\n\n", Ty::Varchar),
        ];
        for (csv, ty) in cases {
            let (f, _) = open(csv, b',');
            assert_eq!(types(&f), vec![*ty], "{}", String::from_utf8_lossy(csv));
        }
    }

    #[test]
    fn inference_uses_only_the_first_rows() {
        // 先頭 SAMPLE_ROWS 行は整数、その後に文字列を混ぜる。
        let mut s = String::from("c\n");
        for i in 0..SAMPLE_ROWS {
            s.push_str(&std::format!("{i}\n"));
        }
        s.push_str("oops\n");
        let (f, src) = open(s.as_bytes(), b',');
        assert_eq!(types(&f), vec![Ty::BigInt]);
        // 推定を外した行はエラーではなく NULL。
        let got = read_all(&f, &src, &[0]);
        assert_eq!(got[0].len(), SAMPLE_ROWS + 1);
        assert_eq!(got[0][SAMPLE_ROWS], Value::Null);
    }

    #[test]
    fn value_parsers() {
        assert_eq!(parse_i64(b"-9223372036854775808"), Some(i64::MIN));
        assert_eq!(parse_i64(b"9223372036854775808"), None);
        assert_eq!(parse_i64(b"+7"), Some(7));
        assert_eq!(parse_i64(b""), None);
        assert_eq!(parse_i64(b"-"), None);
        assert_eq!(parse_i64(b"1 "), None);
        assert_eq!(parse_f64(b"1.5"), Some(1.5));
        assert_eq!(parse_f64(b"-.5"), Some(-0.5));
        assert_eq!(parse_f64(b"1e3"), Some(1000.0));
        assert_eq!(parse_f64(b"inf"), None);
        assert_eq!(parse_f64(b"NaN"), None);
        assert_eq!(parse_f64(b"1e"), None);
        assert_eq!(parse_f64(b"0x10"), None);
        // duckdb: datediff('day', DATE '1970-01-01', DATE '2024-01-01') = 19723
        assert_eq!(parse_date(b"2024-01-01"), Some(19723));
        assert_eq!(parse_date(b"1900-03-01"), Some(-25508));
        assert_eq!(parse_date(b"2023-02-29"), None);
        assert_eq!(parse_date(b"2024-02-29"), Some(19782));
        assert_eq!(parse_date(b"2024-1-1"), None);
        // duckdb: epoch_us(TIMESTAMP '2024-01-01 00:00:00') = 1704067200000000
        assert_eq!(parse_ts(b"2024-01-01 00:00:00"), Some(1_704_067_200_000_000));
        assert_eq!(parse_ts(b"2024-01-01T00:00:00"), Some(1_704_067_200_000_000));
        assert_eq!(parse_ts(b"1969-12-31 23:59:59.999999"), Some(-1));
        // 7 桁目以降は切り捨て。
        assert_eq!(parse_ts(b"1970-01-01 00:00:00.1234567"), Some(123_456));
        assert_eq!(parse_ts(b"2024-01-01 24:00:00"), None);
        assert_eq!(parse_ts(b"2024-01-01 00:00:00Z"), None);
    }

    // --- 引用符 -------------------------------------------------------------

    #[test]
    fn quoted_file() {
        let bytes = read_data("quoted.csv");
        let (f, src) = open(&bytes, b',');
        assert_eq!(names(&f), vec!["a", "b", "c"]);
        assert_eq!(types(&f), vec![Ty::BigInt, Ty::Varchar, Ty::Double]);
        let got = read_all(&f, &src, &[0, 1, 2]);
        assert_eq!(got[0], vec![Value::I64(1), Value::I64(2), Value::I64(3)]);
        assert_eq!(text(&got[1][0]), "he said \"hi\"");
        // 引用符の内側の改行は行終端ではない。
        assert_eq!(text(&got[1][1]), "multi\nline");
        assert_eq!(got[1][2], Value::Null);
        assert_eq!(got[2], vec![Value::F64(2.5), Value::F64(3.5), Value::F64(4.5)]);
    }

    #[test]
    fn quoted_fields_contain_delimiter_and_crlf() {
        let (f, src) = open(b"a,b\n\"x,y\",\"p\r\nq\"\n", b',');
        let got = read_all(&f, &src, &[0, 1]);
        assert_eq!(text(&got[0][0]), "x,y");
        assert_eq!(text(&got[1][0]), "p\r\nq");
    }

    #[test]
    fn empty_versus_quoted_empty() {
        let (f, src) = open(b"a,b\n,\"\"\n", b',');
        let got = read_all(&f, &src, &[0, 1]);
        // 引用符なしの空は NULL、`""` は空文字列。
        assert_eq!(got[0][0], Value::Null);
        assert_eq!(got[1][0], Value::Bytes(vec![]));
    }

    #[test]
    fn quoted_numbers_still_infer_numeric() {
        let (f, src) = open(b"a\n\"1\"\n\"2\"\n", b',');
        assert_eq!(types(&f), vec![Ty::BigInt]);
        let got = read_all(&f, &src, &[0]);
        assert_eq!(got[0], vec![Value::I64(1), Value::I64(2)]);
    }

    // --- 分割 ---------------------------------------------------------------

    #[test]
    fn split_ranges_ask_for_the_overhang() {
        let mut s = String::from("a\n");
        for i in 0..1000 {
            s.push_str(&std::format!("{i}\n"));
        }
        let (mut f, _) = open(s.as_bytes(), b',');
        f.split_bytes = 100;
        f.max_record = 50;
        let n = f.num_splits();
        assert!(n > 5);
        let mut r = Vec::new();
        f.split_ranges(1, &[0], &mut r).unwrap();
        let (a, b) = r[0];
        let ds = f.data_start;
        // 直前 1 バイトを含み、後ろは max_record ぶん読み越す。
        assert_eq!(a, ds + 100 - 1);
        assert_eq!(b, ds + 200 + 50);
        // 射影を変えても範囲は変わらない（行指向なので減らせない）。
        let mut r2 = Vec::new();
        f.split_ranges(1, &[], &mut r2).unwrap();
        assert_eq!(r2[0], (a, b));
    }

    #[test]
    fn single_split_reads_basic_csv() {
        let bytes = read_data("basic.csv");
        let (f, src) = open(&bytes, b',');
        assert_eq!(f.num_splits(), 1, "8MiB 分割なら 1 つに収まる");
        assert_eq!(names(&f), vec!["id", "name", "score", "flag", "big", "d"]);
        // duckdb の DESCRIBE と一致する（id は BIGINT に推定される）。
        assert_eq!(
            types(&f),
            vec![Ty::BigInt, Ty::Varchar, Ty::Double, Ty::Boolean, Ty::BigInt, Ty::Timestamp]
        );
        let got = read_all(&f, &src, &[0, 4]);
        assert_eq!(got[0].len(), 1000);
        // duckdb: count(big) = 800、5 行おきに NULL。
        let nulls = got[1].iter().filter(|v| **v == Value::Null).count();
        assert_eq!(nulls, 200);
    }

    #[test]
    fn basic_csv_matches_the_parquet_file() {
        // Parquet 側も `TableFormat` 越しに読む。両者を同じ入口で比べられるのが
        // この抽象を挟んだ意味なので、テストもその形にしておく。
        let psrc = Source::from_bytes(read_data("basic.parquet"));
        let mut pf = crate::format::parquet::ParquetFormat::new();
        assert_eq!(pf.resolve(&psrc).expect("parquet resolve"), Ok(()));
        let proj: Vec<usize> = (0..pf.schema().len()).collect();
        let mut expect: Vec<Vec<Value>> = proj.iter().map(|_| Vec::new()).collect();
        for s in 0..pf.num_splits() {
            let cols = pf.read_split(&psrc, s, &proj).expect("parquet split");
            for (i, c) in cols.iter().enumerate() {
                for r in 0..c.len() {
                    expect[i].push(c.value_at(r));
                }
            }
        }

        let bytes = read_data("basic.csv");
        // 分割をまたいでも一致すること（境界の取りこぼしを検出する）。
        let (f, src) = open_split(&bytes, b',', 4096);
        assert!(f.num_splits() > 5);
        let got = read_all(&f, &src, &[0, 1, 2, 3, 4, 5]);

        for c in 0..6 {
            assert_eq!(got[c].len(), 1000, "列 {c} の行数");
            for r in 0..1000 {
                // Parquet 側の id は INTEGER なので i64 に揃えて比べる。
                let e = match &expect[c][r] {
                    Value::I32(x) if c == 0 => Value::I64(*x as i64),
                    v => v.clone(),
                };
                assert_eq!(got[c][r], e, "列 {c} 行 {r}");
            }
        }
    }

    #[test]
    fn multi_split_row_counts_and_boundary_values() {
        let bytes = read_data("multi_rg.csv");
        // 既定の分割サイズでは 1 分割なので、小さくして境界を作る。
        let (f, src) = open_split(&bytes, b',', 64 * 1024);
        assert!(f.num_splits() >= 16, "分割数 {}", f.num_splits());

        // 分割ごとの行数を足すと全体になること、境界の行が欠けも重複もしないこと。
        let mut total = 0usize;
        let mut next = 0i64;
        for s in 0..f.num_splits() {
            let cols = f.read_split(&src, s, &[0, 3]).expect("read_split");
            let n = cols[0].len();
            for r in 0..n {
                // id は 0 から連番。1 つでもずれれば境界処理の誤り。
                assert_eq!(cols[0].value_at(r), Value::I64(next), "split {s} 行 {r}");
                assert_eq!(cols[1].value_at(r), Value::F64(next as f64 * 0.25));
                next += 1;
            }
            total += n;
        }
        // duckdb: count(*) = 50000
        assert_eq!(total, 50_000);
    }

    #[test]
    fn split_size_does_not_change_the_result() {
        let bytes = read_data("multi_rg.csv");
        let (f1, s1) = open_split(&bytes, b',', 1 << 20);
        let (f2, s2) = open_split(&bytes, b',', 997);
        let a = read_all(&f1, &s1, &[0, 1, 2, 3]);
        let b = read_all(&f2, &s2, &[0, 1, 2, 3]);
        assert!(f2.num_splits() > f1.num_splits());
        for c in 0..4 {
            assert_eq!(a[c].len(), 50_000);
            assert_eq!(a[c], b[c], "列 {c}");
        }
    }

    #[test]
    fn records_straddling_every_boundary() {
        // 1 レコード 8 バイト、分割 8 バイトだと境界がレコード先頭に一致する。
        // 半開区間の所有規則を守れていないと、ここで 1 行ずつ消える。
        let mut s = String::from("a,b\n");
        for i in 0..64 {
            s.push_str(&std::format!("{i:03},{i:03}\n"));
        }
        for size in 1..=17u64 {
            let (f, src) = open_split(s.as_bytes(), b',', size);
            let got = read_all(&f, &src, &[0]);
            assert_eq!(got[0].len(), 64, "split_bytes={size}");
            for (i, v) in got[0].iter().enumerate() {
                assert_eq!(*v, Value::I64(i as i64), "split_bytes={size}");
            }
        }
    }

    #[test]
    fn last_record_without_trailing_newline() {
        let (f, src) = open_split(b"a\n1\n2\n3", b',', 2);
        let got = read_all(&f, &src, &[0]);
        assert_eq!(got[0], vec![Value::I64(1), Value::I64(2), Value::I64(3)]);
    }

    #[test]
    fn record_longer_than_max_record_is_rejected() {
        let mut s = String::from("a,b\n");
        s.push_str("1,");
        for _ in 0..200 {
            s.push('x');
        }
        s.push('\n');
        s.push_str("2,y\n");
        let (mut f, src) = open(s.as_bytes(), b',');
        f.split_bytes = 8;
        f.max_record = 16;
        // 長いレコードにぶつかる分割が LimitExceeded を返す。
        let bad =
            (0..f.num_splits()).filter_map(|s| code_of(f.read_split(&src, s, &[0, 1]))).next();
        assert_eq!(bad, Some(Code::LimitExceeded));
    }

    // --- 射影 ---------------------------------------------------------------

    #[test]
    fn projection_returns_requested_columns_in_order() {
        let (f, src) = open(b"a,b,c\n1,x,2.5\n3,y,4.5\n", b',');
        let got = read_all(&f, &src, &[2, 0]);
        assert_eq!(got.len(), 2);
        assert_eq!(got[0], vec![Value::F64(2.5), Value::F64(4.5)]);
        assert_eq!(got[1], vec![Value::I64(1), Value::I64(3)]);
        // 空の射影でも列 0 個を返す（行数は呼び出し側が知る術が無い）。
        let cols = f.read_split(&src, 0, &[]).unwrap();
        assert!(cols.is_empty());
        // 同じ列を 2 度要求しても長さの揃った列が返る。
        let dup = read_all(&f, &src, &[1, 1]);
        assert_eq!(dup[0], dup[1]);
        assert_eq!(text(&dup[0][0]), "x");
        // 範囲外の列は内部エラー。
        assert_eq!(code_of(f.read_split(&src, 0, &[9])), Some(Code::Internal));
    }

    // --- 異常入力 -----------------------------------------------------------

    #[test]
    fn unterminated_quote_is_a_syntax_error() {
        let (f, src) = open(b"a,b\n1,\"oops\n", b',');
        assert_eq!(code_of(f.read_split(&src, 0, &[0, 1])), Some(Code::SyntaxError));
    }

    #[test]
    fn short_and_long_rows() {
        // 足りないフィールドは NULL、多い分は捨てる。
        let (f, src) = open(b"a,b,c\n1,x,2.5\n2\n3,y,4.5,extra\n", b',');
        let got = read_all(&f, &src, &[0, 1, 2]);
        assert_eq!(got[0], vec![Value::I64(1), Value::I64(2), Value::I64(3)]);
        assert_eq!(got[1][1], Value::Null);
        assert_eq!(got[2][1], Value::Null);
        assert_eq!(got[2][2], Value::F64(4.5));
    }

    #[test]
    fn values_that_do_not_match_the_inferred_type_become_null() {
        // 推定はサンプル行しか見ないので、外れる行はサンプルより後ろに置く。
        let mut s = String::from("i,b,d,t\n");
        for _ in 0..SAMPLE_ROWS {
            s.push_str("1,true,2024-01-01,2024-01-01 00:00:00\n");
        }
        s.push_str("x,x,x,x\n");
        // 引用符付きの空文字列も、VARCHAR 以外の列では NULL になる。
        s.push_str("\"\",\"\",\"\",\"\"\n");
        let (f, src) = open(s.as_bytes(), b',');
        assert_eq!(types(&f), vec![Ty::BigInt, Ty::Boolean, Ty::Date, Ty::Timestamp]);
        let got = read_all(&f, &src, &[0, 1, 2, 3]);
        for (c, col) in got.iter().enumerate() {
            assert_ne!(col[0], Value::Null, "列 {c}");
            assert_eq!(col[SAMPLE_ROWS], Value::Null, "列 {c}");
            assert_eq!(col[SAMPLE_ROWS + 1], Value::Null, "列 {c}");
        }
    }

    #[test]
    fn garbage_after_a_closing_quote_is_dropped() {
        let (f, src) = open(b"a,b\n\"x\"junk,1\n", b',');
        let got = read_all(&f, &src, &[0, 1]);
        assert_eq!(text(&got[0][0]), "x");
        assert_eq!(got[1][0], Value::I64(1));
    }

    #[test]
    fn binary_garbage_does_not_panic() {
        let mut bytes = std::vec![b'a', b',', b'b', b'\n'];
        for i in 0u32..4096 {
            bytes.push((i.wrapping_mul(2654435761) >> 13) as u8);
        }
        let src = Source::from_bytes(bytes);
        let mut f = CsvFormat::new(b',');
        if f.resolve(&src).is_ok() && f.is_resolved() {
            f.split_bytes = 64;
            for s in 0..f.num_splits() {
                let _ = f.read_split(&src, s, &[0, 1]);
            }
        }
    }

    // --- 解決の I/O ---------------------------------------------------------

    #[test]
    fn resolve_requests_the_sample_range_first() {
        let mut f = CsvFormat::new(b',');
        let src = Source::remote(10 * 1024 * 1024);
        match f.resolve(&src).unwrap() {
            Err((off, len)) => {
                assert_eq!(off, 0);
                assert_eq!(len, SAMPLE_BYTES);
            }
            Ok(()) => panic!("バイトが無いのに解決できるはずがない"),
        }
        assert!(!f.is_resolved());
        assert_eq!(f.num_splits(), 0);
        assert!(f.schema().is_empty());
    }

    #[test]
    fn resolve_requests_only_the_file_when_shorter_than_the_sample() {
        let mut f = CsvFormat::new(b',');
        let src = Source::remote(10);
        assert_eq!(f.resolve(&src).unwrap(), Err((0, 10)));
    }

    #[test]
    fn resolve_progresses_once_the_sample_arrives() {
        let body = b"a,b\n1,x\n";
        let mut src = Source::remote(body.len() as u64);
        let mut f = CsvFormat::new(b',');
        let (off, len) = f.resolve(&src).unwrap().expect_err("最初は要求");
        src.insert(off, body[off as usize..(off + len) as usize].to_vec());
        assert_eq!(f.resolve(&src).unwrap(), Ok(()));
        assert_eq!(names(&f), vec!["a", "b"]);
    }

    #[test]
    fn header_longer_than_the_sample_is_rejected() {
        // ヘッダに改行が無いまま SAMPLE_BYTES を超えるファイル。
        let mut bytes = std::vec![b'a'; (SAMPLE_BYTES + 10) as usize];
        bytes[SAMPLE_BYTES as usize + 5] = b'\n';
        let src = Source::from_bytes(bytes);
        let mut f = CsvFormat::new(b',');
        assert_eq!(code_of(f.resolve(&src)), Some(Code::LimitExceeded));
    }
}
