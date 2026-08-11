//! テーブルカタログとバイト供給元。
//!
//! `Source` は「取得済みのバイト範囲の集合」として表現する。この 1 つの型で
//! 2 つの経路を統一できるのが要点:
//!
//! - メモリ上に全体がある場合 … `[0, len)` を覆う範囲が 1 本だけある状態
//! - ホストからレンジ取得する場合 … 必要な範囲が届くたびに増えていく状態
//!
//! 実行側は `get()` を呼ぶだけでよく、`None` が返ったら「その範囲を取ってきて
//! ほしい」と要求を出す（DESIGN.md §6 の RowGroup 境界バリア）。

use crate::format::{self, FormatKind, TableFormat};
use crate::prelude::*;
use crate::rt::hash::eq_ascii_ci;
#[cfg(feature = "ddl")]
use crate::vector::Value;
use crate::vector::{Field, Ty};

/// 取得済みバイト範囲の集合。
pub struct Source {
    pub total_len: u64,
    /// `(開始オフセット, データ)`。開始オフセット昇順に保つ。
    chunks: Vec<(u64, Vec<u8>)>,
    /// ホストに展開してもらったページ。キーは圧縮ページ本体の
    /// `(ファイル上のオフセット, 長さ)`（DESIGN.md §6 のコーデック委譲）。
    decoded: Vec<((u64, u32), Vec<u8>)>,
}

impl Source {
    /// ファイル全体がメモリにある場合。
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Source { total_len: bytes.len() as u64, chunks: vec![(0, bytes)], decoded: Vec::new() }
    }

    /// ホストが保持していて、レンジ取得で読み出す場合。
    pub fn remote(total_len: u64) -> Self {
        Source { total_len, chunks: Vec::new(), decoded: Vec::new() }
    }

    /// `[off, off+len)` が取得済みならそのスライスを返す。
    pub fn get(&self, off: u64, len: usize) -> Option<&[u8]> {
        let end = off.checked_add(len as u64)?;
        // 範囲は多くても数十本なので線形探索で足りる。
        for (start, data) in &self.chunks {
            let cend = start + data.len() as u64;
            if *start <= off && end <= cend {
                let s = (off - start) as usize;
                return Some(&data[s..s + len]);
            }
        }
        None
    }

    /// 取得したバイト列を登録する。隣接・重複する範囲は結合する。
    pub fn insert(&mut self, off: u64, data: Vec<u8>) {
        if data.is_empty() {
            return;
        }
        self.chunks.push((off, data));
        self.chunks.sort_by_key(|(o, _)| *o);
        // 隣接・重複を 1 本にまとめる。放っておくと `get` の線形探索が伸びる。
        let mut merged: Vec<(u64, Vec<u8>)> = Vec::with_capacity(self.chunks.len());
        for (off, data) in self.chunks.drain(..) {
            match merged.last_mut() {
                Some((po, pd)) if off <= *po + pd.len() as u64 => {
                    let overlap = (*po + pd.len() as u64 - off) as usize;
                    if overlap < data.len() {
                        pd.extend_from_slice(&data[overlap..]);
                    }
                }
                _ => merged.push((off, data)),
            }
        }
        self.chunks = merged;
    }

    /// 要求範囲のうち、まだ取得できていない部分を返す。
    pub fn missing(&self, off: u64, len: u64) -> Option<(u64, u64)> {
        if len == 0 || self.get(off, len as usize).is_some() {
            None
        } else {
            Some((off, len.min(self.total_len.saturating_sub(off))))
        }
    }

    pub fn is_complete(&self) -> bool {
        matches!(self.chunks.first(), Some((0, d)) if d.len() as u64 == self.total_len)
    }

    /// ホストが展開したページを登録する。
    pub fn insert_decoded(&mut self, offset: u64, len: u32, data: Vec<u8>) {
        if let Some(e) = self.decoded.iter_mut().find(|(k, _)| *k == (offset, len)) {
            e.1 = data;
            return;
        }
        self.decoded.push(((offset, len), data));
    }

    /// 展開済みページが既にあるか。
    pub fn has_decoded(&self, offset: u64, len: u32) -> bool {
        self.decoded.iter().any(|(k, _)| *k == (offset, len))
    }

    pub fn decoded_bytes(&self) -> usize {
        self.decoded.iter().map(|(_, d)| d.len()).sum()
    }

    /// 展開済みページを捨てる。分割を 1 つ処理し終えたら呼ぶ。
    /// 溜め込むと圧縮前のファイルより大きなメモリを抱えることになる。
    pub fn clear_decoded(&mut self) {
        self.decoded.clear();
    }
}

impl crate::parquet::reader::PageCache for Source {
    fn get(&self, offset: u64, len: u32) -> Option<&[u8]> {
        self.decoded.iter().find(|(k, _)| *k == (offset, len)).map(|(_, d)| d.as_slice())
    }
}

/// テーブルを構成する 1 ファイル分。
///
/// `path` は登録時に渡された名前（多くはファイルパスや URL）で、フォーマット
/// 自動判定と Hive パーティション解析の両方に使う（`session.rs` 側の仕事）。
/// ここでは単なる識別子として持つだけで、意味づけはしない。
pub struct TablePart {
    pub path: String,
    pub source: Source,
    pub format: Box<dyn TableFormat>,
}

/// 登録済みテーブル。
///
/// 1 つの論理テーブルは 1 個以上の `TablePart` からなる（Hive スタイルの
/// パーティションディレクトリのように、複数ファイルが 1 つの表を構成する
/// ケースを表現するため）。各パートは独立に「自分のバイト範囲」を持ち、
/// 独立に解決する。`Scan` オペレータから見ると、これはパート境界をまたいで
/// 分割 (split) 番号を振り直しているだけの平坦な列に見える
/// （`exec::Scan` 参照）。
pub struct Table {
    pub name: String,
    pub parts: Vec<TablePart>,
    /// 全パートを解決した後に確定する統一スキーマ。未解決の間は `None`。
    schema: Option<Vec<Field>>,
}

/// Per-part I/O still needed before `Table::resolve` can finish: `(part index,
/// offset, len)` triples, one per part still missing bytes.
type PendingPartReads = Vec<(usize, u64, u64)>;

impl Table {
    /// 全パートのスキーマを解決する。
    ///
    /// バイトが足りないパートがあれば、その場で 1 個ずつ止まらずに**全パート
    /// を見てから**まとめて要求を返す。ホストが全ファイルのフッタを並列で
    /// 取得できるようにするための工夫（1 パートごとに往復すると、パート数だけ
    /// ラウンドトリップが直列に発生してしまう）。
    ///
    /// 全パートが揃った時点でスキーマの互換性を確認し、統一スキーマを 1 度
    /// だけ計算してキャッシュする。
    pub fn resolve(&mut self) -> Result<core::result::Result<(), PendingPartReads>> {
        let mut need = Vec::new();
        for (i, part) in self.parts.iter_mut().enumerate() {
            match part.format.resolve(&part.source)? {
                Ok(()) => {}
                Err((offset, len)) => need.push((i, offset, len)),
            }
        }
        if !need.is_empty() {
            return Ok(Err(need));
        }
        if self.schema.is_none() {
            self.schema = Some(unify_schema(&self.parts)?);
        }
        Ok(Ok(()))
    }

    /// 統一スキーマが確定しているか。
    pub fn is_resolved(&self) -> bool {
        self.schema.is_some()
    }

    /// 統一スキーマ。最初のパートの列名・並び・NULL 可否を基準に、型は
    /// `Ty::unify` で広げ、NULL 可否はどれか 1 パートでも NULL を許せば
    /// 全体も NULL 許容にする（`unify_schema` 参照）。未解決なら空。
    pub fn schema(&self) -> &[Field] {
        self.schema.as_deref().unwrap_or(&[])
    }

    /// 全パートを通した分割の総数。進捗表示程度にしか使わない。
    pub fn num_splits(&self) -> usize {
        self.parts.iter().map(|p| p.format.num_splits()).sum()
    }
}

/// 全パートのスキーマを 1 つに合わせる。
///
/// 列数が異なる、列名が（大文字小文字を無視して）同じ位置で揃っていない、
/// または列の型を共通化できない組み合わせがあれば `TypeMismatch`。
/// `plan/bind.rs` の `unify_setop_schema`（`UNION` の型合わせ）と同じ考え方
/// だが、`catalog` は `plan` に依存させたくないのでここに独立して置いてある。
///
/// 列名も見るのは意図的: 型だけ見て位置で結合すると、パートごとに列の
/// 並びが違う（が型はたまたま両立する）場合に、意味の異なる列を静かに
/// 1 列として merge してしまう。`Scan`（`exec::mod.rs`）は統一後の列番号を
/// そのまま各パートの物理列番号として使い、`Pruner`（統計プルーニング）も
/// 列番号を埋め込んでいるため、**列番号を基準にパートをまたいで並べ替える
/// ことはできない**（並べ替えるには `Pruner` を含めた列番号の付け替えが
/// パート単位で要り、統計プルーニングの正しさに関わる大きな変更になる）。
/// そのため「並びが違うファイルは受け付けて並べ替える」ではなく「並びが
/// 違うファイルは明確に拒否する」という安全側の設計にしてある。
fn unify_schema(parts: &[TablePart]) -> Result<Vec<Field>> {
    let mut iter = parts.iter();
    let first = match iter.next() {
        Some(p) => p,
        None => return Ok(Vec::new()),
    };
    let mut out: Vec<Field> = first.format.schema().to_vec();
    for part in iter {
        let s = part.format.schema();
        ensure!(s.len() == out.len(), TypeMismatch);
        for (o, f) in out.iter_mut().zip(s) {
            ensure!(eq_ascii_ci(o.name.as_bytes(), f.name.as_bytes()), TypeMismatch);
            let ty = match Ty::unify(o.ty, f.ty) {
                Some(t) => t,
                None => err!(TypeMismatch),
            };
            o.ty = ty;
            o.nullable = o.nullable || f.nullable;
        }
    }
    Ok(out)
}

/// `ddl`/`dml` フィーチャ専用のインメモリ表。
///
/// `Table`（ファイル由来、読み取り専用）とは完全に別系統。`Source` の
/// 不変条件（一度入ったバイトは書き換わらない）に触れずに DML を実現する
/// ための設計で、行の追加・更新・削除はここにしか効かない（DESIGN.md §16）。
///
/// 行指向で持つ: DML は行単位の更新・削除が中心で、列指向にしても
/// このエンジンの主戦場（大きな Parquet ファイルの読み取り）には効かない
/// ので、実装の単純さを優先した。
#[cfg(feature = "ddl")]
pub struct MemTable {
    pub name: String,
    pub schema: Vec<Field>,
    pub rows: Vec<Vec<Value>>,
}

#[cfg(feature = "ddl")]
impl MemTable {
    /// `[start, end)` 行を `Batch` へ変換する。`Scan`（`exec::MemScan`）と
    /// DML（`dml::update`/`dml::delete`）の行 → ベクタ変換をここに集約する。
    /// 常にメモリ上のデータから組み立てるだけなので、`Source` のような
    /// 分割待ち（`NeedIo`）は原理的に起こらない。
    pub fn batch(&self, start: usize, end: usize) -> crate::vector::Batch {
        // 列が 0 本（`ALTER TABLE ... DROP COLUMN` で最後の列を落とした直後）
        // だと `cols` が空になり、`Batch::new` は行数を追跡できない
        // （`num_rows()` は `cols.first()` が無いと `empty_rows`（既定 0）を
        // 見るため、実際の行数を静かに 0 と誤報してしまう）。その場合は
        // `Batch::rows_only` で行数だけを明示的に持たせる。
        if self.schema.is_empty() {
            return crate::vector::Batch::rows_only(end - start);
        }
        let mut cols: Vec<crate::vector::Vector> = self
            .schema
            .iter()
            .map(|f| crate::vector::Vector::with_capacity(f.ty, end - start))
            .collect();
        for row in &self.rows[start..end] {
            for (c, v) in cols.iter_mut().zip(row.iter()) {
                c.push_value(v);
            }
        }
        crate::vector::Batch::new(cols)
    }
}

#[derive(Default)]
pub struct Catalog {
    tables: Vec<Table>,
    #[cfg(feature = "ddl")]
    mem: Vec<MemTable>,
    /// ビューは `(名前, クエリ本体の生 SQL)`。参照されるたびに束縛時
    /// （`plan::bind::flatten_from`）に再パースする。`ExprArena`/`QueryStmt`
    /// を持たせると `catalog` が `sql::ast` に依存してしまうので避けている。
    #[cfg(feature = "ddl")]
    views: Vec<(String, String)>,
}

/// 名前の大文字小文字を無視して線形探索する。`index_of`/`mem_index_of`/
/// `view_index_of` が共有する検索規則（テーブル・ビューを通して同じ）。
fn find_ci_index<'a>(mut names: impl Iterator<Item = &'a str>, target: &str) -> Option<usize> {
    names.position(|n| eq_ascii_ci(n.as_bytes(), target.as_bytes()))
}

impl Catalog {
    pub fn new() -> Self {
        Catalog {
            tables: Vec::new(),
            #[cfg(feature = "ddl")]
            mem: Vec::new(),
            #[cfg(feature = "ddl")]
            views: Vec::new(),
        }
    }

    /// 1 ファイルだけのテーブルを登録する。同名があれば置き換える
    /// （再登録をエラーにしない）。
    ///
    /// この時点では I/O を行わない。スキーマ解決は最初のクエリまで遅延する。
    pub fn register(&mut self, name: &str, source: Source, kind: FormatKind) -> Result<usize> {
        let fmt = format::make(kind, name)?;
        let part = TablePart { path: name.into(), source, format: fmt };
        self.register_multi(name, vec![part])
    }

    /// 複数ファイルを 1 つの論理テーブルとして登録する。
    ///
    /// 各パートのフォーマット（Hive パーティション列のラップも含む）は
    /// 呼び出し側（`session.rs`）が組み立て済みであることを前提にする。
    /// `catalog` はそれをそのまま束ねるだけで、`format::partitioned` の
    /// 存在を知る必要がない。
    pub fn register_multi(&mut self, name: &str, parts: Vec<TablePart>) -> Result<usize> {
        ensure!(!parts.is_empty(), Internal);
        let t = Table { name: name.into(), parts, schema: None };
        Ok(match self.index_of(name) {
            Some(i) => {
                self.tables[i] = t;
                i
            }
            None => {
                self.tables.push(t);
                self.tables.len() - 1
            }
        })
    }

    pub fn index_of(&self, name: &str) -> Option<usize> {
        find_ci_index(self.tables.iter().map(|t| t.name.as_str()), name)
    }

    pub fn get(&self, i: usize) -> Option<&Table> {
        self.tables.get(i)
    }

    pub fn get_mut(&mut self, i: usize) -> Option<&mut Table> {
        self.tables.get_mut(i)
    }

    pub fn len(&self) -> usize {
        self.tables.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tables.iter().map(|t| t.name.as_str())
    }

    // --- インメモリ表（`ddl`） -----------------------------------------------

    #[cfg(feature = "ddl")]
    pub fn mem_index_of(&self, name: &str) -> Option<usize> {
        find_ci_index(self.mem.iter().map(|t| t.name.as_str()), name)
    }

    #[cfg(feature = "ddl")]
    pub fn mem_get(&self, i: usize) -> Option<&MemTable> {
        self.mem.get(i)
    }

    #[cfg(feature = "ddl")]
    pub fn mem_get_mut(&mut self, i: usize) -> Option<&mut MemTable> {
        self.mem.get_mut(i)
    }

    #[cfg(feature = "ddl")]
    pub fn mem_names(&self) -> impl Iterator<Item = &str> {
        self.mem.iter().map(|t| t.name.as_str())
    }

    /// この名前が、書き込み可能な入れ物として空いているか（他のファイル
    /// テーブル/ビューに使われていないか）。`CREATE TABLE`/`CREATE VIEW` の
    /// 衝突検査に使う。
    #[cfg(feature = "ddl")]
    fn name_taken_by_other(&self, name: &str) -> bool {
        self.index_of(name).is_some() || self.view_index_of(name).is_some()
    }

    /// `CREATE TABLE t (...)` / `CREATE TABLE t AS SELECT ...`。
    /// `replace` なら既存の同名インメモリ表を静かに置き換える。
    #[cfg(feature = "ddl")]
    pub fn mem_create(&mut self, name: &str, schema: Vec<Field>, replace: bool) -> Result<usize> {
        ensure!(!self.name_taken_by_other(name), DuplicateTable);
        match self.mem_index_of(name) {
            Some(i) => {
                ensure!(replace, DuplicateTable);
                self.mem[i] = MemTable { name: name.into(), schema, rows: Vec::new() };
                Ok(i)
            }
            None => {
                self.mem.push(MemTable { name: name.into(), schema, rows: Vec::new() });
                Ok(self.mem.len() - 1)
            }
        }
    }

    /// `DROP TABLE t`。ファイルテーブルは対象外（読み取り専用のため常に
    /// `TableNotFound`）。
    #[cfg(feature = "ddl")]
    pub fn mem_drop(&mut self, name: &str) -> Result<()> {
        match self.mem_index_of(name) {
            Some(i) => {
                self.mem.remove(i);
                Ok(())
            }
            None => err!(TableNotFound),
        }
    }

    /// 名前が書き込み可能なインメモリ表を指しているか確認し、添字を返す。
    /// ファイルテーブル（読み取り専用）なら `ReadOnlyTable`、どちらにも
    /// 無ければ `TableNotFound`。`dml::insert`/`update`/`delete` と
    /// `ALTER TABLE` の両方がこの規則を共有する。
    #[cfg(feature = "ddl")]
    pub fn mem_index_writable(&self, name: &str) -> Result<usize> {
        if let Some(i) = self.mem_index_of(name) {
            return Ok(i);
        }
        if self.index_of(name).is_some() {
            err!(ReadOnlyTable);
        }
        err!(TableNotFound)
    }

    /// `ALTER TABLE t ADD COLUMN col ty ...`。列名の重複（大文字小文字無視）
    /// は拒否する。`value` は呼び出し側（`ddl::alter_table`）が既に
    /// DEFAULT を評価済みの値（無ければ `Value::Null`）で、全既存行の
    /// 末尾に同じ値を積む。
    #[cfg(feature = "ddl")]
    pub fn mem_add_column(&mut self, idx: usize, field: Field, value: Value) -> Result<()> {
        let mt = match self.mem.get_mut(idx) {
            Some(t) => t,
            None => err!(TableNotFound),
        };
        ensure!(
            !mt.schema.iter().any(|f| eq_ascii_ci(f.name.as_bytes(), field.name.as_bytes())),
            DuplicateColumn
        );
        mt.schema.push(field);
        for row in &mut mt.rows {
            row.push(value.clone());
        }
        Ok(())
    }

    /// `ALTER TABLE t DROP COLUMN col`。列が無ければ `ColumnNotFound`。
    #[cfg(feature = "ddl")]
    pub fn mem_drop_column(&mut self, idx: usize, col_name: &str) -> Result<()> {
        let mt = match self.mem.get_mut(idx) {
            Some(t) => t,
            None => err!(TableNotFound),
        };
        let pos = match mt
            .schema
            .iter()
            .position(|f| eq_ascii_ci(f.name.as_bytes(), col_name.as_bytes()))
        {
            Some(p) => p,
            None => err!(ColumnNotFound),
        };
        mt.schema.remove(pos);
        for row in &mut mt.rows {
            row.remove(pos);
        }
        Ok(())
    }

    /// `ALTER TABLE t RENAME COLUMN old TO new`。`old` が無ければ
    /// `ColumnNotFound`、`new` が既存の別の列と衝突すれば `DuplicateColumn`。
    #[cfg(feature = "ddl")]
    pub fn mem_rename_column(&mut self, idx: usize, old: &str, new: &str) -> Result<()> {
        let mt = match self.mem.get_mut(idx) {
            Some(t) => t,
            None => err!(TableNotFound),
        };
        let pos =
            match mt.schema.iter().position(|f| eq_ascii_ci(f.name.as_bytes(), old.as_bytes())) {
                Some(p) => p,
                None => err!(ColumnNotFound),
            };
        let conflict = mt
            .schema
            .iter()
            .enumerate()
            .any(|(i, f)| i != pos && eq_ascii_ci(f.name.as_bytes(), new.as_bytes()));
        ensure!(!conflict, DuplicateColumn);
        mt.schema[pos].name = new.into();
        Ok(())
    }

    /// `ALTER TABLE t RENAME TO new_name`。`new_name` が他のファイルテーブル
    /// /ビュー/別のインメモリ表に使われていれば `DuplicateTable`（自分自身
    /// への改名、つまり大文字小文字だけの変更は許す）。
    #[cfg(feature = "ddl")]
    pub fn mem_rename_table(&mut self, idx: usize, new_name: &str) -> Result<()> {
        let taken_by_other_mem = match self.mem_index_of(new_name) {
            Some(i) => i != idx,
            None => false,
        };
        ensure!(!self.name_taken_by_other(new_name) && !taken_by_other_mem, DuplicateTable);
        self.mem[idx].name = new_name.into();
        Ok(())
    }

    // --- ビュー（`ddl`） ------------------------------------------------------

    #[cfg(feature = "ddl")]
    pub fn view_index_of(&self, name: &str) -> Option<usize> {
        find_ci_index(self.views.iter().map(|(n, _)| n.as_str()), name)
    }

    /// ビュー本体（`SELECT ...` の生テキスト）。
    #[cfg(feature = "ddl")]
    pub fn view_get(&self, i: usize) -> Option<&str> {
        self.views.get(i).map(|(_, sql)| sql.as_str())
    }

    #[cfg(feature = "ddl")]
    pub fn view_names(&self) -> impl Iterator<Item = &str> {
        self.views.iter().map(|(n, _)| n.as_str())
    }

    #[cfg(feature = "ddl")]
    pub fn view_create(
        &mut self,
        name: &str,
        query_sql: String,
        or_replace: bool,
    ) -> Result<usize> {
        ensure!(self.index_of(name).is_none() && self.mem_index_of(name).is_none(), DuplicateTable);
        match self.view_index_of(name) {
            Some(i) => {
                ensure!(or_replace, DuplicateTable);
                self.views[i] = (name.into(), query_sql);
                Ok(i)
            }
            None => {
                self.views.push((name.into(), query_sql));
                Ok(self.views.len() - 1)
            }
        }
    }

    #[cfg(feature = "ddl")]
    pub fn view_drop(&mut self, name: &str) -> Result<()> {
        match self.view_index_of(name) {
            Some(i) => {
                self.views.remove(i);
                Ok(())
            }
            None => err!(TableNotFound),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{CodecTask, Pruner, ResolveStep};
    use crate::vector::Vector;

    #[test]
    fn in_memory_source_serves_any_range() {
        let s = Source::from_bytes((0u8..=255).collect());
        assert_eq!(s.get(0, 4), Some(&[0u8, 1, 2, 3][..]));
        assert_eq!(s.get(250, 6), Some(&[250u8, 251, 252, 253, 254, 255][..]));
        assert_eq!(s.get(250, 7), None);
        assert!(s.is_complete());
    }

    #[test]
    fn remote_source_reports_missing_then_serves() {
        let mut s = Source::remote(1000);
        assert_eq!(s.missing(100, 50), Some((100, 50)));
        s.insert(100, vec![7u8; 50]);
        assert_eq!(s.missing(100, 50), None);
        assert_eq!(s.get(120, 10), Some(&[7u8; 10][..]));
        // 別の範囲はまだ無い。
        assert_eq!(s.missing(300, 10), Some((300, 10)));
    }

    #[test]
    fn adjacent_ranges_merge() {
        let mut s = Source::remote(300);
        s.insert(100, vec![1u8; 50]);
        s.insert(150, vec![2u8; 50]);
        // 結合されているので 1 本のスライスとして取り出せる。
        let got = s.get(140, 20).unwrap();
        assert_eq!(&got[..10], &[1u8; 10]);
        assert_eq!(&got[10..], &[2u8; 10]);
    }

    #[test]
    fn overlapping_ranges_merge_without_duplication() {
        let mut s = Source::remote(300);
        s.insert(0, vec![1u8; 100]);
        s.insert(50, vec![2u8; 100]);
        assert_eq!(s.get(0, 150).unwrap().len(), 150);
        // 重なった部分は先着を優先し、後続の非重複部分だけを足す。
        assert_eq!(s.get(0, 150).unwrap()[99], 1);
        assert_eq!(s.get(0, 150).unwrap()[100], 2);
    }

    #[test]
    fn missing_range_is_clamped_to_file_length() {
        let s = Source::remote(120);
        assert_eq!(s.missing(100, 500), Some((100, 20)));
    }

    #[test]
    fn register_replaces_same_name_case_insensitively() {
        let mut c = Catalog::new();
        let a = c.register("Trips", Source::from_bytes(vec![1]), FormatKind::Parquet).unwrap();
        let b = c.register("trips", Source::from_bytes(vec![2]), FormatKind::Parquet).unwrap();
        assert_eq!(a, b);
        assert_eq!(c.len(), 1);
        assert_eq!(c.index_of("TRIPS"), Some(0));
    }

    // --- multi-file 用のモック TableFormat --------------------------------
    //
    // Parquet/CSV の本物のパーサを持ち出さずに、複数パートの束ね方だけを
    // 単体テストしたい。「全体が `total` バイトあり、それが揃うまでは
    // `(0, total)` を要求する」という最小のフォーマットを自作する。

    struct MockFormat {
        schema: Vec<Field>,
        total: u64,
        resolved: bool,
        rows_per_split: u64,
        splits: usize,
    }

    impl MockFormat {
        fn new(schema: Vec<Field>, total: u64, splits: usize, rows_per_split: u64) -> Self {
            MockFormat { schema, total, resolved: false, rows_per_split, splits }
        }
    }

    impl TableFormat for MockFormat {
        fn resolve(&mut self, src: &Source) -> Result<ResolveStep> {
            if self.resolved {
                return Ok(Ok(()));
            }
            if src.get(0, self.total as usize).is_none() {
                return Ok(Err((0, self.total)));
            }
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
            self.splits
        }

        fn split_rows(&self, _split: usize) -> Option<u64> {
            Some(self.rows_per_split)
        }

        fn split_ranges(
            &self,
            _split: usize,
            _projection: &[usize],
            out: &mut Vec<(u64, u64)>,
        ) -> Result<()> {
            out.push((0, self.total));
            Ok(())
        }

        fn codec_tasks(
            &self,
            _src: &Source,
            _split: usize,
            _projection: &[usize],
            _out: &mut Vec<CodecTask>,
        ) -> Result<()> {
            Ok(())
        }

        fn may_match(&self, _split: usize, _pruners: &[Pruner], _projection: &[usize]) -> bool {
            true
        }

        fn read_split(
            &self,
            _src: &Source,
            _split: usize,
            projection: &[usize],
        ) -> Result<Vec<Vector>> {
            let n = self.rows_per_split as usize;
            Ok(projection
                .iter()
                .map(|&c| {
                    let mut v = Vector::with_capacity(self.schema[c].ty, n);
                    for i in 0..n {
                        v.push_value(&crate::vector::Value::I64(i as i64));
                    }
                    v
                })
                .collect())
        }
    }

    fn mock_part(path: &str, schema: Vec<Field>, total: u64) -> TablePart {
        TablePart {
            path: path.into(),
            source: Source::remote(total),
            format: Box::new(MockFormat::new(schema, total, 1, 4)),
        }
    }

    #[test]
    fn resolve_unions_needio_across_parts() {
        let schema = vec![Field::new("id", Ty::BigInt, false)];
        let parts = vec![
            mock_part("a.parquet", schema.clone(), 100),
            mock_part("b.parquet", schema.clone(), 200),
            mock_part("c.parquet", schema, 300),
        ];
        let mut c = Catalog::new();
        let i = c.register_multi("t", parts).unwrap();
        let t = c.get_mut(i).unwrap();

        // どのパートもまだバイトが無いので、3 パート分まとめて要求が返る。
        let need = match t.resolve().unwrap() {
            Err(need) => need,
            Ok(()) => panic!("expected NeedIo"),
        };
        assert_eq!(need.len(), 3);
        let mut sorted = need.clone();
        sorted.sort_by_key(|(p, _, _)| *p);
        assert_eq!(sorted, vec![(0, 0, 100), (1, 0, 200), (2, 0, 300)]);
        assert!(!t.is_resolved());
    }

    #[test]
    fn resolve_progresses_as_parts_arrive_incrementally() {
        let schema = vec![Field::new("id", Ty::BigInt, false)];
        let parts =
            vec![mock_part("a.parquet", schema.clone(), 100), mock_part("b.parquet", schema, 200)];
        let mut c = Catalog::new();
        let i = c.register_multi("t", parts).unwrap();

        // 1 パート目だけ届ける。
        {
            let t = c.get_mut(i).unwrap();
            t.parts[0].source.insert(0, vec![0u8; 100]);
            let need = match t.resolve().unwrap() {
                Err(need) => need,
                Ok(()) => panic!("expected NeedIo"),
            };
            // 2 パート目だけがまだ要る。
            assert_eq!(need, vec![(1, 0, 200)]);
        }

        // 2 パート目も届ける。
        {
            let t = c.get_mut(i).unwrap();
            t.parts[1].source.insert(0, vec![0u8; 200]);
            assert!(t.resolve().unwrap().is_ok());
            assert!(t.is_resolved());
            assert_eq!(t.schema().len(), 1);
        }
    }

    #[test]
    fn schema_mismatch_across_parts_is_rejected() {
        let a_schema = vec![Field::new("id", Ty::BigInt, false)];
        let b_schema =
            vec![Field::new("id", Ty::Varchar, false), Field::new("extra", Ty::Int, true)];
        let mut c = Catalog::new();
        let i = c
            .register_multi(
                "t",
                vec![mock_part("a.parquet", a_schema, 10), mock_part("b.parquet", b_schema, 10)],
            )
            .unwrap();
        let t = c.get_mut(i).unwrap();
        for p in &mut t.parts {
            p.source.insert(0, vec![0u8; 10]);
        }
        // 列数が違うので TypeMismatch。
        assert!(t.resolve().is_err());
    }

    #[test]
    fn schema_type_mismatch_is_rejected_when_unify_fails() {
        // VARCHAR と INTEGER は unify できない組み合わせ。
        let a_schema = vec![Field::new("id", Ty::Varchar, false)];
        let b_schema = vec![Field::new("id", Ty::Int, false)];
        let mut c = Catalog::new();
        let i = c
            .register_multi(
                "t",
                vec![mock_part("a.parquet", a_schema, 10), mock_part("b.parquet", b_schema, 10)],
            )
            .unwrap();
        let t = c.get_mut(i).unwrap();
        for p in &mut t.parts {
            p.source.insert(0, vec![0u8; 10]);
        }
        assert!(t.resolve().is_err());
    }

    #[test]
    fn nullable_widens_and_type_unifies_across_parts() {
        // INT と BIGINT は BIGINT に、NOT NULL と NULL は NULL 許容に広がる。
        let a_schema = vec![Field::new("id", Ty::Int, false)];
        let b_schema = vec![Field::new("id", Ty::BigInt, true)];
        let mut c = Catalog::new();
        let i = c
            .register_multi(
                "t",
                vec![mock_part("a.parquet", a_schema, 10), mock_part("b.parquet", b_schema, 10)],
            )
            .unwrap();
        let t = c.get_mut(i).unwrap();
        for p in &mut t.parts {
            p.source.insert(0, vec![0u8; 10]);
        }
        assert!(t.resolve().unwrap().is_ok());
        let s = t.schema();
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].ty, Ty::BigInt);
        assert!(s[0].nullable);
        // 列名は先頭パートを正とする。
        assert_eq!(s[0].name, "id");
    }

    #[test]
    fn columns_swapped_across_parts_are_rejected_even_when_types_would_unify() {
        // a.parquet: (id INT, region VARCHAR) / b.parquet: (region VARCHAR, id INT)。
        // 位置だけで揃えると INT<->VARCHAR は unify できないので気づけるが、
        // 仮に両方 VARCHAR のような「型が偶然両立する」組み合わせだと、
        // 列名を見ずに位置だけで揃えた場合は意味の違う列を静かに 1 列として
        // merge してしまう。列名の位置一致も要求することでこれを拒否する。
        let a_schema =
            vec![Field::new("id", Ty::Varchar, false), Field::new("region", Ty::Varchar, false)];
        let b_schema =
            vec![Field::new("region", Ty::Varchar, false), Field::new("id", Ty::Varchar, false)];
        let mut c = Catalog::new();
        let i = c
            .register_multi(
                "t",
                vec![mock_part("a.parquet", a_schema, 10), mock_part("b.parquet", b_schema, 10)],
            )
            .unwrap();
        let t = c.get_mut(i).unwrap();
        for p in &mut t.parts {
            p.source.insert(0, vec![0u8; 10]);
        }
        assert!(t.resolve().is_err(), "列の並びが違うパートは型が両立しても拒否されるべき");
    }

    #[test]
    fn column_name_case_differs_across_parts_still_unifies() {
        // 列名の比較は大文字小文字を無視する（`index_of` など、このファイルの
        // 他の名前比較と同じ規約）。
        let a_schema = vec![Field::new("ID", Ty::Int, false)];
        let b_schema = vec![Field::new("id", Ty::Int, false)];
        let mut c = Catalog::new();
        let i = c
            .register_multi(
                "t",
                vec![mock_part("a.parquet", a_schema, 10), mock_part("b.parquet", b_schema, 10)],
            )
            .unwrap();
        let t = c.get_mut(i).unwrap();
        for p in &mut t.parts {
            p.source.insert(0, vec![0u8; 10]);
        }
        assert!(t.resolve().unwrap().is_ok());
    }

    #[test]
    fn single_part_register_is_unchanged() {
        let mut c = Catalog::new();
        let i = c.register("t", Source::from_bytes(vec![1, 2, 3]), FormatKind::Csv).unwrap();
        let t = c.get(i).unwrap();
        assert_eq!(t.parts.len(), 1);
        assert_eq!(t.parts[0].path, "t");
    }

    // --- MemTable の直接単体テスト（`ddl`） -----------------------------------
    //
    // ここまでの ADD/DROP/RENAME COLUMN のテストは全部 `ddl.rs`/統合テスト
    // 経由（SQL 文字列を投げる形）だった。`Catalog::mem_*` メソッドそのものを
    // 直接呼び、schema と各行の長さが常に一致し続けるという不変条件を確認する。

    #[cfg(feature = "ddl")]
    fn mem_catalog_with_two_rows() -> (Catalog, usize) {
        let mut c = Catalog::new();
        let i = c
            .mem_create(
                "t",
                vec![Field::new("a", Ty::Int, false), Field::new("b", Ty::Varchar, true)],
                false,
            )
            .unwrap();
        c.mem_get_mut(i).unwrap().rows = vec![
            vec![Value::I32(1), Value::Bytes(b"x".to_vec())],
            vec![Value::I32(2), Value::Bytes(b"y".to_vec())],
        ];
        (c, i)
    }

    #[cfg(feature = "ddl")]
    fn assert_schema_and_rows_stay_in_sync(c: &Catalog, i: usize) {
        let mt = c.mem_get(i).unwrap();
        for (r, row) in mt.rows.iter().enumerate() {
            assert_eq!(
                row.len(),
                mt.schema.len(),
                "行 {r} の長さ({})がスキーマの列数({})とずれている",
                row.len(),
                mt.schema.len()
            );
        }
    }

    #[cfg(feature = "ddl")]
    #[test]
    fn mem_add_column_backfills_existing_rows_and_rejects_duplicate_name() {
        let (mut c, i) = mem_catalog_with_two_rows();
        c.mem_add_column(i, Field::new("c", Ty::Boolean, true), Value::Bool(true)).unwrap();
        let mt = c.mem_get(i).unwrap();
        assert_eq!(mt.schema.len(), 3);
        assert_eq!(mt.rows[0][2], Value::Bool(true));
        assert_eq!(mt.rows[1][2], Value::Bool(true));
        assert_schema_and_rows_stay_in_sync(&c, i);

        // 既存列名（大文字小文字無視）との衝突は拒否。
        let r = c.mem_add_column(i, Field::new("A", Ty::Int, true), Value::Null);
        assert!(r.is_err());
    }

    #[cfg(feature = "ddl")]
    #[test]
    fn mem_drop_column_removes_slot_from_every_row_down_to_zero_columns() {
        let (mut c, i) = mem_catalog_with_two_rows();
        c.mem_drop_column(i, "b").unwrap();
        assert_eq!(c.mem_get(i).unwrap().schema.len(), 1);
        assert_schema_and_rows_stay_in_sync(&c, i);

        // 最後の1列も落とせる（DuckDB と違い列指向ではないので制約を課していない）。
        c.mem_drop_column(i, "a").unwrap();
        let mt = c.mem_get(i).unwrap();
        assert_eq!(mt.schema.len(), 0);
        assert!(mt.rows.iter().all(|r| r.is_empty()));
        assert_schema_and_rows_stay_in_sync(&c, i);

        // 存在しない列。
        assert!(c.mem_drop_column(i, "nope").is_err());
    }

    #[cfg(feature = "ddl")]
    #[test]
    fn mem_rename_column_updates_name_without_touching_data() {
        let (mut c, i) = mem_catalog_with_two_rows();
        c.mem_rename_column(i, "a", "a2").unwrap();
        let mt = c.mem_get(i).unwrap();
        assert_eq!(mt.schema[0].name, "a2");
        assert_eq!(mt.rows[0][0], Value::I32(1), "データは変わらない");

        // 存在しない旧名。
        assert!(c.mem_rename_column(i, "nope", "x").is_err());
        // 既存の別列名との衝突。
        assert!(c.mem_rename_column(i, "a2", "b").is_err());
    }

    #[cfg(feature = "ddl")]
    #[test]
    fn mem_rename_table_allows_case_only_change_but_rejects_real_collisions() {
        let (mut c, i) = mem_catalog_with_two_rows();
        // 大文字小文字だけの変更（自分自身への改名）は許す。
        c.mem_rename_table(i, "T").unwrap();
        assert_eq!(c.mem_get(i).unwrap().name, "T");

        // 別のインメモリ表・ファイル表との衝突は拒否。
        c.mem_create("other", vec![Field::new("x", Ty::Int, true)], false).unwrap();
        assert!(c.mem_rename_table(i, "other").is_err());
        c.register("filetable", Source::from_bytes(vec![1]), FormatKind::Csv).unwrap();
        assert!(c.mem_rename_table(i, "filetable").is_err());
    }

    #[cfg(feature = "ddl")]
    #[test]
    fn mem_schema_row_length_invariant_survives_add_drop_rename_sequence() {
        let (mut c, i) = mem_catalog_with_two_rows();
        c.mem_add_column(i, Field::new("c", Ty::Boolean, true), Value::Null).unwrap();
        assert_schema_and_rows_stay_in_sync(&c, i);
        c.mem_rename_column(i, "b", "b2").unwrap();
        assert_schema_and_rows_stay_in_sync(&c, i);
        c.mem_drop_column(i, "a").unwrap();
        assert_schema_and_rows_stay_in_sync(&c, i);
        c.mem_add_column(i, Field::new("d", Ty::Varchar, true), Value::Bytes(b"z".to_vec()))
            .unwrap();
        assert_schema_and_rows_stay_in_sync(&c, i);
        let mt = c.mem_get(i).unwrap();
        assert_eq!(mt.schema.len(), 3);
        assert_eq!(mt.schema[0].name, "b2");
        assert_eq!(mt.schema[1].name, "c");
        assert_eq!(mt.schema[2].name, "d");
    }
}
