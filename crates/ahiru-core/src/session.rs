//! セッション: カタログを持ち、SQL を受け取ってバッチを返す。
//!
//! 非同期 I/O は「実行を止めて必要なバイト範囲を返す」形で表現する。
//! Asyncify を使わずに済むので wasm のコードサイズが増えない（DESIGN.md §6）。

use crate::catalog::{Catalog, Source};
use crate::exec::{build, CodecRequest, ExecContext, IoRequest, Operator, Step};
use crate::expr::vm::Vm;
use crate::format::FormatKind;
use crate::plan::bind::{bind_select, referenced_tables, resolve_from};
use crate::prelude::*;
use crate::sql::ast::{FromItem, Stmt};
use crate::sql::parse;
use crate::vector::{Batch, Field, Value};

/// 準備済みクエリ。
pub struct Query {
    root: Box<dyn Operator>,
    pub schema: Vec<Field>,
}

/// `Session::prepare` の結果。
pub enum Prepared {
    Ready(Query),
    /// フッタを読むためにバイトが足りない。
    NeedIo(Vec<IoRequest>),
}

/// `Session::step` の結果。
pub enum QueryStep {
    Batch(Batch),
    NeedIo(Vec<IoRequest>),
    /// 内蔵していないコーデックの展開をホストに依頼する（DESIGN.md §6）。
    NeedCodec(Vec<CodecRequest>),
    Done,
}

pub struct Session {
    pub catalog: Catalog,
    vm: Vm,
}

impl Session {
    pub fn new() -> Self {
        Session { catalog: Catalog::new(), vm: Vm::new() }
    }

    /// ファイル全体をメモリに持つテーブルを登録する。
    /// フォーマットは名前（拡張子）から推定する。
    pub fn register_bytes(&mut self, name: &str, bytes: Vec<u8>) -> Result<usize> {
        self.register_bytes_as(name, bytes, FormatKind::Auto)
    }

    pub fn register_bytes_as(
        &mut self,
        name: &str,
        bytes: Vec<u8>,
        kind: FormatKind,
    ) -> Result<usize> {
        self.catalog.register(name, Source::from_bytes(bytes), kind)
    }

    /// ホストがレンジ取得で供給するテーブルを登録する。I/O は発生しない。
    pub fn register_remote(&mut self, name: &str, total_len: u64) -> Result<usize> {
        self.register_remote_as(name, total_len, FormatKind::Auto)
    }

    pub fn register_remote_as(
        &mut self,
        name: &str,
        total_len: u64,
        kind: FormatKind,
    ) -> Result<usize> {
        self.catalog.register(name, Source::remote(total_len), kind)
    }

    /// `NeedIo` で要求したバイト列を渡す。
    pub fn provide(&mut self, table: usize, offset: u64, data: Vec<u8>) -> Result<()> {
        match self.catalog.get_mut(table) {
            Some(t) => {
                t.source.insert(offset, data);
                Ok(())
            }
            None => err!(TableNotFound),
        }
    }

    /// SQL をプランに落とす。フッタ未取得ならバイト範囲を要求して戻る。
    pub fn prepare(&mut self, sql: &str, params: &[Value]) -> Result<Prepared> {
        let parsed = parse(sql)?;
        let sel = match &parsed.stmt {
            Stmt::Select(s) => s,
            _ => err!(UnsupportedFeature),
        };
        let from = match &sel.from {
            Some(f) => f,
            None => err!(UnsupportedFeature),
        };

        // FROM に出てくるテーブルのスキーマを先に全部解決する。
        // 結合があると複数になるので、足りない範囲はまとめて要求する。
        let mut tables = Vec::new();
        referenced_tables(&self.catalog, from, &mut tables)?;
        let mut io = Vec::new();
        for &table in &tables {
            let t = match self.catalog.get_mut(table) {
                Some(t) => t,
                None => err!(TableNotFound),
            };
            if let Err((offset, len)) = t.resolve()? {
                io.push(IoRequest { table, offset, len });
            }
        }
        if !io.is_empty() {
            return Ok(Prepared::NeedIo(io));
        }

        let plan = bind_select(&self.catalog, &parsed.arena, sel, params)?;
        let schema = plan.root.schema().to_vec();
        Ok(Prepared::Ready(Query { root: build(plan.root)?, schema }))
    }

    /// ホストが展開した圧縮ブロックを渡す。
    pub fn provide_decoded(
        &mut self,
        table: usize,
        offset: u64,
        len: u32,
        data: Vec<u8>,
    ) -> Result<()> {
        match self.catalog.get_mut(table) {
            Some(t) => {
                t.source.insert_decoded(offset, len, data);
                Ok(())
            }
            None => err!(TableNotFound),
        }
    }

    /// 次のバッチを取り出す。
    pub fn step(&mut self, q: &mut Query) -> Result<QueryStep> {
        let mut ctx = ExecContext {
            catalog: &mut self.catalog,
            vm: &mut self.vm,
            io: Vec::new(),
            codec: Vec::new(),
        };
        match q.root.next(&mut ctx)? {
            Step::Ready(b) => Ok(QueryStep::Batch(b)),
            Step::NeedIo => Ok(QueryStep::NeedIo(ctx.io)),
            Step::NeedCodec => Ok(QueryStep::NeedCodec(ctx.codec)),
            Step::Done => Ok(QueryStep::Done),
        }
    }

    /// テーブル名を列挙する（`SHOW TABLES`）。
    pub fn table_names(&self) -> Vec<String> {
        self.catalog.names().map(String::from).collect()
    }

    /// `DESCRIBE` 用。フッタが未解決なら要求を返す。
    pub fn describe(
        &mut self,
        from: &FromItem,
    ) -> Result<core::result::Result<Vec<Field>, IoRequest>> {
        let table = resolve_from(&self.catalog, from)?;
        let t = match self.catalog.get_mut(table) {
            Some(t) => t,
            None => err!(TableNotFound),
        };
        if let Err((offset, len)) = t.resolve()? {
            return Ok(Err(IoRequest { table, offset, len }));
        }
        Ok(Ok(t.schema().to_vec()))
    }
}

impl Default for Session {
    fn default() -> Self {
        Session::new()
    }
}
