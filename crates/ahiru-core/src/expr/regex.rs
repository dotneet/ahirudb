//! `regexp_matches` / `regexp_extract` / `regexp_replace` 用の正規表現エンジン。
//!
//! 外部クレートを使わない自前実装。素朴な再帰バックトラックは
//! `(a+)+b` のようなパターンで指数時間になりうるため採用せず、
//! **Thompson NFA + Pike's VM**（Russ Cox の一連の記事で解説されている手法）
//! で実装する。パターンを NFA へコンパイルし、入力バイトごとに「今生きている
//! 状態の集合」を並行して前進させる。状態は 1 バイトあたり高々 命令数ぶんしか
//! 増えない（同じ命令番号への到達は 1 回だけ数える）ので、最悪計算量は
//! `O(命令数 × 入力長)` に収まり、バックトラックによる爆発が原理的に起きない。
//! `LIKE` (`expr/kernels.rs` の `like_match`) が 2 ポインタ法で同じ問題を
//! 避けているのと動機は同じ。
//!
//! ## 対応する構文（これ以外は `Err(UnsupportedFeature)` か `Err(SyntaxError)`）
//!
//! - リテラル文字（**バイト単位**。`upper`/`lower` 等と違い UTF-8 の
//!   コードポイントは意識しない。マルチバイト文字はバイト列の完全一致として
//!   扱われるので、リテラルとしては問題なく動くが `.` は「バイト 1 個」に
//!   一致する点に注意）。
//! - `.`（改行以外の任意の 1 バイト）
//! - `*` `+` `?`（貪欲な量指定子）
//! - `{n}` `{n,}` `{n,m}`（有界繰り返し。DuckDB の内部エンジン RE2 と同じく
//!   `n`,`m` は 1000 まで）
//! - 文字クラス `[abc]` `[^abc]`、範囲 `[a-z]`、短縮形 `\d \D \w \W \s \S`
//! - アンカー `^` `$`（`(?m)` 相当のマルチライン化はしない。`^` は入力の
//!   先頭、`$` は入力の末尾にのみ一致する）
//! - 選択 `|`
//! - グループ `(...)`（捕獲）と `(?:...)`（非捕獲）
//! - メタ文字のエスケープ `\. \* \+ \? \( \) \[ \] \{ \} \| \^ \$ \\`
//!
//! **非対応**（`resolve`/コンパイル時に必ずエラーにする。黙って別の意味に
//! 解釈してしまうと「間違った検索結果」という一番まずい failure mode になる
//! ため）:
//! - 先読み・後読み `(?=...)` `(?!...)` `(?<=...)` `(?<!...)`
//! - パターン内バックリファレンス `\1`（**置換文字列側の** `\1`/`\2` は別物で
//!   `regexp_replace` に必須の機能として対応する。詳細は `parse_repl` 参照）
//! - 名前付きグループ `(?P<name>...)` `(?<name>...)`
//! - 非貪欲量指定子 `*?` `+?` `??` `{n,m}?`
//! - 単語境界 `\b` `\B`
//! - 大文字小文字を無視するマッチ（`(?i)` やフラグ引数）。DuckDB は対応するが
//!   実装コストと引き換えるほどの頻度ではないと判断し v1 では見送る。
//!   ASCII 大小無視だけなら安く足せるが、`LIKE` 同様この単位でも需要が
//!   確認できるまでは増やさない。
//!
//! ## サイズ・実行時間の上限
//!
//! すべて「小さいコードで確実に上限を掛けられる」値を選んでいる。
//! 個々の理由はコード中の定数コメントを参照。
//!
//! ## `resolve`/`call` との関係
//!
//! `resolve` は型しか見られず、パターン文字列（多くの場合クエリ全体で
//! 定数）を見られない。そのため「1 クエリに 1 回だけコンパイルする」ことは
//! 計画時にはできない。代わりに `call` の中で、パターン列の stride が 0
//! （＝定数列）なら行ループの外で 1 回だけ `compile` し、そうでなければ
//! （パターンが列ごとに変わる稀なケース）行ごとに再コンパイルする。
//! `LIKE` は現状パターンを毎行スキャンしているだけなので、これでも既存水準
//! より悪化はしない。

use crate::prelude::*;
use crate::vector::{Bitmap, BytesData, Data, Ty, Vector};

// ============================================================================
// 上限値
// ============================================================================

/// パターン文字列の最大バイト数。これより長い場合はコンパイルに掛ける前に
/// 拒否する。以降の上限（命令数など）から見て、これより長くても意味のある
/// パターンは書けない。
const MAX_PATTERN_LEN: usize = 4096;

/// コンパイル後の命令列の最大長。「数千命令」という指示に沿って 4096 に
/// 設定。`{n,m}` の展開や入れ子repeatはこの値を超えた時点で即座に打ち切る
/// ので、`(a{1000}){1000}` のような入れ子でも実際に確保するメモリは
/// この上限に比例した量で頭打ちになる（詳細は `Compiler::emit` 参照）。
const MAX_PROG_INSTS: usize = 4096;

/// 捕獲グループの最大数。スレッドごとに持つ捕獲位置配列
/// `[u32; SLOTS]`（`SLOTS = (MAX_GROUPS+1)*2`）のサイズを決める値なので、
/// 大きくするほど NFA シミュレーションの 1 ステップが重くなる。32 は
/// 実用的なパターンでまず超えない数。
const MAX_GROUPS: usize = 32;

/// 捕獲位置配列のサイズ。group 0 は全体一致用に予約する。
const SLOTS: usize = (MAX_GROUPS + 1) * 2;

/// 括弧の入れ子の最大深さ。パーサ・コンパイラとも再帰で書いているので、
/// これでネイティブスタックの深さを間接的に抑える（wasm はスタックが
/// 小さいホストもある）。
const MAX_NEST_DEPTH: u32 = 64;

/// `{n,m}` の `n`/`m` の最大値。DuckDB が内部で使う RE2 も同じ 1000 を
/// 上限にしている（`duckdb -c "select regexp_matches('a','a{1001}')"` が
/// `invalid repetition size` で弾かれることを確認済み）ので、それに揃える。
const MAX_REPEAT: u32 = 1000;

/// 1 回の NFA シミュレーション（`exec` 1 呼び出し＝1 回のマッチ試行）で
/// 許す「命令に到達した」延べ回数。Thompson NFA の最悪計算量は
/// `O(命令数 × 入力長)` なので、命令数上限 4096 と掛け合わせると
/// 8,000,000 は「上限いっぱいの命令数のパターンを ~2000 バイトの入力に
/// 掛ける」「20 命令程度の普通のパターンを ~400KB の入力に掛ける」の
/// どちらも許容しつつ、変な入力（`regexp_replace` の `g` 付きグローバル
/// 置換で `exec` を繰り返し呼ぶケースなど）で 1 回のマッチ試行が
/// wasm 上で数十 ms を超えないよう頭打ちにする値として選んだ。
/// バッチ全体ではなく `exec` 呼び出し単位の上限であることに注意
/// （行数分の正当な作業量まで制限すると通常のクエリが動かなくなるため）。
const MAX_STEPS: u32 = 8_000_000;

// ============================================================================
// 文字クラス
// ============================================================================

/// 256 バイト値の集合をビットマスクで持つ。範囲・否定・和集合のどれも
/// 固定長ループで済み、パターンのサイズに関わらずコード量が増えない。
#[derive(Clone, Copy)]
struct ClassSet([u64; 4]);

impl ClassSet {
    fn new() -> Self {
        ClassSet([0; 4])
    }

    fn set(&mut self, b: u8) {
        self.0[(b >> 6) as usize] |= 1u64 << (b & 63);
    }

    fn set_range(&mut self, lo: u8, hi: u8) {
        let mut b = lo;
        loop {
            self.set(b);
            if b == hi {
                break;
            }
            b += 1;
        }
    }

    fn negate(&mut self) {
        for w in self.0.iter_mut() {
            *w = !*w;
        }
    }

    fn union(&mut self, other: &ClassSet) {
        for i in 0..4 {
            self.0[i] |= other.0[i];
        }
    }

    fn test(&self, b: u8) -> bool {
        (self.0[(b >> 6) as usize] >> (b & 63)) & 1 != 0
    }
}

fn digit_set() -> ClassSet {
    let mut s = ClassSet::new();
    s.set_range(b'0', b'9');
    s
}

fn word_set() -> ClassSet {
    let mut s = digit_set();
    s.set_range(b'a', b'z');
    s.set_range(b'A', b'Z');
    s.set(b'_');
    s
}

/// RE2 の `\s` は `[\t\n\f\r ]`（`\v` を含まない）。DuckDB 経由で実測して
/// 確認済み（`chr(11)`（`\v`）は不一致、`chr(12)`（`\f`）は一致）。
fn space_set() -> ClassSet {
    let mut s = ClassSet::new();
    for &b in &[b' ', b'\t', b'\n', 0x0c, b'\r'] {
        s.set(b);
    }
    s
}

// ============================================================================
// AST とパーサ
// ============================================================================

enum Ast {
    Empty,
    Char(u8),
    Any,
    Class(ClassSet),
    Concat(Vec<Ast>),
    Alt(Vec<Ast>),
    Star(Box<Ast>),
    Plus(Box<Ast>),
    Quest(Box<Ast>),
    Repeat(Box<Ast>, u32, Option<u32>),
    /// `Some(g)` は捕獲グループ（`g` は 1 始まり）、`None` は非捕獲。
    Group(Box<Ast>, Option<u16>),
    Bol,
    Eol,
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
    depth: u32,
    n_groups: u16,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }

    fn parse_alt(&mut self) -> Result<Ast> {
        let mut branches = vec![self.parse_concat()?];
        while self.peek() == Some(b'|') {
            self.i += 1;
            branches.push(self.parse_concat()?);
        }
        Ok(if branches.len() == 1 { branches.pop().unwrap() } else { Ast::Alt(branches) })
    }

    fn parse_concat(&mut self) -> Result<Ast> {
        let mut items = Vec::new();
        while let Some(c) = self.peek() {
            if c == b'|' || c == b')' {
                break;
            }
            items.push(self.parse_repeat()?);
        }
        Ok(match items.len() {
            0 => Ast::Empty,
            1 => items.pop().unwrap(),
            _ => Ast::Concat(items),
        })
    }

    /// 原子 1 つ＋末尾の量指定子（あれば 1 つだけ）を読む。
    fn parse_repeat(&mut self) -> Result<Ast> {
        let atom = self.parse_atom()?;
        // まず「量指定子を適用するかどうか」だけを、atom を消費せずに決める。
        enum Q {
            None,
            Star,
            Plus,
            Quest,
            Repeat(u32, Option<u32>),
        }
        let q = match self.peek() {
            Some(b'*') => {
                self.i += 1;
                Q::Star
            }
            Some(b'+') => {
                self.i += 1;
                Q::Plus
            }
            Some(b'?') => {
                self.i += 1;
                Q::Quest
            }
            Some(b'{') => match self.try_parse_bound() {
                Some((n, m)) => {
                    ensure!(n <= MAX_REPEAT, LimitExceeded);
                    if let Some(mm) = m {
                        ensure!(mm <= MAX_REPEAT, LimitExceeded);
                        ensure!(mm >= n, SyntaxError);
                    }
                    Q::Repeat(n, m)
                }
                // 有効な `{n,m}` の形をしていない `{` は単なるリテラル文字
                // （RE2/DuckDB の挙動。`self.i` は '{' の手前のまま戻すので
                // 次の parse_repeat 呼び出しで通常の原子として読み直される）。
                None => Q::None,
            },
            _ => Q::None,
        };
        if matches!(q, Q::None) {
            return Ok(atom);
        }
        // 量指定子の直後にさらに量指定子は許さない。`*?` 等の非貪欲
        // 記法は非対応として明示的に拒否する（黙って貪欲扱いすると
        // 「間違った結果」になってしまうため、それだけは絶対に避ける）。
        match self.peek() {
            Some(b'?') => err!(UnsupportedFeature),
            Some(b'*') | Some(b'+') | Some(b'{') => err!(SyntaxError),
            _ => {}
        }
        Ok(match q {
            Q::None => unreachable!(),
            Q::Star => Ast::Star(Box::new(atom)),
            Q::Plus => Ast::Plus(Box::new(atom)),
            Q::Quest => Ast::Quest(Box::new(atom)),
            Q::Repeat(n, m) => Ast::Repeat(Box::new(atom), n, m),
        })
    }

    /// `{n}` / `{n,}` / `{n,m}` を読む。形が違えば `self.i` を戻して `None`。
    fn try_parse_bound(&mut self) -> Option<(u32, Option<u32>)> {
        let start = self.i;
        self.i += 1; // '{'
        let n = self.scan_u32()?;
        let bound = if self.peek() == Some(b',') {
            self.i += 1;
            if self.peek() == Some(b'}') {
                Some((n, None))
            } else {
                let m = self.scan_u32()?;
                Some((n, Some(m)))
            }
        } else {
            Some((n, Some(n)))
        };
        if self.peek() == Some(b'}') && bound.is_some() {
            self.i += 1;
            bound
        } else {
            self.i = start;
            None
        }
    }

    fn scan_u32(&mut self) -> Option<u32> {
        let start = self.i;
        let mut v: u32 = 0;
        while let Some(c) = self.peek() {
            if !c.is_ascii_digit() {
                break;
            }
            // MAX_REPEAT より十分大きい値で早々に飽和させる（オーバーフロー
            // 回避。後段の ensure! でどのみち LimitExceeded になる）。
            v = v.saturating_mul(10).saturating_add((c - b'0') as u32);
            self.i += 1;
        }
        if self.i == start {
            None
        } else {
            Some(v)
        }
    }

    fn parse_atom(&mut self) -> Result<Ast> {
        let c = match self.peek() {
            Some(c) => c,
            None => err!(SyntaxError),
        };
        match c {
            b'*' | b'+' | b'?' => err!(SyntaxError), // 対象がない量指定子
            b'.' => {
                self.i += 1;
                Ok(Ast::Any)
            }
            b'^' => {
                self.i += 1;
                Ok(Ast::Bol)
            }
            b'$' => {
                self.i += 1;
                Ok(Ast::Eol)
            }
            b'[' => self.parse_class(),
            b'(' => self.parse_group(),
            b'\\' => self.parse_escape(),
            // '{' はここでは常にリテラル（有効な {n,m} は parse_repeat 側で
            // 先に処理される。ここに来るのは「原子の先頭に来た {」なので
            // 対象がなく、常にリテラル扱いで良い）。
            _ => {
                self.i += 1;
                Ok(Ast::Char(c))
            }
        }
    }

    fn parse_group(&mut self) -> Result<Ast> {
        self.i += 1; // '('
        let capturing = if self.peek() == Some(b'?') {
            if self.s.get(self.i + 1) == Some(&b':') {
                self.i += 2;
                false
            } else {
                // (?=  (?!  (?<  (?P  など、非捕獲以外の拡張構文はすべて非対応。
                err!(UnsupportedFeature);
            }
        } else {
            true
        };
        let g = if capturing {
            ensure!((self.n_groups as usize) < MAX_GROUPS, LimitExceeded);
            self.n_groups += 1;
            Some(self.n_groups)
        } else {
            None
        };
        self.depth += 1;
        ensure!(self.depth <= MAX_NEST_DEPTH, LimitExceeded);
        let inner = self.parse_alt()?;
        self.depth -= 1;
        ensure!(self.peek() == Some(b')'), SyntaxError);
        self.i += 1;
        Ok(Ast::Group(Box::new(inner), g))
    }

    fn parse_escape(&mut self) -> Result<Ast> {
        self.i += 1; // '\'
        let c = match self.peek() {
            Some(c) => c,
            None => err!(SyntaxError),
        };
        self.i += 1;
        Ok(match c {
            b'.' | b'*' | b'+' | b'?' | b'(' | b')' | b'[' | b']' | b'{' | b'}' | b'|' | b'^'
            | b'$' | b'\\' => Ast::Char(c),
            b'd' => Ast::Class(digit_set()),
            b'D' => {
                let mut s = digit_set();
                s.negate();
                Ast::Class(s)
            }
            b'w' => Ast::Class(word_set()),
            b'W' => {
                let mut s = word_set();
                s.negate();
                Ast::Class(s)
            }
            b's' => Ast::Class(space_set()),
            b'S' => {
                let mut s = space_set();
                s.negate();
                Ast::Class(s)
            }
            // パターン内バックリファレンス・単語境界は非対応（モジュール冒頭参照）。
            b'1'..=b'9' | b'b' | b'B' => err!(UnsupportedFeature),
            _ => err!(SyntaxError),
        })
    }

    fn parse_class(&mut self) -> Result<Ast> {
        self.i += 1; // '['
        let negate = self.peek() == Some(b'^');
        if negate {
            self.i += 1;
        }
        let mut set = ClassSet::new();
        let mut first = true;
        loop {
            match self.peek() {
                None => err!(SyntaxError),
                Some(b']') if !first => {
                    self.i += 1;
                    break;
                }
                Some(b'\\') => {
                    self.i += 1;
                    let c = match self.peek() {
                        Some(c) => c,
                        None => err!(SyntaxError),
                    };
                    self.i += 1;
                    match c {
                        b'd' => set.union(&digit_set()),
                        b'D' => {
                            let mut t = digit_set();
                            t.negate();
                            set.union(&t);
                        }
                        b'w' => set.union(&word_set()),
                        b'W' => {
                            let mut t = word_set();
                            t.negate();
                            set.union(&t);
                        }
                        b's' => set.union(&space_set()),
                        b'S' => {
                            let mut t = space_set();
                            t.negate();
                            set.union(&t);
                        }
                        // クラス内では `]` `\` `^` `-` などをエスケープする
                        // 必要が実務上あるので、それ以外の文字もリテラルとして
                        // 許す（クラス外の厳格なエスケープ規則とは別）。
                        _ => set.set(c),
                    }
                    first = false;
                }
                Some(lo) => {
                    self.i += 1;
                    first = false;
                    // 範囲 `a-z`。末尾の `-`（次が `]`）や先頭の `-` は
                    // リテラルとして扱う。
                    if self.peek() == Some(b'-')
                        && self.s.get(self.i + 1).is_some()
                        && self.s.get(self.i + 1) != Some(&b']')
                    {
                        self.i += 1;
                        let hi = self.s[self.i];
                        ensure!(hi != b'\\', SyntaxError);
                        self.i += 1;
                        ensure!(lo <= hi, SyntaxError);
                        set.set_range(lo, hi);
                    } else {
                        set.set(lo);
                    }
                }
            }
        }
        if negate {
            set.negate();
        }
        Ok(Ast::Class(set))
    }
}

// ============================================================================
// コンパイル（AST → NFA 命令列）
// ============================================================================

#[derive(Clone, Copy)]
enum Inst {
    Char(u8),
    Any,
    Class(u16),
    Split(u32, u32),
    Jmp(u32),
    Save(u8),
    Bol,
    Eol,
    Match,
}

pub struct Program {
    insts: Vec<Inst>,
    classes: Vec<ClassSet>,
    n_groups: u16,
}

struct Compiler {
    prog: Vec<Inst>,
    classes: Vec<ClassSet>,
}

impl Compiler {
    fn emit(&mut self, inst: Inst) -> Result<u32> {
        // ここが唯一の増加点。`{n,m}` や入れ子 repeat の展開はすべて
        // `compile` の再帰呼び出し経由でここを通るので、パターンがどれだけ
        // 入れ子になっていても実際に確保する命令数はこの上限で頭打ちになる
        // （AST 自体を複製して膨らませることは一切しない設計のため）。
        ensure!(self.prog.len() < MAX_PROG_INSTS, LimitExceeded);
        self.prog.push(inst);
        Ok((self.prog.len() - 1) as u32)
    }

    fn compile(&mut self, ast: &Ast) -> Result<()> {
        match ast {
            Ast::Empty => Ok(()),
            Ast::Char(c) => {
                self.emit(Inst::Char(*c))?;
                Ok(())
            }
            Ast::Any => {
                self.emit(Inst::Any)?;
                Ok(())
            }
            Ast::Class(set) => {
                let idx = self.classes.len() as u16;
                self.classes.push(*set);
                self.emit(Inst::Class(idx))?;
                Ok(())
            }
            Ast::Bol => {
                self.emit(Inst::Bol)?;
                Ok(())
            }
            Ast::Eol => {
                self.emit(Inst::Eol)?;
                Ok(())
            }
            Ast::Concat(items) => {
                for it in items {
                    self.compile(it)?;
                }
                Ok(())
            }
            Ast::Alt(branches) => self.compile_alt(branches),
            Ast::Star(x) => self.compile_star(x),
            Ast::Plus(x) => self.compile_plus(x),
            Ast::Quest(x) => self.compile_quest(x),
            Ast::Repeat(x, n, m) => self.compile_repeat(x, *n, *m),
            Ast::Group(x, Some(g)) => {
                self.emit(Inst::Save(2 * (*g as u8)))?;
                self.compile(x)?;
                self.emit(Inst::Save(2 * (*g as u8) + 1))?;
                Ok(())
            }
            Ast::Group(x, None) => self.compile(x),
        }
    }

    fn compile_alt(&mut self, branches: &[Ast]) -> Result<()> {
        if branches.len() == 1 {
            return self.compile(&branches[0]);
        }
        let split_pc = self.emit(Inst::Split(0, 0))?;
        self.compile(&branches[0])?;
        let jmp_pc = self.emit(Inst::Jmp(0))?;
        let l2 = self.prog.len() as u32;
        self.compile_alt(&branches[1..])?;
        let end = self.prog.len() as u32;
        self.prog[split_pc as usize] = Inst::Split(split_pc + 1, l2);
        self.prog[jmp_pc as usize] = Inst::Jmp(end);
        Ok(())
    }

    /// 貪欲な `x*`。「続ける」を先、「抜ける」を後にした `Split` で貪欲優先を表す。
    fn compile_star(&mut self, x: &Ast) -> Result<()> {
        let l1 = self.emit(Inst::Split(0, 0))?;
        self.compile(x)?;
        self.emit(Inst::Jmp(l1))?;
        let end = self.prog.len() as u32;
        self.prog[l1 as usize] = Inst::Split(l1 + 1, end);
        Ok(())
    }

    fn compile_plus(&mut self, x: &Ast) -> Result<()> {
        let l1 = self.prog.len() as u32;
        self.compile(x)?;
        let split_pc = self.emit(Inst::Split(0, 0))?;
        self.prog[split_pc as usize] = Inst::Split(l1, split_pc + 1);
        Ok(())
    }

    fn compile_quest(&mut self, x: &Ast) -> Result<()> {
        let l1 = self.emit(Inst::Split(0, 0))?;
        self.compile(x)?;
        let end = self.prog.len() as u32;
        self.prog[l1 as usize] = Inst::Split(l1 + 1, end);
        Ok(())
    }

    /// `{n,m}` は `x` を `n` 回並べたあと、`x?` を `m-n` 回並べる
    /// （`{n,}` は `m` を省略して `x* ` を続ける）だけで表せる。
    /// 各 `x?` は独立した Split なので、貪欲優先の順序も自然に保たれる。
    /// AST を複製して展開するのではなく、同じ `x` を指す参照へ
    /// `compile` を繰り返し呼ぶだけなので、展開後のサイズは
    /// （`emit` が数える）命令数として現れる分だけで、入れ子になっていても
    /// メモリ上で先に膨れ上がることが無い。
    fn compile_repeat(&mut self, x: &Ast, n: u32, m: Option<u32>) -> Result<()> {
        for _ in 0..n {
            self.compile(x)?;
        }
        match m {
            None => self.compile_star(x),
            Some(mm) => {
                for _ in 0..(mm - n) {
                    self.compile_quest(x)?;
                }
                Ok(())
            }
        }
    }
}

/// パターン文字列をコンパイルする。呼び出し側（`funcs.rs`）はパターン列の
/// stride が 0（定数列）ならこれを行ループの外で 1 回だけ呼ぶ。
pub fn compile(pattern: &[u8]) -> Result<Program> {
    ensure!(pattern.len() <= MAX_PATTERN_LEN, LimitExceeded);
    let mut p = Parser { s: pattern, i: 0, depth: 0, n_groups: 0 };
    let ast = p.parse_alt()?;
    // 消費しきれなかった残りがあるのは、対応する '(' の無い ')' が
    // 混ざっていたとき。
    ensure!(p.i == pattern.len(), SyntaxError);
    let mut c = Compiler { prog: Vec::new(), classes: Vec::new() };
    c.emit(Inst::Save(0))?;
    c.compile(&ast)?;
    c.emit(Inst::Save(1))?;
    c.emit(Inst::Match)?;
    Ok(Program { insts: c.prog, classes: c.classes, n_groups: p.n_groups })
}

// ============================================================================
// Pike's VM によるシミュレーション
// ============================================================================

#[derive(Clone, Copy)]
struct Thread {
    pc: u32,
    saves: [u32; SLOTS],
}

/// `pc` から到達できる「バイトを 1 つ消費する」命令（`Char`/`Any`/`Class`）と
/// `Match` まで、ε 遷移（`Split`/`Jmp`/`Save`/`Bol`/`Eol`）を辿って `list` に
/// 積む（NFA の ε-閉包）。
///
/// `gen`/`gen_id` は「この位置で既にこの命令番号を追加したか」を覚えるための
/// もの。これにより同じ命令番号が 1 ステップに 2 回積まれることが無くなり、
/// `Split` がいくつ入れ子になっていても 1 ステップの計算量は命令数で頭打ちに
/// なる（＝バックトラック法のような指数的な再訪問が起きない）。
///
/// 再帰ではなく明示スタックで書いているのは、`Split` の枝を再帰で辿ると
/// パターン次第でネイティブスタックを命令数ぶん消費しうるため
/// （wasm はホストによってスタックが小さいことがある）。
#[allow(clippy::too_many_arguments)]
fn add_thread(
    prog: &Program,
    list: &mut Vec<Thread>,
    gen: &mut [u32],
    gen_id: u32,
    pc0: u32,
    saves0: [u32; SLOTS],
    sp: usize,
    input_len: usize,
    steps: &mut u32,
) -> Result<()> {
    let mut stack: Vec<(u32, [u32; SLOTS])> = vec![(pc0, saves0)];
    while let Some((pc, mut saves)) = stack.pop() {
        if gen[pc as usize] == gen_id {
            continue;
        }
        gen[pc as usize] = gen_id;
        *steps += 1;
        ensure!(*steps <= MAX_STEPS, LimitExceeded);
        match prog.insts[pc as usize] {
            Inst::Jmp(x) => stack.push((x, saves)),
            Inst::Split(a, b) => {
                // 貪欲優先: 先に積んだほうが「あと」に処理される LIFO なので、
                // 優先度の高い枝 `a` をあとに積んで先に取り出させる。
                stack.push((b, saves));
                stack.push((a, saves));
            }
            Inst::Save(slot) => {
                if (slot as usize) < SLOTS {
                    saves[slot as usize] = sp as u32;
                }
                stack.push((pc + 1, saves));
            }
            Inst::Bol => {
                if sp == 0 {
                    stack.push((pc + 1, saves));
                }
            }
            Inst::Eol => {
                if sp == input_len {
                    stack.push((pc + 1, saves));
                }
            }
            Inst::Char(_) | Inst::Any | Inst::Class(_) | Inst::Match => {
                list.push(Thread { pc, saves });
            }
        }
    }
    Ok(())
}

/// 入力中の最左マッチを探す。見つかれば捕獲位置（group 0 = 全体一致、
/// スロット値 `u32::MAX` は「未捕獲」）を返す。
///
/// アンカーされた探索ではなく、開始位置を 1 バイトずつ後ろへずらしながら
/// 探す「検索」。ただし専用のループを回すのではなく、まだマッチが
/// 見つかっていない間は新しい開始スレッドを**既存スレッドより低い優先度**で
/// 追加し続けるだけで実現する（Pike's VM の定石）。これにより
/// 「最初に見つかった＝最左」が自動的に保証される。
pub fn find(prog: &Program, input: &[u8]) -> Result<Option<[u32; SLOTS]>> {
    let mut gen = vec![0u32; prog.insts.len()];
    let mut gen_id: u32 = 0;
    let mut steps: u32 = 0;
    let mut clist: Vec<Thread> = Vec::new();
    let mut nlist: Vec<Thread> = Vec::new();
    let mut matched: Option<[u32; SLOTS]> = None;
    let len = input.len();

    gen_id += 1;
    add_thread(prog, &mut clist, &mut gen, gen_id, 0, [u32::MAX; SLOTS], 0, len, &mut steps)?;

    let mut sp = 0usize;
    loop {
        if clist.is_empty() && (matched.is_some() || sp > len) {
            break;
        }
        let byte = input.get(sp).copied();
        gen_id += 1;
        let mut idx = 0;
        while idx < clist.len() {
            let th = clist[idx];
            match prog.insts[th.pc as usize] {
                Inst::Char(c) => {
                    if byte == Some(c) {
                        add_thread(
                            prog,
                            &mut nlist,
                            &mut gen,
                            gen_id,
                            th.pc + 1,
                            th.saves,
                            sp + 1,
                            len,
                            &mut steps,
                        )?;
                    }
                }
                Inst::Any => {
                    if let Some(b) = byte {
                        if b != b'\n' {
                            add_thread(
                                prog,
                                &mut nlist,
                                &mut gen,
                                gen_id,
                                th.pc + 1,
                                th.saves,
                                sp + 1,
                                len,
                                &mut steps,
                            )?;
                        }
                    }
                }
                Inst::Class(ci) => {
                    if let Some(b) = byte {
                        if prog.classes[ci as usize].test(b) {
                            add_thread(
                                prog,
                                &mut nlist,
                                &mut gen,
                                gen_id,
                                th.pc + 1,
                                th.saves,
                                sp + 1,
                                len,
                                &mut steps,
                            )?;
                        }
                    }
                }
                Inst::Match => {
                    // これより優先度の低い（＝ clist の後ろにある）スレッドは
                    // 採用されえないので打ち切る。優先度の高いスレッドは
                    // このループより前で既に nlist へ積み終えている。
                    matched = Some(th.saves);
                    break;
                }
                Inst::Jmp(_) | Inst::Split(_, _) | Inst::Save(_) | Inst::Bol | Inst::Eol => {
                    // ε 命令は add_thread が展開済みなのでスレッドとしては現れない。
                }
            }
            idx += 1;
        }
        if byte.is_some() && matched.is_none() {
            add_thread(
                prog,
                &mut nlist,
                &mut gen,
                gen_id,
                0,
                [u32::MAX; SLOTS],
                sp + 1,
                len,
                &mut steps,
            )?;
        }
        core::mem::swap(&mut clist, &mut nlist);
        nlist.clear();
        if byte.is_none() {
            break;
        }
        sp += 1;
    }
    Ok(matched)
}

pub fn is_match(prog: &Program, input: &[u8]) -> Result<bool> {
    Ok(find(prog, input)?.is_some())
}

// ============================================================================
// SQL 関数本体（Vector 単位）
// ============================================================================

/// 行数と各引数の stride。`funcs.rs::strides` と同じ規則（長さ 1 は定数,
/// それ以外は全部同じ長さでなければならない）だが、他ファイルを触らない
/// という制約のためここに独立して持つ。
fn plan(args: &[&Vector]) -> Result<(usize, Vec<usize>)> {
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

fn geti(v: &Vector, i: usize) -> i64 {
    match v.data() {
        Data::I64(d) => d.get(i).copied().unwrap_or(0),
        Data::I32(d) => d.get(i).copied().unwrap_or(0) as i64,
        _ => 0,
    }
}

/// `regexp_matches(str, pattern)`。
pub fn eval_matches(args: &[&Vector]) -> Result<Vector> {
    ensure!(args.len() == 2, WrongArgCount);
    let (n, s) = plan(args)?;
    let valid = combine(args, &s, n);
    let live = |i: usize| valid.as_ref().is_none_or(|b| b.get(i));
    let (sv, pv) = (args[0], args[1]);
    let pat_const = s[1] == 0;
    let cached =
        if pat_const && n > 0 && pv.is_valid(0) { Some(compile(pv.bytes().get(0))?) } else { None };
    let mut bits = Bitmap::with_capacity(n);
    for i in 0..n {
        if !live(i) {
            bits.push(false);
            continue;
        }
        let compiled;
        let prog: &Program = match &cached {
            Some(p) => p,
            None => {
                compiled = compile(pv.bytes().get(i * s[1]))?;
                &compiled
            }
        };
        bits.push(is_match(prog, sv.bytes().get(i * s[0]))?);
    }
    let mut out = Vector::from_data(Ty::Boolean, Data::Bool(bits), valid);
    out.compact_validity();
    Ok(out)
}

/// `regexp_extract(str, pattern[, group])`。
///
/// DuckDB は「マッチしない」「グループがそのマッチでは捕獲されなかった」
/// 「グループ番号がパターンの捕獲数を超える（ただし 0〜9 の範囲内）」の
/// いずれも **NULL ではなく空文字列** を返す（`duckdb -c "select
/// regexp_extract('xxx','foo')"` 等で実測）。NULL になるのは str/pattern/
/// group 引数そのものが NULL のときだけ。
pub fn eval_extract(args: &[&Vector]) -> Result<Vector> {
    ensure!(args.len() == 2 || args.len() == 3, WrongArgCount);
    let (n, s) = plan(args)?;
    let valid = combine(args, &s, n);
    let live = |i: usize| valid.as_ref().is_none_or(|b| b.get(i));
    let (sv, pv) = (args[0], args[1]);
    let pat_const = s[1] == 0;
    let cached =
        if pat_const && n > 0 && pv.is_valid(0) { Some(compile(pv.bytes().get(0))?) } else { None };
    let mut out = BytesData::with_capacity(n, n * 8);
    for i in 0..n {
        if !live(i) {
            out.push_empty();
            continue;
        }
        let compiled;
        let prog: &Program = match &cached {
            Some(p) => p,
            None => {
                compiled = compile(pv.bytes().get(i * s[1]))?;
                &compiled
            }
        };
        let g = if args.len() == 3 { geti(args[2], i * s[2]) } else { 0 };
        // DuckDB 実測: group は 0〜9 のみ許され、範囲外は行ではなくクエリ
        // 自体のエラーになる（`select regexp_extract('a','a',10)` は
        // Invalid Input Error）。
        ensure!((0..=9).contains(&g), ValueOutOfRange);
        let text = sv.bytes().get(i * s[0]);
        match find(prog, text)? {
            None => out.push_empty(),
            Some(saves) => {
                if g as u16 > prog.n_groups {
                    out.push_empty();
                } else {
                    let (st, en) = (saves[2 * g as usize], saves[2 * g as usize + 1]);
                    if st == u32::MAX || en == u32::MAX {
                        out.push_empty();
                    } else {
                        out.push(&text[st as usize..en as usize]);
                    }
                }
            }
        }
    }
    let mut v = Vector::from_data(Ty::Varchar, Data::Bytes(out), valid);
    v.compact_validity();
    Ok(v)
}

/// 置換文字列のトークン。`\0`〜`\9` はグループ参照、`\\` はリテラル `\`、
/// それ以外の `\X` は不正。
enum ReplTok {
    Lit(u8),
    Group(u8),
}

/// 置換文字列を構文解析する。DuckDB（内部の RE2）実測では、置換文字列が
/// 不正（末尾が `\` で終わる／`\` の次が数字でも `\` でもない）だったり、
/// 参照しているグループ番号がパターンの実際の捕獲数を超えていたりすると、
/// **エラーにはならず、その行の結果は入力文字列がそのまま返る**
/// （置換が一切行われない）。`select regexp_replace('abc','b','X\9')` が
/// `'abc'` を返す、`select regexp_replace('a','(a)','X\2Y')` が `'a'` を
/// 返す、等で確認済み。ここではその「無効なら `None`」をそのまま
/// `replace_into` 側の「無効なら丸ごと素通し」に対応させる。
///
/// 戻り値の第 2 要素は参照された最大グループ番号（`prog.n_groups` との
/// 比較は呼び出し側で行う。ここではパターンを知らないため）。
fn parse_repl(repl: &[u8]) -> Option<(Vec<ReplTok>, u8)> {
    let mut toks = Vec::with_capacity(repl.len());
    let mut max_g: u8 = 0;
    let mut i = 0usize;
    while i < repl.len() {
        let c = repl[i];
        if c == b'\\' {
            let d = *repl.get(i + 1)?;
            if d == b'\\' {
                toks.push(ReplTok::Lit(b'\\'));
            } else if d.is_ascii_digit() {
                let g = d - b'0';
                toks.push(ReplTok::Group(g));
                if g > max_g {
                    max_g = g;
                }
            } else {
                return None;
            }
            i += 2;
        } else {
            toks.push(ReplTok::Lit(c));
            i += 1;
        }
    }
    Some((toks, max_g))
}

fn emit_repl(toks: &[ReplTok], text: &[u8], saves: &[u32; SLOTS], out: &mut Vec<u8>) {
    for t in toks {
        match t {
            ReplTok::Lit(b) => out.push(*b),
            ReplTok::Group(g) => {
                let (st, en) = (saves[2 * *g as usize], saves[2 * *g as usize + 1]);
                // 宣言はされているが今回のマッチでは捕獲されなかったグループ
                // （例: `(a)|(b)` で `a` 側にマッチしたときの group 2）は
                // 空文字列扱い（DuckDB 実測: `select regexp_replace('a',
                // '(a)|(b)', 'X\2Y')` は `'XY'`）。
                if st != u32::MAX && en != u32::MAX {
                    out.extend_from_slice(&text[st as usize..en as usize]);
                }
            }
        }
    }
}

/// `regexp_replace` 本体。`global=false` なら最初の 1 致だけ、`true` なら
/// 全マッチを置換する。
///
/// グローバル置換で空文字列にマッチしうるパターン（例: `'x*'`）を扱う際、
/// 「直前に採用したマッチの終端と同じ位置に始まる空マッチ」は無視して
/// 1 バイトそのまま出力して先へ進める。これを入れないと `'aXbXc'` に対する
/// `regexp_replace(.., 'X*', '-', 'g')` が `-a--b--c-` になってしまうが、
/// DuckDB の実測は `-a-b-c-`（Python の `re.sub` 等と同じ「直前のマッチに
/// 隣接する空マッチは採用しない」規則）だったため、それに合わせている。
fn replace_into(prog: &Program, text: &[u8], repl: &[u8], global: bool, out: &mut Vec<u8>) {
    let toks = match parse_repl(repl) {
        Some((t, mg)) if mg as u16 <= prog.n_groups => t,
        _ => {
            out.extend_from_slice(text);
            return;
        }
    };
    if !global {
        match find(prog, text) {
            Ok(Some(saves)) => {
                let (mstart, mend) = (saves[0] as usize, saves[1] as usize);
                out.extend_from_slice(&text[..mstart]);
                emit_repl(&toks, text, &saves, out);
                out.extend_from_slice(&text[mend..]);
            }
            _ => out.extend_from_slice(text),
        }
        return;
    }
    let len = text.len();
    let mut pos = 0usize;
    let mut prev_end: Option<usize> = None;
    while pos <= len {
        let mut saves = match find(prog, &text[pos..]) {
            Ok(Some(sv)) => sv,
            _ => break,
        };
        for x in saves.iter_mut() {
            if *x != u32::MAX {
                *x += pos as u32;
            }
        }
        let mstart = saves[0] as usize;
        let mend = saves[1] as usize;
        if mstart == mend && Some(mstart) == prev_end {
            if mstart < len {
                out.push(text[mstart]);
            }
            pos = mstart + 1;
            continue;
        }
        out.extend_from_slice(&text[pos..mstart]);
        emit_repl(&toks, text, &saves, out);
        if mend > mstart {
            pos = mend;
        } else {
            if mend < len {
                out.push(text[mend]);
            }
            pos = mend + 1;
        }
        prev_end = Some(mend);
    }
    out.extend_from_slice(&text[pos.min(len)..]);
}

/// `regexp_replace(str, pattern, replacement[, 'g'])`。
pub fn eval_replace(args: &[&Vector]) -> Result<Vector> {
    ensure!(args.len() == 3 || args.len() == 4, WrongArgCount);
    let (n, s) = plan(args)?;
    let valid = combine(args, &s, n);
    let live = |i: usize| valid.as_ref().is_none_or(|b| b.get(i));
    let (sv, pv, rv) = (args[0], args[1], args[2]);
    let pat_const = s[1] == 0;
    let cached =
        if pat_const && n > 0 && pv.is_valid(0) { Some(compile(pv.bytes().get(0))?) } else { None };
    let mut out = BytesData::with_capacity(n, n * 8);
    let mut buf = Vec::new();
    for i in 0..n {
        if !live(i) {
            out.push_empty();
            continue;
        }
        let compiled;
        let prog: &Program = match &cached {
            Some(p) => p,
            None => {
                compiled = compile(pv.bytes().get(i * s[1]))?;
                &compiled
            }
        };
        let global = if args.len() == 4 {
            let f = args[3].bytes().get(i * s[3]);
            ensure!(f.is_empty() || f == b"g", UnsupportedFeature);
            f == b"g"
        } else {
            false
        };
        buf.clear();
        replace_into(prog, sv.bytes().get(i * s[0]), rv.bytes().get(i * s[2]), global, &mut buf);
        out.push(&buf);
    }
    let mut v = Vector::from_data(Ty::Varchar, Data::Bytes(out), valid);
    v.compact_validity();
    Ok(v)
}

// ============================================================================
// テスト
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::code_of;
    use crate::vector::Value;
    use std::time::Instant;

    fn m(s: &str, p: &str) -> bool {
        is_match(&compile(p.as_bytes()).unwrap(), s.as_bytes()).unwrap()
    }

    fn ext(s: &str, p: &str, g: u16) -> Option<String> {
        let prog = compile(p.as_bytes()).unwrap();
        let text = s.as_bytes();
        find(&prog, text).unwrap().map(|saves| {
            let (st, en) = (saves[2 * g as usize], saves[2 * g as usize + 1]);
            if st == u32::MAX || en == u32::MAX {
                String::new()
            } else {
                String::from_utf8(text[st as usize..en as usize].to_vec()).unwrap()
            }
        })
    }

    // --- 個々の構文 ----------------------------------------------------------

    #[test]
    fn literal_and_any() {
        assert!(m("foobar", "foo"));
        assert!(!m("foo", "bar"));
        assert!(m("a", "."));
        assert!(!m("\n", "."));
        assert!(m("axb", "a.b"));
    }

    #[test]
    fn quantifiers() {
        assert!(m("", "a*"));
        assert!(m("aaa", "a*"));
        assert!(!m("", "a+"));
        assert!(m("a", "a+"));
        assert!(m("", "a?"));
        assert!(m("a", "a?"));
        assert!(!m("aa", "^a?$"));
    }

    #[test]
    fn bounded_repeat() {
        assert!(m("aaa", "a{3}"));
        assert!(!m("aa", "^a{3}$"));
        assert!(m("aaaa", "^a{2,}$"));
        assert!(!m("a", "^a{2,}$"));
        assert!(m("aa", "^a{1,3}$"));
        assert!(m("aaa", "^a{1,3}$"));
        assert!(!m("aaaa", "^a{1,3}$"));
        assert!(m("b", "a{0}b"));
    }

    #[test]
    fn char_classes() {
        assert!(m("b", "[abc]"));
        assert!(!m("d", "[abc]"));
        assert!(m("d", "[^abc]"));
        assert!(m("m", "[a-z]"));
        assert!(!m("M", "[a-z]"));
        assert!(m("a.b", "a[.]b"));
        assert!(m("a-b", "[a-b-]"));
        assert!(m("-", "[-ab]"));
    }

    #[test]
    fn shorthand_classes() {
        assert!(m("5", "\\d"));
        assert!(!m("x", "^\\d$"));
        assert!(m("x", "\\D"));
        assert!(m("_", "\\w"));
        assert!(m(" ", "\\s"));
        assert!(!m("\u{0b}", "^\\s$")); // RE2 の \s は \v を含まない（実測済み）
        assert!(m("\u{0c}", "^\\s$"));
        assert!(m("!", "\\S"));
        assert!(m("3.14", "\\d+\\.\\d+"));
    }

    #[test]
    fn anchors() {
        assert!(m("bar", "^bar"));
        assert!(!m("foobar", "^bar"));
        assert!(m("ab", "ab$"));
        assert!(!m("\n", "^$")); // $ は入力末尾のみ（改行の手前ではない）
        assert!(m("", "^$"));
    }

    #[test]
    fn alternation_and_groups() {
        assert!(m("cat", "cat|dog"));
        assert!(m("dog", "cat|dog"));
        assert!(!m("fish", "^(cat|dog)$"));
        assert_eq!(ext("foobar", "(foo)(bar)", 1).as_deref(), Some("foo"));
        assert_eq!(ext("foobar", "(foo)(bar)", 2).as_deref(), Some("bar"));
        assert_eq!(ext("abc", "(?:a)(b)", 1).as_deref(), Some("b"));
    }

    #[test]
    fn escapes() {
        for c in [".", "*", "+", "?", "(", ")", "[", "]", "{", "}", "|", "^", "$", "\\"] {
            let pat = format!("\\{c}");
            assert!(m(c, &pat), "escape {c} failed");
        }
    }

    // --- 貪欲・最左優先の意味論（DuckDB と突き合わせ済み） --------------------

    #[test]
    fn greedy_and_leftmost() {
        // duckdb: regexp_extract('aaa','a*') = 'aaa'
        assert_eq!(ext("aaa", "a*", 0).as_deref(), Some("aaa"));
        // duckdb: regexp_extract('xaaay','a*') = ''（先頭 'x' の位置で 0 回一致）
        assert_eq!(ext("xaaay", "a*", 0).as_deref(), Some(""));
        // duckdb: regexp_extract('abc','a|ab') = 'a'（選択は左優先、最長優先ではない）
        assert_eq!(ext("abc", "a|ab", 0).as_deref(), Some("a"));
        // duckdb: regexp_extract('xabcx','ab|abc') = 'ab'
        assert_eq!(ext("xabcx", "ab|abc", 0).as_deref(), Some("ab"));
    }

    // --- コンパイルエラー ------------------------------------------------------

    #[test]
    fn rejects_unsupported_constructs() {
        assert_eq!(code_of(compile(b"(?=a)").map(|_| ())), Some(Code::UnsupportedFeature));
        assert_eq!(code_of(compile(b"(?!a)").map(|_| ())), Some(Code::UnsupportedFeature));
        assert_eq!(code_of(compile(b"(?<=a)").map(|_| ())), Some(Code::UnsupportedFeature));
        assert_eq!(code_of(compile(b"(?P<n>a)").map(|_| ())), Some(Code::UnsupportedFeature));
        assert_eq!(code_of(compile(b"(a)\\1").map(|_| ())), Some(Code::UnsupportedFeature));
        assert_eq!(code_of(compile(b"a*?").map(|_| ())), Some(Code::UnsupportedFeature));
        assert_eq!(code_of(compile(b"a+?").map(|_| ())), Some(Code::UnsupportedFeature));
        assert_eq!(code_of(compile(b"\\bfoo").map(|_| ())), Some(Code::UnsupportedFeature));
    }

    #[test]
    fn rejects_malformed_patterns() {
        assert_eq!(code_of(compile(b"(a").map(|_| ())), Some(Code::SyntaxError));
        assert_eq!(code_of(compile(b"a)").map(|_| ())), Some(Code::SyntaxError));
        assert_eq!(code_of(compile(b"[a").map(|_| ())), Some(Code::SyntaxError));
        assert_eq!(code_of(compile(b"*a").map(|_| ())), Some(Code::SyntaxError));
        assert_eq!(code_of(compile(b"a**").map(|_| ())), Some(Code::SyntaxError));
        assert_eq!(code_of(compile(b"a{2,1}").map(|_| ())), Some(Code::SyntaxError));
        assert_eq!(code_of(compile(b"\\").map(|_| ())), Some(Code::SyntaxError));
        assert!(compile(b"foobar").is_ok());
    }

    // --- 病的な入力でも高速に完了すること ---------------------------------------

    #[test]
    fn pathological_pattern_is_not_exponential() {
        // (a+)+b は素朴な再帰バックトラックだと典型的に指数時間になるパターン。
        // Thompson NFA なら O(命令数 × 入力長) で完了するはず。
        let subject = "a".repeat(5000);
        let prog = compile(b"(a+)+b").unwrap();
        let start = Instant::now();
        let hit = is_match(&prog, subject.as_bytes()).unwrap();
        let elapsed = start.elapsed();
        assert!(!hit);
        assert!(elapsed.as_millis() < 2000, "took too long: {elapsed:?}");
    }

    #[test]
    fn step_limit_triggers_on_adversarial_input() {
        // 典型的な NFA スレッド爆発パターン `a?{n}a{n}` を長大な非マッチ入力に
        // 当てて MAX_STEPS に到達させる。命令数自体は MAX_PROG_INSTS
        // (4096) に収まる範囲（1000*2 + 1000 + 3 = 3003）に抑えつつ、
        // 1 ステップあたりの生存スレッド数が最大 ~1000 になるようにして
        // 長い入力なら早々に MAX_STEPS を超えるようにする。
        let pat: String = "a?".repeat(1000) + &"a".repeat(1000);
        let prog = compile(pat.as_bytes()).unwrap();
        // 999 個の 'a' ごとに 'b' を挟み、連続する 'a' が 1000 個未満にしか
        // ならないようにする（マッチしない）。これにより探索は最後まで
        // 「あと少しで届かない」状態を延々繰り返し、早期にマッチが見つかって
        // 打ち切られることなく MAX_STEPS 判定まで到達する。
        let block = "a".repeat(999) + "b";
        let subject = block.repeat(5000);
        let start = Instant::now();
        let r = is_match(&prog, subject.as_bytes());
        let elapsed = start.elapsed();
        assert_eq!(code_of(r.map(|_| ())), Some(Code::LimitExceeded));
        assert!(elapsed.as_millis() < 5000, "took too long: {elapsed:?}");
    }

    #[test]
    fn prog_size_limit_triggers() {
        // {1000} を何個も連結して命令数上限を超えさせる。
        let pat = "a{1000}".repeat(10);
        assert_eq!(code_of(compile(pat.as_bytes()).map(|_| ())), Some(Code::LimitExceeded));
    }

    #[test]
    fn nested_repeat_does_not_blow_up_before_limit_check() {
        // 展開すると 1000^3 命令相当になるはずのパターンでも、即座に
        // LimitExceeded で終わること（AST を複製していないので OOM しない）。
        let start = Instant::now();
        let r = compile(b"((a{1000}){1000}){1000}");
        let elapsed = start.elapsed();
        assert_eq!(code_of(r.map(|_| ())), Some(Code::LimitExceeded));
        assert!(elapsed.as_millis() < 2000, "took too long: {elapsed:?}");
    }

    // --- Vector 単位の SQL 関数 -------------------------------------------------

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

    fn vi(vals: &[Option<i64>]) -> Vector {
        let mut v = Vector::new(Ty::BigInt);
        for x in vals {
            match x {
                Some(n) => v.push_value(&Value::I64(*n)),
                None => v.push_null(),
            }
        }
        v
    }

    fn str_at(v: &Vector, i: usize) -> Option<String> {
        if !v.is_valid(i) {
            return None;
        }
        Some(String::from_utf8(v.bytes().get(i).to_vec()).unwrap())
    }

    fn bool_at(v: &Vector, i: usize) -> Option<bool> {
        if !v.is_valid(i) {
            return None;
        }
        match v.data() {
            Data::Bool(b) => Some(b.get(i)),
            _ => None,
        }
    }

    #[test]
    fn sql_matches_basics_and_nulls() {
        let s = vs(&[Some("foobar"), Some("xxx"), None, Some("abc")]);
        let p = vs(&[Some("o+b"), Some("foo"), Some("a"), None]);
        let r = eval_matches(&[&s, &p]).unwrap();
        assert_eq!(bool_at(&r, 0), Some(true));
        assert_eq!(bool_at(&r, 1), Some(false));
        assert_eq!(bool_at(&r, 2), None);
        assert_eq!(bool_at(&r, 3), None);
    }

    #[test]
    fn sql_matches_constant_pattern_len1_result() {
        let s = vs(&[Some("abc")]);
        let p = vs(&[Some("b")]);
        let r = eval_matches(&[&s, &p]).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(bool_at(&r, 0), Some(true));
    }

    #[test]
    fn sql_extract_no_match_and_group_range() {
        let s = vs(&[Some("xxx"), Some("foobar"), Some("foobar")]);
        let p = vs(&[Some("foo"), Some("(foo)(bar)"), Some("(foo)(bar)")]);
        let g = vi(&[Some(0), Some(1), Some(9)]);
        let r = eval_extract(&[&s, &p, &g]).unwrap();
        // マッチしない場合も空文字列（NULL ではない）。DuckDB 実測どおり。
        assert_eq!(str_at(&r, 0).as_deref(), Some(""));
        assert_eq!(str_at(&r, 1).as_deref(), Some("foo"));
        // group=9 は 0〜9 の範囲内だが、このパターンの捕獲数(2)を超えるので空文字列。
        assert_eq!(str_at(&r, 2).as_deref(), Some(""));
    }

    #[test]
    fn sql_extract_group_out_of_hard_range_errors() {
        let s = vs(&[Some("a")]);
        let p = vs(&[Some("a")]);
        let g = vi(&[Some(10)]);
        assert_eq!(code_of(eval_extract(&[&s, &p, &g]).map(|_| ())), Some(Code::ValueOutOfRange));
        let g2 = vi(&[Some(-1)]);
        assert_eq!(code_of(eval_extract(&[&s, &p, &g2]).map(|_| ())), Some(Code::ValueOutOfRange));
    }

    #[test]
    fn sql_extract_null_propagation() {
        let s = vs(&[None]);
        let p = vs(&[Some("a")]);
        let r = eval_extract(&[&s, &p]).unwrap();
        assert_eq!(str_at(&r, 0), None);
    }

    #[test]
    fn sql_replace_first_vs_global() {
        let s = vs(&[Some("hello world")]);
        let p = vs(&[Some("o")]);
        let r0 = vs(&[Some("0")]);
        let first = eval_replace(&[&s, &p, &r0]).unwrap();
        assert_eq!(str_at(&first, 0).as_deref(), Some("hell0 world"));
        let flag = vs(&[Some("g")]);
        let global = eval_replace(&[&s, &p, &r0, &flag]).unwrap();
        assert_eq!(str_at(&global, 0).as_deref(), Some("hell0 w0rld"));
    }

    #[test]
    fn sql_replace_backreferences() {
        let s = vs(&[Some("foobar")]);
        let p = vs(&[Some("(foo)(bar)")]);
        let r1 = vs(&[Some("\\2\\1")]);
        let out = eval_replace(&[&s, &p, &r1]).unwrap();
        assert_eq!(str_at(&out, 0).as_deref(), Some("barfoo"));

        let flag = vs(&[Some("g")]);
        let r2 = vs(&[Some("\\2-\\1")]);
        let out2 = eval_replace(&[&s, &p, &r2, &flag]).unwrap();
        assert_eq!(str_at(&out2, 0).as_deref(), Some("bar-foo"));
    }

    #[test]
    fn sql_replace_invalid_backreference_is_noop() {
        // DuckDB 実測: グループ参照がパターンの捕獲数を超えると置換自体が
        // 行われず元の文字列がそのまま返る。
        let s = vs(&[Some("abc")]);
        let p = vs(&[Some("b")]);
        let r = vs(&[Some("X\\9")]);
        let out = eval_replace(&[&s, &p, &r]).unwrap();
        assert_eq!(str_at(&out, 0).as_deref(), Some("abc"));
    }

    #[test]
    fn sql_replace_global_empty_match_adjacency() {
        // duckdb: regexp_replace('aXbXc','X*','-','g') = '-a-b-c-'
        let s = vs(&[Some("aXbXc")]);
        let p = vs(&[Some("X*")]);
        let r = vs(&[Some("-")]);
        let flag = vs(&[Some("g")]);
        let out = eval_replace(&[&s, &p, &r, &flag]).unwrap();
        assert_eq!(str_at(&out, 0).as_deref(), Some("-a-b-c-"));
    }

    #[test]
    fn sql_replace_null_propagation() {
        let s = vs(&[None]);
        let p = vs(&[Some("a")]);
        let r = vs(&[Some("b")]);
        let out = eval_replace(&[&s, &p, &r]).unwrap();
        assert_eq!(str_at(&out, 0), None);
    }

    #[test]
    fn sql_replace_bad_flag_errors() {
        let s = vs(&[Some("abc")]);
        let p = vs(&[Some("a")]);
        let r = vs(&[Some("x")]);
        let bad = vs(&[Some("i")]);
        assert_eq!(
            code_of(eval_replace(&[&s, &p, &r, &bad]).map(|_| ())),
            Some(Code::UnsupportedFeature)
        );
    }

    #[test]
    fn sql_matches_len1_result_for_constant_folding() {
        let s = vs(&[Some("abc")]);
        let p = vs(&[Some("b")]);
        let r = eval_matches(&[&s, &p]).unwrap();
        assert_eq!(r.len(), 1);
    }
}
