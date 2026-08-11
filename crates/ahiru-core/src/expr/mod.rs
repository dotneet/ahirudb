//! 式のバイトコード VM。
//!
//! 式ツリーを再帰評価する代わりに、フラットな命令列にコンパイルして
//! `kernel_table[op][phys_type]` で実行する（DESIGN.md §9）。ジェネリクスに
//! よる単相化爆発をテーブル 1 枚に置き換えられるので、演算子や型が増えても
//! コードサイズが線形にしか伸びない。
//!
//! ## 設計上の判断: 制御フローを持たない
//!
//! DESIGN.md では `AND`/`OR`/`CASE` を分岐命令で短絡評価する想定だったが、
//! 実装では**分岐を持たず両辺を評価して `Select` で合成する**方式に変えた。
//! 行ごとに分岐するとベクトル化が壊れるうえ、命令ポインタの巻き戻しが必要に
//! なり VM が大きくなるため。短絡が必要になる唯一のケースはゼロ除算だが、
//! これは DuckDB と同じく**エラーではなく NULL を返す**ことで解消している。
//!
//! ## NULL の扱い
//!
//! カーネルは「値の計算」と「validity の計算」を分離する。ほとんどの演算では
//! 結果の validity = 入力の validity の AND だが、`AND`/`OR` だけは三値論理の
//! ため値と validity を同時に決める必要がある。

pub mod funcs;
pub mod kernels;
pub mod regex;
pub mod vm;

use crate::prelude::*;
use crate::vector::{PhysType, Ty, Value};

/// レジスタ番号。
pub type Reg = u16;

/// 命令。1 命令 12 バイト。
#[derive(Clone, Copy)]
pub struct Instr {
    pub op: OpCode,
    /// 演算対象の物理型。比較では「入力の」型を指す（出力は常に Bool）。
    pub ty: PhysType,
    pub dst: Reg,
    pub a: Reg,
    pub b: Reg,
    /// 命令ごとに意味が異なる補助フィールド。
    /// `LoadCol` は列番号、`LoadConst` は定数プール添字、`Cast` はキャスト表の
    /// 添字、`Select` は第 3 オペランドのレジスタ番号。
    pub aux: u16,
}

impl Instr {
    pub fn new(op: OpCode, ty: PhysType, dst: Reg, a: Reg, b: Reg) -> Self {
        Instr { op, ty, dst, a, b, aux: 0 }
    }

    pub fn with_aux(op: OpCode, ty: PhysType, dst: Reg, a: Reg, b: Reg, aux: u16) -> Self {
        Instr { op, ty, dst, a, b, aux }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(Debug))]
#[repr(u8)]
pub enum OpCode {
    /// dst = 入力バッチの aux 列
    LoadCol,
    /// dst = 定数プール[aux] を長さ 1 のベクタとして
    LoadConst,

    // 算術。ty ∈ {I32, I64, I128, F64}
    Add,
    Sub,
    Mul,
    /// ゼロ除算は NULL を返す（エラーにしない）。
    Div,
    Mod,
    Neg,

    // 比較。入力は ty、出力は Bool。
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    // 三値論理。入力・出力とも Bool。
    And,
    Or,
    Not,

    /// dst = a が NULL か。結果は決して NULL にならない。
    IsNull,
    IsNotNull,

    /// dst = キャスト表[aux] に従って a を変換。
    Cast,
    /// `TRY_CAST`。`Cast` と同じキャスト表を使うが、変換できない組み合わせ
    /// （`kernels::cast` がエラーを返す場合）はエラーにせず全行 NULL にする。
    /// 行単位の変換失敗（範囲外・パース不能）は `Cast` と同じくその行だけ
    /// NULL になる（`kernels::cast` 自体の契約）。
    TryCast,

    /// dst = a LIKE b。ty は Bytes。
    Like,
    /// dst = a || b。ty は Bytes。
    Concat,

    /// dst = a(Bool) が真なら b、偽または NULL なら レジスタ aux。
    /// `CASE` と `COALESCE` はこれに落とす。
    Select,

    /// dst = a が NULL なら b、そうでなければ a。
    Coalesce,

    /// スカラ関数呼び出し。`aux` は `Program::calls` の添字。
    ///
    /// 引数はレジスタ番号の可変長リストなので、`a`/`b` には収まらない。
    /// 命令を可変長にすると VM のループが複雑になるため、引数リストだけを
    /// 別表に逃がしている。
    Call,

    // INTERVAL 演算。a・b の物理型が異なる（TIMESTAMP は I64、INTERVAL は
    // I128）ため、`arith` の物理型 1 種前提のテーブルには乗せられず専用に
    // 分けてある。カレンダー演算（月末クランプ）が要るのも他の算術と違う点。
    /// dst(I64) = a(TIMESTAMP, I64) + b(INTERVAL, I128)。DATE は呼び出し側が
    /// TIMESTAMP へキャストしてから渡す。
    TsAddInterval,
    /// dst(I128) = a(INTERVAL) + b(INTERVAL)。フィールドごとの加算。
    IntervalAdd,
    /// dst(I128) = a(INTERVAL) の符号反転。`Sub`/単項 `-` はこれと
    /// `IntervalAdd`/`TsAddInterval` の組み合わせに展開する。
    IntervalNeg,
    /// dst(I128) = a(INTERVAL) * b(BIGINT)。フィールドごとの乗算。
    IntervalMul,
}

/// スカラ関数呼び出しの引数。
#[derive(Clone)]
pub struct CallSpec {
    pub func: u16,
    pub args: Vec<Reg>,
    /// 関数が返す論理型。
    pub result_ty: Ty,
}

/// キャストの指定。論理型が要るのは DECIMAL のスケール調整のため。
#[derive(Clone, Copy)]
pub struct CastSpec {
    pub from: Ty,
    pub to: Ty,
}

/// コンパイル済みの式。
///
/// `Clone` は GROUPING SETS 対応で要る: 同じグルーピング列や集約の式を、
/// グルーピングセットの数だけ作る `Node::Aggregate` それぞれに複製して積む
/// （入力スコープはどのセットでも同じなのでレジスタ割り当てはそのまま使い回せる）。
#[derive(Clone)]
pub struct Program {
    pub instrs: Vec<Instr>,
    /// スカラ関数呼び出しの引数表。
    pub calls: Vec<CallSpec>,
    /// 定数プール。`Ty` は `Value` だけでは決まらない論理型（DATE など）を保持する。
    pub consts: Vec<(Ty, Value)>,
    pub casts: Vec<CastSpec>,
    pub num_regs: u16,
    /// 結果が入るレジスタ。
    pub result: Reg,
    /// 結果の論理型。
    pub result_ty: Ty,
}

impl Program {
    pub fn new() -> Self {
        Program {
            instrs: Vec::new(),
            calls: Vec::new(),
            consts: Vec::new(),
            casts: Vec::new(),
            num_regs: 0,
            result: 0,
            result_ty: Ty::Null,
        }
    }

    /// 新しいレジスタを割り当てる。
    pub fn alloc_reg(&mut self) -> Reg {
        let r = self.num_regs;
        self.num_regs += 1;
        r
    }

    pub fn add_const(&mut self, ty: Ty, v: Value) -> u16 {
        // 同じ定数は共有する。定数はたいてい少数なので線形探索で足りる。
        for (i, (t, existing)) in self.consts.iter().enumerate() {
            if *t == ty && *existing == v {
                return i as u16;
            }
        }
        self.consts.push((ty, v));
        (self.consts.len() - 1) as u16
    }

    pub fn add_call(&mut self, func: u16, args: Vec<Reg>, result_ty: Ty) -> u16 {
        self.calls.push(CallSpec { func, args, result_ty });
        (self.calls.len() - 1) as u16
    }

    pub fn add_cast(&mut self, from: Ty, to: Ty) -> u16 {
        self.casts.push(CastSpec { from, to });
        (self.casts.len() - 1) as u16
    }

    pub fn push(&mut self, i: Instr) {
        self.instrs.push(i);
    }

    /// 定数だけで構成されているか（定数畳み込みの判定用）。
    pub fn is_constant(&self) -> bool {
        !self.instrs.iter().any(|i| i.op == OpCode::LoadCol)
    }
}

impl Default for Program {
    fn default() -> Self {
        Program::new()
    }
}
