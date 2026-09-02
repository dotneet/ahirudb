//! The expression bytecode VM.
//!
//! Instead of recursively evaluating an expression tree, it is compiled into a flat instruction
//! sequence and executed via `kernel_table[op][phys_type]` (DESIGN.md §9). Replacing generics'
//! monomorphization blowup with a single table means code size grows only linearly as operators
//! and types are added.
//!
//! ## A design decision: no control flow
//!
//! DESIGN.md envisioned short-circuiting `AND`/`OR`/`CASE` with branch instructions, but the
//! implementation instead **has no branches and evaluates both sides, combining them with
//! `Select`**. Branching per row would break vectorization, and it would need rewinding the
//! instruction pointer, growing the VM. The one case that would need short-circuiting is division
//! by zero, and that is resolved, as in DuckDB, by **returning NULL rather than an error**.
//!
//! ## Handling NULLs
//!
//! Kernels separate "computing the value" from "computing the validity". For most operations the
//! result's validity is the AND of the inputs' validity, but `AND`/`OR` alone need the value and
//! the validity decided together, because of three-valued logic.

pub mod funcs;
pub mod kernels;
pub mod regex;
pub mod vm;

// The shortest-round-trip `f64` -> text formatter. It physically lives in
// `write/float.rs` (where it started life, shared by the CSV and JSONL
// writers), but the always-on `CAST(<double> AS VARCHAR)` path needs the very
// same rendering, and `write` as a whole is gated behind the opt-in `export`
// feature. Including the file here by path, rather than copying it, keeps a
// single source of truth for float formatting across the whole crate: a fix
// to the algorithm cannot land on the cast path and miss the writers, or the
// other way round. See the note in the module's own docs about why one shared
// copy matters. (Follow-up for whoever owns `write/`: once `write/mod.rs`
// switches its `mod float;` to `use crate::expr::float;`, the module stops
// being compiled twice in builds that enable `export` + `csv`/`jsonl`.)
// `clippy::duplicate_mod` is exactly the follow-up above: with `export` +
// `csv`/`jsonl` all enabled, this file is compiled as both `expr::float` and
// `write::float`. The alternative -- a second copy of a 600-line intricate
// algorithm -- is strictly worse, and this crate's own `write/float.rs` module
// doc explains why one shared copy matters. Removing the duplication needs a
// one-line change in `write/mod.rs`, which is not this module's to make.
#[allow(clippy::duplicate_mod)]
#[path = "../write/float.rs"]
pub(crate) mod float;

use crate::prelude::*;
use crate::vector::{PhysType, Ty, Value};

/// A register number.
pub type Reg = u16;

/// An instruction. 12 bytes each.
#[derive(Clone, Copy)]
pub struct Instr {
    pub op: OpCode,
    /// The physical type operated on. For comparisons it means the **input's** type (the output is always Bool).
    pub ty: PhysType,
    pub dst: Reg,
    pub a: Reg,
    pub b: Reg,
    /// An auxiliary field whose meaning depends on the instruction.
    /// For `LoadCol` it is the column number, for `LoadConst` the constant pool index, for `Cast`
    /// the cast table index, and for `Select` the third operand's register number.
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
    /// dst = column aux of the input batch
    LoadCol,
    /// dst = `constant pool[aux]` as a length-1 vector
    LoadConst,

    // Arithmetic. ty is one of {I32, I64, I128, F64}
    Add,
    Sub,
    Mul,
    /// Division by zero returns NULL (rather than an error).
    Div,
    Mod,
    Neg,

    // Comparison. The input is ty and the output is Bool.
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    // Three-valued logic. Both input and output are Bool.
    And,
    Or,
    Not,

    /// dst = whether a is NULL. The result is never NULL.
    IsNull,
    IsNotNull,

    /// dst = a converted according to `cast table[aux]`.
    Cast,
    /// `TRY_CAST`. Uses the same cast table as `Cast`, but an unconvertible combination
    /// (where `kernels::cast` returns an error) becomes all-NULL rather than an error.
    /// A per-row conversion failure (out of range, unparsable) makes just that row NULL, as with
    /// `Cast` (the contract of `kernels::cast` itself).
    TryCast,

    /// dst = a LIKE b. ty is Bytes.
    Like,
    /// dst = a || b. ty is Bytes.
    Concat,

    /// dst = b if a (Bool) is true, or register aux if it is false or NULL.
    /// `CASE` and `COALESCE` lower to this.
    Select,

    /// dst = b if a is NULL, otherwise a.
    Coalesce,

    /// A scalar function call. `aux` is an index into `Program::calls`.
    ///
    /// The arguments are a variable-length list of register numbers and do not fit in `a`/`b`.
    /// Making instructions variable-length would complicate the VM's loop, so only the argument
    /// list is moved to a separate table.
    Call,

    // INTERVAL operations. `a` and `b` have different physical types (TIMESTAMP is I64 and INTERVAL
    // is I128), so they cannot ride `arith`'s table, which assumes a single physical type, and are
    // separated out. Needing calendar arithmetic (month-end clamping) is another difference from the other arithmetic.
    /// dst(I64) = a(TIMESTAMP, I64) + b(INTERVAL, I128). DATE is cast to TIMESTAMP by the caller
    /// before being passed in.
    TsAddInterval,
    /// dst(I128) = a(INTERVAL) + b(INTERVAL). Field-wise addition.
    IntervalAdd,
    /// dst(I128) = a(INTERVAL) negated. `Sub` and unary `-` expand into a combination of this and
    /// `IntervalAdd`/`TsAddInterval`.
    IntervalNeg,
    /// dst(I128) = a(INTERVAL) * b(BIGINT). Field-wise multiplication.
    IntervalMul,
}

/// The arguments of a scalar function call.
#[derive(Clone)]
pub struct CallSpec {
    pub func: u16,
    pub args: Vec<Reg>,
    /// The logical type the function returns.
    pub result_ty: Ty,
    /// The lambda body of `list_transform`/`list_filter`/`list_reduce`.
    /// An index into `Program::lambdas` (`None` for an ordinary scalar function call).
    /// See the `Call` instruction in `expr::vm::exec` and `expr::funcs::call_lambda`.
    pub lambda: Option<u16>,
}

/// A cast specification. The logical type is needed for DECIMAL scale adjustment.
#[derive(Clone, Copy)]
pub struct CastSpec {
    pub from: Ty,
    pub to: Ty,
}

/// A compiled expression.
///
/// `Clone` is needed for GROUPING SETS support: the same grouping-column and aggregate expressions
/// are duplicated into each of the `Node::Aggregate`s built per grouping set
/// (the input scope is the same for every set, so the register allocation can be reused as is).
#[derive(Clone)]
pub struct Program {
    pub instrs: Vec<Instr>,
    /// The argument table for scalar function calls.
    pub calls: Vec<CallSpec>,
    /// The constant pool. `Ty` retains the logical type (DATE and so on) that `Value` alone does not determine.
    pub consts: Vec<(Ty, Value)>,
    pub casts: Vec<CastSpec>,
    /// The lambda body programs of `list_transform`/`list_filter`/`list_reduce`.
    /// Referenced by index from `CallSpec::lambda`. A body is compiled in an isolated scope that
    /// can reference only its own parameters (columns of the enclosing scope are unreachable; see
    /// `plan::compile::Compiler::lambda_call`).
    pub lambdas: Vec<Program>,
    pub num_regs: u16,
    /// Set when one of the `u16`-wide counters below (registers, constants,
    /// calls, casts, lambdas) ran out of room while this program was being
    /// built.
    ///
    /// The counters cannot simply be widened: `Instr` is a packed 12-byte
    /// record whose operand fields are `u16` by design (DESIGN.md §9). So
    /// instead of truncating an index — which silently aliases two different
    /// values onto one register, and in `profile.wasm` (overflow checks off,
    /// `panic = "abort"`) does so with no diagnostic at all — the builder
    /// saturates and records it here, and `expr::vm::Vm::eval` refuses to run
    /// the program with `LimitExceeded`. That turns "wrong answer, silently"
    /// into "clean error", which is the rule the rest of the engine follows
    /// for resource ceilings (DESIGN.md §9).
    pub overflow: bool,
    /// The register the result lands in.
    pub result: Reg,
    /// The result's logical type.
    pub result_ty: Ty,
}

impl Program {
    pub fn new() -> Self {
        Program {
            instrs: Vec::new(),
            calls: Vec::new(),
            consts: Vec::new(),
            casts: Vec::new(),
            lambdas: Vec::new(),
            num_regs: 0,
            overflow: false,
            result: 0,
            result_ty: Ty::Null,
        }
    }

    /// Allocates a new register.
    ///
    /// A register number is a `u16`, so at most `u16::MAX` of them exist. Past
    /// that the allocation saturates and sets [`Program::overflow`] instead of
    /// wrapping around and handing out a register that is already in use.
    pub fn alloc_reg(&mut self) -> Reg {
        match self.num_regs.checked_add(1) {
            Some(next) => {
                let r = self.num_regs;
                self.num_regs = next;
                r
            }
            None => {
                self.overflow = true;
                Reg::MAX
            }
        }
    }

    /// Reserves the next index in a side table, or records the overflow.
    /// `len` is the table's length *before* the push the caller is about to do.
    fn next_index(&mut self, len: usize) -> Option<u16> {
        if self.overflow || len >= u16::MAX as usize {
            self.overflow = true;
            return None;
        }
        Some(len as u16)
    }

    pub fn add_const(&mut self, ty: Ty, v: Value) -> u16 {
        // Identical constants are shared. Constants are usually few, so a linear scan suffices.
        // The scan is skipped once the program is already doomed, so a pathologically large
        // literal list stops costing quadratic work the moment the limit is hit.
        if !self.overflow {
            for (i, (t, existing)) in self.consts.iter().enumerate() {
                if *t == ty && *existing == v {
                    return i as u16;
                }
            }
        }
        let Some(i) = self.next_index(self.consts.len()) else { return 0 };
        self.consts.push((ty, v));
        i
    }

    pub fn add_call(&mut self, func: u16, args: Vec<Reg>, result_ty: Ty) -> u16 {
        let Some(i) = self.next_index(self.calls.len()) else { return 0 };
        self.calls.push(CallSpec { func, args, result_ty, lambda: None });
        i
    }

    /// For `list_transform`/`list_filter`/`list_reduce`. `lambda` is the index `add_lambda` returned.
    pub fn add_lambda_call(
        &mut self,
        func: u16,
        args: Vec<Reg>,
        result_ty: Ty,
        lambda: u16,
    ) -> u16 {
        let Some(i) = self.next_index(self.calls.len()) else { return 0 };
        self.calls.push(CallSpec { func, args, result_ty, lambda: Some(lambda) });
        i
    }

    /// Embeds a lambda body program and returns the index to pass to `CallSpec::lambda`.
    ///
    /// A body that itself overflowed poisons the enclosing program: the body is
    /// executed through the same `Vm::eval` gate, but failing here as well means
    /// the outer program never starts either.
    pub fn add_lambda(&mut self, body: Program) -> u16 {
        if body.overflow {
            self.overflow = true;
        }
        let Some(i) = self.next_index(self.lambdas.len()) else { return 0 };
        self.lambdas.push(body);
        i
    }

    pub fn add_cast(&mut self, from: Ty, to: Ty) -> u16 {
        let Some(i) = self.next_index(self.casts.len()) else { return 0 };
        self.casts.push(CastSpec { from, to });
        i
    }

    pub fn push(&mut self, i: Instr) {
        self.instrs.push(i);
    }

    /// Whether it consists only of constants (for the constant-folding check).
    pub fn is_constant(&self) -> bool {
        !self.instrs.iter().any(|i| i.op == OpCode::LoadCol)
    }
}

impl Default for Program {
    fn default() -> Self {
        Program::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Register numbers are `u16`. Past `u16::MAX` the allocator must not hand
    /// out a register that is already live: two values sharing one register is a
    /// wrong answer, and in `profile.wasm` (overflow checks off) it used to
    /// happen silently.
    #[test]
    fn register_allocation_saturates_instead_of_wrapping() {
        let mut p = Program::new();
        for i in 0..u16::MAX {
            assert_eq!(p.alloc_reg(), i);
        }
        assert!(!p.overflow, "exactly u16::MAX registers still fit");
        assert_eq!(p.alloc_reg(), u16::MAX);
        assert!(p.overflow);
        // It stays saturated rather than wrapping back to 0.
        assert_eq!(p.alloc_reg(), u16::MAX);
        assert_eq!(p.num_regs, u16::MAX);
    }

    /// The side tables are indexed by `u16` too, and `(len - 1) as u16` used to
    /// truncate silently once a table passed 65536 entries.
    #[test]
    fn side_table_indices_report_overflow_instead_of_truncating() {
        let mut p = Program::new();
        for i in 0..(u16::MAX as i32) {
            assert_eq!(p.add_const(Ty::Int, Value::I32(i)), i as u16);
        }
        assert!(!p.overflow);
        assert_eq!(p.add_const(Ty::Int, Value::I32(-1)), 0);
        assert!(p.overflow);
        // An index already in the pool still resolves; only new entries fail.
        assert_eq!(p.consts.len(), u16::MAX as usize);
    }

    #[test]
    fn cast_and_call_tables_share_the_same_guard() {
        let mut p = Program::new();
        for _ in 0..u16::MAX {
            p.add_cast(Ty::Int, Ty::BigInt);
        }
        assert!(!p.overflow);
        p.add_cast(Ty::Int, Ty::BigInt);
        assert!(p.overflow);

        let mut q = Program::new();
        for _ in 0..u16::MAX {
            q.add_call(0, Vec::new(), Ty::Int);
        }
        assert!(!q.overflow);
        q.add_call(0, Vec::new(), Ty::Int);
        assert!(q.overflow);
    }

    /// A lambda body that overflowed poisons the program that embeds it, so the
    /// outer program never starts either.
    #[test]
    fn a_poisoned_lambda_body_poisons_its_parent() {
        let mut body = Program::new();
        body.overflow = true;
        let mut p = Program::new();
        p.add_lambda(body);
        assert!(p.overflow);
    }
}
