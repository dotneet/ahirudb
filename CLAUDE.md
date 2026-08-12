# ahirudb

A lightweight SQL engine that queries Parquet (and CSV/JSONL/JSON) directly,
built to run in under 1 MiB of WASM. See [docs/DESIGN.md](docs/DESIGN.md) for
the full architecture and design rationale before making non-trivial changes —
several constraints there (the 1 MiB budget, the 6-physical-type limit, the
split-boundary I/O barrier, `no_std`/no `core::fmt`) are load-bearing and easy
to violate by accident.

## Language policy

Write all documentation and code comments and commit comment in **English** — README files,
`docs/`, doc comments, and inline comments in new or edited code.

The existing codebase has a large amount of Japanese in comments and internal
docs from earlier development. Leave those as-is when touching nearby code
unless the edit already requires rewriting the comment; don't do drive-by
translation. But any comment or documentation you *write* — new code, edits to
existing comments, new docs — should be in English.

Code identifiers (function/type/variable names), commit messages, and
conversation with the user follow whatever the user has separately instructed;
this policy is about documentation and comments specifically.

## Verifying changes

Before considering a change done, run:

```bash
cargo build --workspace
cargo test --workspace --all-features
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features
```

Keep `--all-features`: the `dml`- and `export`-gated integration tests are
compiled out without it, and `cargo test` then passes without running them.

If touching `crates/ahiru-core`, also check the opt-in feature combinations
that apply (`csv`, `jsonl`, `export`, `export-parquet`, `ddl`, `dml`) on the
wasm target, and the wasm size budget. `--target wasm32-unknown-unknown` is
required: a `no_std` build for the host target fails, because the host is
`panic=unwind` while the crate brings its own panic handler. Use
`--profile wasm` so the check matches what actually ships.

```bash
cargo check -p ahiru-core --target wasm32-unknown-unknown --profile wasm \
  --no-default-features --features zstd,csv,jsonl,export,export-parquet,dml
```

```bash
./scripts/size.sh
```

If touching `js/`, run the JS host test suite:

```bash
node --test 'js/test/*.test.mjs'
```
