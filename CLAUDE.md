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
cargo test --workspace
cargo fmt --all --check
cargo clippy --workspace --all-targets
```

If touching `crates/ahiru-core`, also check the opt-in feature combinations
that apply (`csv`, `jsonl`, `export`, `export-parquet`, `ddl`, `dml`) and the
wasm size budget:

```bash
./scripts/size.sh
```

If touching `js/`, run the JS host test suite:

```bash
node --test 'js/test/*.test.mjs'
```
