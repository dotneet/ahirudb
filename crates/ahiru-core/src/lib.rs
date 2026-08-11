//! ahirudb — WASM 1MB 以内で動く Parquet 用 SQL エンジン。
//!
//! 設計の詳細は `docs/DESIGN.md` を参照。
//!
//! ビルド構成:
//! - ネイティブ (既定): `std` フィーチャ有効。テストとデバッグはこちらで行う。
//! - wasm 配布: `--no-default-features` で `no_std` になり、`core::fmt` を
//!   一切リンクしない。

#![cfg_attr(not(feature = "std"), no_std)]
// no_std でも alloc は常に使う。
extern crate alloc;

#[macro_use]
pub mod error;

pub mod rt;

pub mod vector;

pub mod parquet;

pub mod catalog;

pub mod format;

pub mod sql;

pub mod plan;

pub mod expr;

pub mod exec;

pub mod session;

#[cfg(target_arch = "wasm32")]
pub mod abi;

/// クレート内で共通に使う import。
pub(crate) mod prelude {
    #![allow(unused_imports)]

    pub use alloc::borrow::ToOwned;
    pub use alloc::boxed::Box;
    pub use alloc::string::String;
    pub use alloc::vec;
    pub use alloc::vec::Vec;

    pub use crate::error::{Code, Error, Result};
}

pub use catalog::{Catalog, Source};
pub use format::{FormatKind, TableFormat};
pub use session::{Prepared, Query, QueryStep, Session};
