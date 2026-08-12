//! ahirudb -- a SQL engine for Parquet that runs in under 1 MB of WASM.
//!
//! See `docs/DESIGN.md` for the design in detail.
//!
//! Build configurations:
//! - native (default): the `std` feature is on. Development and debugging happen here.
//! - wasm distribution: `--no-default-features` makes it `no_std` and never links
//!   `core::fmt` at all.

#![cfg_attr(not(feature = "std"), no_std)]
// alloc is always used, even under no_std.
extern crate alloc;

#[macro_use]
pub mod error;

pub mod rt;

pub mod vector;

pub mod json;

pub mod parquet;

pub mod catalog;

pub mod format;

pub mod sql;

pub mod plan;

pub mod expr;

pub mod exec;

pub mod session;

// Every mutating feature is opt-out capable. Disabled by default (DESIGN.md §16).
#[cfg(feature = "export")]
pub mod write;

#[cfg(feature = "ddl")]
pub mod ddl;
#[cfg(feature = "dml")]
pub mod dml;

#[cfg(target_arch = "wasm32")]
pub mod abi;

/// The imports shared across the crate.
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
