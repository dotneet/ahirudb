//! Rendering the plan tree as text. For `EXPLAIN`.
//!
//! `format!` is unavailable given the size budget, so strings are assembled by hand,
//! including number formatting (`push_u64`).

use crate::plan::Node;
use crate::prelude::*;

/// Renders a plan as text, one node per line. Depth is shown by indentation.
pub fn explain(root: &Node) -> Vec<String> {
    let mut out = Vec::new();
    walk(root, 0, &mut out);
    out
}

fn walk(n: &Node, depth: usize, out: &mut Vec<String>) {
    let mut line = String::new();
    for _ in 0..depth {
        line.push_str("  ");
    }
    match n {
        Node::Scan(s) => {
            line.push_str("Scan  columns=");
            push_u64(&mut line, s.columns.len() as u64);
            if !s.pruners.is_empty() {
                line.push_str(" pruners=");
                push_u64(&mut line, s.pruners.len() as u64);
            }
            out.push(line);
        }
        #[cfg(feature = "ddl")]
        Node::MemScan(s) => {
            line.push_str("MemScan  columns=");
            push_u64(&mut line, s.schema.len() as u64);
            out.push(line);
        }
        Node::Filter { input, .. } => {
            line.push_str("Filter");
            out.push(line);
            walk(input, depth + 1, out);
        }
        Node::Project { input, exprs, .. } => {
            line.push_str("Project  exprs=");
            push_u64(&mut line, exprs.len() as u64);
            out.push(line);
            walk(input, depth + 1, out);
        }
        Node::Aggregate { input, groups, aggs, having, .. } => {
            line.push_str("Aggregate  groups=");
            push_u64(&mut line, groups.len() as u64);
            line.push_str(" aggs=");
            push_u64(&mut line, aggs.len() as u64);
            if having.is_some() {
                line.push_str(" having");
            }
            out.push(line);
            walk(input, depth + 1, out);
        }
        Node::Sort { input, keys, limit } => {
            line.push_str("Sort  keys=");
            push_u64(&mut line, keys.len() as u64);
            if let Some(l) = limit {
                // Whether it lowered to a Top-N matters for reading performance.
                line.push_str(" topn=");
                push_u64(&mut line, *l as u64);
            }
            out.push(line);
            walk(input, depth + 1, out);
        }
        Node::Join { left, right, kind, left_keys, residual, .. } => {
            line.push_str("Join  ");
            line.push_str(match kind {
                crate::sql::ast::JoinKind::Inner => "inner",
                crate::sql::ast::JoinKind::Left => "left",
                crate::sql::ast::JoinKind::Right => "right",
                crate::sql::ast::JoinKind::Full => "full",
                crate::sql::ast::JoinKind::Cross => "cross",
                crate::sql::ast::JoinKind::Semi => "semi",
                crate::sql::ast::JoinKind::Anti => "anti",
                crate::sql::ast::JoinKind::AntiNullAware => "anti (null aware)",
            });
            if left_keys.is_empty() {
                line.push_str(" (nested loop)");
            } else {
                line.push_str(" keys=");
                push_u64(&mut line, left_keys.len() as u64);
            }
            if residual.is_some() {
                line.push_str(" residual");
            }
            out.push(line);
            walk(left, depth + 1, out);
            walk(right, depth + 1, out);
        }
        Node::Window { input, windows, .. } => {
            line.push_str("Window  funcs=");
            push_u64(&mut line, windows.len() as u64);
            out.push(line);
            walk(input, depth + 1, out);
        }
        Node::SetOp { left, right, op, all, .. } => {
            line.push_str(match op {
                crate::plan::SetOpKind::Union => "Union",
                crate::plan::SetOpKind::Intersect => "Intersect",
                crate::plan::SetOpKind::Except => "Except",
            });
            if *all {
                line.push_str(" all");
            }
            out.push(line);
            walk(left, depth + 1, out);
            walk(right, depth + 1, out);
        }
        Node::Limit { input, limit, offset } => {
            line.push_str("Limit  n=");
            match limit {
                Some(l) => push_u64(&mut line, *l),
                None => line.push_str("all"),
            }
            if *offset > 0 {
                line.push_str(" offset=");
                push_u64(&mut line, *offset);
            }
            out.push(line);
            walk(input, depth + 1, out);
        }
        Node::DistinctOn { input, keys } => {
            line.push_str("DistinctOn  keys=");
            push_u64(&mut line, keys.len() as u64);
            out.push(line);
            walk(input, depth + 1, out);
        }
        Node::RecursiveCte { anchor, recursive_term, union_all, .. } => {
            line.push_str("RecursiveCte");
            if *union_all {
                line.push_str(" all");
            }
            out.push(line);
            walk(anchor, depth + 1, out);
            walk(recursive_term, depth + 1, out);
        }
        Node::WorkingTable { .. } => {
            line.push_str("WorkingTable");
            out.push(line);
        }
        Node::Unnest { input, elem_ty, .. } => {
            line.push_str("Unnest  elem_ty=");
            line.push_str(elem_ty.name());
            out.push(line);
            walk(input, depth + 1, out);
        }
        Node::GenerateSeries { .. } => {
            line.push_str("GenerateSeries");
            out.push(line);
        }
        Node::Sample { input, spec } => {
            line.push_str(if spec.is_rows { "Sample  rows=" } else { "Sample  percent=" });
            push_u64(&mut line, spec.amount as u64);
            out.push(line);
            walk(input, depth + 1, out);
        }
        Node::AssertMaxOneRow { input, keys } => {
            line.push_str("AssertMaxOneRow  keys=");
            push_u64(&mut line, keys.len() as u64);
            out.push(line);
            walk(input, depth + 1, out);
        }
    }
}

fn push_u64(s: &mut String, mut v: u64) {
    let mut buf = [0u8; 20];
    let mut n = 0;
    loop {
        buf[n] = b'0' + (v % 10) as u8;
        n += 1;
        v /= 10;
        if v == 0 {
            break;
        }
    }
    for i in (0..n).rev() {
        s.push(buf[i] as char);
    }
}
