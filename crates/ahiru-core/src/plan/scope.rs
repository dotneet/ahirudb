//! The name-resolution scope.
//!
//! With joins, "the same column name exists in several tables" is routine, so columns are
//! held as `(qualifier, field)` pairs. The qualifier is a table name or alias.
//!
//! The index doubles as the column number within a batch. A join's output is "the left
//! schema ++ the right schema", so concatenating scopes in the same order suffices.

use crate::prelude::*;
use crate::rt::hash::eq_ascii_ci;
use crate::vector::Field;

#[derive(Default, Clone)]
pub struct Scope {
    quals: Vec<Option<String>>,
    fields: Vec<Field>,
}

impl Scope {
    pub fn new() -> Self {
        Scope { quals: Vec::new(), fields: Vec::new() }
    }

    /// A scope with no qualifier. Used for the intermediate results of aggregation and sorting.
    pub fn from_fields(fields: Vec<Field>) -> Self {
        Scope { quals: vec![None; fields.len()], fields }
    }

    pub fn push(&mut self, qual: Option<String>, field: Field) {
        self.quals.push(qual);
        self.fields.push(field);
    }

    /// Concatenates another scope on the right. Used to build a join's output schema.
    pub fn extend(&mut self, other: &Scope) {
        self.quals.extend_from_slice(&other.quals);
        self.fields.extend_from_slice(&other.fields);
    }

    pub fn fields(&self) -> &[Field] {
        &self.fields
    }

    pub fn qualifier(&self, i: usize) -> Option<&str> {
        self.quals.get(i).and_then(|q| q.as_deref())
    }

    pub fn len(&self) -> usize {
        self.fields.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Whether at least one column carries this qualifier. Used to validate `t.*`.
    pub fn has_qualifier(&self, qual: &str) -> bool {
        self.quals
            .iter()
            .any(|q| q.as_deref().is_some_and(|q| eq_ascii_ci(q.as_bytes(), qual.as_bytes())))
    }

    /// Enumerates only the qualified columns. Used to expand `t.*`.
    pub fn indices_for_qualifier(&self, qual: &str) -> Vec<usize> {
        self.quals
            .iter()
            .enumerate()
            .filter(|(_, q)| {
                q.as_deref().is_some_and(|q| eq_ascii_ci(q.as_bytes(), qual.as_bytes()))
            })
            .map(|(i, _)| i)
            .collect()
    }

    /// Resolves a column.
    ///
    /// With a qualifier, only columns carrying it are considered. Without one, the whole
    /// scope is searched, and two or more matches give `AmbiguousColumn`. Name comparison is
    /// case-insensitive (identifier spelling is preserved as the output name, but not for matching).
    pub fn resolve(&self, qual: Option<&str>, name: &str) -> Result<usize> {
        let mut found = None;
        for i in 0..self.fields.len() {
            if let Some(q) = qual {
                match self.quals[i].as_deref() {
                    Some(cq) if eq_ascii_ci(cq.as_bytes(), q.as_bytes()) => {}
                    _ => continue,
                }
            }
            if !eq_ascii_ci(self.fields[i].name.as_bytes(), name.as_bytes()) {
                continue;
            }
            if found.is_some() {
                err!(AmbiguousColumn);
            }
            found = Some(i);
        }
        match found {
            Some(i) => Ok(i),
            None => {
                // Flattened STRUCT leaves are stored as a single field named
                // `qual.name` (e.g. `address.city`). `address.city` parses as
                // qualifier `address` + name `city`, which misses both the
                // table-column lookup above and an unqualified lookup of the
                // dotted field. Fall back to that dotted name before deciding
                // the qualifier is a missing table.
                if let Some(q) = qual {
                    let mut dotted = String::with_capacity(q.len() + 1 + name.len());
                    dotted.push_str(q);
                    dotted.push('.');
                    dotted.push_str(name);
                    let mut flat = None;
                    for i in 0..self.fields.len() {
                        if !eq_ascii_ci(self.fields[i].name.as_bytes(), dotted.as_bytes()) {
                            continue;
                        }
                        if flat.is_some() {
                            err!(AmbiguousColumn);
                        }
                        flat = Some(i);
                    }
                    if let Some(i) = flat {
                        return Ok(i);
                    }
                    if !self.has_qualifier(q) {
                        err!(TableNotFound);
                    }
                }
                err!(ColumnNotFound)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::code_of;
    use crate::vector::Ty;

    fn scope() -> Scope {
        let mut s = Scope::new();
        s.push(Some("a".into()), Field::new("id", Ty::Int, false));
        s.push(Some("a".into()), Field::new("name", Ty::Varchar, true));
        s.push(Some("b".into()), Field::new("id", Ty::BigInt, false));
        s.push(Some("b".into()), Field::new("score", Ty::Double, true));
        s
    }

    #[test]
    fn qualified_resolution_picks_the_right_side() {
        let s = scope();
        assert_eq!(s.resolve(Some("a"), "id").unwrap(), 0);
        assert_eq!(s.resolve(Some("b"), "id").unwrap(), 2);
        // Case-insensitive.
        assert_eq!(s.resolve(Some("B"), "ID").unwrap(), 2);
    }

    #[test]
    fn unqualified_duplicate_is_ambiguous() {
        let s = scope();
        assert_eq!(code_of(s.resolve(None, "id")), Some(Code::AmbiguousColumn));
        // A name present on only one side is not ambiguous.
        assert_eq!(s.resolve(None, "score").unwrap(), 3);
    }

    #[test]
    fn unknown_qualifier_reports_missing_table() {
        let s = scope();
        assert_eq!(code_of(s.resolve(Some("z"), "id")), Some(Code::TableNotFound));
        // With a qualifier present but no such column, the column is the problem.
        assert_eq!(code_of(s.resolve(Some("a"), "score")), Some(Code::ColumnNotFound));
    }

    #[test]
    fn dotted_struct_field_resolves_without_a_qualifier_table() {
        let mut s = Scope::new();
        s.push(Some("t".into()), Field::new("id", Ty::Int, false));
        s.push(Some("t".into()), Field::new("address.city", Ty::Varchar, true));
        // `address.city` parsed as qualifier + name.
        assert_eq!(s.resolve(Some("address"), "city").unwrap(), 1);
        // Quoted `"address.city"` is a single identifier.
        assert_eq!(s.resolve(None, "address.city").unwrap(), 1);
        // `t.address.city` after the parser joins the tail into the name.
        assert_eq!(s.resolve(Some("t"), "address.city").unwrap(), 1);
        // A qualifier that is neither a table nor a struct prefix stays missing.
        assert_eq!(code_of(s.resolve(Some("z"), "city")), Some(Code::TableNotFound));
    }

    #[test]
    fn star_expansion_by_qualifier() {
        let s = scope();
        assert_eq!(s.indices_for_qualifier("a"), vec![0, 1]);
        assert_eq!(s.indices_for_qualifier("b"), vec![2, 3]);
        assert!(s.indices_for_qualifier("z").is_empty());
    }

    #[test]
    fn extend_concatenates_for_joins() {
        let mut l = Scope::new();
        l.push(Some("l".into()), Field::new("x", Ty::Int, false));
        let mut r = Scope::new();
        r.push(Some("r".into()), Field::new("y", Ty::Int, false));
        l.extend(&r);
        assert_eq!(l.len(), 2);
        assert_eq!(l.resolve(Some("r"), "y").unwrap(), 1);
    }
}
