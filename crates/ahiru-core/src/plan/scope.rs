//! 名前解決のスコープ。
//!
//! 結合が入ると「同じ列名が複数のテーブルにある」状況が普通に起きるので、
//! 列は `(修飾子, フィールド)` の組で持つ。修飾子はテーブル名か別名。
//!
//! 添字はそのままバッチ内の列番号になる。結合の出力は「左のスキーマ ++
//! 右のスキーマ」なので、スコープも同じ順で連結すればよい。

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

    /// 修飾子を持たないスコープ。集約やソートの中間結果に使う。
    pub fn from_fields(fields: Vec<Field>) -> Self {
        Scope { quals: vec![None; fields.len()], fields }
    }

    pub fn push(&mut self, qual: Option<String>, field: Field) {
        self.quals.push(qual);
        self.fields.push(field);
    }

    /// 別のスコープを右側に連結する。結合の出力スキーマを作るのに使う。
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

    /// この修飾子を持つ列が 1 つでもあるか。`t.*` の検証に使う。
    pub fn has_qualifier(&self, qual: &str) -> bool {
        self.quals
            .iter()
            .any(|q| q.as_deref().is_some_and(|q| eq_ascii_ci(q.as_bytes(), qual.as_bytes())))
    }

    /// 修飾子付きの列だけを列挙する。`t.*` の展開に使う。
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

    /// 列を解決する。
    ///
    /// 修飾子があればその修飾子を持つ列だけを見る。無ければ全体から探し、
    /// 2 つ以上一致したら `AmbiguousColumn`。名前の比較は大文字小文字を
    /// 区別しない（識別子の綴りは出力名として保つが、照合はしない）。
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
                // 修飾子が付いていて、その修飾子自体が存在しないなら
                // 「テーブルが無い」と言う方が原因に近い。
                if let Some(q) = qual {
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
        // 大文字小文字は区別しない。
        assert_eq!(s.resolve(Some("B"), "ID").unwrap(), 2);
    }

    #[test]
    fn unqualified_duplicate_is_ambiguous() {
        let s = scope();
        assert_eq!(code_of(s.resolve(None, "id")), Some(Code::AmbiguousColumn));
        // 片方にしか無い名前なら曖昧ではない。
        assert_eq!(s.resolve(None, "score").unwrap(), 3);
    }

    #[test]
    fn unknown_qualifier_reports_missing_table() {
        let s = scope();
        assert_eq!(code_of(s.resolve(Some("z"), "id")), Some(Code::TableNotFound));
        // 修飾子はあるが列が無い場合は列の問題。
        assert_eq!(code_of(s.resolve(Some("a"), "score")), Some(Code::ColumnNotFound));
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
