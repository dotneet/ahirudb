//! ミニ Arrow 相当の列指向ベクタ層。
//!
//! - 1 バッチ = 2048 行。オペレータ間はこの単位で受け渡す。
//! - `Vector` の長さはデータ本体から導出する（別途 `len` を持たない）。
//!   フィールドの二重管理による不整合バグを構造的に潰すため。
//! - フィルタは行をコピーせず `Batch::sel`（selection vector）を絞る。

pub mod bitmap;
pub mod types;
pub mod value;

pub use bitmap::Bitmap;
pub use types::{fmt_interval, pack_interval, unpack_interval, Field, PhysType, Ty};
pub use value::Value;

use crate::prelude::*;

/// 1 バッチあたりの行数。
pub const BATCH_SIZE: usize = 2048;

/// 可変長バイト列の格納。`offsets.len() == 行数 + 1`。
#[derive(Clone)]
pub struct BytesData {
    pub offsets: Vec<u32>,
    pub data: Vec<u8>,
}

impl BytesData {
    pub fn new() -> Self {
        BytesData { offsets: vec![0], data: Vec::new() }
    }

    pub fn with_capacity(rows: usize, bytes: usize) -> Self {
        let mut offsets = Vec::with_capacity(rows + 1);
        offsets.push(0);
        BytesData { offsets, data: Vec::with_capacity(bytes) }
    }

    #[inline]
    pub fn len(&self) -> usize {
        // `saturating_sub` は空 offsets（`BytesData::EMPTY`）を安全に 0 行として扱うため。
        self.offsets.len().saturating_sub(1)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn get(&self, i: usize) -> &[u8] {
        let s = self.offsets[i] as usize;
        let e = self.offsets[i + 1] as usize;
        &self.data[s..e]
    }

    #[inline]
    pub fn push(&mut self, v: &[u8]) {
        self.data.extend_from_slice(v);
        self.offsets.push(self.data.len() as u32);
    }

    /// 空バイト列を追加する。NULL 行のプレースホルダに使う。
    #[inline]
    pub fn push_empty(&mut self) {
        self.offsets.push(self.data.len() as u32);
    }

    pub fn clear(&mut self) {
        self.offsets.truncate(1);
        self.data.clear();
    }
}

impl Default for BytesData {
    fn default() -> Self {
        BytesData::new()
    }
}

/// 物理型ごとのデータ本体。
#[derive(Clone)]
pub enum Data {
    Bool(Bitmap),
    I32(Vec<i32>),
    I64(Vec<i64>),
    F64(Vec<f64>),
    I128(Vec<i128>),
    Bytes(BytesData),
}

impl Data {
    pub fn empty(p: PhysType) -> Self {
        match p {
            PhysType::Bool => Data::Bool(Bitmap::new()),
            PhysType::I32 => Data::I32(Vec::new()),
            PhysType::I64 => Data::I64(Vec::new()),
            PhysType::F64 => Data::F64(Vec::new()),
            PhysType::I128 => Data::I128(Vec::new()),
            PhysType::Bytes => Data::Bytes(BytesData::new()),
        }
    }

    pub fn with_capacity(p: PhysType, cap: usize) -> Self {
        match p {
            PhysType::Bool => Data::Bool(Bitmap::with_capacity(cap)),
            PhysType::I32 => Data::I32(Vec::with_capacity(cap)),
            PhysType::I64 => Data::I64(Vec::with_capacity(cap)),
            PhysType::F64 => Data::F64(Vec::with_capacity(cap)),
            PhysType::I128 => Data::I128(Vec::with_capacity(cap)),
            PhysType::Bytes => Data::Bytes(BytesData::with_capacity(cap, cap * 8)),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Data::Bool(b) => b.len(),
            Data::I32(v) => v.len(),
            Data::I64(v) => v.len(),
            Data::F64(v) => v.len(),
            Data::I128(v) => v.len(),
            Data::Bytes(b) => b.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn phys(&self) -> PhysType {
        match self {
            Data::Bool(_) => PhysType::Bool,
            Data::I32(_) => PhysType::I32,
            Data::I64(_) => PhysType::I64,
            Data::F64(_) => PhysType::F64,
            Data::I128(_) => PhysType::I128,
            Data::Bytes(_) => PhysType::Bytes,
        }
    }

    pub fn clear(&mut self) {
        match self {
            Data::Bool(b) => *b = Bitmap::new(),
            Data::I32(v) => v.clear(),
            Data::I64(v) => v.clear(),
            Data::F64(v) => v.clear(),
            Data::I128(v) => v.clear(),
            Data::Bytes(b) => b.clear(),
        }
    }
}

/// 1 列分のベクタ。
#[derive(Clone)]
pub struct Vector {
    ty: Ty,
    /// `None` は「全行 valid」を意味する。NULL の無い列でビットマップを持たない。
    validity: Option<Bitmap>,
    data: Data,
}

impl Vector {
    pub fn new(ty: Ty) -> Self {
        Vector { ty, validity: None, data: Data::empty(ty.phys()) }
    }

    pub fn with_capacity(ty: Ty, cap: usize) -> Self {
        Vector { ty, validity: None, data: Data::with_capacity(ty.phys(), cap) }
    }

    pub fn from_data(ty: Ty, data: Data, validity: Option<Bitmap>) -> Self {
        debug_assert_eq!(ty.phys(), data.phys());
        debug_assert!(validity.as_ref().is_none_or(|v| v.len() == data.len()));
        Vector { ty, validity, data }
    }

    #[inline]
    pub fn ty(&self) -> Ty {
        self.ty
    }

    /// 論理型だけを差し替える。物理型が同じ場合のみ許される（例: INT → DATE）。
    pub fn retype(&mut self, ty: Ty) {
        debug_assert_eq!(ty.phys(), self.data.phys());
        self.ty = ty;
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn data(&self) -> &Data {
        &self.data
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut Data {
        &mut self.data
    }

    #[inline]
    pub fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    #[inline]
    pub fn has_nulls(&self) -> bool {
        self.validity.is_some()
    }

    #[inline]
    pub fn is_valid(&self, i: usize) -> bool {
        match &self.validity {
            None => true,
            Some(b) => b.get(i),
        }
    }

    pub fn set_validity(&mut self, v: Option<Bitmap>) {
        debug_assert!(v.as_ref().is_none_or(|b| b.len() == self.data.len()));
        self.validity = v;
    }

    /// validity ビットマップを実体化して返す。無ければ全 1 で作る。
    pub fn validity_mut(&mut self) -> &mut Bitmap {
        if self.validity.is_none() {
            self.validity = Some(Bitmap::ones(self.data.len()));
        }
        let b = self.validity.as_mut().unwrap();
        if b.len() < self.data.len() {
            b.resize(self.data.len(), true);
        }
        b
    }

    /// NULL が 1 つも無ければ validity を捨てる。以降の演算が速くなる。
    pub fn compact_validity(&mut self) {
        if let Some(b) = &self.validity {
            if b.all_set() {
                self.validity = None;
            }
        }
    }

    // --- 型別アクセサ -------------------------------------------------------
    // 物理型の不一致は呼び出し側のバグ。no_std でパニックさせないため、
    // デバッグビルドで検出しリリースでは空を返す。

    #[inline]
    pub fn i32s(&self) -> &[i32] {
        match &self.data {
            Data::I32(v) => v,
            _ => {
                debug_assert!(false, "expected I32");
                &[]
            }
        }
    }

    #[inline]
    pub fn i64s(&self) -> &[i64] {
        match &self.data {
            Data::I64(v) => v,
            _ => {
                debug_assert!(false, "expected I64");
                &[]
            }
        }
    }

    #[inline]
    pub fn f64s(&self) -> &[f64] {
        match &self.data {
            Data::F64(v) => v,
            _ => {
                debug_assert!(false, "expected F64");
                &[]
            }
        }
    }

    #[inline]
    pub fn i128s(&self) -> &[i128] {
        match &self.data {
            Data::I128(v) => v,
            _ => {
                debug_assert!(false, "expected I128");
                &[]
            }
        }
    }

    #[inline]
    pub fn bools(&self) -> &Bitmap {
        match &self.data {
            Data::Bool(b) => b,
            _ => {
                debug_assert!(false, "expected Bool");
                Bitmap::empty_ref()
            }
        }
    }

    #[inline]
    pub fn bytes(&self) -> &BytesData {
        match &self.data {
            Data::Bytes(b) => b,
            _ => {
                debug_assert!(false, "expected Bytes");
                BytesData::empty_ref()
            }
        }
    }

    /// i 番目の値をスカラとして取り出す。結果出力と定数畳み込み用。
    /// ホットループでは使わない。
    pub fn value_at(&self, i: usize) -> Value {
        if !self.is_valid(i) {
            return Value::Null;
        }
        match &self.data {
            Data::Bool(b) => Value::Bool(b.get(i)),
            Data::I32(v) => Value::I32(v[i]),
            Data::I64(v) => Value::I64(v[i]),
            Data::F64(v) => Value::F64(v[i]),
            Data::I128(v) => Value::I128(v[i]),
            Data::Bytes(b) => Value::Bytes(b.get(i).to_vec()),
        }
    }

    /// 末尾に NULL を 1 行追加する。物理データにはダミー値を入れる。
    pub fn push_null(&mut self) {
        match &mut self.data {
            Data::Bool(b) => b.push(false),
            Data::I32(v) => v.push(0),
            Data::I64(v) => v.push(0),
            Data::F64(v) => v.push(0.0),
            Data::I128(v) => v.push(0),
            Data::Bytes(b) => b.push_empty(),
        }
        let n = self.data.len();
        let bm = match &mut self.validity {
            Some(b) => b,
            None => {
                self.validity = Some(Bitmap::ones(n - 1));
                self.validity.as_mut().unwrap()
            }
        };
        bm.resize(n - 1, true);
        bm.push(false);
    }

    pub fn push_value(&mut self, v: &Value) {
        match v {
            Value::Null => {
                self.push_null();
                return;
            }
            Value::Bool(x) => {
                if let Data::Bool(b) = &mut self.data {
                    b.push(*x)
                }
            }
            Value::I32(x) => {
                if let Data::I32(d) = &mut self.data {
                    d.push(*x)
                }
            }
            Value::I64(x) => {
                if let Data::I64(d) = &mut self.data {
                    d.push(*x)
                }
            }
            Value::F64(x) => {
                if let Data::F64(d) = &mut self.data {
                    d.push(*x)
                }
            }
            Value::I128(x) => {
                if let Data::I128(d) = &mut self.data {
                    d.push(*x)
                }
            }
            Value::Bytes(x) => {
                if let Data::Bytes(d) = &mut self.data {
                    d.push(x)
                }
            }
        }
        if let Some(b) = &mut self.validity {
            b.resize(self.data.len(), true);
        }
    }

    /// selection vector に従って行を抽出した新しいベクタを作る。
    /// 結果を materialize する必要が生じたときにだけ呼ぶ。
    pub fn gather(&self, sel: &[u32]) -> Vector {
        let mut out = Vector::with_capacity(self.ty, sel.len());
        match &self.data {
            Data::Bool(b) => {
                let mut nb = Bitmap::with_capacity(sel.len());
                for &i in sel {
                    nb.push(b.get(i as usize));
                }
                out.data = Data::Bool(nb);
            }
            Data::I32(v) => out.data = Data::I32(sel.iter().map(|&i| v[i as usize]).collect()),
            Data::I64(v) => out.data = Data::I64(sel.iter().map(|&i| v[i as usize]).collect()),
            Data::F64(v) => out.data = Data::F64(sel.iter().map(|&i| v[i as usize]).collect()),
            Data::I128(v) => out.data = Data::I128(sel.iter().map(|&i| v[i as usize]).collect()),
            Data::Bytes(b) => {
                let mut nb = BytesData::with_capacity(sel.len(), b.data.len());
                for &i in sel {
                    nb.push(b.get(i as usize));
                }
                out.data = Data::Bytes(nb);
            }
        }
        if let Some(val) = &self.validity {
            let mut nv = Bitmap::with_capacity(sel.len());
            for &i in sel {
                nv.push(val.get(i as usize));
            }
            out.validity = Some(nv);
            out.compact_validity();
        }
        out
    }
}

// `bytes()` の型不一致時に返す空データ。`Vec::new()` は const fn。
static EMPTY_BYTES: BytesData = BytesData { offsets: Vec::new(), data: Vec::new() };

impl BytesData {
    pub fn empty_ref() -> &'static BytesData {
        &EMPTY_BYTES
    }
}

/// オペレータ間で受け渡す 1 バッチ。
pub struct Batch {
    pub cols: Vec<Vector>,
    /// 有効行のインデックス。`None` は「全行有効」。
    pub sel: Option<Vec<u32>>,
    /// 列が 0 個のとき（`COUNT(*)` など）に行数を保持する。
    empty_rows: usize,
}

impl Batch {
    pub fn new(cols: Vec<Vector>) -> Self {
        Batch { cols, sel: None, empty_rows: 0 }
    }

    /// 列を持たず行数だけを表すバッチ。
    pub fn rows_only(n: usize) -> Self {
        Batch { cols: Vec::new(), sel: None, empty_rows: n }
    }

    /// 物理行数（selection 適用前）。
    pub fn num_rows(&self) -> usize {
        match self.cols.first() {
            Some(c) => c.len(),
            None => self.empty_rows,
        }
    }

    /// 有効行数（selection 適用後）。
    pub fn card(&self) -> usize {
        match &self.sel {
            Some(s) => s.len(),
            None => self.num_rows(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.card() == 0
    }

    /// selection を実体化して全列を絞り込む。
    /// ハッシュ表への投入など、ランダムアクセスが増える手前で呼ぶ。
    pub fn materialize(&mut self) {
        if let Some(sel) = self.sel.take() {
            if sel.len() == self.num_rows() {
                // すべて選択されているならコピー不要。
                let full = sel.iter().enumerate().all(|(i, &v)| i as u32 == v);
                if full {
                    return;
                }
            }
            for c in self.cols.iter_mut() {
                *c = c.gather(&sel);
            }
            self.empty_rows = sel.len();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn len_is_derived_from_data() {
        let mut v = Vector::new(Ty::Int);
        if let Data::I32(d) = v.data_mut() {
            d.extend_from_slice(&[1, 2, 3]);
        }
        assert_eq!(v.len(), 3);
        assert_eq!(v.i32s(), &[1, 2, 3]);
    }

    #[test]
    fn push_null_materializes_validity() {
        let mut v = Vector::new(Ty::BigInt);
        v.push_value(&Value::I64(10));
        v.push_null();
        v.push_value(&Value::I64(30));
        assert_eq!(v.len(), 3);
        assert!(v.is_valid(0));
        assert!(!v.is_valid(1));
        assert!(v.is_valid(2));
        assert_eq!(v.value_at(1), Value::Null);
        assert_eq!(v.value_at(2), Value::I64(30));
    }

    #[test]
    fn bytes_roundtrip() {
        let mut v = Vector::new(Ty::Varchar);
        v.push_value(&Value::Bytes(b"hello".to_vec()));
        v.push_null();
        v.push_value(&Value::Bytes(b"world!".to_vec()));
        assert_eq!(v.len(), 3);
        assert_eq!(v.bytes().get(0), b"hello");
        assert_eq!(v.bytes().get(2), b"world!");
        assert!(!v.is_valid(1));
    }

    #[test]
    fn gather_respects_validity() {
        let mut v = Vector::new(Ty::Int);
        for i in 0..10 {
            if i % 3 == 0 {
                v.push_null();
            } else {
                v.push_value(&Value::I32(i));
            }
        }
        let g = v.gather(&[1, 3, 4]);
        assert_eq!(g.len(), 3);
        assert_eq!(g.value_at(0), Value::I32(1));
        assert_eq!(g.value_at(1), Value::Null);
        assert_eq!(g.value_at(2), Value::I32(4));
    }

    #[test]
    fn compact_validity_drops_all_valid_bitmap() {
        let mut v = Vector::new(Ty::Int);
        v.push_value(&Value::I32(1));
        v.validity_mut();
        assert!(v.has_nulls());
        v.compact_validity();
        assert!(!v.has_nulls());
    }

    #[test]
    fn batch_cardinality_follows_selection() {
        let mut v = Vector::new(Ty::Int);
        for i in 0..10 {
            v.push_value(&Value::I32(i));
        }
        let mut b = Batch::new(vec![v]);
        assert_eq!(b.card(), 10);
        b.sel = Some(vec![2, 4, 6]);
        assert_eq!(b.card(), 3);
        assert_eq!(b.num_rows(), 10);
        b.materialize();
        assert_eq!(b.num_rows(), 3);
        assert_eq!(b.cols[0].value_at(0), Value::I32(2));
    }
}
