//! ネイティブ (std) でのみ走るテスト。
//!
//! フィクスチャは `zstd` CLI (v1.5.7) で作った実際のフレームを
//! `tests/data/*.zst` に置き、平文は同じ生成規則を Rust で書き直して再現する
//! （平文をリポジトリに置くと無駄に太るため）。生成規則が食い違えば、そのまま
//! 比較テストが落ちるので、ズレは検出できる。
//!
//! 各フィクスチャは下の生成関数の出力を
//! `zstd -f -q --single-thread --no-check -<level> <平文> -o <名前>.zst`
//! で圧縮したもの（`text_crc` だけ `--check` 付き）。名前の末尾が圧縮レベル。

use super::*;
use std::time::Instant;

// --- 平文の生成規則 (tests/data の .zst と対で維持する) ---------------------

fn lcg(n: usize) -> Vec<u8> {
    let mut s: u32 = 0x1234_5678;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
        v.push((s >> 16) as u8);
    }
    v
}

const WORDS: [&str; 8] =
    ["ahiru", "duck", "parquet", "zstandard", "column", "vector", "wasm", "query"];

fn prose(n: usize) -> Vec<u8> {
    let mut s = String::with_capacity(n + 16);
    let mut st: u32 = 0x2468_ACE1;
    while s.len() < n {
        st = st.wrapping_mul(1_103_515_245).wrapping_add(12345);
        s.push_str(WORDS[((st >> 16) % 8) as usize]);
        s.push(if (st >> 24).is_multiple_of(13) { '\n' } else { ' ' });
    }
    s.truncate(n);
    s.into_bytes()
}

fn lines(n: usize) -> Vec<u8> {
    let mut s = String::with_capacity(n + 32);
    let mut i = 0usize;
    while s.len() < n {
        s.push_str(&format!("record {i:08} ahirudb\n"));
        i += 1;
    }
    s.truncate(n);
    s.into_bytes()
}

/// 140000 バイトの一意な行 + その先頭 60000 バイトの複製。
/// 複製部分は 128 KiB のブロック境界をまたいだ後方参照でしか再現できない。
fn xblock() -> Vec<u8> {
    let base = lines(140_000);
    let mut v = base.clone();
    v.extend_from_slice(&base[..60_000]);
    v
}

/// 小さい記号値 (0..5) の偏った分布。maxSymbolValue が小さいため zstd は
/// ハフマン weight を FSE 圧縮せず 4 ビット直接形式で書く。
fn low_alphabet(n: usize) -> Vec<u8> {
    let mut s: u32 = 0x1357_2468;
    let mut v = Vec::with_capacity(n);
    while v.len() < n {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
        v.push(((s >> 16) % 11).min(5) as u8);
    }
    v
}

/// 長大な同一バイト列。offset 1 の重なりマッチになる。
fn runs(n: usize) -> Vec<u8> {
    let mut v = lcg(64);
    v.resize(v.len() + n, b'q');
    v.extend_from_slice(&lcg(64));
    v
}

// --- フィクスチャ -----------------------------------------------------------

const EMPTY_3: &[u8] = include_bytes!("../tests/data/empty_3.zst");
const HELLO_3: &[u8] = include_bytes!("../tests/data/hello_3.zst");
const RLE_3: &[u8] = include_bytes!("../tests/data/rle_3.zst");
const RLELIT_3: &[u8] = include_bytes!("../tests/data/rlelit_3.zst");
const RAND_3: &[u8] = include_bytes!("../tests/data/rand_3.zst");
const TEXT_1: &[u8] = include_bytes!("../tests/data/text_1.zst");
const TEXT_3: &[u8] = include_bytes!("../tests/data/text_3.zst");
const TEXT_19: &[u8] = include_bytes!("../tests/data/text_19.zst");
const TEXT_CRC: &[u8] = include_bytes!("../tests/data/text_crc.zst");
const BIG_1: &[u8] = include_bytes!("../tests/data/big_1.zst");
const BIG_3: &[u8] = include_bytes!("../tests/data/big_3.zst");
const BIG_19: &[u8] = include_bytes!("../tests/data/big_19.zst");
const XBLOCK_3: &[u8] = include_bytes!("../tests/data/xblock_3.zst");
const RUNS_3: &[u8] = include_bytes!("../tests/data/runs_3.zst");
const LOW_3: &[u8] = include_bytes!("../tests/data/low_3.zst");

/// シーケンステーブルの RLE モードを手で組んだフレーム。実際の zstd が
/// このモードを選ぶ入力は稀なので、仕様どおりに組んで直接確かめる。
/// 参照実装 (`zstd -d`) がこのバイト列を "aaaabbbbcccc" に展開することは確認済み。
#[rustfmt::skip]
const RLE_SEQ_FRAME: &[u8] = &[
    0x28, 0xb5, 0x2f, 0xfd, // マジック
    0x20,                   // Single_Segment, FCS 1 バイト
    12,                     // Frame_Content_Size
    0x55, 0x00, 0x00,       // last=1, Compressed, size=10
    0x18, 0x61, 0x62, 0x63, // Raw リテラル 3 バイト "abc"
    0x03,                   // Nb_Sequences = 3
    0x54,                   // LL=RLE, OF=RLE, ML=RLE
    0x01, 0x00, 0x00,       // LL sym 1 (ll=1), OF sym 0 (rep1), ML sym 0 (ml=3)
    0x01,                   // ビットストリーム (全表が 0 ビットなので中身は空)
];

#[test]
fn rle_sequence_tables() {
    assert_eq!(decompress(RLE_SEQ_FRAME, 12).unwrap(), b"aaaabbbbcccc");
}

fn check(comp: &[u8], plain: &[u8]) {
    let got = decompress(comp, plain.len()).expect("decompress");
    assert_eq!(got.len(), plain.len(), "length mismatch");
    if got != plain {
        let at = got.iter().zip(plain).position(|(a, b)| a != b).unwrap_or(0);
        panic!("content mismatch at byte {at}");
    }
}

// --- 往復 -------------------------------------------------------------------

#[test]
fn roundtrip_empty() {
    check(EMPTY_3, b"");
}

#[test]
fn roundtrip_short() {
    check(HELLO_3, b"hello, ahiru");
}

#[test]
fn roundtrip_repetitive() {
    check(RLE_3, &vec![b'a'; 100_000]);
    check(RLELIT_3, &vec![b'a'; 200_000]);
}

#[test]
fn roundtrip_incompressible() {
    check(RAND_3, &lcg(32_768));
}

/// 記号数の少ない入力。ハフマン weight が 4 ビット直接形式になる。
#[test]
fn roundtrip_low_alphabet() {
    check(LOW_3, &low_alphabet(60_000));
}

#[test]
fn roundtrip_text_levels() {
    let plain = prose(40_000);
    check(TEXT_1, &plain);
    check(TEXT_3, &plain);
    check(TEXT_19, &plain);
}

#[test]
fn roundtrip_multi_block() {
    let plain = prose(300_000);
    check(BIG_1, &plain);
    check(BIG_3, &plain);
    check(BIG_19, &plain);
}

/// ブロック境界をまたいだ後方参照。
#[test]
fn roundtrip_cross_block_match() {
    let plain = xblock();
    assert!(plain.len() > 128 * 1024, "multi-block になっていない");
    check(XBLOCK_3, &plain);
}

/// offset 1 の長大マッチ（複製元と複製先が完全に重なる）。
#[test]
fn roundtrip_overlapping_match() {
    check(RUNS_3, &runs(70_000));
}

#[test]
fn content_checksum_is_verified() {
    let plain = prose(40_000);
    check(TEXT_CRC, &plain);
    assert!(stats::take() & (1 << stats::CHECKSUM) != 0, "チェックサムを通っていない");

    // 末尾 4 バイトのチェックサムを 1 ビット壊す。
    let mut bad = TEXT_CRC.to_vec();
    let n = bad.len();
    bad[n - 1] ^= 0x80;
    assert_eq!(decompress(&bad, plain.len()), Err(Error::ChecksumMismatch));
}

/// `decompress_into` は既存の内容の後ろに追記する。
#[test]
fn decompress_into_appends() {
    let mut out = b"prefix:".to_vec();
    decompress_into(HELLO_3, &mut out, 64).unwrap();
    assert_eq!(out, b"prefix:hello, ahiru");
}

/// wasm ABI はホスト所有のバッファを `Vec::from_raw_parts(ptr, 0, cap)` で包む。
/// 再確保が起きるとホスト側のポインタが無効になるので、`max_len` が容量以下で
/// ある限りバッファが動かないことを保証する（ABI の安全性の前提）。
#[test]
fn never_reallocates_within_limit() {
    let cases: [(&[u8], usize); 5] =
        [(HELLO_3, 12), (RLE_3, 100_000), (RUNS_3, 70_128), (XBLOCK_3, 200_000), (LOW_3, 60_000)];
    for (comp, n) in cases {
        // 正常系と、途中で切れた入力（エラー経路）の両方。
        for src in [comp, &comp[..comp.len() / 2], &comp[..1]] {
            let mut out: Vec<u8> = Vec::with_capacity(n);
            let (ptr, cap) = (out.as_ptr(), out.capacity());
            let _ = decompress_into(src, &mut out, n);
            assert_eq!(out.as_ptr(), ptr, "バッファが動いた");
            assert_eq!(out.capacity(), cap, "再確保が起きた");
            assert!(out.len() <= n);
        }
    }
}

// --- フォーマット分岐の網羅 -------------------------------------------------

/// フィクスチャ群が仕様上の分岐を実際に踏んでいることを確かめる。
/// 「テストは通るが分岐は未実行」を防ぐための担保。
#[test]
fn covers_format_variants() {
    let mut seen = 0u32;
    let _ = stats::take();
    for (c, n) in [
        (EMPTY_3, 0usize),
        (HELLO_3, 12),
        (RLE_3, 100_000),
        (RLELIT_3, 200_000),
        (RAND_3, 32_768),
        (TEXT_1, 40_000),
        (TEXT_3, 40_000),
        (TEXT_19, 40_000),
        (BIG_1, 300_000),
        (BIG_3, 300_000),
        (BIG_19, 300_000),
        (XBLOCK_3, 200_000),
        (RUNS_3, 70_128),
        (LOW_3, 60_000),
        (RLE_SEQ_FRAME, 12),
    ] {
        decompress(c, n).unwrap();
        seen |= stats::take();
    }
    // スキッパブルフレームは別途合成する。
    decompress(&with_skippable(HELLO_3), 12).unwrap();
    seen |= stats::take();

    let want: [(u32, &str); 17] = [
        (stats::BLOCK_RAW, "Raw ブロック"),
        (stats::BLOCK_RLE, "RLE ブロック"),
        (stats::BLOCK_COMPRESSED, "Compressed ブロック"),
        (stats::LIT_RAW, "Raw リテラル"),
        (stats::LIT_RLE, "RLE リテラル"),
        (stats::LIT_COMPRESSED, "Compressed リテラル"),
        (stats::LIT_TREELESS, "Treeless リテラル"),
        (stats::HUF_W_FSE, "FSE 圧縮された weight"),
        (stats::HUF_W_DIRECT, "4 ビット直接 weight"),
        (stats::HUF_1STREAM, "1 ストリーム"),
        (stats::HUF_4STREAM, "4 ストリーム"),
        (stats::SEQ_PREDEFINED, "Predefined モード"),
        (stats::SEQ_RLE, "RLE モード"),
        (stats::SEQ_FSE, "FSE_Compressed モード"),
        (stats::SEQ_REPEAT, "Repeat モード"),
        (stats::REPEAT_OFFSET, "繰り返しオフセット"),
        (stats::SKIPPABLE, "スキッパブルフレーム"),
    ];
    let missing: Vec<&str> =
        want.iter().filter(|(b, _)| seen & (1 << b) == 0).map(|(_, s)| *s).collect();
    assert!(missing.is_empty(), "未実行の分岐: {missing:?}");
}

// --- スキッパブルフレーム ---------------------------------------------------

fn skippable(magic_low: u8, payload: &[u8]) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(&(0x184D_2A50u32 | magic_low as u32).to_le_bytes());
    v.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    v.extend_from_slice(payload);
    v
}

fn with_skippable(frame: &[u8]) -> Vec<u8> {
    let mut v = skippable(0, b"before");
    v.extend_from_slice(frame);
    v.extend_from_slice(&skippable(0x0F, b"after!!"));
    v
}

#[test]
fn skippable_frames_are_ignored() {
    assert_eq!(decompress(&with_skippable(HELLO_3), 12).unwrap(), b"hello, ahiru");
    // 前だけ / 後ろだけ / 中身 0 バイト。
    let mut only_before = skippable(3, b"x");
    only_before.extend_from_slice(HELLO_3);
    assert_eq!(decompress(&only_before, 12).unwrap(), b"hello, ahiru");

    let mut only_after = HELLO_3.to_vec();
    only_after.extend_from_slice(&skippable(0, b""));
    assert_eq!(decompress(&only_after, 12).unwrap(), b"hello, ahiru");
}

#[test]
fn concatenated_frames() {
    let mut v = HELLO_3.to_vec();
    v.extend_from_slice(HELLO_3);
    assert_eq!(decompress(&v, 24).unwrap(), b"hello, ahiruhello, ahiru");
}

// --- 上限 -------------------------------------------------------------------

#[test]
fn output_limit_is_respected() {
    // 宣言サイズが実際より小さい場合はエラー。バッファ溢れは起こさない。
    for short in [0usize, 1, 11] {
        assert_eq!(decompress(HELLO_3, short), Err(Error::LimitExceeded));
    }
    assert_eq!(decompress(RLE_3, 99_999), Err(Error::LimitExceeded));
    // Frame_Content_Size を持たない入力でも書き込み時点で止まる。
    let plain = prose(300_000);
    assert_eq!(decompress(BIG_3, plain.len() - 1), Err(Error::LimitExceeded));
    // ちょうど / 余裕があるときは成功する。
    assert_eq!(decompress(HELLO_3, 12).unwrap().len(), 12);
    assert_eq!(decompress(HELLO_3, 1 << 20).unwrap().len(), 12);
}

// --- 壊れた入力 -------------------------------------------------------------

#[test]
fn bad_magic() {
    let mut v = HELLO_3.to_vec();
    v[0] ^= 0xFF;
    assert_eq!(decompress(&v, 12), Err(Error::BadMagic));
    assert_eq!(decompress(b"", 0), Err(Error::UnexpectedEof));
    assert_eq!(decompress(b"\x28\xb5", 0), Err(Error::UnexpectedEof));
    // 実フレームの後ろに 4 バイト未満のゴミ。
    let mut tail = HELLO_3.to_vec();
    tail.push(0);
    assert_eq!(decompress(&tail, 12), Err(Error::UnexpectedEof));
}

#[test]
fn dictionary_id_is_rejected() {
    // Dictionary_ID_flag = 1、Single_Segment、FCS 1 バイト。
    let mut v = 0xFD2F_B528u32.to_le_bytes().to_vec();
    v.push(0x20 | 0x01); // single segment + DID 1 バイト
    v.push(0xAB); // Dictionary_ID
    v.push(4); // FCS
    v.extend_from_slice(&[0x21, 0x00, 0x00]); // Raw ブロック 4 バイト
    v.extend_from_slice(b"abcd");
    assert_eq!(decompress(&v, 4), Err(Error::DictionaryUnsupported));
}

#[test]
fn reserved_block_type_is_rejected() {
    let mut v = 0xFD2F_B528u32.to_le_bytes().to_vec();
    v.push(0x20); // single segment, FCS 1 バイト
    v.push(4);
    v.extend_from_slice(&[0x27, 0x00, 0x00]); // last=1, type=3 (予約)
    assert_eq!(decompress(&v, 4), Err(Error::BadBlock));
}

#[test]
fn absurd_sequence_count_is_rejected() {
    // 圧縮ブロック: Raw リテラル 0 バイト + シーケンス数 0xFFFF+ の宣言。
    let block: [u8; 8] = [
        0x00, // Raw リテラル, size_format 0, regen 0
        0xFF, 0xFF, 0xFF, // Nb_Sequences = 0x7F00 + 65535
        0x00, // 全て Predefined
        0x00, 0x00, 0x01, // ビットストリーム (中身は破綻している)
    ];
    let mut v = 0xFD2F_B528u32.to_le_bytes().to_vec();
    v.push(0x20);
    v.push(200); // FCS = 200
    let h = ((block.len() as u32) << 3) | (2 << 1) | 1;
    v.extend_from_slice(&h.to_le_bytes()[..3]);
    v.extend_from_slice(&block);
    assert!(decompress(&v, 200).is_err());
}

#[test]
fn offset_before_start_of_output_is_rejected() {
    // Predefined テーブルで「最初のシーケンスがいきなり巨大な offset」を作る。
    // 手で正しいビット列を組むのは大変なので、既存フレームのシーケンス部を
    // 総当たりで壊して「BadOffset を必ず 1 回以上観測する」ことを確認する。
    let mut seen = false;
    for i in 0..HELLO_3.len() {
        for bit in 0..8 {
            let mut v = HELLO_3.to_vec();
            v[i] ^= 1 << bit;
            if decompress(&v, 12) == Err(Error::BadOffset) {
                seen = true;
            }
        }
    }
    // hello は短くシーケンスが無いので、より大きいフレームでも試す。
    for i in 0..200 {
        for bit in 0..8 {
            let mut v = TEXT_3.to_vec();
            v[i] ^= 1 << bit;
            if decompress(&v, 40_000) == Err(Error::BadOffset) {
                seen = true;
            }
        }
    }
    assert!(seen, "BadOffset を観測できなかった");
}

/// 有効なフレームの「あらゆる前置き長」で切り詰める。壊れた入力の探索としては
/// いちばん費用対効果が高い。パニックせず、止まらず、必ず `Err` になること。
#[test]
fn truncation_at_every_prefix() {
    let t0 = Instant::now();
    let cases: [(&[u8], usize); 6] = [
        (EMPTY_3, 0),
        (HELLO_3, 12),
        (RLE_3, 100_000),
        (RUNS_3, 70_128),
        (XBLOCK_3, 200_000),
        (TEXT_19, 40_000),
    ];
    let mut n = 0usize;
    for (comp, out_len) in cases {
        for cut in 0..comp.len() {
            let r = decompress(&comp[..cut], out_len);
            assert!(r.is_err(), "prefix {cut} of {} bytes が成功した", comp.len());
            n += 1;
        }
    }
    // 無限ループが混ざっていればここで落ちる。
    assert!(t0.elapsed().as_secs() < 60, "{n} 件の打ち切りに時間がかかりすぎ");
}

/// ビット反転。`Ok` でも `Err` でもよいが、パニックもハングもしないこと。
#[test]
fn bit_flips_never_panic() {
    let t0 = Instant::now();
    for (comp, out_len) in [(HELLO_3, 12usize), (RUNS_3, 70_128), (RLE_3, 100_000)] {
        for i in 0..comp.len() {
            for bit in 0..8 {
                let mut v = comp.to_vec();
                v[i] ^= 1 << bit;
                let _ = decompress(&v, out_len);
                // 上限を絞った場合も同様に安全であること。
                let _ = decompress(&v, out_len / 2);
            }
        }
    }
    // 大きいフレームは先頭 512 バイト (ヘッダ・表の記述) の上下 2 ビットだけ。
    // 全ビットを叩くとデバッグビルドでは時間がかかりすぎる。
    for comp in [TEXT_3, BIG_19, XBLOCK_3] {
        for i in 0..comp.len().min(512) {
            for bit in [0u32, 7] {
                let mut v = comp.to_vec();
                v[i] ^= 1 << bit;
                let _ = decompress(&v, 300_000);
            }
        }
    }
    assert!(t0.elapsed().as_secs() < 60, "ビット反転テストに時間がかかりすぎ");
}

#[test]
fn garbage_never_panics() {
    // ランダムバイト列に本物のマジックを被せたもの。
    let mut noise = lcg(4096);
    for start in (0..noise.len() - 64).step_by(37) {
        noise[start..start + 4].copy_from_slice(&0xFD2F_B528u32.to_le_bytes());
        let _ = decompress(&noise[start..], 1 << 16);
        let _ = decompress(&noise[start..], 0);
    }
}

// --- XXH64 ------------------------------------------------------------------

#[test]
fn xxh64_known_vectors() {
    // 公式のテストベクタ (seed 0)。
    assert_eq!(xxh64::hash(b""), 0xEF46_DB37_51D8_E999);
    assert_eq!(xxh64::hash(b"a"), 0xD24E_C4F1_A98C_6E5B);
    assert_eq!(xxh64::hash(b"abc"), 0x44BC_2CF5_AD77_0999);
    assert_eq!(xxh64::hash(b"The quick brown fox jumps over the lazy dog"), 0x0B24_2D36_1FDA_71BC);
}

// --- Parquet 実データ -------------------------------------------------------

/// Thrift compact protocol の最小読み取り（PageHeader だけを読む）。
struct Thrift<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Thrift<'a> {
    fn varint(&mut self) -> Option<u64> {
        let mut v = 0u64;
        let mut sh = 0u32;
        loop {
            let x = *self.b.get(self.p)?;
            self.p += 1;
            v |= ((x & 0x7F) as u64) << sh;
            if x & 0x80 == 0 {
                return Some(v);
            }
            sh += 7;
            if sh > 63 {
                return None;
            }
        }
    }
    fn zigzag(&mut self) -> Option<i64> {
        let v = self.varint()?;
        Some(((v >> 1) as i64) ^ -((v & 1) as i64))
    }
    /// 型 `t` の値を読み飛ばす。
    fn skip(&mut self, t: u8) -> Option<()> {
        match t {
            1 | 2 => Some(()),
            3 => {
                self.p += 1;
                Some(())
            }
            4..=6 => self.zigzag().map(|_| ()),
            7 => {
                self.p += 8;
                Some(())
            }
            8 => {
                let n = self.varint()? as usize;
                self.p += n;
                Some(())
            }
            9 | 10 => {
                let h = *self.b.get(self.p)?;
                self.p += 1;
                let n = if h >> 4 == 15 { self.varint()? as usize } else { (h >> 4) as usize };
                for _ in 0..n {
                    self.skip(h & 0x0F)?;
                }
                Some(())
            }
            12 => self.strukt(&mut |_, _, _| Some(false)),
            _ => None,
        }
    }
    /// 構造体を走査し、各フィールドを `f` に渡す。`f` が読まなければ読み飛ばす。
    fn strukt(
        &mut self,
        f: &mut dyn FnMut(&mut Thrift<'a>, i16, u8) -> Option<bool>,
    ) -> Option<()> {
        let mut id: i16 = 0;
        loop {
            let h = *self.b.get(self.p)?;
            self.p += 1;
            if h == 0 {
                return Some(());
            }
            let t = h & 0x0F;
            if h >> 4 == 0 {
                id = self.zigzag()? as i16;
            } else {
                id += (h >> 4) as i16;
            }
            if !f(self, id, t)? {
                self.skip(t)?;
            }
        }
    }
}

/// PageHeader を 1 個読み、(type, uncompressed, compressed, num_values, 終端位置)。
fn page_header(b: &[u8], at: usize) -> Option<(i64, usize, usize, i64, usize)> {
    let mut t = Thrift { b, p: at };
    let (mut ty, mut un, mut co, mut nv) = (-1i64, 0usize, 0usize, 0i64);
    t.strukt(&mut |t, id, ft| {
        match (id, ft) {
            (1, 5) => ty = t.zigzag()?,
            (2, 5) => un = t.zigzag()? as usize,
            (3, 5) => co = t.zigzag()? as usize,
            // DataPageHeader / DataPageHeaderV2 / DictionaryPageHeader の
            // 先頭フィールドが num_values。
            (5, 12) | (7, 12) | (8, 12) => {
                t.strukt(&mut |t, sid, sft| {
                    if (sid, sft) == (1, 5) {
                        nv = t.zigzag()?;
                        return Some(true);
                    }
                    Some(false)
                })?;
            }
            _ => return Some(false),
        }
        Some(true)
    })?;
    if ty < 0 || co == 0 {
        return None;
    }
    Some((ty, un, co, nv, t.p))
}

/// ファイル先頭 (PAR1 の直後) から連続するページを辿り、全て展開する。
/// 返り値は (ページ数, データページの値の総数)。
fn walk_pages(file: &[u8]) -> (usize, i64) {
    let mut at = 4usize;
    let (mut pages, mut values) = (0usize, 0i64);
    while let Some((ty, un, co, nv, end)) = page_header(file, at) {
        let data = match file.get(end..end + co) {
            Some(d) => d,
            None => break,
        };
        if data.len() < 4 || data[..4] != 0xFD2F_B528u32.to_le_bytes() {
            break;
        }
        let out = decompress(data, un).expect("page decompress");
        assert_eq!(out.len(), un, "宣言された展開後サイズと一致しない");
        // 0 = DATA_PAGE, 3 = DATA_PAGE_V2 (2 = DICTIONARY_PAGE は数えない)
        if ty == 0 || ty == 3 {
            values += nv;
        }
        pages += 1;
        at = end + co;
    }
    (pages, values)
}

/// duckdb が書いた実際の ZSTD Parquet のページを全て展開し、宣言された展開後
/// サイズと、duckdb が数えた行数に突き合わせる。
#[test]
fn parquet_zstd_pages() {
    // 単一 INTEGER 列。
    let f1 = std::fs::read(data_path("zstd.parquet")).expect("zstd.parquet");
    assert_eq!(&f1[..4], b"PAR1");
    let (pages, values) = walk_pages(&f1);
    assert!(pages > 0, "ZSTD ページを 1 つも読めていない");
    let rows = duckdb_scalar("SELECT count(*) FROM 'tests/data/zstd.parquet'");
    assert_eq!(values, rows, "ページの値数が duckdb の行数と合わない");

    // INTEGER + VARCHAR + DOUBLE の 3 列。辞書ページや複数カラムチャンクを含む。
    let f2 = std::fs::read(data_path("zstd2.parquet")).expect("zstd2.parquet");
    let (pages2, values2) = walk_pages(&f2);
    assert!(pages2 >= 3, "カラムチャンクを辿れていない: {pages2} ページ");
    let rows2 = duckdb_scalar("SELECT count(*) FROM 'tests/data/zstd2.parquet'");
    assert_eq!(values2 % rows2, 0, "値の総数が行数の倍数でない");
    assert_eq!(values2 / rows2, 3, "3 列分のデータページを辿れていない");
}

fn data_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data")).join(name)
}

fn duckdb_scalar(sql: &str) -> i64 {
    let out = std::process::Command::new("duckdb")
        .current_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
        .args(["-noheader", "-list", "-c", sql])
        .output()
        .expect("duckdb を実行できない");
    assert!(out.status.success(), "duckdb 失敗: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).trim().parse().expect("duckdb の出力が数値でない")
}
