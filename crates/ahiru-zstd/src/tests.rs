//! Tests that run only on native (std).
//!
//! The fixtures are real frames produced by the `zstd` CLI (v1.5.7), stored in
//! `tests/data/*.zst`; the plaintext is reproduced by rewriting the same generation
//! rule in Rust (keeping the plaintext in the repository would bloat it for nothing).
//! If the generation rules diverge, the comparison tests simply fail, so drift is detectable.
//!
//! Each fixture is the output of a generator function below, compressed with
//! `zstd -f -q --single-thread --no-check -<level> <plaintext> -o <name>.zst`
//! (only `text_crc` uses `--check`). The suffix of the name is the compression level.

use super::*;
use std::time::Instant;

// --- Plaintext generation rules (kept in sync with the .zst files in tests/data) ---

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

/// 140000 bytes of unique lines plus a copy of their first 60000 bytes.
/// The copied part is only reproducible via a back-reference across the 128 KiB block boundary.
fn xblock() -> Vec<u8> {
    let base = lines(140_000);
    let mut v = base.clone();
    v.extend_from_slice(&base[..60_000]);
    v
}

/// A skewed distribution over small symbol values (0..5). Because maxSymbolValue is
/// small, zstd writes the Huffman weights in the 4-bit direct form rather than FSE-compressing them.
fn low_alphabet(n: usize) -> Vec<u8> {
    let mut s: u32 = 0x1357_2468;
    let mut v = Vec::with_capacity(n);
    while v.len() < n {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
        v.push(((s >> 16) % 11).min(5) as u8);
    }
    v
}

/// A very long run of identical bytes. Becomes an overlapping match at offset 1.
fn runs(n: usize) -> Vec<u8> {
    let mut v = lcg(64);
    v.resize(v.len() + n, b'q');
    v.extend_from_slice(&lcg(64));
    v
}

// --- Fixtures ---------------------------------------------------------------

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

/// A frame hand-assembled to use RLE mode for the sequence tables. Real zstd rarely
/// picks this mode for any input, so it is built to spec and checked directly.
/// The reference implementation (`zstd -d`) has been confirmed to expand these bytes to "aaaabbbbcccc".
#[rustfmt::skip]
const RLE_SEQ_FRAME: &[u8] = &[
    0x28, 0xb5, 0x2f, 0xfd, // magic
    0x20,                   // Single_Segment, 1-byte FCS
    12,                     // Frame_Content_Size
    0x55, 0x00, 0x00,       // last=1, Compressed, size=10
    0x18, 0x61, 0x62, 0x63, // 3 bytes of Raw literals, "abc"
    0x03,                   // Nb_Sequences = 3
    0x54,                   // LL=RLE, OF=RLE, ML=RLE
    0x01, 0x00, 0x00,       // LL sym 1 (ll=1), OF sym 0 (rep1), ML sym 0 (ml=3)
    0x01,                   // bitstream (empty, since every table takes 0 bits)
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

// --- Round trips ------------------------------------------------------------

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

/// Input with few distinct symbols. The Huffman weights end up in the 4-bit direct form.
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

/// A back-reference across a block boundary.
#[test]
fn roundtrip_cross_block_match() {
    let plain = xblock();
    assert!(plain.len() > 128 * 1024, "not actually multi-block");
    check(XBLOCK_3, &plain);
}

/// A very long match at offset 1 (source and destination overlap completely).
#[test]
fn roundtrip_overlapping_match() {
    check(RUNS_3, &runs(70_000));
}

#[test]
fn content_checksum_is_verified() {
    let plain = prose(40_000);
    check(TEXT_CRC, &plain);
    assert!(stats::take() & (1 << stats::CHECKSUM) != 0, "the checksum path was not taken");

    // Flip one bit of the 4-byte checksum at the end.
    let mut bad = TEXT_CRC.to_vec();
    let n = bad.len();
    bad[n - 1] ^= 0x80;
    assert_eq!(decompress(&bad, plain.len()), Err(Error::ChecksumMismatch));
}

/// `decompress_into` appends after the existing contents.
#[test]
fn decompress_into_appends() {
    let mut out = b"prefix:".to_vec();
    decompress_into(HELLO_3, &mut out, 64).unwrap();
    assert_eq!(out, b"prefix:hello, ahiru");
}

/// The wasm ABI wraps a host-owned buffer with `Vec::from_raw_parts(ptr, 0, cap)`.
/// A reallocation would invalidate the host's pointer, so this guarantees the buffer
/// never moves as long as `max_len` is within capacity (a safety precondition of the ABI).
#[test]
fn never_reallocates_within_limit() {
    let cases: [(&[u8], usize); 5] =
        [(HELLO_3, 12), (RLE_3, 100_000), (RUNS_3, 70_128), (XBLOCK_3, 200_000), (LOW_3, 60_000)];
    for (comp, n) in cases {
        // Both the happy path and truncated input (the error path).
        for src in [comp, &comp[..comp.len() / 2], &comp[..1]] {
            let mut out: Vec<u8> = Vec::with_capacity(n);
            let (ptr, cap) = (out.as_ptr(), out.capacity());
            let _ = decompress_into(src, &mut out, n);
            assert_eq!(out.as_ptr(), ptr, "the buffer moved");
            assert_eq!(out.capacity(), cap, "a reallocation happened");
            assert!(out.len() <= n);
        }
    }
}

// --- Coverage of the format branches ----------------------------------------

/// Confirms the fixtures actually exercise the branches the spec defines.
/// A safeguard against "the tests pass but the branch never ran".
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
    // Skippable frames are synthesized separately.
    decompress(&with_skippable(HELLO_3), 12).unwrap();
    seen |= stats::take();

    let want: [(u32, &str); 17] = [
        (stats::BLOCK_RAW, "Raw block"),
        (stats::BLOCK_RLE, "RLE block"),
        (stats::BLOCK_COMPRESSED, "Compressed block"),
        (stats::LIT_RAW, "Raw literals"),
        (stats::LIT_RLE, "RLE literals"),
        (stats::LIT_COMPRESSED, "Compressed literals"),
        (stats::LIT_TREELESS, "Treeless literals"),
        (stats::HUF_W_FSE, "FSE-compressed weights"),
        (stats::HUF_W_DIRECT, "4-bit direct weights"),
        (stats::HUF_1STREAM, "1 stream"),
        (stats::HUF_4STREAM, "4 streams"),
        (stats::SEQ_PREDEFINED, "Predefined mode"),
        (stats::SEQ_RLE, "RLE mode"),
        (stats::SEQ_FSE, "FSE_Compressed mode"),
        (stats::SEQ_REPEAT, "Repeat mode"),
        (stats::REPEAT_OFFSET, "repeat offsets"),
        (stats::SKIPPABLE, "skippable frame"),
    ];
    let missing: Vec<&str> =
        want.iter().filter(|(b, _)| seen & (1 << b) == 0).map(|(_, s)| *s).collect();
    assert!(missing.is_empty(), "branches never taken: {missing:?}");
}

// --- Skippable frames -------------------------------------------------------

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
    // Leading only / trailing only / zero-byte contents.
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

// --- Limits -----------------------------------------------------------------

#[test]
fn output_limit_is_respected() {
    // A declared size smaller than the actual one is an error. No buffer overflow occurs.
    for short in [0usize, 1, 11] {
        assert_eq!(decompress(HELLO_3, short), Err(Error::LimitExceeded));
    }
    assert_eq!(decompress(RLE_3, 99_999), Err(Error::LimitExceeded));
    // Input without a Frame_Content_Size also stops at the point of writing.
    let plain = prose(300_000);
    assert_eq!(decompress(BIG_3, plain.len() - 1), Err(Error::LimitExceeded));
    // Exactly enough, or with room to spare, succeeds.
    assert_eq!(decompress(HELLO_3, 12).unwrap().len(), 12);
    assert_eq!(decompress(HELLO_3, 1 << 20).unwrap().len(), 12);
}

// --- Corrupt input ----------------------------------------------------------

#[test]
fn bad_magic() {
    let mut v = HELLO_3.to_vec();
    v[0] ^= 0xFF;
    assert_eq!(decompress(&v, 12), Err(Error::BadMagic));
    assert_eq!(decompress(b"", 0), Err(Error::UnexpectedEof));
    assert_eq!(decompress(b"\x28\xb5", 0), Err(Error::UnexpectedEof));
    // Fewer than 4 bytes of garbage after a real frame.
    let mut tail = HELLO_3.to_vec();
    tail.push(0);
    assert_eq!(decompress(&tail, 12), Err(Error::UnexpectedEof));
}

#[test]
fn dictionary_id_is_rejected() {
    // Dictionary_ID_flag = 1, Single_Segment, 1-byte FCS.
    let mut v = 0xFD2F_B528u32.to_le_bytes().to_vec();
    v.push(0x20 | 0x01); // single segment + 1-byte DID
    v.push(0xAB); // Dictionary_ID
    v.push(4); // FCS
    v.extend_from_slice(&[0x21, 0x00, 0x00]); // 4-byte Raw block
    v.extend_from_slice(b"abcd");
    assert_eq!(decompress(&v, 4), Err(Error::DictionaryUnsupported));
}

#[test]
fn reserved_block_type_is_rejected() {
    let mut v = 0xFD2F_B528u32.to_le_bytes().to_vec();
    v.push(0x20); // single segment, 1-byte FCS
    v.push(4);
    v.extend_from_slice(&[0x27, 0x00, 0x00]); // last=1, type=3 (reserved)
    assert_eq!(decompress(&v, 4), Err(Error::BadBlock));
}

#[test]
fn absurd_sequence_count_is_rejected() {
    // Compressed block: 0 bytes of Raw literals plus a declared sequence count of 0xFFFF+.
    let block: [u8; 8] = [
        0x00, // Raw literals, size_format 0, regen 0
        0xFF, 0xFF, 0xFF, // Nb_Sequences = 0x7F00 + 65535
        0x00, // all Predefined
        0x00, 0x00, 0x01, // bitstream (the contents are broken)
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
    // Make "the very first sequence has a huge offset" with the Predefined tables.
    // Hand-assembling the correct bit sequence is painful, so instead the sequences
    // section of an existing frame is corrupted exhaustively, confirming BadOffset is observed at least once.
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
    // hello is short and has no sequences, so try a larger frame too.
    for i in 0..200 {
        for bit in 0..8 {
            let mut v = TEXT_3.to_vec();
            v[i] ^= 1 << bit;
            if decompress(&v, 40_000) == Err(Error::BadOffset) {
                seen = true;
            }
        }
    }
    assert!(seen, "BadOffset was never observed");
}

/// Truncates a valid frame at every possible prefix length. As a search for corrupt
/// input this has the best cost/benefit. It must not panic, must terminate, and must always be `Err`.
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
            assert!(r.is_err(), "prefix {cut} of {} bytes succeeded", comp.len());
            n += 1;
        }
    }
    // Any infinite loop in there would fail here.
    assert!(t0.elapsed().as_secs() < 60, "truncating {n} prefixes took too long");
}

/// Bit flips. `Ok` or `Err` are both fine, as long as it neither panics nor hangs.
#[test]
fn bit_flips_never_panic() {
    let t0 = Instant::now();
    for (comp, out_len) in [(HELLO_3, 12usize), (RUNS_3, 70_128), (RLE_3, 100_000)] {
        for i in 0..comp.len() {
            for bit in 0..8 {
                let mut v = comp.to_vec();
                v[i] ^= 1 << bit;
                let _ = decompress(&v, out_len);
                // The same must hold safely with a tightened limit.
                let _ = decompress(&v, out_len / 2);
            }
        }
    }
    // For large frames, only the top and bottom 2 bits of the first 512 bytes (header
    // and table descriptions). Hitting every bit takes far too long in debug builds.
    for comp in [TEXT_3, BIG_19, XBLOCK_3] {
        for i in 0..comp.len().min(512) {
            for bit in [0u32, 7] {
                let mut v = comp.to_vec();
                v[i] ^= 1 << bit;
                let _ = decompress(&v, 300_000);
            }
        }
    }
    assert!(t0.elapsed().as_secs() < 60, "the bit-flip test took too long");
}

#[test]
fn garbage_never_panics() {
    // Random bytes with a genuine magic number laid over them.
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
    // The official test vector (seed 0).
    assert_eq!(xxh64::hash(b""), 0xEF46_DB37_51D8_E999);
    assert_eq!(xxh64::hash(b"a"), 0xD24E_C4F1_A98C_6E5B);
    assert_eq!(xxh64::hash(b"abc"), 0x44BC_2CF5_AD77_0999);
    assert_eq!(xxh64::hash(b"The quick brown fox jumps over the lazy dog"), 0x0B24_2D36_1FDA_71BC);
}

// --- Real Parquet data ------------------------------------------------------

/// A minimal read of the Thrift compact protocol (enough for PageHeader only).
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
    /// Skips a value of type `t`.
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
    /// Walks a struct, passing each field to `f`. Anything `f` does not read is skipped.
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

/// Reads one PageHeader, giving (type, uncompressed, compressed, num_values, end position).
fn page_header(b: &[u8], at: usize) -> Option<(i64, usize, usize, i64, usize)> {
    let mut t = Thrift { b, p: at };
    let (mut ty, mut un, mut co, mut nv) = (-1i64, 0usize, 0usize, 0i64);
    t.strukt(&mut |t, id, ft| {
        match (id, ft) {
            (1, 5) => ty = t.zigzag()?,
            (2, 5) => un = t.zigzag()? as usize,
            (3, 5) => co = t.zigzag()? as usize,
            // The first field of DataPageHeader / DataPageHeaderV2 / DictionaryPageHeader
            // is num_values.
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

/// Walks the consecutive pages from the start of the file (just after PAR1) and decompresses them all.
/// Returns (page count, total value count across data pages).
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
        assert_eq!(out.len(), un, "does not match the declared decompressed size");
        // 0 = DATA_PAGE, 3 = DATA_PAGE_V2 (2 = DICTIONARY_PAGE is not counted)
        if ty == 0 || ty == 3 {
            values += nv;
        }
        pages += 1;
        at = end + co;
    }
    (pages, values)
}

/// Decompresses every page of a real ZSTD Parquet file written by duckdb and
/// cross-checks against the declared decompressed sizes and duckdb's row count.
#[test]
fn parquet_zstd_pages() {
    // A single INTEGER column.
    let f1 = std::fs::read(data_path("zstd.parquet")).expect("zstd.parquet");
    assert_eq!(&f1[..4], b"PAR1");
    let (pages, values) = walk_pages(&f1);
    assert!(pages > 0, "not a single ZSTD page was read");
    let rows = duckdb_scalar("SELECT count(*) FROM 'tests/data/zstd.parquet'");
    assert_eq!(values, rows, "the page value count disagrees with duckdb's row count");

    // INTEGER + VARCHAR + DOUBLE, three columns. Includes dictionary pages and multiple column chunks.
    let f2 = std::fs::read(data_path("zstd2.parquet")).expect("zstd2.parquet");
    let (pages2, values2) = walk_pages(&f2);
    assert!(pages2 >= 3, "failed to walk the column chunks: {pages2} pages");
    let rows2 = duckdb_scalar("SELECT count(*) FROM 'tests/data/zstd2.parquet'");
    assert_eq!(values2 % rows2, 0, "the total value count is not a multiple of the row count");
    assert_eq!(values2 / rows2, 3, "failed to walk the data pages of all three columns");
}

fn data_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data")).join(name)
}

fn duckdb_scalar(sql: &str) -> i64 {
    let out = std::process::Command::new("duckdb")
        .current_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
        .args(["-noheader", "-list", "-c", sql])
        .output()
        .expect("cannot run duckdb");
    assert!(out.status.success(), "duckdb failed: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).trim().parse().expect("duckdb output is not a number")
}
