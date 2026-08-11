//! XXH64。フレーム末尾の Content_Checksum 検証にだけ使う。
//!
//! 展開結果は全て手元にあるので、逐次更新版は要らず一発計算で足りる。

const P1: u64 = 0x9E37_79B1_85EB_CA87;
const P2: u64 = 0xC2B2_AE3D_27D4_EB4F;
const P3: u64 = 0x1656_67B1_9E37_79F9;
const P4: u64 = 0x85EB_CA77_C2B2_AE63;
const P5: u64 = 0x27D4_EB2F_1656_67C5;

#[inline]
fn round(acc: u64, v: u64) -> u64 {
    acc.wrapping_add(v.wrapping_mul(P2)).rotate_left(31).wrapping_mul(P1)
}

#[inline]
fn merge(h: u64, v: u64) -> u64 {
    (h ^ round(0, v)).wrapping_mul(P1).wrapping_add(P4)
}

#[inline]
fn u64le(b: &[u8]) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(b);
    u64::from_le_bytes(a)
}

#[inline]
fn u32le(b: &[u8]) -> u32 {
    let mut a = [0u8; 4];
    a.copy_from_slice(b);
    u32::from_le_bytes(a)
}

pub fn hash(data: &[u8]) -> u64 {
    let mut h;
    let mut p = 0usize;
    if data.len() >= 32 {
        let (mut v1, mut v2, mut v3, mut v4) =
            (P1.wrapping_add(P2), P2, 0u64, 0u64.wrapping_sub(P1));
        while p + 32 <= data.len() {
            v1 = round(v1, u64le(&data[p..p + 8]));
            v2 = round(v2, u64le(&data[p + 8..p + 16]));
            v3 = round(v3, u64le(&data[p + 16..p + 24]));
            v4 = round(v4, u64le(&data[p + 24..p + 32]));
            p += 32;
        }
        h = v1
            .rotate_left(1)
            .wrapping_add(v2.rotate_left(7))
            .wrapping_add(v3.rotate_left(12))
            .wrapping_add(v4.rotate_left(18));
        h = merge(h, v1);
        h = merge(h, v2);
        h = merge(h, v3);
        h = merge(h, v4);
    } else {
        h = P5;
    }
    h = h.wrapping_add(data.len() as u64);

    while p + 8 <= data.len() {
        h = (h ^ round(0, u64le(&data[p..p + 8])))
            .rotate_left(27)
            .wrapping_mul(P1)
            .wrapping_add(P4);
        p += 8;
    }
    if p + 4 <= data.len() {
        h = (h ^ (u32le(&data[p..p + 4]) as u64).wrapping_mul(P1))
            .rotate_left(23)
            .wrapping_mul(P2)
            .wrapping_add(P3);
        p += 4;
    }
    while p < data.len() {
        h = (h ^ (data[p] as u64).wrapping_mul(P5)).rotate_left(11).wrapping_mul(P1);
        p += 1;
    }

    h ^= h >> 33;
    h = h.wrapping_mul(P2);
    h ^= h >> 29;
    h = h.wrapping_mul(P3);
    h ^= h >> 32;
    h
}
