//! 基本型・定数・盤面の方向テーブル・Zobrist ハッシュ。
//!
//! 升の番号: sq = (筋 - 1) * 9 + 段 (段は a=0 .. i=8)。
//! 先手(BLACK)は段が小さくなる方向（a段側）へ進む。

pub type Move = u32;
pub const MOVE_NONE: Move = 0;

pub const BLACK: usize = 0;
pub const WHITE: usize = 1;

// 駒種。成駒 = 元の駒種 + PROMOTE（歩・香・桂・銀・角・飛のみ）
pub const PAWN: u8 = 1;
pub const LANCE: u8 = 2;
pub const KNIGHT: u8 = 3;
pub const SILVER: u8 = 4;
pub const BISHOP: u8 = 5;
pub const ROOK: u8 = 6;
pub const GOLD: u8 = 7;
pub const KING: u8 = 8;
pub const PRO_PAWN: u8 = 9;
pub const PRO_LANCE: u8 = 10;
pub const PRO_KNIGHT: u8 = 11;
pub const PRO_SILVER: u8 = 12;
pub const HORSE: u8 = 13;
pub const DRAGON: u8 = 14;
pub const PROMOTE: u8 = 8;

/// 盤上の駒 = 駒種 | (手番 << 4)。0 は空き升。
#[inline]
pub fn make_piece(c: usize, pt: u8) -> u8 {
    pt | ((c as u8) << 4)
}
#[inline]
pub fn color_of(p: u8) -> usize {
    (p >> 4) as usize
}
#[inline]
pub fn type_of(p: u8) -> u8 {
    p & 15
}
#[inline]
pub fn can_promote(pt: u8) -> bool {
    (PAWN..=ROOK).contains(&pt)
}
#[inline]
pub fn unpromote(pt: u8) -> u8 {
    if pt > KING {
        pt - PROMOTE
    } else {
        pt
    }
}
/// 手番から見た段（0 が敵陣の一番奥）
#[inline]
pub fn rel_rank(c: usize, rank: usize) -> usize {
    if c == BLACK {
        rank
    } else {
        8 - rank
    }
}

// 先手視点の方向 (筋の増分, 段の増分)。8,9 は桂馬跳び。
const DIRS: [(i32, i32); 10] = [
    (0, -1),
    (-1, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (1, 1),
    (0, 1),
    (-1, -2),
    (1, -2),
];

/// 駒種ごとの 1 マス移動方向（DIRS のビット）
pub const STEP_MASK: [u16; 15] = [
    0,          // なし
    1,          // 歩
    0,          // 香
    0b11 << 8,  // 桂
    0b1100111,  // 銀: 前・斜め前・斜め後ろ
    0,          // 角
    0,          // 飛
    0b10011111, // 金
    0b11111111, // 玉
    0b10011111, // と
    0b10011111, // 成香
    0b10011111, // 成桂
    0b10011111, // 成銀
    0b10011001, // 馬: 縦横 1 マス
    0b01100110, // 龍: 斜め 1 マス
];

/// 駒種ごとの走り方向
pub const SLIDE_MASK: [u16; 15] = [
    0, 0, 1, 0, 0, 0b01100110, 0b10011001, 0, 0, 0, 0, 0, 0, 0b01100110, 0b10011001,
];

/// NEIGHBOR[手番][升][方向] = 移動先の升（盤外なら -1）
pub static NEIGHBOR: [[[i8; 10]; 81]; 2] = build_neighbors();

const fn build_neighbors() -> [[[i8; 10]; 81]; 2] {
    let mut t = [[[-1i8; 10]; 81]; 2];
    let mut c = 0;
    while c < 2 {
        let mut sq = 0;
        while sq < 81 {
            let f = (sq / 9) as i32;
            let r = (sq % 9) as i32;
            let mut d = 0;
            while d < 10 {
                let (mut df, mut dr) = DIRS[d];
                if c == 1 {
                    df = -df;
                    dr = -dr;
                }
                let nf = f + df;
                let nr = r + dr;
                if nf >= 0 && nf < 9 && nr >= 0 && nr < 9 {
                    t[c][sq][d] = (nf * 9 + nr) as i8;
                }
                d += 1;
            }
            sq += 1;
        }
        c += 1;
    }
    t
}

// ---- 指し手のエンコード ----
// bit 0-6: 移動先, bit 7-13: 移動元（駒打ちは 81 + 駒種）, bit 14: 成り

#[inline]
pub fn mv_to(m: Move) -> usize {
    (m & 0x7f) as usize
}
#[inline]
pub fn mv_from(m: Move) -> usize {
    ((m >> 7) & 0x7f) as usize
}
#[inline]
pub fn mv_is_drop(m: Move) -> bool {
    mv_from(m) >= 81
}
#[inline]
pub fn mv_drop_pt(m: Move) -> u8 {
    (mv_from(m) - 81) as u8
}
#[inline]
pub fn mv_is_promo(m: Move) -> bool {
    m & (1 << 14) != 0
}
#[inline]
pub fn make_move(from: usize, to: usize, promo: bool) -> Move {
    to as u32 | (from as u32) << 7 | (promo as u32) << 14
}
#[inline]
pub fn make_drop(pt: u8, to: usize) -> Move {
    to as u32 | (81 + pt as u32) << 7
}

const HAND_CHARS: &[u8] = b"?PLNSBRG";

pub fn sq_to_usi(sq: usize) -> String {
    format!("{}{}", sq / 9 + 1, (b'a' + (sq % 9) as u8) as char)
}

pub fn move_to_usi(m: Move) -> String {
    if m == MOVE_NONE {
        return "none".to_string();
    }
    if mv_is_drop(m) {
        format!("{}*{}", HAND_CHARS[mv_drop_pt(m) as usize] as char, sq_to_usi(mv_to(m)))
    } else {
        format!(
            "{}{}{}",
            sq_to_usi(mv_from(m)),
            sq_to_usi(mv_to(m)),
            if mv_is_promo(m) { "+" } else { "" }
        )
    }
}

pub fn parse_sq(f: u8, r: u8) -> Option<usize> {
    if (b'1'..=b'9').contains(&f) && (b'a'..=b'i').contains(&r) {
        Some((f - b'1') as usize * 9 + (r - b'a') as usize)
    } else {
        None
    }
}

// ---- Zobrist ハッシュ ----

pub struct Zobrist {
    pub psq: [[u64; 81]; 32],
    pub hand: [[[u64; 20]; 8]; 2],
    pub side: u64,
}

pub static ZOB: Zobrist = build_zobrist();

const fn xorshift(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

const fn build_zobrist() -> Zobrist {
    let mut z = Zobrist { psq: [[0; 81]; 32], hand: [[[0; 20]; 8]; 2], side: 0 };
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut p = 0;
    while p < 32 {
        let mut sq = 0;
        while sq < 81 {
            s = xorshift(s);
            z.psq[p][sq] = s;
            sq += 1;
        }
        p += 1;
    }
    let mut c = 0;
    while c < 2 {
        let mut pt = 0;
        while pt < 8 {
            let mut n = 0;
            while n < 20 {
                s = xorshift(s);
                z.hand[c][pt][n] = s;
                n += 1;
            }
            pt += 1;
        }
        c += 1;
    }
    s = xorshift(s);
    z.side = s;
    z
}
