//! ビットボード（実験 速さ4 V17〜、cargo の機能 v_bb）：盤の81升を u128 の下位81ビットで表す。升の番号は position と同じ（筋×9＋段）。
//! 表はすべて組み立てのときに計算する。

use crate::types::*;

const NB: [[[i8; 10]; 81]; 2] = build_neighbors();

pub struct BbTables {
    /// ray[g][sq]：升 sq から先手の向きの方向 g（0〜7）へ進んだ升（sq は含まない）
    pub ray: [[u128; 81]; 8],
    /// rev_step[by][pt][sq]：手番 by の駒種 pt がそこにいると、1升の動き（桂は跳ぶ）で sq に利く升
    pub rev_step: [[[u128; 81]; 15]; 2],
    /// 縦横・斜めの4本の線を合わせたもの（飛・角が利きうるか、先に大まかに調べる用）
    pub rook_reach: [u128; 81],
    pub bishop_reach: [u128; 81],
    /// step_att[c][pt][sq]：手番 c の駒種 pt が sq にいるとき、1升の動き（桂は跳ぶ）で利く升
    pub step_att: [[[u128; 81]; 15]; 2],
    /// dir[by][to][s]：手番 by の駒が s から to に利くとき、`least_attacker` が調べる方向の番号（NEIGHBOR[by ^ 1][to][d] の d。線の上か桂の升でなければ 255）
    pub dir: [[[u8; 81]; 81]; 2],
}

/// 方向 g へ進むと升の番号が増えるか（増えるなら、いちばん近い駒は最も下のビット）
pub const RAY_UP: [bool; 8] = {
    let mut up = [false; 8];
    let mut g = 0;
    while g < 8 {
        up[g] = DIRS[g].0 * 9 + DIRS[g].1 > 0;
        g += 1;
    }
    up
};

pub static BB: BbTables = build();

/// 盤の81升
pub const ALL: u128 = (1u128 << 81) - 1;

/// FILE[f]：筋 f（0〜8）の9升
pub const FILE: [u128; 9] = {
    let mut t = [0u128; 9];
    let mut f = 0;
    while f < 9 {
        t[f] = 0x1ffu128 << (f * 9);
        f += 1;
    }
    t
};

/// DROP_OK[us][pt]：手番 us が駒種 pt を打てる段の升（歩・香は1段目に、桂は1・2段目に打てない。二歩は別に調べる）
pub const DROP_OK: [[u128; 8]; 2] = {
    let mut t = [[0u128; 8]; 2];
    let mut us = 0;
    while us < 2 {
        let mut pt = 1;
        while pt < 8 {
            let mut sq = 0;
            while sq < 81 {
                let r = sq % 9;
                let rr = if us == BLACK { r } else { 8 - r };
                let ok = match pt as u8 {
                    PAWN | LANCE => rr != 0,
                    KNIGHT => rr > 1,
                    _ => true,
                };
                if ok {
                    t[us][pt] |= 1u128 << sq;
                }
                sq += 1;
            }
            pt += 1;
        }
        us += 1;
    }
    t
};

const fn build() -> BbTables {
    let mut t = BbTables {
        ray: [[0; 81]; 8],
        rev_step: [[[0; 81]; 15]; 2],
        rook_reach: [0; 81],
        bishop_reach: [0; 81],
        step_att: [[[0; 81]; 15]; 2],
        dir: [[[255; 81]; 81]; 2],
    };
    let mut sq = 0;
    while sq < 81 {
        let mut by = 0;
        while by < 2 {
            let mut d = 0;
            while d < 10 {
                let mut s = sq;
                loop {
                    let n = NB[by ^ 1][s][d];
                    if n < 0 {
                        break;
                    }
                    s = n as usize;
                    t.dir[by][sq][s] = d as u8;
                    if d >= 8 {
                        break;
                    }
                }
                d += 1;
            }
            by += 1;
        }
        let mut g = 0;
        while g < 8 {
            let mut s = sq;
            loop {
                let n = NB[BLACK][s][g];
                if n < 0 {
                    break;
                }
                s = n as usize;
                t.ray[g][sq] |= 1u128 << s;
            }
            g += 1;
        }
        t.rook_reach[sq] = t.ray[0][sq] | t.ray[3][sq] | t.ray[4][sq] | t.ray[7][sq];
        t.bishop_reach[sq] = t.ray[1][sq] | t.ray[2][sq] | t.ray[5][sq] | t.ray[6][sq];
        let mut by = 0;
        while by < 2 {
            let mut pt = 0;
            while pt < 15 {
                let mut d = 0;
                while d < 10 {
                    let a = NB[by ^ 1][sq][d];
                    if STEP_MASK[pt] & (1 << d) != 0 && a >= 0 {
                        t.rev_step[by][pt][sq] |= 1u128 << a;
                    }
                    let to = NB[by][sq][d];
                    if STEP_MASK[pt] & (1 << d) != 0 && to >= 0 {
                        t.step_att[by][pt][sq] |= 1u128 << to;
                    }
                    d += 1;
                }
                pt += 1;
            }
            by += 1;
        }
        sq += 1;
    }
    t
}

/// 升 sq から方向 g へ走る駒の利く升（occ のうち最初にある升まで、その升を含む）
#[inline(always)]
pub fn ray_attack(occ: u128, g: usize, sq: usize) -> u128 {
    let r = BB.ray[g][sq];
    match first_on_ray(occ, g, sq) {
        Some(b) => r & !BB.ray[g][b],
        None => r,
    }
}

/// 升 sq から方向 g へ進んで、occ のうち最初にある升（なければ None）
#[inline(always)]
pub fn first_on_ray(occ: u128, g: usize, sq: usize) -> Option<usize> {
    let r = BB.ray[g][sq] & occ;
    if r == 0 {
        None
    } else if RAY_UP[g] {
        Some(r.trailing_zeros() as usize)
    } else {
        Some(127 - r.leading_zeros() as usize)
    }
}
