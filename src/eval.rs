//! 評価関数（駒割り + 玉の安全度 + 大駒の働き + 玉との距離による攻め・守り）。
//! 値は手番側から見たセンチポーン単位（歩 ≒ 100）。

use crate::params::PST;
use crate::position::{Position, NO_SQ};
use crate::types::*;
use std::sync::atomic::{AtomicI32, Ordering};

/// 駒の点数表の添字（後手は盤を 180 度回して先手の向きにそろえる）
#[inline]
pub fn pst_index(c: usize, sq: usize, pt: u8) -> usize {
    pt as usize * 81 + if c == BLACK { sq } else { 80 - sq }
}

/// 調整用の重み（USI の隠しオプション EvalKingDanger / EvalMobility で変更できる）
pub static KING_DANGER_PCT: AtomicI32 = AtomicI32::new(100);
pub static MOBILITY_WEIGHT: AtomicI32 = AtomicI32::new(3);

pub const PIECE_VALUE: [i32; 15] = [
    0, 90, 315, 405, 495, 855, 990, 540, 0, // なし 歩 香 桂 銀 角 飛 金 玉
    540, 540, 540, 540, 945, 1395, // と 杏 圭 全 馬 龍
];

/// 持ち駒の価値（盤上より少し高め）。添字は駒種 (歩..金)
pub const HAND_VALUE: [i32; 8] = [0, 100, 340, 440, 540, 950, 1100, 600];

/// 敵玉との距離ごとの攻めボーナス
const ATTACK_BONUS: [i32; 9] = [0, 28, 20, 12, 6, 2, 0, 0, 0];
/// 自玉との距離ごとの守り駒（金銀・小駒の成り駒）ボーナス
const DEFEND_BONUS: [i32; 9] = [0, 30, 20, 8, 0, 0, 0, 0, 0];
/// 玉の段（自陣の奥 = 8）ごとのボーナス
const KING_RANK: [i32; 9] = [-80, -70, -60, -40, -20, -5, 5, 15, 10];

/// 玉を攻めるときの持ち駒の重み（歩 香 桂 銀 角 飛 金）
const HAND_ATTACK: [i32; 8] = [0, 1, 2, 3, 4, 5, 6, 4];

/// 評価で使う表（実験 V9。前は毎回 NEIGHBOR をたどって求めていたもの。中身は同じ）
struct EvalTables {
    /// 2升の距離（筋・段の差の大きい方）
    dist: [[u8; 81]; 81],
    /// 1升動く駒の利く升 [手番][駒種][升]（個数と升）
    step: Vec<(u8, [u8; 8])>,
    /// 走る駒の升の並び [手番][升][方向]（個数と、近い順の升）
    ray: Vec<(u8, [u8; 8])>,
}

fn tables() -> &'static EvalTables {
    static T: std::sync::OnceLock<EvalTables> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut dist = [[0u8; 81]; 81];
        for a in 0..81 {
            for b in 0..81 {
                dist[a][b] = (a / 9).abs_diff(b / 9).max((a % 9).abs_diff(b % 9)) as u8;
            }
        }
        let mut step = vec![(0u8, [0u8; 8]); 2 * 15 * 81];
        let mut ray = vec![(0u8, [0u8; 8]); 2 * 81 * 10];
        for c in 0..2 {
            for sq in 0..81 {
                for pt in 0..15 {
                    let e = &mut step[(c * 15 + pt) * 81 + sq];
                    let mut mask = STEP_MASK[pt];
                    while mask != 0 {
                        let d = mask.trailing_zeros() as usize;
                        mask &= mask - 1;
                        let to = NEIGHBOR[c][sq][d];
                        if to >= 0 {
                            e.1[e.0 as usize] = to as u8;
                            e.0 += 1;
                        }
                    }
                }
                for d in 0..10 {
                    let e = &mut ray[(c * 81 + sq) * 10 + d];
                    let mut s = sq;
                    loop {
                        let to = NEIGHBOR[c][s][d];
                        if to < 0 || e.0 == 8 {
                            break;
                        }
                        e.1[e.0 as usize] = to as u8;
                        e.0 += 1;
                        if d >= 8 {
                            break;
                        }
                        s = to as usize;
                    }
                }
            }
        }
        EvalTables { dist, step, ray }
    })
}

#[inline]
fn dist(a: usize, b: usize) -> usize {
    tables().dist[a][b] as usize
}

/// 盤上の各升への利きの数（手番ごと）と、大駒の動ける升の数を数える
fn attack_map(pos: &Position, att: &mut [[u8; 81]; 2], mobility: &mut [i32; 2]) {
    let t = tables();
    for sq in 0..81 {
        let p = pos.board[sq];
        if p == 0 {
            continue;
        }
        let c = color_of(p);
        let pt = type_of(p) as usize;
        let (n, tos) = &t.step[(c * 15 + pt) * 81 + sq];
        for &to in &tos[..*n as usize] {
            att[c][to as usize] += 1;
        }
        let mut mask = SLIDE_MASK[pt];
        while mask != 0 {
            let d = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            let (n, tos) = &t.ray[(c * 81 + sq) * 10 + d];
            for &to in &tos[..*n as usize] {
                let to = to as usize;
                att[c][to] += 1;
                if pt != LANCE as usize {
                    mobility[c] += 1;
                }
                if pos.board[to] != 0 {
                    break;
                }
            }
        }
    }
}

/// 手番 c の玉の危険度（大きいほど危ない）
/// att(手番, 升) はその手番の駒がその升に利いている数
#[inline(always)]
fn king_danger(pos: &Position, c: usize, att: impl Fn(usize, usize) -> i32) -> i32 {
    let k = pos.king[c];
    if k == NO_SQ {
        return 0;
    }
    let e = c ^ 1;
    let mut attacked = 0; // 敵の利きがある玉の周りの升の数
    let mut weak = 0; // 敵の利きが守りより多い升の数
    let mut pressure = 0; // 敵の利きの合計
    let mut escapes = 0; // 逃げ道
    for d in 0..8 {
        let s = NEIGHBOR[c][k][d];
        if s < 0 {
            continue;
        }
        let s = s as usize;
        let a = att(e, s);
        let def = att(c, s) - 1; // 玉自身の利きを除く
        if a > 0 {
            attacked += 1;
            pressure += a;
            if a > def {
                weak += 1;
            }
        }
        let p = pos.board[s];
        if a == 0 && (p == 0 || color_of(p) == e) {
            escapes += 1;
        }
    }
    // 玉そのものへの利き（王手）
    let check = att(e, k);
    let mut hand = 0;
    for pt in 1..8 {
        hand += HAND_ATTACK[pt] * pos.hand[e][pt].min(3) as i32;
    }
    let mut danger = attacked * 18 + weak * 30 + pressure * 10 + check * 40 + hand * 7;
    danger += match escapes {
        0 => 90,
        1 => 45,
        2 => 15,
        _ => 0,
    };
    // 危険が重なるほど急に悪くなるよう、二乗で効かせる（上限あり）
    (danger * danger / 260).min(2500)
}

/// 手番側から見た評価値
pub fn evaluate(pos: &Position) -> i32 {
    let score = if cfg!(feature = "v_bbeval") {
        eval_bb(pos)
    } else if cfg!(feature = "v_evalfast") {
        eval_fast(pos)
    } else {
        eval_impl::<true>(pos)
    };
    let s = score[BLACK] - score[WHITE];
    if pos.side == BLACK {
        s
    } else {
        -s
    }
}

/// 駒の点数表を除いた評価値（先手から見た値）。学習用。
pub fn eval_without_pst(pos: &Position) -> i32 {
    let score = eval_impl::<false>(pos);
    score[BLACK] - score[WHITE]
}

/// 実験 速さ4 V15（cargo の機能 v_evalfast）：駒のある升の印を先に作り、駒のある升だけをたどる。
/// 足す順番が違うだけで、値は `eval_impl::<true>` と同じ
fn eval_fast(pos: &Position) -> [i32; 2] {
    let t = tables();
    let mut occ = 0u128;
    for sq in 0..81 {
        occ |= ((pos.board[sq] != 0) as u128) << sq;
    }
    let mut att = [[0u8; 81]; 2];
    let mut mobility = [0i32; 2];
    let mut rest = occ;
    while rest != 0 {
        let sq = rest.trailing_zeros() as usize;
        rest &= rest - 1;
        let p = pos.board[sq];
        let c = color_of(p);
        let pt = type_of(p) as usize;
        let (n, tos) = &t.step[(c * 15 + pt) * 81 + sq];
        for &to in &tos[..*n as usize] {
            att[c][to as usize] += 1;
        }
        let mut mask = SLIDE_MASK[pt];
        while mask != 0 {
            let d = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            let (n, tos) = &t.ray[(c * 81 + sq) * 10 + d];
            for &to in &tos[..*n as usize] {
                let to = to as usize;
                att[c][to] += 1;
                if pt != LANCE as usize {
                    mobility[c] += 1;
                }
                if (occ >> to) & 1 != 0 {
                    break;
                }
            }
        }
    }

    let mut score = [0i32; 2];
    let mut rest = occ;
    while rest != 0 {
        let sq = rest.trailing_zeros() as usize;
        rest &= rest - 1;
        let p = pos.board[sq];
        let c = color_of(p);
        let pt = type_of(p);
        score[c] += PST[pt as usize][if c == BLACK { sq } else { 80 - sq }] as i32;
        if pt == KING {
            score[c] += KING_RANK[rel_rank(c, sq % 9)];
            continue;
        }
        score[c] += PIECE_VALUE[pt as usize];
        let ek = pos.king[c ^ 1];
        if ek != NO_SQ {
            let b = ATTACK_BONUS[t.dist[sq][ek] as usize];
            score[c] += match pt {
                PAWN | BISHOP | ROOK => b / 2,
                _ => b,
            };
        }
        let ok = pos.king[c];
        if ok != NO_SQ && matches!(pt, GOLD | SILVER | PRO_PAWN | PRO_LANCE | PRO_KNIGHT | PRO_SILVER) {
            score[c] += DEFEND_BONUS[t.dist[sq][ok] as usize];
        }
        if att[c ^ 1][sq] > 0 && att[c][sq] == 0 {
            score[c] -= PIECE_VALUE[pt as usize] / 16;
        }
    }
    let kd = KING_DANGER_PCT.load(Ordering::Relaxed);
    let mw = MOBILITY_WEIGHT.load(Ordering::Relaxed);
    for c in 0..2 {
        for pt in 1..8 {
            score[c] += HAND_VALUE[pt] * pos.hand[c][pt] as i32;
        }
        score[c] += mobility[c] * mw;
        score[c] -= king_danger(pos, c, |c, s| att[c][s] as i32) * kd / 100;
    }
    score
}

/// 実験 速さ4 V20（cargo の機能 v_bbeval）：利きを駒ごとの印で求め、升ごとの利きの数は印を4枚重ねたビットごとの足し算で数える。
/// 大駒の動ける升の数は、線の印のビットの数。値は `eval_impl::<true>` と同じ
fn eval_bb(pos: &Position) -> [i32; 2] {
    use crate::bb::{ray_attack, BB};
    let t = tables();
    let occ = pos.occ;
    // planes[c][i]：手番 c の利きの数の、2 の i 乗の位
    let mut planes = [[0u128; 4]; 2];
    let mut mobility = [0i32; 2];
    for c in 0..2 {
        for pt in 1..15 {
            let mut set = pos.bb[c][pt];
            while set != 0 {
                let sq = set.trailing_zeros() as usize;
                set &= set - 1;
                let mut a = BB.step_att[c][pt][sq];
                let mut mask = SLIDE_MASK[pt];
                while mask != 0 {
                    let d = mask.trailing_zeros() as usize;
                    mask &= mask - 1;
                    let g = if c == BLACK { d } else { 7 - d };
                    let r = ray_attack(occ, g, sq);
                    if pt != LANCE as usize {
                        mobility[c] += r.count_ones() as i32;
                    }
                    a |= r;
                }
                // 1升の動きと走る動きの升は重ならないので、駒ごとにまとめて1つ足す
                let p = &mut planes[c];
                let c0 = p[0] & a;
                p[0] ^= a;
                let c1 = p[1] & c0;
                p[1] ^= c0;
                let c2 = p[2] & c1;
                p[2] ^= c1;
                p[3] ^= c2;
            }
        }
    }
    let any = [planes[0][0] | planes[0][1] | planes[0][2] | planes[0][3], planes[1][0] | planes[1][1] | planes[1][2] | planes[1][3]];
    let count = |c: usize, s: usize| -> i32 {
        let p = &planes[c];
        ((p[0] >> s) & 1 | ((p[1] >> s) & 1) << 1 | ((p[2] >> s) & 1) << 2 | ((p[3] >> s) & 1) << 3) as i32
    };

    let mut score = [0i32; 2];
    let mut rest = occ;
    while rest != 0 {
        let sq = rest.trailing_zeros() as usize;
        rest &= rest - 1;
        let p = pos.board[sq];
        let c = color_of(p);
        let pt = type_of(p);
        score[c] += PST[pt as usize][if c == BLACK { sq } else { 80 - sq }] as i32;
        if pt == KING {
            score[c] += KING_RANK[rel_rank(c, sq % 9)];
            continue;
        }
        score[c] += PIECE_VALUE[pt as usize];
        let ek = pos.king[c ^ 1];
        if ek != NO_SQ {
            let b = ATTACK_BONUS[t.dist[sq][ek] as usize];
            score[c] += match pt {
                PAWN | BISHOP | ROOK => b / 2,
                _ => b,
            };
        }
        let ok = pos.king[c];
        if ok != NO_SQ && matches!(pt, GOLD | SILVER | PRO_PAWN | PRO_LANCE | PRO_KNIGHT | PRO_SILVER) {
            score[c] += DEFEND_BONUS[t.dist[sq][ok] as usize];
        }
        if (any[c ^ 1] >> sq) & 1 != 0 && (any[c] >> sq) & 1 == 0 {
            score[c] -= PIECE_VALUE[pt as usize] / 16;
        }
    }
    let kd = KING_DANGER_PCT.load(Ordering::Relaxed);
    let mw = MOBILITY_WEIGHT.load(Ordering::Relaxed);
    for c in 0..2 {
        for pt in 1..8 {
            score[c] += HAND_VALUE[pt] * pos.hand[c][pt] as i32;
        }
        score[c] += mobility[c] * mw;
        score[c] -= king_danger(pos, c, count) * kd / 100;
    }
    score
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 実験 速さ4 V15：駒のある升だけをたどる評価が、前の評価とすべての局面で一致する
    #[test]
    fn fast_eval_matches_eval_impl() {
        let mut rng = 55667788u64;
        let mut pos = Position::startpos();
        let mut ply = 0;
        for i in 0..20000 {
            assert_eq!(eval_fast(&pos), eval_impl::<true>(&pos), "{} 局面目", i);
            if cfg!(feature = "v_bb") {
                assert_eq!(eval_bb(&pos), eval_impl::<true>(&pos), "{} 局面目", i);
            }
            let legal = pos.legal_moves();
            if legal.is_empty() || ply > 250 {
                pos = Position::startpos();
                ply = 0;
                continue;
            }
            ply += 1;
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            pos.do_move(legal[(rng % legal.len() as u64) as usize]);
        }
    }
}

fn eval_impl<const PST_ON: bool>(pos: &Position) -> [i32; 2] {
    let mut score = [0i32; 2];
    let mut att = [[0u8; 81]; 2];
    let mut mobility = [0i32; 2];
    attack_map(pos, &mut att, &mut mobility);
    let dist = |a: usize, b: usize| tables().dist[a][b] as usize;

    for sq in 0..81 {
        let p = pos.board[sq];
        if p == 0 {
            continue;
        }
        let c = color_of(p);
        let pt = type_of(p);
        if PST_ON {
            score[c] += PST[pt as usize][if c == BLACK { sq } else { 80 - sq }] as i32;
        }
        if pt == KING {
            score[c] += KING_RANK[rel_rank(c, sq % 9)];
            continue;
        }
        score[c] += PIECE_VALUE[pt as usize];

        let ek = pos.king[c ^ 1];
        if ek != NO_SQ {
            let b = ATTACK_BONUS[dist(sq, ek)];
            score[c] += match pt {
                PAWN | BISHOP | ROOK => b / 2,
                _ => b,
            };
        }
        let ok = pos.king[c];
        if ok != NO_SQ && matches!(pt, GOLD | SILVER | PRO_PAWN | PRO_LANCE | PRO_KNIGHT | PRO_SILVER) {
            score[c] += DEFEND_BONUS[dist(sq, ok)];
        }
        // 相手の利きがあって、自分の利きがない駒（取られそうな駒）は少し減点
        if att[c ^ 1][sq] > 0 && att[c][sq] == 0 {
            score[c] -= PIECE_VALUE[pt as usize] / 16;
        }
    }
    let kd = KING_DANGER_PCT.load(Ordering::Relaxed);
    let mw = MOBILITY_WEIGHT.load(Ordering::Relaxed);
    for c in 0..2 {
        for pt in 1..8 {
            score[c] += HAND_VALUE[pt] * pos.hand[c][pt] as i32;
        }
        score[c] += mobility[c] * mw;
        score[c] -= king_danger(pos, c, |c, s| att[c][s] as i32) * kd / 100;
    }
    score
}
