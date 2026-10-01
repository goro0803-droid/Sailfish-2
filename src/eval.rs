//! 評価関数（駒割り + 玉の安全度 + 大駒の働き + 玉との距離による攻め・守り）。
//! 値は手番側から見たセンチポーン単位（歩 ≒ 100）。

use crate::position::{Position, NO_SQ};
use crate::types::*;
use std::sync::atomic::{AtomicI32, Ordering};

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

#[inline]
fn dist(a: usize, b: usize) -> usize {
    let df = (a / 9).abs_diff(b / 9);
    let dr = (a % 9).abs_diff(b % 9);
    df.max(dr)
}

/// 盤上の各升への利きの数（手番ごと）と、大駒の動ける升の数を数える
fn attack_map(pos: &Position, att: &mut [[u8; 81]; 2], mobility: &mut [i32; 2]) {
    for sq in 0..81 {
        let p = pos.board[sq];
        if p == 0 {
            continue;
        }
        let c = color_of(p);
        let pt = type_of(p) as usize;
        let mut mask = STEP_MASK[pt];
        while mask != 0 {
            let d = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            let to = NEIGHBOR[c][sq][d];
            if to >= 0 {
                att[c][to as usize] += 1;
            }
        }
        let mut mask = SLIDE_MASK[pt];
        while mask != 0 {
            let d = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            let mut s = sq;
            loop {
                let to = NEIGHBOR[c][s][d];
                if to < 0 {
                    break;
                }
                let to = to as usize;
                att[c][to] += 1;
                if pt != LANCE as usize {
                    mobility[c] += 1;
                }
                if pos.board[to] != 0 {
                    break;
                }
                s = to;
            }
        }
    }
}

/// 手番 c の玉の危険度（大きいほど危ない）
fn king_danger(pos: &Position, c: usize, att: &[[u8; 81]; 2]) -> i32 {
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
        let a = att[e][s] as i32;
        let def = att[c][s] as i32 - 1; // 玉自身の利きを除く
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
    let check = att[e][k] as i32;
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

pub fn evaluate(pos: &Position) -> i32 {
    let mut score = [0i32; 2];
    let mut att = [[0u8; 81]; 2];
    let mut mobility = [0i32; 2];
    attack_map(pos, &mut att, &mut mobility);

    for sq in 0..81 {
        let p = pos.board[sq];
        if p == 0 {
            continue;
        }
        let c = color_of(p);
        let pt = type_of(p);
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
        score[c] -= king_danger(pos, c, &att) * kd / 100;
    }
    let s = score[BLACK] - score[WHITE];
    if pos.side == BLACK {
        s
    } else {
        -s
    }
}
