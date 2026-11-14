//! ニューラルネットワークによる評価（NNUE 風）。
//!
//! 入力（視点ごとに作る。相手側の視点は盤を 180 度回して持ち主を入れ替える）:
//!   1. 駒の種類 × 持ち主 × 升（one-hot）
//!   2. 持ち駒の個数（個数ぶんだけ特徴を立てる）
//!   3. 駒の種類 × 持ち主 × 「自玉からの相対位置」、および「敵玉からの相対位置」
//!      （玉と周りの駒の位置関係。玉がどこにいても同じ形なら同じ特徴になる）
//! 第 1 層: 視点ごとに H 個のアキュムレータ（駒が動くたびに差分だけ更新する。玉が動いたときだけ作り直す）
//! 第 2 層: 手番側・相手側のアキュムレータをつないだ 2H 個 → L2 個
//! 出力  : 1 個。評価値 = 土台の評価値（手番側から見た） + 出力 × OUT_SCALE
//!   土台は、駒割りだけ（base_hc = false）か、手作りの評価関数（base_hc = true）。
//!   手作りの評価関数を土台にすると、ネットワークはその「補正」だけを学べばよいので、少ないデータでも使える。

use crate::eval::{evaluate, HAND_VALUE, PIECE_VALUE};
use crate::position::{Position, NO_SQ};
use crate::types::*;
use std::sync::{Arc, RwLock};

/// 第 1 層（アキュムレータ）の幅。2.4 から 256 が標準（2.3 までは 64）。
/// cargo の機能 h64 / h128 で切り替える（h256 は以前のビルド手順との互換のために残してあり、既定と同じ）。
/// ネットのファイルの先頭に H が書かれているので、幅の違うネットは読み込み時に拒否される
#[cfg(not(any(feature = "h64", feature = "h128")))]
pub const H: usize = 256;
#[cfg(all(feature = "h64", not(feature = "h128")))]
pub const H: usize = 64;
#[cfg(feature = "h128")]
pub const H: usize = 128;
/// 第 2 層の幅（段階6の実験では cargo の機能 l2_64 で 64。ネットとエンジンで同じ幅にそろえる。8 の倍数にする）
pub const L2: usize = if cfg!(feature = "l2_64") { 64 } else { 32 };
/// 実験 LS（11/10、cargo の機能 ls4）：第2層・第3層の重みの組の数。盤上の駒の数で組を選ぶ（第1層と出力の偏り b3 は共有）。
/// 組が1つ（既定）なら、ファイルも計算も今までと同じ
pub const NB: usize = if cfg!(feature = "ls4") { 4 } else { 1 };
/// 組の境目：盤上の駒の数が LS_T[k] 以上なら組 k まで（多い順に組 0, 1, 2, 3）。学習データ（train_b5.psv の先頭・中ほど・終わり 各100万局面）の4分位：36以上 25.1%、33〜35 30.5%、31〜32 17.5%、30以下 26.9%
pub const LS_T: [u8; 3] = [36, 33, 31];
#[cfg(all(feature = "ls4", feature = "v_sparse"))]
compile_error!("ls4 と v_sparse は同時に使えない");
/// 盤上の駒の数から組の番号を決める
#[inline]
pub fn bucket_of(npc: u8) -> usize {
    if NB == 1 { return 0; }
    if npc >= LS_T[0] { 0 } else if npc >= LS_T[1] { 1 } else if npc >= LS_T[2] { 2 } else { 3 }
}
/// 盤上の駒の数（玉を含む）
pub fn board_count(board: &[u8; 81]) -> u8 {
    board.iter().filter(|&&p| p != 0).count() as u8
}
/// 持ち駒の種類ごとの最大枚数（歩 香 桂 銀 角 飛 金）
const HAND_CAP: [usize; 8] = [0, 18, 4, 4, 4, 2, 2, 4];
/// 持ち駒の特徴の開始位置（駒種ごと）
const HAND_START: [usize; 8] = [0, 0, 18, 22, 26, 30, 32, 34];
/// 片方の持ち主の持ち駒の特徴の数
const HAND_PER_OWNER: usize = 38;
/// 出力 1.0 が何 cp にあたるか
pub const OUT_SCALE: f32 = 400.0;

const MAGIC: &[u8; 4] = b"SFNN";
const VERSION: u32 = 3;
/// ファイルの先頭（マジック 4 + バージョン・H・L2・特徴数・土台の種類 各 4）
const HEADER_LEN: usize = 4 + 5 * 4;
/// ファイルのバージョン。組が2つ以上なら、上位の16ビットに（組の数 − 1）を入れる（組が1つなら今までと同じ 3）
fn file_version() -> u32 {
    VERSION | ((NB as u32 - 1) << 16)
}

// 入力の特徴は2種類。どちらも N_FEATS・piece_feats・hand_feat・mirror_feature を同じ形で用意する。
//   kp（標準）  : 駒の升・持ち駒・玉からの相対位置（約4,900）
//   hkp（実験C2、cargo の機能 halfkp）: 自玉の升 × 駒（124,254）
#[cfg(not(feature = "halfkp"))]
pub use kp::*;
#[cfg(feature = "halfkp")]
pub use hkp::*;

#[cfg(not(feature = "halfkp"))]
mod kp {
use super::*;
/// 盤上の駒の特徴の数: 持ち主(手番側/相手側) × 駒種 14 × 升 81
const N_PS: usize = 2 * 14 * 81;
const N_HAND: usize = 2 * HAND_PER_OWNER;
/// 玉からの相対位置の範囲（筋・段それぞれ ±KR_R まで。これより遠い駒には玉との位置関係の特徴を付けない）
#[cfg(feature = "kr2")]
const KR_R: i32 = 2;
#[cfg(feature = "kr4")]
const KR_R: i32 = 4;
#[cfg(feature = "kr8")]
const KR_R: i32 = 8;
#[cfg(not(any(feature = "kr2", feature = "kr4", feature = "kr8")))]
const KR_R: i32 = 3;
const KR_W: usize = (2 * KR_R + 1) as usize;
/// 玉からの相対位置の特徴: クラス（玉を除く駒種 13 × 持ち主 2）× 相対位置（KR_W × KR_W）
const KR_CLASSES: usize = 26;
const KR_OFFS: usize = KR_W * KR_W;
const KR_BLOCK: usize = KR_CLASSES * KR_OFFS;
const KR_BASE: usize = N_PS + N_HAND;
/// 特徴の総数（自玉基準のブロックと敵玉基準のブロックがある）
pub const N_FEATS: usize = KR_BASE + 2 * KR_BLOCK;
/// 学習のときだけ足す「因数分解」の特徴の数（kp では使わない）
pub const N_FACTOR: usize = 0;
/// 特徴 f に対応する因数分解の特徴（kp では無い）
#[inline]
pub fn factor_of(_f: usize) -> Option<usize> { None }

/// 視点 persp から見た升の座標（筋, 段）
#[inline]
fn rel_coords(persp: usize, sq: usize) -> (i32, i32) {
    let s = if persp == BLACK { sq } else { 80 - sq };
    ((s / 9) as i32, (s % 9) as i32)
}

/// 視点 persp から見た、盤上の駒の特徴番号
#[inline]
fn ps_feat(persp: usize, color: usize, pt: u8, sq: usize) -> usize {
    let rel_color = (color != persp) as usize;
    let rel_sq = if persp == BLACK { sq } else { 80 - sq };
    (rel_color * 14 + (pt as usize - 1)) * 81 + rel_sq
}

/// 玉 king_sq からの相対位置の特徴番号（enemy_ref が true なら敵玉基準のブロック）。範囲外なら None
#[inline]
fn kr_feat(persp: usize, color: usize, pt: u8, sq: usize, king_sq: usize, enemy_ref: bool) -> Option<usize> {
    let (pf, pr) = rel_coords(persp, sq);
    let (kf, kr) = rel_coords(persp, king_sq);
    let (dx, dy) = (pf - kf, pr - kr);
    if dx.abs() > KR_R || dy.abs() > KR_R {
        return None;
    }
    let rel_color = (color != persp) as usize;
    let kt = pt as usize - 1 - (pt > KING) as usize; // 玉を除いた駒種 0..12
    let class = rel_color * 13 + kt;
    let off = ((dx + KR_R) as usize) * KR_W + (dy + KR_R) as usize;
    Some(KR_BASE + (enemy_ref as usize) * KR_BLOCK + class * KR_OFFS + off)
}

/// 視点 persp から見た、持ち駒の特徴番号（k は 1 始まりの枚数目）。kp では玉の位置は使わない
#[inline]
pub fn hand_feat(persp: usize, owner: usize, pt: u8, k: usize, _kings: [usize; 2]) -> usize {
    let rel_owner = (owner != persp) as usize;
    N_PS + rel_owner * HAND_PER_OWNER + HAND_START[pt as usize] + (k - 1)
}

/// 盤上の駒 1 つの特徴を out に書く（個数を返す）。玉は升の特徴だけ。
#[inline]
pub fn piece_feats(persp: usize, color: usize, pt: u8, sq: usize, kings: [usize; 2], out: &mut [usize; 3]) -> usize {
    out[0] = ps_feat(persp, color, pt, sq);
    let mut n = 1;
    if pt != KING {
        if kings[persp] != NO_SQ {
            if let Some(f) = kr_feat(persp, color, pt, sq, kings[persp], false) {
                out[n] = f;
                n += 1;
            }
        }
        if kings[persp ^ 1] != NO_SQ {
            if let Some(f) = kr_feat(persp, color, pt, sq, kings[persp ^ 1], true) {
                out[n] = f;
                n += 1;
            }
        }
    }
    n
}

/// 左右反転した特徴番号（盤を筋について反転しても形勢は同じなので、学習の水増しに使う）
pub fn mirror_feature(f: usize) -> usize {
    if f < N_PS {
        let (b, s) = (f / 81, f % 81);
        b * 81 + (8 - s / 9) * 9 + s % 9
    } else if f < KR_BASE {
        f
    } else {
        let g = f - KR_BASE;
        let (blk, r) = (g / KR_BLOCK, g % KR_BLOCK);
        let (cls, off) = (r / KR_OFFS, r % KR_OFFS);
        let (dx, dy) = (off / KR_W, off % KR_W);
        KR_BASE + blk * KR_BLOCK + cls * KR_OFFS + (KR_W - 1 - dx) * KR_W + dy
    }
}

/// 実験 速さ6 V28（cargo の機能 v_kinginc）：玉が動いた手も差分で更新できる（kp だけ）
pub const KING_INC: bool = true;

/// V28：手番 us の玉が old_k から new_k へ動いたとき、視点 persp で変わる「その玉からの相対位置」の行を out に足す。
/// board は指したあとの盤（取られた駒は載っていない）。駒ごとに、前の位置関係の行を引く（plus_bit なし）→ 新しい位置関係の行を足す（plus_bit つき）の順に並べる
pub fn king_move_ops(board: &[u8; 81], persp: usize, us: usize, old_k: usize, new_k: usize, out: &mut [u32], n: &mut usize, plus_bit: u32) {
    let eref = persp != us;
    let (of, or) = ((old_k / 9) as i32, (old_k % 9) as i32);
    let (nf, nr) = ((new_k / 9) as i32, (new_k % 9) as i32);
    let (f0, f1) = ((of.min(nf) - KR_R).max(0), (of.max(nf) + KR_R).min(8));
    let (r0, r1) = ((or.min(nr) - KR_R).max(0), (or.max(nr) + KR_R).min(8));
    for f in f0..=f1 {
        for r in r0..=r1 {
            let s = (f * 9 + r) as usize;
            let p = board[s];
            if p == 0 || type_of(p) == KING {
                continue;
            }
            let (c, t) = (color_of(p), type_of(p));
            if let Some(x) = kr_feat(persp, c, t, s, old_k, eref) {
                out[*n] = x as u32;
                *n += 1;
            }
            if let Some(x) = kr_feat(persp, c, t, s, new_k, eref) {
                out[*n] = x as u32 | plus_bit;
                *n += 1;
            }
        }
    }
}
} // mod kp

/// HalfKP 型（実験C2）：視点ごとに「その視点の玉の升 × 駒」。強い将棋ソフトの HalfKP と同じ考え方。
/// 駒（BonaPiece 相当）は、盤上の駒（玉を除く）が 持ち主 2 × 種類 9 × 升 81 = 1,458、持ち駒が 持ち主 2 × 38 = 76。
/// 成香・成桂・成銀・と金は金と同じ種類にまとめる（動きが金と同じなので。駒の価値の違いは土台の手作りの評価関数が持つ）。
#[cfg(feature = "halfkp")]
mod hkp {
use super::*;
/// 盤上の駒の種類（0..9）。玉は特徴にしない
const KIND9: [usize; 15] = [usize::MAX, 0, 1, 2, 3, 5, 7, 4, usize::MAX, 4, 4, 4, 4, 6, 8];
const N_BOARD: usize = 2 * 9 * 81;
/// 駒（BonaPiece 相当）の数
pub const N_P: usize = N_BOARD + 2 * HAND_PER_OWNER;
/// 特徴の総数: 玉の升 81 × 駒
pub const N_FEATS: usize = 81 * N_P;
/// 学習のときだけ足す「因数分解」の特徴（玉の位置によらない駒の特徴）の数
pub const N_FACTOR: usize = N_P;

/// 特徴 f の、玉の位置によらない部分（因数分解の特徴の番号 0..N_P）
#[inline]
pub fn factor_of(f: usize) -> Option<usize> { Some(f % N_P) }

/// 視点 persp から見た自玉の升（玉がいない局面では 0 とする）
#[inline]
fn king_index(persp: usize, kings: [usize; 2]) -> usize {
    let k = kings[persp];
    if k == NO_SQ { 0 } else if persp == BLACK { k } else { 80 - k }
}

/// 視点 persp から見た、持ち駒の特徴番号（k は 1 始まりの枚数目）
#[inline]
pub fn hand_feat(persp: usize, owner: usize, pt: u8, k: usize, kings: [usize; 2]) -> usize {
    let rel_owner = (owner != persp) as usize;
    king_index(persp, kings) * N_P + N_BOARD + rel_owner * HAND_PER_OWNER + HAND_START[pt as usize] + (k - 1)
}

/// 盤上の駒 1 つの特徴を out に書く（個数を返す）。玉は特徴にしない（0 個）
#[inline]
pub fn piece_feats(persp: usize, color: usize, pt: u8, sq: usize, kings: [usize; 2], out: &mut [usize; 3]) -> usize {
    if pt == KING {
        return 0;
    }
    let rel_color = (color != persp) as usize;
    let rel_sq = if persp == BLACK { sq } else { 80 - sq };
    out[0] = king_index(persp, kings) * N_P + (rel_color * 9 + KIND9[pt as usize]) * 81 + rel_sq;
    1
}

/// 左右反転した特徴番号（玉の升と駒の升の両方を反転する。持ち駒はそのまま）
pub fn mirror_feature(f: usize) -> usize {
    let mirror_sq = |s: usize| (8 - s / 9) * 9 + s % 9;
    let (k, p) = (f / N_P, f % N_P);
    let p = if p < N_BOARD { p / 81 * 81 + mirror_sq(p % 81) } else { p };
    mirror_sq(k) * N_P + p
}

/// V28：HalfKP 型は玉が動くと全部の特徴が変わるので、差分にしない
pub const KING_INC: bool = false;
pub fn king_move_ops(_board: &[u8; 81], _persp: usize, _us: usize, _old_k: usize, _new_k: usize, _out: &mut [u32], _n: &mut usize, _plus_bit: u32) {}
} // mod hkp

/// 因数分解の特徴の左右反転（玉の位置によらない駒の特徴 0..N_FACTOR）
pub fn mirror_factor(p: usize) -> usize {
    // 玉の升 0 の特徴として反転し、駒の部分だけを取り出す（mirror_feature は玉の升と駒を別々に反転するので同じ結果）
    if N_FACTOR == 0 { return p; }
    mirror_feature(p) % N_FACTOR
}

/// 盤面の配列から、視点 persp の特徴をすべて out に追加する
pub fn features_arrays(board: &[u8; 81], hand: &[[u8; 8]; 2], kings: [usize; 2], persp: usize, out: &mut Vec<u32>) {
    let mut f = [0usize; 3];
    for sq in 0..81 {
        let p = board[sq];
        if p != 0 {
            let n = piece_feats(persp, color_of(p), type_of(p), sq, kings, &mut f);
            for &i in &f[..n] {
                out.push(i as u32);
            }
        }
    }
    for owner in 0..2 {
        for pt in 1..8u8 {
            let n = (hand[owner][pt as usize] as usize).min(HAND_CAP[pt as usize]);
            for k in 1..=n {
                out.push(hand_feat(persp, owner, pt, k, kings) as u32);
            }
        }
    }
}

/// 局面の特徴をすべて out に追加する（視点 persp から見たもの）
pub fn features(pos: &Position, persp: usize, out: &mut Vec<u32>) {
    features_arrays(&pos.board, &pos.hand, pos.king, persp, out);
}

fn material_arrays(board: &[u8; 81], hand: &[[u8; 8]; 2]) -> i32 {
    let mut s = [0i32; 2];
    for sq in 0..81 {
        let p = board[sq];
        if p != 0 {
            s[color_of(p)] += PIECE_VALUE[type_of(p) as usize];
        }
    }
    for c in 0..2 {
        for pt in 1..8 {
            s[c] += HAND_VALUE[pt] * hand[c][pt] as i32;
        }
    }
    s[BLACK] - s[WHITE]
}

/// 先手から見た駒割り（盤上 + 持ち駒）
pub fn material(pos: &Position) -> i32 {
    material_arrays(&pos.board, &pos.hand)
}

#[derive(Clone)]
pub struct Acc {
    /// 視点ごとのアキュムレータ（第 1 層の出力、活性化の前）
    pub v: [[f32; H]; 2],
    /// 整数化したネット（Network::q）を使うときのアキュムレータ（v の S1 倍を int16 にしたもの）。使うのはどちらか一方だけ
    pub q: [[i16; H]; 2],
    /// 先手から見た駒割り
    pub mat: i32,
    /// 盤上の駒の数（実験 LS の組を選ぶ）
    pub npc: u8,
}

impl Default for Acc {
    fn default() -> Self {
        Acc { v: [[0.0; H]; 2], q: [[0; H]; 2], mat: 0, npc: 0 }
    }
}

impl Acc {
    /// 使っている方のアキュムレータと駒割りだけを写す（両方を写すと、使わない方の分だけ遅くなる）
    #[inline]
    pub fn copy_from(&mut self, o: &Acc, quant: bool) {
        if quant { self.q = o.q; } else { self.v = o.v; }
        self.mat = o.mat;
        self.npc = o.npc;
    }
}

/// 整数化（実験C1）の倍率。第1層は S1 倍して int16、第2層の重みは S2 倍して int16（積和は int32）。
/// 誤差はほとんど S1 で決まる（B2 のネットで、S1 = 1024 なら平均 1.8〜2.8cp、2048 なら 0.9〜1.4cp）。
/// S1 = 2048 だと int16 に入るのは ±16 まで。教師局面のアキュムレータは中間64 で ±7、中間256 で −10〜+5
/// （ランダムな対局では −13 まで）なので、まれにあふれうる。差分更新は飽和演算にして、あふれても
/// 大きな負の値が正に化けないようにする（負の値は活性化で 0 になるので、評価への影響はほとんどない）。
pub const Q_S1: i32 = 2048;
pub const Q_S2: i32 = 8192;

/// 整数化したネット（f32 のネットから作る。出力層だけは f32 のまま）
#[derive(Clone)]
pub struct QNet {
    /// 第1層の倍率（活性化の上限も S1）と、第2層の重みの倍率
    s1: i32,
    s2: i32,
    w1: Vec<i16>,
    b1: Vec<i16>,
    w2: Vec<i16>,
    /// 実験 速さ5 V24（v_sparse）：第2層の重みを入力の組（i16 2つ）ごとに並べ替えた写し。[組 p][出力 j][2]
    #[cfg(feature = "v_sparse")]
    w2p: Vec<i16>,
    b2: Vec<i32>,
    /// 出力層の重みを S1·S2 で割ったもの（第2層の出力を整数のまま掛けられるように）
    w3: Vec<f32>,
    b3: f32,
}

impl QNet {
    fn from_network(n: &Network, s1: i32, s2: i32) -> QNet {
        let q16 = |x: f32, s: i32| (x * s as f32).round().clamp(-32767.0, 32767.0) as i16;
        let s12 = (s1 as i64 * s2 as i64) as f32;
        let w2: Vec<i16> = n.w2.iter().map(|&x| q16(x, s2)).collect();
        #[cfg(feature = "v_sparse")]
        let w2p = {
            let mut v = vec![0i16; w2.len()];
            for p in 0..H {
                for j in 0..L2 {
                    for t in 0..2 {
                        v[p * 2 * L2 + 2 * j + t] = w2[j * 2 * H + 2 * p + t];
                    }
                }
            }
            v
        };
        QNet {
            s1,
            s2,
            w1: n.w1.iter().map(|&x| q16(x, s1)).collect(),
            b1: n.b1.iter().map(|&x| q16(x, s1)).collect(),
            w2,
            #[cfg(feature = "v_sparse")]
            w2p,
            b2: n.b2.iter().map(|&x| (x * s12).round() as i32).collect(),
            w3: n.w3.iter().map(|&x| x / s12).collect(),
            b3: n.b3,
        }
    }

    #[inline]
    fn row(&self, f: usize) -> &[i16; H] {
        self.w1[f * H..f * H + H].try_into().unwrap()
    }
}

/// int16 のアキュムレータに1行を足す（plus = false なら引く）。飽和演算（AVX2 と同じ結果）
#[inline]
fn q_update(v: &mut [i16; H], row: &[i16; H], plus: bool) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 の有無を実行時に確かめた。H は 16 の倍数。
        unsafe { q_update_avx2(v, row, plus) };
        return;
    }
    q_update_scalar(v, row, plus);
}

#[inline]
fn q_update_scalar(v: &mut [i16; H], row: &[i16; H], plus: bool) {
    for i in 0..H {
        v[i] = if plus { v[i].saturating_add(row[i]) } else { v[i].saturating_sub(row[i]) };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q_update_avx2(v: &mut [i16; H], row: &[i16; H], plus: bool) {
    use std::arch::x86_64::*;
    for i in (0..H).step_by(16) {
        let a = _mm256_loadu_si256(v.as_ptr().add(i) as *const __m256i);
        let b = _mm256_loadu_si256(row.as_ptr().add(i) as *const __m256i);
        let r = if plus { _mm256_adds_epi16(a, b) } else { _mm256_subs_epi16(a, b) };
        _mm256_storeu_si256(v.as_mut_ptr().add(i) as *mut __m256i, r);
    }
}

/// 実験 速さ4 V13：指す前のアキュムレータ src に、ops の行（特徴番号と、足すなら true）を順に足し引きして dst に書く。
/// 16要素ずつ読み、全部の行を足してから1回だけ書く。要素ごとの飽和演算の順番は1行ずつのときと同じなので、値も同じ
#[inline]
fn q_apply(q: &QNet, src: &[i16; H], dst: &mut [i16; H], ops: &[(usize, bool)]) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 の有無を実行時に確かめた。H は 16 の倍数で、特徴番号は w1 の範囲内。
        unsafe { q_apply_avx2(q, src, dst, ops) };
        return;
    }
    *dst = *src;
    for &(f, plus) in ops {
        q_update_scalar(dst, q.row(f), plus);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q_apply_avx2(q: &QNet, src: &[i16; H], dst: &mut [i16; H], ops: &[(usize, bool)]) {
    use std::arch::x86_64::*;
    let w = q.w1.as_ptr();
    for i in (0..H).step_by(16) {
        let mut a = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
        for &(f, plus) in ops {
            let b = _mm256_loadu_si256(w.add(f * H + i) as *const __m256i);
            a = if plus { _mm256_adds_epi16(a, b) } else { _mm256_subs_epi16(a, b) };
        }
        _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, a);
    }
}

/// 実験 速さ6 V25（cargo の機能 v_accpf）：差分更新で足し引きする行の一覧（視点ごと）。指した直後に作って控えておく
#[derive(Clone, Copy)]
pub struct AccOps {
    /// ACC_OPS_DIFF なら一覧で更新する。ACC_OPS_OTHER なら今までの道（玉が動いた手の作り直し、整数でないネット）
    pub kind: u8,
    pub n: [u8; 2],
    /// 特徴番号。ACC_OPS_PLUS のビットが立っていれば足す、なければ引く
    pub f: [[u32; ACC_OPS_MAX]; 2],
    /// 駒割りと、盤上の駒の数の変化
    pub mat: i32,
    pub npc: i8,
}
pub const ACC_OPS_OTHER: u8 = 0;
pub const ACC_OPS_DIFF: u8 = 1;
/// 1つの視点で変わる行の数の上限（動かした駒 3・置いた駒 3・取った駒 3・持ち駒 1）
pub const ACC_OPS_MAX: usize = 10;
const ACC_OPS_PLUS: u32 = 1 << 31;
/// V28：玉が動いた手で、1つの視点で変わる行の数の上限（玉以外の駒 38 × 2・玉の升 2・取った駒 3・持ち駒 1）
const KING_OPS_MAX: usize = 96;
/// 1行のうち先読みさせる大きさ（バイト）。行は H×2 バイト
const ACC_PF_BYTES: usize = if cfg!(feature = "v_accpf_pf2") { 128 } else { 512 };

impl Default for AccOps {
    fn default() -> Self {
        AccOps { kind: ACC_OPS_OTHER, n: [0; 2], f: [[0; ACC_OPS_MAX]; 2], mat: 0, npc: 0 }
    }
}

/// V25：指す前のアキュムレータ src に、ops の行を順に足し引きして dst に書く。
/// 要素ごとの飽和演算の順番は1行ずつのときと同じなので、値も同じ
#[inline]
fn q_apply_ops(q: &QNet, src: &[i16; H], dst: &mut [i16; H], ops: &[u32]) {
    #[cfg(target_arch = "x86_64")]
    if !cfg!(feature = "v_accpf_rows") && H % 128 == 0 && std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 の有無を実行時に確かめた。H は 128 の倍数で、特徴番号は w1 の範囲内。
        unsafe { q_apply_ops_avx2(q, src, dst, ops) };
        return;
    }
    q_apply_ops_rows(q, src, dst, ops);
}

/// `q_apply_ops` の、1行ずつ足し引きする道（AVX2 のない CPU と、切り分け用の v_accpf_rows）。値は同じ
fn q_apply_ops_rows(q: &QNet, src: &[i16; H], dst: &mut [i16; H], ops: &[u32]) {
    *dst = *src;
    for &op in ops {
        if cfg!(feature = "v_accpf_rows") {
            q_update(dst, q.row((op & !ACC_OPS_PLUS) as usize), op & ACC_OPS_PLUS != 0);
            continue;
        }
        q_update_scalar(dst, q.row((op & !ACC_OPS_PLUS) as usize), op & ACC_OPS_PLUS != 0);
    }
}

/// レジスタ8本（128要素）ずつ載せて、全部の行を足し引きしてから書く。行は先頭から順に読む
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q_apply_ops_avx2(q: &QNet, src: &[i16; H], dst: &mut [i16; H], ops: &[u32]) {
    use std::arch::x86_64::*;
    const T: usize = 8;
    let w = q.w1.as_ptr();
    let mut i = 0;
    while i < H {
        let mut r = [_mm256_setzero_si256(); T];
        for k in 0..T {
            r[k] = _mm256_loadu_si256(src.as_ptr().add(i + 16 * k) as *const __m256i);
        }
        for &op in ops {
            let row = w.add((op & !ACC_OPS_PLUS) as usize * H + i);
            if op & ACC_OPS_PLUS != 0 {
                for k in 0..T {
                    r[k] = _mm256_adds_epi16(r[k], _mm256_loadu_si256(row.add(16 * k) as *const __m256i));
                }
            } else {
                for k in 0..T {
                    r[k] = _mm256_subs_epi16(r[k], _mm256_loadu_si256(row.add(16 * k) as *const __m256i));
                }
            }
        }
        for k in 0..T {
            _mm256_storeu_si256(dst.as_mut_ptr().add(i + 16 * k) as *mut __m256i, r[k]);
        }
        i += 16 * T;
    }
}

/// V13：盤上の駒1つの特徴を、視点ごとの足し引きの並びに加える（`Network::piece` と同じ順番）
#[inline]
fn push_piece_ops(ops: &mut [[(usize, bool); 12]; 2], n: &mut [usize; 2], plus: bool, color: usize, pt: u8, sq: usize, kings: [usize; 2]) {
    let mut f = [0usize; 3];
    for persp in 0..2 {
        let k = piece_feats(persp, color, pt, sq, kings, &mut f);
        for &i in &f[..k] {
            ops[persp][n[persp]] = (i, plus);
            n[persp] += 1;
        }
    }
}

/// 整数のネットの出力（土台を足す前）。第2層までは整数で計算するので、AVX2 とスカラーの結果は完全に一致する
fn forward_q_scalar(q: &QNet, acc: &Acc, stm: usize) -> f32 {
    let mut x = [0i16; 2 * H];
    for i in 0..H {
        x[i] = acc.q[stm][i].clamp(0, q.s1 as i16);
        x[H + i] = acc.q[stm ^ 1][i].clamp(0, q.s1 as i16);
    }
    let mut out = q.b3;
    let o = bucket_of(acc.npc) * L2;
    for j in o..o + L2 {
        let w = &q.w2[j * 2 * H..(j + 1) * 2 * H];
        let mut s = 0i32;
        for i in 0..2 * H {
            s = s.wrapping_add(x[i] as i32 * w[i] as i32);
        }
        let z = q.b2[j].wrapping_add(s).clamp(0, q.s1 * q.s2);
        out += q.w3[j] * z as f32;
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn forward_q_avx2(q: &QNet, acc: &Acc, stm: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut x = [0i16; 2 * H];
    let zero = _mm256_setzero_si256();
    let top = _mm256_set1_epi16(q.s1 as i16);
    for persp in 0..2 {
        let vals = &acc.q[stm ^ persp];
        for i in (0..H).step_by(16) {
            let v = _mm256_loadu_si256(vals.as_ptr().add(i) as *const __m256i);
            let c = _mm256_min_epi16(_mm256_max_epi16(v, zero), top);
            _mm256_storeu_si256(x.as_mut_ptr().add(persp * H + i) as *mut __m256i, c);
        }
    }
    // V24：0 でない入力の組だけ、32出力ぶんの重みの列と積和する（整数の和なので順番を変えても値は同じ）
    #[cfg(feature = "v_sparse")]
    let sums = {
        const NB: usize = L2 / 8;
        let mut s = [_mm256_setzero_si256(); NB];
        let xi = x.as_ptr() as *const i32;
        let w = q.w2p.as_ptr();
        for c in (0..H).step_by(8) {
            let v = _mm256_loadu_si256(xi.add(c) as *const __m256i);
            let zmask = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(v, zero))) as u32;
            let mut nz = !zmask & 0xff;
            while nz != 0 {
                let p = c + nz.trailing_zeros() as usize;
                nz &= nz - 1;
                let input = _mm256_set1_epi32(*xi.add(p));
                let wp = w.add(p * 2 * L2);
                for b in 0..NB {
                    let wk = _mm256_loadu_si256(wp.add(b * 16) as *const __m256i);
                    s[b] = _mm256_add_epi32(s[b], _mm256_madd_epi16(input, wk));
                }
            }
        }
        let mut sums = [0i32; L2];
        for b in 0..NB {
            _mm256_storeu_si256(sums.as_mut_ptr().add(b * 8) as *mut __m256i, s[b]);
        }
        sums
    };
    // 第2層の8行をまとめて計算し、入力の読み込みを共有する。8行の和は hadd でまとめて1本にする（実験 V2。整数の和なので順番を変えても同じ値）
    #[cfg(not(feature = "v_sparse"))]
    let mut sums = [0i32; L2];
    #[cfg(not(feature = "v_sparse"))]
    let mut j = 0;
    #[cfg(not(feature = "v_sparse"))]
    while j < L2 {
        let w = q.w2.as_ptr().add((bucket_of(acc.npc) * L2 + j) * 2 * H);
        let mut s = [_mm256_setzero_si256(); 8];
        for i in (0..2 * H).step_by(16) {
            let input = _mm256_loadu_si256(x.as_ptr().add(i) as *const __m256i);
            for k in 0..8 {
                let wk = _mm256_loadu_si256(w.add(k * 2 * H + i) as *const __m256i);
                s[k] = _mm256_add_epi32(s[k], _mm256_madd_epi16(input, wk));
            }
        }
        let t0 = _mm256_hadd_epi32(s[0], s[1]);
        let t1 = _mm256_hadd_epi32(s[2], s[3]);
        let t2 = _mm256_hadd_epi32(s[4], s[5]);
        let t3 = _mm256_hadd_epi32(s[6], s[7]);
        let u0 = _mm256_hadd_epi32(t0, t1);
        let u1 = _mm256_hadd_epi32(t2, t3);
        let r = _mm256_add_epi32(_mm256_permute2x128_si256(u0, u1, 0x20), _mm256_permute2x128_si256(u0, u1, 0x31));
        _mm256_storeu_si256(sums.as_mut_ptr().add(j) as *mut __m256i, r);
        j += 8;
    }
    // 出力層は小数なので、これまでと同じ順番（j の小さい順）で足す
    let mut out = q.b3;
    let o = bucket_of(acc.npc) * L2;
    for j in 0..L2 {
        let z = q.b2[o + j].wrapping_add(sums[j]).clamp(0, q.s1 * q.s2);
        out += q.w3[o + j] * z as f32;
    }
    out
}

/// 診断（cargo の機能 prof）：玉が動いたときの作り直しの回数とクロック数
static PROF_REFRESH_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PROF_REFRESH_TSC: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[derive(Clone)]
pub struct Network {
    /// 第 1 層の重み: 特徴ごとに H 個
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    /// 第 2 層の重み: 出力ごとに 2H 個（先頭 H が手番側、続く H が相手側）
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
    pub w3: Vec<f32>,
    pub b3: f32,
    /// true なら手作りの評価関数を土台にする（false なら駒割りだけ）
    pub base_hc: bool,
    /// 整数化したネット（実験C1、USI の NNQuant）。Some なら探索の計算はすべてこちらで行う
    pub q: Option<QNet>,
}

// AVX keeps the original eight independent sums and their reduction order.
// Do not use FMA here: rounding changes could alter search decisions.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx") {
        // SAFETY: runtime detection includes OS support; slices are bounded below.
        return unsafe { dot_avx(a, b) };
    }
    dot_scalar(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn dot_avx(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let len = a.len().min(b.len()) / 8 * 8;
    let mut sum = _mm256_setzero_ps();
    for i in (0..len).step_by(8) {
        let x = _mm256_loadu_ps(a.as_ptr().add(i));
        let y = _mm256_loadu_ps(b.as_ptr().add(i));
        sum = _mm256_add_ps(sum, _mm256_mul_ps(x, y));
    }
    let mut lanes = [0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
    lanes.iter().sum()
}

#[inline]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut s = [0f32; 8];
    for (ca, cb) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        for k in 0..8 {
            s[k] += ca[k] * cb[k];
        }
    }
    s.iter().sum()
}

impl Network {
    /// 乱数で初期化（学習の開始用）
    pub fn random(seed: u64) -> Network {
        let mut s = seed | 1;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0 // -1..1
        };
        // 1 つの局面で立つ特徴は多い（玉からの相対位置を含めて 100 前後）ので、第 1 層は小さめに初期化する
        Network {
            w1: (0..N_FEATS * H).map(|_| rnd() * 0.06).collect(),
            b1: vec![0.3; H],
            w2: (0..NB * L2 * 2 * H).map(|_| rnd() * 0.12).collect(),
            b2: vec![0.5; NB * L2],
            w3: (0..NB * L2).map(|_| rnd() * 0.1).collect(),
            b3: 0.0,
            base_hc: false,
            q: None,
        }
    }

    /// 整数化したネットを付ける（on = false なら外して f32 の計算に戻す）
    pub fn set_quant(&mut self, on: bool) {
        self.set_quant_scales(on, Q_S1, Q_S2);
    }

    /// 倍率を指定して整数化する（倍率を決めるための測定用）
    pub fn set_quant_scales(&mut self, on: bool, s1: i32, s2: i32) {
        self.q = if on { Some(QNet::from_network(self, s1, s2)) } else { None };
    }

    #[inline]
    fn row(&self, f: usize) -> &[f32; H] {
        self.w1[f * H..f * H + H].try_into().unwrap()
    }

    /// 盤面の配列から、アキュムレータを最初から作る
    pub fn refresh_arrays(&self, board: &[u8; 81], hand: &[[u8; 8]; 2], kings: [usize; 2], acc: &mut Acc) {
        let mut feats = Vec::with_capacity(128);
        for persp in 0..2 {
            feats.clear();
            features_arrays(board, hand, kings, persp, &mut feats);
            if let Some(q) = &self.q {
                let v = &mut acc.q[persp];
                v.copy_from_slice(&q.b1);
                for &f in &feats {
                    q_update(v, q.row(f as usize), true);
                }
                continue;
            }
            let v = &mut acc.v[persp];
            v.copy_from_slice(&self.b1);
            for &f in &feats {
                let row = self.row(f as usize);
                for i in 0..H {
                    v[i] += row[i];
                }
            }
        }
        acc.mat = material_arrays(board, hand);
        acc.npc = board_count(board);
    }

    /// 局面から最初のアキュムレータを作る
    pub fn refresh(&self, pos: &Position, acc: &mut Acc) {
        self.refresh_arrays(&pos.board, &pos.hand, pos.king, acc);
    }

    /// 診断（cargo の機能 prof）：作り直しの回数とクロック数を読み出して 0 に戻す
    pub fn prof_refresh_take() -> (u64, u64) {
        use std::sync::atomic::Ordering::Relaxed;
        (PROF_REFRESH_N.swap(0, Relaxed), PROF_REFRESH_TSC.swap(0, Relaxed))
    }

    #[inline]
    fn add(&self, acc: &mut Acc, persp: usize, f: usize) {
        if let Some(q) = &self.q {
            q_update(&mut acc.q[persp], q.row(f), true);
            return;
        }
        let row = self.row(f);
        let v = &mut acc.v[persp];
        for i in 0..H {
            v[i] += row[i];
        }
    }

    #[inline]
    fn sub(&self, acc: &mut Acc, persp: usize, f: usize) {
        if let Some(q) = &self.q {
            q_update(&mut acc.q[persp], q.row(f), false);
            return;
        }
        let row = self.row(f);
        let v = &mut acc.v[persp];
        for i in 0..H {
            v[i] -= row[i];
        }
    }

    /// 盤上の駒の特徴を、両方の視点で足す（加算なら plus = true）
    #[inline]
    fn piece(&self, acc: &mut Acc, plus: bool, color: usize, pt: u8, sq: usize, kings: [usize; 2]) {
        let mut f = [0usize; 3];
        for persp in 0..2 {
            let n = piece_feats(persp, color, pt, sq, kings, &mut f);
            for &i in &f[..n] {
                if plus {
                    self.add(acc, persp, i);
                } else {
                    self.sub(acc, persp, i);
                }
            }
        }
    }

    /// 手 m を指したあとの局面 pos と、指す前のアキュムレータ from から、指したあとのアキュムレータ to を作る。
    /// cap は m で取った駒（取らなければ 0）。指したあとの盤からは取った駒がわからないので、別に渡す。
    pub fn apply_after(&self, from: &Acc, pos: &Position, m: Move, cap: u8, to: &mut Acc) {
        let dst = mv_to(m);
        // 玉が動くと、玉からの相対位置の特徴がすべて変わるので、作り直す（玉の移動は少ない）
        if !mv_is_drop(m) && type_of(pos.board[dst]) == KING {
            #[cfg(all(feature = "prof", target_arch = "x86_64"))]
            let t0 = unsafe { core::arch::x86_64::_rdtsc() };
            match &self.q {
                // 実験 速さ6 V28（cargo の機能 v_kinginc）：玉が動いた手も、変わる行だけ足し引きする
                Some(q) if cfg!(feature = "v_kinginc") && KING_INC => Self::apply_king_move(q, from, pos, m, cap, to),
                _ => self.refresh_arrays(&pos.board, &pos.hand, pos.king, to),
            }
            #[cfg(all(feature = "prof", target_arch = "x86_64"))]
            {
                use std::sync::atomic::Ordering::Relaxed;
                PROF_REFRESH_N.fetch_add(1, Relaxed);
                PROF_REFRESH_TSC.fetch_add(unsafe { core::arch::x86_64::_rdtsc() } - t0, Relaxed);
            }
            return;
        }
        if cfg!(feature = "v_accfuse") {
            if let Some(q) = &self.q {
                Self::apply_after_q(q, from, pos, m, cap, to);
                return;
            }
        }
        self.apply_after_rows(from, pos, m, cap, to);
    }

    /// V28：玉が動いた手 m の差分更新（pos は指したあとの局面、cap は取った駒）。
    /// 変わるのは、玉の升の行、動いた玉からの相対位置の行（前後の玉の周りの駒の分）、取った駒の行と持ち駒の行だけ。
    /// 足し引きの途中で値が大きく振れないよう、駒ごとに「引く → 足す」の順に並べる（飽和が起きなければ、作り直しと同じ値になる）
    fn apply_king_move(q: &QNet, from: &Acc, pos: &Position, m: Move, cap: u8, to: &mut Acc) {
        let us = pos.side ^ 1;
        let (src, dst) = (mv_from(m), mv_to(m));
        let kings = pos.king;
        let mut old_kings = kings;
        old_kings[us] = src;
        let mut f = [0usize; 3];
        for persp in 0..2 {
            let mut ops = [0u32; KING_OPS_MAX];
            let mut n = 0;
            if cap != 0 {
                // 取った駒に付いていた行（指す前の玉の位置で）を引く
                let k = piece_feats(persp, color_of(cap), type_of(cap), dst, old_kings, &mut f);
                for &i in &f[..k] {
                    ops[n] = i as u32;
                    n += 1;
                }
            }
            piece_feats(persp, us, KING, src, old_kings, &mut f);
            ops[n] = f[0] as u32;
            piece_feats(persp, us, KING, dst, kings, &mut f);
            ops[n + 1] = f[0] as u32 | ACC_OPS_PLUS;
            n += 2;
            king_move_ops(&pos.board, persp, us, src, dst, &mut ops, &mut n, ACC_OPS_PLUS);
            if cap != 0 {
                let hpt = unpromote(type_of(cap));
                let k = (pos.hand[us][hpt as usize] as usize).min(HAND_CAP[hpt as usize]);
                ops[n] = hand_feat(persp, us, hpt, k, kings) as u32 | ACC_OPS_PLUS;
                n += 1;
            }
            q_apply_ops(q, &from.q[persp], &mut to.q[persp], &ops[..n]);
        }
        to.mat = from.mat;
        to.npc = from.npc;
        if cap != 0 {
            let cpt = type_of(cap);
            let sign = if us == BLACK { 1 } else { -1 };
            to.mat += sign * (HAND_VALUE[unpromote(cpt) as usize] + PIECE_VALUE[cpt as usize]);
            to.npc -= 1;
        }
    }

    /// 差分更新を1行ずつ行う（玉が動いた手は呼ぶ前に除く）。V13 の比べる相手としてテストでも使う
    fn apply_after_rows(&self, from: &Acc, pos: &Position, m: Move, cap: u8, to: &mut Acc) {
        let us = pos.side ^ 1;
        let dst = mv_to(m);
        to.copy_from(from, self.q.is_some());
        let kings = pos.king;
        let sign = if us == BLACK { 1 } else { -1 };
        if mv_is_drop(m) {
            let pt = mv_drop_pt(m);
            // 打つ前の持ち駒の枚数は、打ったあとの枚数 + 1
            let before = (pos.hand[us][pt as usize] as usize + 1).min(HAND_CAP[pt as usize]);
            for persp in 0..2 {
                self.sub(to, persp, hand_feat(persp, us, pt, before, kings));
            }
            self.piece(to, true, us, pt, dst, kings);
            to.mat += sign * (PIECE_VALUE[pt as usize] - HAND_VALUE[pt as usize]);
            to.npc += 1;
        } else {
            let src = mv_from(m);
            let npt = type_of(pos.board[dst]); // 指したあとの駒種（成っていれば成り駒）
            let pt = if mv_is_promo(m) { npt - PROMOTE } else { npt };
            self.piece(to, false, us, pt, src, kings);
            self.piece(to, true, us, npt, dst, kings);
            to.mat += sign * (PIECE_VALUE[npt as usize] - PIECE_VALUE[pt as usize]);
            if cap != 0 {
                let cpt = type_of(cap);
                self.piece(to, false, color_of(cap), cpt, dst, kings);
                let hpt = unpromote(cpt);
                // 取ったあとの持ち駒の枚数
                let n = (pos.hand[us][hpt as usize] as usize).min(HAND_CAP[hpt as usize]);
                for persp in 0..2 {
                    self.add(to, persp, hand_feat(persp, us, hpt, n, kings));
                }
                to.mat += sign * (HAND_VALUE[hpt as usize] + PIECE_VALUE[cpt as usize]);
                to.npc -= 1;
            }
        }
    }

    /// V13：整数化したネットの差分更新を、視点ごとに1回の読み書きで行う（玉が動いた手は呼ぶ前に除く）。
    /// 足し引きの順番と駒割りの計算は `apply_after` と同じ
    fn apply_after_q(q: &QNet, from: &Acc, pos: &Position, m: Move, cap: u8, to: &mut Acc) {
        let us = pos.side ^ 1;
        let dst = mv_to(m);
        let kings = pos.king;
        let sign = if us == BLACK { 1 } else { -1 };
        let mut ops = [[(0usize, false); 12]; 2];
        let mut n = [0usize; 2];
        let mut mat = from.mat;
        let mut npc = from.npc;
        if mv_is_drop(m) {
            let pt = mv_drop_pt(m);
            let before = (pos.hand[us][pt as usize] as usize + 1).min(HAND_CAP[pt as usize]);
            for persp in 0..2 {
                ops[persp][n[persp]] = (hand_feat(persp, us, pt, before, kings), false);
                n[persp] += 1;
            }
            push_piece_ops(&mut ops, &mut n, true, us, pt, dst, kings);
            mat += sign * (PIECE_VALUE[pt as usize] - HAND_VALUE[pt as usize]);
            npc += 1;
        } else {
            let src = mv_from(m);
            let npt = type_of(pos.board[dst]);
            let pt = if mv_is_promo(m) { npt - PROMOTE } else { npt };
            push_piece_ops(&mut ops, &mut n, false, us, pt, src, kings);
            push_piece_ops(&mut ops, &mut n, true, us, npt, dst, kings);
            mat += sign * (PIECE_VALUE[npt as usize] - PIECE_VALUE[pt as usize]);
            if cap != 0 {
                let cpt = type_of(cap);
                push_piece_ops(&mut ops, &mut n, false, color_of(cap), cpt, dst, kings);
                let hpt = unpromote(cpt);
                let k = (pos.hand[us][hpt as usize] as usize).min(HAND_CAP[hpt as usize]);
                for persp in 0..2 {
                    ops[persp][n[persp]] = (hand_feat(persp, us, hpt, k, kings), true);
                    n[persp] += 1;
                }
                mat += sign * (HAND_VALUE[hpt as usize] + PIECE_VALUE[cpt as usize]);
                npc -= 1;
            }
        }
        for persp in 0..2 {
            q_apply(q, &from.q[persp], &mut to.q[persp], &ops[persp][..n[persp]]);
        }
        to.mat = mat;
        to.npc = npc;
    }

    /// 実験 速さ6 V25（cargo の機能 v_accpf）：手 m を指したあとの局面 pos から、差分更新で足し引きする行の一覧を作る。
    /// 作りながら行を先読みさせる（子の局面で評価が要るまでの間に、行がキャッシュに届く）。
    /// 一覧の順番は `apply_after_rows` と同じ（視点ごとに、動かした駒を引く → 置いた駒を足す → 取った駒を引く → 持ち駒を足す）
    #[inline]
    pub fn prepare_ops(&self, pos: &Position, m: Move, cap: u8) -> AccOps {
        let mut o = AccOps { kind: ACC_OPS_OTHER, n: [0; 2], f: [[0; ACC_OPS_MAX]; 2], mat: 0, npc: 0 };
        let dst = mv_to(m);
        // 玉が動いた手は作り直し、整数でないネットは今までの道で更新する
        if !mv_is_drop(m) && type_of(pos.board[dst]) == KING {
            return o;
        }
        let Some(q) = &self.q else { return o };
        o.kind = ACC_OPS_DIFF;
        let us = pos.side ^ 1;
        let kings = pos.king;
        let sign = if us == BLACK { 1 } else { -1 };
        let push = |o: &mut AccOps, persp: usize, f: usize, plus: bool| {
            o.f[persp][o.n[persp] as usize] = f as u32 | if plus { ACC_OPS_PLUS } else { 0 };
            o.n[persp] += 1;
        };
        let piece = |o: &mut AccOps, plus: bool, color: usize, pt: u8, sq: usize| {
            let mut f = [0usize; 3];
            for persp in 0..2 {
                let k = piece_feats(persp, color, pt, sq, kings, &mut f);
                for &i in &f[..k] {
                    o.f[persp][o.n[persp] as usize] = i as u32 | if plus { ACC_OPS_PLUS } else { 0 };
                    o.n[persp] += 1;
                }
            }
        };
        if mv_is_drop(m) {
            let pt = mv_drop_pt(m);
            let before = (pos.hand[us][pt as usize] as usize + 1).min(HAND_CAP[pt as usize]);
            for persp in 0..2 {
                push(&mut o, persp, hand_feat(persp, us, pt, before, kings), false);
            }
            piece(&mut o, true, us, pt, dst);
            o.mat = sign * (PIECE_VALUE[pt as usize] - HAND_VALUE[pt as usize]);
            o.npc = 1;
        } else {
            let src = mv_from(m);
            let npt = type_of(pos.board[dst]);
            let pt = if mv_is_promo(m) { npt - PROMOTE } else { npt };
            piece(&mut o, false, us, pt, src);
            piece(&mut o, true, us, npt, dst);
            o.mat = sign * (PIECE_VALUE[npt as usize] - PIECE_VALUE[pt as usize]);
            if cap != 0 {
                let cpt = type_of(cap);
                piece(&mut o, false, color_of(cap), cpt, dst);
                let hpt = unpromote(cpt);
                let k = (pos.hand[us][hpt as usize] as usize).min(HAND_CAP[hpt as usize]);
                for persp in 0..2 {
                    push(&mut o, persp, hand_feat(persp, us, hpt, k, kings), true);
                }
                o.mat += sign * (HAND_VALUE[hpt as usize] + PIECE_VALUE[cpt as usize]);
                o.npc = -1;
            }
        }
        #[cfg(all(target_arch = "x86_64", not(feature = "v_accpf_nopf")))]
        {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            let w = q.w1.as_ptr() as *const i8;
            for persp in 0..2 {
                for &op in &o.f[persp][..o.n[persp] as usize] {
                    let base = (op & !ACC_OPS_PLUS) as usize * H * 2;
                    let mut off = 0;
                    while off < ACC_PF_BYTES.min(H * 2) {
                        // SAFETY: 先読みは番地を読まない（範囲の外でも害はない）。特徴番号は w1 の範囲内
                        unsafe { _mm_prefetch(w.add(base + off), _MM_HINT_T0) };
                        off += 64;
                    }
                }
            }
        }
        o
    }

    /// V25：控えておいた行の一覧で、指す前のアキュムレータ from から指したあとのアキュムレータ to を作る（`kind` が差分のときだけ呼ぶ）
    #[inline]
    pub fn apply_ops(&self, from: &Acc, ops: &AccOps, to: &mut Acc) {
        let q = self.q.as_ref().expect("apply_ops は整数のネットだけ");
        for persp in 0..2 {
            q_apply_ops(q, &from.q[persp], &mut to.q[persp], &ops.f[persp][..ops.n[persp] as usize]);
        }
        to.mat = from.mat + ops.mat;
        to.npc = (from.npc as i8 + ops.npc) as u8;
    }

    /// ネットワークの出力（土台を足す前、-1〜1 くらいの値）
    #[inline]
    pub fn forward_raw(&self, acc: &Acc, stm: usize) -> f32 {
        if let Some(q) = &self.q {
            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 の有無を実行時に確かめた。
                return unsafe { forward_q_avx2(q, acc, stm) };
            }
            return forward_q_scalar(q, acc, stm);
        }
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: AVX is available; forward_avx checks slice bounds.
            return unsafe { self.forward_avx(acc, stm) };
        }
        let mut x = [0f32; 2 * H];
        for i in 0..H {
            x[i] = acc.v[stm][i].clamp(0.0, 1.0);
            x[H + i] = acc.v[stm ^ 1][i].clamp(0.0, 1.0);
        }
        let mut out = self.b3;
        let o = bucket_of(acc.npc) * L2;
        for j in o..o + L2 {
            let z = self.b2[j] + dot(&self.w2[j * 2 * H..(j + 1) * 2 * H], &x);
            out += self.w3[j] * z.clamp(0.0, 1.0);
        }
        out
    }

    // Compute four second-layer neurons together, sharing each input load.
    // Keep the scalar lane sums and output accumulation order (no FMA).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    unsafe fn forward_avx(&self, acc: &Acc, stm: usize) -> f32 {
        use std::arch::x86_64::*;
        let mut x = [0f32; 2 * H];
        let zero = _mm256_setzero_ps();
        let one = _mm256_set1_ps(1.0);
        for persp in 0..2 {
            let values = &acc.v[stm ^ persp];
            for i in (0..H).step_by(8) {
                let v = _mm256_loadu_ps(values.as_ptr().add(i));
                // The second operand wins on equality/NaN. This order keeps -0
                // and matches f32::clamp instead of changing activation bits.
                let clipped = _mm256_min_ps(one, _mm256_max_ps(zero, v));
                _mm256_storeu_ps(x.as_mut_ptr().add(persp * H + i), clipped);
            }
        }
        let mut out = self.b3;
        let o = bucket_of(acc.npc) * L2;
        let mut j = o;
        while j + 4 <= o + L2 {
            let weights = &self.w2[j * 2 * H..(j + 4) * 2 * H];
            let mut sums = [_mm256_setzero_ps(); 4];
            for i in (0..2 * H).step_by(8) {
                let input = _mm256_loadu_ps(x.as_ptr().add(i));
                for k in 0..4 {
                    let weight = _mm256_loadu_ps(weights.as_ptr().add(k * 2 * H + i));
                    sums[k] = _mm256_add_ps(sums[k], _mm256_mul_ps(input, weight));
                }
            }
            for k in 0..4 {
                let mut lanes = [0f32; 8];
                _mm256_storeu_ps(lanes.as_mut_ptr(), sums[k]);
                let z = self.b2[j + k] + lanes.iter().sum::<f32>();
                out += self.w3[j + k] * z.clamp(0.0, 1.0);
            }
            j += 4;
        }
        for k in j..o + L2 {
            let z = self.b2[k] + dot_avx(&self.w2[k * 2 * H..(k + 1) * 2 * H], &x);
            out += self.w3[k] * z.clamp(0.0, 1.0);
        }
        out
    }
    /// 局面 pos（手番側から見た）の評価値（cp）。acc は pos に対応するアキュムレータ。
    pub fn forward(&self, acc: &Acc, pos: &Position) -> i32 {
        let stm = pos.side;
        let out = self.forward_raw(acc, stm);
        let base = if self.base_hc {
            evaluate(pos)
        } else if stm == BLACK {
            acc.mat
        } else {
            -acc.mat
        };
        Self::combine(base, out)
    }

    /// 評価の土台だけをキャッシュした探索から呼ぶ。NN 出力は毎回計算する。
    pub fn forward_with_base(&self, acc: &Acc, stm: usize, base: i32) -> i32 {
        Self::combine(base, self.forward_raw(acc, stm))
    }

    #[inline]
    fn combine(base: i32, out: f32) -> i32 {
        ((base as f32 + out * OUT_SCALE) as i32).clamp(-20000, 20000)
    }

    // ---- ファイルへの保存・読み込み ----

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(MAGIC);
        for v in [file_version(), H as u32, L2 as u32, N_FEATS as u32, self.base_hc as u32] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for arr in [&self.w1, &self.b1, &self.w2, &self.b2, &self.w3] {
            for &x in arr.iter() {
                b.extend_from_slice(&x.to_le_bytes());
            }
        }
        b.extend_from_slice(&self.b3.to_le_bytes());
        b
    }

    pub fn from_bytes(b: &[u8]) -> Result<Network, String> {
        if b.len() < HEADER_LEN || &b[0..4] != MAGIC {
            return Err("ネットワークファイルではありません".into());
        }
        let u = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap());
        if u(4) != file_version() || u(8) as usize != H || u(12) as usize != L2 || u(16) as usize != N_FEATS {
            return Err("ネットワークの形（バージョン・幅）がこのエンジンと合いません".into());
        }
        let base_hc = u(20) != 0;
        let total = N_FEATS * H + H + NB * (L2 * 2 * H + L2 + L2) + 1;
        if b.len() != HEADER_LEN + total * 4 {
            return Err("ネットワークファイルの大きさが合いません".into());
        }
        let mut pos = HEADER_LEN;
        let mut take = |n: usize| -> Vec<f32> {
            let v = (0..n).map(|i| f32::from_le_bytes(b[pos + i * 4..pos + i * 4 + 4].try_into().unwrap())).collect();
            pos += n * 4;
            v
        };
        let w1 = take(N_FEATS * H);
        let b1 = take(H);
        let w2 = take(NB * L2 * 2 * H);
        let b2 = take(NB * L2);
        let w3 = take(NB * L2);
        let b3 = take(1)[0];
        let mut net = Network { w1, b1, w2, b2, w3, b3, base_hc, q: None };
        if QUANT.load(std::sync::atomic::Ordering::Relaxed) {
            net.set_quant(true);
        }
        Ok(net)
    }
}

// ---- 読み込み済みのネットワーク（エンジン全体で 1 つ） ----

static GLOBAL: RwLock<Option<Arc<Network>>> = RwLock::new(None);

/// 整数化したネットを使うか（USI の NNQuant。既定は使わない）。読み込むネットにも引き継ぐ
static QUANT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// 整数化の有無を切り替え、読み込み済みのネットにも反映する
pub fn set_quant_global(on: bool) {
    QUANT.store(on, std::sync::atomic::Ordering::Relaxed);
    if let Some(net) = global() {
        if net.q.is_some() != on {
            let mut n = (*net).clone();
            n.set_quant(on);
            set_global(n);
        }
    }
}

/// 読み込み済みのネットワークを返す
pub fn global() -> Option<Arc<Network>> {
    GLOBAL.read().ok().and_then(|g| g.clone())
}

/// ファイルからネットワークを読み込んで、エンジン全体で使うようにする
pub fn load_global(path: &str) -> Result<(), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {}", path, e))?;
    set_global(Network::from_bytes(&bytes)?);
    Ok(())
}

fn set_global(net: Network) {
    if let Ok(mut g) = GLOBAL.write() {
        *g = Some(Arc::new(net));
    }
}

/// 実行ファイルに埋め込んだ標準のネットワーク（nets/default.nn。11/10 から B5 に戻した。11/7〜11/10 は段階6の L64H だった）。
/// cargo の機能 net_b4 なら実験B4（20エポック）の、net_b5 なら実験B5（水匠10 のデータを追加）の、
/// net_nh なら実験NH（土台を駒の損得だけにして、手作りの評価を使わない）のネットを埋め込む。
/// net_l64h・net_l64m は段階6の実験（第2層を 64 に。土台はそれぞれ手作りの評価・駒の損得だけ）のネットを埋め込む。
#[cfg(not(any(feature = "net_b4", feature = "net_b5", feature = "net_nh", feature = "net_l64h", feature = "net_l64m", feature = "net_ls4")))]
static DEFAULT_NET: &[u8] = include_bytes!("../nets/default.nn");
/// 実験 LS（11/10）：組を4つにしたネット（nets/ls4.nn。学習が終わってから置く）
#[cfg(feature = "net_ls4")]
static DEFAULT_NET: &[u8] = include_bytes!("../nets/ls4.nn");
#[cfg(feature = "net_l64h")]
static DEFAULT_NET: &[u8] = include_bytes!("../nets/l64h.nn");
#[cfg(feature = "net_l64m")]
static DEFAULT_NET: &[u8] = include_bytes!("../nets/l64m.nn");
#[cfg(all(feature = "net_nh", not(any(feature = "net_b4", feature = "net_b5"))))]
static DEFAULT_NET: &[u8] = include_bytes!("../nets/nh.nn");
#[cfg(feature = "net_b4")]
static DEFAULT_NET: &[u8] = include_bytes!("../nets/b4.nn");
#[cfg(all(feature = "net_b5", not(feature = "net_b4")))]
static DEFAULT_NET: &[u8] = include_bytes!("../nets/b5.nn");

/// 埋め込んだ標準のネットワークを読み込んで、エンジン全体で使うようにする
pub fn load_default() -> Result<(), String> {
    set_global(Network::from_bytes(DEFAULT_NET)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_forward_is_bit_exact() {
        let nets = test_nets(19);
        let mut rng = 123456789u64;
        for net in &nets {
            let mut pos = Position::startpos();
            for _ in 0..500 {
                let mut acc = Acc::default();
                net.refresh(&pos, &mut acc);
                for stm in 0..2 {
                    let mut x = [0f32; 2 * H];
                    for i in 0..H {
                        x[i] = acc.v[stm][i].clamp(0.0, 1.0);
                        x[H + i] = acc.v[stm ^ 1][i].clamp(0.0, 1.0);
                    }
                    let mut reference = net.b3;
                    let o = bucket_of(acc.npc) * L2;
                    for j in o..o + L2 {
                        let w = &net.w2[j * 2 * H..(j + 1) * 2 * H];
                        let expected = dot_scalar(w, &x);
                        assert_eq!(dot(w, &x).to_bits(), expected.to_bits());
                        let z = net.b2[j] + expected;
                        reference += net.w3[j] * z.clamp(0.0, 1.0);
                    }
                    assert_eq!(net.forward_raw(&acc, stm).to_bits(), reference.to_bits());
                }
                let moves = pos.legal_moves();
                if moves.is_empty() {
                    pos = Position::startpos();
                } else {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    pos.do_move(moves[(rng % moves.len() as u64) as usize]);
                }
            }
        }
    }

    #[test]
    fn simd_activation_boundaries_match_scalar() {
        let net = Network::random(91);
        let boundaries = [f32::NEG_INFINITY, -1.0, -0.0, 0.0, 0.5, 1.0, 2.0, f32::INFINITY];
        let mut acc = Acc::default();
        for persp in 0..2 {
            for i in 0..H { acc.v[persp][i] = boundaries[(i + persp) % boundaries.len()]; }
        }
        for stm in 0..2 {
            let mut x = [0f32; 2 * H];
            for i in 0..H {
                x[i] = acc.v[stm][i].clamp(0.0, 1.0);
                x[H + i] = acc.v[stm ^ 1][i].clamp(0.0, 1.0);
            }
            let mut expected = net.b3;
            let o = bucket_of(acc.npc) * L2;
            for j in o..o + L2 {
                let z = net.b2[j] + dot_scalar(&net.w2[j * 2 * H..(j + 1) * 2 * H], &x);
                expected += net.w3[j] * z.clamp(0.0, 1.0);
            }
            assert_eq!(net.forward_raw(&acc, stm).to_bits(), expected.to_bits());
        }
    }

    /// 手を指すたびの差分更新が、局面から作り直した結果と一致するか（ランダムな対局で確かめる）
    #[test]
    fn incremental_matches_refresh() {
        let net = Network::random(7);
        let mut rng = 88172645463325252u64;
        let (mut caps, mut drops, mut promos, mut kings) = (0, 0, 0, 0);
        for _game in 0..80 {
            let mut pos = Position::startpos();
            let mut acc = Acc::default();
            net.refresh(&pos, &mut acc);
            for ply in 0..250 {
                let moves = pos.legal_moves();
                if moves.is_empty() {
                    break;
                }
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let m = moves[(rng % moves.len() as u64) as usize];
                if mv_is_drop(m) {
                    drops += 1;
                } else {
                    if pos.board[mv_to(m)] != 0 {
                        caps += 1;
                    }
                    if mv_is_promo(m) {
                        promos += 1;
                    }
                    if type_of(pos.board[mv_from(m)]) == KING {
                        kings += 1;
                    }
                }
                let cap = if mv_is_drop(m) { 0 } else { pos.board[mv_to(m)] };
                pos.do_move(m);
                let mut next = Acc::default();
                net.apply_after(&acc, &pos, m, cap, &mut next);
                let mut fresh = Acc::default();
                net.refresh(&pos, &mut fresh);
                assert_eq!(next.mat, fresh.mat, "駒割りが合わない（{} 手目）", ply);
                assert_eq!(next.npc, fresh.npc, "盤上の駒の数が合わない（{} 手目）", ply);
                for p in 0..2 {
                    for i in 0..H {
                        assert!(
                            (next.v[p][i] - fresh.v[p][i]).abs() < 1e-3,
                            "アキュムレータが合わない（{} 手目, 視点 {}, 要素 {}）",
                            ply,
                            p,
                            i
                        );
                    }
                }
                acc = next;
            }
        }
        assert!(
            caps > 100 && drops > 100 && promos > 20 && kings > 50,
            "テストの対局に駒取り・打ち・成り・玉の移動が足りない: {} {} {} {}",
            caps,
            drops,
            promos,
            kings
        );
    }

    /// 左右反転した局面の特徴は、特徴番号を反転したものと一致するか
    #[test]
    fn mirror_matches_mirrored_position() {
        let mut rng = 12345u64;
        for _ in 0..30 {
            let mut pos = Position::startpos();
            for _ in 0..60 {
                let moves = pos.legal_moves();
                if moves.is_empty() {
                    break;
                }
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                pos.do_move(moves[(rng % moves.len() as u64) as usize]);
            }
            // 筋を反転した盤を作る
            let mut board = [0u8; 81];
            for sq in 0..81 {
                board[(8 - sq / 9) * 9 + sq % 9] = pos.board[sq];
            }
            let kings = [
                (8 - pos.king[0] / 9) * 9 + pos.king[0] % 9,
                (8 - pos.king[1] / 9) * 9 + pos.king[1] % 9,
            ];
            for persp in 0..2 {
                let mut a = Vec::new();
                features(&pos, persp, &mut a);
                let mut mapped: Vec<usize> = a.iter().map(|&f| mirror_feature(f as usize)).collect();
                let mut b = Vec::new();
                features_arrays(&board, &pos.hand, kings, persp, &mut b);
                let mut b: Vec<usize> = b.iter().map(|&f| f as usize).collect();
                mapped.sort();
                b.sort();
                assert_eq!(mapped, b, "左右反転した局面の特徴が一致しない");
            }
        }
    }

    /// 飽和演算のスカラーと AVX2 が、あふれる値でも同じ結果になる
    #[test]
    fn quant_update_saturates_like_avx2() {
        let mut a = [0i16; H];
        let mut row = [0i16; H];
        for i in 0..H {
            a[i] = [32000, -32000, 100, -100][i % 4];
            row[i] = [1000, 1000, -1000, 30000][i % 4];
        }
        for plus in [true, false] {
            let (mut s, mut v) = (a, a);
            q_update_scalar(&mut s, &row, plus);
            q_update(&mut v, &row, plus);
            assert_eq!(s, v);
        }
        let mut s = a;
        q_update_scalar(&mut s, &row, true);
        assert_eq!(s[0], i16::MAX);
    }

    /// テストに使うネット：乱数のネットと、形が合えば内蔵のネット
    fn test_nets(seed: u64) -> Vec<Network> {
        let mut v = vec![Network::random(seed)];
        if let Ok(n) = Network::from_bytes(DEFAULT_NET) { v.push(n); }
        v
    }

    /// HalfKP の特徴：1局面の特徴の数・範囲・重なり（同じ特徴が2回立たない）
    #[test]
    fn features_are_in_range_and_unique() {
        for (pos, _, _) in random_positions(20, 150, 99) {
            for persp in 0..2 {
                let mut f = Vec::new();
                features(&pos, persp, &mut f);
                assert!(f.iter().all(|&x| (x as usize) < N_FEATS));
                let mut s = f.clone();
                s.sort();
                s.dedup();
                assert_eq!(s.len(), f.len(), "同じ特徴が2回立っている");
                for &x in &f {
                    assert_eq!(mirror_feature(mirror_feature(x as usize)), x as usize);
                    if let Some(p) = factor_of(x as usize) {
                        assert!(p < N_FACTOR);
                        assert_eq!(mirror_factor(p), factor_of(mirror_feature(x as usize)).unwrap());
                    }
                }
            }
        }
    }

    /// ランダムな対局の局面を返す（整数化のテスト用）
    fn random_positions(n_games: usize, plies: usize, seed: u64) -> Vec<(Position, Move, u8)> {
        let mut rng = seed;
        let mut out = vec![];
        for _ in 0..n_games {
            let mut pos = Position::startpos();
            for _ in 0..plies {
                let moves = pos.legal_moves();
                if moves.is_empty() { break; }
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let m = moves[(rng % moves.len() as u64) as usize];
                let cap = if mv_is_drop(m) { 0 } else { pos.board[mv_to(m)] };
                pos.do_move(m);
                out.push((pos.clone(), m, cap));
            }
        }
        out
    }

    /// 整数化したネットでも、差分更新が作り直した結果と完全に一致し、AVX2 とスカラーの出力が完全に一致する
    #[test]
    fn quant_incremental_and_simd_are_exact() {
        for mut net in test_nets(5) {
            net.set_quant(true);
            let q = net.q.as_ref().unwrap();
            let mut rng = 4242u64;
            for _game in 0..40 {
                let mut pos = Position::startpos();
                let mut acc = Acc::default();
                net.refresh(&pos, &mut acc);
                for _ in 0..200 {
                    let moves = pos.legal_moves();
                    if moves.is_empty() { break; }
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    let m = moves[(rng % moves.len() as u64) as usize];
                    let cap = if mv_is_drop(m) { 0 } else { pos.board[mv_to(m)] };
                    pos.do_move(m);
                    let mut next = Acc::default();
                    net.apply_after(&acc, &pos, m, cap, &mut next);
                    let mut fresh = Acc::default();
                    net.refresh(&pos, &mut fresh);
                    assert_eq!(next.q, fresh.q, "整数のアキュムレータが合わない");
                    assert_eq!(next.mat, fresh.mat);
                    for stm in 0..2 {
                        let scalar = forward_q_scalar(q, &next, stm);
                        assert_eq!(net.forward_raw(&next, stm).to_bits(), scalar.to_bits(), "AVX2 とスカラーが合わない");
                    }
                    acc = next;
                }
            }
        }
    }

    /// V13：まとめた差分更新は、1行ずつの差分更新と（飽和するときも）完全に同じ値になる
    #[test]
    fn fused_quant_update_matches_row_by_row() {
        for mut net in test_nets(7) {
            // 倍率を大きくして、アキュムレータが飽和する場面も作る
            for s1 in [Q_S1, 16384] {
                net.set_quant_scales(true, s1, Q_S2);
                let q = net.q.as_ref().unwrap();
                let mut rng = 777u64;
                for _game in 0..30 {
                    let mut pos = Position::startpos();
                    let mut acc = Acc::default();
                    net.refresh(&pos, &mut acc);
                    for _ in 0..200 {
                        let moves = pos.legal_moves();
                        if moves.is_empty() { break; }
                        rng ^= rng << 13;
                        rng ^= rng >> 7;
                        rng ^= rng << 17;
                        let m = moves[(rng % moves.len() as u64) as usize];
                        let cap = if mv_is_drop(m) { 0 } else { pos.board[mv_to(m)] };
                        pos.do_move(m);
                        let mut next = Acc::default();
                        if !mv_is_drop(m) && type_of(pos.board[mv_to(m)]) == KING {
                            net.refresh(&pos, &mut next);
                        } else {
                            let mut rows = Acc::default();
                            net.apply_after_rows(&acc, &pos, m, cap, &mut rows);
                            Network::apply_after_q(q, &acc, &pos, m, cap, &mut next);
                            assert_eq!(next.q, rows.q, "まとめた差分更新が1行ずつと合わない");
                            assert_eq!(next.mat, rows.mat);
                        }
                        acc = next;
                    }
                }
            }
        }
    }

    /// 実験 速さ6 V25・V28：控えた行の一覧での更新と、玉が動いた手の差分更新は、作り直しと同じ値になる
    #[test]
    fn ops_and_king_incremental_match_refresh() {
        for mut net in test_nets(9) {
            net.set_quant(true);
            let q = net.q.as_ref().unwrap();
            let mut rng = 20261114u64;
            let (mut kings, mut others) = (0, 0);
            for _game in 0..40 {
                let mut pos = Position::startpos();
                let mut acc = Acc::default();
                net.refresh(&pos, &mut acc);
                for _ in 0..250 {
                    let moves = pos.legal_moves();
                    if moves.is_empty() { break; }
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    // 玉の手をよく選ぶ（3回に1回は、玉の手があればその中から）
                    let km: Vec<Move> = moves.iter().copied().filter(|&m| !mv_is_drop(m) && type_of(pos.board[mv_from(m)]) == KING).collect();
                    let m = if rng % 3 == 0 && !km.is_empty() { km[((rng >> 8) % km.len() as u64) as usize] } else { moves[((rng >> 8) % moves.len() as u64) as usize] };
                    let cap = if mv_is_drop(m) { 0 } else { pos.board[mv_to(m)] };
                    pos.do_move(m);
                    let mut fresh = Acc::default();
                    net.refresh(&pos, &mut fresh);
                    let mut next = Acc::default();
                    if !mv_is_drop(m) && type_of(pos.board[mv_to(m)]) == KING {
                        if KING_INC {
                            Network::apply_king_move(q, &acc, &pos, m, cap, &mut next);
                            assert_eq!(next.q, fresh.q, "玉が動いた手の差分更新が作り直しと合わない");
                            assert_eq!((next.mat, next.npc), (fresh.mat, fresh.npc));
                            kings += 1;
                        }
                    } else {
                        let ops = net.prepare_ops(&pos, m, cap);
                        assert_eq!(ops.kind, ACC_OPS_DIFF);
                        net.apply_ops(&acc, &ops, &mut next);
                        assert_eq!(next.q, fresh.q, "行の一覧での更新が作り直しと合わない");
                        for persp in 0..2 {
                            let mut rows = [0i16; H];
                            q_apply_ops_rows(q, &acc.q[persp], &mut rows, &ops.f[persp][..ops.n[persp] as usize]);
                            assert_eq!(rows, next.q[persp], "1行ずつの道（AVX2 のない CPU）が合わない");
                        }
                        assert_eq!((next.mat, next.npc), (fresh.mat, fresh.npc));
                        others += 1;
                    }
                    acc = fresh;
                }
            }
            assert!(others > 1000 && (!KING_INC || kings > 300), "確かめた手が少ない（玉 {}、ほか {}）", kings, others);
        }
    }

    /// 整数化しても、標準のネットの評価値は f32 とほとんど変わらない（平均 2cp 以下）
    #[test]
    fn quant_error_is_small() {
        // 学習済みのネットで測る（halfkp のビルドでは内蔵のネットの形が合わないので測らない）
        let Ok(f) = Network::from_bytes(DEFAULT_NET) else { return };
        let mut q = f.clone();
        q.set_quant(true);
        let (mut sum, mut max, mut n) = (0f64, 0f32, 0usize);
        for (pos, _, _) in random_positions(60, 150, 777) {
            let (mut a, mut b) = (Acc::default(), Acc::default());
            f.refresh(&pos, &mut a);
            q.refresh(&pos, &mut b);
            for stm in 0..2 {
                let d = (f.forward_raw(&a, stm) - q.forward_raw(&b, stm)).abs() * OUT_SCALE;
                sum += d as f64;
                max = max.max(d);
                n += 1;
            }
        }
        let mean = sum / n as f64;
        assert!(mean <= 2.0, "整数化の誤差が大きい: 平均 {:.3}cp 最大 {:.3}cp", mean, max);
    }

    #[test]
    fn save_load_roundtrip() {
        let net = Network::random(3);
        let back = Network::from_bytes(&net.to_bytes()).unwrap();
        let pos = Position::startpos();
        let (mut a, mut b) = (Acc::default(), Acc::default());
        net.refresh(&pos, &mut a);
        back.refresh(&pos, &mut b);
        assert_eq!(net.forward(&a, &pos), back.forward(&b, &pos));
    }
}
