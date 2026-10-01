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

/// 第 1 層（アキュムレータ）の幅
pub const H: usize = 64;
/// 第 2 層の幅
pub const L2: usize = 32;
/// 盤上の駒の特徴の数: 持ち主(手番側/相手側) × 駒種 14 × 升 81
const N_PS: usize = 2 * 14 * 81;
/// 持ち駒の種類ごとの最大枚数（歩 香 桂 銀 角 飛 金）
const HAND_CAP: [usize; 8] = [0, 18, 4, 4, 4, 2, 2, 4];
/// 持ち駒の特徴の開始位置（駒種ごと）
const HAND_START: [usize; 8] = [0, 0, 18, 22, 26, 30, 32, 34];
/// 片方の持ち主の持ち駒の特徴の数
const HAND_PER_OWNER: usize = 38;
const N_HAND: usize = 2 * HAND_PER_OWNER;
/// 玉からの相対位置の特徴: クラス（玉を除く駒種 13 × 持ち主 2）× 相対位置（17 × 17）
const KR_CLASSES: usize = 26;
const KR_OFFS: usize = 17 * 17;
const KR_BLOCK: usize = KR_CLASSES * KR_OFFS;
const KR_BASE: usize = N_PS + N_HAND;
/// 特徴の総数（自玉基準のブロックと敵玉基準のブロックがある）
pub const N_FEATS: usize = KR_BASE + 2 * KR_BLOCK;
/// 出力 1.0 が何 cp にあたるか
pub const OUT_SCALE: f32 = 400.0;

const MAGIC: &[u8; 4] = b"SFNN";
const VERSION: u32 = 3;
/// ファイルの先頭（マジック 4 + バージョン・H・L2・特徴数・土台の種類 各 4）
const HEADER_LEN: usize = 4 + 5 * 4;

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

/// 玉 king_sq からの相対位置の特徴番号（enemy_ref が true なら敵玉基準のブロック）
#[inline]
fn kr_feat(persp: usize, color: usize, pt: u8, sq: usize, king_sq: usize, enemy_ref: bool) -> usize {
    let rel_color = (color != persp) as usize;
    let kt = pt as usize - 1 - (pt > KING) as usize; // 玉を除いた駒種 0..12
    let class = rel_color * 13 + kt;
    let (pf, pr) = rel_coords(persp, sq);
    let (kf, kr) = rel_coords(persp, king_sq);
    let off = ((pf - kf + 8) * 17 + (pr - kr + 8)) as usize;
    KR_BASE + (enemy_ref as usize) * KR_BLOCK + class * KR_OFFS + off
}

/// 視点 persp から見た、持ち駒の特徴番号（k は 1 始まりの枚数目）
#[inline]
pub fn hand_feat(persp: usize, owner: usize, pt: u8, k: usize) -> usize {
    let rel_owner = (owner != persp) as usize;
    N_PS + rel_owner * HAND_PER_OWNER + HAND_START[pt as usize] + (k - 1)
}

/// 盤上の駒 1 つの特徴を out に書く（個数を返す）。玉は升の特徴だけ。
#[inline]
fn piece_feats(persp: usize, color: usize, pt: u8, sq: usize, kings: [usize; 2], out: &mut [usize; 3]) -> usize {
    out[0] = ps_feat(persp, color, pt, sq);
    let mut n = 1;
    if pt != KING {
        if kings[persp] != NO_SQ {
            out[n] = kr_feat(persp, color, pt, sq, kings[persp], false);
            n += 1;
        }
        if kings[persp ^ 1] != NO_SQ {
            out[n] = kr_feat(persp, color, pt, sq, kings[persp ^ 1], true);
            n += 1;
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
        let (dx, dy) = (off / 17, off % 17);
        KR_BASE + blk * KR_BLOCK + cls * KR_OFFS + (16 - dx) * 17 + dy
    }
}

/// 盤面の配列から、視点 persp の特徴をすべて out に追加する
pub fn features_arrays(board: &[u8; 81], hand: &[[u8; 8]; 2], kings: [usize; 2], persp: usize, out: &mut Vec<u16>) {
    let mut f = [0usize; 3];
    for sq in 0..81 {
        let p = board[sq];
        if p != 0 {
            let n = piece_feats(persp, color_of(p), type_of(p), sq, kings, &mut f);
            for &i in &f[..n] {
                out.push(i as u16);
            }
        }
    }
    for owner in 0..2 {
        for pt in 1..8u8 {
            let n = (hand[owner][pt as usize] as usize).min(HAND_CAP[pt as usize]);
            for k in 1..=n {
                out.push(hand_feat(persp, owner, pt, k) as u16);
            }
        }
    }
}

/// 局面の特徴をすべて out に追加する（視点 persp から見たもの）
pub fn features(pos: &Position, persp: usize, out: &mut Vec<u16>) {
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
    /// 先手から見た駒割り
    pub mat: i32,
}

impl Default for Acc {
    fn default() -> Self {
        Acc { v: [[0.0; H]; 2], mat: 0 }
    }
}

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
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
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
            w2: (0..L2 * 2 * H).map(|_| rnd() * 0.12).collect(),
            b2: vec![0.5; L2],
            w3: (0..L2).map(|_| rnd() * 0.1).collect(),
            b3: 0.0,
            base_hc: false,
        }
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
    }

    /// 局面から最初のアキュムレータを作る
    pub fn refresh(&self, pos: &Position, acc: &mut Acc) {
        self.refresh_arrays(&pos.board, &pos.hand, pos.king, acc);
    }

    #[inline]
    fn add(&self, acc: &mut Acc, persp: usize, f: usize) {
        let row = self.row(f);
        let v = &mut acc.v[persp];
        for i in 0..H {
            v[i] += row[i];
        }
    }

    #[inline]
    fn sub(&self, acc: &mut Acc, persp: usize, f: usize) {
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

    /// 手 m を指す前の局面 pos とそのアキュムレータ from から、指したあとのアキュムレータ to を作る
    pub fn apply(&self, from: &Acc, pos: &Position, m: Move, to: &mut Acc) {
        let us = pos.side;
        let dst = mv_to(m);
        // 玉が動くと、玉からの相対位置の特徴がすべて変わるので、作り直す（玉の移動は少ない）
        if !mv_is_drop(m) && type_of(pos.board[mv_from(m)]) == KING {
            let mut board = pos.board;
            let mut hand = pos.hand;
            let cap = board[dst];
            if cap != 0 {
                hand[us][unpromote(type_of(cap)) as usize] += 1;
            }
            board[dst] = board[mv_from(m)];
            board[mv_from(m)] = 0;
            let mut kings = pos.king;
            kings[us] = dst;
            self.refresh_arrays(&board, &hand, kings, to);
            return;
        }
        to.clone_from(from);
        let kings = pos.king;
        let sign = if us == BLACK { 1 } else { -1 };
        if mv_is_drop(m) {
            let pt = mv_drop_pt(m);
            let n = (pos.hand[us][pt as usize] as usize).min(HAND_CAP[pt as usize]);
            if n >= 1 {
                for persp in 0..2 {
                    self.sub(to, persp, hand_feat(persp, us, pt, n));
                }
            }
            self.piece(to, true, us, pt, dst, kings);
            to.mat += sign * (PIECE_VALUE[pt as usize] - HAND_VALUE[pt as usize]);
        } else {
            let src = mv_from(m);
            let p = pos.board[src];
            let pt = type_of(p);
            let npt = if mv_is_promo(m) { pt + PROMOTE } else { pt };
            self.piece(to, false, us, pt, src, kings);
            self.piece(to, true, us, npt, dst, kings);
            to.mat += sign * (PIECE_VALUE[npt as usize] - PIECE_VALUE[pt as usize]);
            let cap = pos.board[dst];
            if cap != 0 {
                let cpt = type_of(cap);
                self.piece(to, false, color_of(cap), cpt, dst, kings);
                let hpt = unpromote(cpt);
                let n = (pos.hand[us][hpt as usize] as usize + 1).min(HAND_CAP[hpt as usize]);
                for persp in 0..2 {
                    self.add(to, persp, hand_feat(persp, us, hpt, n));
                }
                to.mat += sign * (HAND_VALUE[hpt as usize] + PIECE_VALUE[cpt as usize]);
            }
        }
    }

    /// 局面 pos（手番側から見た）の評価値（cp）。acc は pos に対応するアキュムレータ。
    pub fn forward(&self, acc: &Acc, pos: &Position) -> i32 {
        let stm = pos.side;
        let mut x = [0f32; 2 * H];
        for i in 0..H {
            x[i] = acc.v[stm][i].clamp(0.0, 1.0);
            x[H + i] = acc.v[stm ^ 1][i].clamp(0.0, 1.0);
        }
        let mut out = self.b3;
        for j in 0..L2 {
            let z = self.b2[j] + dot(&self.w2[j * 2 * H..(j + 1) * 2 * H], &x);
            out += self.w3[j] * z.clamp(0.0, 1.0);
        }
        let base = if self.base_hc {
            evaluate(pos)
        } else if stm == BLACK {
            acc.mat
        } else {
            -acc.mat
        };
        ((base as f32 + out * OUT_SCALE) as i32).clamp(-20000, 20000)
    }

    // ---- ファイルへの保存・読み込み ----

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(MAGIC);
        for v in [VERSION, H as u32, L2 as u32, N_FEATS as u32, self.base_hc as u32] {
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
        if u(4) != VERSION || u(8) as usize != H || u(12) as usize != L2 || u(16) as usize != N_FEATS {
            return Err("ネットワークの形（バージョン・幅）がこのエンジンと合いません".into());
        }
        let base_hc = u(20) != 0;
        let total = N_FEATS * H + H + L2 * 2 * H + L2 + L2 + 1;
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
        let w2 = take(L2 * 2 * H);
        let b2 = take(L2);
        let w3 = take(L2);
        let b3 = take(1)[0];
        Ok(Network { w1, b1, w2, b2, w3, b3, base_hc })
    }
}

// ---- 読み込み済みのネットワーク（エンジン全体で 1 つ） ----

static GLOBAL: RwLock<Option<Arc<Network>>> = RwLock::new(None);

/// 読み込み済みのネットワークを返す
pub fn global() -> Option<Arc<Network>> {
    GLOBAL.read().ok().and_then(|g| g.clone())
}

/// ファイルからネットワークを読み込んで、エンジン全体で使うようにする
pub fn load_global(path: &str) -> Result<(), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {}", path, e))?;
    let net = Network::from_bytes(&bytes)?;
    if let Ok(mut g) = GLOBAL.write() {
        *g = Some(Arc::new(net));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
                let mut next = Acc::default();
                net.apply(&acc, &pos, m, &mut next);
                pos.do_move(m);
                let mut fresh = Acc::default();
                net.refresh(&pos, &mut fresh);
                assert_eq!(next.mat, fresh.mat, "駒割りが合わない（{} 手目）", ply);
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
