//! 診断用の外部評価関数（実験 diagSE、cargo の機能 ext_eval）。
//! やねうら王形式の NNUE（HalfKP(Friend) 256x2-32-32）のファイルを読み、Sailfish の評価をまるごと置き換える。
//! 探索と評価のどちらが足りないかを切り分けるためだけに使う（公開しない、学習に使わない）。
//! 速さは問わないので、差分更新はせず局面ごとに全計算する（比べるときは読む局面数をそろえる）。

use crate::position::Position;
use crate::types::*;
use std::sync::OnceLock;

/// 盤上の駒の番号（BonaPiece）の始まり（自分の駒 f、相手の駒 e）。成った小駒は金と同じ。
const BOARD_F: [usize; 15] = [0, 90, 252, 414, 576, 900, 1224, 738, 0, 738, 738, 738, 738, 1062, 1386];
/// 持ち駒の番号の始まり（歩・香・桂・銀・角・飛・金）。n 枚持っていれば始まり+0〜n-1 がすべて立つ
const HAND_F: [usize; 8] = [0, 1, 39, 49, 59, 79, 85, 69];
const HAND_E: [usize; 8] = [0, 20, 44, 54, 64, 82, 88, 74];
const FE_END: usize = 1548;
const N_IN: usize = 81 * FE_END;
const HALF: usize = 256;
const VERSION: u32 = 0x7AF3_2F16;
const FV_SCALE: i32 = 24;

pub struct ExtNet {
    ft_b: Vec<i16>,
    ft_w: Vec<i16>,
    l1_b: Vec<i32>,
    l1_w: Vec<i8>,
    l2_b: Vec<i32>,
    l2_w: Vec<i8>,
    out_b: i32,
    out_w: Vec<i8>,
}

struct Reader<'a> {
    d: &'a [u8],
    p: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], String> {
        let s = self.d.get(self.p..self.p + n).ok_or("ファイルが途中で終わっている")?;
        self.p += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i16s(&mut self, n: usize) -> Result<Vec<i16>, String> {
        Ok(self.take(2 * n)?.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect())
    }
    fn i32s(&mut self, n: usize) -> Result<Vec<i32>, String> {
        Ok(self.take(4 * n)?.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect())
    }
    fn i8s(&mut self, n: usize) -> Result<Vec<i8>, String> {
        Ok(self.take(n)?.iter().map(|&b| b as i8).collect())
    }
}

impl ExtNet {
    pub fn load(path: &str) -> Result<ExtNet, String> {
        let d = std::fs::read(path).map_err(|e| e.to_string())?;
        let mut r = Reader { d: &d, p: 0 };
        if r.u32()? != VERSION {
            return Err("やねうら王形式の NNUE ではない".into());
        }
        r.u32()?; // 構造のハッシュ
        let len = r.u32()? as usize;
        let arch = String::from_utf8_lossy(r.take(len)?).to_string();
        if !arch.contains("HalfKP(Friend)[125388->256x2]") || !arch.contains("AffineTransform[32<-512]") {
            return Err(format!("対応していない構造: {}", arch));
        }
        r.u32()?;
        let ft_b = r.i16s(HALF)?;
        let ft_w = r.i16s(HALF * N_IN)?;
        r.u32()?;
        let l1_b = r.i32s(32)?;
        let l1_w = r.i8s(32 * 512)?;
        let l2_b = r.i32s(32)?;
        let l2_w = r.i8s(32 * 32)?;
        let out_b = r.i32s(1)?[0];
        let out_w = r.i8s(32)?;
        if r.p != d.len() {
            return Err(format!("大きさが合わない（読んだ {} / ファイル {}）", r.p, d.len()));
        }
        Ok(ExtNet { ft_b, ft_w, l1_b, l1_w, l2_b, l2_w, out_b, out_w })
    }

    /// 視点 persp から見た、立っている特徴の番号（玉の升 × BonaPiece）
    fn features(pos: &Position, persp: usize, out: &mut Vec<usize>) {
        out.clear();
        let inv = |sq: usize| if persp == BLACK { sq } else { 80 - sq };
        let base = inv(pos.king[persp]) * FE_END;
        for sq in 0..81 {
            let p = pos.board[sq];
            if p == 0 || type_of(p) == KING {
                continue;
            }
            let f = BOARD_F[type_of(p) as usize] + if color_of(p) == persp { 0 } else { 81 };
            out.push(base + f + inv(sq));
        }
        for c in [BLACK, WHITE] {
            for pt in PAWN as usize..=GOLD as usize {
                let s = if c == persp { HAND_F[pt] } else { HAND_E[pt] };
                for i in 0..pos.hand[c][pt] as usize {
                    out.push(base + s + i);
                }
            }
        }
    }

    /// 手番側から見た評価値
    pub fn evaluate(&self, pos: &Position) -> i32 {
        let mut input = [0u8; 2 * HALF];
        let mut feats = Vec::with_capacity(40);
        for (k, persp) in [pos.side, pos.side ^ 1].into_iter().enumerate() {
            Self::features(pos, persp, &mut feats);
            let mut acc = [0i16; HALF];
            acc.copy_from_slice(&self.ft_b);
            for &f in &feats {
                let row = &self.ft_w[f * HALF..(f + 1) * HALF];
                for j in 0..HALF {
                    acc[j] = acc[j].wrapping_add(row[j]);
                }
            }
            for j in 0..HALF {
                input[k * HALF + j] = acc[j].clamp(0, 127) as u8;
            }
        }
        let mut h1 = [0u8; 32];
        for o in 0..32 {
            let w = &self.l1_w[o * 512..(o + 1) * 512];
            let s = self.l1_b[o] + w.iter().zip(input.iter()).map(|(&a, &b)| a as i32 * b as i32).sum::<i32>();
            h1[o] = (s >> 6).clamp(0, 127) as u8;
        }
        let mut h2 = [0u8; 32];
        for o in 0..32 {
            let w = &self.l2_w[o * 32..(o + 1) * 32];
            let s = self.l2_b[o] + w.iter().zip(h1.iter()).map(|(&a, &b)| a as i32 * b as i32).sum::<i32>();
            h2[o] = (s >> 6).clamp(0, 127) as u8;
        }
        let out = self.out_b + self.out_w.iter().zip(h2.iter()).map(|(&a, &b)| a as i32 * b as i32).sum::<i32>();
        out / FV_SCALE
    }
}

/// 読み込みは起動後に一度だけ（探索中に読み直さないので、毎回の評価で鍵を取らずに済む）
static GLOBAL: OnceLock<ExtNet> = OnceLock::new();

#[inline]
pub fn global() -> Option<&'static ExtNet> {
    GLOBAL.get()
}

pub fn load_global(path: &str) -> Result<(), String> {
    if GLOBAL.get().is_some() {
        return Err("外部の評価関数は一度だけ読み込める".into());
    }
    let net = ExtNet::load(path)?;
    let _ = GLOBAL.set(net);
    Ok(())
}
