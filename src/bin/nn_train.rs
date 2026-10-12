//! ニューラルネットワーク評価の学習ツール。
//!
//! 使い方: nn_train <出力.nn> <エポック数> <λ> <棋譜ファイル>... [--init 既存.nn] [--lr 0.001] [--batch 16384] [--k 600] [--teacher-k 600] [--nomirror] [--quiet-drops] [--skip-opening] [--dedup]
//!
//! 棋譜（tools/datagen.mjs の出力）の静かな局面から、次のように学習する。
//!   予測勝率 p = σ(評価値 / K)       （評価値 = 駒割り + ネットワークの出力 × OUT_SCALE）
//!   正解     y = λ·σ(そのときの探索の評価値 / K) + (1-λ)·対局結果
//! 損失はクロスエントロピー。Adam で最適化し、検証用の棋譜（最後の 5%）で手作りの評価関数と比べる。
//! 将棋は盤を左右に反転しても同じ形勢なので、学習のときは半分の局面を左右反転して使う（データの水増し）。
//!
//! --psv <train.psv> <validation.psv>（psv_prepare の出力）: 全局面をメモリに載せず、区切りごとに読んで学習する。
//!   [--chunk 2000000] [--val-max 1000000] [--val-every 5] [--train-limit N] [--resume]（詳しくは train_psv）

use sailfish2::eval::evaluate;
use sailfish2::nn::*;
use sailfish2::position::Position;
use sailfish2::psv::{self, RECORD_BYTES};
use sailfish2::types::*;
use sailfish2::training::*;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::collections::HashSet;
use std::sync::OnceLock;
use std::thread;
use std::time::Instant;

struct Data {
    fa: Vec<u32>,    // 手番側の視点の特徴
    fb: Vec<u32>,    // 相手側の視点の特徴
    off: Vec<u32>,   // 特徴の開始位置（長さ n+1。fa と fb で共通）
    mat: Vec<f32>,   // 手番側から見た駒割り
    hc: Vec<f32>,    // 手作りの評価関数（手番側から見た値）
    score: Vec<f32>, // 探索の評価値（手番側から見た値）
    res: Vec<f32>,   // 手番側から見た結果 1 / 0.5 / 0
}

impl Data {
    fn new() -> Data {
        Data { fa: vec![], fb: vec![], off: vec![0], mat: vec![], hc: vec![], score: vec![], res: vec![] }
    }
    fn len(&self) -> usize {
        self.mat.len()
    }
    /// 1局面を加える（load_positions と同じ値の作り方）。score・res は手番側から見た値
    fn push(&mut self, pos: &Position, base_hc: bool, score: f32, res: f32, tmp: &mut Vec<u32>) {
        let stm = pos.side;
        tmp.clear();
        features(pos, stm, tmp);
        self.fa.extend_from_slice(tmp);
        tmp.clear();
        features(pos, stm ^ 1, tmp);
        self.fb.extend_from_slice(tmp);
        self.off.push(self.fa.len() as u32);
        let mat = material(pos) as f32;
        let hc = evaluate(pos) as f32;
        self.mat.push(if base_hc { hc } else if stm == BLACK { mat } else { -mat });
        self.hc.push(hc);
        self.score.push(score);
        self.res.push(res);
    }
    fn append(&mut self, o: Data) {
        let base = *self.off.last().unwrap();
        self.fa.extend_from_slice(&o.fa);
        self.fb.extend_from_slice(&o.fb);
        self.off.extend(o.off[1..].iter().map(|&x| base + x));
        self.mat.extend_from_slice(&o.mat);
        self.hc.extend_from_slice(&o.hc);
        self.score.extend_from_slice(&o.score);
        self.res.extend_from_slice(&o.res);
    }
}

/// 左右反転した特徴番号の表
fn mirror_table() -> &'static [u32] {
    static T: OnceLock<Vec<u32>> = OnceLock::new();
    T.get_or_init(|| (0..N_FEATS).map(|f| mirror_feature(f) as u32).collect())
}

/// 棋譜を読み、静かな局面を集める。戻り値は (データ, 検証用に分ける最初のサンプル番号)
fn load(paths: &[String], base_hc: bool, filter: TrainingFilter, dedup: bool) -> (Data, usize) {
    let mut d = Data { fa: vec![], fb: vec![], off: vec![0], mat: vec![], hc: vec![], score: vec![], res: vec![] };
    let mut game_starts: Vec<usize> = vec![];
    let mut seen = HashSet::new();
    let mut tmp: Vec<u32> = Vec::with_capacity(64);
    for path in paths {
        let text = fs::read_to_string(path).unwrap_or_else(|_| panic!("{} を読めません", path));
        for line in text.lines() {
            let mut it = line.split_whitespace();
            if it.next() != Some("R") {
                continue;
            }
            let result: f32 = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.5);
            let open: usize = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
            game_starts.push(d.len());
            let mut pos = Position::startpos();
            for (ply, tok) in it.enumerate() {
                let Some((mv, sc)) = tok.rsplit_once(':') else { break };
                let Some(m) = pos.parse_usi_move(mv) else { break };
                let score: f32 = sc.parse().unwrap_or(0.0);
                // 静かな局面だけ使う: 次の手が駒取り・成りでなく、王手がかかっておらず、詰みや大差でないもの
                if eligible_sample(&mut pos, m, score, ply, open, filter)
                    && (!dedup || seen.insert((pos.board, pos.hand, pos.side))) {
                    let stm = pos.side;
                    tmp.clear();
                    features(&pos, stm, &mut tmp);
                    d.fa.extend_from_slice(&tmp);
                    tmp.clear();
                    features(&pos, stm ^ 1, &mut tmp);
                    d.fb.extend_from_slice(&tmp);
                    d.off.push(d.fa.len() as u32);
                    let mat = material(&pos) as f32;
                    let hc = evaluate(&pos) as f32;
                    // 土台の評価値: 手作りの評価関数、または駒割りだけ（どちらも手番側から見た値）
                    d.mat.push(if base_hc { hc } else if stm == BLACK { mat } else { -mat });
                    d.hc.push(hc);
                    d.score.push(score);
                    d.res.push(if stm == BLACK { result } else { 1.0 - result });
                }
                pos.do_move(m);
            }
        }
    }
    let games = game_starts.len();
    let val_start = game_starts.get(games * 95 / 100).copied().unwrap_or(d.len());
    println!("棋譜 {} 局、局面 {} 個（学習 {} / 検証 {}）", games, d.len(), val_start, d.len() - val_start);
    (d, val_start)
}


/// Independent-position teacher rows: P<TAB>SFEN<TAB>side score<TAB>side result<TAB>move<TAB>source ply.
/// Explicit separate files are required; producer groups positions and mirrors before splitting.
fn load_positions(paths: &[String], base_hc: bool, filter: TrainingFilter) -> (Data, usize) {
    assert_eq!(paths.len(), 2, "--positions requires train.tsv and validation.tsv");
    let mut d = Data { fa: vec![], fb: vec![], off: vec![0], mat: vec![], hc: vec![], score: vec![], res: vec![] };
    let mut seen = HashSet::new();
    let mut tmp = Vec::with_capacity(64);
    let mut val_start = 0;
    let mut accepted = [0usize; 2];
    for (file_idx, path) in paths.iter().enumerate() {
        if file_idx == 1 { val_start = d.len(); }
        let text = fs::read_to_string(path).expect("Cannot read position teacher");
        for line in text.lines().filter(|l| !l.is_empty()) {
            let cols: Vec<&str> = line.split('\t').collect();
            assert_eq!(cols.len(), 6, "Invalid position teacher columns");
            assert_eq!(cols[0], "P");
            let mut pos = Position::from_sfen(cols[1]).expect("Invalid teacher SFEN");
            let score: f32 = cols[2].parse().expect("Invalid teacher score");
            let result: f32 = cols[3].parse().expect("Invalid side-to-move result");
            assert!(score.is_finite() && result.is_finite() && (0.0..=1.0).contains(&result));
            let m = pos.parse_usi_move(cols[4]).expect("Invalid teacher move");
            assert!(pos.do_move_legal(m), "Illegal teacher move");
            pos.undo_move();
            assert!(seen.insert((pos.board, pos.hand, pos.side)), "Duplicate position across teacher files");
            if !eligible_sample(&mut pos, m, score, 0, 0, filter) { continue; }
            // These teacher results are already relative to the current side.
            d.push(&pos, base_hc, score, result, &mut tmp);
            accepted[file_idx] += 1;
        }
    }
    println!("Independent positions: train {} / validation {}", accepted[0], accepted[1]);
    (d, val_start)
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn ce(p: f32, y: f32) -> f64 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    -(y * p.ln() + (1.0 - y) * (1.0 - p).ln()) as f64
}

struct Grad {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    w3: Vec<f32>,
    b3: f32,
}

/// 学習中の第1層の行数：本来の特徴 N_FEATS ＋ 因数分解の特徴 N_FACTOR（halfkp のときだけ。保存のときに足し込む）
const W1_ROWS: usize = N_FEATS + N_FACTOR;
/// 第1層の勾配を、スレッドごとに全体で持たず、局面ごとの勾配から行を分担して足し込む（halfkp。第1層が大きいため）
const SPARSE_W1: bool = N_FACTOR > 0;

impl Grad {
    /// 勾配の入れ物。full = false なら第1層の重みの分を持たない（SPARSE_W1 のときのスレッドごとの入れ物）
    fn with_w1(full: bool) -> Grad {
        Grad {
            w1: if full { vec![0.0; W1_ROWS * H] } else { vec![] },
            b1: vec![0.0; H],
            w2: vec![0.0; L2 * 2 * H],
            b2: vec![0.0; L2],
            w3: vec![0.0; L2],
            b3: 0.0,
        }
    }
    fn new() -> Grad {
        Grad {
            w1: vec![0.0; W1_ROWS * H],
            b1: vec![0.0; H],
            w2: vec![0.0; L2 * 2 * H],
            b2: vec![0.0; L2],
            w3: vec![0.0; L2],
            b3: 0.0,
        }
    }
    fn clear(&mut self) {
        for v in [&mut self.w1, &mut self.b1, &mut self.w2, &mut self.b2, &mut self.w3] {
            v.iter_mut().for_each(|x| *x = 0.0);
        }
        self.b3 = 0.0;
    }
    fn add(&mut self, o: &Grad) {
        for (a, b) in [
            (&mut self.w1, &o.w1),
            (&mut self.b1, &o.b1),
            (&mut self.w2, &o.w2),
            (&mut self.b2, &o.b2),
            (&mut self.w3, &o.w3),
        ] {
            for i in 0..a.len() {
                a[i] += b[i];
            }
        }
        self.b3 += o.b3;
    }
}

/// 1 サンプルの順伝播の中間値
struct Fwd {
    a: [[f32; H]; 2],
    x: [f32; 2 * H],
    z2: [f32; L2],
    h: [f32; L2],
    out: f32,
}

#[inline]
fn feat(f: u32, flip: bool) -> usize {
    if flip { mirror_table()[f as usize] as usize } else { f as usize }
}

fn forward(net: &Network, d: &Data, i: usize, flip: bool) -> Fwd {
    let (o0, o1) = (d.off[i] as usize, d.off[i + 1] as usize);
    let mut a = [[0f32; H]; 2];
    a[0].copy_from_slice(&net.b1);
    a[1].copy_from_slice(&net.b1);
    for (side, feats) in [&d.fa[o0..o1], &d.fb[o0..o1]].into_iter().enumerate() {
        for &f in feats {
            let f = feat(f, flip);
            let row = &net.w1[f * H..f * H + H];
            for k in 0..H {
                a[side][k] += row[k];
            }
            // 因数分解の特徴（halfkp のときだけ。玉の位置によらない駒の重み）
            if let Some(p) = factor_of(f) {
                let r = N_FEATS + p;
                let row = &net.w1[r * H..r * H + H];
                for k in 0..H {
                    a[side][k] += row[k];
                }
            }
        }
    }
    let mut x = [0f32; 2 * H];
    for k in 0..H {
        x[k] = a[0][k].clamp(0.0, 1.0);
        x[H + k] = a[1][k].clamp(0.0, 1.0);
    }
    let mut z2 = [0f32; L2];
    let mut h = [0f32; L2];
    let mut out = net.b3;
    for j in 0..L2 {
        let row = &net.w2[j * 2 * H..(j + 1) * 2 * H];
        let mut s = [0f32; 8];
        for (cr, cx) in row.chunks_exact(8).zip(x.chunks_exact(8)) {
            for k in 0..8 {
                s[k] += cr[k] * cx[k];
            }
        }
        z2[j] = net.b2[j] + s.iter().sum::<f32>();
        h[j] = z2[j].clamp(0.0, 1.0);
        out += net.w3[j] * h[j];
    }
    Fwd { a, x, z2, h, out }
}

fn teacher_target(score: f32, result: f32, teacher_k: f32, lam: f32) -> f32 {
    lam * sigmoid(score / teacher_k) + (1.0 - lam) * result
}

/// 1 サンプルの損失と勾配（g に加算）。第1層の重みの勾配も g.w1 に足す
fn sample_grad(net: &Network, d: &Data, i: usize, k: f32, teacher_k: f32, lam: f32, flip: bool, g: &mut Grad) -> f64 {
    let mut ga = [[0f32; H]; 2];
    let loss = sample_grad_core(net, d, i, k, teacher_k, lam, flip, g, &mut ga);
    add_w1_rows(&mut g.w1, d, i, flip, &ga, |_| true);
    loss
}

/// 局面 i の特徴の行（因数分解の行を含む）に、視点ごとのアキュムレータの勾配 ga を足す。owns(行) が false の行は飛ばす
#[inline]
fn add_w1_rows(w1: &mut [f32], d: &Data, i: usize, flip: bool, ga: &[[f32; H]; 2], owns: impl Fn(usize) -> bool) {
    let (o0, o1) = (d.off[i] as usize, d.off[i + 1] as usize);
    for (side, feats) in [&d.fa[o0..o1], &d.fb[o0..o1]].into_iter().enumerate() {
        for &ft in feats {
            let ft = feat(ft, flip);
            for r in [Some(ft), factor_of(ft).map(|p| N_FEATS + p)].into_iter().flatten() {
                if !owns(r) { continue; }
                let row = &mut w1[r * H..r * H + H];
                for q in 0..H {
                    row[q] += ga[side][q];
                }
            }
        }
    }
}

/// 1 サンプルの損失と、第1層の重み以外の勾配（g に加算）。第1層のアキュムレータの勾配を ga に返す
fn sample_grad_core(net: &Network, d: &Data, i: usize, k: f32, teacher_k: f32, lam: f32, flip: bool, g: &mut Grad, ga: &mut [[f32; H]; 2]) -> f64 {
    let f = forward(net, d, i, flip);
    let eval = d.mat[i] + f.out * OUT_SCALE;
    let p = sigmoid(eval / k);
    let y = teacher_target(d.score[i], d.res[i], teacher_k, lam);
    let loss = ce(p, y);
    let g_out = (p - y) / k * OUT_SCALE;
    g.b3 += g_out;
    let mut gx = [0f32; 2 * H];
    for j in 0..L2 {
        g.w3[j] += g_out * f.h[j];
        let gh = g_out * net.w3[j];
        let gz = if f.z2[j] > 0.0 && f.z2[j] < 1.0 { gh } else { 0.0 };
        if gz != 0.0 {
            g.b2[j] += gz;
            let wrow = &net.w2[j * 2 * H..(j + 1) * 2 * H];
            let grow = &mut g.w2[j * 2 * H..(j + 1) * 2 * H];
            for q in 0..2 * H {
                grow[q] += gz * f.x[q];
                gx[q] += gz * wrow[q];
            }
        }
    }
    for p_ in 0..2 {
        for q in 0..H {
            let act = f.a[p_][q];
            ga[p_][q] = if act > 0.0 && act < 1.0 { gx[p_ * H + q] } else { 0.0 };
            g.b1[q] += ga[p_][q];
        }
    }
    loss
}

struct Adam {
    m: Grad,
    v: Grad,
    t: i32,
}

/// Adam の1要素の更新
#[inline]
fn adam_upd(w: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32], lr: f32, scale: f32, c1: f32, c2: f32) {
    let (b1, b2) = (0.9f32, 0.999f32);
    for i in 0..w.len() {
        let gi = g[i] * scale;
        m[i] = b1 * m[i] + (1.0 - b1) * gi;
        v[i] = b2 * v[i] + (1.0 - b2) * gi * gi;
        w[i] -= lr * (m[i] / c1) / ((v[i] / c2).sqrt() + 1e-8);
    }
}

/// 複数のスレッドで、重なりのない行を書き換えるための生ポインタ（行の分担は「行番号 % スレッド数」で決め、重ならない）
#[derive(Clone, Copy)]
struct RowPtr(*mut f32);
unsafe impl Send for RowPtr {}
unsafe impl Sync for RowPtr {}
impl RowPtr {
    /// SAFETY: 呼び出し側は、同じ行を同時に2つのスレッドから使わないこと
    unsafe fn row(self, r: usize) -> &'static mut [f32] {
        std::slice::from_raw_parts_mut(self.0.add(r * H), H)
    }
}

/// 第1層の勾配を「行を分担して」足し込み、勾配のあった行だけ Adam で更新する（SPARSE_W1 のとき）。
/// ga[n] はバッチの n 番目の局面（slice[n]、左右反転 flips[n]）のアキュムレータの勾配。足し込んだ行の勾配は 0 に戻す。
fn sparse_w1_step(net: &mut Network, opt: &mut Adam, gw1: &mut [f32], d: &Data, slice: &[u32], flips: &[bool], ga: &[[[f32; H]; 2]],
                  lr: f32, scale: f32, threads: usize) {
    let c1 = 1.0 - 0.9f32.powi(opt.t);
    let c2 = 1.0 - 0.999f32.powi(opt.t);
    let (w, g, m, v) = (RowPtr(net.w1.as_mut_ptr()), RowPtr(gw1.as_mut_ptr()), RowPtr(opt.m.w1.as_mut_ptr()), RowPtr(opt.v.w1.as_mut_ptr()));
    thread::scope(|s| {
        for t in 0..threads {
            s.spawn(move || {
                let mut touched: Vec<usize> = vec![];
                let mut seen = vec![false; W1_ROWS];
                for (n, &si) in slice.iter().enumerate() {
                    let i = si as usize;
                    let (o0, o1) = (d.off[i] as usize, d.off[i + 1] as usize);
                    for (side, feats) in [&d.fa[o0..o1], &d.fb[o0..o1]].into_iter().enumerate() {
                        for &ft in feats {
                            let ft = feat(ft, flips[n]);
                            for r in [Some(ft), factor_of(ft).map(|p| N_FEATS + p)].into_iter().flatten() {
                                if r % threads != t { continue; }
                                // SAFETY: 行 r はこのスレッドだけが扱う
                                let row = unsafe { g.row(r) };
                                for q in 0..H { row[q] += ga[n][side][q]; }
                                if !seen[r] { seen[r] = true; touched.push(r); }
                            }
                        }
                    }
                }
                for r in touched {
                    // SAFETY: 行 r はこのスレッドだけが扱う
                    unsafe {
                        adam_upd(w.row(r), g.row(r), m.row(r), v.row(r), lr, scale, c1, c2);
                        g.row(r).iter_mut().for_each(|x| *x = 0.0);
                    }
                }
            });
        }
    });
}

impl Adam {
    /// 1回の更新。SPARSE_W1 のときは第1層の重みを更新しない（sparse_w1_step で行う。t はここで進める）
    fn step(&mut self, net: &mut Network, g: &Grad, lr: f32, scale: f32) {
        self.t += 1;
        let (b1, b2) = (0.9f32, 0.999f32);
        let c1 = 1.0 - b1.powi(self.t);
        let c2 = 1.0 - b2.powi(self.t);
        let upd = |w: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32]| adam_upd(w, g, m, v, lr, scale, c1, c2);
        if !SPARSE_W1 {
            upd(&mut net.w1, &g.w1, &mut self.m.w1, &mut self.v.w1);
        }
        upd(&mut net.b1, &g.b1, &mut self.m.b1, &mut self.v.b1);
        upd(&mut net.w2, &g.w2, &mut self.m.w2, &mut self.v.w2);
        upd(&mut net.b2, &g.b2, &mut self.m.b2, &mut self.v.b2);
        upd(&mut net.w3, &g.w3, &mut self.m.w3, &mut self.v.w3);
        let gi = g.b3 * scale;
        self.m.b3 = b1 * self.m.b3 + (1.0 - b1) * gi;
        self.v.b3 = b2 * self.v.b3 + (1.0 - b2) * gi * gi;
        net.b3 -= lr * (self.m.b3 / c1) / ((self.v.b3 / c2).sqrt() + 1e-8);
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// 検証用の範囲での指標: (ネットワークの損失, 手作りの損失, ネットワークの cp 誤差, 手作りの cp 誤差)
/// cp 誤差は、評価値と探索の評価値との差の絶対値の平均（±1500 に丸めて比べる）
fn validate(net: &Network, d: &Data, k: f32, teacher_k: f32, lam: f32, lo: usize, hi: usize, threads: usize) -> [f64; 4] {
    if lo >= hi {
        return [0.0; 4];
    }
    let chunk = (hi - lo + threads - 1) / threads;
    let mut tot = [0.0f64; 4];
    thread::scope(|s| {
        let mut hs = vec![];
        for t in 0..threads {
            let (a, b) = (lo + t * chunk, (lo + (t + 1) * chunk).min(hi));
            hs.push(s.spawn(move || {
                let mut r = [0.0f64; 4];
                for i in a..b {
                    let y = teacher_target(d.score[i], d.res[i], teacher_k, lam);
                    let f = forward(net, d, i, false);
                    let nn_eval = d.mat[i] + f.out * OUT_SCALE;
                    r[0] += ce(sigmoid(nn_eval / k), y);
                    r[1] += ce(sigmoid(d.hc[i] / k), y);
                    r[2] += (nn_eval.clamp(-1500.0, 1500.0) - d.score[i].clamp(-1500.0, 1500.0)).abs() as f64;
                    r[3] += (d.hc[i].clamp(-1500.0, 1500.0) - d.score[i].clamp(-1500.0, 1500.0)).abs() as f64;
                }
                r
            }));
        }
        for h in hs {
            let r = h.join().unwrap();
            for q in 0..4 {
                tot[q] += r[q];
            }
        }
    });
    let n = (hi - lo) as f64;
    [tot[0] / n, tot[1] / n, tot[2] / n, tot[3] / n]
}

/// psv_prepare の出力（40バイトの教師局面の並び）から特徴を作る（並列）。
/// 局面は psv_prepare で静かな局面に絞ってあるので、ここでは選ばない。
fn build_psv(recs: &[u8], base_hc: bool, threads: usize) -> Data {
    let n = recs.len() / RECORD_BYTES;
    let per = ((n + threads - 1) / threads).max(1);
    let parts: Vec<Data> = thread::scope(|s| {
        let hs: Vec<_> = recs.chunks(per * RECORD_BYTES).map(|part| s.spawn(move || {
            let mut d = Data::new();
            let mut tmp = Vec::with_capacity(64);
            for rec in part.chunks_exact(RECORD_BYTES) {
                let r = psv::decode(rec).expect("教師局面を読めません");
                d.push(&r.pos, base_hc, r.score as f32, (r.result as f32 + 1.0) / 2.0, &mut tmp);
            }
            d
        })).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut d = Data::new();
    for p in parts { d.append(p); }
    d
}

fn read_records(path: &str, start: u64, count: usize) -> Vec<u8> {
    let mut f = fs::File::open(path).unwrap_or_else(|_| panic!("{} を読めません", path));
    f.seek(SeekFrom::Start(start * RECORD_BYTES as u64)).unwrap();
    let mut buf = vec![0u8; count * RECORD_BYTES];
    f.read_exact(&mut buf).unwrap_or_else(|_| panic!("{} を読めません", path));
    buf
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (x ^ (x >> 31)) | 1
}

fn shuffle<T>(v: &mut [T], rng: &mut Rng) {
    for j in (1..v.len()).rev() {
        let r = (rng.next() % (j as u64 + 1)) as usize;
        v.swap(j, r);
    }
}

/// --psv の設定（チェックポイントで再開するとき、同じ設定かを確かめる）
struct PsvConfig {
    train: String,
    val: String,
    n_train: u64,
    chunk: usize,
    val_max: usize,
    val_every: usize,
    epochs: usize,
    batch: usize,
    lr0: f32,
    lam: f32,
    k: f32,
    teacher_k: f32,
    mirror: bool,
    threads: usize,
}

impl PsvConfig {
    /// 再開のときに一致が必要な値（学習の結果を変えるもの）
    fn signature(&self) -> String {
        format!("H={} L2={} n_train={} chunk={} val_max={} val_every={} epochs={} batch={} lr={} lam={} k={} teacher_k={} mirror={} threads={}",
            H, L2, self.n_train, self.chunk, self.val_max, self.val_every, self.epochs, self.batch,
            self.lr0.to_bits(), self.lam.to_bits(), self.k.to_bits(), self.teacher_k.to_bits(), self.mirror, self.threads)
    }
}

fn put_f32s(out: &mut Vec<u8>, v: &[f32]) {
    for x in v { out.extend_from_slice(&x.to_le_bytes()); }
}

fn grad_to_bytes(g: &Grad, out: &mut Vec<u8>) {
    for v in [&g.w1, &g.b1, &g.w2, &g.b2, &g.w3] { put_f32s(out, v); }
    out.extend_from_slice(&g.b3.to_le_bytes());
}

fn grad_from_bytes(g: &mut Grad, b: &[u8], at: &mut usize) {
    let f32_at = |at: &mut usize| { let x = f32::from_le_bytes(b[*at..*at + 4].try_into().unwrap()); *at += 4; x };
    for v in [&mut g.w1, &mut g.b1, &mut g.w2, &mut g.b2, &mut g.w3] {
        for x in v.iter_mut() { *x = f32_at(at); }
    }
    g.b3 = f32_at(at);
}

/// 学習の途中経過。区切りごとに <出力>.ckpt へ書く
struct Progress {
    epoch: usize,      // 今のエポック（1から）
    chunks_done: usize, // このエポックで終えた区切りの数
    step: usize,
    best_val: f64,
    loss_sum: f64,     // 前回の検証からの学習損失の合計
    loss_cnt: u64,
}

/// 02：ネットを、因数分解の行も含めてそのままの並びで保存する（01 は Network::to_bytes の形）
const CKPT_MAGIC: &[u8; 8] = b"SFCKPT02";

/// 学習中のネット（因数分解の行つき）を、エンジンで読める形にする（因数分解の重みを各特徴の行へ足し込む）
fn export(net: &Network) -> Network {
    let mut o = net.clone();
    o.q = None;
    if N_FACTOR == 0 {
        return o;
    }
    o.w1.truncate(N_FEATS * H);
    for f in 0..N_FEATS {
        let r = N_FEATS + factor_of(f).unwrap();
        for k in 0..H {
            o.w1[f * H + k] += net.w1[r * H + k];
        }
    }
    o
}

/// エンジンの形のネットに、因数分解の行（最初は 0）を足して学習用にする
fn for_training(mut net: Network) -> Network {
    if net.w1.len() == N_FEATS * H {
        net.w1.resize(W1_ROWS * H, 0.0);
    }
    net
}

/// 学習中のネットを、そのままの並び（因数分解の行を含む）でバイト列にする
fn net_raw_bytes(net: &Network, out: &mut Vec<u8>) {
    out.extend_from_slice(&(net.base_hc as u32).to_le_bytes());
    for v in [&net.w1, &net.b1, &net.w2, &net.b2, &net.w3] { put_f32s(out, v); }
    out.extend_from_slice(&net.b3.to_le_bytes());
}

fn net_from_raw(b: &[u8], at: &mut usize) -> Network {
    let base_hc = u32::from_le_bytes(b[*at..*at + 4].try_into().unwrap()) != 0;
    *at += 4;
    let mut g = Grad::new();
    grad_from_bytes(&mut g, b, at);
    let mut net = Network::random(1);
    (net.w1, net.b1, net.w2, net.b2, net.w3, net.b3, net.base_hc) = (g.w1, g.b1, g.w2, g.b2, g.w3, g.b3, base_hc);
    net
}

fn save_ckpt(path: &str, cfg: &PsvConfig, p: &Progress, net: &Network, opt: &Adam) {
    let mut b = Vec::new();
    b.extend_from_slice(CKPT_MAGIC);
    let sig = cfg.signature();
    b.extend_from_slice(&(sig.len() as u64).to_le_bytes());
    b.extend_from_slice(sig.as_bytes());
    for v in [p.epoch as u64, p.chunks_done as u64, p.step as u64, p.loss_cnt, opt.t as u64] { b.extend_from_slice(&v.to_le_bytes()); }
    b.extend_from_slice(&p.best_val.to_le_bytes());
    b.extend_from_slice(&p.loss_sum.to_le_bytes());
    net_raw_bytes(net, &mut b);
    grad_to_bytes(&opt.m, &mut b);
    grad_to_bytes(&opt.v, &mut b);
    let tmp = format!("{}.tmp", path);
    fs::write(&tmp, &b).expect("チェックポイントを書けません");
    fs::rename(&tmp, path).expect("チェックポイントを書けません");
}

fn load_ckpt(path: &str, cfg: &PsvConfig) -> (Progress, Network, Adam) {
    let b = fs::read(path).expect("チェックポイントを読めません");
    assert_eq!(&b[..8], CKPT_MAGIC, "チェックポイントの形式が違います");
    let mut at = 8;
    let u64_at = |at: &mut usize| { let x = u64::from_le_bytes(b[*at..*at + 8].try_into().unwrap()); *at += 8; x };
    let sl = u64_at(&mut at) as usize;
    let sig = String::from_utf8(b[at..at + sl].to_vec()).unwrap();
    at += sl;
    if sig != cfg.signature() {
        eprintln!("チェックポイントと設定が違うので再開できません\n  保存: {}\n  今回: {}", sig, cfg.signature());
        std::process::exit(1);
    }
    let (epoch, chunks_done, step, loss_cnt, t) = (u64_at(&mut at), u64_at(&mut at), u64_at(&mut at), u64_at(&mut at), u64_at(&mut at));
    let best_val = f64::from_bits(u64_at(&mut at));
    let loss_sum = f64::from_bits(u64_at(&mut at));
    let net = net_from_raw(&b, &mut at);
    let mut opt = Adam { m: Grad::new(), v: Grad::new(), t: t as i32 };
    grad_from_bytes(&mut opt.m, &b, &mut at);
    grad_from_bytes(&mut opt.v, &b, &mut at);
    assert_eq!(at, b.len(), "チェックポイントの長さが合いません");
    let p = Progress { epoch: epoch as usize, chunks_done: chunks_done as usize, step: step as usize, best_val, loss_sum, loss_cnt };
    (p, net, opt)
}

fn write_atomic(path: &str, data: &[u8]) {
    let tmp = format!("{}.tmp", path);
    fs::write(&tmp, data).expect("出力できません");
    fs::rename(&tmp, path).expect("出力できません");
}

/// --psv: 学習ファイルを区切り（既定 200万局面）ごとに読み込んで学習する。全局面をメモリに載せない。
/// - エポックごとに、区切りの順番と区切りの中の順番を乱数で並べ替える（乱数はエポック・区切りの番号から決まる）。
/// - 区切りを終えるたびに <出力>.ckpt を保存する。--resume でそこから続ける（止めずに学習した場合と同じ結果になる）。
/// - val_every 区切りごと、およびエポックの終わりに検証し、検証損失が最も小さいネットを <出力> に保存する。
fn train_psv(out_path: &str, cfg: &PsvConfig, init_net: Network, resume: bool) {
    let ckpt_path = format!("{}.ckpt", out_path);
    let threads = cfg.threads;
    let base_hc = init_net.base_hc;
    let t0 = Instant::now();
    let n_val = ((fs::metadata(&cfg.val).expect("検証ファイルがありません").len() / RECORD_BYTES as u64) as usize).min(cfg.val_max);
    let vd = build_psv(&read_records(&cfg.val, 0, n_val), base_hc, threads);
    let n_train = cfg.n_train as usize;
    let n_chunks = (n_train + cfg.chunk - 1) / cfg.chunk;
    let chunk_len = |c: usize| cfg.chunk.min(n_train - c * cfg.chunk);
    let steps_per_epoch: usize = (0..n_chunks).map(|c| (chunk_len(c) + cfg.batch - 1) / cfg.batch).sum();
    let total_steps = (cfg.epochs * steps_per_epoch).max(1);
    println!("学習 {} 局面（区切り {} 個 × 最大 {}）/ 検証 {} 局面  1エポック {} 回の更新  {:.0}秒",
        n_train, n_chunks, cfg.chunk, n_val, steps_per_epoch, t0.elapsed().as_secs_f32());
    println!("K = {}  λ = {}  スレッド {}  左右反転 {}  土台 {}", cfg.k, cfg.lam, threads,
        if cfg.mirror { "あり" } else { "なし" }, if base_hc { "手作りの評価関数" } else { "駒割りだけ" });

    let (mut p, mut net, mut opt) = if resume && fs::metadata(&ckpt_path).is_ok() {
        let (p, net, opt) = load_ckpt(&ckpt_path, cfg);
        println!("再開: エポック {} の区切り {} 個目まで終わっている（更新 {} 回、最良の検証 {:.5}）", p.epoch, p.chunks_done, p.step, p.best_val);
        (p, net, opt)
    } else {
        if fs::metadata(&ckpt_path).is_ok() {
            eprintln!("{} があります。続きから学習するなら --resume を付けてください", ckpt_path);
            std::process::exit(1);
        }
        let net = for_training(init_net);
        let v = validate(&net, &vd, cfg.k, cfg.teacher_k, cfg.lam, 0, vd.len(), threads);
        println!("開始時の検証  損失 ネットワーク {:.5} / 手作り {:.5}   cp 誤差 ネットワーク {:.1} / 手作り {:.1}", v[0], v[1], v[2], v[3]);
        // ウォームスタートのネットも最良候補に含め、初期値より悪い学習で上書きしない。
        write_atomic(out_path, &export(&net).to_bytes());
        (Progress { epoch: 1, chunks_done: 0, step: 0, best_val: v[0], loss_sum: 0.0, loss_cnt: 0 }, net, Adam { m: Grad::new(), v: Grad::new(), t: 0 })
    };

    let mut grads: Vec<Grad> = (0..threads).map(|_| Grad::with_w1(!SPARSE_W1)).collect();
    let mut total = Grad::with_w1(!SPARSE_W1);
    // SPARSE_W1 のとき：第1層の勾配（全体で1つ）、バッチの局面ごとのアキュムレータの勾配、左右反転の有無
    let mut gw1: Vec<f32> = if SPARSE_W1 { vec![0.0; W1_ROWS * H] } else { vec![] };
    let mut ga_store: Vec<[[f32; H]; 2]> = if SPARSE_W1 { vec![[[0.0; H]; 2]; cfg.batch] } else { vec![] };
    let mut flips: Vec<bool> = vec![false; cfg.batch];
    if SPARSE_W1 {
        println!("第1層 {} 行（因数分解 {} 行を含む）の勾配は、行を分担して足し込む", W1_ROWS, N_FACTOR);
    }
    const SEED: u64 = 88172645463325252;
    while p.epoch <= cfg.epochs {
        let mut order: Vec<usize> = (0..n_chunks).collect();
        shuffle(&mut order, &mut Rng(splitmix(SEED ^ p.epoch as u64)));
        while p.chunks_done < n_chunks {
            let c = order[p.chunks_done];
            let len = chunk_len(c);
            let d = build_psv(&read_records(&cfg.train, (c * cfg.chunk) as u64, len), base_hc, threads);
            let mut rng = Rng(splitmix(SEED ^ ((p.epoch as u64) << 32) ^ c as u64 ^ 0x5555));
            let mut idx: Vec<u32> = (0..len as u32).collect();
            shuffle(&mut idx, &mut rng);
            let flip_seed = rng.next();
            for slice in idx.chunks(cfg.batch) {
                // 学習率: コサインで 1/50 まで下げる
                let prog = p.step as f32 / total_steps as f32;
                let lr = cfg.lr0 * (0.02 + 0.98 * 0.5 * (1.0 + (std::f32::consts::PI * prog).cos()));
                let chunk = (slice.len() + threads - 1) / threads;
                let (netr, dr) = (&net, &d);
                let mirror = cfg.mirror;
                let (k, teacher_k, lam) = (cfg.k, cfg.teacher_k, cfg.lam);
                for (n, &si) in slice.iter().enumerate() {
                    flips[n] = mirror && ((si as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ flip_seed) >> 63 == 1;
                }
                let flipsr = &flips;
                let losses: Vec<f64> = thread::scope(|s| {
                    let mut hs = vec![];
                    let mut ga_parts = ga_store.chunks_mut(chunk.max(1));
                    for (t, g) in grads.iter_mut().enumerate() {
                        let (lo, hi) = ((t * chunk).min(slice.len()), ((t + 1) * chunk).min(slice.len()));
                        let part = &slice[lo..hi];
                        let ga_part = if SPARSE_W1 { ga_parts.next() } else { None };
                        hs.push(s.spawn(move || {
                            g.clear();
                            let mut l = 0.0f64;
                            match ga_part {
                                // 第1層の勾配はあとで行を分担して足すので、ここではアキュムレータの勾配だけを控える
                                Some(gp) => {
                                    for (j, &si) in part.iter().enumerate() {
                                        l += sample_grad_core(netr, dr, si as usize, k, teacher_k, lam, flipsr[lo + j], g, &mut gp[j]);
                                    }
                                }
                                None => {
                                    for (j, &si) in part.iter().enumerate() {
                                        l += sample_grad(netr, dr, si as usize, k, teacher_k, lam, flipsr[lo + j], g);
                                    }
                                }
                            }
                            l
                        }));
                    }
                    hs.into_iter().map(|h| h.join().unwrap()).collect()
                });
                total.clear();
                for g in &grads {
                    total.add(g);
                }
                let scale = 1.0 / slice.len() as f32;
                opt.step(&mut net, &total, lr, scale);
                if SPARSE_W1 {
                    sparse_w1_step(&mut net, &mut opt, &mut gw1, &d, slice, &flips[..slice.len()], &ga_store[..slice.len()], lr, scale, threads);
                }
                p.loss_sum += losses.iter().sum::<f64>();
                p.loss_cnt += slice.len() as u64;
                p.step += 1;
            }
            drop(d);
            p.chunks_done += 1;
            let epoch_end = p.chunks_done == n_chunks;
            if epoch_end || p.chunks_done % cfg.val_every == 0 {
                let v = validate(&net, &vd, cfg.k, cfg.teacher_k, cfg.lam, 0, vd.len(), threads);
                let better = v[0] < p.best_val;
                println!(
                    "エポック {:3}/{} 区切り {:4}/{}  学習 {:.5}  検証 {:.5}（手作り {:.5}）  cp 誤差 {:.1}（手作り {:.1}）  {:.0}秒 {}",
                    p.epoch, cfg.epochs, p.chunks_done, n_chunks, p.loss_sum / p.loss_cnt.max(1) as f64, v[0], v[1], v[2], v[3],
                    t0.elapsed().as_secs_f32(), if better { "★" } else { "" }
                );
                p.loss_sum = 0.0;
                p.loss_cnt = 0;
                // 検証の損失が最も小さかった時点のネットワークを保存する（過学習の前のもの）
                if better {
                    p.best_val = v[0];
                    write_atomic(out_path, &export(&net).to_bytes());
                }
            }
            if epoch_end {
                p.epoch += 1;
                p.chunks_done = 0;
            }
            save_ckpt(&ckpt_path, cfg, &p, &net, &opt);
            if epoch_end { break; }
        }
    }
    println!("→ {}（検証損失 {:.5}）", out_path, p.best_val);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut pos_args = vec![];
    let (mut init, mut lr0, mut batch, mut k, mut mirror) = (None, 0.001f32, 16384usize, 600.0f32, true);
    let mut teacher_k: Option<f32> = None;
    let mut base_hc = true;
    let mut filter = TrainingFilter::default();
    let mut dedup = false;
    let mut positions = false;
    let mut thread_limit: Option<usize> = None;
    let (mut psv_mode, mut resume) = (false, false);
    let (mut chunk, mut val_max, mut val_every, mut train_limit) = (2_000_000usize, 1_000_000usize, 5usize, None::<u64>);
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--psv" => { psv_mode = true; i += 1; }
            "--resume" => { resume = true; i += 1; }
            "--chunk" => { chunk = args[i + 1].parse().unwrap(); assert!(chunk > 0); i += 2; }
            "--val-max" => { val_max = args[i + 1].parse().unwrap(); i += 2; }
            "--val-every" => { val_every = args[i + 1].parse().unwrap(); assert!(val_every > 0); i += 2; }
            "--train-limit" => { train_limit = Some(args[i + 1].parse().unwrap()); i += 2; }
            "--threads" => { let n: usize = args[i + 1].parse().expect("Invalid threads"); assert!(n > 0); thread_limit = Some(n); i += 2; }
            "--positions" => { positions = true; i += 1; }
            "--dedup" => { dedup = true; i += 1; }
            "--quiet-drops" => { filter.quiet_drops = true; i += 1; }
            "--skip-opening" => { filter.skip_opening = true; i += 1; }
            "--base-mat" => { base_hc = false; i += 1; }
            "--init" => { init = Some(args[i + 1].clone()); i += 2; }
            "--lr" => { lr0 = args[i + 1].parse().unwrap(); i += 2; }
            "--batch" => { batch = args[i + 1].parse().unwrap(); i += 2; }
            "--teacher-k" => { teacher_k = Some(args[i + 1].parse().unwrap()); i += 2; }
            "--k" => { k = args[i + 1].parse().unwrap(); i += 2; }
            "--nomirror" => { mirror = false; i += 1; }
            _ => { pos_args.push(args[i].clone()); i += 1; }
        }
    }
    if pos_args.len() < 4 {
        eprintln!("使い方: nn_train <出力.nn> <エポック数> <λ> <棋譜ファイル>... [--init 既存.nn] [--lr 0.001] [--batch 16384] [--k 600] [--teacher-k 600] [--nomirror] [--quiet-drops] [--skip-opening] [--dedup]");
        std::process::exit(1);
    }
    let teacher_k = teacher_k.unwrap_or(k);
    assert!(k.is_finite() && k > 0.0 && teacher_k.is_finite() && teacher_k > 0.0, "K must be finite and positive");
    println!("Teacher K = {} (output K = {})", teacher_k, k);
    let out_path = pos_args[0].clone();
    let epochs: usize = pos_args[1].parse().unwrap();
    let lam: f32 = pos_args[2].parse().unwrap();
    let mut net = match &init {
        Some(p) => Network::from_bytes(&fs::read(p).expect("--init のファイルを読めません")).expect("ネットワークの形が合いません"),
        None => {
            let mut n = Network::random(12345);
            n.base_hc = base_hc;
            if base_hc {
                // 補正だけを学ぶので、最初の出力はほぼ 0 にしておく（手作りの評価関数から始める）
                n.w3.iter_mut().for_each(|w| *w *= 0.05);
            }
            n
        }
    };
    if psv_mode {
        assert_eq!(pos_args.len(), 5, "--psv には train.psv と validation.psv の2つを指定する");
        let n_file = fs::metadata(&pos_args[3]).expect("学習ファイルがありません").len() / RECORD_BYTES as u64;
        let cfg = PsvConfig {
            train: pos_args[3].clone(),
            val: pos_args[4].clone(),
            n_train: train_limit.map_or(n_file, |l| l.min(n_file)),
            chunk,
            val_max,
            val_every,
            epochs,
            batch,
            lr0,
            lam,
            k,
            teacher_k,
            mirror,
            threads: thread_limit.unwrap_or_else(|| thread::available_parallelism().map(|v| v.get()).unwrap_or(4).min(16)),
        };
        println!("学習データ（psv_prepare で静かな局面に絞り済み）: {} / {}", cfg.train, cfg.val);
        train_psv(&out_path, &cfg, net, resume);
        return;
    }
    if SPARSE_W1 {
        eprintln!("halfkp のビルドでは --psv の学習だけに対応しています");
        std::process::exit(1);
    }
    println!("学習フィルタ: 静かな駒打ち={} ランダム序盤を除外={} 同一局面除外={}", filter.quiet_drops, filter.skip_opening, dedup);
    let (d, val_start) = if positions {
        load_positions(&pos_args[3..], net.base_hc, filter)
    } else { load(&pos_args[3..], net.base_hc, filter, dedup) };
    let n = d.len();
    if val_start == 0 || val_start >= n {
        eprintln!("学習と検証の両方に有効な局面が必要です");
        std::process::exit(1);
    }
    let threads = thread_limit.unwrap_or_else(|| thread::available_parallelism().map(|v| v.get()).unwrap_or(4).min(16));
    println!(
        "K = {}  λ = {}  スレッド {}  左右反転 {}  土台 {}",
        k, lam, threads, if mirror { "あり" } else { "なし" }, if net.base_hc { "手作りの評価関数" } else { "駒割りだけ" }
    );
    let v = validate(&net, &d, k, teacher_k, lam, val_start, n, threads);
    println!("開始時の検証  損失 ネットワーク {:.5} / 手作り {:.5}   cp 誤差 ネットワーク {:.1} / 手作り {:.1}", v[0], v[1], v[2], v[3]);

    let mut opt = Adam { m: Grad::new(), v: Grad::new(), t: 0 };
    let mut grads: Vec<Grad> = (0..threads).map(|_| Grad::new()).collect();
    let mut total = Grad::new();
    let mut idx: Vec<u32> = (0..val_start as u32).collect();
    let mut rng = Rng(88172645463325252);
    let steps_per_epoch = (val_start + batch - 1) / batch;
    let total_steps = (epochs * steps_per_epoch).max(1);
    let mut step = 0usize;
    // ウォームスタートのネットも最良候補に含め、初期値より悪い学習で上書きしない。
    let mut best_val = v[0];
    fs::write(&out_path, net.to_bytes()).expect("出力できません");
    let t0 = Instant::now();
    for ep in 1..=epochs {
        for j in (1..idx.len()).rev() {
            let r = (rng.next() % (j as u64 + 1)) as usize;
            idx.swap(j, r);
        }
        let ep_seed = rng.next();
        let mut loss_sum = 0.0f64;
        let mut cnt = 0usize;
        for b in 0..steps_per_epoch {
            let slice = &idx[b * batch..((b + 1) * batch).min(idx.len())];
            // 学習率: コサインで 1/50 まで下げる
            let prog = step as f32 / total_steps as f32;
            let lr = lr0 * (0.02 + 0.98 * 0.5 * (1.0 + (std::f32::consts::PI * prog).cos()));
            let chunk = (slice.len() + threads - 1) / threads;
            let netr = &net;
            let dr = &d;
            let losses: Vec<f64> = thread::scope(|s| {
                let mut hs = vec![];
                for (t, g) in grads.iter_mut().enumerate() {
                    let part = &slice[(t * chunk).min(slice.len())..((t + 1) * chunk).min(slice.len())];
                    hs.push(s.spawn(move || {
                        g.clear();
                        let mut l = 0.0f64;
                        for &si in part {
                            let flip = mirror && ((si as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ep_seed) >> 63 == 1;
                            l += sample_grad(netr, dr, si as usize, k, teacher_k, lam, flip, g);
                        }
                        l
                    }));
                }
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            total.clear();
            for g in &grads {
                total.add(g);
            }
            opt.step(&mut net, &total, lr, 1.0 / slice.len() as f32);
            loss_sum += losses.iter().sum::<f64>();
            cnt += slice.len();
            step += 1;
        }
        let v = validate(&net, &d, k, teacher_k, lam, val_start, n, threads);
        let mark = if v[0] < best_val { "★" } else { "" };
        println!(
            "エポック {:3}/{}  学習 {:.5}  検証 {:.5}（手作り {:.5}）  cp 誤差 {:.1}（手作り {:.1}）  {:.0}秒 {}",
            ep, epochs, loss_sum / cnt as f64, v[0], v[1], v[2], v[3], t0.elapsed().as_secs_f32(), mark
        );
        // 検証の損失が最も小さかった時点のネットワークを保存する（過学習の前のもの）
        if v[0] < best_val {
            best_val = v[0];
            fs::write(&out_path, net.to_bytes()).expect("出力できません");
        }
    }
    println!("→ {}（検証損失 {:.5}）", out_path, best_val);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ランダムな対局の局面から作った小さなデータ
    fn random_data(n_games: usize, seed: u64) -> Data {
        let mut d = Data::new();
        let mut tmp = Vec::new();
        let mut rng = seed;
        for _ in 0..n_games {
            let mut pos = Position::startpos();
            for _ in 0..80 {
                let moves = pos.legal_moves();
                if moves.is_empty() { break; }
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                pos.do_move(moves[(rng % moves.len() as u64) as usize]);
                d.push(&pos, true, (rng % 1000) as f32 - 500.0, (rng % 2) as f32, &mut tmp);
            }
        }
        d
    }

    /// 第1層の勾配を「行を分担して足し込む」方法（sparse_w1_step）と、従来の方法（スレッドごとに全体）で、
    /// 更新後の重みがほぼ一致する（足す順番が違うので、丸めの差だけ許す）
    #[test]
    fn sparse_w1_matches_dense() {
        let d = random_data(4, 31);
        let net0 = for_training(Network::random(77));
        let slice: Vec<u32> = (0..d.len() as u32).collect();
        let flips: Vec<bool> = slice.iter().map(|&s| s % 3 == 0).collect();
        let (k, tk, lam, lr) = (600.0, 600.0, 0.7, 0.01);
        let scale = 1.0 / slice.len() as f32;
        // 従来の方法
        let mut dense = net0.clone();
        let mut g = Grad::new();
        for (n, &si) in slice.iter().enumerate() {
            sample_grad(&dense, &d, si as usize, k, tk, lam, flips[n], &mut g);
        }
        let mut opt_d = Adam { m: Grad::new(), v: Grad::new(), t: 0 };
        opt_d.t += 1;
        let (c1, c2) = (1.0 - 0.9f32.powi(1), 1.0 - 0.999f32.powi(1));
        adam_upd(&mut dense.w1, &g.w1, &mut opt_d.m.w1, &mut opt_d.v.w1, lr, scale, c1, c2);
        // 行を分担する方法（3スレッド）
        let mut sparse = net0.clone();
        let mut g2 = Grad::with_w1(false);
        let mut ga = vec![[[0f32; H]; 2]; slice.len()];
        for (n, &si) in slice.iter().enumerate() {
            sample_grad_core(&sparse, &d, si as usize, k, tk, lam, flips[n], &mut g2, &mut ga[n]);
        }
        let mut opt_s = Adam { m: Grad::new(), v: Grad::new(), t: 1 };
        let mut gw1 = vec![0f32; W1_ROWS * H];
        sparse_w1_step(&mut sparse, &mut opt_s, &mut gw1, &d, &slice, &flips, &ga, lr, scale, 3);
        assert!(gw1.iter().all(|&x| x == 0.0), "足し込んだ勾配が 0 に戻っていない");
        let mut changed = 0;
        for i in 0..dense.w1.len() {
            assert!((dense.w1[i] - sparse.w1[i]).abs() < 1e-6, "第1層の重みが合わない（{}）", i);
            if dense.w1[i] != net0.w1[i] { changed += 1; }
        }
        assert!(changed > 1000, "更新された重みが少なすぎる: {}", changed);
    }

    /// 因数分解の重みを足し込んだネット（export）の出力が、学習中のネットの出力とほぼ一致する
    #[test]
    fn export_matches_training_net() {
        let d = random_data(2, 5);
        let mut net = for_training(Network::random(9));
        // 因数分解の行にも値を入れる（halfkp のとき）
        for (i, w) in net.w1[N_FEATS * H..].iter_mut().enumerate() { *w = ((i % 97) as f32 - 48.0) * 1e-3; }
        let e = export(&net);
        assert_eq!(e.w1.len(), N_FEATS * H);
        let mut acc = Acc::default();
        for i in 0..d.len().min(60) {
            let f = forward(&net, &d, i, false);
            // データの特徴から、足し込んだネットでアキュムレータを作り、エンジン側の出力（forward_raw）と比べる
            for (side, persp_feats) in [(0, &d.fa), (1, &d.fb)] {
                let (o0, o1) = (d.off[i] as usize, d.off[i + 1] as usize);
                acc.v[side].copy_from_slice(&e.b1);
                for &ft in &persp_feats[o0..o1] {
                    for q in 0..H { acc.v[side][q] += e.w1[ft as usize * H + q]; }
                }
                for q in 0..H { assert!((acc.v[side][q] - f.a[side][q]).abs() < 1e-4); }
            }
            assert!((e.forward_raw(&acc, 0) - f.out).abs() < 1e-4);
        }
    }
    #[test]
    fn teacher_scale_keeps_side_sign_and_legacy_target() {
        let k = 600.0; let lam = 0.9;
        for score in [-1200.0, -200.0, 0.0, 200.0, 1200.0] {
            for result in [0.0, 0.5, 1.0] {
                assert_eq!(teacher_target(score, result, k, lam), lam * sigmoid(score / k) + (1.0 - lam) * result);
                assert!((teacher_target(score, result, 465.0, lam) + teacher_target(-score, 1.0-result, 465.0, lam) - 1.0).abs() < 1e-6);
            }
        }
        assert!(teacher_target(200.0, 1.0, 465.0, lam) > teacher_target(200.0, 1.0, k, lam));
        assert!(teacher_target(-200.0, 0.0, 465.0, lam) < teacher_target(-200.0, 0.0, k, lam));
    }
    #[test]
    fn independent_positions_keep_side_result_and_explicit_split() {
        let stem = std::env::temp_dir().join(format!("sailfish_positions_{}", std::process::id()));
        let train = stem.with_extension("train.tsv");
        let validation = stem.with_extension("validation.tsv");
        let black_sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
        let white_sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2";
        fs::write(&train, format!("P\t{}\t42\t0\t7g7f\t1\n", black_sfen)).unwrap();
        fs::write(&validation, format!("P\t{}\t-42\t1\t3c3d\t2\n", white_sfen)).unwrap();
        let paths = vec![train.to_string_lossy().into_owned(), validation.to_string_lossy().into_owned()];
        let (d, split) = load_positions(&paths, true, TrainingFilter::default());
        fs::remove_file(train).unwrap();
        fs::remove_file(validation).unwrap();
        assert_eq!(split, 1);
        assert_eq!(d.len(), 2);
        assert_eq!(d.score, vec![42.0, -42.0]);
        assert_eq!(d.res, vec![0.0, 1.0]);
    }
    #[test]
    fn dedup_removes_transpositions_across_games() {
        let first = "R 1 0 7g7f:30000 3c3d:30000 2g2f:30000 8c8d:30000 6i7h:42";
        let second = "R 0 0 2g2f:30000 8c8d:30000 7g7f:30000 3c3d:30000 6i7h:-42";
        let path = std::env::temp_dir().join(format!("sailfish_dedup_{}.txt", std::process::id()));
        fs::write(&path, (0..20).map(|i| if i % 2 == 0 { first } else { second }).collect::<Vec<_>>().join("\n")).unwrap();
        let paths = vec![path.to_string_lossy().into_owned()];
        let (legacy, split) = load(&paths, true, TrainingFilter::default(), false);
        let (unique, _) = load(&paths, true, TrainingFilter::default(), true);
        fs::remove_file(path).unwrap();
        assert_eq!(legacy.len(), 20);
        assert_eq!(split, 19);
        assert_eq!(unique.len(), 1);
        assert_eq!(unique.score, vec![42.0]);
        assert_eq!(unique.res, vec![1.0]);
    }
}
