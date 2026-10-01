//! ニューラルネットワーク評価の学習ツール。
//!
//! 使い方: nn_train <出力.nn> <エポック数> <λ> <棋譜ファイル>... [--init 既存.nn] [--lr 0.001] [--batch 16384] [--k 600] [--nomirror]
//!
//! 棋譜（tools/datagen.mjs の出力）の静かな局面から、次のように学習する。
//!   予測勝率 p = σ(評価値 / K)       （評価値 = 駒割り + ネットワークの出力 × OUT_SCALE）
//!   正解     y = λ·σ(そのときの探索の評価値 / K) + (1-λ)·対局結果
//! 損失はクロスエントロピー。Adam で最適化し、検証用の棋譜（最後の 5%）で手作りの評価関数と比べる。
//! 将棋は盤を左右に反転しても同じ形勢なので、学習のときは半分の局面を左右反転して使う（データの水増し）。

use sailfish2::eval::evaluate;
use sailfish2::nn::*;
use sailfish2::position::Position;
use sailfish2::types::*;
use std::fs;
use std::sync::OnceLock;
use std::thread;
use std::time::Instant;

struct Data {
    fa: Vec<u16>,    // 手番側の視点の特徴
    fb: Vec<u16>,    // 相手側の視点の特徴
    off: Vec<u32>,   // 特徴の開始位置（長さ n+1。fa と fb で共通）
    mat: Vec<f32>,   // 手番側から見た駒割り
    hc: Vec<f32>,    // 手作りの評価関数（手番側から見た値）
    score: Vec<f32>, // 探索の評価値（手番側から見た値）
    res: Vec<f32>,   // 手番側から見た結果 1 / 0.5 / 0
}

impl Data {
    fn len(&self) -> usize {
        self.mat.len()
    }
}

/// 左右反転した特徴番号の表
fn mirror_table() -> &'static [u16] {
    static T: OnceLock<Vec<u16>> = OnceLock::new();
    T.get_or_init(|| (0..N_FEATS).map(|f| mirror_feature(f) as u16).collect())
}

/// 棋譜を読み、静かな局面を集める。戻り値は (データ, 検証用に分ける最初のサンプル番号)
fn load(paths: &[String], base_hc: bool) -> (Data, usize) {
    let mut d = Data { fa: vec![], fb: vec![], off: vec![0], mat: vec![], hc: vec![], score: vec![], res: vec![] };
    let mut game_starts: Vec<usize> = vec![];
    let mut tmp: Vec<u16> = Vec::with_capacity(64);
    for path in paths {
        let text = fs::read_to_string(path).unwrap_or_else(|_| panic!("{} を読めません", path));
        for line in text.lines() {
            let mut it = line.split_whitespace();
            if it.next() != Some("R") {
                continue;
            }
            let result: f32 = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.5);
            let _open = it.next();
            game_starts.push(d.len());
            let mut pos = Position::startpos();
            for tok in it {
                let Some((mv, sc)) = tok.rsplit_once(':') else { break };
                let Some(m) = pos.parse_usi_move(mv) else { break };
                let score: f32 = sc.parse().unwrap_or(0.0);
                // 静かな局面だけ使う: 次の手が駒取り・成りでなく、王手がかかっておらず、詰みや大差でないもの
                let quiet = !mv_is_drop(m) && pos.board[mv_to(m)] == 0 && !mv_is_promo(m);
                if quiet && !pos.in_check() && score.abs() < 2500.0 {
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

impl Grad {
    fn new() -> Grad {
        Grad {
            w1: vec![0.0; N_FEATS * H],
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
fn feat(f: u16, flip: bool) -> usize {
    if flip { mirror_table()[f as usize] as usize } else { f as usize }
}

fn forward(net: &Network, d: &Data, i: usize, flip: bool) -> Fwd {
    let (o0, o1) = (d.off[i] as usize, d.off[i + 1] as usize);
    let mut a = [[0f32; H]; 2];
    a[0].copy_from_slice(&net.b1);
    a[1].copy_from_slice(&net.b1);
    for &f in &d.fa[o0..o1] {
        let f = feat(f, flip);
        let row = &net.w1[f * H..f * H + H];
        for k in 0..H {
            a[0][k] += row[k];
        }
    }
    for &f in &d.fb[o0..o1] {
        let f = feat(f, flip);
        let row = &net.w1[f * H..f * H + H];
        for k in 0..H {
            a[1][k] += row[k];
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

/// 1 サンプルの損失と勾配（g に加算）
fn sample_grad(net: &Network, d: &Data, i: usize, k: f32, lam: f32, flip: bool, g: &mut Grad) -> f64 {
    let f = forward(net, d, i, flip);
    let eval = d.mat[i] + f.out * OUT_SCALE;
    let p = sigmoid(eval / k);
    let y = lam * sigmoid(d.score[i] / k) + (1.0 - lam) * d.res[i];
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
    let mut ga = [[0f32; H]; 2];
    for p_ in 0..2 {
        for q in 0..H {
            let act = f.a[p_][q];
            ga[p_][q] = if act > 0.0 && act < 1.0 { gx[p_ * H + q] } else { 0.0 };
            g.b1[q] += ga[p_][q];
        }
    }
    let (o0, o1) = (d.off[i] as usize, d.off[i + 1] as usize);
    for &ft in &d.fa[o0..o1] {
        let ft = feat(ft, flip);
        let row = &mut g.w1[ft * H..ft * H + H];
        for q in 0..H {
            row[q] += ga[0][q];
        }
    }
    for &ft in &d.fb[o0..o1] {
        let ft = feat(ft, flip);
        let row = &mut g.w1[ft * H..ft * H + H];
        for q in 0..H {
            row[q] += ga[1][q];
        }
    }
    loss
}

struct Adam {
    m: Grad,
    v: Grad,
    t: i32,
}

impl Adam {
    fn step(&mut self, net: &mut Network, g: &Grad, lr: f32, scale: f32) {
        self.t += 1;
        let (b1, b2) = (0.9f32, 0.999f32);
        let c1 = 1.0 - b1.powi(self.t);
        let c2 = 1.0 - b2.powi(self.t);
        let upd = |w: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32]| {
            for i in 0..w.len() {
                let gi = g[i] * scale;
                m[i] = b1 * m[i] + (1.0 - b1) * gi;
                v[i] = b2 * v[i] + (1.0 - b2) * gi * gi;
                w[i] -= lr * (m[i] / c1) / ((v[i] / c2).sqrt() + 1e-8);
            }
        };
        upd(&mut net.w1, &g.w1, &mut self.m.w1, &mut self.v.w1);
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
fn validate(net: &Network, d: &Data, k: f32, lam: f32, lo: usize, hi: usize, threads: usize) -> [f64; 4] {
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
                    let y = lam * sigmoid(d.score[i] / k) + (1.0 - lam) * d.res[i];
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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut pos_args = vec![];
    let (mut init, mut lr0, mut batch, mut k, mut mirror) = (None, 0.001f32, 16384usize, 600.0f32, true);
    let mut base_hc = true;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--base-mat" => { base_hc = false; i += 1; }
            "--init" => { init = Some(args[i + 1].clone()); i += 2; }
            "--lr" => { lr0 = args[i + 1].parse().unwrap(); i += 2; }
            "--batch" => { batch = args[i + 1].parse().unwrap(); i += 2; }
            "--k" => { k = args[i + 1].parse().unwrap(); i += 2; }
            "--nomirror" => { mirror = false; i += 1; }
            _ => { pos_args.push(args[i].clone()); i += 1; }
        }
    }
    if pos_args.len() < 4 {
        eprintln!("使い方: nn_train <出力.nn> <エポック数> <λ> <棋譜ファイル>... [--init 既存.nn] [--lr 0.001] [--batch 16384] [--k 600] [--nomirror]");
        std::process::exit(1);
    }
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
    let (d, val_start) = load(&pos_args[3..], net.base_hc);
    let n = d.len();
    let threads = thread::available_parallelism().map(|v| v.get()).unwrap_or(4).min(16);
    println!(
        "K = {}  λ = {}  スレッド {}  左右反転 {}  土台 {}",
        k, lam, threads, if mirror { "あり" } else { "なし" }, if net.base_hc { "手作りの評価関数" } else { "駒割りだけ" }
    );
    let v = validate(&net, &d, k, lam, val_start, n, threads);
    println!("開始時の検証  損失 ネットワーク {:.5} / 手作り {:.5}   cp 誤差 ネットワーク {:.1} / 手作り {:.1}", v[0], v[1], v[2], v[3]);

    let mut opt = Adam { m: Grad::new(), v: Grad::new(), t: 0 };
    let mut grads: Vec<Grad> = (0..threads).map(|_| Grad::new()).collect();
    let mut total = Grad::new();
    let mut idx: Vec<u32> = (0..val_start as u32).collect();
    let mut rng = Rng(88172645463325252);
    let steps_per_epoch = (val_start + batch - 1) / batch;
    let total_steps = (epochs * steps_per_epoch).max(1);
    let mut step = 0usize;
    let mut best_val = f64::MAX;
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
                            l += sample_grad(netr, dr, si as usize, k, lam, flip, g);
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
        let v = validate(&net, &d, k, lam, val_start, n, threads);
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
