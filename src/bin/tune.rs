//! 評価関数の学習ツール（Texel 法）。
//! 自己対戦の棋譜（tools/datagen.mjs の出力）の局面から、駒の点数表（PST）を求める。
//!
//! 使い方: tune <棋譜ファイル> <出力 params.rs> [エポック数=300] [λ=0.5] [正則化=1e-7]
//!
//! モデル:  評価値 = 今の評価関数（点数表なし） + Σ 点数表[駒][升]（先手の駒は +、後手の駒は −）
//! 予測勝率 p = σ(評価値 / K)、正解 y = λ·対局結果 + (1-λ)·σ(そのときの探索の評価値 / K)
//! 損失はクロスエントロピー。Adam で最適化する。

use sailfish2::eval::{eval_without_pst, pst_index};
use sailfish2::position::Position;
use sailfish2::types::*;
use std::fs;
use std::thread;

const NPARAM: usize = 15 * 81;

struct Data {
    base: Vec<f32>,   // 点数表を除いた評価値（先手から見た cp）
    search: Vec<f32>, // そのときの探索の評価値（先手から見た cp）
    result: Vec<f32>, // 先手から見た結果 1 / 0.5 / 0
    off: Vec<u32>,    // 特徴の開始位置（長さ n+1）
    feat: Vec<u16>,   // 特徴: 添字*2 + (後手の駒なら 1)
    game: Vec<u32>,   // どの棋譜か（検証用の分割に使う）
}

fn load(path: &str) -> Data {
    let text = fs::read_to_string(path).expect("棋譜ファイルを読めません");
    let mut d = Data { base: vec![], search: vec![], result: vec![], off: vec![0], feat: vec![], game: vec![] };
    let mut games = 0u32;
    for line in text.lines() {
        let mut it = line.split_whitespace();
        if it.next() != Some("R") {
            continue;
        }
        let result: f32 = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.5);
        let open: usize = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
        let mut pos = Position::startpos();
        let mut ok = true;
        for (ply, tok) in it.enumerate() {
            let Some((mv, sc)) = tok.rsplit_once(':') else { ok = false; break };
            let Some(m) = pos.parse_usi_move(mv) else { ok = false; break };
            let mover_score: f32 = sc.parse().unwrap_or(0.0);
            // 静かな局面だけを使う: 序盤の乱しが終わったあとで、王手がかかっておらず、
            // 次の手が駒取り・成り・王手でなく、詰みや大差の局面でないもの
            let quiet_move = !mv_is_drop(m) && pos.board[mv_to(m)] == 0 && !mv_is_promo(m);
            let usable = ply >= open + 2 && quiet_move && !pos.in_check() && mover_score.abs() < 1500.0;
            if usable {
                let black_score = if pos.side == BLACK { mover_score } else { -mover_score };
                d.base.push(eval_without_pst(&pos) as f32);
                d.search.push(black_score);
                d.result.push(result);
                d.game.push(games);
                for sq in 0..81 {
                    let p = pos.board[sq];
                    if p == 0 {
                        continue;
                    }
                    let c = color_of(p);
                    let i = pst_index(c, sq, type_of(p));
                    d.feat.push((i * 2 + c) as u16);
                }
                d.off.push(d.feat.len() as u32);
            }
            pos.do_move(m);
            // 駒取り・王手のあとの局面は使わないので、ここでは何もしない
        }
        if ok {
            games += 1;
        }
    }
    println!("棋譜 {} 局、局面 {} 個、特徴 {} 個", games, d.base.len(), d.feat.len());
    d
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// 勾配と損失を求める（スレッドで分割）。valid: 検証用に除く棋譜の条件
fn grad_loss(d: &Data, w: &[f32], k: f32, lambda: f32, range: (usize, usize), grad: &mut [f32]) -> (f64, usize) {
    let mut loss = 0.0f64;
    let mut n = 0usize;
    for i in range.0..range.1 {
        let mut s = d.base[i];
        for &f in &d.feat[d.off[i] as usize..d.off[i + 1] as usize] {
            let wi = w[(f >> 1) as usize];
            if f & 1 == 0 { s += wi } else { s -= wi }
        }
        let p = sigmoid(s / k);
        let y = lambda * d.result[i] + (1.0 - lambda) * sigmoid(d.search[i] / k);
        let pc = p.clamp(1e-6, 1.0 - 1e-6);
        loss += -(y * pc.ln() + (1.0 - y) * (1.0 - pc).ln()) as f64;
        n += 1;
        let g = (p - y) / k;
        for &f in &d.feat[d.off[i] as usize..d.off[i + 1] as usize] {
            let idx = (f >> 1) as usize;
            if f & 1 == 0 { grad[idx] += g } else { grad[idx] -= g }
        }
    }
    (loss, n)
}

fn loss_only(d: &Data, w: &[f32], k: f32, lambda: f32, range: (usize, usize)) -> f64 {
    let mut g = vec![0f32; NPARAM];
    let (l, n) = grad_loss(d, w, k, lambda, range, &mut g);
    l / n.max(1) as f64
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("使い方: tune <棋譜ファイル> <出力 params.rs> [エポック数=300] [λ=0.5] [正則化=1e-7]");
        std::process::exit(1);
    }
    let epochs: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(300);
    let lambda: f32 = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(0.5);
    let reg: f32 = args.get(5).and_then(|v| v.parse().ok()).unwrap_or(1e-7);

    let d = load(&args[1]);
    let n = d.base.len();
    // 検証用: 棋譜番号が 10 で割って 0 のものは学習に使わない。棋譜は順に並んでいるので範囲で分ける。
    let split = d.game.iter().position(|&g| g >= (d.game.last().copied().unwrap_or(0) * 9 / 10)).unwrap_or(n);
    println!("学習 {} 個 / 検証 {} 個", split, n - split);

    let mut w = vec![0f32; NPARAM];
    // K（勝率への換算の係数）を点数表なしの状態で最もよく合う値にする
    let mut best_k = 400.0f32;
    let mut best_l = f64::MAX;
    for k in (150..=1200).step_by(25) {
        let l = loss_only(&d, &w, k as f32, lambda, (0, split));
        if l < best_l {
            best_l = l;
            best_k = k as f32;
        }
    }
    println!("K = {}  初期の損失 = {:.5}  検証 = {:.5}", best_k, best_l, loss_only(&d, &w, best_k, lambda, (split, n)));

    let threads = thread::available_parallelism().map(|v| v.get()).unwrap_or(4).min(16);
    let mut m = vec![0f32; NPARAM];
    let mut v = vec![0f32; NPARAM];
    let (b1, b2, lr) = (0.9f32, 0.999f32, 1.0f32);
    let chunk = (split + threads - 1) / threads;
    for ep in 1..=epochs {
        let mut grad = vec![0f32; NPARAM];
        let mut total_loss = 0.0f64;
        thread::scope(|s| {
            let mut hs = vec![];
            for t in 0..threads {
                let (lo, hi) = (t * chunk, ((t + 1) * chunk).min(split));
                let (d, w) = (&d, &w);
                hs.push(s.spawn(move || {
                    let mut g = vec![0f32; NPARAM];
                    let (l, _) = if lo < hi { grad_loss(d, w, best_k, lambda, (lo, hi), &mut g) } else { (0.0, 0) };
                    (g, l)
                }));
            }
            for h in hs {
                let (g, l) = h.join().unwrap();
                for i in 0..NPARAM {
                    grad[i] += g[i];
                }
                total_loss += l;
            }
        });
        for i in 0..NPARAM {
            let g = grad[i] / split as f32 + reg * w[i];
            m[i] = b1 * m[i] + (1.0 - b1) * g;
            v[i] = b2 * v[i] + (1.0 - b2) * g * g;
            let mh = m[i] / (1.0 - b1.powi(ep as i32));
            let vh = v[i] / (1.0 - b2.powi(ep as i32));
            w[i] -= lr * mh / (vh.sqrt() + 1e-12);
        }
        if ep % 25 == 0 || ep == epochs {
            let val = loss_only(&d, &w, best_k, lambda, (split, n));
            println!("エポック {:4}  学習 {:.5}  検証 {:.5}", ep, total_loss / split as f64, val);
        }
    }

    // 駒の種類ごとに平均を引く（駒の価値は PIECE_VALUE が受け持つので、点数表は「位置による差」だけにする）
    let mut out = String::new();
    out.push_str("//! 学習で求めた評価関数の点数表（tune コマンドが生成する）。\n");
    out.push_str("//! PST[駒種][升] は、その駒がその升にいるときの加点（持ち主から見た向き）。\n");
    out.push_str("pub static PST: [[i16; 81]; 15] = [\n");
    for pt in 0..15 {
        let row = &w[pt * 81..(pt + 1) * 81];
        let used = row.iter().any(|&x| x != 0.0);
        let mean = if used { row.iter().sum::<f32>() / 81.0 } else { 0.0 };
        out.push_str("    [");
        for (i, &x) in row.iter().enumerate() {
            let val = if used { (x - mean).round().clamp(-300.0, 300.0) as i32 } else { 0 };
            out.push_str(&format!("{}{}", val, if i + 1 < 81 { ", " } else { "" }));
        }
        out.push_str("],\n");
    }
    out.push_str("];\n");
    fs::write(&args[2], out).expect("出力できません");
    println!("→ {}", args[2]);
}
