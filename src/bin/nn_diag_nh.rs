//! 段階6 NH の診断：2つのネットの誤差を局面の種類ごとに比べ、ネット B の残りの誤差と手作りの評価の部品の関係を数える。
//! 使い方: nn_diag_nh <ネットA.nn> <ネットB.nn> <局面.psv> [局面数=1000000]
//! 損失は学習の道具と同じ（K=600、λ=0.7 の交差エントロピー）。全体の値が学習の記録と合うことで、計算を確かめられる。

use sailfish2::eval::{eval_components, evaluate, EVAL_COMPONENT_NAMES};
use sailfish2::nn::*;
use sailfish2::psv::{decode, RECORD_BYTES};
use std::fs;
use std::io::Read;

const K: f64 = 600.0;
const LAM: f64 = 0.7;
const N_COMP: usize = EVAL_COMPONENT_NAMES.len();

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
fn loss(eval: f64, t: f64) -> f64 {
    let p = sigmoid(eval / K).clamp(1e-7, 1.0 - 1e-7);
    -(t * p.ln() + (1.0 - t) * (1.0 - p).ln())
}

#[derive(Default, Clone)]
struct Row {
    n: u64,
    loss_a: f64,
    loss_b: f64,
    loss_hc: f64,
    abs_a: f64,
    abs_b: f64,
    // B − A の評価値（手番側）の平均
    diff: f64,
}
impl Row {
    fn add(&mut self, la: f64, lb: f64, lh: f64, ea: f64, eb: f64, s: f64) {
        self.n += 1;
        self.loss_a += la;
        self.loss_b += lb;
        self.loss_hc += lh;
        self.abs_a += (s - ea).abs();
        self.abs_b += (s - eb).abs();
        self.diff += eb - ea;
    }
    fn print(&self, label: &str, total: u64) {
        if self.n == 0 {
            println!("{:<24} 0局面", label);
            return;
        }
        let n = self.n as f64;
        println!(
            "{:<24} {:>8} ({:>5.1}%)  損失 A {:.5} B {:.5} 差 {:+.5} 手作り {:.5}  cp誤差 A {:>6.1} B {:>6.1}  B−A 平均 {:+7.1}",
            label, self.n, 100.0 * n / total as f64,
            self.loss_a / n, self.loss_b / n, (self.loss_b - self.loss_a) / n, self.loss_hc / n,
            self.abs_a / n, self.abs_b / n, self.diff / n
        );
    }
}

/// 残りの誤差 y と部品 x の直線の関係（相関と傾き）
#[derive(Default, Clone, Copy)]
struct Corr {
    n: f64,
    sx: f64,
    sy: f64,
    sxx: f64,
    syy: f64,
    sxy: f64,
}
impl Corr {
    fn add(&mut self, x: f64, y: f64) {
        self.n += 1.0;
        self.sx += x;
        self.sy += y;
        self.sxx += x * x;
        self.syy += y * y;
        self.sxy += x * y;
    }
    fn cov(&self) -> f64 {
        self.sxy - self.sx * self.sy / self.n
    }
    fn varx(&self) -> f64 {
        self.sxx - self.sx * self.sx / self.n
    }
    fn vary(&self) -> f64 {
        self.syy - self.sy * self.sy / self.n
    }
    fn r(&self) -> f64 {
        self.cov() / (self.varx() * self.vary()).sqrt().max(1e-9)
    }
    fn slope(&self) -> f64 {
        self.cov() / self.varx().max(1e-9)
    }
    fn sd(&self) -> f64 {
        (self.varx() / self.n).sqrt()
    }
}

fn bucket(v: f64, edges: &[f64]) -> usize {
    edges.iter().position(|&e| v < e).unwrap_or(edges.len())
}
fn labels(name: &str, edges: &[f64]) -> Vec<String> {
    let mut out = vec![];
    for i in 0..=edges.len() {
        out.push(if i == 0 {
            format!("{} <{}", name, edges[0])
        } else if i == edges.len() {
            format!("{} ≥{}", name, edges[i - 1])
        } else {
            format!("{} {}〜{}", name, edges[i - 1], edges[i])
        });
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("使い方: nn_diag_nh <ネットA.nn> <ネットB.nn> <局面.psv> [局面数=1000000]");
        std::process::exit(2);
    }
    let a = Network::from_bytes(&fs::read(&args[0]).expect("ネットAを読めません")).expect("ネットAの形が合いません");
    let b = Network::from_bytes(&fs::read(&args[1]).expect("ネットBを読めません")).expect("ネットBの形が合いません");
    let n: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(1_000_000);
    let mut bytes = vec![0u8; n * RECORD_BYTES];
    fs::File::open(&args[2]).expect("局面を読めません").read_exact(&mut bytes).expect("局面が足りません");
    println!("A={} (base_hc={})  B={} (base_hc={})  L2={}  局面 {}", args[0], a.base_hc, args[1], b.base_hc, L2, n);

    // 自玉の危険度・敵玉の危険度は、部品の「玉の危険度」と別に、手番側の損得で分ける
    let danger_edges = [-600.0, -200.0, -50.0, 50.0, 200.0, 600.0];
    let piece_edges = [-200.0, -80.0, -30.0, 30.0, 80.0, 200.0];
    let score_edges = [100.0, 300.0, 600.0, 1000.0, 2000.0];
    let hand_edges = [2.0, 5.0, 10.0, 15.0];
    let mut all = Row::default();
    let mut by_danger = vec![Row::default(); danger_edges.len() + 1];
    let mut by_hang = vec![Row::default(); piece_edges.len() + 1];
    let mut by_score = vec![Row::default(); score_edges.len() + 1];
    let mut by_hand = vec![Row::default(); hand_edges.len() + 1];
    let mut corr_a = [Corr::default(); N_COMP];
    let mut corr_b = [Corr::default(); N_COMP];
    let (mut acc_a, mut acc_b) = (Acc::default(), Acc::default());
    // 学習し直さずに部品を α 倍して足したときの損失（ネットの出力は変えない）。効果の下限の目安
    const ALPHAS: [f64; 8] = [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0, 3.0];
    let mut scan_a = [[0f64; ALPHAS.len()]; N_COMP];
    let mut scan_b = [[0f64; ALPHAS.len()]; N_COMP];

    for rec in bytes.chunks_exact(RECORD_BYTES) {
        let r = decode(rec).expect("教師局面を読めません");
        let pos = &r.pos;
        let s = r.score as f64;
        let res = (r.result as f64 + 1.0) / 2.0;
        let t = LAM * sigmoid(s / K) + (1.0 - LAM) * res;
        a.refresh(pos, &mut acc_a);
        b.refresh(pos, &mut acc_b);
        let ea = a.forward(&acc_a, pos) as f64;
        let eb = b.forward(&acc_b, pos) as f64;
        let hc = evaluate(pos) as f64;
        let (la, lb, lh) = (loss(ea, t), loss(eb, t), loss(hc, t));
        let comp = eval_components(pos);
        let hand: i32 = (0..2).map(|c| (1..8).map(|pt| pos.hand[c][pt] as i32).sum::<i32>()).sum();

        for i in 0..N_COMP {
            for (j, &al) in ALPHAS.iter().enumerate() {
                scan_a[i][j] += loss(ea + al * comp[i] as f64, t);
                scan_b[i][j] += loss(eb + al * comp[i] as f64, t);
            }
        }
        all.add(la, lb, lh, ea, eb, s);
        by_danger[bucket(comp[0] as f64, &danger_edges)].add(la, lb, lh, ea, eb, s);
        by_hang[bucket(comp[2] as f64, &piece_edges)].add(la, lb, lh, ea, eb, s);
        by_score[bucket(s.abs(), &score_edges)].add(la, lb, lh, ea, eb, s);
        by_hand[bucket(hand as f64, &hand_edges)].add(la, lb, lh, ea, eb, s);
        // 残りの誤差は、大差の局面（教師の値が飽和する）を除いて数える
        if s.abs() < 2000.0 {
            for i in 0..N_COMP {
                corr_a[i].add(comp[i] as f64, s - ea);
                corr_b[i].add(comp[i] as f64, s - eb);
            }
        }
    }

    let total = all.n;
    println!("\n== 全体（学習の記録の検証損失と合うこと。手作りは 0.70144）");
    all.print("全体", total);
    println!("\n== 玉の危険度の部品（手番側の損得。負なら自玉のほうが危ない）");
    for (row, l) in by_danger.iter().zip(labels("危険度", &danger_edges)) { row.print(&l, total); }
    println!("\n== 取られそうな駒の部品（手番側の損得）");
    for (row, l) in by_hang.iter().zip(labels("取られそう", &piece_edges)) { row.print(&l, total); }
    println!("\n== 教師の評価値の大きさ");
    for (row, l) in by_score.iter().zip(labels("|教師|", &score_edges)) { row.print(&l, total); }
    println!("\n== 持ち駒の枚数（両者の合計）");
    for (row, l) in by_hand.iter().zip(labels("持ち駒", &hand_edges)) { row.print(&l, total); }
    println!("\n== 残りの誤差（教師 − ネット、|教師|<2000）と部品の関係。傾き 0.10 は部品の 10cp が誤差 1cp を説明する");
    for i in 0..N_COMP {
        println!(
            "{:<14} 部品の広がり {:>6.1}cp  A: 相関 {:+.4} 傾き {:+.4}  B: 相関 {:+.4} 傾き {:+.4}",
            EVAL_COMPONENT_NAMES[i], corr_b[i].sd(), corr_a[i].r(), corr_a[i].slope(), corr_b[i].r(), corr_b[i].slope()
        );
    }
    println!("\n== 部品を α 倍して足したときの損失の変化（負なら良くなる。学習はし直していない）。α = {:?}", ALPHAS);
    let (base_a, base_b) = (all.loss_a / total as f64, all.loss_b / total as f64);
    for i in 0..N_COMP {
        let f = |row: &[f64; ALPHAS.len()], base: f64| row.iter().map(|x| format!("{:+.5}", x / total as f64 - base)).collect::<Vec<_>>().join(" ");
        println!("{:<14} A: {}", EVAL_COMPONENT_NAMES[i], f(&scan_a[i], base_a));
        println!("{:<14} B: {}", "", f(&scan_b[i], base_b));
    }
}
