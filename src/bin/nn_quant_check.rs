//! 評価関数の整数化（実験C1）の確認用。
//! 使い方: nn_quant_check <ネット.nn> <局面.psv> [局面数=200000]
//! ネットの重みと、教師局面でのアキュムレータの範囲を表示する（整数の倍率 S1・S2 を決めるため）。

use sailfish2::nn::*;
use sailfish2::position::Position;
use sailfish2::psv::{decode, RECORD_BYTES};
use std::fs;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 { eprintln!("使い方: nn_quant_check <ネット.nn> <局面.psv> [局面数=200000]"); std::process::exit(2); }
    let net = Network::from_bytes(&fs::read(&args[0]).expect("ネットを読めません")).expect("ネットの形が合いません");
    let n: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(200_000);
    let maxabs = |v: &[f32]| v.iter().fold(0f32, |m, &x| m.max(x.abs()));
    println!("H={} 特徴数={}  max|w1|={:.4} max|b1|={:.4} max|w2|={:.4} max|b2|={:.4} max|w3|={:.4} b3={:.4}",
        H, N_FEATS, maxabs(&net.w1), maxabs(&net.b1), maxabs(&net.w2), maxabs(&net.b2), maxabs(&net.w3), net.b3);
    let row_sum = (0..L2).map(|j| net.w2[j * 2 * H..(j + 1) * 2 * H].iter().map(|x| x.abs()).sum::<f32>()).fold(0f32, f32::max);
    println!("第2層の1行の Σ|w2| の最大 {:.3}", row_sum);
    let bytes = fs::read(&args[1]).expect("局面を読めません");
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    let mut zmax = 0f32;
    let mut acc = Acc::default();
    for rec in bytes.chunks_exact(RECORD_BYTES).take(n) {
        let pos = decode(rec).unwrap().pos;
        net.refresh(&pos, &mut acc);
        for p in 0..2 { for &v in acc.v[p].iter() { lo = lo.min(v); hi = hi.max(v); } }
        for stm in 0..2 {
            for j in 0..L2 {
                let mut z = net.b2[j];
                for i in 0..H {
                    z += net.w2[j * 2 * H + i] * acc.v[stm][i].clamp(0.0, 1.0) + net.w2[j * 2 * H + H + i] * acc.v[stm ^ 1][i].clamp(0.0, 1.0);
                }
                zmax = zmax.max(z.abs());
            }
        }
    }
    println!("局面 {} 個：アキュムレータの範囲 {:.3} 〜 {:.3}、第2層の |z| の最大 {:.3}", n.min(bytes.len() / RECORD_BYTES), lo, hi, zmax);

    // エンジン側の計算（refresh → forward。土台の手作り評価を含む）で、学習ツールと同じ損失を測る（f32 と整数）
    {
        let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
        let mut qn = net.clone();
        qn.set_quant(true);
        for (name, m) in [("f32", &net), ("整数", &qn)] {
            let (mut loss, mut cp, mut cnt) = (0f64, 0f64, 0usize);
            let mut acc = Acc::default();
            for rec in bytes.chunks_exact(RECORD_BYTES).take(n) {
                let r = decode(rec).unwrap();
                m.refresh(&r.pos, &mut acc);
                let e = m.forward(&acc, &r.pos) as f32;
                let y = 0.7 * sig(r.score as f32 / 600.0) + 0.3 * (r.result as f32 + 1.0) / 2.0;
                let p = sig(e / 600.0).clamp(1e-6, 1.0 - 1e-6);
                loss += -(y * p.ln() + (1.0 - y) * (1.0 - p).ln()) as f64;
                cp += (e.clamp(-1500.0, 1500.0) - (r.score as f32).clamp(-1500.0, 1500.0)).abs() as f64;
                cnt += 1;
            }
            println!("エンジン側の計算（{}）：{} 局面  損失 {:.5}  cp 誤差 {:.1}", name, cnt, loss / cnt as f64, cp / cnt as f64);
        }
    }

    // 整数化したネットとの評価値の差（cp）。教師局面と、ランダムな対局の局面で、倍率の組み合わせごとに測る
    let teacher: Vec<Position> = bytes.chunks_exact(RECORD_BYTES).take(n).map(|r| decode(r).unwrap().pos).collect();
    let mut random = vec![];
    let mut rng = 777u64;
    for _ in 0..300 {
        let mut pos = Position::startpos();
        for _ in 0..200 {
            let moves = pos.legal_moves();
            if moves.is_empty() { break; }
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            pos.do_move(moves[(rng % moves.len() as u64) as usize]);
            random.push(pos.clone());
        }
    }
    let scales: Vec<(i32, i32)> = match std::env::var("QSCALES") {
        Ok(s) => s.split(',').map(|p| { let (a, b) = p.split_once('x').unwrap(); (a.parse().unwrap(), b.parse().unwrap()) }).collect(),
        Err(_) => vec![(1024, 8192)],
    };
    for (s1, s2) in scales {
        let mut qnet = net.clone();
        qnet.set_quant_scales(true, s1, s2);
        let worst_sum = row_sum as f64 * s1 as f64 * s2 as f64;
        print!("S1={:5} S2={:5}（第2層の最悪の和 {:.2e}、int32 の上限 2.15e9）", s1, s2, worst_sum);
        for (name, set) in [("教師", &teacher), ("ランダム", &random)] {
            let mut d: Vec<f32> = set.iter().map(|pos| {
                let (mut a, mut b) = (Acc::default(), Acc::default());
                net.refresh(pos, &mut a);
                qnet.refresh(pos, &mut b);
                (net.forward_raw(&a, pos.side) - qnet.forward_raw(&b, pos.side)).abs() * OUT_SCALE
            }).collect();
            d.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mean = d.iter().map(|&x| x as f64).sum::<f64>() / d.len() as f64;
            print!("  {} 平均 {:.3} 99% {:.2} 最大 {:.2}", name, mean, d[d.len() * 99 / 100], d[d.len() - 1]);
        }
        println!();
    }
}
