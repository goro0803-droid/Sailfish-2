//! 診断 diagEP 用：SFEN の一覧の各局面を、ネット（整数化、土台の手作りの評価を含む）と手作りの評価関数で評価する。
//! 使い方: nn_eval_list <ネット.nn> <入力.tsv（1列目が SFEN）> <出力.tsv>
//! 出力: 行番号 <TAB> ネットの評価値 <TAB> 手作りの評価値（どちらも手番から見た値）

use sailfish2::eval::evaluate;
use sailfish2::nn::{Acc, Network};
use sailfish2::position::Position;
use std::fs;
use std::io::{BufWriter, Write};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() != 3 { eprintln!("使い方: nn_eval_list <ネット.nn> <入力.tsv> <出力.tsv>"); std::process::exit(2); }
    let mut net = Network::from_bytes(&fs::read(&a[0]).expect("ネットを読めません")).expect("ネットの形が合いません");
    net.set_quant(true);
    let mut out = BufWriter::new(fs::File::create(&a[2]).unwrap());
    let mut acc = Acc::default();
    let mut n = 0;
    for (i, line) in fs::read_to_string(&a[1]).unwrap().lines().enumerate() {
        let sfen = line.split('\t').next().unwrap();
        let pos = Position::from_sfen(sfen).unwrap_or_else(|e| panic!("{} 行目の SFEN を読めません: {}", i + 1, e));
        net.refresh(&pos, &mut acc);
        writeln!(out, "{}\t{}\t{}", i, net.forward(&acc, &pos), evaluate(&pos)).unwrap();
        n += 1;
    }
    println!("{} 局面を評価しました", n);
}
