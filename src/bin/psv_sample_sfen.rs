//! 診断 diagEP 用：教師局面（PSV）から一定の間隔で局面を取り出し、SFEN と教師の評価値を書く。王手をかけられた局面は飛ばす。
//! 使い方: psv_sample_sfen <局面.psv> <間隔> <局面数> <出力.tsv>
//! 出力: SFEN（手数 1）<TAB> 教師の評価値（手番から見た値） <TAB> 結果

use sailfish2::psv::{decode, sfen_key, RECORD_BYTES};
use std::fs;
use std::io::{BufWriter, Write};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() != 4 { eprintln!("使い方: psv_sample_sfen <局面.psv> <間隔> <局面数> <出力.tsv>"); std::process::exit(2); }
    let bytes = fs::read(&a[0]).expect("局面を読めません");
    let (every, n): (usize, usize) = (a[1].parse().unwrap(), a[2].parse().unwrap());
    let mut out = BufWriter::new(fs::File::create(&a[3]).unwrap());
    let mut count = 0;
    for rec in bytes.chunks_exact(RECORD_BYTES).step_by(every.max(1)) {
        if count >= n { break; }
        let r = decode(rec).unwrap();
        if r.pos.in_check() { continue; }
        writeln!(out, "{} 1\t{}\t{}", sfen_key(&r.pos, false), r.score, r.result).unwrap();
        count += 1;
    }
    println!("{} 局面を書きました", count);
}
