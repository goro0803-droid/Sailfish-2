//! 実験B3：2つの教師局面のファイル（40バイトの並び）を、件数の比率に合わせて交互に並べた1つのファイルにする。
//! 学習の区切り（200万局面）の中で、種類が偏らないようにするため。
//!
//! 使い方: psv_merge <A.psv> <B.psv> <出力.psv>

use sailfish2::psv::RECORD_BYTES;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        eprintln!("使い方: psv_merge <A.psv> <B.psv> <出力.psv>");
        std::process::exit(2);
    }
    let count = |p: &str| fs::metadata(p).unwrap_or_else(|_| panic!("{} がありません", p)).len() / RECORD_BYTES as u64;
    let (na, nb) = (count(&args[0]), count(&args[1]));
    let total = na + nb;
    let mut a = BufReader::with_capacity(1 << 22, File::open(&args[0]).unwrap());
    let mut b = BufReader::with_capacity(1 << 22, File::open(&args[1]).unwrap());
    let mut w = BufWriter::with_capacity(1 << 22, File::create(&args[2]).unwrap());
    let mut rec = [0u8; RECORD_BYTES];
    let (mut wa, mut wb) = (0u64, 0u64);
    for i in 0..total {
        // A の書いた数が、全体に占める A の割合に追いつくように選ぶ
        let take_a = wb >= nb || (wa < na && (wa as u128) * (total as u128) < ((i + 1) as u128) * (na as u128));
        if take_a { a.read_exact(&mut rec).unwrap(); wa += 1; } else { b.read_exact(&mut rec).unwrap(); wb += 1; }
        w.write_all(&rec).unwrap();
        if (i + 1) % 20_000_000 == 0 { println!("{} / {} 局面", i + 1, total); }
    }
    w.flush().unwrap();
    println!("A {} 局面、B {} 局面、合計 {} 局面を書いた", wa, wb, total);
}
