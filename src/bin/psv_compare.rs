//! psv_prepare の出力（.psv）が、旧 tools/convert-mekitto-tsv.py の TSV のうち nn_train --positions --quiet-drops で
//! 使われる局面と、順番どおり一致するかを調べる（作り替えの確認用）。
//! 使い方: psv_compare <新.psv> <旧.tsv>

use sailfish2::position::Position;
use sailfish2::psv::{decode, move16_to_usi, RECORD_BYTES};
use sailfish2::training::{eligible_sample, TrainingFilter};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 { eprintln!("使い方: psv_compare <新.psv> <旧.tsv>"); std::process::exit(2); }
    let mut psv = BufReader::with_capacity(1 << 22, File::open(&args[0]).expect(".psv を開けません"));
    let tsv = BufReader::with_capacity(1 << 22, File::open(&args[1]).expect(".tsv を開けません"));
    let filter = TrainingFilter { quiet_drops: true, skip_opening: false };
    let (mut ok, mut bad, mut skipped) = (0u64, 0u64, 0u64);
    let mut rec = [0u8; RECORD_BYTES];
    for line in tsv.lines() {
        let line = line.unwrap();
        if line.is_empty() { continue; }
        let c: Vec<&str> = line.split('\t').collect();
        let mut want = Position::from_sfen(c[1]).expect("旧TSV の sfen が読めません");
        let score: f32 = c[2].parse().unwrap();
        let m = want.parse_usi_move(c[4]).expect("旧TSV の指し手が不正");
        if !eligible_sample(&mut want, m, score, 0, 0, filter) { skipped += 1; continue; }
        if psv.read_exact(&mut rec).is_err() {
            println!("新.psv が先に終わりました（一致 {}）", ok);
            std::process::exit(1);
        }
        let r = decode(&rec).expect("新.psv を読めません");
        let same = (r.pos.board, r.pos.hand, r.pos.side) == (want.board, want.hand, want.side)
            && r.score.to_string() == c[2]
            && format!("{:?}", (r.result as f32 + 1.0) / 2.0) == c[3]
            && move16_to_usi(r.move16) == c[4];
        if same { ok += 1; } else {
            bad += 1;
            if bad <= 5 { println!("不一致 {}件目: 旧 {} / 新 評価値 {} 勝敗 {} 指し手 {}", ok + bad, line, r.score, r.result, move16_to_usi(r.move16)); }
        }
    }
    let extra = psv.read(&mut rec).unwrap() > 0;
    println!("一致 {} / 不一致 {}（旧TSV のうち静かでない局面 {}）{}", ok, bad, skipped, if extra { "、新.psv に余りあり" } else { "" });
    std::process::exit(if bad == 0 && ok > 0 && !extra { 0 } else { 1 });
}
