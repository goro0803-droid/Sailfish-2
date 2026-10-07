//! Rust の PackedSfenValue 読み込み（src/psv.rs）を、cshogi で書き出した正解と突き合わせる検査。
//! 使い方: psv_check <教師.bin> <正解.tsv>
//! 正解.tsv の各行: 番号 TAB sfen TAB 指し手(USI) TAB 評価値 TAB 手数 TAB 勝敗（tools/psv-reference.py が作る）

use sailfish2::position::Position;
use sailfish2::psv::{decode, move16_to_usi, RECORD_BYTES};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 { eprintln!("使い方: psv_check <教師.bin> <正解.tsv>"); std::process::exit(2); }
    let mut bin = File::open(&args[0]).expect("教師.bin を開けません");
    let reference = BufReader::new(File::open(&args[1]).expect("正解.tsv を開けません"));
    let (mut ok, mut bad) = (0usize, 0usize);
    let mut rec = [0u8; RECORD_BYTES];
    for line in reference.lines() {
        let line = line.unwrap();
        let c: Vec<&str> = line.split('\t').collect();
        let index: u64 = c[0].parse().unwrap();
        bin.seek(SeekFrom::Start(index * RECORD_BYTES as u64)).unwrap();
        bin.read_exact(&mut rec).unwrap();
        let r = match decode(&rec) { Ok(r) => r, Err(e) => { bad += 1; if bad <= 5 { println!("{}: 読み込み失敗 {}", index, e); } continue; } };
        let mut want = Position::from_sfen(c[1]).expect("正解の sfen が読めません");
        let usi = move16_to_usi(r.move16);
        let same_pos = (r.pos.board, r.pos.hand, r.pos.side, r.pos.king, r.pos.key) == (want.board, want.hand, want.side, want.king, want.key);
        let legal = want.parse_usi_move(&usi).map(|m| { let l = want.do_move_legal(m); if l { want.undo_move(); } l }).unwrap_or(false);
        let same_vals = usi == c[2] && r.score.to_string() == c[3] && r.ply.to_string() == c[4] && r.result.to_string() == c[5];
        if same_pos && same_vals && legal { ok += 1; } else {
            bad += 1;
            if bad <= 5 { println!("{}: 不一致 局面={} 値={} 合法={} 指し手 {} / 正解 {}", index, same_pos, same_vals, legal, usi, c[2]); }
        }
    }
    println!("一致 {} / 不一致 {}", ok, bad);
    std::process::exit(if bad == 0 && ok > 0 { 0 } else { 1 });
}
