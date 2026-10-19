//! 実験B3：静かでない教師局面から、静止探索で取り合いを読み切った先の局面（読み筋の末端）を作る。
//!
//! 使い方: psv_qleaf <入力.psv> <出力.psv> [--threads 8] [--max-diff 300] [--score-limit 2500]
//!
//! 各局面 P（評価値 s・勝敗 r は P の手番から見た値）について、Sailfish の静止探索（全幅の窓、埋め込みのネット・整数化）を行い、
//!   - 静止探索の値 v と s の差が --max-diff を超えたら捨てる（静止探索で読み切れない手筋がある）
//!   - 読み筋の末端 L が王手中なら捨てる
//!   - L の評価値・勝敗は、読み筋の手数が偶数なら s・r、奇数なら −s・−r。|評価値| が --score-limit 以上なら捨てる
//!   - L を40バイトの形で書く（指し手の欄は 0。学習では使わない）。件数は <出力>.json に残す。

use sailfish2::psv::{decode, encode, RECORD_BYTES};
use sailfish2::search::{Searcher, Shared};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

const BLOCK: usize = 1 << 18; // 一度に読む局面数

#[derive(Default, Clone, Copy)]
struct Stats { read: u64, kept: u64, diff: u64, check: u64, score: u64, len: [u64; 4] }

fn convert(s: &mut Searcher, rec: &[u8], max_diff: i32, limit: i32, st: &mut Stats, out: &mut Vec<u8>) {
    st.read += 1;
    let r = decode(rec).expect("教師局面を読めません");
    let mut pos = r.pos;
    let (v, pv) = s.qsearch_pv(&mut pos);
    if (v - r.score as i32).abs() > max_diff { st.diff += 1; return; }
    for &m in &pv { pos.do_move(m); }
    if pos.in_check() { st.check += 1; return; }
    let flip = pv.len() % 2 == 1;
    let score = if flip { -(r.score as i32) } else { r.score as i32 };
    if score.abs() >= limit { st.score += 1; return; }
    let result = if flip { -r.result } else { r.result };
    let rec = encode(&pos, score as i16, 0, r.ply.saturating_add(pv.len() as u16), result).expect("局面を書けません");
    out.extend_from_slice(&rec);
    st.kept += 1;
    st.len[pv.len().min(3)] += 1;
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut pos_args = vec![];
    let (mut threads, mut max_diff, mut limit) = (8usize, 300i32, 2500i32);
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--threads" => { threads = args[i + 1].parse().unwrap(); i += 2; }
            "--max-diff" => { max_diff = args[i + 1].parse().unwrap(); i += 2; }
            "--score-limit" => { limit = args[i + 1].parse().unwrap(); i += 2; }
            _ => { pos_args.push(args[i].clone()); i += 1; }
        }
    }
    if pos_args.len() != 2 {
        eprintln!("使い方: psv_qleaf <入力.psv> <出力.psv> [--threads 8] [--max-diff 300] [--score-limit 2500]");
        std::process::exit(2);
    }
    sailfish2::nn::set_quant_global(true);
    sailfish2::nn::load_default().expect("埋め込みのネットワークを読み込めません");
    let t0 = Instant::now();
    let total = fs::metadata(&pos_args[0]).expect("入力がありません").len() / RECORD_BYTES as u64;
    let mut f = BufReader::with_capacity(1 << 22, File::open(&pos_args[0]).unwrap());
    let mut w = BufWriter::with_capacity(1 << 22, File::create(&pos_args[1]).unwrap());
    let mut searchers: Vec<Searcher> = (0..threads).map(|_| Searcher::new(16, Arc::new(Shared::new()))).collect();
    let mut stats = Stats::default();
    let mut buf = vec![0u8; BLOCK * RECORD_BYTES];
    loop {
        let mut got = 0;
        while got < buf.len() {
            let n = f.read(&mut buf[got..]).unwrap();
            if n == 0 { break; }
            got += n;
        }
        if got == 0 { break; }
        let recs = &buf[..got - got % RECORD_BYTES];
        let per = ((recs.len() / RECORD_BYTES + threads - 1) / threads).max(1) * RECORD_BYTES;
        let results: Vec<(Stats, Vec<u8>)> = thread::scope(|sc| {
            let hs: Vec<_> = recs.chunks(per).zip(searchers.iter_mut()).map(|(part, s)| {
                sc.spawn(move || {
                    let mut st = Stats::default();
                    let mut out = Vec::with_capacity(part.len());
                    for rec in part.chunks_exact(RECORD_BYTES) { convert(s, rec, max_diff, limit, &mut st, &mut out); }
                    (st, out)
                })
            }).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (st, out) in results {
            w.write_all(&out).unwrap();
            stats.read += st.read; stats.kept += st.kept; stats.diff += st.diff; stats.check += st.check; stats.score += st.score;
            for k in 0..4 { stats.len[k] += st.len[k]; }
        }
        println!("{} / {} 局面（残した {}、{:.0}秒）", stats.read, total, stats.kept, t0.elapsed().as_secs_f32());
    }
    w.flush().unwrap();
    let json = format!(
        "{{\n \"input\": \"{}\",\n \"maxDiff\": {},\n \"scoreLimit\": {},\n \"read\": {},\n \"kept\": {},\n \"droppedDiff\": {},\n \"droppedCheck\": {},\n \"droppedScore\": {},\n \"pvLength\": {{ \"0\": {}, \"1\": {}, \"2\": {}, \"3+\": {} }},\n \"seconds\": {:.0}\n}}\n",
        pos_args[0].replace('\\', "\\\\"), max_diff, limit, stats.read, stats.kept, stats.diff, stats.check, stats.score,
        stats.len[0], stats.len[1], stats.len[2], stats.len[3], t0.elapsed().as_secs_f32()
    );
    fs::write(format!("{}.json", pos_args[1]), &json).unwrap();
    print!("{}", json);
}
