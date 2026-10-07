//! めきっと氏の公開教師データ（PackedSfenValue）から、nn_train --psv 用の学習・検証ファイルを作る（全件をメモリに載せない）。
//!
//! 使い方: psv_prepare <教師.bin> <出力フォルダ> [--max-accepted N] [--score-limit 3000] [--threads 8]
//!
//! 選び方は tools/convert-mekitto-tsv.py と同じ（ファイルは元々シャッフル済みなので先頭から読む）:
//!   - 不正な手・勝敗があれば中止。手数16未満、王手がかかっている局面、|評価値| > 上限 を除外。
//!   - 左右反転を同一視した SFEN の SHA-256 で、同じ局面は最初の1つだけ残し、50分の1 を検証用に分ける。
//!   - --max-accepted N なら、ここまでで N 局面を採用したところで止める（convert-mekitto-tsv.py の「採用する局面数」）。
//! そのうえで、nn_train --positions --quiet-drops と同じ「静かな局面」（eligible_sample）だけを、
//! 40バイトの元の形のまま train.psv / validation.psv に書く。件数は prepare.json に残す。
//! 重複の判定は、鍵（8バイト）を一時ファイルに書き、上位2ビットで4つに分けて順に行う（メモリを抑えるため）。

use sailfish2::position::Position;
use sailfish2::psv::{canonical_hash, decode, move16_to_usi, RECORD_BYTES};
use sailfish2::training::{eligible_sample, TrainingFilter};
use std::collections::HashSet;
use std::fs::{self, File};
use std::hash::{BuildHasherDefault, Hasher};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::thread;
use std::time::Instant;

const STAGE1: u8 = 1; // 不正でなく、手数・王手・評価値の条件を満たす
const VAL: u8 = 2; // 検証用
const QUIET: u8 = 4; // 静かな局面（学習に使う）
const DUP: u8 = 8; // 前に同じ局面があった
const INVALID: u8 = 16;
const BLOCK: usize = 1 << 20; // 一度に読む局面数
const PARTS: u64 = 4;

/// 鍵はすでに乱数なので、そのまま（混ぜるだけ）ハッシュ値に使う
#[derive(Default)]
struct KeyHasher(u64);
impl Hasher for KeyHasher {
    fn finish(&self) -> u64 { self.0.wrapping_mul(0x9E37_79B9_7F4A_7C15) }
    fn write(&mut self, _: &[u8]) { unreachable!() }
    fn write_u64(&mut self, v: u64) { self.0 = v; }
}

fn classify(rec: &[u8], limit: i32) -> Result<(u8, u64), String> {
    let r = decode(rec)?;
    let mut pos: Position = r.pos;
    let usi = move16_to_usi(r.move16);
    let m = pos.parse_usi_move(&usi).ok_or_else(|| format!("不正な手 {}", usi))?;
    if !(-1..=1).contains(&r.result) { return Err(format!("不正な勝敗 {}", r.result)); }
    if r.ply < 16 || pos.in_check() || (r.score as i32).abs() > limit { return Ok((0, 0)); }
    let (key, val) = canonical_hash(&pos);
    let filter = TrainingFilter { quiet_drops: true, skip_opening: false };
    let quiet = eligible_sample(&mut pos, m, r.score as f32, 0, 0, filter);
    Ok((STAGE1 | if val { VAL } else { 0 } | if quiet { QUIET } else { 0 }, key))
}

fn read_block(f: &mut impl Read, buf: &mut Vec<u8>) -> usize {
    buf.resize(BLOCK * RECORD_BYTES, 0);
    let mut got = 0;
    while got < buf.len() {
        let n = f.read(&mut buf[got..]).expect("教師データを読めません");
        if n == 0 { break; }
        got += n;
    }
    assert!(got % RECORD_BYTES == 0, "教師データの長さが40バイトの倍数ではありません");
    buf.truncate(got);
    got / RECORD_BYTES
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut pos_args = vec![];
    let (mut max_accepted, mut limit, mut threads) = (None::<u64>, 3000i32, 8usize);
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--max-accepted" => { max_accepted = Some(args[i + 1].parse().unwrap()); i += 2; }
            "--score-limit" => { limit = args[i + 1].parse().unwrap(); i += 2; }
            "--threads" => { threads = args[i + 1].parse().unwrap(); i += 2; }
            _ => { pos_args.push(args[i].clone()); i += 1; }
        }
    }
    if pos_args.len() != 2 {
        eprintln!("使い方: psv_prepare <教師.bin> <出力フォルダ> [--max-accepted N] [--score-limit 3000] [--threads 8]");
        std::process::exit(2);
    }
    let (src, out) = (&pos_args[0], Path::new(&pos_args[1]));
    fs::create_dir_all(out).unwrap();
    if out.join("train.psv").exists() { panic!("出力が既にあります: {}", out.display()); }
    let t0 = Instant::now();
    let total = fs::metadata(src).expect("教師データがありません").len() / RECORD_BYTES as u64;
    // 段階1の採用がこれだけあれば、重複を除いても max_accepted に届く（届かなければ最後に中止する）
    let stage1_goal = max_accepted.map(|n| n + n / 20 + 100_000);

    // 1回目: 各局面を調べ、印を flags に、鍵を一時ファイルに書く
    let keys_path = out.join("keys.tmp");
    let mut flags: Vec<u8> = Vec::with_capacity(total as usize);
    let mut part_count = [0usize; PARTS as usize];
    let mut errors: Vec<String> = vec![];
    {
        let mut f = BufReader::with_capacity(1 << 22, File::open(src).unwrap());
        let mut keys = BufWriter::with_capacity(1 << 22, File::create(&keys_path).unwrap());
        let mut buf = vec![];
        let mut stage1 = 0u64;
        loop {
            if stage1_goal.is_some_and(|g| stage1 >= g) { break; }
            let n = read_block(&mut f, &mut buf);
            if n == 0 { break; }
            let per = (n + threads - 1) / threads;
            let results: Vec<(Vec<u8>, Vec<u64>, Vec<String>)> = thread::scope(|s| {
                let hs: Vec<_> = buf.chunks(per * RECORD_BYTES).map(|part| s.spawn(move || {
                    let (mut fl, mut ks, mut er) = (vec![], vec![], vec![]);
                    for rec in part.chunks_exact(RECORD_BYTES) {
                        match classify(rec, limit) {
                            Ok((flag, key)) => { fl.push(flag); ks.push(key); }
                            Err(e) => { fl.push(INVALID); ks.push(0); er.push(e); }
                        }
                    }
                    (fl, ks, er)
                })).collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            for (fl, ks, er) in results {
                for (&flag, &key) in fl.iter().zip(&ks) {
                    if flag & STAGE1 != 0 { stage1 += 1; part_count[(key >> 62) as usize] += 1; }
                    keys.write_all(&key.to_le_bytes()).unwrap();
                }
                flags.extend_from_slice(&fl);
                errors.extend(er);
            }
            println!("1回目 {} / {} 局面（{:.0}秒）", flags.len(), total, t0.elapsed().as_secs_f32());
        }
        keys.flush().unwrap();
    }

    // 2回目: 鍵の上位2ビットごとに、前に同じ局面があったものへ印を付ける
    for p in 0..PARTS {
        let mut seen: HashSet<u64, BuildHasherDefault<KeyHasher>> = HashSet::with_capacity_and_hasher(part_count[p as usize], Default::default());
        let mut f = BufReader::with_capacity(1 << 22, File::open(&keys_path).unwrap());
        let mut kb = [0u8; 8];
        for flag in flags.iter_mut() {
            f.read_exact(&mut kb).unwrap();
            let key = u64::from_le_bytes(kb);
            if *flag & STAGE1 != 0 && key >> 62 == p && !seen.insert(key) { *flag |= DUP; }
        }
        println!("重複の判定 {}/{}（{:.0}秒）", p + 1, PARTS, t0.elapsed().as_secs_f32());
    }
    fs::remove_file(&keys_path).unwrap();

    // 3回目: 採用した静かな局面を、元の40バイトのまま書く
    let mut c = std::collections::BTreeMap::<&str, u64>::new();
    {
        let mut f = BufReader::with_capacity(1 << 22, File::open(src).unwrap());
        let mut w = [
            BufWriter::with_capacity(1 << 22, File::create(out.join("train.psv")).unwrap()),
            BufWriter::with_capacity(1 << 22, File::create(out.join("validation.psv")).unwrap()),
        ];
        let mut rec = [0u8; RECORD_BYTES];
        let mut accepted = 0u64;
        for &flag in &flags {
            if max_accepted.is_some_and(|m| accepted >= m) { break; }
            f.read_exact(&mut rec).unwrap();
            *c.entry("read").or_default() += 1;
            if flag & INVALID != 0 { *c.entry("invalid").or_default() += 1; continue; }
            if flag & STAGE1 == 0 { *c.entry("filtered").or_default() += 1; continue; }
            if flag & DUP != 0 { *c.entry("duplicate").or_default() += 1; continue; }
            accepted += 1;
            let val = (flag & VAL != 0) as usize;
            *c.entry(["train", "validation"][val]).or_default() += 1;
            if flag & QUIET != 0 {
                *c.entry(["trainQuiet", "validationQuiet"][val]).or_default() += 1;
                w[val].write_all(&rec).unwrap();
            }
        }
        *c.entry("accepted").or_default() = accepted;
        for x in &mut w { x.flush().unwrap(); }
        if let Some(m) = max_accepted {
            if accepted < m { panic!("採用が {} に届きません（{}）", m, accepted); }
        }
    }
    let counts = c.iter().map(|(k, v)| format!("  \"{}\": {}", k, v)).collect::<Vec<_>>().join(",\n");
    let json = format!(
        "{{\n \"source\": \"{}\",\n \"maxAccepted\": {},\n \"scoreLimit\": {},\n \"counts\": {{\n{}\n }},\n \"note\": \"tools/convert-mekitto-tsv.py と同じ選び方（先頭から、左右反転を同一視して重複除外・SHA-256で50分の1を検証）。そのうち nn_train --positions --quiet-drops と同じ静かな局面だけを、40バイトのまま書いた。\"\n}}\n",
        src.replace('\\', "\\\\"), max_accepted.map(|v| v.to_string()).unwrap_or("null".into()), limit, counts
    );
    fs::write(out.join("prepare.json"), &json).unwrap();
    print!("{}", json);
    if !errors.is_empty() {
        for e in errors.iter().take(5) { eprintln!("{}", e); }
        eprintln!("不正な局面が {} 件あります", errors.len());
        std::process::exit(1);
    }
    println!("完了（{:.0}秒）", t0.elapsed().as_secs_f32());
}
