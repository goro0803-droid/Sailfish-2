//! Sailfish - USI プロトコル対応の将棋エンジン

use sailfish2::position::{perft, Position, START_SFEN};
use sailfish2::search::{now_us, Limits, Searcher, Shared};
use sailfish2::types::*;
use sailfish2::{eval, tt};
use std::io::{self, BufRead};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

const ENGINE_NAME: &str = "Sailfish 2.7";
const ENGINE_AUTHOR: &str = "dax";
const STACK_SIZE: usize = 256 * 1024 * 1024;

struct Engine {
    pos: Position,
    shared: Arc<Shared>,
    searcher: Option<Searcher>,
    handle: Option<JoinHandle<Searcher>>,
    hash_mb: usize,
    network_delay: u64,
    multipv: usize,
    threads: usize,
    game_ply: usize,
    balanced_time: bool,
    /// PonderTime=total: 先読み（go ponder）の時間も思考時間に数える。false（reset）は ponderhit で時計を 0 に戻す
    ponder_total: bool,
    /// GUI が Ponder を使う（USI_Ponder=true）
    usi_ponder: bool,
    /// 定跡を使う（cargo の機能 book のときだけ意味がある）
    use_book: bool,
    /// go ponder の局面が定跡にあったとき、ponderhit か stop で返す bestmove の行
    book_pending: Option<String>,
    rng: u64,
}

impl Engine {
    /// 探索スレッドが動いていれば終了を待ち、Searcher を回収する
    fn join(&mut self) {
        if let Some(h) = self.handle.take() {
            self.searcher = Some(h.join().expect("search thread panicked"));
        }
    }

    /// 定跡で先読み中だった手を返す
    fn flush_book_pending(&mut self) {
        if let Some(line) = self.book_pending.take() {
            println!("{}", line);
        }
    }

    /// 定跡にあれば、探索せずに手を決める（bestmove の行と info の行）
    fn book_move(&mut self) -> Option<(String, String)> {
        if !cfg!(feature = "book") || !self.use_book {
            return None;
        }
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        let b = sailfish2::book::global().pick(&self.pos, self.rng)?;
        let (best, info) = match &b.ponder {
            Some(p) => (format!("bestmove {} ponder {}", b.mv, p), format!("info depth 0 score cp {} pv {} {}", b.value, b.mv, p)),
            None => (format!("bestmove {}", b.mv), format!("info depth 0 score cp {} pv {}", b.value, b.mv)),
        };
        Some((best, info))
    }

    fn stop(&mut self) {
        self.flush_book_pending();
        self.shared.stop.store(true, Ordering::SeqCst);
        self.shared.ponder.store(false, Ordering::SeqCst);
        self.join();
    }

    fn ensure_searcher(&mut self) {
        self.join();
        let need_new = match &self.searcher {
            Some(s) => s.tt.size_mb != self.hash_mb,
            None => true,
        };
        if need_new {
            self.searcher = None; // 古い置換表を先に解放
            self.searcher = Some(Searcher::new(self.hash_mb, self.shared.clone()));
        }
    }

    fn setoption(&mut self, tokens: &[&str]) {
        // setoption name <id> [value <x>]
        let name_pos = tokens.iter().position(|&t| t == "name");
        let value_pos = tokens.iter().position(|&t| t == "value");
        let (Some(np), Some(vp)) = (name_pos, value_pos) else { return };
        let name = tokens[np + 1..vp].join(" ");
        let value = tokens[vp + 1..].join(" ");
        match name.as_str() {
            "USI_Hash" => {
                if let Ok(v) = value.parse() {
                    self.hash_mb = v;
                }
            }
            "TimeManagement" => { self.balanced_time = value != "legacy"; }
            // 先読みが当たったときの時間の使い方（隠しオプション。実験P で total を既定にした。reset は 2.4 までの動作）
            "PonderTime" => { self.ponder_total = value == "total"; }
            "USI_Ponder" => { self.usi_ponder = value.eq_ignore_ascii_case("true"); }
            "UseBook" => { self.use_book = value.eq_ignore_ascii_case("true"); }
            "NetworkDelay" => {
                if let Ok(v) = value.parse() {
                    self.network_delay = v;
                }
            }
            // 調整用の隠しオプション（GUI には表示しない）
            "EvalKingDanger" => {
                if let Ok(v) = value.parse() {
                    eval::KING_DANGER_PCT.store(v, std::sync::atomic::Ordering::Relaxed);
                }
            }
            "EvalMobility" => {
                if let Ok(v) = value.parse() {
                    eval::MOBILITY_WEIGHT.store(v, std::sync::atomic::Ordering::Relaxed);
                }
            }
            "Feat" => {
                if let Ok(v) = value.parse() {
                    sailfish2::search::FEAT.store(v, std::sync::atomic::Ordering::Relaxed);
                }
            }
            // ニューラルネットワークの評価関数ファイル（開発用の隠しオプション。Feat に 32 を足すと使う）
            "EvalFile" => {
                if let Err(e) = sailfish2::nn::load_global(&value) {
                    println!("info string EvalFile を読み込めません: {}", e);
                }
            }
            // 評価関数を整数（int16）で計算する（隠しオプション。2.4 から既定 true）
            "NNQuant" => {
                sailfish2::nn::set_quant_global(value.eq_ignore_ascii_case("true"));
            }
            "MultiPV" => {
                if let Ok(v) = value.parse::<usize>() {
                    self.multipv = v.clamp(1, 10);
                }
            }
            "Threads" => {
                if let Ok(v) = value.parse::<usize>() {
                    self.threads = v.clamp(1, 64);
                }
            }
            "UseNeuralNet" => {
                use sailfish2::search::{F_NN, FEAT};
                if value.eq_ignore_ascii_case("true") {
                    FEAT.fetch_or(F_NN, std::sync::atomic::Ordering::Relaxed);
                } else {
                    FEAT.fetch_and(!F_NN, std::sync::atomic::Ordering::Relaxed);
                }
            }
            _ => {}
        }
    }

    fn position(&mut self, tokens: &[&str]) {
        let moves_idx = tokens.iter().position(|&t| t == "moves").unwrap_or(tokens.len());
        let sfen = match tokens.get(1) {
            Some(&"startpos") => START_SFEN.to_string(),
            Some(&"sfen") => tokens[2..moves_idx].join(" "),
            _ => return,
        };
        let mut pos = match Position::from_sfen(&sfen) {
            Ok(p) => p,
            Err(e) => {
                println!("info string {}", e);
                return;
            }
        };
        let mut game_ply = sfen.split_whitespace().nth(3).and_then(|v| v.parse::<usize>().ok()).unwrap_or(1).saturating_sub(1);
        for &s in tokens.iter().skip(moves_idx + 1) {
            match pos.parse_usi_move(s) {
                Some(m) => { pos.do_move(m); game_ply = game_ply.saturating_add(1); },
                None => {
                    println!("info string illegal move: {}", s);
                    break;
                }
            }
        }
        self.pos = pos;
        self.game_ply = game_ply;
    }

    fn go(&mut self, tokens: &[&str]) {
        let mut btime = 0u64;
        let mut wtime = 0u64;
        let mut byoyomi = 0u64;
        let mut binc = 0u64;
        let mut winc = 0u64;
        let mut movetime = 0u64;
        let mut limits = Limits::default();
        let mut ponder = false;

        let mut it = tokens.iter().skip(1);
        while let Some(&t) = it.next() {
            let mut num = || it.next().and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
            match t {
                "btime" => btime = num(),
                "wtime" => wtime = num(),
                "byoyomi" => byoyomi = num(),
                "binc" => binc = num(),
                "winc" => winc = num(),
                "movetime" => movetime = num(),
                "depth" => limits.depth = num() as i32,
                "nodes" => limits.nodes = num(),
                "infinite" => limits.infinite = true,
                "ponder" => ponder = true,
                "mate" => {
                    println!("checkmate notimplemented");
                    return;
                }
                _ => {}
            }
        }

        self.book_pending = None;
        // 入玉宣言（cargo の機能 nyugyoku）：条件を満たしていれば宣言する（検討の go infinite では宣言しない）
        if cfg!(feature = "nyugyoku") && !limits.infinite && self.pos.can_declare() {
            println!("info string declare");
            if ponder {
                self.book_pending = Some("bestmove win".to_string());
            } else {
                println!("bestmove win");
            }
            return;
        }
        // 定跡（対局のときだけ。検討の go infinite や、深さ・ノード数を決めた探索では使わない）
        if !limits.infinite && limits.depth == 0 && limits.nodes == 0 {
            if let Some((best, info)) = self.book_move() {
                println!("info string book");
                println!("{}", info);
                if ponder {
                    self.book_pending = Some(best);
                } else {
                    println!("{}", best);
                }
                return;
            }
        }

        let (time, inc) = if self.pos.side == BLACK { (btime, binc) } else { (wtime, winc) };
        let delay = self.network_delay;
        if movetime > 0 {
            limits.soft_ms = movetime.saturating_sub(delay).max(10);
            limits.hard_ms = limits.soft_ms;
        } else if time > 0 || byoyomi > 0 || inc > 0 {
            let budget = sailfish2::time_control::allocate(time, inc, byoyomi, delay, self.game_ply, self.balanced_time);
            limits.soft_ms = budget.soft_ms;
            limits.hard_ms = budget.hard_ms;
            limits.adaptive_time = self.balanced_time && inc > 0 && byoyomi == 0;
            // 実験T2（cargo の機能 ponder_bonus）：Ponder を使う対局では、当たった手が早く終わって時間が余るので、soft を 1.25 倍（hard まで）にする
            if cfg!(feature = "ponder_bonus") && self.usi_ponder {
                limits.soft_ms = (limits.soft_ms.saturating_mul(5) / 4).min(limits.hard_ms);
            }
            // 実験T4（cargo の機能 time_bank）：41手目以降で、残り時間が加算の10手分より多い（時間がたまっている）なら、
            // soft を 1.5 倍（hard まで）にして中盤を長く考える。hard は変えない
            if cfg!(feature = "time_bank") && self.balanced_time && inc > 0 && byoyomi == 0
                && self.game_ply >= 40 && time > inc.saturating_mul(10)
            {
                limits.soft_ms = (limits.soft_ms.saturating_mul(3) / 2).min(limits.hard_ms);
            }
        }

        self.ensure_searcher();
        self.shared.stop.store(false, Ordering::SeqCst);
        self.shared.ponder.store(ponder, Ordering::SeqCst);
        self.shared.soft_ms.store(limits.soft_ms, Ordering::SeqCst);
        let now = now_us();
        self.shared.start_us.store(now, Ordering::SeqCst);
        self.shared.hard_from_us.store(now, Ordering::SeqCst);

        let mut searcher = self.searcher.take().unwrap();
        searcher.limits = limits;
        searcher.multipv = self.multipv;
        searcher.threads = self.threads;
        let mut pos = self.pos.clone();
        self.handle = Some(
            thread::Builder::new()
                .stack_size(STACK_SIZE)
                .spawn(move || {
                    searcher.think(&mut pos);
                    searcher
                })
                .expect("failed to spawn search thread"),
        );
    }
}

/// Threads の初期値: 論理コア数の半分（物理コア数のめやす）。多すぎると GUI や他のエンジンの邪魔になるので 4 までにする。
fn default_threads() -> usize {
    let logical = thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    (logical / 2).clamp(1, 4)
}

fn main() {
    now_us(); // 時計の基準を初期化
    // 2.4 から、評価関数は整数（int16）で計算するのが標準（実験C1）。USI の NNQuant=false で f32 に戻せる
    sailfish2::nn::set_quant_global(true);
    if let Err(e) = sailfish2::nn::load_default() {
        eprintln!("埋め込みのネットワークを読み込めません: {}", e);
    }
    let shared = Arc::new(Shared::new());
    let mut engine = Engine {
        pos: Position::startpos(),
        shared,
        searcher: None,
        handle: None,
        hash_mb: 256,
        network_delay: 120,
        multipv: 1,
        threads: default_threads(),
        game_ply: 0,
        balanced_time: true,
        ponder_total: true,
        usi_ponder: false,
        use_book: true,
        book_pending: None,
        rng: (now_us() ^ 0x9E37_79B9_7F4A_7C15) | 1,
    };

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let line = line.trim_start_matches('\u{feff}');
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let Some(&cmd) = tokens.first() else { continue };
        match cmd {
            "usi" => {
                println!("id name {}", ENGINE_NAME);
                println!("id author {}", ENGINE_AUTHOR);
                println!("option name USI_Hash type spin default 256 min 1 max 65536");
                println!("option name NetworkDelay type spin default 120 min 0 max 10000");
                println!("option name TimeManagement type combo default balanced var balanced var legacy");
                println!("option name MultiPV type spin default 1 min 1 max 10");
                println!("option name Threads type spin default {} min 1 max 64", default_threads());
                println!("option name UseNeuralNet type check default true");
                // GUI は名乗ったオプションしか送らないことがあるので、実験T2 のビルドでは USI_Ponder を名乗る
                if cfg!(feature = "ponder_bonus") {
                    println!("option name USI_Ponder type check default false");
                }
                if cfg!(feature = "book") {
                    println!("option name UseBook type check default true");
                }
                println!("usiok");
            }
            "setoption" => engine.setoption(&tokens),
            "isready" => {
                engine.ensure_searcher();
                println!("readyok");
            }
            "usinewgame" => {
                engine.ensure_searcher();
                if let Some(s) = engine.searcher.as_mut() {
                    s.tt.clear();
                    s.new_game();
                }
            }
            "position" => {
                engine.join();
                engine.position(&tokens);
            }
            "go" => {
                engine.join();
                engine.go(&tokens);
            }
            "stop" => engine.stop(),
            "ponderhit" if engine.book_pending.is_some() => engine.flush_book_pending(),
            "ponderhit" => {
                // 打ち切りの上限は、自分の時計が動き出したこの時点から数える
                let now = now_us();
                engine.shared.hard_from_us.store(now, Ordering::SeqCst);
                if engine.ponder_total {
                    // 先読みの時間も数える。もう soft を過ぎていれば、読み終えた反復の最善手ですぐ指す
                    let elapsed_ms = now.saturating_sub(engine.shared.start_us.load(Ordering::SeqCst)) / 1000;
                    if elapsed_ms >= engine.shared.soft_ms.load(Ordering::SeqCst) {
                        engine.shared.stop.store(true, Ordering::SeqCst);
                    }
                } else {
                    engine.shared.start_us.store(now, Ordering::SeqCst);
                }
                engine.shared.ponder.store(false, Ordering::SeqCst);
            }
            "gameover" => {
                engine.book_pending = None;
                engine.stop();
            }
            "quit" => {
                engine.stop();
                break;
            }
            // ---- デバッグ用コマンド ----
            "d" => print!("{}", engine.pos.display()),
            "eval" => println!("eval = {}", eval::evaluate(&engine.pos)),
            // 手番の偏りの確認: 今の局面の静的評価と、駒を取らず・成らず・王手でない手を指した後の静的評価（符号を反転）を比べる
            // 出力: evalgap hc <親> <子の最大> <子の中央値> nn <親> <子の最大> <子の中央値> moves <数>
            "evalgap" => {
                let st = |p: &Position| -> (i32, i32) {
                    let nn = sailfish2::nn::global().map_or(0, |net| {
                        let mut acc = sailfish2::nn::Acc::default();
                        net.refresh(p, &mut acc);
                        net.forward(&acc, p)
                    });
                    (eval::evaluate(p), nn)
                };
                let mut pos = engine.pos.clone();
                let (h0, n0) = st(&pos);
                let (mut hs, mut ns) = (Vec::new(), Vec::new());
                for m in pos.legal_moves() {
                    if mv_is_promo(m) || (!mv_is_drop(m) && pos.board[mv_to(m)] != 0) {
                        continue;
                    }
                    pos.do_move(m);
                    if !pos.in_check() {
                        let (h, n) = st(&pos);
                        hs.push(-h);
                        ns.push(-n);
                    }
                    pos.undo_move();
                }
                hs.sort();
                ns.sort();
                if hs.is_empty() {
                    println!("evalgap none");
                } else {
                    println!("evalgap hc {} {} {} nn {} {} {} moves {}", h0, hs[hs.len() - 1], hs[hs.len() / 2],
                        n0, ns[ns.len() - 1], ns[ns.len() / 2], hs.len());
                }
            }
            // 評価の速さの計測: evalbench [回数]（今の局面で、手作りの評価・ネットワークの出力・差分更新の 1 回あたりの時間を出す）
            "evalbench" => {
                use std::time::Instant;
                let n: usize = tokens.get(1).and_then(|v| v.parse().ok()).unwrap_or(1_000_000);
                let pos = engine.pos.clone();
                let mut sink = 0i64;
                let t = Instant::now();
                for _ in 0..n {
                    sink += eval::evaluate(std::hint::black_box(&pos)) as i64;
                }
                println!("手作りの評価    {:6.1} ns/回", t.elapsed().as_nanos() as f64 / n as f64);
                if let Some(net) = sailfish2::nn::global() {
                    let mut acc = sailfish2::nn::Acc::default();
                    net.refresh(&pos, &mut acc);
                    let t = Instant::now();
                    for _ in 0..n {
                        sink += (net.forward_raw(std::hint::black_box(&acc), pos.side) * 1000.0) as i64;
                    }
                    println!("ネットワークの出力 {:6.1} ns/回", t.elapsed().as_nanos() as f64 / n as f64);
                    // 指したあとの局面を先に作っておき、差分更新だけを測る
                    let mut p2 = pos.clone();
                    let moves = p2.legal_moves();
                    let after: Vec<(Position, Move, u8)> = moves
                        .iter()
                        .map(|&m| {
                            let cap = if mv_is_drop(m) { 0 } else { pos.board[mv_to(m)] };
                            let mut q = pos.clone();
                            q.do_move(m);
                            (q, m, cap)
                        })
                        .collect();
                    let mut next = sailfish2::nn::Acc::default();
                    let t = Instant::now();
                    let rounds = (n / moves.len().max(1)).max(1);
                    for _ in 0..rounds {
                        for (q, m, cap) in &after {
                            net.apply_after(std::hint::black_box(&acc), q, *m, *cap, &mut next);
                            sink += next.mat as i64;
                        }
                    }
                    println!("差分更新（1 手）  {:6.1} ns/回", t.elapsed().as_nanos() as f64 / (rounds * moves.len().max(1)) as f64);
                    let t = Instant::now();
                    for _ in 0..(n / 20).max(1) {
                        net.refresh(std::hint::black_box(&pos), &mut acc);
                        sink += acc.mat as i64;
                    }
                    println!("作り直し         {:6.1} ns/回", t.elapsed().as_nanos() as f64 / (n / 20).max(1) as f64);
                }
                let _ = sink;
            }
            "moves" => {
                let ms: Vec<String> = engine.pos.legal_moves().iter().map(|&m| move_to_usi(m)).collect();
                println!("{} moves: {}", ms.len(), ms.join(" "));
            }
            "perft" => {
                let depth: u32 = tokens.get(1).and_then(|v| v.parse().ok()).unwrap_or(3);
                let mut pos = engine.pos.clone();
                for d in 1..=depth {
                    let t = std::time::Instant::now();
                    let n = perft(&mut pos, d);
                    println!("perft {} = {} ({} ms)", d, n, t.elapsed().as_millis());
                }
            }
            _ => println!("info string unknown command: {}", cmd),
        }
    }
}
