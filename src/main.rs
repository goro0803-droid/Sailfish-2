//! Sailfish - USI プロトコル対応の将棋エンジン

mod eval;
mod position;
mod search;
mod tt;
mod types;

use position::{perft, Position, START_SFEN};
use search::{now_us, Limits, Searcher, Shared};
use std::io::{self, BufRead};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use types::*;

const ENGINE_NAME: &str = "Sailfish 2.0-dev";
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
}

impl Engine {
    /// 探索スレッドが動いていれば終了を待ち、Searcher を回収する
    fn join(&mut self) {
        if let Some(h) = self.handle.take() {
            self.searcher = Some(h.join().expect("search thread panicked"));
        }
    }

    fn stop(&mut self) {
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
        for &s in tokens.iter().skip(moves_idx + 1) {
            match pos.parse_usi_move(s) {
                Some(m) => pos.do_move(m),
                None => {
                    println!("info string illegal move: {}", s);
                    break;
                }
            }
        }
        self.pos = pos;
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

        let (time, inc) = if self.pos.side == BLACK { (btime, binc) } else { (wtime, winc) };
        let delay = self.network_delay;
        if movetime > 0 {
            limits.soft_ms = movetime.saturating_sub(delay).max(10);
            limits.hard_ms = limits.soft_ms;
        } else if time > 0 || byoyomi > 0 || inc > 0 {
            // 持ち時間 + 秒読み（または加算）の範囲で使える最大時間
            let cap = time + byoyomi;
            let hard = (time / 8 + byoyomi + inc).min(cap);
            let soft = (time / 40 + byoyomi + inc * 3 / 4).min(hard);
            limits.hard_ms = hard.saturating_sub(delay).max(10);
            limits.soft_ms = soft.saturating_sub(delay).max(10);
        }

        self.ensure_searcher();
        self.shared.stop.store(false, Ordering::SeqCst);
        self.shared.ponder.store(ponder, Ordering::SeqCst);
        self.shared.start_us.store(now_us(), Ordering::SeqCst);

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

fn main() {
    now_us(); // 時計の基準を初期化
    let shared = Arc::new(Shared::new());
    let mut engine = Engine {
        pos: Position::startpos(),
        shared,
        searcher: None,
        handle: None,
        hash_mb: 256,
        network_delay: 120,
        multipv: 1,
        threads: 1,
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
                println!("option name MultiPV type spin default 1 min 1 max 10");
                println!("option name Threads type spin default 1 min 1 max 64");
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
            "ponderhit" => {
                engine.shared.start_us.store(now_us(), Ordering::SeqCst);
                engine.shared.ponder.store(false, Ordering::SeqCst);
            }
            "gameover" => engine.stop(),
            "quit" => {
                engine.stop();
                break;
            }
            // ---- デバッグ用コマンド ----
            "d" => print!("{}", engine.pos.display()),
            "eval" => println!("eval = {}", eval::evaluate(&engine.pos)),
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
