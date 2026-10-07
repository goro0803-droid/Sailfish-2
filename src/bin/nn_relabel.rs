//! 自己対戦の既存局面を選び、Sailfish自身の深い探索で教師値を付け直す。
//! nn_relabel <engine> <net> <games> <out> [samples=8000] [nodes=80000] [workers=8] [--uniform]
//! --uniform: 外部棋譜の仮評価値に依存せず、すべての対局から一様抽出。
//! 偶数の棋譜は静かな局面を一様抽出、奇数はNN静的評価と既存探索値の差が最大の局面。
//! 1局から1局面だけ取り、ランダム序盤と王手になる駒打ちを除外。元の対局結果は維持する。
use sailfish2::nn::{Acc, Network};
use sailfish2::position::Position;
use sailfish2::training::{eligible_sample, TrainingFilter};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Debug, PartialEq)]
struct Sample {
    index: usize,
    result: String,
    open: usize,
    moves: String,
    next: String,
    old: i32,
}
fn rng(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn select(path: &str, net: &Network, limit: usize, uniform: bool) -> Result<Vec<Sample>, String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let filter = TrainingFilter {
        quiet_drops: true,
        skip_opening: true,
    };
    let mut selected = Vec::new();
    let mut seed = 712367821u64;
    for (index, line) in text.lines().enumerate() {
        let mut tokens = line.split_whitespace();
        if tokens.next() != Some("R") {
            continue;
        }
        let result = tokens.next().ok_or("missing result")?.to_owned();
        let open: usize = tokens
            .next()
            .ok_or("missing opening")?
            .parse()
            .map_err(|_| "bad opening")?;
        let mut pos = Position::startpos();
        let mut history = Vec::new();
        let mut chosen = None;
        let mut eligible = 0u64;
        let mut largest = -1;
        for (ply, token) in tokens.enumerate() {
            let (usi, cp) = token.rsplit_once(':').ok_or("bad game token")?;
            let cp: i32 = cp.parse().map_err(|_| "bad score")?;
            let m = pos
                .parse_usi_move(usi)
                .ok_or_else(|| format!("illegal move in game {}: {}", index, usi))?;
            if eligible_sample(&mut pos, m, cp as f32, ply, open, filter) {
                eligible += 1;
                let keep = if uniform || index % 2 == 0 {
                    rng(&mut seed) % eligible == 0
                } else {
                    let mut acc = Acc::default();
                    net.refresh(&pos, &mut acc);
                    let gap = (net.forward(&acc, &pos) - cp).abs();
                    let keep = gap > largest;
                    largest = largest.max(gap);
                    keep
                };
                if keep {
                    chosen = Some(Sample {
                        index,
                        result: result.clone(),
                        open,
                        moves: history.join(" "),
                        next: usi.to_owned(),
                        old: cp,
                    });
                }
            }
            history.push(usi.to_owned());
            pos.do_move(m);
        }
        if let Some(s) = chosen {
            selected.push(s);
        }
    }
    // Deterministic shuffle before taking the requested count; no prefix-only sampling.
    for i in (1..selected.len()).rev() {
        let j = (rng(&mut seed) % (i as u64 + 1)) as usize;
        selected.swap(i, j);
    }
    selected.truncate(limit);
    Ok(selected)
}

fn labelled_line(sample: &Sample, cp: i32) -> String {
    let mut tokens: Vec<String> = sample
        .moves
        .split_whitespace()
        .map(|s| format!("{}:30000", s))
        .collect();
    tokens.push(format!("{}:{}", sample.next, cp));
    format!("R {} {} {}", sample.result, sample.open, tokens.join(" "))
}

struct Engine {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
}
impl Engine {
    fn send(&mut self, line: &str) -> Result<(), String> {
        writeln!(self.input, "{}", line)
            .and_then(|_| self.input.flush())
            .map_err(|e| e.to_string())
    }
    fn line(&mut self) -> Result<String, String> {
        let mut s = String::new();
        if self.output.read_line(&mut s).map_err(|e| e.to_string())? == 0 {
            return Err("engine closed stdout".into());
        }
        if s.starts_with("info string") {
            return Err(s);
        }
        Ok(s)
    }
    fn start(exe: &std::path::Path, net: &std::path::Path) -> Result<Self, String> {
        let mut cmd = Command::new(exe);
        cmd.current_dir(exe.parent().ok_or("engine directory")?)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x08000000);
        }
        let mut child = cmd.spawn().map_err(|e| e.to_string())?;
        let input = child.stdin.take().ok_or("stdin")?;
        let output = BufReader::new(child.stdout.take().ok_or("stdout")?);
        let mut e = Self {
            child,
            input,
            output,
        };
        e.send("usi")?;
        let mut own_engine = false;
        loop {
            let line = e.line()?;
            if line.starts_with("id name Sailfish 2") {
                own_engine = true;
            }
            if line.starts_with("usiok") {
                break;
            }
        }
        if !own_engine {
            return Err("教師はSailfish 2のみ使用できます".into());
        }
        e.send("setoption name Threads value 1")?;
        e.send("setoption name USI_Hash value 32")?;
        e.send(&format!("setoption name EvalFile value {}", net.display()))?;
        e.send("isready")?;
        loop {
            if e.line()?.starts_with("readyok") {
                break;
            }
        }
        Ok(e)
    }
    fn score(&mut self, sample: &Sample, nodes: u64) -> Result<i32, String> {
        self.send("usinewgame")?;
        self.send(&format!("position startpos moves {}", sample.moves))?;
        self.send(&format!("go nodes {}", nodes))?;
        let mut score = None;
        loop {
            let line = self.line()?;
            if line.starts_with("bestmove") {
                return score.ok_or("no completed score".into());
            }
            let words: Vec<_> = line.split_whitespace().collect();
            if let Some(i) = words.iter().position(|&w| w == "score") {
                if let (Some(&kind), Some(&value)) = (words.get(i + 1), words.get(i + 2)) {
                    let value: i32 = value.parse().map_err(|_| "bad teacher score")?;
                    score = Some(if kind == "mate" {
                        if value >= 0 {
                            30000
                        } else {
                            -30000
                        }
                    } else {
                        value
                    });
                }
            }
        }
    }
}
impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.send("quit");
        // Kill only the child created by this worker; never unrelated GUI engines.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 4 {
        return Err("nn_relabel <engine> <net> <games> <out> [samples=8000] [nodes=80000] [workers=8] [--uniform]".into());
    }
    let count: usize = args
        .get(4)
        .map(|s| s.parse())
        .transpose()
        .map_err(|_| "samples")?
        .unwrap_or(8000);
    let nodes: u64 = args
        .get(5)
        .map(|s| s.parse())
        .transpose()
        .map_err(|_| "nodes")?
        .unwrap_or(80000);
    let workers: usize = args
        .get(6)
        .map(|s| s.parse())
        .transpose()
        .map_err(|_| "workers")?
        .unwrap_or(8);
    if count == 0 || nodes < 1024 || workers == 0 || workers > 32 {
        return Err("invalid limits".into());
    }
    if std::path::Path::new(&args[3]).exists() {
        return Err("出力ファイルが存在します。新しい名前を指定してください".into());
    }
    let exe = fs::canonicalize(&args[0]).map_err(|e| e.to_string())?;
    let net_path = fs::canonicalize(&args[1]).map_err(|e| e.to_string())?;
    // Strip the Windows extended-path prefix for the engine's textual EvalFile option.
    let net_path = std::path::PathBuf::from(net_path.to_string_lossy().trim_start_matches(r"\\?\"));
    let net = Network::from_bytes(&fs::read(&net_path).map_err(|e| e.to_string())?)?;
    let uniform = args.iter().any(|s| s == "--uniform");
    let samples = select(&args[2], &net, count, uniform)?;
    let n = samples.len();
    if n == 0 {
        return Err("no eligible samples".into());
    }
    // 途中結果は <出力>.partial に1局面ずつ追記する（長時間の処理が止まっても、再実行で続きから進める）
    // 1行の形式: 対局の番号 TAB 教師値の変化の絶対値 TAB 学習用の行
    let partial_path = format!("{}.partial", args[3]);
    let mut resumed = Vec::new();
    if let Ok(text) = fs::read_to_string(&partial_path) {
        for row in text.lines() {
            let mut cols = row.splitn(3, '\t');
            if let (Some(i), Some(d), Some(line)) = (cols.next(), cols.next(), cols.next()) {
                if let (Ok(i), Ok(d)) = (i.parse::<usize>(), d.parse::<i32>()) {
                    resumed.push((i, line.to_owned(), d));
                }
            }
        }
    }
    let finished: std::collections::HashSet<usize> = resumed.iter().map(|r| r.0).collect();
    let samples: Vec<Sample> = samples.into_iter().filter(|s| !finished.contains(&s.index)).collect();
    println!(
        "選択 {} 局面（uniform={}）、教師 {} nodes、並列 {}、途中結果から再開 {} 局面",
        n, uniform, nodes, workers, resumed.len()
    );
    let partial = Arc::new(Mutex::new(
        fs::OpenOptions::new().create(true).append(true).open(&partial_path).map_err(|e| e.to_string())?,
    ));
    let queue = Arc::new(Mutex::new(samples));
    let done = Arc::new(AtomicUsize::new(resumed.len()));
    let output = Arc::new(Mutex::new(resumed));
    let started = Instant::now();
    let results: Vec<Result<(), String>> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..workers {
            let (queue, output, done, partial) = (queue.clone(), output.clone(), done.clone(), partial.clone());
            let (exe, net_path) = (&exe, &net_path);
            let started = &started;
            handles.push(scope.spawn(move || {
                let mut engine = Engine::start(exe, net_path)?;
                loop {
                    let Some(sample) = queue.lock().unwrap().pop() else {
                        break;
                    };
                    let cp = engine.score(&sample, nodes)?;
                    let line = labelled_line(&sample, cp);
                    let delta = (cp - sample.old).abs();
                    {
                        let mut f = partial.lock().unwrap();
                        writeln!(f, "{}\t{}\t{}", sample.index, delta, line)
                            .and_then(|_| f.flush())
                            .map_err(|e| e.to_string())?;
                    }
                    output.lock().unwrap().push((sample.index, line, delta));
                    let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if d % 200 == 0 || d == n {
                        println!("{}/{}  {:.0}秒", d, n, started.elapsed().as_secs_f64());
                    }
                }
                Ok(())
            }));
        }
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .map_err(|_| "worker panic".to_owned())
                    .and_then(|r| r)
            })
            .collect()
    });
    for result in results {
        result?;
    }
    let mut rows = output.lock().unwrap();
    rows.sort_by_key(|r| r.0);
    let delta: f64 = rows.iter().map(|r| r.2 as f64).sum::<f64>() / n as f64;
    let body = rows
        .iter()
        .map(|r| r.1.as_str())
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    fs::write(&args[3], body).map_err(|e| e.to_string())?;
    println!(
        "完了 {} 局面、教師値の平均変化 {:.1}cp、{:.0}秒 -> {}",
        n,
        delta,
        started.elapsed().as_secs_f64(),
        args[3]
    );
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{}", e);
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn relabel_keeps_one_valid_sample_per_game_and_masks_prefix() {
        let game = "R 1 8 7g7f:0 3c3d:0 2g2f:0 8c8d:0 2f2e:0 8d8e:0 6i7h:0 4a3b:0 2e2d:0 2c2d:0 2h2d:0 7a6b:0";
        let path =
            std::env::temp_dir().join(format!("sailfish_relabel_test_{}.txt", std::process::id()));
        fs::write(&path, format!("{}\n{}\n{}\n{}\n", game, game, game, game)).unwrap();
        let net = Network::random(79);
        let samples = select(path.to_str().unwrap(), &net, 3, false).unwrap();
        let repeated = select(path.to_str().unwrap(), &net, 3, false).unwrap();

        assert_eq!(samples, repeated);
        let uniform = select(path.to_str().unwrap(), &net, 3, true).unwrap();
        fs::remove_file(path).unwrap();
        assert_eq!(uniform.len(), 3);
        assert_eq!(samples.len(), 3);
        let mut indices = samples.iter().map(|s| s.index).collect::<Vec<_>>();
        indices.sort();
        indices.dedup();
        assert_eq!(indices.len(), 3);
        for sample in &samples {
            assert!(sample.moves.split_whitespace().count() >= sample.open);
            let mut pos = Position::startpos();
            for usi in sample.moves.split_whitespace() {
                let m = pos.parse_usi_move(usi).unwrap();
                pos.do_move(m);
            }
            let next = pos.parse_usi_move(&sample.next).unwrap();
            assert!(eligible_sample(
                &mut pos,
                next,
                42.0,
                sample.moves.split_whitespace().count(),
                sample.open,
                TrainingFilter {
                    quiet_drops: true,
                    skip_opening: true
                }
            ));
            let line = labelled_line(sample, 42);
            assert!(line.starts_with("R 1 8 "));
            let scores = line
                .split_whitespace()
                .skip(3)
                .map(|t| t.rsplit_once(':').unwrap().1)
                .collect::<Vec<_>>();
            assert_eq!(scores.last(), Some(&"42"));
            assert!(scores[..scores.len() - 1].iter().all(|&s| s == "30000"));
        }
    }
}
