//! 探索部: 反復深化 + PVS(αβ) + 静止探索。
//! 置換表・キラー・ヒストリー・ヌルムーブ・LMR・王手延長を使用。

use crate::eval::{evaluate, PIECE_VALUE};
use crate::nn::{self, Acc, Network};
use crate::position::{MoveList, Position, MAX_MOVES};
use crate::tt::*;
use crate::types::*;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

/// 探索の改良を個別に入れ替えるためのビット（USI の隠しオプション Feat で指定。強さの比較用）
pub static FEAT: AtomicU32 = AtomicU32::new(F_QS_TT | F_NN);
/// 静止探索でも置換表を使う
pub const F_QS_TT: u32 = 1;
/// 置換表に手がないとき、深さを 1 減らす（Internal Iterative Reduction）
pub const F_IIR: u32 = 2;
/// 評価値が 2 手前より良くなっているかで、枝刈りと LMR の強さを変える
pub const F_IMPROVING: u32 = 4;
/// ヌルムーブの減らし方を、深さと評価値の余裕に応じて大きくする
pub const F_NULL2: u32 = 8;
/// 評価値を局面ごとに覚えておく（同じ局面を何度も評価しない）
pub const F_ECACHE: u32 = 16;
/// 手作りの評価関数のかわりに、ニューラルネットワーク（EvalFile で読み込んだもの）で評価する
pub const F_NN: u32 = 32;

const ECACHE_BITS: usize = 18;

pub const MAX_PLY: usize = 128;
pub const INF: i32 = 31000;
pub const MATE: i32 = 30000;
pub const MATE_BOUND: i32 = MATE - MAX_PLY as i32;

static EPOCH: OnceLock<Instant> = OnceLock::new();

pub fn now_us() -> u64 {
    EPOCH.get_or_init(Instant::now).elapsed().as_micros() as u64
}

/// USI スレッドと探索スレッドで共有する状態
pub struct Shared {
    pub stop: AtomicBool,
    pub ponder: AtomicBool,
    pub start_us: AtomicU64,
    /// 主スレッドが読み終えたので、補助スレッドも止める
    pub helpers_stop: AtomicBool,
    /// 全スレッドの合計ノード数（1024 ノードごとに加算）
    pub nodes: AtomicU64,
}

impl Shared {
    pub fn new() -> Shared {
        Shared {
            stop: AtomicBool::new(false),
            ponder: AtomicBool::new(false),
            start_us: AtomicU64::new(0),
            helpers_stop: AtomicBool::new(false),
            nodes: AtomicU64::new(0),
        }
    }
}

/// 補助スレッドごとに読む深さをずらす表（Lazy SMP）
const SKIP_SIZE: [i32; 20] = [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4];
const SKIP_PHASE: [i32; 20] = [0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7];

#[derive(Clone, Copy)]
pub struct Limits {
    /// この時間を過ぎたら次の反復を始めない
    pub soft_ms: u64,
    /// この時間を過ぎたら探索を打ち切る
    pub hard_ms: u64,
    pub depth: i32,
    pub nodes: u64,
    /// go infinite: stop が来るまで bestmove を返さない
    pub infinite: bool,
}

impl Default for Limits {
    fn default() -> Self {
        Limits { soft_ms: u64::MAX, hard_ms: u64::MAX, depth: 0, nodes: 0, infinite: false }
    }
}

pub struct Searcher {
    pub tt: Arc<TT>,
    pub shared: Arc<Shared>,
    pub limits: Limits,
    /// 候補手の数（USI の MultiPV オプション）
    pub multipv: usize,
    /// 探索スレッドの数（USI の Threads オプション）
    pub threads: usize,
    /// 0 = 主スレッド（info の出力・時間管理を行う）、1 以上 = 補助スレッド
    helper_idx: usize,
    /// 有効な探索の改良（F_ のビット）
    feat: u32,
    /// 各 ply の静的評価値（王手を受けているときは -INF）
    static_evals: [i32; MAX_PLY + 2],
    /// 評価値のキャッシュ（局面のキー, 評価値）。スレッドごとに持つ。
    ecache: Vec<(u64, i32)>,
    /// ニューラルネットワーク（F_NN が有効で、読み込み済みのとき）
    nn: Option<Arc<Network>>,
    /// 各 ply の局面に対応するアキュムレータ（nn があるときだけ使う）
    accs: Vec<Acc>,
    /// 各 ply の局面に至った手と、そのとき取った駒（アキュムレータの更新待ち）
    pend: Vec<(Move, u8)>,
    root_excluded: Vec<Move>,
    nodes: u64,
    stopped: bool,
    sel_depth: usize,
    killers: [[Move; 2]; MAX_PLY + 2],
    history: Vec<i32>,
    /// 直前の相手の手（駒と行き先）に対してよく効いた応手
    counter: Vec<Move>,
    /// 読みを浅くする量の表 [深さ][何手目]
    lmr: Vec<i32>,
    pv: Vec<Move>,
    pv_len: [usize; MAX_PLY + 2],
}

const PV_STRIDE: usize = MAX_PLY + 2;

#[inline]
fn score_to_tt(s: i32, ply: usize) -> i32 {
    if s >= MATE_BOUND {
        s + ply as i32
    } else if s <= -MATE_BOUND {
        s - ply as i32
    } else {
        s
    }
}

#[inline]
fn score_from_tt(s: i32, ply: usize) -> i32 {
    if s >= MATE_BOUND {
        s - ply as i32
    } else if s <= -MATE_BOUND {
        s + ply as i32
    } else {
        s
    }
}

fn score_to_usi(s: i32) -> String {
    if s >= MATE_BOUND {
        format!("mate {}", MATE - s)
    } else if s <= -MATE_BOUND {
        format!("mate -{}", MATE + s)
    } else {
        format!("cp {}", s)
    }
}

impl Searcher {
    pub fn new(hash_mb: usize, shared: Arc<Shared>) -> Searcher {
        Searcher::with_tt(Arc::new(TT::new(hash_mb)), shared)
    }

    fn with_tt(tt: Arc<TT>, shared: Arc<Shared>) -> Searcher {
        Searcher {
            tt,
            shared,
            limits: Limits::default(),
            multipv: 1,
            threads: 1,
            helper_idx: 0,
            feat: 0,
            static_evals: [-INF; MAX_PLY + 2],
            ecache: vec![(0, 0); 1 << ECACHE_BITS],
            nn: None,
            accs: Vec::new(),
            pend: Vec::new(),
            root_excluded: Vec::new(),
            nodes: 0,
            stopped: false,
            sel_depth: 0,
            killers: [[MOVE_NONE; 2]; MAX_PLY + 2],
            history: vec![0; 32 * 81],
            counter: vec![MOVE_NONE; 32 * 81],
            lmr: (0..64 * 64)
                .map(|i| {
                    let (d, m) = ((i / 64) as f64, (i % 64) as f64);
                    if d < 1.0 || m < 1.0 { 0 } else { (0.75 + d.ln() * m.ln() / 2.25) as i32 }
                })
                .collect(),
            pv: vec![MOVE_NONE; PV_STRIDE * PV_STRIDE],
            pv_len: [0; MAX_PLY + 2],
        }
    }

    fn elapsed_ms(&self) -> u64 {
        now_us().saturating_sub(self.shared.start_us.load(Ordering::Relaxed)) / 1000
    }

    /// 静的評価（F_ECACHE が有効なら、同じ局面の評価値を覚えておく）
    #[inline]
    fn eval(&mut self, pos: &Position, ply: usize) -> i32 {
        if let Some(nn) = &self.nn {
            return nn.forward(&self.accs[ply], pos);
        }
        if self.feat & F_ECACHE == 0 {
            return evaluate(pos);
        }
        let i = pos.key as usize & ((1 << ECACHE_BITS) - 1);
        let e = self.ecache[i];
        if e.0 == pos.key {
            return e.1;
        }
        let v = evaluate(pos);
        self.ecache[i] = (pos.key, v);
        v
    }

    /// 補助スレッド用の探索状態を作る（置換表と停止フラグは主スレッドと共有）
    fn make_helper(&self, idx: usize) -> Searcher {
        let mut h = Searcher::with_tt(self.tt.clone(), self.shared.clone());
        h.limits = self.limits;
        h.helper_idx = idx;
        h.feat = self.feat;
        h.set_network(self.nn.clone());
        h
    }

    /// ニューラルネットワークを使う設定にする（None なら手作りの評価関数）
    fn set_network(&mut self, nn: Option<Arc<Network>>) {
        if nn.is_some() && self.accs.len() != MAX_PLY + 2 {
            self.accs = vec![Acc::default(); MAX_PLY + 2];
            self.pend = vec![(MOVE_NONE, 0); MAX_PLY + 2];
        }
        self.nn = nn;
    }

    /// 子の局面の評価に必要な情報（指した手と取った駒）を、指す前に控えておく。
    /// アキュムレータの更新は、子の局面で評価が必要になったときに ensure_acc で行う
    /// （枝刈りや違法手で捨てる手のぶんは計算しない）。
    #[inline]
    fn set_pending(&mut self, pos: &Position, m: Move, ply: usize) {
        if self.nn.is_some() {
            let cap = if mv_is_drop(m) { 0 } else { pos.board[mv_to(m)] };
            self.pend[ply + 1] = (m, cap);
        }
    }

    /// null move のあとは駒の配置が変わらないので、アキュムレータはそのまま写す（MOVE_NONE が目印）
    #[inline]
    fn set_pending_null(&mut self, ply: usize) {
        if self.nn.is_some() {
            self.pend[ply + 1] = (MOVE_NONE, 0);
        }
    }

    /// この ply の局面（pos）のアキュムレータを、一つ前の ply のものと控えておいた手から作る
    #[inline]
    fn ensure_acc(&mut self, pos: &Position, ply: usize) {
        if ply == 0 {
            return;
        }
        if let Some(nn) = &self.nn {
            let (m, cap) = self.pend[ply];
            let (a, b) = self.accs.split_at_mut(ply);
            if m == MOVE_NONE {
                b[0].clone_from(&a[ply - 1]);
            } else {
                nn.apply_after(&a[ply - 1], pos, m, cap, &mut b[0]);
            }
        }
    }

    #[inline]
    fn check_time(&mut self) {
        if self.nodes & 1023 != 0 {
            return;
        }
        let total = self.shared.nodes.fetch_add(1024, Ordering::Relaxed) + 1024;
        if self.shared.stop.load(Ordering::Relaxed) {
            self.stopped = true;
            return;
        }
        if self.helper_idx > 0 {
            if self.shared.helpers_stop.load(Ordering::Relaxed) {
                self.stopped = true;
            }
            return;
        }
        if self.limits.nodes > 0 && total >= self.limits.nodes {
            self.stopped = true;
            return;
        }
        if !self.shared.ponder.load(Ordering::Relaxed) && self.elapsed_ms() >= self.limits.hard_ms {
            self.stopped = true;
        }
    }

    fn update_pv(&mut self, ply: usize, m: Move) {
        let child_len = self.pv_len[ply + 1];
        let (head, tail) = self.pv.split_at_mut((ply + 1) * PV_STRIDE);
        let row = &mut head[ply * PV_STRIDE..];
        row[0] = m;
        row[1..=child_len].copy_from_slice(&tail[..child_len]);
        self.pv_len[ply] = child_len + 1;
    }

    /// 探索を行い、bestmove を出力する。
    pub fn think(&mut self, pos: &mut Position) {
        self.shared.nodes.store(0, Ordering::Relaxed);
        self.shared.helpers_stop.store(false, Ordering::Relaxed);
        self.feat = FEAT.load(Ordering::Relaxed);
        self.set_network(if self.feat & F_NN != 0 { nn::global() } else { None });
        self.history.iter_mut().for_each(|h| *h /= 4);

        let root_moves = pos.legal_moves();
        if root_moves.is_empty() {
            self.wait_for_stop();
            println!("bestmove resign");
            return;
        }

        // 指せる手が 1 つなら補助スレッドは要らない
        let n_helpers = if root_moves.len() > 1 { self.threads.saturating_sub(1) } else { 0 };
        let (best, ponder) = std::thread::scope(|s| {
            for idx in 1..=n_helpers {
                let mut helper = self.make_helper(idx);
                let mut hp = pos.clone();
                let roots = root_moves.clone();
                std::thread::Builder::new()
                    .stack_size(64 << 20)
                    .spawn_scoped(s, move || {
                        helper.search(&mut hp, &roots);
                    })
                    .expect("failed to spawn helper thread");
            }
            let r = self.search(pos, &root_moves);
            self.shared.helpers_stop.store(true, Ordering::Relaxed);
            r
        });

        self.wait_for_stop();
        if ponder != MOVE_NONE {
            println!("bestmove {} ponder {}", move_to_usi(best), move_to_usi(ponder));
        } else {
            println!("bestmove {}", move_to_usi(best));
        }
    }

    /// 反復深化。主スレッドは info を出して時間を管理し、補助スレッドは置換表を温めるために深さをずらして読む。
    fn search(&mut self, pos: &mut Position, root_moves: &[Move]) -> (Move, Move) {
        self.nodes = 0;
        self.stopped = false;
        self.killers = [[MOVE_NONE; 2]; MAX_PLY + 2];
        let is_main = self.helper_idx == 0;
        if let Some(nn) = &self.nn {
            nn.refresh(pos, &mut self.accs[0]);
        }

        let mut best = root_moves[0];
        let mut ponder = MOVE_NONE;
        let max_depth = if self.limits.depth > 0 { self.limits.depth } else { MAX_PLY as i32 - 8 };
        let mut prev_score: i32 = 0;

        for depth in 1..=max_depth {
            if !is_main {
                let i = (self.helper_idx - 1) % 20;
                if ((depth + SKIP_PHASE[i]) / SKIP_SIZE[i]) % 2 == 1 {
                    continue;
                }
            }
            self.sel_depth = 0;

            // アスピレーションウィンドウ
            let mut delta = 60;
            let (mut alpha, mut beta) = if depth >= 4 && prev_score.abs() < MATE_BOUND {
                (prev_score - delta, prev_score + delta)
            } else {
                (-INF, INF)
            };
            let score = loop {
                let s = self.negamax(pos, alpha, beta, depth, 0, false);
                if self.stopped {
                    break s;
                }
                if s <= alpha {
                    alpha = (alpha - delta).max(-INF);
                } else if s >= beta {
                    beta = (beta + delta).min(INF);
                } else {
                    break s;
                }
                delta *= 2;
                if delta > 1000 {
                    alpha = -INF;
                    beta = INF;
                }
            };

            if self.stopped && depth > 1 {
                break;
            }
            prev_score = score;
            if self.pv_len[0] > 0 {
                best = self.pv[0];
                ponder = if self.pv_len[0] > 1 { self.pv[1] } else { MOVE_NONE };
            }

            if !is_main {
                continue;
            }
            let ms = self.elapsed_ms();
            self.print_info(depth, 1, score, ms);

            // MultiPV: 最善手を除外して 2 番目以降の候補手を探す
            if self.multipv > 1 && !self.stopped {
                self.root_excluded.clear();
                self.root_excluded.push(best);
                for k in 1..self.multipv.min(root_moves.len()) {
                    self.sel_depth = 0;
                    let s = self.negamax(pos, -INF, INF, depth, 0, false);
                    if self.stopped || self.pv_len[0] == 0 {
                        break;
                    }
                    self.root_excluded.push(self.pv[0]);
                    let ms = self.elapsed_ms();
                    self.print_info(depth, k + 1, s, ms);
                }
                self.root_excluded.clear();
            }

            if self.stopped {
                break;
            }
            let pondering = self.shared.ponder.load(Ordering::Relaxed);
            if !self.limits.infinite && !pondering {
                if ms >= self.limits.soft_ms || root_moves.len() == 1 {
                    break;
                }
                // 詰みを読み切った
                if score.abs() >= MATE_BOUND && depth > MATE - score.abs() + 2 {
                    break;
                }
            }
        }

        (best, ponder)
    }

    fn print_info(&self, depth: i32, multipv: usize, score: i32, ms: u64) {
        let pv: Vec<String> = self.pv[..self.pv_len[0]].iter().map(|&m| move_to_usi(m)).collect();
        let mpv = if self.multipv > 1 { format!(" multipv {}", multipv) } else { String::new() };
        // 全スレッドの合計（共有の合計は 1024 ノード単位なので、主スレッドの端数を足す）
        let total_nodes = self.shared.nodes.load(Ordering::Relaxed) + (self.nodes & 1023);
        println!(
            "info depth {} seldepth {}{} score {} nodes {} nps {} time {} hashfull {} pv {}",
            depth,
            self.sel_depth,
            mpv,
            score_to_usi(score),
            total_nodes,
            total_nodes * 1000 / ms.max(1),
            ms,
            self.tt.hashfull(),
            pv.join(" ")
        );
    }

    /// go infinite / go ponder 中は stop（または ponderhit）が来るまで bestmove を返さない
    fn wait_for_stop(&self) {
        while (self.limits.infinite || self.shared.ponder.load(Ordering::Relaxed))
            && !self.shared.stop.load(Ordering::Relaxed)
        {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    /// 直前の手に対するカウンタームーブ表の添字
    #[inline]
    fn counter_index(pos: &Position) -> Option<usize> {
        let prev = pos.last_move();
        if prev == MOVE_NONE {
            return None;
        }
        let to = mv_to(prev);
        Some(pos.board[to] as usize * 81 + to)
    }

    fn order_moves(&self, pos: &Position, list: &MoveList, scores: &mut [i32], tt_move: Move, ply: usize) {
        let counter = Self::counter_index(pos).map(|i| self.counter[i]).unwrap_or(MOVE_NONE);
        for i in 0..list.len {
            let m = list.moves[i];
            scores[i] = if m == tt_move {
                1_000_000
            } else if !mv_is_drop(m) && pos.board[mv_to(m)] != 0 {
                let victim = PIECE_VALUE[type_of(pos.board[mv_to(m)]) as usize];
                let attacker = PIECE_VALUE[type_of(pos.board[mv_from(m)]) as usize];
                let base = victim * 16 - attacker / 16 + if mv_is_promo(m) { 1000 } else { 0 };
                // 取り合いで損をする駒取りは、静かな手より後ろに回す
                if pos.see(m) >= 0 { 200_000 + base } else { -60_000 + base }
            } else if mv_is_promo(m) {
                150_000
            } else if m == self.killers[ply][0] {
                120_000
            } else if m == self.killers[ply][1] {
                110_000
            } else if m == counter {
                100_000
            } else {
                let piece = if mv_is_drop(m) {
                    make_piece(pos.side, mv_drop_pt(m))
                } else {
                    pos.board[mv_from(m)]
                };
                self.history[piece as usize * 81 + mv_to(m)]
            };
        }
    }

    #[inline]
    fn pick_move(list: &mut MoveList, scores: &mut [i32], i: usize) -> Move {
        let mut best = i;
        for j in i + 1..list.len {
            if scores[j] > scores[best] {
                best = j;
            }
        }
        list.moves.swap(i, best);
        scores.swap(i, best);
        list.moves[i]
    }

    fn negamax(
        &mut self,
        pos: &mut Position,
        mut alpha: i32,
        mut beta: i32,
        mut depth: i32,
        ply: usize,
        null_ok: bool,
    ) -> i32 {
        let pv_node = beta - alpha > 1;
        self.pv_len[ply] = 0;
        if depth <= 0 {
            return self.qsearch(pos, alpha, beta, ply);
        }
        self.nodes += 1;
        self.check_time();
        if self.stopped {
            return 0;
        }
        self.sel_depth = self.sel_depth.max(ply);

        if ply > 0 {
            if pos.is_repetition() {
                return 0;
            }
            // 詰み距離による枝刈り
            alpha = alpha.max(-MATE + ply as i32);
            beta = beta.min(MATE - ply as i32 - 1);
            if alpha >= beta {
                return alpha;
            }
        }
        if ply >= MAX_PLY - 4 {
            self.ensure_acc(pos, ply);
            return self.eval(pos, ply);
        }

        let key = pos.key;
        let mut tt_move = MOVE_NONE;
        if let Some(e) = self.tt.probe(key) {
            tt_move = e.mv;
            if !pv_node && ply > 0 && e.depth as i32 >= depth {
                let s = score_from_tt(e.score as i32, ply);
                if e.flag == BOUND_EXACT
                    || (e.flag == BOUND_LOWER && s >= beta)
                    || (e.flag == BOUND_UPPER && s <= alpha)
                {
                    return s;
                }
            }
        }

        // ここから先でこの局面のアキュムレータが要る（評価と、子の局面への更新の元になる）
        self.ensure_acc(pos, ply);
        let in_check = pos.in_check();
        let static_eval = if in_check { -INF } else { self.eval(pos, ply) };
        self.static_evals[ply] = static_eval;
        // 2 手前（自分の手番）より評価値が良くなっているか（わからないときは良くなっている扱い）
        let imp_on = self.feat & F_IMPROVING != 0;
        let improving = ply < 2 || self.static_evals[ply - 2] == -INF || static_eval > self.static_evals[ply - 2];

        // Internal Iterative Reduction: 置換表に手がない局面は、深く読んでも並べ替えが効かないので浅くする
        if self.feat & F_IIR != 0 && tt_move == MOVE_NONE && depth >= 4 && !in_check {
            depth -= 1;
        }

        if !pv_node && !in_check && ply > 0 {
            // Reverse futility pruning
            let rfp_depth = if imp_on && improving { depth - 1 } else { depth };
            if depth <= 3 && static_eval - 150 * rfp_depth >= beta && static_eval.abs() < MATE_BOUND {
                return static_eval;
            }
            // Null move pruning
            if null_ok && depth >= 3 && static_eval >= beta {
                let r = if self.feat & F_NULL2 != 0 {
                    3 + depth / 3 + ((static_eval - beta) / 200).clamp(0, 2)
                } else {
                    2 + depth / 4
                };
                self.set_pending_null(ply);
                pos.do_null();
                let s = -self.negamax(pos, -beta, -beta + 1, depth - 1 - r, ply + 1, false);
                pos.undo_null();
                if self.stopped {
                    return 0;
                }
                if s >= beta {
                    return if s >= MATE_BOUND { beta } else { s };
                }
            }
        }

        // 王手延長
        if in_check {
            depth += 1;
        }

        let mut list = MoveList::new();
        pos.gen_moves(&mut list, false);
        let mut scores = [0i32; MAX_MOVES];
        self.order_moves(pos, &list, &mut scores, tt_move, ply);

        let us = pos.side;
        let orig_alpha = alpha;
        let mut best = -INF;
        let mut best_move = MOVE_NONE;
        let mut legal = 0;
        // 試して失敗した静かな手（β カット時にヒストリーを下げる）
        let mut quiets_tried = [(0u8, 0usize); 64];
        let mut n_quiets = 0;

        for i in 0..list.len {
            let m = Self::pick_move(&mut list, &mut scores, i);
            let to = mv_to(m);
            let is_cap = !mv_is_drop(m) && pos.board[to] != 0;
            let is_promo = mv_is_promo(m);
            let quiet = !is_cap && !is_promo;
            let piece = if mv_is_drop(m) { make_piece(us, mv_drop_pt(m)) } else { pos.board[mv_from(m)] };

            if ply == 0 && self.root_excluded.contains(&m) {
                continue;
            }
            self.set_pending(pos, m, ply);
            if !pos.do_move_legal(m) {
                continue;
            }
            legal += 1;
            let gives_check = pos.in_check();

            if !pv_node && !in_check && !gives_check && quiet && best > -MATE_BOUND {
                // Late move pruning（浅い深さでの後半の静かな手を省略）
                let lmp_base = 4 + depth * depth * 3;
                let lmp_limit = if imp_on && !improving { lmp_base * 2 / 3 } else { lmp_base };
                if depth <= 3 && legal > lmp_limit {
                    pos.undo_move();
                    continue;
                }
                // Futility pruning（評価値が α に遠く届かない局面の静かな手を省略）
                if depth <= 3 && legal > 1 && static_eval + 110 + 100 * depth <= alpha {
                    pos.undo_move();
                    continue;
                }
            }
            if quiet && n_quiets < 64 {
                quiets_tried[n_quiets] = (piece, to);
                n_quiets += 1;
            }

            let new_depth = depth - 1;
            let mut s;
            if legal == 1 {
                s = -self.negamax(pos, -beta, -alpha, new_depth, ply + 1, true);
            } else {
                let mut r = 0;
                if depth >= 3 && legal > 2 && quiet && !in_check && !gives_check {
                    r = self.lmr[(depth.min(63) as usize) * 64 + legal.min(63) as usize];
                    if pv_node {
                        r -= 1;
                    }
                    if imp_on && !improving {
                        r += 1;
                    }
                    // ヒストリーの良い手は浅くしすぎない
                    if self.history[piece as usize * 81 + to] > 4000 {
                        r -= 1;
                    }
                    r = r.clamp(0, (new_depth - 1).max(0));
                }
                s = -self.negamax(pos, -alpha - 1, -alpha, new_depth - r, ply + 1, true);
                if s > alpha && r > 0 {
                    s = -self.negamax(pos, -alpha - 1, -alpha, new_depth, ply + 1, true);
                }
                if s > alpha && s < beta {
                    s = -self.negamax(pos, -beta, -alpha, new_depth, ply + 1, true);
                }
            }
            pos.undo_move();
            if self.stopped {
                return 0;
            }

            if s > best {
                best = s;
                best_move = m;
                if s > alpha {
                    alpha = s;
                    self.update_pv(ply, m);
                    if s >= beta {
                        if quiet {
                            if self.killers[ply][0] != m {
                                self.killers[ply][1] = self.killers[ply][0];
                                self.killers[ply][0] = m;
                            }
                            if let Some(ci) = Self::counter_index(pos) {
                                self.counter[ci] = m;
                            }
                            // ヒストリー: カットした手を上げ、先に試して失敗した手を下げる
                            let bonus = (depth * depth).min(400);
                            for k in 0..n_quiets {
                                let (p, t) = quiets_tried[k];
                                let h = &mut self.history[p as usize * 81 + t];
                                let b = if p == piece && t == to { bonus } else { -bonus };
                                *h += b - *h * b.abs() / 16384;
                            }
                        }
                        break;
                    }
                }
            }
        }

        if legal == 0 {
            // 将棋では指し手が無ければ負け
            return -MATE + ply as i32;
        }

        let flag = if best >= beta {
            BOUND_LOWER
        } else if best > orig_alpha {
            BOUND_EXACT
        } else {
            BOUND_UPPER
        };
        self.tt.store(key, best_move, score_to_tt(best, ply), depth, flag);
        best
    }

    fn qsearch(&mut self, pos: &mut Position, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        self.pv_len[ply] = 0;
        self.nodes += 1;
        self.check_time();
        if self.stopped {
            return 0;
        }
        self.sel_depth = self.sel_depth.max(ply);
        if ply >= MAX_PLY - 4 {
            self.ensure_acc(pos, ply);
            return self.eval(pos, ply);
        }

        let orig_alpha = alpha;
        let pv_node = beta - alpha > 1;
        let key = pos.key;
        let use_tt = self.feat & F_QS_TT != 0;
        let mut tt_move = MOVE_NONE;
        if use_tt {
            if let Some(e) = self.tt.probe(key) {
                tt_move = e.mv;
                if !pv_node {
                    let s = score_from_tt(e.score as i32, ply);
                    if e.flag == BOUND_EXACT
                        || (e.flag == BOUND_LOWER && s >= beta)
                        || (e.flag == BOUND_UPPER && s <= alpha)
                    {
                        return s;
                    }
                }
            }
        }

        self.ensure_acc(pos, ply);
        let in_check = pos.in_check();
        let mut best_move = MOVE_NONE;
        let mut list = MoveList::new();
        let mut best;
        if in_check {
            best = -INF;
            pos.gen_moves(&mut list, false);
        } else {
            let stand = self.eval(pos, ply);
            if stand >= beta {
                return stand;
            }
            alpha = alpha.max(stand);
            best = stand;
            pos.gen_moves(&mut list, true);
        }

        let mut scores = [0i32; MAX_MOVES];
        self.order_moves(pos, &list, &mut scores, tt_move, ply);

        let mut legal = 0;
        for i in 0..list.len {
            let m = Self::pick_move(&mut list, &mut scores, i);
            if !in_check {
                // Delta pruning
                let gain = PIECE_VALUE[type_of(pos.board[mv_to(m)]) as usize] + if mv_is_promo(m) { 600 } else { 0 };
                if best + gain + 200 < alpha {
                    continue;
                }
                // 取り合いで損をする駒取りは読まない
                if pos.see(m) < 0 {
                    continue;
                }
            }
            self.set_pending(pos, m, ply);
            if !pos.do_move_legal(m) {
                continue;
            }
            legal += 1;
            let s = -self.qsearch(pos, -beta, -alpha, ply + 1);
            pos.undo_move();
            if self.stopped {
                return 0;
            }
            if s > best {
                best = s;
                best_move = m;
                if s > alpha {
                    alpha = s;
                    self.update_pv(ply, m);
                    if s >= beta {
                        break;
                    }
                }
            }
        }

        if in_check && legal == 0 {
            return -MATE + ply as i32;
        }
        if use_tt {
            let flag = if best >= beta {
                BOUND_LOWER
            } else if best > orig_alpha {
                BOUND_EXACT
            } else {
                BOUND_UPPER
            };
            self.tt.store_qs(key, best_move, score_to_tt(best, ply), flag);
        }
        best
    }
}
