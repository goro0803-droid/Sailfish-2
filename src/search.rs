//! 探索部: 反復深化 + PVS(αβ) + 静止探索。
//! 置換表・キラー・ヒストリー・ヌルムーブ・LMR・王手延長を使用。

use crate::eval::{evaluate, PIECE_VALUE};
use crate::nn::{self, Acc, Network};
use crate::position::{MoveList, Position, CAPTURE_VALUE, MAX_MOVES};
use crate::tt::*;
use crate::types::*;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

/// 探索の改良を個別に入れ替えるためのビット（USI の隠しオプション Feat で指定。強さの比較用）
pub static FEAT: AtomicU32 = AtomicU32::new(F_QS_TT | F_NN | F_ECACHE | F_DELTA_HAND);
/// 静止探索でも置換表を使う
pub const F_QS_TT: u32 = 1;
/// 置換表に手がないとき、深さを 1 減らす（Internal Iterative Reduction）
pub const F_IIR: u32 = 2;
/// 評価値が 2 手前より良くなっているかで、枝刈りと LMR の強さを変える
pub const F_IMPROVING: u32 = 4;
/// ヌルムーブの減らし方を、深さと評価値の余裕に応じて大きくする
pub const F_NULL2: u32 = 8;
/// 手作り評価を局面ごとに覚える（NN の出力と差分更新は毎回計算する）
pub const F_ECACHE: u32 = 16;
/// 手作りの評価関数のかわりに、ニューラルネットワーク（EvalFile で読み込んだもの）で評価する
pub const F_NN: u32 = 32;
// Diagnostic ablations only; default FEAT sets none of these bits.
pub const D_NO_LMR: u32 = 64;
pub const D_NO_NULL: u32 = 128;
pub const D_NO_RFP: u32 = 256;
pub const D_NO_LMP: u32 = 512;
pub const D_NO_FUTILITY: u32 = 1024;
pub const D_NO_QSEE: u32 = 2048;
pub const D_NO_QDELTA: u32 = 4096;
/// Preserve checking captures on the principal quiescence line.
pub const F_QCHECK_CAPTURE: u32 = 8192;
/// Search quiet checks on the principal line, for at most the first three QS plies.
pub const F_QCHECK_QUIET: u32 = 16384;
/// Exact mate-in-one before stand-pat at the first quiescence ply.
pub const F_QMATE1: u32 = 32768;
pub const F_SEE_HAND: u32 = 65536;
pub const F_DELTA_HAND: u32 = 131072;
/// Correct exchange values for move ordering without changing QS pruning.
pub const F_SEE_HAND_ORDER: u32 = 262144;

/// Extend a reliable TT move only when excluding it fails below a margin.
pub const F_SINGULAR: u32 = 8388608;

/// Reduce LMR for safe escapes and defenses of attacked, undefended pieces.
pub const F_TACTICAL_GUARD: u32 = 16777216;
/// Also preserve those replies from shallow quiet-move pruning (requires GUARD).
pub const F_TACTICAL_PRUNE: u32 = 33554432;

/// Experimental root ordering for non-losing quiet checks with few replies.
pub const F_GOAL_ORDER: u32 = 67108864;

/// 王手にならない駒打ちで、相手の銀以上の駒に当たる手（脅威を作る駒打ち）は、LMR で浅くする量を 1 減らす（実験用、既定は無効）
pub const F_THREAT_DROP: u32 = 134217728;

/// 升 sq の駒が、相手の駒のうち駒の価値が銀以上のもの（玉は除く）に直接利いているか
fn threatens_enemy_piece(pos: &Position, sq: usize) -> bool {
    let p = pos.board[sq];
    if p == 0 { return false; }
    let (c, pt) = (color_of(p), type_of(p) as usize);
    let (step, slide) = (STEP_MASK[pt], SLIDE_MASK[pt]);
    for d in 0..10 {
        if (step | slide) & (1 << d) == 0 { continue; }
        let mut s = sq;
        loop {
            let n = NEIGHBOR[c][s][d];
            if n < 0 { break; }
            s = n as usize;
            let q = pos.board[s];
            if q != 0 {
                if color_of(q) != c && PIECE_VALUE[type_of(q) as usize] >= PIECE_VALUE[SILVER as usize] { return true; }
                break;
            }
            if d >= 8 || slide & (1 << d) == 0 { break; }
        }
    }
    false
}

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
    /// 打ち切りの上限（hard_ms）を数え始めた時刻。ふだんは start_us と同じで、PonderTime=total では ponderhit の時刻
    pub hard_from_us: AtomicU64,
    /// この探索の soft_ms（ponderhit のときに USI の側で見る）
    pub soft_ms: AtomicU64,
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
            hard_from_us: AtomicU64::new(0),
            soft_ms: AtomicU64::new(0),
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
    /// Predict whether the next iteration fits the clock target (increment play only).
    pub adaptive_time: bool,
}

impl Default for Limits {
    fn default() -> Self {
        Limits { soft_ms: u64::MAX, hard_ms: u64::MAX, depth: 0, nodes: 0, infinite: false, adaptive_time: false }
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
    /// キャッシュした土台評価に対応する USI 評価オプション。
    ecache_params: (i32, i32),
    /// ニューラルネットワーク（F_NN が有効で、読み込み済みのとき）
    nn: Option<Arc<Network>>,
    /// 各 ply の局面に対応するアキュムレータ（nn があるときだけ使う）
    accs: Vec<Acc>,
    /// 各 ply の局面に至った手と、そのとき取った駒（アキュムレータの更新待ち）
    pend: Vec<(Move, u8)>,
    root_excluded: Vec<Move>,
    root_goal_moves: Vec<Move>,
    iteration_depth: i32,
    singular_verifying: bool,
    singular_probes: u64,
    singular_extensions: u64,
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

// Attack maps are only a tactical hint; king legality remains the move
// generator's responsibility. Source-square tests are cached per search node.
struct TacticalThreats {
    checked: u128,
    hanging: u128,
    majors: u128,
}

impl TacticalThreats {
    fn new(pos: &Position, enabled: bool) -> Self {
        let mut threats = Self { checked: 0, hanging: 0, majors: 0 };
        if enabled {
            for sq in 0..81 {
                let p = pos.board[sq];
                if p != 0 && color_of(p) == pos.side
                    && matches!(unpromote(type_of(p)), BISHOP | ROOK)
                {
                    if threats.hanging_from(pos, sq) { threats.majors |= 1u128 << sq; }
                }
            }
        }
        threats
    }

    fn hanging_from(&mut self, pos: &Position, sq: usize) -> bool {
        let bit = 1u128 << sq;
        if self.checked & bit == 0 {
            self.checked |= bit;
            let p = pos.board[sq];
            if p != 0 && color_of(p) == pos.side && type_of(p) != KING
                && PIECE_VALUE[type_of(p) as usize] >= PIECE_VALUE[KNIGHT as usize]
                && pos.attacked(sq, pos.side ^ 1) && !pos.attacked(sq, pos.side)
            { self.hanging |= bit; }
        }
        self.hanging & bit != 0
    }

    // Called after a legal move, with its pre-move SEE >= 0 already checked.
    fn protects_after(&self, pos: &Position, us: usize, escape: bool) -> bool {
        if escape { return true; }
        let mut targets = self.majors;
        while targets != 0 {
            let sq = targets.trailing_zeros() as usize;
            targets &= targets - 1;
            let p = pos.board[sq];
            if p != 0 && color_of(p) == us
                && matches!(unpromote(type_of(p)), BISHOP | ROOK)
                && (pos.attacked(sq, us) || !pos.attacked(sq, us ^ 1))
            { return true; }
        }
        false
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
            ecache: vec![(0, i32::MIN); 1 << ECACHE_BITS],
            ecache_params: (
                crate::eval::KING_DANGER_PCT.load(Ordering::Relaxed),
                crate::eval::MOBILITY_WEIGHT.load(Ordering::Relaxed),
            ),
            nn: None,
            accs: Vec::new(),
            pend: Vec::new(),
            root_excluded: Vec::new(),
            root_goal_moves: Vec::new(),
            iteration_depth: 0,
            singular_verifying: false,
            singular_probes: 0,
            singular_extensions: 0,
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

    /// 打ち切りの上限（hard_ms）と比べる経過時間
    fn hard_elapsed_ms(&self) -> u64 {
        now_us().saturating_sub(self.shared.hard_from_us.load(Ordering::Relaxed)) / 1000
    }

    /// NN の土台の手作り評価だけを再利用する。差分更新の浮動小数点値はキャッシュしない。
    #[inline]
    fn eval(&mut self, pos: &Position, ply: usize) -> i32 {
        if self.feat & F_ECACHE != 0 && self.nn.as_ref().is_some_and(|net| net.base_hc) {
            let base = self.cached_base_eval(pos);
            return self.nn.as_ref().unwrap().forward_with_base(&self.accs[ply], pos.side, base);
        }
        if let Some(nn) = &self.nn {
            return nn.forward(&self.accs[ply], pos);
        }
        self.cached_base_eval(pos)
    }

    #[inline]
    fn cached_base_eval(&mut self, pos: &Position) -> i32 {
        if self.feat & F_ECACHE == 0 {
            return evaluate(pos);
        }
        let i = pos.key as usize & ((1 << ECACHE_BITS) - 1);
        let e = self.ecache[i];
        if e.0 == pos.key && e.1 != i32::MIN {
            return e.1;
        }
        let v = evaluate(pos);
        self.ecache[i] = (pos.key, v);
        v
    }

    // 局面キーは手番・持ち駒を含む。探索間でも土台評価を再利用できる。
    // 評価オプションが変わったときだけ無効化する。
    fn sync_eval_cache(&mut self) {
        let params = (
            crate::eval::KING_DANGER_PCT.load(Ordering::Relaxed),
            crate::eval::MOBILITY_WEIGHT.load(Ordering::Relaxed),
        );
        if params != self.ecache_params {
            self.ecache.fill((0, i32::MIN));
            self.ecache_params = params;
        }
    }

    /// 補助スレッド用の探索状態を作る（置換表と停止フラグは主スレッドと共有）
    fn make_helper(&self, idx: usize) -> Searcher {
        let mut h = Searcher::with_tt(self.tt.clone(), self.shared.clone());
        h.limits = self.limits;
        h.helper_idx = idx;
        h.feat = self.feat;
        h.root_goal_moves = self.root_goal_moves.clone();
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
                b[0].copy_from(&a[ply - 1], nn.q.is_some());
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
        if !self.shared.ponder.load(Ordering::Relaxed) && self.hard_elapsed_ms() >= self.limits.hard_ms {
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
        self.sync_eval_cache();
        self.set_network(if self.feat & F_NN != 0 { nn::global() } else { None });
        self.history.iter_mut().for_each(|h| *h /= 4);

        let root_moves = pos.legal_moves();
        self.prepare_goal_moves(pos, &root_moves);
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
        self.singular_verifying = false;
        self.singular_probes = 0;
        self.singular_extensions = 0;
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
        let mut previous_iteration_ms = 0;
        let mut stable_iterations = 0;

        for depth in 1..=max_depth {
            if !is_main {
                let i = (self.helper_idx - 1) % 20;
                if ((depth + SKIP_PHASE[i]) / SKIP_SIZE[i]) % 2 == 1 {
                    continue;
                }
            }
            self.iteration_depth = depth;
            let iteration_start = self.elapsed_ms();
            let prior_score = prev_score;
            let prior_best = best;
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
            let iteration_ms = self.elapsed_ms().saturating_sub(iteration_start);
            stable_iterations = if best == prior_best && (score-prior_score).abs()<80 { stable_iterations+1 } else { 0 };
            let predicted_stop = self.limits.adaptive_time && depth >= 5 && crate::time_control::stop_before_next(self.elapsed_ms(), iteration_ms, previous_iteration_ms, self.limits.soft_ms, self.limits.hard_ms, stable_iterations>=2);
            previous_iteration_ms = iteration_ms;
            let pondering = self.shared.ponder.load(Ordering::Relaxed);
            if !self.limits.infinite && !pondering {
                if ms >= self.limits.soft_ms || predicted_stop || root_moves.len() == 1 {
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

    #[inline]
    fn exchange_score(&self, pos: &Position, m: Move) -> i32 {
        if self.feat & F_SEE_HAND != 0 { pos.see_with_hand(m) } else { pos.see(m) }
    }

    // Only prioritize a hypothesis; the ordinary search must still refute it.
    // Include captures, blocks and king escapes among all legal replies.
    fn prepare_goal_moves(&mut self, pos: &mut Position, root_moves: &[Move]) {
        self.root_goal_moves.clear();
        if self.feat & F_GOAL_ORDER == 0 { return; }
        for &m in root_moves {
            if self.shared.stop.load(Ordering::Relaxed)
                || (!self.shared.ponder.load(Ordering::Relaxed)
                    && self.hard_elapsed_ms() >= self.limits.hard_ms) { break; }
            if mv_is_promo(m) || (!mv_is_drop(m) && pos.board[mv_to(m)] != 0)
                || pos.see(m) < 0 || !pos.gives_check(m) { continue; }
            if !pos.do_move_legal(m) { continue; }
            let mut list = MoveList::new();
            pos.gen_moves(&mut list, false);
            let mut replies = 0;
            for i in 0..list.len {
                if pos.do_move_legal(list.moves[i]) {
                    pos.undo_move();
                    replies += 1;
                    if replies > 2 { break; }
                }
            }
            pos.undo_move();
            if replies <= 2 { self.root_goal_moves.push(m); }
        }
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
                let see = if self.feat & (F_SEE_HAND | F_SEE_HAND_ORDER) != 0 {
                    pos.see_with_hand(m)
                } else { pos.see(m) };
                if see >= 0 { 200_000 + base } else { -60_000 + base }
            } else if mv_is_promo(m) {
                150_000
            } else if ply == 0 && self.feat & F_GOAL_ORDER != 0
                && self.root_goal_moves.contains(&m) {
                125_000
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

    fn negamax(&mut self, pos: &mut Position, alpha: i32, beta: i32,
               depth: i32, ply: usize, null_ok: bool) -> i32 {
        self.negamax_excluding(pos, alpha, beta, depth, ply, null_ok, MOVE_NONE)
    }

    // The excluded root is a different search problem: never reuse its normal
    // TT bound, store its result under the normal key, or treat no alternatives
    // as checkmate. Descendants still search ordinary legal positions.
    fn negamax_excluding(
        &mut self,
        pos: &mut Position,
        mut alpha: i32,
        mut beta: i32,
        mut depth: i32,
        ply: usize,
        null_ok: bool,
        excluded: Move,
    ) -> i32 {
        let pv_node = beta - alpha > 1;
        self.pv_len[ply] = 0;
        if depth <= 0 {
            return self.qsearch(pos, alpha, beta, ply, 0);
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
        let tt_entry = if excluded == MOVE_NONE { self.tt.probe(key) } else { None };
        if let Some(e) = tt_entry {
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
        if excluded == MOVE_NONE && self.feat & F_IIR != 0 && tt_move == MOVE_NONE && depth >= 4 && !in_check {
            depth -= 1;
        }

        if excluded == MOVE_NONE && !pv_node && !in_check && ply > 0 {
            // Reverse futility pruning
            let rfp_depth = if imp_on && improving { depth - 1 } else { depth };
            if self.feat & D_NO_RFP == 0 && depth <= 3 && static_eval - 150 * rfp_depth >= beta && static_eval.abs() < MATE_BOUND {
                return static_eval;
            }
            // Null move pruning
            if self.feat & D_NO_NULL == 0 && null_ok && depth >= 3 && static_eval >= beta {
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

        // A single-ply extension, bounded to twice the root iteration horizon.
        // Verification has half depth and excludes the TT move explicitly.
        // Nested singular verification is disabled throughout that subtree.
        let mut singular_move = MOVE_NONE;
        if self.feat & F_SINGULAR != 0 && excluded == MOVE_NONE
            && !self.singular_verifying && !in_check && ply > 0 && depth >= 7
            && ply < (2 * self.iteration_depth) as usize
        {
            if let Some(e) = tt_entry {
                let tt_score = score_from_tt(e.score as i32, ply);
                if tt_move != MOVE_NONE && e.depth as i32 >= depth - 3
                    && (e.flag == BOUND_LOWER || e.flag == BOUND_EXACT)
                    && tt_score.abs() < MATE_BOUND
                {
                    // Do not verify a stale or illegal hash move.
                    let mut candidates = MoveList::new();
                    pos.gen_moves(&mut candidates, false);
                    if candidates.moves[..candidates.len].contains(&tt_move)
                        && pos.do_move_legal(tt_move)
                    {
                        pos.undo_move();
                        self.singular_probes += 1;
                        self.singular_verifying = true;
                        let threshold = tt_score - 3 * depth;
                        let alternative = self.negamax_excluding(pos, threshold - 1,
                            threshold, (depth - 1) / 2, ply, false, tt_move);
                        self.singular_verifying = false;
                        self.pv_len[ply] = 0;
                        self.static_evals[ply] = static_eval;
                        if self.stopped { return 0; }
                        if alternative < threshold {
                            singular_move = tt_move;
                            self.singular_extensions += 1;
                        }
                    }
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
        let guard_on = self.feat & F_TACTICAL_GUARD != 0 && excluded == MOVE_NONE
            && !in_check && depth >= 2;
        let mut threats = TacticalThreats::new(pos, guard_on);
        let orig_alpha = alpha;
        let mut best = -INF;
        let mut best_move = MOVE_NONE;
        let mut legal = 0;
        // 試して失敗した静かな手（β カット時にヒストリーを下げる）
        let mut quiets_tried = [(0u8, 0usize); 64];
        let mut n_quiets = 0;

        for i in 0..list.len {
            let m = Self::pick_move(&mut list, &mut scores, i);
            if m == excluded { continue; }
            let to = mv_to(m);
            let is_cap = !mv_is_drop(m) && pos.board[to] != 0;
            let is_promo = mv_is_promo(m);
            let quiet = !is_cap && !is_promo;
            let piece = if mv_is_drop(m) { make_piece(us, mv_drop_pt(m)) } else { pos.board[mv_from(m)] };

            if ply == 0 && self.root_excluded.contains(&m) {
                continue;
            }
            let escape = guard_on && quiet && !mv_is_drop(m)
                && threats.hanging_from(pos, mv_from(m));
            let safe_reply = guard_on && quiet && (escape || threats.majors != 0)
                && pos.see(m) >= 0;
            self.set_pending(pos, m, ply);
            if !pos.do_move_legal(m) {
                continue;
            }
            legal += 1;
            let gives_check = pos.in_check();
            let protected_reply = safe_reply && threats.protects_after(pos, us, escape);
            let threat_drop = self.feat & F_THREAT_DROP != 0 && mv_is_drop(m) && !gives_check
                && threatens_enemy_piece(pos, to);

            if !(protected_reply && self.feat & F_TACTICAL_PRUNE != 0)
                && excluded == MOVE_NONE && !pv_node && !in_check && !gives_check && quiet && best > -MATE_BOUND {
                // Late move pruning（浅い深さでの後半の静かな手を省略）
                let lmp_base = 4 + depth * depth * 3;
                let lmp_limit = if imp_on && !improving { lmp_base * 2 / 3 } else { lmp_base };
                if self.feat & D_NO_LMP == 0 && depth <= 3 && legal > lmp_limit {
                    pos.undo_move();
                    continue;
                }
                // Futility pruning（評価値が α に遠く届かない局面の静かな手を省略）
                if self.feat & D_NO_FUTILITY == 0 && depth <= 3 && legal > 1 && static_eval + 110 + 100 * depth <= alpha {
                    pos.undo_move();
                    continue;
                }
            }
            if quiet && n_quiets < 64 {
                quiets_tried[n_quiets] = (piece, to);
                n_quiets += 1;
            }

            let new_depth = depth - 1 + i32::from(m == singular_move);
            let mut s;
            if legal == 1 {
                s = -self.negamax(pos, -beta, -alpha, new_depth, ply + 1, true);
            } else {
                let mut r = 0;
                if excluded == MOVE_NONE && self.feat & D_NO_LMR == 0 && depth >= 3 && legal > 2 && quiet && !in_check && !gives_check {
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
                    if protected_reply { r -= 1; }
                    if threat_drop { r -= 1; }
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
                    if excluded == MOVE_NONE { self.update_pv(ply, m); }
                    if s >= beta {
                        if quiet && excluded == MOVE_NONE {
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
            if excluded != MOVE_NONE { return orig_alpha; }
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
        if excluded == MOVE_NONE {
            self.tt.store(key, best_move, score_to_tt(best, ply), depth, flag);
        }
        best
    }

    fn qsearch(&mut self, pos: &mut Position, mut alpha: i32, beta: i32, ply: usize, qdepth: usize) -> i32 {
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

        if self.feat & (F_QCHECK_CAPTURE | F_QCHECK_QUIET | F_QMATE1) != 0 && pos.is_repetition() { return 0; }
        let orig_alpha = alpha;
        let pv_node = beta - alpha > 1;
        let key = pos.key;
        // Entry-only quiet checks/mate verification have a different horizon.
        // Do not reuse or store capture-only QS bounds for that context.
        let extra_checks = qdepth < 3 && pv_node && self.feat & F_QCHECK_QUIET != 0;
        let mate_probe = qdepth == 0 && pv_node && self.feat & F_QMATE1 != 0;
        let use_tt = self.feat & F_QS_TT != 0 && !extra_checks && !mate_probe;
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
        if mate_probe {
            if let Some(m) = pos.mate_in_one() {
                self.pv[ply * PV_STRIDE] = m;
                self.pv_len[ply] = 1;
                return MATE - ply as i32 - 1;
            }
        }
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
            pos.gen_moves(&mut list, !extra_checks);
        }

        let mut scores = [0i32; MAX_MOVES];
        self.order_moves(pos, &list, &mut scores, tt_move, ply);

        let mut legal = 0;
        for i in 0..list.len {
            let m = Self::pick_move(&mut list, &mut scores, i);
            if !in_check {
                let captured = pos.board[mv_to(m)] != 0;
                let checking = (extra_checks
                    || (pv_node && self.feat & F_QCHECK_CAPTURE != 0))
                    && pos.gives_check(m);
                // Full generation at the entry includes quiet moves; only checks belong here.
                if !captured && !checking { continue; }
                // Delta pruning
                let gain = if self.feat & F_DELTA_HAND != 0 {
                    let captured = type_of(pos.board[mv_to(m)]) as usize;
                    let promotion = if mv_is_promo(m) {
                        let pt = type_of(pos.board[mv_from(m)]) as usize;
                        PIECE_VALUE[pt + PROMOTE as usize] - PIECE_VALUE[pt]
                    } else { 0 };
                    CAPTURE_VALUE[captured] + promotion
                } else {
                    PIECE_VALUE[type_of(pos.board[mv_to(m)]) as usize] + if mv_is_promo(m) { 600 } else { 0 }
                };
                if !checking && self.feat & D_NO_QDELTA == 0 && best + gain + 200 < alpha {
                    continue;
                }
                // 取り合いで損をする駒取りは読まない
                if !checking && self.feat & D_NO_QSEE == 0 && self.exchange_score(pos, m) < 0 {
                    continue;
                }
            }
            self.set_pending(pos, m, ply);
            if !pos.do_move_legal(m) {
                continue;
            }
            legal += 1;
            let s = -self.qsearch(pos, -beta, -alpha, ply + 1, qdepth + 1);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_neural_base_matches_uncached_evaluation() {
        let mut net = Network::random(71);
        net.base_hc = true;
        let net = Arc::new(net);
        let mut searcher = Searcher::new(1, Arc::new(Shared::new()));
        searcher.set_network(Some(net.clone()));
        searcher.feat = F_NN | F_ECACHE;
        let mut pos = Position::startpos();
        let mut rng = 918273645u64;
        for _ in 0..1000 {
            net.refresh(&pos, &mut searcher.accs[0]);
            let expected = net.forward(&searcher.accs[0], &pos);
            assert_eq!(searcher.eval(&pos, 0), expected);
            assert_eq!(searcher.eval(&pos, 0), expected);
            // The cache must not replace NN outputs from another accumulator.
            searcher.accs[0].v[0][0] += 0.25;
            assert_eq!(searcher.eval(&pos, 0), net.forward(&searcher.accs[0], &pos));
            let moves = pos.legal_moves();
            if moves.is_empty() {
                pos = Position::startpos();
            } else {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                pos.do_move(moves[(rng % moves.len() as u64) as usize]);
            }
        }
        // Changing evaluation parameters invalidates retained entries.
        searcher.ecache_params = (i32::MIN, i32::MIN);
        searcher.sync_eval_cache();
        assert!(searcher.ecache.iter().all(|&(_, value)| value == i32::MIN));
        // A key of zero must not be confused with an empty cache slot.
        searcher.ecache.fill((0, i32::MIN));
        pos.key = 0;
        assert_eq!(searcher.cached_base_eval(&pos), evaluate(&pos));
    }
}



#[cfg(test)]
mod tactical_tests {
    use super::*;
    #[test]
    fn qsearch_finds_quiet_mate_before_stand_pat() {
        let mut p = Position::from_sfen("3pkp3/9/4G4/9/9/9/9/9/4K4 b R 1").unwrap();
        let original = (p.board, p.hand, p.side, p.king, p.key);
        let mut searcher = Searcher::new(1, Arc::new(Shared::new()));
        searcher.feat = F_QMATE1;
        assert_eq!(searcher.qsearch(&mut p, -INF, -1000, 0, 0), MATE - 1);
        assert_eq!((p.board, p.hand, p.side, p.king, p.key), original);
        assert_eq!(searcher.pv_len[0], 1);
        let m = searcher.pv[0];
        assert!(p.do_move_legal(m));
        assert!(p.in_check());
        assert!(p.legal_moves().is_empty());
    }
}

#[cfg(test)]
mod delta_tests {
    use super::*;
    #[test]
    fn quiescence_does_not_discard_winning_rook_capture() {
        // A gold captures an undefended rook. Two distant enemy golds make the
        // initial stand-pat low, but the captured rook also becomes a hand rook.
        let mut p = Position::from_sfen("kgg6/9/4r4/4G4/9/9/9/9/8K b - 1").unwrap();
        let original = (p.board, p.hand, p.side, p.king, p.key);
        let mut legacy = Searcher::new(1, Arc::new(Shared::new()));
        let old_score = legacy.qsearch(&mut p, 0, 1000, 0, 0);
        let mut improved = Searcher::new(1, Arc::new(Shared::new()));
        improved.feat = F_DELTA_HAND;
        let new_score = improved.qsearch(&mut p, 0, 1000, 0, 0);
        assert!(old_score < 0, "old {old_score}");
        assert!(new_score > 0, "new {new_score}");
        assert_eq!(move_to_usi(improved.pv[0]), "5d5c");
        assert_eq!((p.board, p.hand, p.side, p.king, p.key), original);
    }
}


#[cfg(test)]
mod singular_tests {
    use super::*;

    #[test]
    fn exclusion_ignores_normal_tt_bound_and_does_not_store_fake_bound() {
        let mut pos = Position::startpos();
        let excluded = pos.legal_moves()[0];
        let key = pos.key;
        let board = pos.board;
        let mut s = Searcher::new(1, Arc::new(Shared::new()));
        s.tt.store(key, excluded, 20000, 20, BOUND_EXACT);
        let value = s.negamax_excluding(&mut pos, -10, -9, 2, 1, false, excluded);
        assert!(value < 20000);
        let e = s.tt.probe(key).unwrap();
        assert_eq!((e.mv, e.score, e.depth, e.flag), (excluded, 20000, 20, BOUND_EXACT));
        assert_eq!(pos.key, key);
        assert_eq!(pos.board, board);
        assert_eq!(s.pv_len[1], 0);
    }

    #[test]
    fn excluding_the_only_legal_move_returns_fail_low_not_mate() {
        let mut pos = Position::from_sfen(
            "l5SR1/9/n6sk/p2pp1P2/9/1pPP1+b2P/P1NsP4/Lg1G+r1+p2/K2L5 b B2GS2N6Plp 205"
        ).unwrap();
        let legal = pos.legal_moves();
        assert_eq!(legal.len(), 1);
        let key = pos.key;
        let mut s = Searcher::new(1, Arc::new(Shared::new()));
        assert_eq!(s.negamax_excluding(&mut pos, 99, 100, 3, 1, false, legal[0]), 99);
        assert_eq!(pos.key, key);
        assert!(s.tt.probe(key).is_none());
    }

    #[test]
    fn stopping_inside_verification_restores_position_and_guard() {
        let mut pos = Position::startpos();
        let key = pos.key;
        let board = pos.board;
        let shared = Arc::new(Shared::new());
        shared.stop.store(true, Ordering::Relaxed);
        let mut s = Searcher::new(1, shared);
        s.feat = F_SINGULAR;
        s.iteration_depth = 7;
        s.nodes = 1022; // The verification's entry reaches the stop check.
        s.tt.store(key, pos.legal_moves()[0], 20000, 7, BOUND_LOWER);
        assert_eq!(s.negamax(&mut pos, -INF, INF, 7, 1, false), 0);
        assert!(s.stopped);
        assert_eq!(s.singular_probes, 1);
        assert_eq!(s.singular_extensions, 0);
        assert!(!s.singular_verifying);
        assert_eq!(pos.key, key);
        assert_eq!(pos.board, board);
    }

    #[test]
    fn singular_extension_activates_and_restores_verification_state() {
        let mut pos = Position::startpos();
        let legal = pos.legal_moves();
        let key = pos.key;
        let mut s = Searcher::new(1, Arc::new(Shared::new()));
        s.feat = F_SINGULAR;
        s.iteration_depth = 7;
        // Wide parent window prevents a normal hash cutoff. A deliberately
        // strong lower bound makes every real alternative fail verification.
        s.tt.store(key, legal[0], 20000, 7, BOUND_LOWER);
        s.negamax(&mut pos, -INF, INF, 7, 1, false);
        assert!(s.singular_probes > 0);
        assert!(s.singular_extensions > 0);
        assert!(!s.singular_verifying);
        assert_eq!(pos.key, key);
    }
}


#[cfg(test)]
mod tactical_guard_tests {
    use super::*;
    fn position() -> Position {
        Position::from_sfen("4k4/9/4r4/9/4R4/9/9/9/K8 b G 1").unwrap()
    }
    fn guarded(pos: &mut Position, text: &str) -> bool {
        let m = pos.legal_moves().into_iter().find(|&m| move_to_usi(m) == text).unwrap();
        let original = (pos.board, pos.hand, pos.side, pos.king, pos.key);
        let us = pos.side;
        let mut threats = TacticalThreats::new(pos, true);
        let escape = !mv_is_drop(m) && threats.hanging_from(pos, mv_from(m));
        let safe = pos.see(m) >= 0;
        assert!(pos.do_move_legal(m));
        let result = safe && threats.protects_after(pos, us, escape);
        pos.undo_move();
        assert_eq!((pos.board, pos.hand, pos.side, pos.king, pos.key), original);
        result
    }
    #[test]
    fn safe_escape_is_protected_but_retreat_into_attack_is_not() {
        let mut pos = position();
        assert!(guarded(&mut pos, "5e4e"));
        assert!(!guarded(&mut pos, "5e5d"));
    }
    #[test]
    fn safe_defender_and_interposition_are_protected() {
        let mut pos = position();
        assert!(guarded(&mut pos, "G*4f"));
        // The gold is attacked, but the rook can recapture: its SEE is safe.
        assert!(guarded(&mut pos, "G*5d"));
    }
    #[test]
    fn unrelated_drop_is_not_protected() {
        assert!(!guarded(&mut position(), "G*1h"));
    }
    #[test]
    fn already_defended_piece_does_not_expand_guard_targets() {
        let pos = Position::from_sfen("4k4/9/4r4/9/4R4/4G4/9/9/K8 b - 1").unwrap();
        let mut threats = TacticalThreats::new(&pos, true);
        assert_eq!(threats.majors, 0);
        assert!(!threats.hanging_from(&pos, 4 * 9 + 4));
        assert_eq!(TacticalThreats::new(&position(), false).majors, 0);
    }
}

#[cfg(test)]
mod threat_drop_tests {
    use super::*;
    /// sfen の局面で手 text を指し、打った駒が相手の銀以上の駒に当たるかを返す（局面は元に戻す）
    fn threat(sfen: &str, text: &str) -> bool {
        let mut pos = Position::from_sfen(sfen).unwrap();
        let m = pos.legal_moves().into_iter().find(|&m| move_to_usi(m) == text).unwrap();
        let original = (pos.board, pos.hand, pos.side, pos.king, pos.key);
        assert!(pos.do_move_legal(m));
        let result = threatens_enemy_piece(&pos, mv_to(m));
        pos.undo_move();
        assert_eq!((pos.board, pos.hand, pos.side, pos.king, pos.key), original);
        result
    }
    #[test]
    fn bishop_drop_hitting_rook_is_a_threat() {
        assert!(threat("4k4/9/4r4/9/9/9/9/9/K8 b B 1", "B*3e"));
        // 後手の角打ちでも同じ（向きが逆）
        assert!(threat("k8/9/9/9/9/9/4R4/9/4K4 w b 1", "B*7e"));
    }
    #[test]
    fn drop_hitting_only_a_pawn_is_not_a_threat() {
        assert!(!threat("4k4/9/4p4/9/9/9/9/9/K8 b B 1", "B*3e"));
    }
    #[test]
    fn blocked_line_is_not_a_threat() {
        assert!(!threat("4k4/9/4r4/5p3/9/9/9/9/K8 b B 1", "B*3e"));
    }
    #[test]
    fn knight_and_lance_drops_hitting_gold_or_silver_are_threats() {
        assert!(threat("4k4/9/5g3/9/9/9/9/9/K8 b N 1", "N*5e"));
        assert!(threat("4k4/4s4/9/9/9/9/9/9/K8 b L 1", "L*5e"));
        assert!(!threat("4k4/9/9/9/5g3/9/9/9/K8 b N 1", "N*5e"));
    }
}

#[cfg(test)]
mod goal_order_tests {
 use super::*;
 #[test]
 fn all_legal_replies_and_root_restoration_match_independent_fixtures() {
  let fixtures: &[(&str,&[&str])] = &[
("1S7/l8/2n1gsb1+R/p1p1p1kPp/1p1p1pp2/P1P5P/1PS5+r/LKGS5/1N7 b BGNL2Pgnl3p 139", &["1c2c","B*2c"]),
("1S7/l8/2n1gs1+P+R/p1p1p3p/1p1p1ppk1/P1P5P/1PG5+r/LK1S5/1N7 b 2BGNL2Pgsnl3p 143", &["1c2d"]),
("1S7/lg7/2n1+B2+P1/p1p1p3p/1p1p1p+R2/P1P4pP/1PG3nk1/LK6+r/1NS3G2 b BG2SNL3Pl2p 157", &["S*2h","B*3h","G*2h"]),
("1S7/lg7/2n1+B2+P1/p1p1p3p/1p1p1p+R2/P1P4pP/1PG3n1B/LK4k2/1NS6 b RG2SNL2Pgl3p 163", &["G*3i"]),
("1S7/lg7/2n1+B2+P1/p1p1p3p/1p1p1p+R2/P1P4pP/1PG3nkB/LK7/1NS3R2 b G2SNL2Pgl3p 165", &["S*2h","G*2h"]),
("1S7/lg7/2n1+B2+P1/p1p1p3p/1p1p1p3/P1P4+RP/1PG3n1B/LK6k/1NS3R2 b G2SNL3Pgl3p 167", &["2f2h","L*1i","S*1i","S*2g","G*1i","G*2h"]),
("l6nl/3g2gk1/p5ppp/1p+Rp1n3/7sP/3PPS3/P2+bSKPP1/2B3G2/LR5NL w N3Pgs3p 116", &["S*5h"]),
("l6nl/3g2gk1/p5ppp/1p+Rp1n3/7sP/3PPS3/P2+bS1PP1/2B1sKG2/LR5NL w N3Pg3p 118", &["G*4i"]),
("l7l/3+P2gk1/4p1ns1/5pppp/1G1N4s/PB1S2P2/1P1+s1PN1P/1K7/L1g5L w RBN4Prg3p 134", &["6g7h","7i7h","R*7h","R*8i","G*7h","G*8i"]),
("9/3+P2gk1/4p1nsl/5pppp/1G1N4s/LB1S2P2/1P1+s1PN1P/1K7/2g5L w NL4P2rbg4p 144", &["R*7h","R*8i","G*8i"]),
("9/3+P2gk1/4p1nsl/5pppp/1G1N4s/LB1S2P2/KP1+s1PN1P/2r6/2g5L w NL4Prbg4p 146", &["B*8h","R*9h","G*9h"])
  ];
  let mut s=Searcher::new(1,Arc::new(Shared::new()));s.feat=F_GOAL_ORDER;
  for &(sfen,expected) in fixtures {
   let mut p=Position::from_sfen(sfen).unwrap();let before=(p.board,p.hand,p.side,p.king,p.key);let roots=p.legal_moves();
   s.prepare_goal_moves(&mut p,&roots);assert!(before==(p.board,p.hand,p.side,p.king,p.key));
   let mut got:Vec<_>=s.root_goal_moves.iter().map(|&m|move_to_usi(m)).collect();got.sort();let mut want:Vec<_>=expected.iter().map(|x|x.to_string()).collect();want.sort();assert_eq!(got,want);
   let mut list=MoveList::new();p.gen_moves(&mut list,false);let mut scores=[0;MAX_MOVES];let tt=s.root_goal_moves[0];s.order_moves(&p,&list,&mut scores,tt,0);
   assert_eq!(scores[list.moves[..list.len].iter().position(|&m|m==tt).unwrap()],1_000_000);
   for &m in &s.root_goal_moves {if m!=tt {assert_eq!(scores[list.moves[..list.len].iter().position(|&x|x==m).unwrap()],125_000);}}
  }
  s.feat=0;let mut p=Position::startpos();let roots=p.legal_moves();s.prepare_goal_moves(&mut p,&roots);assert!(s.root_goal_moves.is_empty());
  s.feat=F_GOAL_ORDER;s.shared.stop.store(true,Ordering::Relaxed);s.prepare_goal_moves(&mut p,&roots);assert!(s.root_goal_moves.is_empty());
 }
}
