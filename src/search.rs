//! 探索部: 反復深化 + PVS(αβ) + 静止探索。
//! 置換表・キラー・ヒストリー・ヌルムーブ・LMR・王手延長を使用。

use crate::eval::{evaluate, PIECE_VALUE};
use crate::nn::{self, Acc, Network};
use crate::position::{MoveList, Position, CAPTURE_VALUE, MAX_MOVES};
use crate::tt::*;
use crate::types::*;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

/// 診断 diagEP（cargo の機能 node_dump、既定で無効）：通常の探索の局面を一定の間隔で書き出す。
/// 書き出す先は環境変数 SAILFISH_DUMP_FILE、間隔は SAILFISH_DUMP_EVERY（既定 1000 局面に1つ）
#[cfg(feature = "node_dump")]
mod node_dump {
    use std::fs::File;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Mutex, OnceLock};
    static OUT: OnceLock<Option<(Mutex<File>, u64)>> = OnceLock::new();
    static CTR: AtomicU64 = AtomicU64::new(0);
    pub fn write(line: impl FnOnce() -> String) {
        let Some((f, every)) = OUT.get_or_init(|| {
            let path = std::env::var("SAILFISH_DUMP_FILE").ok()?;
            let every = std::env::var("SAILFISH_DUMP_EVERY").ok().and_then(|s| s.parse().ok()).unwrap_or(1000u64);
            Some((Mutex::new(File::create(path).ok()?), every.max(1)))
        }) else { return };
        if CTR.fetch_add(1, Ordering::Relaxed) % *every == 0 {
            let _ = writeln!(f.lock().unwrap(), "{}", line());
        }
    }
}

/// 探索の改良を個別に入れ替えるためのビット（USI の隠しオプション Feat で指定。強さの比較用）
/// 実験S12（10/28）で singular の +1 の延長を既定で有効にした（AobaNNUE 相手 800局で +7.8ポイント）
pub static FEAT: AtomicU32 = AtomicU32::new(F_QS_TT | F_NN | F_ECACHE | F_DELTA_HAND | F_SINGULAR);
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

/// 実験SPSA（cargo の機能 tune）：探索の値。USI の隠しオプション T_0〜T_15 で変える。機能が無効なら既定の値を使う。
/// 0 RFP の幅／1 RFP の深さの上限／2 futility の基本の幅／3 futility の幅（1手あたり）／4 futility の深さの上限／
/// 5 LMP の基本の手数／6 LMP の手数（深さ²あたり）／7 LMP の深さの上限／8 ヌルムーブで減らす基本／9 ヌルムーブの深さの割り算／
/// 10 LMR の基本（×100）／11 LMR の割り算（×100）／12 LMR のヒストリーの境目／13 アスピレーションの幅／
/// 14 静止探索の駒取りの余裕／15 ヒストリーの加点の上限
pub const TUNE_DEFAULT: [i32; 16] = [150, 3, 110, 100, 3, 4, 3, 3, 2, 4, 75, 225, 4000, 60, 200, 400];
pub static TUNE: [AtomicI32; 16] = [
    AtomicI32::new(150), AtomicI32::new(3), AtomicI32::new(110), AtomicI32::new(100),
    AtomicI32::new(3), AtomicI32::new(4), AtomicI32::new(3), AtomicI32::new(3),
    AtomicI32::new(2), AtomicI32::new(4), AtomicI32::new(75), AtomicI32::new(225),
    AtomicI32::new(4000), AtomicI32::new(60), AtomicI32::new(200), AtomicI32::new(400),
];

#[inline]
fn tp(i: usize) -> i32 {
    if cfg!(feature = "tune") { TUNE[i].load(Ordering::Relaxed) } else { TUNE_DEFAULT[i] }
}

/// LMR で浅くする量の表 [深さ][何手目]
fn lmr_table() -> Vec<i32> {
    let (base, div) = (tp(10) as f64 / 100.0, (tp(11) as f64 / 100.0).max(0.1));
    (0..64 * 64)
        .map(|i| {
            let (d, m) = ((i / 64) as f64, (i % 64) as f64);
            if d < 1.0 || m < 1.0 { 0 } else { (base + d.ln() * m.ln() / div) as i32 }
        })
        .collect()
}

const ECACHE_BITS: usize = 18;

pub const MAX_PLY: usize = 128;
pub const INF: i32 = 31000;
pub const MATE: i32 = 30000;
pub const MATE_BOUND: i32 = MATE - MAX_PLY as i32;

/// 評価補正履歴の表の大きさ（手番ごと）と、補正の上限（点）
const CORR_SIZE: usize = 1 << 14;
const CORR_MAX: i32 = 150;

/// 歩の配置（各筋の先手・後手の歩の段。二歩が無いので筋ごとに高々1枚）と手番から、補正の表の添字を作る
fn corr_index(pos: &Position) -> usize {
    let mut sig = [[0u64; 9]; 2];
    for sq in 0..81 {
        let p = pos.board[sq];
        if p != 0 && type_of(p) == PAWN {
            sig[color_of(p)][sq / 9] = (sq % 9) as u64 + 1;
        }
    }
    let mut h = 0u64;
    for c in 0..2 {
        for f in 0..9 {
            h = h.wrapping_mul(11).wrapping_add(sig[c][f]);
        }
    }
    let i = (h.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - 14)) as usize;
    pos.side * CORR_SIZE + i
}

/// 連続王手の千日手・入玉宣言で勝つときの値。評価値（±20000 まで）より大きく、詰みの値（MATE_BOUND 以上）より小さい。近いほど大きい
#[inline]
fn repetition_win(ply: usize) -> i32 {
    MATE_BOUND - 1 - ply as i32
}

static EPOCH: OnceLock<Instant> = OnceLock::new();

/// 省略の診断の書き出し先（環境変数 SAILFISH_DIAG のファイル。cargo の機能 prune_diag のときだけ使う）
static DIAG_OUT: OnceLock<std::sync::Mutex<Option<std::io::BufWriter<std::fs::File>>>> = OnceLock::new();

/// 1 行を書く: 種類（1=局面全体の省略、2=静かな手の省略）,深さ,省略の判定に使う深さ,改善しているか,評価と窓の差,読んだ結果と窓の差
fn diag_write(kind: u8, depth: i32, eff_depth: i32, improving: bool, gap: i32, result: i32) {
    use std::io::Write;
    let m = DIAG_OUT.get_or_init(|| {
        std::sync::Mutex::new(std::env::var("SAILFISH_DIAG").ok().and_then(|p| {
            std::fs::OpenOptions::new().create(true).append(true).open(p).ok().map(std::io::BufWriter::new)
        }))
    });
    if let Some(w) = m.lock().unwrap().as_mut() {
        let _ = writeln!(w, "{},{},{},{},{},{}", kind, depth, eff_depth, improving as u8, gap, result);
    }
}

fn diag_flush() {
    use std::io::Write;
    if let Some(m) = DIAG_OUT.get() {
        if let Some(w) = m.lock().unwrap().as_mut() {
            let _ = w.flush();
        }
    }
}

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
    /// 実験S12（cargo の機能 s_sing2）：根からその ply までに2手延長した回数
    dext: [i32; MAX_PLY + 2],
    /// 省略の診断（cargo の機能 prune_diag）：確かめるための探索の最中か、省略しない ply、間引きの数
    diag_verifying: bool,
    diag_noprune_ply: usize,
    diag_ctr: u64,
    diag_every: u64,
    /// 評価補正履歴（cargo の機能 corrhist）：手番 × 歩の配置ごとの「探索の値 − 補正前の静的評価」（1/16 点単位）
    corr: Vec<i32>,
    /// 各 ply の補正前の静的評価と、表の添字
    raw_evals: [i32; MAX_PLY + 2],
    corr_idx: [usize; MAX_PLY + 2],
    /// 記録：値が確定した局面での、補正前・補正後のずれの合計と件数
    corr_stats: (u64, u64, u64),
    singular_probes: u64,
    singular_extensions: u64,
    nodes: u64,
    stopped: bool,
    sel_depth: usize,
    killers: [[Move; 2]; MAX_PLY + 2],
    history: Vec<i32>,
    /// 直前の相手の手（駒と行き先）に対してよく効いた応手
    counter: Vec<Move>,
    /// 実験S4（cargo の機能 s_conthist）：継続ヒストリー [1手前・2手前の (駒, 行き先)][(駒, 行き先)]
    cont: Vec<i16>,
    /// 実験S5（cargo の機能 s_caphist）：駒取りヒストリー [(駒, 行き先)][取った駒の種類]
    caph: Vec<i16>,
    /// 各 ply で指した手の (駒, 行き先)。null move は (0, 0)（継続ヒストリーの添字）
    ss_pm: [(u8, usize); MAX_PLY + 2],
    ts: TreeStats,
    /// 診断（cargo の機能 prof）：欄ごとのクロック数と回数
    prof: [std::cell::Cell<u64>; 17],
    prof_n: [std::cell::Cell<u64>; 17],
    /// 読みを浅くする量の表 [深さ][何手目]
    lmr: Vec<i32>,
    pv: Vec<Move>,
    pv_len: [usize; MAX_PLY + 2],
}

const PV_STRIDE: usize = MAX_PLY + 2;

/// 診断（cargo の機能 prof）：式にかかった CPU のクロック数を、欄 i に足す（機能が無効なら式をそのまま返す）
/// 欄：0 手の生成（通常）／1 手の生成（静止探索）／2 並べ替えの点数づけ／3 指す（合法の確認こみ）／4 戻す／5 王手になるかの判定／
/// 6 SEE／7 アキュムレータの更新／8 評価／9 置換表を引く／10 置換表に書く／11 千日手の判定／12 入玉宣言の判定／13 手を選ぶ／14 思考の全体／
/// 15 LMP・futility で省いた手を戻す（指す方は 3 に入る）／16 手作りの評価（覚えておいた値が使えず計算した分。8 の内数）
macro_rules! prof {
    ($s:expr, $i:expr, $e:expr) => {{
        if cfg!(feature = "prof") {
            let t0 = unsafe { core::arch::x86_64::_rdtsc() };
            let r = $e;
            let dt = unsafe { core::arch::x86_64::_rdtsc() } - t0;
            $s.prof[$i].set($s.prof[$i].get() + dt);
            $s.prof_n[$i].set($s.prof_n[$i].get() + 1);
            r
        } else {
            $e
        }
    }};
}

/// tree_stats のときだけ数を1足す（機能が無効なら何もしない）
macro_rules! ts {
    ($s:expr, $f:ident) => {
        if cfg!(feature = "tree_stats") {
            $s.ts.$f += 1;
        }
    };
    ($s:expr, $f:ident, $n:expr) => {
        if cfg!(feature = "tree_stats") {
            $s.ts.$f += $n as u64;
        }
    };
}

/// 診断（cargo の機能 tree_stats）：読みの木の形を数える。思考の終わりに info string で出す
#[derive(Default, Clone, Copy)]
struct TreeStats {
    /// 通常の探索・静止探索の局面数、静止探索のうち王手をかけられた局面
    main: u64,
    qs: u64,
    qs_check: u64,
    /// 置換表に載っていた局面、それで打ち切った局面
    tt_hit: u64,
    tt_cut: u64,
    /// 局面ごと打ち切り：RFP、ヌルムーブ（試した数・打ち切った数）
    rfp: u64,
    null_try: u64,
    null_cut: u64,
    /// 手ごとの省略：LMP、futility
    lmp: u64,
    fut: u64,
    /// β カットした局面、1手目でカットした局面、カットまでに読んだ手の合計
    cut: u64,
    cut_first: u64,
    cut_moves: u64,
    /// どの手も α を超えなかった局面
    all_nodes: u64,
    /// 読んだ手の合計（通常の探索）
    moves: u64,
    /// 王手の延長、LMR で浅くした手、LMR の読み直し、PVS の読み直し
    ext_check: u64,
    lmr: u64,
    lmr_re: u64,
    pvs_re: u64,
    /// 静止探索：その場で打ち切った局面（stand pat）、5手目より先の局面、読んだ手（王手をかけられていない局面・王手をかけられた局面）
    qs_standpat: u64,
    qs_deep: u64,
    qs_moves: u64,
    qs_evasions: u64,
    /// 静止探索で、王手をかけられていない局面のうち、3手目以降の駒取りまで読んだ局面
    qs_over2: u64,
    /// 残りの深さごと（入ったときの depth、23 以上は 23 にまとめる）：局面数、置換表で打ち切り、RFP、ヌルムーブで打ち切り、
    /// 読んだ手、読んだ手のうち LMR で浅くした手、浅くした量の合計、子の深さが親以上になった手（延長）
    d_main: [u64; 24],
    d_tt_cut: [u64; 24],
    d_rfp: [u64; 24],
    d_null_cut: [u64; 24],
    d_moves: [u64; 24],
    d_lmr: [u64; 24],
    d_red: [u64; 24],
    d_ext: [u64; 24],
    /// どの手も α を超えなかった局面の数と、そこで読んだ手・省いた手の合計
    d_all: [u64; 24],
    d_all_moves: [u64; 24],
    d_all_pruned: [u64; 24],
    /// どの手も α を超えなかった局面で実際に読んだ手の合計、β カットした局面の数・1手目でカットした数・カットまでに試した手（省いた手も含む）の合計
    d_all_searched: [u64; 24],
    d_cut: [u64; 24],
    d_cut_first: [u64; 24],
    d_cut_moves: [u64; 24],
}

/// tree_stats のときだけ、残りの深さ d の欄に数を足す
macro_rules! tsd {
    ($s:expr, $f:ident, $d:expr) => {
        if cfg!(feature = "tree_stats") {
            $s.ts.$f[($d).clamp(0, 23) as usize] += 1;
        }
    };
    ($s:expr, $f:ident, $d:expr, $n:expr) => {
        if cfg!(feature = "tree_stats") {
            $s.ts.$f[($d).clamp(0, 23) as usize] += $n as u64;
        }
    };
}
/// (駒, 行き先) の数（ヒストリーの添字の大きさ）
const PT: usize = 32 * 81;

/// ヒストリーの値を、上限 16384 に向かって縮めながら b だけ動かす
#[inline]
fn hist_update(h: &mut i16, b: i32) {
    let v = *h as i32;
    *h = (v + b - v * b.abs() / 16384) as i16;
}

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
    /// 静止探索（全幅の窓）の値と読み筋を返す（学習データづくり用、実験B3）。
    /// 局面の手番から見た値。読み筋は pos に指さない（pos は元のまま返す）。
    pub fn qsearch_pv(&mut self, pos: &mut Position) -> (i32, Vec<Move>) {
        self.feat = FEAT.load(Ordering::Relaxed);
        self.sync_eval_cache();
        self.set_network(if self.feat & F_NN != 0 { nn::global() } else { None });
        if let Some(nn) = &self.nn {
            nn.refresh(pos, &mut self.accs[0]);
        }
        self.stopped = false;
        self.nodes = 0;
        let v = self.qsearch(pos, -INF, INF, 0, 0);
        (v, self.pv[..self.pv_len[0]].to_vec())
    }

    /// 新しい対局：評価補正履歴の表を消す
    pub fn new_game(&mut self) {
        self.corr.iter_mut().for_each(|e| *e = 0);
    }

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
            dext: [0; MAX_PLY + 2],
            diag_verifying: false,
            diag_noprune_ply: usize::MAX,
            diag_ctr: 0,
            corr: if cfg!(feature = "corrhist") { vec![0; 2 * CORR_SIZE] } else { Vec::new() },
            raw_evals: [0; MAX_PLY + 2],
            corr_idx: [0; MAX_PLY + 2],
            corr_stats: (0, 0, 0),
            diag_every: std::env::var("SAILFISH_DIAG_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(8).max(1),
            singular_probes: 0,
            singular_extensions: 0,
            nodes: 0,
            stopped: false,
            sel_depth: 0,
            killers: [[MOVE_NONE; 2]; MAX_PLY + 2],
            history: vec![0; 32 * 81],
            counter: vec![MOVE_NONE; 32 * 81],
            cont: if cfg!(feature = "s_conthist") { vec![0; PT * PT] } else { Vec::new() },
            caph: if cfg!(feature = "s_caphist") { vec![0; PT * 16] } else { Vec::new() },
            ss_pm: [(0, 0); MAX_PLY + 2],
            ts: TreeStats::default(),
            prof: Default::default(),
            prof_n: Default::default(),
            lmr: lmr_table(),
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
        prof!(self, 8, self.eval_inner(pos, ply))
    }

    #[inline]
    fn eval_inner(&mut self, pos: &Position, ply: usize) -> i32 {
        // 診断（実験 diagSE）：外部の評価関数を読み込んでいれば、評価をまるごと置き換える
        #[cfg(feature = "ext_eval")]
        if let Some(ext) = crate::ext_eval::global() {
            return ext.evaluate(pos);
        }
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
        let v = prof!(self, 16, evaluate(pos));
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
        prof!(self, 7, self.ensure_acc_inner(pos, ply))
    }

    #[inline]
    fn ensure_acc_inner(&mut self, pos: &Position, ply: usize) {
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
        if cfg!(feature = "tune") {
            self.lmr = lmr_table();
        }
        self.cont.iter_mut().for_each(|h| *h /= 2);
        self.caph.iter_mut().for_each(|h| *h /= 2);

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
            let r = prof!(self, 14, self.search(pos, &root_moves));
            self.shared.helpers_stop.store(true, Ordering::Relaxed);
            r
        });

        self.wait_for_stop();
        if cfg!(feature = "prune_diag") {
            diag_flush();
        }
        if cfg!(feature = "prof") {
            let c = self.prof.iter().map(|x| x.take().to_string()).collect::<Vec<_>>().join(",");
            let n = self.prof_n.iter().map(|x| x.take().to_string()).collect::<Vec<_>>().join(",");
            println!("info string prof cycles {} calls {} nodes {}", c, n, self.nodes);
        }
        if cfg!(feature = "tree_stats") {
            let t = std::mem::take(&mut self.ts);
            println!("info string tree main {} qs {} qs_check {} tt_hit {} tt_cut {} rfp {} null_try {} null_cut {} lmp {} fut {} cut {} cut_first {} cut_moves {} all {} moves {} ext_check {} lmr {} lmr_re {} pvs_re {} qs_standpat {} qs_deep {} qs_moves {} qs_evasions {} qs_over2 {}",
                t.main, t.qs, t.qs_check, t.tt_hit, t.tt_cut, t.rfp, t.null_try, t.null_cut, t.lmp, t.fut,
                t.cut, t.cut_first, t.cut_moves, t.all_nodes, t.moves, t.ext_check, t.lmr, t.lmr_re, t.pvs_re,
                t.qs_standpat, t.qs_deep, t.qs_moves, t.qs_evasions, t.qs_over2);
            let j = |a: &[u64; 24]| a.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",");
            println!("info string treed main {} tt_cut {} rfp {} null_cut {} moves {} lmr {} red {} ext {} all {} all_moves {} all_pruned {} all_searched {} cut {} cut_first {} cut_moves {}",
                j(&t.d_main), j(&t.d_tt_cut), j(&t.d_rfp), j(&t.d_null_cut), j(&t.d_moves), j(&t.d_lmr), j(&t.d_red), j(&t.d_ext),
                j(&t.d_all), j(&t.d_all_moves), j(&t.d_all_pruned), j(&t.d_all_searched), j(&t.d_cut), j(&t.d_cut_first), j(&t.d_cut_moves));
        }
        if cfg!(feature = "corrhist") {
            let (raw, cor, n) = std::mem::take(&mut self.corr_stats);
            if n > 0 {
                println!("info string corrhist n {} raw {:.1} corrected {:.1}", n, raw as f64 / n as f64, cor as f64 / n as f64);
            }
        }
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
            let mut delta = tp(13);
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
                // 実験T1（cargo の機能 time_unstable）：最善手・評価が落ち着いていなければ、soft の 1.25 倍（hard まで）まで次の反復を始める。
                // stop_before_next の目標と同じ。既定では soft で必ず打ち切るので、この延長は働かない。
                let soft_limit = if cfg!(feature = "time_unstable") && self.limits.adaptive_time && stable_iterations < 2 {
                    (self.limits.soft_ms.saturating_mul(5) / 4).min(self.limits.hard_ms)
                } else {
                    self.limits.soft_ms
                };
                if ms >= soft_limit || predicted_stop || root_moves.len() == 1 {
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
        prof!(self, 6, if self.feat & F_SEE_HAND != 0 { pos.see_with_hand(m) } else { pos.see(m) })
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

    /// k 手前（この ply から見て）に指した手の (駒, 行き先) の添字。無ければ 0（null move と同じ番兵の行）
    #[inline]
    fn prev_pm(&self, ply: usize, k: usize) -> usize {
        if ply >= k { let (p, t) = self.ss_pm[ply - k]; p as usize * 81 + t } else { 0 }
    }

    /// 静かな手のヒストリーの合計（駒×行き先、実験S4 では 1手前・2手前からの継続ヒストリーも足す）
    #[inline]
    fn quiet_stat(&self, piece: u8, to: usize, ply: usize) -> i32 {
        let idx = piece as usize * 81 + to;
        let mut s = self.history[idx];
        if cfg!(feature = "s_conthist") {
            s += self.cont[self.prev_pm(ply, 1) * PT + idx] as i32 + self.cont[self.prev_pm(ply, 2) * PT + idx] as i32;
        }
        s
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
                // 実験S5（cargo の機能 s_caphist）：取る駒の価値に、駒取りヒストリーを足す（取る側の駒の価値のかわり）
                let tiebreak = if cfg!(feature = "s_caphist") {
                    let idx = pos.board[mv_from(m)] as usize * 81 + mv_to(m);
                    self.caph[idx * 16 + type_of(pos.board[mv_to(m)]) as usize] as i32 / 16
                } else { -attacker / 16 };
                let base = victim * 16 + tiebreak + if mv_is_promo(m) { 1000 } else { 0 };
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
                self.quiet_stat(piece, mv_to(m), ply)
            };
        }
    }

    #[inline]
    fn pick_move(list: &mut MoveList, scores: &mut [i32], i: usize) -> Move {
        // 残りの中で点数が最大の、最初の手（これまでと同じ決まり）。最大値を先にまとめて求めてから位置を探す（実験 V5）
        let rest = &scores[i..list.len];
        let max = rest.iter().copied().fold(i32::MIN, i32::max);
        let best = i + rest.iter().position(|&s| s == max).unwrap_or(0);
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
        ts!(self, main);
        // 入ったときの残りの深さ（tree_stats の深さごとの欄）
        let d0 = depth;
        tsd!(self, d_main, d0);
        self.check_time();
        if self.stopped {
            return 0;
        }
        self.sel_depth = self.sel_depth.max(ply);

        if ply > 0 {
            if cfg!(feature = "perpetual_check") {
                // 連続王手の千日手は、王手をかけ続けた側の負け（それ以外の繰り返しは引き分け）
                if let Some(r) = prof!(self, 11, pos.repetition_result()) {
                    return r * repetition_win(ply);
                }
            } else if pos.is_repetition() {
                return 0;
            }
            // 入玉宣言（cargo の機能 nyugyoku）：手番側が宣言できる局面は勝ち
            if cfg!(feature = "nyugyoku") && prof!(self, 12, pos.can_declare()) {
                return repetition_win(ply);
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
        let tt_entry = if excluded == MOVE_NONE { prof!(self, 9, self.tt.probe(key)) } else { None };
        if let Some(e) = tt_entry {
            tt_move = e.mv;
            ts!(self, tt_hit);
            if !pv_node && ply > 0 && e.depth as i32 >= depth {
                let s = score_from_tt(e.score as i32, ply);
                if e.flag == BOUND_EXACT
                    || (e.flag == BOUND_LOWER && s >= beta)
                    || (e.flag == BOUND_UPPER && s <= alpha)
                {
                    ts!(self, tt_cut);
                    tsd!(self, d_tt_cut, d0);
                    return s;
                }
            }
        }

        // ここから先でこの局面のアキュムレータが要る（評価と、子の局面への更新の元になる）
        self.ensure_acc(pos, ply);
        let in_check = pos.in_check();
        let mut static_eval = if in_check { -INF } else { self.eval(pos, ply) };
        // 評価補正履歴（cargo の機能 corrhist）：補正前の値と添字を控え、表の値を足す
        if cfg!(feature = "corrhist") && !in_check && static_eval.abs() < MATE_BOUND {
            let idx = corr_index(pos);
            self.raw_evals[ply] = static_eval;
            self.corr_idx[ply] = idx;
            static_eval = (static_eval + self.corr[idx] / 16).clamp(-20000, 20000);
        }
        self.static_evals[ply] = static_eval;
        #[cfg(feature = "node_dump")]
        if !in_check && excluded == MOVE_NONE && ply > 0 {
            node_dump::write(|| format!("{} 1\t{}\t{}\t{}", crate::psv::sfen_key(pos, false), depth, ply, static_eval));
        }
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
            // 省略の診断：静的評価が β 以上の浅い局面を、この局面では省略せずに同じ深さで読み、結果を記録する（SAILFISH_DIAG_EVERY 回に 1 回、既定 8）
            if cfg!(feature = "prune_diag") && !self.diag_verifying && depth <= 3
                && static_eval >= beta && static_eval.abs() < MATE_BOUND && beta.abs() < MATE_BOUND
            {
                self.diag_ctr += 1;
                if self.diag_ctr % self.diag_every == 0 {
                    self.diag_verifying = true;
                    self.diag_noprune_ply = ply;
                    let v = self.negamax(pos, beta - 1, beta, depth, ply, null_ok);
                    self.diag_verifying = false;
                    self.diag_noprune_ply = usize::MAX;
                    self.pv_len[ply] = 0;
                    self.static_evals[ply] = static_eval;
                    if self.stopped {
                        return 0;
                    }
                    diag_write(1, depth, rfp_depth, improving, static_eval - beta, v - beta);
                }
            }
            // 実験S2（cargo の機能 s_prune）：やねうら王と同じく、残り8手まで・幅 214×(深さ − improving) で省略する
            let rfp = if cfg!(feature = "s_prune") {
                depth < 9 && static_eval - 214 * (depth - i32::from(improving)) >= beta
            } else if cfg!(feature = "s_prune2") {
                // 実験S6（cargo の機能 s_prune2）：幅はこれまでと同じ 150×深さのまま、深さ6まで広げる
                depth <= 6 && static_eval - 150 * rfp_depth >= beta
            } else {
                depth <= tp(1) && static_eval - tp(0) * rfp_depth >= beta
            };
            if self.feat & D_NO_RFP == 0 && ply != self.diag_noprune_ply && rfp && static_eval.abs() < MATE_BOUND {
                ts!(self, rfp);
                tsd!(self, d_rfp, d0);
                return static_eval;
            }
            // Null move pruning
            if self.feat & D_NO_NULL == 0 && null_ok && depth >= 3 && static_eval >= beta {
                let r = if self.feat & F_NULL2 != 0 {
                    3 + depth / 3 + ((static_eval - beta) / 200).clamp(0, 2)
                } else {
                    tp(8) + depth / tp(9).max(1)
                };
                self.set_pending_null(ply);
                self.ss_pm[ply] = (0, 0);
                ts!(self, null_try);
                if cfg!(feature = "s_sing2") { self.dext[ply + 1] = self.dext[ply]; }
                pos.do_null();
                let s = -self.negamax(pos, -beta, -beta + 1, depth - 1 - r, ply + 1, false);
                pos.undo_null();
                if self.stopped {
                    return 0;
                }
                if s >= beta {
                    ts!(self, null_cut);
                    tsd!(self, d_null_cut, d0);
                    return if s >= MATE_BOUND { beta } else { s };
                }
            }
        }

        // A single-ply extension, bounded to twice the root iteration horizon.
        // Verification has half depth and excludes the TT move explicitly.
        // Nested singular verification is disabled throughout that subtree.
        let mut singular_move = MOVE_NONE;
        // 実験S12（s_sing2）：置換表の手の延長量（+1・+2・−2）
        let mut sing_ext = 0;
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
                            sing_ext = 1;
                            // 実験S12：ほかの手がはっきり悪ければ2手延長（PV でない局面、根からの2手延長は6回まで）
                            if cfg!(feature = "s_sing2") && !pv_node && alternative < threshold - 75 && self.dext[ply] <= 6 {
                                sing_ext = 2;
                            }
                        } else if cfg!(feature = "s_sing2") {
                            if threshold >= beta {
                                // 実験S12 multi-cut：置換表の手を除いても β を超える手があるので、この局面は β カットとみなす
                                return threshold;
                            } else if tt_score >= beta {
                                // 実験S12 負の延長：置換表の手だけが β を超えそうとは言えないので、置換表の手を2手浅く読む
                                singular_move = tt_move;
                                sing_ext = -2;
                            }
                        }
                    }
                }
            }
        }

        // 王手延長（実験S1、cargo の機能 s_check では、王手をかけられた局面ではなく、損をしない王手の手を延長する）
        if in_check && !cfg!(feature = "s_check") {
            depth += 1;
        }

        let mut list = MoveList::new();
        prof!(self, 0, pos.gen_moves(&mut list, false));
        // 実験M2（cargo の機能 m_evext）：王手をかけられていて、逃げ方が2通り以下なら1手延長する（追い詰められた筋を深く読む）。
        // S1 で王手をかけられた局面の延長をやめたので、その副作用（自玉への詰めろの読み落とし）を減らすため
        if cfg!(feature = "m_evext") && in_check && excluded == MOVE_NONE && ply > 0
            && ply < (2 * self.iteration_depth) as usize
        {
            let mut n = 0;
            for i in 0..list.len {
                if pos.do_move_legal(list.moves[i]) {
                    pos.undo_move();
                    n += 1;
                    if n > 2 { break; }
                }
            }
            if n <= 2 {
                depth += 1;
            }
        }
        let mut scores = [0i32; MAX_MOVES];
        prof!(self, 2, self.order_moves(pos, &list, &mut scores, tt_move, ply));

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
        // 実験S5：試した駒取り（駒, 行き先, 取った駒の種類）
        let mut caps_tried = [(0u8, 0usize, 0u8); 32];
        let mut n_caps = 0;
        // 実験S2：指す前に省略した手の数（やねうら王の moveCount と同じく、省略した手も数える）
        let mut pruned = 0;
        // tree_stats：実際に読んだ手の数（LMP・futility で省いた手を除く）
        let mut ts_searched = 0;
        // 実験S3：置換表の手が駒を取る手か
        let tt_capture = tt_move != MOVE_NONE && !mv_is_drop(tt_move) && pos.board[mv_to(tt_move)] != 0;

        for i in 0..list.len {
            let m = prof!(self, 13, Self::pick_move(&mut list, &mut scores, i));
            if m == excluded { continue; }
            let to = mv_to(m);
            let is_cap = !mv_is_drop(m) && pos.board[to] != 0;
            let is_promo = mv_is_promo(m);
            let quiet = !is_cap && !is_promo;
            let piece = if mv_is_drop(m) { make_piece(us, mv_drop_pt(m)) } else { pos.board[mv_from(m)] };

            if ply == 0 && self.root_excluded.contains(&m) {
                continue;
            }
            // 指す前の王手の判定は、実験S2・S6 のときだけ（S1 の延長は、指した後に王手かどうかで決める。実験 V6）
            let pre_check = (cfg!(feature = "s_prune") || cfg!(feature = "s_prune2")) && prof!(self, 5, pos.gives_check(m));
            // 実験S2（cargo の機能 s_prune）：指す前の省略（やねうら王の Step 13 と同じ考え方）
            if cfg!(feature = "s_prune") && ply > 0 && excluded == MOVE_NONE && !in_check && best > -MATE_BOUND {
                let mc = legal + pruned + 1;
                if is_cap || is_promo || pre_check {
                    // 取り合いで大きく損をする駒取り・成り・王手
                    if self.exchange_score(pos, m) < -218 * depth {
                        pruned += 1;
                        continue;
                    }
                } else {
                    let lmr_depth = (depth - 1 - self.lmr[(depth.min(63) as usize) * 64 + mc.min(63) as usize]).max(0);
                    // 後半の静かな手・評価が α に届きそうにない静かな手・取り合いで損をする静かな手
                    if mc >= (3 + depth * depth) / (2 - i32::from(improving))
                        || (lmr_depth < 8 && static_eval + 172 + 145 * lmr_depth <= alpha)
                        || self.exchange_score(pos, m) < -21 * lmr_depth * lmr_depth - 21 * lmr_depth
                    {
                        pruned += 1;
                        continue;
                    }
                }
            }
            // 実験S6（cargo の機能 s_prune2）：S2 の軽い版。PV でない局面で、取り合いで大きく損をする手だけを指す前に省略する
            if cfg!(feature = "s_prune2") && ply > 0 && !pv_node && excluded == MOVE_NONE && !in_check
                && best > -MATE_BOUND && depth <= 6
            {
                let limit = if is_cap || is_promo || pre_check { -218 * depth } else if depth <= 4 { -21 * depth * depth - 21 * depth } else { i32::MIN };
                if limit > i32::MIN && self.exchange_score(pos, m) < limit {
                    continue;
                }
            }
            let escape = guard_on && quiet && !mv_is_drop(m)
                && threats.hanging_from(pos, mv_from(m));
            let safe_reply = guard_on && quiet && (escape || threats.majors != 0)
                && pos.see(m) >= 0;
            self.set_pending(pos, m, ply);
            let cap_type = if is_cap { type_of(pos.board[to]) } else { 0 };
            if !prof!(self, 3, pos.do_move_legal(m)) {
                continue;
            }
            legal += 1;
            self.ss_pm[ply] = (piece, to);
            let gives_check = pos.in_check();
            // 実験S1：損をしない王手（SEE が 0 以上）を1手延長する。延長はルートの深さの2倍の ply まで。
            // SEE は指す前の局面で調べるので、王手のときだけ戻して調べ、指し直す（実験 V6）
            let check_ext = cfg!(feature = "s_check") && gives_check && ply < (2 * self.iteration_depth) as usize && {
                pos.undo_move();
                let ok = self.exchange_score(pos, m) >= 0;
                pos.do_move(m);
                ok
            };
            let protected_reply = safe_reply && threats.protects_after(pos, us, escape);
            let threat_drop = self.feat & F_THREAT_DROP != 0 && mv_is_drop(m) && !gives_check
                && threatens_enemy_piece(pos, to);

            if !cfg!(feature = "s_prune") && !(protected_reply && self.feat & F_TACTICAL_PRUNE != 0)
                && excluded == MOVE_NONE && !pv_node && !in_check && !gives_check && quiet && best > -MATE_BOUND {
                // 省略の診断：静的評価が α 以下の浅い局面の静かな手を、省略せずに読み、結果を記録する（SAILFISH_DIAG_EVERY 回に 1 回、既定 8）
                if cfg!(feature = "prune_diag") && !self.diag_verifying && depth <= 3 && legal > 1
                    && static_eval <= alpha && alpha.abs() < MATE_BOUND
                {
                    self.diag_ctr += 1;
                    if self.diag_ctr % self.diag_every == 0 {
                        self.diag_verifying = true;
                        let s = -self.negamax(pos, -alpha - 1, -alpha, depth - 1, ply + 1, true);
                        self.diag_verifying = false;
                        if self.stopped {
                            pos.undo_move();
                            return 0;
                        }
                        diag_write(2, depth, depth, improving, alpha - static_eval, s - alpha);
                    }
                }
                // Late move pruning（浅い深さでの後半の静かな手を省略）
                let lmp_base = tp(5) + depth * depth * tp(6);
                let lmp_limit = if imp_on && !improving { lmp_base * 2 / 3 } else { lmp_base };
                // 実験S10（cargo の機能 s_lmp）：やねうら王と同じ (3+d²)/(2−improving) 手目から省略（深さの上限なし）。
                // 実験S11（s_lmp_mid）：今の 4+3d² とやねうら王の中間 (4+2d²)/(2−improving)
                let lmp = if cfg!(feature = "s_lmp") {
                    legal >= (3 + depth * depth) / (2 - i32::from(improving))
                } else if cfg!(feature = "s_lmp_mid") {
                    legal >= (4 + 2 * depth * depth) / (2 - i32::from(improving))
                } else {
                    depth <= tp(7) && legal > lmp_limit
                };
                if self.feat & D_NO_LMP == 0 && lmp {
                    ts!(self, lmp);
                    prof!(self, 15, pos.undo_move());
                    continue;
                }
                // Futility pruning（評価値が α に遠く届かない局面の静かな手を省略）
                if self.feat & D_NO_FUTILITY == 0 && depth <= tp(4) && legal > 1 && static_eval + tp(2) + tp(3) * depth <= alpha {
                    ts!(self, fut);
                    prof!(self, 15, pos.undo_move());
                    continue;
                }
            }
            if quiet && n_quiets < 64 {
                quiets_tried[n_quiets] = (piece, to);
                n_quiets += 1;
            }
            if cfg!(feature = "s_caphist") && is_cap && n_caps < 32 {
                caps_tried[n_caps] = (piece, to, cap_type);
                n_caps += 1;
            }

            // 実験S12（s_sing2）：置換表の手は singular の結果（+1・+2・−2）だけで延長量を決める（王手の延長と重ねない）
            let new_depth = if cfg!(feature = "s_sing2") && m == singular_move {
                depth - 1 + sing_ext
            } else {
                depth - 1 + i32::from(m == singular_move || check_ext)
            };
            if cfg!(feature = "s_sing2") {
                self.dext[ply + 1] = self.dext[ply] + i32::from(m == singular_move && sing_ext == 2);
            }
            if check_ext { ts!(self, ext_check); }
            ts!(self, moves);
            tsd!(self, d_moves, d0);
            if cfg!(feature = "tree_stats") { ts_searched += 1; }
            if new_depth >= d0 { tsd!(self, d_ext, d0); }
            let mut s;
            if legal == 1 {
                s = -self.negamax(pos, -beta, -alpha, new_depth, ply + 1, true);
            } else {
                let mut r = 0;
                let lmr_from = if cfg!(feature = "s_lmr") { 1 } else { 2 };
                if excluded == MOVE_NONE && self.feat & D_NO_LMR == 0 && depth >= 3 && legal > lmr_from && quiet && !in_check && !gives_check {
                    let mc = if cfg!(feature = "s_prune") { legal + pruned } else { legal };
                    r = self.lmr[(depth.min(63) as usize) * 64 + mc.min(63) as usize];
                    if pv_node {
                        r -= 1;
                    }
                    if imp_on && !improving {
                        r += 1;
                    }
                    if cfg!(feature = "s_lmr") {
                        // 実験S3（cargo の機能 s_lmr）：PV でない局面・置換表の手が駒取りの局面では深めに減らし、
                        // ヒストリーの良し悪しに応じて減らす量を連続的に変える（やねうら王の cutNode・ttCapture・statScore の考え方）
                        if !pv_node {
                            r += 1;
                        }
                        if tt_capture {
                            r += 1;
                        }
                        let div = if cfg!(feature = "s_conthist") { 16384 } else { 8192 };
                        r -= self.quiet_stat(piece, to, ply) / div;
                    } else if self.history[piece as usize * 81 + to] > tp(12) {
                        // ヒストリーの良い手は浅くしすぎない
                        r -= 1;
                    }
                    if protected_reply { r -= 1; }
                    if threat_drop { r -= 1; }
                    r = r.clamp(0, (new_depth - 1).max(0));
                }
                if r > 0 { ts!(self, lmr); tsd!(self, d_lmr, d0); tsd!(self, d_red, d0, r); }
                s = -self.negamax(pos, -alpha - 1, -alpha, new_depth - r, ply + 1, true);
                if s > alpha && r > 0 {
                    ts!(self, lmr_re);
                    s = -self.negamax(pos, -alpha - 1, -alpha, new_depth, ply + 1, true);
                }
                if s > alpha && s < beta {
                    ts!(self, pvs_re);
                    s = -self.negamax(pos, -beta, -alpha, new_depth, ply + 1, true);
                }
            }
            prof!(self, 4, pos.undo_move());
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
                        ts!(self, cut);
                        ts!(self, cut_moves, legal);
                        if legal == 1 { ts!(self, cut_first); }
                        tsd!(self, d_cut, d0);
                        tsd!(self, d_cut_moves, d0, legal + pruned);
                        if ts_searched == 1 { tsd!(self, d_cut_first, d0); }
                        // 実験S5：カットした駒取りを上げ、先に試して失敗した駒取りを下げる
                        if cfg!(feature = "s_caphist") && excluded == MOVE_NONE {
                            let bonus = (depth * depth).min(tp(15));
                            for k in 0..n_caps {
                                let (p, t, c) = caps_tried[k];
                                let b = if is_cap && p == piece && t == to { bonus } else { -bonus };
                                hist_update(&mut self.caph[(p as usize * 81 + t) * 16 + c as usize], b);
                            }
                        }
                        if quiet && excluded == MOVE_NONE {
                            if self.killers[ply][0] != m {
                                self.killers[ply][1] = self.killers[ply][0];
                                self.killers[ply][0] = m;
                            }
                            if let Some(ci) = Self::counter_index(pos) {
                                self.counter[ci] = m;
                            }
                            // ヒストリー: カットした手を上げ、先に試して失敗した手を下げる
                            let bonus = (depth * depth).min(tp(15));
                            for k in 0..n_quiets {
                                let (p, t) = quiets_tried[k];
                                let h = &mut self.history[p as usize * 81 + t];
                                let b = if p == piece && t == to { bonus } else { -bonus };
                                *h += b - *h * b.abs() / 16384;
                                // 実験S4：1手前・2手前の手からの継続ヒストリーも同じように動かす
                                if cfg!(feature = "s_conthist") {
                                    let idx = p as usize * 81 + t;
                                    for k2 in 1..=2 {
                                        if ply >= k2 {
                                            let c = self.prev_pm(ply, k2) * PT + idx;
                                            hist_update(&mut self.cont[c], b);
                                        }
                                    }
                                }
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
        if flag == BOUND_UPPER {
            ts!(self, all_nodes);
            tsd!(self, d_all, d0);
            tsd!(self, d_all_moves, d0, legal);
            tsd!(self, d_all_pruned, d0, list.len as i32 - legal);
            tsd!(self, d_all_searched, d0, ts_searched);
        }
        // 評価補正履歴を覚える（向きの合わない限界値、駒を取る手が最善、詰みの値は使わない）
        if cfg!(feature = "corrhist") && excluded == MOVE_NONE && !in_check
            && best.abs() < MATE_BOUND && static_eval.abs() < MATE_BOUND
            && !(best_move != MOVE_NONE && !mv_is_drop(best_move) && pos.board[mv_to(best_move)] != 0)
            && !(flag == BOUND_LOWER && best <= static_eval)
            && !(flag == BOUND_UPPER && best >= static_eval)
        {
            let raw = self.raw_evals[ply];
            let idx = self.corr_idx[ply];
            if flag == BOUND_EXACT {
                let s = &mut self.corr_stats;
                s.0 += (best - raw).unsigned_abs() as u64;
                s.1 += (best - static_eval).unsigned_abs() as u64;
                s.2 += 1;
            }
            let target = (best - raw).clamp(-CORR_MAX, CORR_MAX) * 16;
            let w = (depth + 1).clamp(1, 16) * 8;
            let e = &mut self.corr[idx];
            *e += (target - *e) * w / 1024;
        }
        if excluded == MOVE_NONE {
            prof!(self, 10, self.tt.store(key, best_move, score_to_tt(best, ply), depth, flag));
        }
        best
    }

    fn qsearch(&mut self, pos: &mut Position, mut alpha: i32, beta: i32, ply: usize, qdepth: usize) -> i32 {
        self.pv_len[ply] = 0;
        self.nodes += 1;
        ts!(self, qs);
        if qdepth >= 5 { ts!(self, qs_deep); }
        self.check_time();
        if self.stopped {
            return 0;
        }
        self.sel_depth = self.sel_depth.max(ply);
        if cfg!(feature = "nyugyoku") && ply > 0 && prof!(self, 12, pos.can_declare()) {
            return repetition_win(ply);
        }
        if ply >= MAX_PLY - 4 {
            self.ensure_acc(pos, ply);
            return self.eval(pos, ply);
        }

        if self.feat & (F_QCHECK_CAPTURE | F_QCHECK_QUIET | F_QMATE1) != 0 {
            if cfg!(feature = "perpetual_check") {
                if let Some(r) = pos.repetition_result() { return r * repetition_win(ply); }
            } else if pos.is_repetition() { return 0; }
        }
        let orig_alpha = alpha;
        let pv_node = beta - alpha > 1;
        let key = pos.key;
        // Entry-only quiet checks/mate verification have a different horizon.
        // Do not reuse or store capture-only QS bounds for that context.
        let extra_checks = qdepth < 3 && pv_node && self.feat & F_QCHECK_QUIET != 0;
        let mate_probe = qdepth == 0 && pv_node && self.feat & F_QMATE1 != 0;
        let use_tt = self.feat & F_QS_TT != 0 && !extra_checks && !mate_probe;
        let mut tt_move = MOVE_NONE;
        let mut tt_hit = false;
        if use_tt {
            if let Some(e) = prof!(self, 9, self.tt.probe(key)) {
                tt_hit = true;
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
        // 実験M1（cargo の機能 m_mate1）：静止探索の入口（駒取りしか読まないので、駒を取らない詰みを見落とす）で、
        // 王手をかけられておらず置換表に無い局面は、1手詰めを確かめる（やねうら王の qsearch の mate_1ply と同じ考え方）
        let mate1 = cfg!(feature = "m_mate1") && qdepth == 0 && !in_check && !tt_hit;
        if mate_probe || mate1 {
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
            ts!(self, qs_check);
            best = -INF;
            prof!(self, 1, pos.gen_moves(&mut list, false));
        } else {
            let mut stand = self.eval(pos, ply);
            if cfg!(feature = "corrhist") && stand.abs() < MATE_BOUND {
                stand = (stand + self.corr[corr_index(pos)] / 16).clamp(-20000, 20000);
            }
            if stand >= beta {
                ts!(self, qs_standpat);
                return stand;
            }
            alpha = alpha.max(stand);
            best = stand;
            prof!(self, 1, pos.gen_moves(&mut list, !extra_checks));
        }

        let mut scores = [0i32; MAX_MOVES];
        prof!(self, 2, self.order_moves(pos, &list, &mut scores, tt_move, ply));

        let mut legal = 0;
        // 実験Q1・Q2：ここまでに選んだ（王手でない）駒取りの数、stand pat の値、直前の手の行き先
        let mut q_picked = 0;
        let stand_pat = best;
        let recapture_sq = { let lm = pos.last_move(); if lm == MOVE_NONE { usize::MAX } else { mv_to(lm) } };
        for i in 0..list.len {
            let m = prof!(self, 13, Self::pick_move(&mut list, &mut scores, i));
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
                if !checking && self.feat & D_NO_QDELTA == 0 && best + gain + tp(14) < alpha {
                    continue;
                }
                // 取り合いで損をする駒取りは読まない
                if !checking && self.feat & D_NO_QSEE == 0 && self.exchange_score(pos, m) < 0 {
                    continue;
                }
                if !checking {
                    q_picked += 1;
                    // 実験Q1（cargo の機能 q_mc2）：やねうら王と同じく、王手でない駒取りは2手まで。
                    // 評価＋155 が α に届かず、取り合いで得をしない手も読まない
                    if cfg!(feature = "q_mc2")
                        && (q_picked > 2 || (stand_pat + 155 <= alpha && self.exchange_score(pos, m) <= 0))
                    {
                        continue;
                    }
                    // 実験Q2（cargo の機能 q_recap）：静止探索の5手目より先は、直前に動いた駒を取り返す手だけ
                    if cfg!(feature = "q_recap") && qdepth >= 5 && mv_to(m) != recapture_sq {
                        continue;
                    }
                }
            } else if cfg!(feature = "q_evsee") && best > -MATE_BOUND && self.exchange_score(pos, m) < 0 {
                // 実験Q3（cargo の機能 q_evsee）：王手をかけられた局面でも、詰まない逃げ方が見つかった後は、取り合いで損をする逃げ方を読まない
                continue;
            }
            self.set_pending(pos, m, ply);
            let piece = if mv_is_drop(m) { make_piece(pos.side, mv_drop_pt(m)) } else { pos.board[mv_from(m)] };
            if !prof!(self, 3, pos.do_move_legal(m)) {
                continue;
            }
            legal += 1;
            self.ss_pm[ply] = (piece, mv_to(m));
            if in_check { ts!(self, qs_evasions); } else {
                ts!(self, qs_moves);
                if legal == 3 { ts!(self, qs_over2); }
            }
            let s = -self.qsearch(pos, -beta, -alpha, ply + 1, qdepth + 1);
            prof!(self, 4, pos.undo_move());
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
            prof!(self, 10, self.tt.store_qs(key, best_move, score_to_tt(best, ply), flag));
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
