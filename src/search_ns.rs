//! 実験 NS（11/17〜）：作り直した探索。cargo の機能 `ns` のときだけ使う（既定で無効）。
//!
//! 反復深化・時間の管理・評価（ネットの差分更新）・置換表・複数スレッド・USI の出力は search.rs のものを使い、
//! ここには通常の探索（ns_search）と静止探索（ns_qsearch）、手の並べ方、ヒストリーを置く。
//! 考え方はチェスの探索で広く知られたもの。やねうら王・Stockfish のコードは写していない（値も Sailfish の物差しで決めた）。
//!
//! 部品は USI の隠しオプション NsOff（ビットの和）で外せる。外したときの損を、組み立て直さずに測るため。

use super::*;

/// NsOff のビット：外す部品
pub(super) const NS_RAZOR: u32 = 1;
pub(super) const NS_RFP: u32 = 2;
pub(super) const NS_NULL: u32 = 4;
pub(super) const NS_PROBCUT: u32 = 8;
pub(super) const NS_IIR: u32 = 16;
pub(super) const NS_LMP: u32 = 32;
pub(super) const NS_FUT: u32 = 64;
pub(super) const NS_SEE: u32 = 128;
pub(super) const NS_CHP: u32 = 256;
pub(super) const NS_CAPFUT: u32 = 512;
pub(super) const NS_SING: u32 = 1024;
pub(super) const NS_SING_MC: u32 = 2048;
pub(super) const NS_SING2: u32 = 4096;
pub(super) const NS_CHECKEXT: u32 = 8192;
pub(super) const NS_MATE1: u32 = 16384;
pub(super) const NS_LMR_ADJ: u32 = 32768;
pub(super) const NS_DEEPER: u32 = 65536;
pub(super) const NS_NULLVER: u32 = 131072;
pub(super) const NS_CONT: u32 = 262144;
pub(super) const NS_QFUT: u32 = 524288;
pub(super) const NS_TTSTAT: u32 = 1048576;

/// 実験 NS の SPSA（cargo の機能 ns_tune）：探索の値。USI の隠しオプション NT_0〜NT_15 で変える。
/// 機能が無効なら既定の値（＝ 11/17 の NS と同じ読み）を使う。
/// 0 RFP の幅（1手あたり）／1 razoring の幅（深さ²あたり）／2 ProbCut の幅／3 LMP の手数（百分率）／
/// 4 駒取りの futility の基本の幅／5 同（1手あたり）／6 駒取り・成りの SEE の幅（1手あたり）／
/// 7 継続ヒストリーによる省略の境目（1手あたり）／8 静かな手の futility の基本の幅／9 同（1手あたり）／
/// 10 静かな手の SEE の幅（深さ²あたり）／11 LMR の表の基本（×100）／12 LMR の表の割り算（×100）／
/// 13 LMR で cut_node に足す量（1/1024 手）／14 LMR のヒストリーの割り算／15 singular の幅（×10、1手あたり）
pub const NT_DEFAULT: [i32; 16] = [170, 280, 200, 100, 200, 180, 190, 3500, 120, 130, 25, 50, 210, 1536, 16000, 30];
pub static NS_TUNE: [AtomicI32; 16] = [
    AtomicI32::new(170), AtomicI32::new(280), AtomicI32::new(200), AtomicI32::new(100),
    AtomicI32::new(200), AtomicI32::new(180), AtomicI32::new(190), AtomicI32::new(3500),
    AtomicI32::new(120), AtomicI32::new(130), AtomicI32::new(25), AtomicI32::new(50),
    AtomicI32::new(210), AtomicI32::new(1536), AtomicI32::new(16000), AtomicI32::new(30),
];

#[inline]
fn nt(i: usize) -> i32 {
    if cfg!(feature = "ns_tune") { NS_TUNE[i].load(Ordering::Relaxed) } else { NT_DEFAULT[i] }
}

/// LMR で浅くする量の表（1/1024 手）[深さ][何手目]
pub(super) fn red_table() -> Vec<i32> {
    let (base, div) = (nt(11) as f64 / 100.0, nt(12) as f64 / 100.0);
    (0..64 * 64)
        .map(|i| {
            let (d, m) = ((i / 64) as f64, (i % 64) as f64);
            if d < 1.0 || m < 1.0 { 0 } else { (1024.0 * (base + d.ln() * m.ln() / div)) as i32 }
        })
        .collect()
}

/// 手の並べ方の点数の帯
const SC_TT: i32 = i32::MAX;
/// 駒取り・成り（まだ SEE を調べていない）。GOOD＋点数（点数は 0〜5千万未満）
const SC_GOOD: i32 = 1_000_000_000;
const SC_GOOD_LO: i32 = 950_000_000;
/// キラー・カウンター
const SC_KILLER: i32 = 900_000_000;
const SC_KILLER_LO: i32 = 899_000_000;
/// 損な駒取り（SEE が 0 未満と分かったもの）
const SC_BAD: i32 = -1_000_000_000;
const SC_BAD_HI: i32 = -500_000_000;

/// 選んだ手の種類
const K_TT: u8 = 0;
const K_GOOD: u8 = 1;
const K_KILLER: u8 = 2;
const K_QUIET: u8 = 3;
const K_BAD: u8 = 4;

/// ヒストリーの上限
const H_MAX: i32 = 16384;
/// 手番×(元の升 か 打つ駒)×行き先
/// （元の升は 0〜80、打つ駒は 81＋駒種 1〜7 で、行は 89）
const MAIN_SIZE: usize = 2 * 89 * 81;

/// 探索のスレッドごとの表
pub(super) struct NsTables {
    pub(super) main: Vec<i16>,
    pub(super) cont: Vec<i16>,
    pub(super) cap: Vec<i16>,
    /// LMR で浅くする量（1/1024 手）[深さ][何手目]
    pub(super) red: Vec<i32>,
    /// 外す部品（NsOff）
    pub(super) off: u32,
    /// ヌルムーブの確かめの最中は、この ply より浅い所でヌルムーブをしない
    pub(super) nmp_min_ply: usize,
    /// 根からその ply までの2手延長の回数
    pub(super) dext: [i32; MAX_PLY + 2],
}

impl NsTables {
    pub(super) fn new(on: bool) -> NsTables {
        let red = if on { red_table() } else { Vec::new() };
        NsTables {
            main: if on { vec![0; MAIN_SIZE] } else { Vec::new() },
            cont: if on { vec![0; PT * PT] } else { Vec::new() },
            cap: if on { vec![0; PT * 16] } else { Vec::new() },
            red,
            off: 0,
            nmp_min_ply: 0,
            dext: [0; MAX_PLY + 2],
        }
    }
}

/// ヒストリーを、上限に向かって縮めながら動かす
#[inline]
fn upd(h: &mut i16, bonus: i32) {
    let v = *h as i32;
    let b = bonus.clamp(-H_MAX, H_MAX);
    *h = (v + b - v * b.abs() / H_MAX) as i16;
}

#[inline]
fn stat_bonus(d: i32) -> i32 {
    (140 * d - 60).clamp(0, 1600)
}

#[inline]
fn stat_malus(d: i32) -> i32 {
    (150 * d - 40).clamp(0, 1500)
}

/// 手番×(元の升 か 打つ駒)×行き先 の添字
#[inline]
fn main_idx(side: usize, m: Move) -> usize {
    let f = if mv_is_drop(m) { 81 + mv_drop_pt(m) as usize } else { mv_from(m) };
    (side * 89 + f) * 81 + mv_to(m)
}

#[inline]
fn piece_of(pos: &Position, m: Move) -> u8 {
    if mv_is_drop(m) { make_piece(pos.side, mv_drop_pt(m)) } else { pos.board[mv_from(m)] }
}

/// 駒を取る・成るときに得る価値（取った駒は持ち駒になるので、盤上の価値＋持ち駒の価値）
#[inline]
fn gain(pos: &Position, m: Move) -> i32 {
    let mut g = 0;
    if !mv_is_drop(m) {
        let t = pos.board[mv_to(m)];
        if t != 0 { g += CAPTURE_VALUE[type_of(t) as usize]; }
        if mv_is_promo(m) {
            let pt = type_of(pos.board[mv_from(m)]) as usize;
            g += PIECE_VALUE[pt + PROMOTE as usize] - PIECE_VALUE[pt];
        }
    }
    g
}

impl Searcher {
    /// 静かな手のヒストリーの合計（継続ヒストリーの 1・2・4手前）
    #[inline]
    fn ns_cont(&self, pidx: usize, ply: usize) -> i32 {
        if self.ns.off & NS_CONT != 0 { return 0; }
        self.ns_cont_at(&self.ns_cont_rows(ply), pidx)
    }

    /// 継続ヒストリーの 1・2・4手前の行の先頭（局面ごとに1回だけ計算する）
    #[inline]
    fn ns_cont_rows(&self, ply: usize) -> [usize; 3] {
        [self.prev_pm(ply, 1) * PT, self.prev_pm(ply, 2) * PT, self.prev_pm(ply, 4) * PT]
    }

    #[inline]
    fn ns_cont_at(&self, rows: &[usize; 3], pidx: usize) -> i32 {
        let c = &self.ns.cont;
        c[rows[0] + pidx] as i32 + c[rows[1] + pidx] as i32 + c[rows[2] + pidx] as i32 / 2
    }

    fn ns_score_moves(&self, pos: &Position, list: &MoveList, scores: &mut [i32], tt_move: Move, ply: usize) {
        let counter = Self::counter_index(pos).map(|i| self.counter[i]).unwrap_or(MOVE_NONE);
        let (k0, k1) = (self.killers[ply][0], self.killers[ply][1]);
        let cont_on = self.ns.off & NS_CONT == 0;
        let rows = self.ns_cont_rows(ply);
        let main_base = pos.side * 89 * 81;
        for i in 0..list.len {
            let m = list.moves[i];
            if m == tt_move {
                scores[i] = SC_TT;
                continue;
            }
            let to = mv_to(m);
            let is_cap = !mv_is_drop(m) && pos.board[to] != 0;
            let piece = piece_of(pos, m);
            if is_cap || mv_is_promo(m) {
                let ch = if is_cap {
                    self.ns.cap[(piece as usize * 81 + to) * 16 + type_of(pos.board[to]) as usize] as i32
                } else { 0 };
                scores[i] = SC_GOOD + (8 * gain(pos, m) + ch / 4).clamp(0, 40_000_000);
            } else if m == k0 {
                scores[i] = SC_KILLER + 2;
            } else if m == k1 {
                scores[i] = SC_KILLER + 1;
            } else if m == counter {
                scores[i] = SC_KILLER;
            } else {
                let pidx = piece as usize * 81 + to;
                let f = if mv_is_drop(m) { 81 + mv_drop_pt(m) as usize } else { mv_from(m) };
                let c = if cont_on { self.ns_cont_at(&rows, pidx) } else { 0 };
                scores[i] = 2 * self.ns.main[main_base + f * 81 + to] as i32 + c;
            }
        }
    }

    /// 残りの手から点数の最も高い手を選ぶ。駒取り・成りは、選ぶときに SEE を調べ、損なら後ろに回して選び直す
    #[inline]
    fn ns_pick(&self, pos: &Position, list: &mut MoveList, scores: &mut [i32], i: usize) -> (Move, u8) {
        loop {
            // 残りの中で最初に現れる最大の点数の手（1回なめるだけ）
            let mut best = i;
            let mut max = scores[i];
            for (j, &s) in scores.iter().enumerate().take(list.len).skip(i + 1) {
                if s > max { max = s; best = j; }
            }
            let kind = if max == SC_TT {
                K_TT
            } else if max >= SC_GOOD_LO {
                if !self.exchange_ge(pos, list.moves[best], 0) {
                    scores[best] = SC_BAD + (max - SC_GOOD);
                    continue;
                }
                K_GOOD
            } else if max >= SC_KILLER_LO {
                K_KILLER
            } else if max > SC_BAD_HI {
                K_QUIET
            } else {
                K_BAD
            };
            list.moves.swap(i, best);
            scores.swap(i, best);
            return (list.moves[i], kind);
        }
    }

    /// LMR で浅くする量（1/1024 手）
    #[inline]
    fn ns_reduction(&self, depth: i32, mc: i32, improving: bool) -> i32 {
        self.ns.red[(depth.clamp(0, 63) as usize) * 64 + mc.clamp(0, 63) as usize] + if improving { 0 } else { 512 }
    }

    /// 静かな手がカットしたとき：その手を上げ、先に試して失敗した静かな手を下げる。キラー・カウンターも覚える
    fn ns_update_quiet(&mut self, pos: &Position, m: Move, ply: usize, bonus: i32) {
        let side = pos.side;
        upd(&mut self.ns.main[main_idx(side, m)], bonus);
        let pidx = piece_of(pos, m) as usize * 81 + mv_to(m);
        self.ns_update_cont(pidx, ply, bonus);
    }

    #[inline]
    fn ns_update_cont(&mut self, pidx: usize, ply: usize, bonus: i32) {
        for (k, w) in [(1usize, 1i32), (2, 1), (4, 2)] {
            if ply >= k {
                let c = self.prev_pm(ply, k) * PT + pidx;
                upd(&mut self.ns.cont[c], bonus / w);
            }
        }
    }

    /// 通常の探索。cut_node はカットが見込まれる局面か（PV でない局面の子は、交互に入れ替わる）
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ns_search(&mut self, pos: &mut Position, mut alpha: i32, mut beta: i32, mut depth: i32,
                            ply: usize, cut_node: bool, excluded: Move) -> i32 {
        let pv_node = beta - alpha > 1;
        let root = ply == 0;
        self.pv_len[ply] = 0;
        if depth <= 0 {
            return self.ns_qsearch(pos, alpha, beta, ply, 0);
        }
        self.nodes += 1;
        ts!(self, main);
        let d0 = depth;
        tsd!(self, d_main, d0);
        self.check_time();
        if self.stopped {
            return 0;
        }
        self.sel_depth = self.sel_depth.max(ply);
        let off = self.ns.off;

        if !root {
            if cfg!(feature = "perpetual_check") {
                if let Some(r) = prof!(self, 11, pos.repetition_result()) {
                    return r * repetition_win(ply);
                }
            } else if prof!(self, 11, pos.is_repetition()) {
                return 0;
            }
            if cfg!(feature = "nyugyoku") && prof!(self, 12, pos.can_declare()) {
                return repetition_win(ply);
            }
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

        // 置換表
        let key = pos.key;
        let tt_entry = if excluded == MOVE_NONE { prof!(self, 9, self.tt.probe(key)) } else { None };
        let tt_move = tt_entry.map_or(MOVE_NONE, |e| e.mv);
        let (tt_score, tt_depth, tt_flag) = match tt_entry {
            Some(e) => (score_from_tt(e.score as i32, ply), e.depth as i32, e.flag),
            None => (-INF, -100, 0),
        };
        if let Some(_) = tt_entry {
            ts!(self, tt_hit);
            if !pv_node && tt_depth >= depth
                && (tt_flag == BOUND_EXACT
                    || (tt_flag == BOUND_LOWER && tt_score >= beta)
                    || (tt_flag == BOUND_UPPER && tt_score <= alpha))
            {
                // 置換表の静かな手で β を超えたら、その手のヒストリーを上げる
                if off & NS_TTSTAT == 0 && tt_move != MOVE_NONE && tt_score >= beta && pos.is_pseudo_legal(tt_move)
                    && !mv_is_promo(tt_move) && (mv_is_drop(tt_move) || pos.board[mv_to(tt_move)] == 0)
                {
                    self.ns_update_quiet(pos, tt_move, ply, stat_bonus(depth));
                }
                ts!(self, tt_cut);
                tsd!(self, d_tt_cut, d0);
                return tt_score;
            }
        }

        let in_check = pos.in_check();
        // 1手詰め（置換表に無い局面）
        // 実験 M1（cargo の機能 ns_mate1d1）：1手詰めを、残りの深さ1の局面だけで調べる
        // 機能 ns_nomate1（11/21 既定に、実験 M1 の MA と同じ読み）：通常の探索では1手詰めを調べない
        if !cfg!(feature = "ns_nomate1") && off & NS_MATE1 == 0 && !root && !in_check && excluded == MOVE_NONE && tt_entry.is_none()
            && (!cfg!(feature = "ns_mate1d1") || depth <= 1)
        {
            if let Some(m) = prof!(self, 15, pos.mate_in_one()) {
                let s = MATE - ply as i32 - 1;
                self.pv[ply * PV_STRIDE] = m;
                self.pv_len[ply] = 1;
                self.tt.store(key, m, score_to_tt(s, ply), depth, BOUND_EXACT);
                return s;
            }
        }

        // 静的評価（置換表の値が評価より確かな向きなら、そちらを使う）
        self.ensure_acc(pos, ply);
        let raw = if in_check { -INF } else { self.eval(pos, ply) };
        self.static_evals[ply] = raw;
        let mut eval = raw;
        if !in_check && tt_entry.is_some() && tt_score.abs() < MATE_BOUND
            && (tt_flag == BOUND_EXACT
                || (tt_flag == BOUND_LOWER && tt_score > raw)
                || (tt_flag == BOUND_UPPER && tt_score < raw))
        {
            eval = tt_score;
        }
        // 2手前（王手をかけられていれば4手前）より良くなっているか
        let improving = !in_check && if ply >= 2 && self.static_evals[ply - 2] != -INF {
            raw > self.static_evals[ply - 2]
        } else if ply >= 4 && self.static_evals[ply - 4] != -INF {
            raw > self.static_evals[ply - 4]
        } else {
            false
        };

        if !pv_node && !in_check && excluded == MOVE_NONE && !root {
            // razoring：評価が α よりかなり低ければ、静止探索で確かめて打ち切る
            if off & NS_RAZOR == 0 && eval < alpha - 450 - nt(1) * depth * depth {
                let v = self.ns_qsearch(pos, alpha - 1, alpha, ply, 0);
                if self.stopped { return 0; }
                if v < alpha && v.abs() < MATE_BOUND {
                    return v;
                }
            }
            // 静的評価による打ち切り
            if off & NS_RFP == 0 && depth < 9 && eval.abs() < MATE_BOUND
                && eval - nt(0) * (depth - i32::from(improving)) >= beta
            {
                ts!(self, rfp);
                tsd!(self, d_rfp, d0);
                return eval;
            }
            // ヌルムーブ
            let prev_null = ply >= 1 && self.ss_pm[ply - 1] == (0, 0);
            if off & NS_NULL == 0 && !prev_null && ply >= self.ns.nmp_min_ply && eval >= beta
                && raw >= beta - 10 * depth + 150 && beta.abs() < MATE_BOUND
            {
                let r = 3 + depth / 3 + ((eval - beta) / 120).clamp(0, 3);
                self.set_pending_null(ply);
                self.ss_pm[ply] = (0, 0);
                ts!(self, null_try);
                pos.do_null();
                let v = -self.ns_search(pos, -beta, -beta + 1, depth - r, ply + 1, !cut_node, MOVE_NONE);
                pos.undo_null();
                if self.stopped { return 0; }
                if v >= beta && v < MATE_BOUND {
                    if off & NS_NULLVER != 0 || self.ns.nmp_min_ply != 0 || depth < 14 {
                        ts!(self, null_cut);
                        tsd!(self, d_null_cut, d0);
                        return v;
                    }
                    // 深いときは、ヌルムーブなしで確かめる
                    self.ns.nmp_min_ply = ply + (3 * (depth - r) / 4).max(0) as usize;
                    let v2 = self.ns_search(pos, beta - 1, beta, depth - r, ply, false, MOVE_NONE);
                    self.ns.nmp_min_ply = 0;
                    self.static_evals[ply] = raw;
                    if self.stopped { return 0; }
                    if v2 >= beta {
                        ts!(self, null_cut);
                        tsd!(self, d_null_cut, d0);
                        return v;
                    }
                }
            }
            // ProbCut：β をはっきり超えそうな駒取りを、静止探索 → 4手浅い探索で確かめる
            let pc_beta = beta + nt(2) - 60 * i32::from(improving);
            if off & NS_PROBCUT == 0 && depth >= 5 && beta.abs() < MATE_BOUND
                && !(tt_entry.is_some() && tt_depth >= depth - 3 && tt_score < pc_beta)
            {
                let mut caps = MoveList::new();
                prof!(self, 0, pos.gen_moves(&mut caps, true));
                let mut sc = [0i32; MAX_MOVES];
                prof!(self, 2, self.ns_score_moves(pos, &caps, &mut sc, tt_move, ply));
                for i in 0..caps.len {
                    let (m, _) = prof!(self, 13, self.ns_pick(pos, &mut caps, &mut sc, i));
                    if !self.exchange_ge(pos, m, pc_beta - raw) { continue; }
                    let piece = piece_of(pos, m);
                    self.set_pending(pos, m, ply);
                    if !prof!(self, 3, pos.do_move_legal_ic(m, in_check)) { continue; }
                    self.after_move(pos, ply);
                    self.ss_pm[ply] = (piece, mv_to(m));
                    let mut v = -self.ns_qsearch(pos, -pc_beta, -pc_beta + 1, ply + 1, 0);
                    if v >= pc_beta {
                        v = -self.ns_search(pos, -pc_beta, -pc_beta + 1, depth - 4, ply + 1, !cut_node, MOVE_NONE);
                    }
                    prof!(self, 4, pos.undo_move());
                    if self.stopped { return 0; }
                    if v >= pc_beta {
                        self.tt.store(key, m, score_to_tt(v, ply), depth - 3, BOUND_LOWER);
                        return v - (pc_beta - beta);
                    }
                }
            }
        }

        // 置換表に手が無いときは、並べ替えが効かないので浅くする
        if off & NS_IIR == 0 && excluded == MOVE_NONE && tt_move == MOVE_NONE && depth >= 4 && (pv_node || cut_node) {
            depth -= 1;
        }

        let mut list = MoveList::new();
        prof!(self, 0, pos.gen_moves(&mut list, false));
        let mut scores = [0i32; MAX_MOVES];
        prof!(self, 2, self.ns_score_moves(pos, &list, &mut scores, tt_move, ply));

        let orig_alpha = alpha;
        let mut best = -INF;
        let mut best_move = MOVE_NONE;
        let mut move_count = 0;
        let mut searched = 0;
        let mut skip_quiets = false;
        let mut quiets_tried = [MOVE_NONE; 64];
        let mut n_quiets = 0;
        let mut caps_tried = [MOVE_NONE; 32];
        let mut n_caps = 0;
        let tt_capture = tt_move != MOVE_NONE && !mv_is_drop(tt_move) && pos.board[mv_to(tt_move)] != 0;
        let lmp_limit = ((3 + depth * depth) * nt(3) / 100) / (2 - i32::from(improving));
        let us = pos.side;

        let mut next = 0;
        while next < list.len {
            let (m, kind) = prof!(self, 13, self.ns_pick(pos, &mut list, &mut scores, next));
            next += 1;
            if m == excluded { continue; }
            if root && self.root_excluded.contains(&m) { continue; }
            let to = mv_to(m);
            let is_cap = !mv_is_drop(m) && pos.board[to] != 0;
            let is_promo = mv_is_promo(m);
            let quiet = !is_cap && !is_promo;
            let piece = piece_of(pos, m);
            let pidx = piece as usize * 81 + to;
            let gives_check = prof!(self, 5, pos.gives_check(m));
            move_count += 1;
            let mut new_depth = depth - 1;
            let r_base = self.ns_reduction(depth, move_count, improving);

            // 指す前の省略
            if !root && !in_check && best > -MATE_BOUND {
                if off & NS_LMP == 0 && move_count >= lmp_limit {
                    skip_quiets = true;
                }
                let lmr_depth = new_depth - r_base / 1024;
                if !quiet || gives_check {
                    if !gives_check && off & NS_CAPFUT == 0 && lmr_depth < 7 {
                        let ch = if is_cap { self.ns.cap[pidx * 16 + type_of(pos.board[to]) as usize] as i32 } else { 0 };
                        if raw + nt(4) + nt(5) * lmr_depth + gain(pos, m) + ch / 16 <= alpha {
                            continue;
                        }
                    }
                    if off & NS_SEE == 0 && kind != K_GOOD && !self.exchange_ge(pos, m, -nt(6) * depth) {
                        continue;
                    }
                } else {
                    if skip_quiets {
                        ts!(self, lmp);
                        continue;
                    }
                    let ch = self.ns_cont(pidx, ply);
                    if off & NS_CHP == 0 && lmr_depth < 6 && ch < -nt(7) * depth {
                        continue;
                    }
                    let hist = ch + 2 * self.ns.main[main_idx(us, m)] as i32;
                    let ld = lmr_depth + hist / 16000;
                    if off & NS_FUT == 0 && ld < 10 && raw + nt(8) + nt(9) * ld <= alpha {
                        ts!(self, fut);
                        continue;
                    }
                    let ld = ld.max(0);
                    if off & NS_SEE == 0 && !self.exchange_ge(pos, m, -nt(10) * ld * ld) {
                        continue;
                    }
                }
            }

            // 延長
            let mut ext = 0;
            if off & NS_SING == 0 && !root && m == tt_move && excluded == MOVE_NONE && !self.singular_verifying
                && depth >= 7 && ply < (2 * self.iteration_depth) as usize
                && tt_depth >= depth - 3 && tt_flag & BOUND_LOWER != 0 && tt_score.abs() < MATE_BOUND
            {
                let legal = prof!(self, 3, pos.do_move_legal_ic(m, in_check));
                if legal {
                    prof!(self, 4, pos.undo_move());
                    let sbeta = tt_score - nt(15) * depth / 10;
                    self.singular_verifying = true;
                    let v = self.ns_search(pos, sbeta - 1, sbeta, (depth - 1) / 2, ply, cut_node, m);
                    self.singular_verifying = false;
                    self.pv_len[ply] = 0;
                    self.static_evals[ply] = raw;
                    if self.stopped { return 0; }
                    if v < sbeta {
                        ext = 1;
                        if off & NS_SING2 == 0 && !pv_node && v < sbeta - 60 && self.ns.dext[ply] < 6 {
                            ext = 2;
                        }
                    } else if off & NS_SING_MC == 0 && v >= beta && v.abs() < MATE_BOUND {
                        // ほかの手でも β を超えるので、この局面はカットとみなす
                        return v;
                    } else if off & NS_SING_MC == 0 && tt_score >= beta {
                        ext = -2;
                    } else if off & NS_SING_MC == 0 && cut_node {
                        ext = -1;
                    }
                }
            }
            if ext == 0 && off & NS_CHECKEXT == 0 && gives_check && ply < (2 * self.iteration_depth) as usize
                && (kind == K_GOOD || self.exchange_ge(pos, m, 0))
            {
                ext = 1;
                ts!(self, ext_check);
            }
            new_depth += ext;
            self.ns.dext[ply + 1] = self.ns.dext[ply] + i32::from(ext == 2);

            self.set_pending(pos, m, ply);
            if !prof!(self, 3, pos.do_move_legal_ic(m, in_check)) {
                move_count -= 1;
                continue;
            }
            self.after_move(pos, ply);
            self.ss_pm[ply] = (piece, to);
            searched += 1;
            ts!(self, moves);
            tsd!(self, d_moves, d0);

            let mut v;
            if pv_node && searched == 1 {
                v = -self.ns_search(pos, -beta, -alpha, new_depth, ply + 1, false, MOVE_NONE);
            } else {
                if depth >= 2 && searched > 1 + usize::from(root) && (quiet || kind == K_BAD) {
                    // LMR
                    let mut r = r_base;
                    if off & NS_LMR_ADJ == 0 {
                        if pv_node { r -= 1024; }
                        if cut_node { r += nt(13); }
                        if tt_capture { r += 1024; }
                        if gives_check { r -= 1024; }
                        if kind == K_KILLER { r -= 768; }
                        if quiet {
                            let ss = 2 * self.ns.main[main_idx(us, m)] as i32 + self.ns_cont(pidx, ply) - 3000;
                            r -= ss * 1024 / nt(14);
                        }
                    }
                    let hi = new_depth.max(1) + i32::from(pv_node);
                    let d = (new_depth - r / 1024).clamp(1, hi);
                    if d < new_depth {
                        ts!(self, lmr);
                        tsd!(self, d_lmr, d0);
                        tsd!(self, d_red, d0, new_depth - d);
                    }
                    v = -self.ns_search(pos, -alpha - 1, -alpha, d, ply + 1, true, MOVE_NONE);
                    if v > alpha && d < new_depth {
                        ts!(self, lmr_re);
                        if off & NS_DEEPER == 0 {
                            let deeper = v > best + 50 + 2 * new_depth;
                            let shallower = v < best + 9;
                            new_depth += i32::from(deeper) - i32::from(shallower);
                        }
                        if new_depth > d {
                            v = -self.ns_search(pos, -alpha - 1, -alpha, new_depth, ply + 1, !cut_node, MOVE_NONE);
                        }
                    }
                } else {
                    v = -self.ns_search(pos, -alpha - 1, -alpha, new_depth, ply + 1, !cut_node, MOVE_NONE);
                }
                if pv_node && v > alpha && (root || v < beta) {
                    ts!(self, pvs_re);
                    v = -self.ns_search(pos, -beta, -alpha, new_depth, ply + 1, false, MOVE_NONE);
                }
            }
            prof!(self, 4, pos.undo_move());
            if self.stopped {
                return 0;
            }

            if v > best {
                best = v;
                if v > alpha {
                    best_move = m;
                    if cfg!(feature = "t_partial") && root && excluded == MOVE_NONE {
                        self.root_partial = m;
                    }
                    if pv_node && excluded == MOVE_NONE {
                        self.update_pv(ply, m);
                    }
                    if v >= beta {
                        ts!(self, cut);
                        tsd!(self, d_cut, d0);
                        if searched == 1 { tsd!(self, d_cut_first, d0); }
                        break;
                    }
                    alpha = v;
                }
            }
            if m != best_move {
                if quiet {
                    if n_quiets < 64 { quiets_tried[n_quiets] = m; n_quiets += 1; }
                } else if is_cap && n_caps < 32 {
                    caps_tried[n_caps] = m;
                    n_caps += 1;
                }
            }
        }

        if searched == 0 {
            if excluded != MOVE_NONE {
                return alpha;
            }
            return -MATE + ply as i32;
        }

        // ヒストリー
        if best >= beta && best_move != MOVE_NONE && excluded == MOVE_NONE {
            let bonus = stat_bonus(depth + i32::from(best > beta + 150));
            let malus = stat_malus(depth);
            let bm = best_move;
            let bm_cap = !mv_is_drop(bm) && pos.board[mv_to(bm)] != 0;
            if !bm_cap && !mv_is_promo(bm) {
                self.ns_update_quiet(pos, bm, ply, bonus);
                if self.killers[ply][0] != bm {
                    self.killers[ply][1] = self.killers[ply][0];
                    self.killers[ply][0] = bm;
                }
                if let Some(ci) = Self::counter_index(pos) {
                    self.counter[ci] = bm;
                }
                for &q in &quiets_tried[..n_quiets] {
                    self.ns_update_quiet(pos, q, ply, -malus);
                }
            } else if bm_cap {
                let i = (piece_of(pos, bm) as usize * 81 + mv_to(bm)) * 16 + type_of(pos.board[mv_to(bm)]) as usize;
                upd(&mut self.ns.cap[i], bonus);
            }
            for &c in &caps_tried[..n_caps] {
                let i = (piece_of(pos, c) as usize * 81 + mv_to(c)) * 16 + type_of(pos.board[mv_to(c)]) as usize;
                upd(&mut self.ns.cap[i], -malus);
            }
        }

        let flag = if best >= beta {
            BOUND_LOWER
        } else if pv_node && best_move != MOVE_NONE {
            BOUND_EXACT
        } else {
            BOUND_UPPER
        };
        let _ = orig_alpha;
        if excluded == MOVE_NONE {
            prof!(self, 10, self.tt.store(key, best_move, score_to_tt(best, ply), depth, flag));
        }
        best
    }

    /// 静止探索
    pub(super) fn ns_qsearch(&mut self, pos: &mut Position, mut alpha: i32, beta: i32, ply: usize, qdepth: usize) -> i32 {
        self.pv_len[ply] = 0;
        self.nodes += 1;
        ts!(self, qs);
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
        let off = self.ns.off;
        let pv_node = beta - alpha > 1;
        let key = pos.key;
        let mut tt_move = MOVE_NONE;
        let mut tt_val: Option<(i32, u8)> = None;
        if let Some(e) = prof!(self, 9, self.tt.probe(key)) {
            tt_move = e.mv;
            let s = score_from_tt(e.score as i32, ply);
            if !pv_node
                && (e.flag == BOUND_EXACT
                    || (e.flag == BOUND_LOWER && s >= beta)
                    || (e.flag == BOUND_UPPER && s <= alpha))
            {
                return s;
            }
            tt_val = Some((s, e.flag));
        }

        self.ensure_acc(pos, ply);
        let in_check = pos.in_check();
        let mut list = MoveList::new();
        let mut best;
        let futility_base;
        if in_check {
            best = -INF;
            futility_base = -INF;
            prof!(self, 1, pos.gen_moves(&mut list, false));
        } else {
            let raw = self.eval(pos, ply);
            let mut stand = raw;
            if let Some((s, f)) = tt_val {
                if s.abs() < MATE_BOUND
                    && (f == BOUND_EXACT || (f == BOUND_LOWER && s > raw) || (f == BOUND_UPPER && s < raw))
                {
                    stand = s;
                }
            }
            if stand >= beta {
                ts!(self, qs_standpat);
                return stand;
            }
            alpha = alpha.max(stand);
            best = stand;
            futility_base = raw + 150;
            prof!(self, 1, pos.gen_moves(&mut list, true));
        }

        let mut scores = [0i32; MAX_MOVES];
        prof!(self, 2, self.ns_score_moves(pos, &list, &mut scores, tt_move, ply));
        let mut best_move = MOVE_NONE;
        let mut searched = 0;
        for i in 0..list.len {
            let (m, kind) = prof!(self, 13, self.ns_pick(pos, &mut list, &mut scores, i));
            if !in_check {
                // 駒取りだけを読む（置換表の手が静かな手なら飛ばす）
                if mv_is_drop(m) || pos.board[mv_to(m)] == 0 { continue; }
                if off & NS_QFUT == 0 {
                    let fv = futility_base + gain(pos, m);
                    if fv <= alpha {
                        best = best.max(fv);
                        continue;
                    }
                    if futility_base <= alpha && !self.exchange_ge(pos, m, 1) {
                        best = best.max(futility_base);
                        continue;
                    }
                }
                // 取り合いで損をする駒取りは読まない
                if kind == K_BAD || (kind == K_TT && !self.exchange_ge(pos, m, 0)) {
                    continue;
                }
            }
            let piece = piece_of(pos, m);
            self.set_pending(pos, m, ply);
            if !prof!(self, 3, pos.do_move_legal_ic(m, in_check)) {
                continue;
            }
            searched += 1;
            self.ss_pm[ply] = (piece, mv_to(m));
            self.after_move(pos, ply);
            if in_check { ts!(self, qs_evasions); } else { ts!(self, qs_moves); }
            let v = -self.ns_qsearch(pos, -beta, -alpha, ply + 1, qdepth + 1);
            prof!(self, 4, pos.undo_move());
            if self.stopped {
                return 0;
            }
            if v > best {
                best = v;
                if v > alpha {
                    best_move = m;
                    self.update_pv(ply, m);
                    if v >= beta {
                        break;
                    }
                    alpha = v;
                }
            }
        }

        if in_check && searched == 0 {
            return -MATE + ply as i32;
        }
        let flag = if best >= beta {
            BOUND_LOWER
        } else if pv_node && best_move != MOVE_NONE {
            BOUND_EXACT
        } else {
            BOUND_UPPER
        };
        prof!(self, 10, self.tt.store_qs(key, best_move, score_to_tt(best, ply), flag));
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn searcher() -> Searcher {
        let mut s = Searcher::new(16, Arc::new(Shared::new()));
        s.ns = NsTables::new(true);
        s
    }

    fn run(s: &mut Searcher, pos: &mut Position, depth: i32) -> (i32, Move) {
        s.stopped = false;
        s.nodes = 0;
        s.iteration_depth = depth;
        s.ns.off = 0;
        let mut v = 0;
        for d in 1..=depth {
            s.iteration_depth = d;
            v = s.ns_search(pos, -INF, INF, d, 0, false, MOVE_NONE);
        }
        (v, s.pv[0])
    }

    #[test]
    fn ns_finds_mate_in_one_and_restores_position() {
        // 頭金で詰む
        let mut p = Position::from_sfen("4k4/9/4P4/9/9/9/9/9/4K4 b G 1").unwrap();
        let original = (p.board, p.hand, p.side, p.key);
        let mut s = searcher();
        let (v, m) = run(&mut s, &mut p, 3);
        assert_eq!(v, MATE - 1);
        assert_eq!(move_to_usi(m), "G*5b");
        assert_eq!((p.board, p.hand, p.side, p.key), original);
    }

    #[test]
    fn ns_qsearch_takes_hanging_rook() {
        // 守られていない飛車を金で取る
        let mut p = Position::from_sfen("kgg6/9/4r4/4G4/9/9/9/9/8K b - 1").unwrap();
        let original = (p.board, p.hand, p.side, p.key);
        let mut s = searcher();
        let v = s.ns_qsearch(&mut p, 0, 1000, 0, 0);
        assert!(v > 0, "v {v}");
        assert_eq!(move_to_usi(s.pv[0]), "5d5c");
        assert_eq!((p.board, p.hand, p.side, p.key), original);
    }

    #[test]
    fn ns_search_from_startpos_is_stable() {
        let mut p = Position::startpos();
        let original = (p.board, p.hand, p.side, p.key);
        let mut s = searcher();
        let (v, m) = run(&mut s, &mut p, 7);
        assert!(v.abs() < 1000, "v {v}");
        assert!(p.legal_moves().contains(&m));
        assert_eq!((p.board, p.hand, p.side, p.key), original);
    }
}
