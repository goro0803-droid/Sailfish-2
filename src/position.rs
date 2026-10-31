//! 局面の表現・SFEN の読み込み・指し手生成・指し手の実行と取り消し。

use crate::types::*;

pub const START_SFEN: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
pub const NO_SQ: usize = 81;
pub const MAX_MOVES: usize = 1024;
/// Material swing when captured: removed board value plus the capturer's hand value.
pub const CAPTURE_VALUE: [i32; 15] = [
    0, 190, 655, 845, 1035, 1805, 2090, 1140, 20000,
    640, 880, 980, 1080, 1895, 2495,
];

pub struct MoveList {
    pub moves: [Move; MAX_MOVES],
    pub len: usize,
}

impl MoveList {
    pub fn new() -> Self {
        MoveList { moves: [0; MAX_MOVES], len: 0 }
    }
    #[inline]
    pub fn push(&mut self, m: Move) {
        self.moves[self.len] = m;
        self.len += 1;
    }
}

#[derive(Clone, Copy)]
struct StateInfo {
    mv: Move, // MOVE_NONE ならヌルムーブ
    captured: u8,
    key: u64, // 指す前の局面のハッシュ
}

#[derive(Clone)]
pub struct Position {
    pub board: [u8; 81],
    pub hand: [[u8; 8]; 2],
    pub side: usize,
    pub king: [usize; 2],
    pub key: u64,
    history: Vec<StateInfo>,
}

fn char_to_piece(ch: char) -> Option<(usize, u8)> {
    let c = if ch.is_ascii_uppercase() { BLACK } else { WHITE };
    let pt = match ch.to_ascii_uppercase() {
        'P' => PAWN,
        'L' => LANCE,
        'N' => KNIGHT,
        'S' => SILVER,
        'B' => BISHOP,
        'R' => ROOK,
        'G' => GOLD,
        'K' => KING,
        _ => return None,
    };
    Some((c, pt))
}

impl Position {
    pub fn startpos() -> Position {
        Position::from_sfen(START_SFEN).unwrap()
    }

    pub fn from_sfen(sfen: &str) -> Result<Position, String> {
        let parts: Vec<&str> = sfen.split_whitespace().collect();
        if parts.len() < 3 {
            return Err(format!("invalid sfen: {}", sfen));
        }
        let mut pos = Position {
            board: [0; 81],
            hand: [[0; 8]; 2],
            side: BLACK,
            king: [NO_SQ; 2],
            key: 0,
            history: Vec::with_capacity(1024),
        };

        // 盤面: a段から i段へ、各段は 9筋から 1筋へ
        let mut rank = 0usize;
        let mut file: i32 = 8;
        let mut promo = false;
        for ch in parts[0].chars() {
            match ch {
                '/' => {
                    rank += 1;
                    file = 8;
                }
                '1'..='9' => file -= ch as i32 - '0' as i32,
                '+' => promo = true,
                _ => {
                    let (c, pt) = char_to_piece(ch).ok_or(format!("bad piece '{}'", ch))?;
                    let pt = if promo { pt + PROMOTE } else { pt };
                    promo = false;
                    if file < 0 || rank > 8 {
                        return Err("board overflow".into());
                    }
                    let sq = file as usize * 9 + rank;
                    pos.board[sq] = make_piece(c, pt);
                    if pt == KING {
                        pos.king[c] = sq;
                    }
                    file -= 1;
                }
            }
        }

        pos.side = match parts[1] {
            "b" => BLACK,
            "w" => WHITE,
            _ => return Err("bad side".into()),
        };

        if parts[2] != "-" {
            let mut n = 0u8;
            for ch in parts[2].chars() {
                if let Some(d) = ch.to_digit(10) {
                    n = n * 10 + d as u8;
                } else {
                    let (c, pt) = char_to_piece(ch).ok_or(format!("bad hand '{}'", ch))?;
                    pos.hand[c][pt as usize] += n.max(1);
                    n = 0;
                }
            }
        }

        pos.key = pos.compute_key();
        Ok(pos)
    }

    /// 盤・持ち駒・手番から局面を作る（SFEN を経由しない。教師データの読み込み用）
    pub fn from_parts(board: [u8; 81], hand: [[u8; 8]; 2], side: usize) -> Position {
        let mut king = [NO_SQ; 2];
        for (sq, &p) in board.iter().enumerate() {
            if p != 0 && type_of(p) == KING { king[color_of(p)] = sq; }
        }
        let mut pos = Position { board, hand, side, king, key: 0, history: Vec::with_capacity(1024) };
        pos.key = pos.compute_key();
        pos
    }

    fn compute_key(&self) -> u64 {
        let mut k = 0u64;
        for sq in 0..81 {
            let p = self.board[sq];
            if p != 0 {
                k ^= ZOB.psq[p as usize][sq];
            }
        }
        for c in 0..2 {
            for pt in 1..8 {
                k ^= ZOB.hand[c][pt][self.hand[c][pt] as usize];
            }
        }
        if self.side == WHITE {
            k ^= ZOB.side;
        }
        k
    }

    #[inline]
    fn add_hand(&mut self, c: usize, pt: u8, delta: i8) {
        let pt = pt as usize;
        let old = self.hand[c][pt];
        let new = (old as i8 + delta) as u8;
        self.key ^= ZOB.hand[c][pt][old as usize] ^ ZOB.hand[c][pt][new as usize];
        self.hand[c][pt] = new;
    }

    pub fn do_move(&mut self, m: Move) {
        let us = self.side;
        let to = mv_to(m);
        let mut st = StateInfo { mv: m, captured: 0, key: self.key };
        if mv_is_drop(m) {
            let pt = mv_drop_pt(m);
            self.add_hand(us, pt, -1);
            let p = make_piece(us, pt);
            self.board[to] = p;
            self.key ^= ZOB.psq[p as usize][to];
        } else {
            let from = mv_from(m);
            let p = self.board[from];
            let cap = self.board[to];
            if cap != 0 {
                st.captured = cap;
                self.key ^= ZOB.psq[cap as usize][to];
                self.add_hand(us, unpromote(type_of(cap)), 1);
            }
            self.key ^= ZOB.psq[p as usize][from];
            self.board[from] = 0;
            let np = if mv_is_promo(m) { p + PROMOTE } else { p };
            self.board[to] = np;
            self.key ^= ZOB.psq[np as usize][to];
            if type_of(p) == KING {
                self.king[us] = to;
            }
        }
        self.history.push(st);
        self.side ^= 1;
        self.key ^= ZOB.side;
    }

    pub fn undo_move(&mut self) {
        let st = self.history.pop().expect("undo_move with empty history");
        self.side ^= 1;
        let us = self.side;
        let m = st.mv;
        let to = mv_to(m);
        if mv_is_drop(m) {
            self.board[to] = 0;
            self.hand[us][mv_drop_pt(m) as usize] += 1;
        } else {
            let from = mv_from(m);
            let p = self.board[to];
            let orig = if mv_is_promo(m) { p - PROMOTE } else { p };
            self.board[from] = orig;
            self.board[to] = st.captured;
            if st.captured != 0 {
                self.hand[us][unpromote(type_of(st.captured)) as usize] -= 1;
            }
            if type_of(orig) == KING {
                self.king[us] = from;
            }
        }
        self.key = st.key;
    }

    pub fn do_null(&mut self) {
        self.history.push(StateInfo { mv: MOVE_NONE, captured: 0, key: self.key });
        self.side ^= 1;
        self.key ^= ZOB.side;
    }

    pub fn undo_null(&mut self) {
        let st = self.history.pop().unwrap();
        self.side ^= 1;
        self.key = st.key;
    }

    /// 升 sq に手番 by の駒が利いているか
    pub fn attacked(&self, sq: usize, by: usize) -> bool {
        let opp = by ^ 1;
        for d in 0..10 {
            let a = NEIGHBOR[opp][sq][d];
            if a < 0 {
                continue;
            }
            let a = a as usize;
            let p = self.board[a];
            if p != 0 {
                if color_of(p) == by {
                    let pt = type_of(p) as usize;
                    if (STEP_MASK[pt] | SLIDE_MASK[pt]) & (1 << d) != 0 {
                        return true;
                    }
                }
                continue;
            }
            if d >= 8 {
                continue;
            }
            let mut s = a;
            loop {
                let n = NEIGHBOR[opp][s][d];
                if n < 0 {
                    break;
                }
                s = n as usize;
                let p = self.board[s];
                if p != 0 {
                    if color_of(p) == by && SLIDE_MASK[type_of(p) as usize] & (1 << d) != 0 {
                        return true;
                    }
                    break;
                }
            }
        }
        false
    }

    /// Check from the post-move occupancy without changing history, hands or hash.
    /// This accepts pseudo-legal moves; callers must still check king safety.
    pub fn gives_check(&mut self, m: Move) -> bool {
        let king = self.king[self.side ^ 1];
        if king == NO_SQ { return false; }
        let to = mv_to(m);
        let captured = self.board[to];
        let from = if mv_is_drop(m) { NO_SQ } else { mv_from(m) };
        let moving = if from == NO_SQ {
            make_piece(self.side, mv_drop_pt(m))
        } else {
            let pc = self.board[from];
            self.board[from] = 0;
            if mv_is_promo(m) { pc + PROMOTE } else { pc }
        };
        self.board[to] = moving;
        let result = self.attacked(king, self.side);
        self.board[to] = captured;
        if from != NO_SQ {
            self.board[from] = if mv_is_promo(m) { moving - PROMOTE } else { moving };
        }
        result
    }

    /// 実験V11：駒を取らない手が自玉を王手にさらさないかを、盤を一時的に書き換えて調べる（`do_move_legal` と同じ判定）。
    /// 王手をかけられていない局面の、王手にならない手だけに使う（打ち歩詰めは王手の手なので、ここでは起きない）
    pub fn is_legal_quiet(&mut self, m: Move) -> bool {
        if mv_is_drop(m) { return true; }
        let us = self.side;
        let from = mv_from(m);
        let to = mv_to(m);
        let pc = self.board[from];
        let moving = if mv_is_promo(m) { pc + PROMOTE } else { pc };
        self.board[from] = 0;
        self.board[to] = moving;
        let k = if from == self.king[us] { to } else { self.king[us] };
        let ok = k == NO_SQ || !self.attacked(k, us ^ 1);
        self.board[to] = 0;
        self.board[from] = pc;
        ok
    }

    /// 実験V10：手 m が `gen_moves` の作る疑似合法手に含まれるかを、手を作らずに調べる（置換表の手の確かめ用）
    pub fn is_pseudo_legal(&self, m: Move) -> bool {
        if m == MOVE_NONE || m >> 15 != 0 {
            return false;
        }
        let us = self.side;
        let to = mv_to(m);
        if to >= 81 || (self.board[to] != 0 && color_of(self.board[to]) == us) {
            return false;
        }
        if mv_is_drop(m) {
            let pt = mv_drop_pt(m);
            if mv_is_promo(m) || !(PAWN..=GOLD).contains(&pt) || self.hand[us][pt as usize] == 0 || self.board[to] != 0 {
                return false;
            }
            let rr = rel_rank(us, to % 9);
            return match pt {
                PAWN => {
                    let own_pawn = make_piece(us, PAWN);
                    rr != 0 && !(0..9).any(|r| self.board[(to / 9) * 9 + r] == own_pawn)
                }
                LANCE => rr != 0,
                KNIGHT => rr > 1,
                _ => true,
            };
        }
        let from = mv_from(m);
        let p = self.board[from];
        if p == 0 || color_of(p) != us {
            return false;
        }
        let pt = type_of(p);
        let mut reach = false;
        let mut mask = STEP_MASK[pt as usize];
        while mask != 0 && !reach {
            let d = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            reach = NEIGHBOR[us][from][d] == to as i8;
        }
        let mut mask = SLIDE_MASK[pt as usize];
        while mask != 0 && !reach {
            let d = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            let mut sq = from;
            loop {
                let n = NEIGHBOR[us][sq][d];
                if n < 0 {
                    break;
                }
                if n == to as i8 {
                    reach = true;
                    break;
                }
                if self.board[n as usize] != 0 {
                    break;
                }
                sq = n as usize;
            }
        }
        if !reach {
            return false;
        }
        // 成り・不成は push_moves と同じ決まり
        let zone = can_promote(pt) && (rel_rank(us, from % 9) <= 2 || rel_rank(us, to % 9) <= 2);
        if mv_is_promo(m) {
            return zone;
        }
        if zone {
            let rr = rel_rank(us, to % 9);
            return match pt {
                PAWN | LANCE => rr != 0,
                KNIGHT => rr > 1,
                _ => true,
            };
        }
        true
    }

    /// Exact one-ply checkmate, including quiet checks and legal drops.
    /// The input position is restored on every path; pawn-drop mate is rejected.
    /// 実験 速い1手詰め（10/31）：答え（返す手も）は `mate_in_one_slow` と同じ。
    /// 駒打ちは玉に利く升だけを調べ、逃げ道は手を作らずに `has_evasion` で調べる。
    pub fn mate_in_one(&mut self) -> Option<Move> {
        let us = self.side;
        let them = us ^ 1;
        let ek = self.king[them];
        if ek == NO_SQ {
            return None;
        }
        // 王手になりうるのは、移動元か移動先が、相手の玉から見て縦・横・斜めの線上か桂の位置にある手だけ
        let mut lines = 0u128;
        for d in 0..10 {
            let mut s = ek;
            loop {
                let n = NEIGHBOR[them][s][d];
                if n < 0 {
                    break;
                }
                s = n as usize;
                lines |= 1 << s;
                if d >= 8 {
                    break;
                }
            }
        }
        let mut moves = MoveList::new();
        self.gen_board_moves(&mut moves, false);
        for i in 0..moves.len {
            let m = moves.moves[i];
            if lines & (1 << mv_from(m) | 1 << mv_to(m)) == 0 { continue; }
            if !self.gives_check(m) || !self.do_move_legal(m) { continue; }
            let escape = self.has_evasion();
            self.undo_move();
            if !escape { return Some(m); }
        }
        // 歩を打つ王手は、詰みなら打ち歩詰めで指せないので調べない（`mate_in_one_slow` でも返らない）
        for pt in LANCE..=GOLD {
            if self.hand[us][pt as usize] == 0 {
                continue;
            }
            // pt を打つと相手の玉に利く空き升
            let mut mask = 0u128;
            let mut steps = STEP_MASK[pt as usize];
            while steps != 0 {
                let d = steps.trailing_zeros() as usize;
                steps &= steps - 1;
                let s = NEIGHBOR[them][ek][d];
                if s >= 0 && self.board[s as usize] == 0 {
                    mask |= 1 << s;
                }
            }
            let mut slides = SLIDE_MASK[pt as usize];
            while slides != 0 {
                let d = slides.trailing_zeros() as usize;
                slides &= slides - 1;
                let mut s = ek;
                loop {
                    let n = NEIGHBOR[them][s][d];
                    if n < 0 || self.board[n as usize] != 0 {
                        break;
                    }
                    s = n as usize;
                    mask |= 1 << s;
                }
            }
            while mask != 0 {
                let to = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                let rr = rel_rank(us, to % 9);
                let ok = match pt {
                    LANCE => rr != 0,
                    KNIGHT => rr > 1,
                    _ => true,
                };
                let m = make_drop(pt, to);
                if !ok || !self.do_move_legal(m) { continue; }
                let escape = self.has_evasion();
                self.undo_move();
                if !escape { return Some(m); }
            }
        }
        None
    }

    /// 手番 by の駒のうち、升 sq に利いている駒の升（方向ごとに最大1つなので最大10）
    fn attackers_of(&self, sq: usize, by: usize, out: &mut [usize; 10]) -> usize {
        let opp = by ^ 1;
        let mut n = 0;
        for d in 0..10 {
            let a = NEIGHBOR[opp][sq][d];
            if a < 0 {
                continue;
            }
            let a = a as usize;
            let p = self.board[a];
            if p != 0 {
                if color_of(p) == by {
                    let pt = type_of(p) as usize;
                    if (STEP_MASK[pt] | SLIDE_MASK[pt]) & (1 << d) != 0 {
                        out[n] = a;
                        n += 1;
                    }
                }
                continue;
            }
            if d >= 8 {
                continue;
            }
            let mut s = a;
            loop {
                let next = NEIGHBOR[opp][s][d];
                if next < 0 {
                    break;
                }
                s = next as usize;
                let p = self.board[s];
                if p != 0 {
                    if color_of(p) == by && SLIDE_MASK[type_of(p) as usize] & (1 << d) != 0 {
                        out[n] = s;
                        n += 1;
                    }
                    break;
                }
            }
        }
        n
    }

    /// 手番側の駒を from から to へ動かしても、自玉が王手にならないか（盤を一時的に書き換えて調べる）
    fn move_keeps_king_safe(&mut self, from: usize, to: usize) -> bool {
        let us = self.side;
        let pc = self.board[from];
        let cap = self.board[to];
        self.board[from] = 0;
        self.board[to] = pc;
        let k = if from == self.king[us] { to } else { self.king[us] };
        let ok = !self.attacked(k, us ^ 1);
        self.board[to] = cap;
        self.board[from] = pc;
        ok
    }

    /// 王手をかけられている手番側に、合法手があるか（手を作らずに調べる。`has_legal_move` と違い打ち歩詰めも見る）
    fn has_evasion(&mut self) -> bool {
        let us = self.side;
        let them = us ^ 1;
        let k = self.king[us];
        // 玉が動く（取る手も含む）
        let mut steps = STEP_MASK[KING as usize];
        while steps != 0 {
            let d = steps.trailing_zeros() as usize;
            steps &= steps - 1;
            let to = NEIGHBOR[us][k][d];
            if to < 0 {
                continue;
            }
            let to = to as usize;
            let t = self.board[to];
            if t != 0 && color_of(t) == us {
                continue;
            }
            if self.move_keeps_king_safe(k, to) {
                return true;
            }
        }
        let mut checkers = [0usize; 10];
        if self.attackers_of(k, them, &mut checkers) != 1 {
            return false;
        }
        let c = checkers[0];
        // 王手をかけた駒を、玉以外の駒で取る（成り・不成のどちらかは必ず指せる）
        let mut att = [0usize; 10];
        let na = self.attackers_of(c, us, &mut att);
        for &a in &att[..na] {
            if a != k && self.move_keeps_king_safe(a, c) {
                return true;
            }
        }
        // 飛び道具の王手なら、間の升に打つ・動く
        let mut between = [0usize; 8];
        let mut nb = 0;
        for d in 0..8 {
            let mut s = k;
            nb = 0;
            loop {
                let n = NEIGHBOR[us][s][d];
                if n < 0 {
                    break;
                }
                s = n as usize;
                if self.board[s] != 0 {
                    break;
                }
                between[nb] = s;
                nb += 1;
            }
            if s == c {
                break;
            }
            nb = 0;
        }
        if nb == 0 {
            return false;
        }
        let mut pawn_files = 0u16;
        if self.hand[us][PAWN as usize] > 0 {
            let own_pawn = make_piece(us, PAWN);
            for f in 0..9 {
                if (0..9).any(|r| self.board[f * 9 + r] == own_pawn) {
                    pawn_files |= 1 << f;
                }
            }
        }
        for &s in &between[..nb] {
            let rr = rel_rank(us, s % 9);
            for pt in PAWN..=GOLD {
                if self.hand[us][pt as usize] == 0 {
                    continue;
                }
                let ok = match pt {
                    PAWN => rr != 0 && pawn_files & (1 << (s / 9)) == 0,
                    LANCE => rr != 0,
                    KNIGHT => rr > 1,
                    _ => true,
                };
                if !ok {
                    continue;
                }
                // 相手の玉の前に歩を打つ手は、打ち歩詰めになりうるので指して確かめる
                if pt == PAWN && NEIGHBOR[us][s][0] == self.king[them] as i8 {
                    if self.do_move_legal(make_drop(PAWN, s)) {
                        self.undo_move();
                        return true;
                    }
                    continue;
                }
                return true;
            }
            let na = self.attackers_of(s, us, &mut att);
            for &a in &att[..na] {
                if a != k && self.move_keeps_king_safe(a, s) {
                    return true;
                }
            }
        }
        false
    }

    /// 前の1手詰めの判定（すべての手を作り、王手のたびに相手の手もすべて指して確かめる）。テストで答えを比べる用
    #[cfg(test)]
    pub fn mate_in_one_slow(&mut self) -> Option<Move> {
        let mut moves = MoveList::new();
        self.gen_moves(&mut moves, false);
        for i in 0..moves.len {
            let m = moves.moves[i];
            if !self.gives_check(m) || !self.do_move_legal(m) { continue; }
            let mut replies = MoveList::new();
            self.gen_moves(&mut replies, false);
            let mut escape = false;
            for j in 0..replies.len {
                if self.do_move_legal(replies.moves[j]) {
                    self.undo_move();
                    escape = true;
                    break;
                }
            }
            self.undo_move();
            if !escape { return Some(m); }
        }
        None
    }

    /// 直前の指し手（なければ MOVE_NONE）
    #[inline]
    pub fn last_move(&self) -> Move {
        self.history.last().map(|s| s.mv).unwrap_or(MOVE_NONE)
    }

    /// 静的交換評価（SEE）: 升 to での駒の取り合いを最後まで続けたときの、指す側の損得
    pub fn see(&self, m: Move) -> i32 { self.see_impl(m, false) }

    /// Shogi exchange values include the captured piece returning to hand unpromoted.
    pub fn see_with_hand(&self, m: Move) -> i32 { self.see_impl(m, true) }

    fn see_impl(&self, m: Move, with_hand: bool) -> i32 {
        const BOARD: [i32; 15] = [0, 90, 315, 405, 495, 855, 990, 540, 20000, 540, 540, 540, 540, 945, 1395];
        let values = if with_hand { &CAPTURE_VALUE } else { &BOARD };
        let to = mv_to(m);
        let mut board = self.board;
        let mut gain = [0i32; 40];
        let moving;
        if mv_is_drop(m) {
            moving = make_piece(self.side, mv_drop_pt(m));
            gain[0] = 0;
        } else {
            let p = board[mv_from(m)];
            moving = if mv_is_promo(m) { p + PROMOTE } else { p };
            gain[0] = values[type_of(board[to]) as usize] + (values[type_of(moving) as usize] - values[type_of(p) as usize]);
            board[mv_from(m)] = 0;
        }
        board[to] = moving;
        let mut on_square = values[type_of(moving) as usize];
        let mut side = self.side ^ 1;
        let mut d = 0;
        while d < 38 {
            let Some((sq, p)) = least_attacker(&board, to, side, with_hand) else { break };
            d += 1;
            gain[d] = on_square - gain[d - 1];
            // どちらを選んでも損なら打ち切ってよい
            if (-gain[d - 1]).max(gain[d]) < 0 {
                break;
            }
            on_square = values[type_of(p) as usize];
            board[sq] = 0;
            board[to] = p;
            side ^= 1;
        }
        while d > 0 {
            gain[d - 1] = -((-gain[d - 1]).max(gain[d]));
            d -= 1;
        }
        gain[0]
    }

    #[inline]
    pub fn in_check(&self) -> bool {
        let k = self.king[self.side];
        k != NO_SQ && self.attacked(k, self.side ^ 1)
    }

    #[inline]
    fn push_moves(list: &mut MoveList, us: usize, pt: u8, from: usize, to: usize) {
        if can_promote(pt) && (rel_rank(us, from % 9) <= 2 || rel_rank(us, to % 9) <= 2) {
            list.push(make_move(from, to, true));
            let rr = rel_rank(us, to % 9);
            let must = match pt {
                PAWN | LANCE => rr == 0,
                KNIGHT => rr <= 1,
                _ => false,
            };
            if !must {
                list.push(make_move(from, to, false));
            }
        } else {
            list.push(make_move(from, to, false));
        }
    }

    /// 疑似合法手を生成する（自玉が取られる手も含む）。
    /// captures_only の場合は駒を取る手だけ。
    pub fn gen_moves(&self, list: &mut MoveList, captures_only: bool) {
        self.gen_board_moves(list, captures_only);
        if captures_only {
            return;
        }
        self.gen_drops(list);
    }

    /// 盤上の駒を動かす疑似合法手（`gen_moves` の前半。順番も同じ）
    #[inline(always)]
    fn gen_board_moves(&self, list: &mut MoveList, captures_only: bool) {
        let us = self.side;
        for from in 0..81 {
            let p = self.board[from];
            if p == 0 || color_of(p) != us {
                continue;
            }
            let pt = type_of(p);

            let mut mask = STEP_MASK[pt as usize];
            while mask != 0 {
                let d = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                let to = NEIGHBOR[us][from][d];
                if to < 0 {
                    continue;
                }
                let to = to as usize;
                let t = self.board[to];
                if t != 0 {
                    if color_of(t) == us {
                        continue;
                    }
                } else if captures_only {
                    continue;
                }
                Self::push_moves(list, us, pt, from, to);
            }

            let mut mask = SLIDE_MASK[pt as usize];
            while mask != 0 {
                let d = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                let mut sq = from;
                loop {
                    let to = NEIGHBOR[us][sq][d];
                    if to < 0 {
                        break;
                    }
                    let to = to as usize;
                    let t = self.board[to];
                    if t != 0 {
                        if color_of(t) != us {
                            Self::push_moves(list, us, pt, from, to);
                        }
                        break;
                    }
                    if !captures_only {
                        Self::push_moves(list, us, pt, from, to);
                    }
                    sq = to;
                }
            }
        }
    }

    /// 駒打ちの疑似合法手（`gen_moves` の後半。順番も同じ）
    #[inline(always)]
    fn gen_drops(&self, list: &mut MoveList) {
        let us = self.side;
        let mut pawn_files = 0u16;
        if self.hand[us][PAWN as usize] > 0 {
            let own_pawn = make_piece(us, PAWN);
            for f in 0..9 {
                if (0..9).any(|r| self.board[f * 9 + r] == own_pawn) {
                    pawn_files |= 1 << f;
                }
            }
        }
        for pt in PAWN..=GOLD {
            if self.hand[us][pt as usize] == 0 {
                continue;
            }
            for to in 0..81 {
                if self.board[to] != 0 {
                    continue;
                }
                let rr = rel_rank(us, to % 9);
                let ok = match pt {
                    PAWN => rr != 0 && pawn_files & (1 << (to / 9)) == 0,
                    LANCE => rr != 0,
                    KNIGHT => rr > 1,
                    _ => true,
                };
                if ok {
                    list.push(make_drop(pt, to));
                }
            }
        }
    }

    /// 疑似合法手を指し、合法なら true（指したまま）、非合法なら元に戻して false。
    /// 王手放置と打ち歩詰めを判定する。
    pub fn do_move_legal(&mut self, m: Move) -> bool {
        let us = self.side;
        self.do_move(m);
        let k = self.king[us];
        if k != NO_SQ && self.attacked(k, us ^ 1) {
            self.undo_move();
            return false;
        }
        if mv_is_drop(m) && mv_drop_pt(m) == PAWN {
            let ek = self.king[us ^ 1];
            if ek != NO_SQ && NEIGHBOR[us][mv_to(m)][0] == ek as i8 && !self.has_legal_move() {
                self.undo_move();
                return false;
            }
        }
        true
    }

    /// 手番側に合法手があるか（打ち歩詰めの判定用。ここでは打ち歩詰め自体は無視する）
    fn has_legal_move(&mut self) -> bool {
        let mut list = MoveList::new();
        self.gen_moves(&mut list, false);
        let us = self.side;
        for i in 0..list.len {
            self.do_move(list.moves[i]);
            let k = self.king[us];
            let ok = k == NO_SQ || !self.attacked(k, us ^ 1);
            self.undo_move();
            if ok {
                return true;
            }
        }
        false
    }

    pub fn legal_moves(&mut self) -> Vec<Move> {
        let mut list = MoveList::new();
        self.gen_moves(&mut list, false);
        let mut out = Vec::new();
        for i in 0..list.len {
            let m = list.moves[i];
            if self.do_move_legal(m) {
                self.undo_move();
                out.push(m);
            }
        }
        out
    }

    /// 探索中の同一局面の出現（千日手）を検出する。
    pub fn is_repetition(&self) -> bool {
        let n = self.history.len();
        let mut j = n;
        let mut count = 0;
        while j > 0 && count < 256 {
            j -= 1;
            count += 1;
            let st = &self.history[j];
            if st.mv == MOVE_NONE {
                break;
            }
            if (n - j) % 2 == 0 && st.key == self.key {
                return true;
            }
        }
        false
    }

    /// 手番側が入玉宣言（CSA ルール、27点法）の条件を満たすか（持ち時間の条件は GUI・サーバーが扱う）。
    /// 玉が敵陣にいる、敵陣に玉以外の自分の駒が 10 枚以上、敵陣の駒と持ち駒の点数（飛・角と成り駒は 5点、ほかは 1点）が
    /// 先手 28点以上・後手 27点以上、王手をかけられていない。玉の位置を先に調べるので、ふだんはすぐ false を返す。
    pub fn can_declare(&self) -> bool {
        let us = self.side;
        let k = self.king[us];
        if k == NO_SQ || rel_rank(us, k % 9) > 2 {
            return false;
        }
        let mut count = 0;
        let mut points = 0;
        for sq in 0..81 {
            let p = self.board[sq];
            if p == 0 || color_of(p) != us || rel_rank(us, sq % 9) > 2 {
                continue;
            }
            let pt = type_of(p);
            if pt == KING {
                continue;
            }
            count += 1;
            points += if matches!(unpromote(pt), BISHOP | ROOK) { 5 } else { 1 };
        }
        if count < 10 {
            return false;
        }
        for pt in PAWN..=GOLD {
            let n = self.hand[us][pt as usize] as i32;
            points += if pt == BISHOP || pt == ROOK { 5 * n } else { n };
        }
        points >= if us == BLACK { 28 } else { 27 } && !self.in_check()
    }

    /// 同じ局面の繰り返しの結果（cargo の機能 perpetual_check で使う）。
    /// None = 繰り返しでない、Some(0) = 千日手（引き分け）、
    /// Some(1) = 手番側の勝ち（相手が王手をかけ続けた）、Some(-1) = 手番側の負け（手番側が王手をかけ続けた）。
    /// 繰り返しを見つけたときだけ、写した局面で手を戻しながら、その間の王手の有無を調べる。
    pub fn repetition_result(&self) -> Option<i32> {
        let n = self.history.len();
        let mut j = n;
        let mut count = 0;
        let mut len = None;
        while j > 0 && count < 256 {
            j -= 1;
            count += 1;
            let st = &self.history[j];
            if st.mv == MOVE_NONE {
                break;
            }
            if (n - j) % 2 == 0 && st.key == self.key {
                len = Some(n - j);
                break;
            }
        }
        let len = len?;
        let mut p = self.clone();
        let (mut us_checked, mut them_checked) = (true, true);
        for k in 0..len {
            // k が偶数: 手番側が指す局面（王手されていれば、直前の相手の手が王手）。奇数: 相手が指す局面
            if k % 2 == 0 {
                us_checked &= p.in_check();
            } else {
                them_checked &= p.in_check();
            }
            p.undo_move();
        }
        Some(if us_checked { 1 } else if them_checked { -1 } else { 0 })
    }

    pub fn parse_usi_move(&mut self, s: &str) -> Option<Move> {
        let b = s.as_bytes();
        if b.len() < 4 {
            return None;
        }
        let m = if b[1] == b'*' {
            let (_, pt) = char_to_piece(b[0] as char)?;
            make_drop(pt, parse_sq(b[2], b[3])?)
        } else {
            let from = parse_sq(b[0], b[1])?;
            let to = parse_sq(b[2], b[3])?;
            make_move(from, to, b.len() >= 5 && b[4] == b'+')
        };
        // 指定された手だけを合法性判定する。全候補を指して戻す必要はない。
        let mut list = MoveList::new();
        self.gen_moves(&mut list, false);
        if list.moves[..list.len].contains(&m) && self.do_move_legal(m) {
            self.undo_move();
            Some(m)
        } else {
            None
        }
    }

    pub fn display(&self) -> String {
        const NAMES: [&str; 15] =
            ["・", "歩", "香", "桂", "銀", "角", "飛", "金", "玉", "と", "杏", "圭", "全", "馬", "龍"];
        let hand_str = |c: usize| {
            let mut s = String::new();
            for pt in (PAWN..=GOLD).rev() {
                let n = self.hand[c][pt as usize];
                if n > 0 {
                    s += NAMES[pt as usize];
                    if n > 1 {
                        s += &n.to_string();
                    }
                }
            }
            if s.is_empty() {
                s = "なし".into();
            }
            s
        };
        let mut out = format!("後手の持駒: {}\n  ９ ８ ７ ６ ５ ４ ３ ２ １\n", hand_str(WHITE));
        for r in 0..9 {
            out += "|";
            for f in (0..9).rev() {
                let p = self.board[f * 9 + r];
                if p == 0 {
                    out += " ・";
                } else {
                    out += if color_of(p) == WHITE { "v" } else { " " };
                    out += NAMES[type_of(p) as usize];
                }
            }
            out += &format!("| {}\n", (b'a' + r as u8) as char);
        }
        out += &format!("先手の持駒: {}\n", hand_str(BLACK));
        out += &format!("手番: {}  key: {:016x}\n", if self.side == BLACK { "先手" } else { "後手" }, self.key);
        out
    }
}

/// 升 sq に利いている手番 by の駒のうち、いちばん安い駒（SEE 用）。駒の後ろの利き（X線）も盤から駒を除けば自然に反映される
fn least_attacker(board: &[u8; 81], sq: usize, by: usize, with_hand: bool) -> Option<(usize, u8)> {
    const ORDER: [i32; 15] = [0, 1, 2, 3, 4, 7, 8, 5, 20, 5, 5, 5, 5, 9, 10];
    let order = if with_hand { &CAPTURE_VALUE } else { &ORDER };
    let opp = by ^ 1;
    let mut best: Option<(usize, u8)> = None;
    let mut consider = |s: usize, p: u8, best: &mut Option<(usize, u8)>| {
        if best.map_or(true, |(_, b)| order[type_of(p) as usize] < order[type_of(b) as usize]) {
            *best = Some((s, p));
        }
    };
    for d in 0..10 {
        let a = NEIGHBOR[opp][sq][d];
        if a < 0 {
            continue;
        }
        let a = a as usize;
        let p = board[a];
        if p != 0 {
            if color_of(p) == by && (STEP_MASK[type_of(p) as usize] | SLIDE_MASK[type_of(p) as usize]) & (1 << d) != 0 {
                consider(a, p, &mut best);
            }
            continue;
        }
        if d >= 8 {
            continue;
        }
        let mut s = a;
        loop {
            let n = NEIGHBOR[opp][s][d];
            if n < 0 {
                break;
            }
            s = n as usize;
            let p = board[s];
            if p != 0 {
                if color_of(p) == by && SLIDE_MASK[type_of(p) as usize] & (1 << d) != 0 {
                    consider(s, p, &mut best);
                }
                break;
            }
        }
    }
    best
}

pub fn perft(pos: &mut Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut list = MoveList::new();
    pos.gen_moves(&mut list, false);
    let mut n = 0;
    for i in 0..list.len {
        if pos.do_move_legal(list.moves[i]) {
            n += perft(pos, depth - 1);
            pos.undo_move();
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_usi_validation_matches_all_legal_moves() {
        let mut pos = Position::startpos();
        let mut rng = 987654321u64;
        let (mut captures, mut drops, mut promotions, mut checks) = (0, 0, 0, 0);
        for _ in 0..1500 {
            let legal = pos.legal_moves();
            let mut candidates = MoveList::new();
            pos.gen_moves(&mut candidates, false);
            let before = (pos.board, pos.hand, pos.king, pos.side, pos.key, pos.history.len());
            if pos.in_check() { checks += 1; }
            for &m in &candidates.moves[..candidates.len] {
                assert_eq!(pos.parse_usi_move(&move_to_usi(m)), legal.contains(&m).then_some(m));
                assert_eq!(before, (pos.board, pos.hand, pos.king, pos.side, pos.key, pos.history.len()));
            }
            assert!(pos.parse_usi_move("5e5e").is_none());
            assert_eq!(before, (pos.board, pos.hand, pos.king, pos.side, pos.key, pos.history.len()));
            if legal.is_empty() {
                pos = Position::startpos();
            } else {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let m = legal[(rng % legal.len() as u64) as usize];
                if mv_is_drop(m) { drops += 1; }
                else if pos.board[mv_to(m)] != 0 { captures += 1; }
                if mv_is_promo(m) { promotions += 1; }
                pos.do_move(m);
            }
        }
        assert!(captures > 20 && drops > 20 && promotions > 10 && checks > 10);
    }

    #[test]
    fn pseudo_legal_check_matches_generated_moves() {
        let mut pos = Position::startpos();
        let mut rng = 192837465u64;
        for _ in 0..800 {
            let mut list = MoveList::new();
            pos.gen_moves(&mut list, false);
            let gen = &list.moves[..list.len];
            // 盤上の手はすべての移動元・移動先・成りの組み合わせ、駒打ちはすべての駒種・升（駒種 0 と玉も含む）
            for from in 0..81 {
                for to in 0..81 {
                    for promo in [false, true] {
                        let m = make_move(from, to, promo);
                        if m == MOVE_NONE { continue; }
                        assert_eq!(pos.is_pseudo_legal(m), gen.contains(&m), "{}", move_to_usi(m));
                    }
                }
            }
            for pt in 0..=KING {
                for to in 0..81 {
                    let m = make_drop(pt, to);
                    assert_eq!(pos.is_pseudo_legal(m), gen.contains(&m));
                    assert!(!pos.is_pseudo_legal(m | 1 << 14));
                }
            }
            assert!(!pos.is_pseudo_legal(MOVE_NONE));
            let legal = pos.legal_moves();
            if legal.is_empty() {
                pos = Position::startpos();
            } else {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                pos.do_move(legal[(rng % legal.len() as u64) as usize]);
            }
        }
    }
}

#[cfg(test)]
mod declare_tests {
    use super::*;

    #[test]
    fn black_needs_28_points_and_10_pieces() {
        // 敵陣に玉以外 10 枚（竜・馬 各5点、金銀成桂成香 7 枚と歩 1 枚）＝ 18点 ＋ 持ち駒の歩
        let ok = Position::from_sfen("+R+BGGSS+N+N+L/4K3P/9/9/9/9/9/9/4k4 b 10P 1").unwrap();
        assert!(ok.can_declare()); // 28点
        let short = Position::from_sfen("+R+BGGSS+N+N+L/4K3P/9/9/9/9/9/9/4k4 b 9P 1").unwrap();
        assert!(!short.can_declare()); // 27点（先手は 28点が要る）
        let nine = Position::from_sfen("+R+BGGSS+N+N+L/4K4/9/9/9/9/9/9/4k4 b 11P 1").unwrap();
        assert!(!nine.can_declare()); // 点数は足りるが、敵陣の駒が 9 枚
        let checked = Position::from_sfen("+R+BGGSS+N+N+L/4K3P/4g4/9/9/9/9/9/4k4 b 10P 1").unwrap();
        assert!(!checked.can_declare()); // 王手をかけられている
        let third = Position::from_sfen("+R+BGGSS+N+N+L/8P/4K4/9/9/9/9/9/4k4 b 10P 1").unwrap();
        assert!(third.can_declare()); // 3段目も敵陣
        let not_in_camp = Position::from_sfen("+R+BGGSS+N+N+L/8P/9/4K4/9/9/9/9/4k4 b 10P 1").unwrap();
        assert!(!not_in_camp.can_declare()); // 玉が敵陣の外
    }

    #[test]
    fn white_needs_27_points() {
        // 敵陣に 10 枚で 18点 ＋ 持ち駒の歩
        let ok = Position::from_sfen("4K4/9/9/9/9/9/9/p3k4/+l+n+nssgg+b+r w 9p 1").unwrap();
        assert!(ok.can_declare()); // 27点
        let short = Position::from_sfen("4K4/9/9/9/9/9/9/p3k4/+l+n+nssgg+b+r w 8p 1").unwrap();
        assert!(!short.can_declare());
    }
}

#[cfg(test)]
mod tactical_tests {
    use super::*;
    #[test]
    fn checking_move_matches_play_and_restores_position() {
        let mut p = Position::startpos();
        let mut rng = 123456789u64;
        let mut drops = 0;
        let mut promotions = 0;
        for _ in 0..300 {
            let moves = p.legal_moves();
            let sfen = (p.board, p.hand, p.side, p.king, p.key);
            let hash = p.key;
            let hands = p.hand;
            for &m in &moves {
                let check = p.gives_check(m);
                assert_eq!((p.board, p.hand, p.side, p.king, p.key), sfen);
                assert_eq!(p.key, hash);
                assert_eq!(p.hand, hands);
                p.do_move(m);
                assert_eq!(check, p.in_check());
                p.undo_move();
                if mv_is_drop(m) { drops += 1; }
                if mv_is_promo(m) { promotions += 1; }
            }
            if moves.is_empty() { p = Position::startpos(); continue; }
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            p.do_move(moves[rng as usize % moves.len()]);
        }
        assert!(drops > 10 && promotions > 10);
    }
    #[test]
    fn mate_in_one_checks_all_replies_and_rejects_pawn_drop_mate() {
        let mut p = Position::from_sfen("3pkp3/9/4G4/9/9/9/9/9/4K4 b R 1").unwrap();
        let original = (p.board, p.hand, p.side, p.king, p.key);
        let key = p.key;
        let m = p.mate_in_one().expect("rook mate");
        assert_eq!((p.board, p.hand, p.side, p.king, p.key), original);
        assert_eq!(p.key, key);
        assert!(p.do_move_legal(m));
        assert!(p.in_check());
        assert!(p.legal_moves().is_empty());
        p.undo_move();
        let mut pawn = Position::from_sfen("3pkp3/9/4G4/9/9/9/9/9/4K4 b P 1").unwrap();
        assert!(pawn.parse_usi_move("P*5b").is_none());
        assert!(pawn.mate_in_one().is_none());
        let mut escape = Position::from_sfen("4k4/9/9/9/9/9/9/9/4K4 b R 1").unwrap();
        assert!(escape.mate_in_one().is_none());
    }

    #[test]
    fn fast_mate_in_one_matches_slow_version() {
        // ランダムに進めた局面と、その局面から王手を1つ指した後の局面（逃げ道を調べる側の当たりが増える）で比べる
        let mut pos = Position::startpos();
        let mut rng = 987654321u64;
        let mut next = |n: usize| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng % n as u64) as usize
        };
        let (mut mates, mut checked) = (0, 0);
        for _ in 0..20000 {
            let original = (pos.board, pos.hand, pos.side, pos.king, pos.key);
            let fast = pos.mate_in_one();
            assert_eq!((pos.board, pos.hand, pos.side, pos.king, pos.key), original);
            assert_eq!(fast, pos.mate_in_one_slow(), "{}", pos.display());
            checked += 1;
            if fast.is_some() { mates += 1; }
            let legal = pos.legal_moves();
            if legal.is_empty() || pos.history.len() > 300 {
                pos = Position::startpos();
                continue;
            }
            // 駒の取り合いが進んで持ち駒が増えるよう、取る手を選びやすくする
            let caps: Vec<Move> = legal.iter().copied().filter(|&m| !mv_is_drop(m) && pos.board[mv_to(m)] != 0).collect();
            let m = if !caps.is_empty() && next(3) != 0 { caps[next(caps.len())] } else { legal[next(legal.len())] };
            pos.do_move(m);
        }
        assert!(checked == 20000 && mates > 200, "mates {}", mates);
    }
}

#[cfg(test)]
mod exchange_tests {
    use super::*;
    #[test]
    fn captures_account_for_demoted_hand_value() {
        for pt in PAWN..=DRAGON {
            if pt == KING { continue; }
            assert_eq!(CAPTURE_VALUE[pt as usize],
                crate::eval::PIECE_VALUE[pt as usize] +
                crate::eval::HAND_VALUE[unpromote(pt) as usize]);
        }
        let mut p = Position::from_sfen("k8/5s3/4g4/4+P4/9/9/9/9/8K b - 1").unwrap();
        let m = p.parse_usi_move("5d5c").unwrap();
        assert_eq!(p.see(m), 0);
        assert_eq!(p.see_with_hand(m), 500);
        let mut p = Position::from_sfen("k8/5s3/4+p4/4G4/9/9/9/9/8K b - 1").unwrap();
        let m = p.parse_usi_move("5d5c").unwrap();
        assert_eq!(p.see(m), 0);
        assert_eq!(p.see_with_hand(m), -500);
    }
}

