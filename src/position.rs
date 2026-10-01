//! 局面の表現・SFEN の読み込み・指し手生成・指し手の実行と取り消し。

use crate::types::*;

pub const START_SFEN: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
pub const NO_SQ: usize = 81;
pub const MAX_MOVES: usize = 1024;

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

    /// 直前の指し手（なければ MOVE_NONE）
    #[inline]
    pub fn last_move(&self) -> Move {
        self.history.last().map(|s| s.mv).unwrap_or(MOVE_NONE)
    }

    /// 静的交換評価（SEE）: 升 to での駒の取り合いを最後まで続けたときの、指す側の損得
    pub fn see(&self, m: Move) -> i32 {
        const V: [i32; 15] = [0, 90, 315, 405, 495, 855, 990, 540, 20000, 540, 540, 540, 540, 945, 1395];
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
            gain[0] = V[type_of(board[to]) as usize] + (V[type_of(moving) as usize] - V[type_of(p) as usize]);
            board[mv_from(m)] = 0;
        }
        board[to] = moving;
        let mut on_square = V[type_of(moving) as usize];
        let mut side = self.side ^ 1;
        let mut d = 0;
        while d < 38 {
            let Some((sq, p)) = least_attacker(&board, to, side) else { break };
            d += 1;
            gain[d] = on_square - gain[d - 1];
            // どちらを選んでも損なら打ち切ってよい
            if (-gain[d - 1]).max(gain[d]) < 0 {
                break;
            }
            on_square = V[type_of(p) as usize];
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

        if captures_only {
            return;
        }

        // 駒打ち
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
        if self.legal_moves().contains(&m) {
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
fn least_attacker(board: &[u8; 81], sq: usize, by: usize) -> Option<(usize, u8)> {
    const ORDER: [i32; 15] = [0, 1, 2, 3, 4, 7, 8, 5, 20, 5, 5, 5, 5, 9, 10];
    let opp = by ^ 1;
    let mut best: Option<(usize, u8)> = None;
    let mut consider = |s: usize, p: u8, best: &mut Option<(usize, u8)>| {
        if best.map_or(true, |(_, b)| ORDER[type_of(p) as usize] < ORDER[type_of(b) as usize]) {
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
