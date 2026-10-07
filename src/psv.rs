//! やねうら王形式の教師局面（PackedSfenValue、1局面40バイト）の読み込み。
//!
//! 40バイト = 圧縮した局面 32 + 評価値 i16 + 指し手 u16 + 手数 u16 + 勝敗 i8 + 詰め物 1（リトルエンディアン）。
//! 評価値・勝敗は手番側から見た値（勝敗: 1 勝ち、0 引き分け、-1 負け）。
//! 圧縮した局面（256ビット、各バイトの下位ビットから読む）:
//!   手番 1ビット、先手玉・後手玉の升 各7ビット、
//!   玉以外の升を 0..80 の順に、ハフマン符号の駒（空き升は "0"）＋成りフラグ（金以外）＋手番、
//!   残りは持ち駒（盤上の符号の先頭1ビットを除いたもの）＋成りフラグ（金以外、常に0）＋手番。
//! 升の番号・駒種の番号は Sailfish とやねうら王で同じ（sq = (筋-1)*9 + 段）。

use crate::position::Position;
use crate::types::*;

pub const RECORD_BYTES: usize = 40;

/// 盤上の駒のハフマン符号（符号, ビット数）。添字は駒種（0 は空き升）。
const HUFFMAN: [(u32, u32); 8] = [(0x00, 1), (0x01, 2), (0x03, 4), (0x0b, 4), (0x07, 4), (0x1f, 6), (0x3f, 6), (0x0f, 5)];

pub struct PsvRecord {
    pub pos: Position,
    pub score: i16,
    pub move16: u16,
    pub ply: u16,
    pub result: i8,
}

struct Bits<'a> { data: &'a [u8], cursor: usize }
impl Bits<'_> {
    fn one(&mut self) -> Result<u32, String> {
        if self.cursor >= 256 { return Err("圧縮した局面のビットが足りない".into()); }
        let b = (self.data[self.cursor / 8] >> (self.cursor & 7)) & 1;
        self.cursor += 1;
        Ok(b as u32)
    }
    fn n(&mut self, n: u32) -> Result<u32, String> {
        let mut v = 0;
        for i in 0..n { v |= self.one()? << i; }
        Ok(v)
    }
}

/// 32バイトの圧縮した局面を、盤・持ち駒・手番に戻す
pub fn unpack_sfen(data: &[u8]) -> Result<Position, String> {
    let mut s = Bits { data, cursor: 0 };
    let side = s.one()? as usize;
    let mut board = [0u8; 81];
    let mut hand = [[0u8; 8]; 2];
    for c in 0..2 {
        let sq = s.n(7)? as usize;
        if sq >= 81 { return Err(format!("玉の升が不正: {}", sq)); }
        board[sq] = make_piece(c, KING);
    }
    for sq in 0..81 {
        if board[sq] != 0 && type_of(board[sq]) == KING { continue; }
        let (mut code, mut bits) = (0u32, 0u32);
        let pt = loop {
            code |= s.one()? << bits;
            bits += 1;
            if let Some(pt) = (0..8).find(|&pt| HUFFMAN[pt] == (code, bits)) { break pt as u8; }
            if bits > 6 { return Err("盤上の駒の符号が不正".into()); }
        };
        if pt == 0 { continue; }
        let promote = if pt == GOLD { false } else { s.one()? == 1 };
        let c = s.one()? as usize;
        board[sq] = make_piece(c, if promote { pt + PROMOTE } else { pt });
    }
    while s.cursor < 256 {
        let (mut code, mut bits) = (0u32, 0u32);
        let pt = loop {
            code |= s.one()? << bits;
            bits += 1;
            if let Some(pt) = (1..8).find(|&pt| (HUFFMAN[pt].0 >> 1, HUFFMAN[pt].1 - 1) == (code, bits)) { break pt as u8; }
            if bits > 5 { return Err("持ち駒の符号が不正".into()); }
        };
        if pt != GOLD { s.one()?; } // 成りフラグ（持ち駒では使わない）
        let c = s.one()? as usize;
        hand[c][pt as usize] += 1;
    }
    Ok(Position::from_parts(board, hand, side))
}

/// 40バイトの教師局面を読む
pub fn decode(rec: &[u8]) -> Result<PsvRecord, String> {
    if rec.len() != RECORD_BYTES { return Err("教師局面の長さが40バイトではない".into()); }
    let pos = unpack_sfen(&rec[..32])?;
    let le16 = |i: usize| u16::from_le_bytes([rec[i], rec[i + 1]]);
    Ok(PsvRecord { pos, score: le16(32) as i16, move16: le16(34), ply: le16(36), result: rec[38] as i8 })
}

/// cshogi の Board.sfen() と同じ書き方の、盤・手番・持ち駒の部分（手数は含めない）。
/// mirror = true なら盤を左右反転して書く（tools/sample-mekitto.py の mirror_key と同じ）。
pub fn sfen_key(pos: &Position, mirror: bool) -> String {
    const CH: [u8; 9] = [0, b'P', b'L', b'N', b'S', b'B', b'R', b'G', b'K'];
    let mut s = String::with_capacity(100);
    for rank in 0..9 {
        let mut empty = 0u8;
        for f in (0..9).rev() {
            let file = if mirror { 8 - f } else { f };
            let p = pos.board[file * 9 + rank];
            if p == 0 { empty += 1; continue; }
            if empty > 0 { s.push((b'0' + empty) as char); empty = 0; }
            let pt = type_of(p);
            if pt > KING { s.push('+'); }
            let ch = CH[if pt > KING { (pt - PROMOTE) as usize } else { pt as usize }];
            s.push(if color_of(p) == BLACK { ch } else { ch.to_ascii_lowercase() } as char);
        }
        if empty > 0 { s.push((b'0' + empty) as char); }
        if rank < 8 { s.push('/'); }
    }
    s.push_str(if pos.side == BLACK { " b " } else { " w " });
    let mut any = false;
    for c in 0..2 {
        for pt in [ROOK, BISHOP, GOLD, SILVER, KNIGHT, LANCE, PAWN] {
            let n = pos.hand[c][pt as usize];
            if n == 0 { continue; }
            if n > 1 { s.push_str(&n.to_string()); }
            let ch = CH[pt as usize];
            s.push(if c == BLACK { ch } else { ch.to_ascii_lowercase() } as char);
            any = true;
        }
    }
    if !any { s.push('-'); }
    s
}

/// 左右反転を同一視した局面の鍵と、検証用かどうか（tools/convert-mekitto-tsv.py と同じ：
/// min(SFEN, 反転SFEN) の SHA-256 の先頭8バイトを鍵、次の8バイトを 50 で割った余りが 0 なら検証用）
pub fn canonical_hash(pos: &Position) -> (u64, bool) {
    let (a, b) = (sfen_key(pos, false), sfen_key(pos, true));
    let h = sha256(if a <= b { a.as_bytes() } else { b.as_bytes() });
    let key = u64::from_le_bytes(h[0..8].try_into().unwrap());
    let split = u64::from_le_bytes(h[8..16].try_into().unwrap());
    (key, split % 50 == 0)
}

/// SHA-256（外部の crate を使わないための最小限の実装）
pub fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut h: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 { msg.push(0); }
    msg.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 { w[i] = u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().unwrap()); }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [t1.wrapping_add(t2), v[0], v[1], v[2], v[3].wrapping_add(t1), v[4], v[5], v[6]];
        }
        for i in 0..8 { h[i] = h[i].wrapping_add(v[i]); }
    }
    let mut out = [0u8; 32];
    for i in 0..8 { out[i * 4..i * 4 + 4].copy_from_slice(&h[i].to_be_bytes()); }
    out
}

/// やねうら王の16ビットの指し手を USI の文字列にする（駒打ち: 1<<14、成り: 1<<15、駒打ちの移動元の欄は駒種）
pub fn move16_to_usi(m: u16) -> String {
    let sq = |s: u16| format!("{}{}", s / 9 + 1, (b'a' + (s % 9) as u8) as char);
    let to = m & 0x7f;
    let from = (m >> 7) & 0x7f;
    if m & (1 << 14) != 0 {
        let p = ["", "P", "L", "N", "S", "B", "R", "G"].get(from as usize).copied().unwrap_or("?");
        format!("{}*{}", p, sq(to))
    } else {
        format!("{}{}{}", sq(from), sq(to), if m & (1 << 15) != 0 { "+" } else { "" })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_values() {
        let hex = |h: [u8; 32]| h.iter().map(|b| format!("{:02x}", b)).collect::<String>();
        assert_eq!(hex(sha256(b"")), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(hex(sha256(b"abc")), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        let long = vec![b'a'; 1000];
        assert_eq!(hex(sha256(&long)), "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3");
    }

    #[test]
    fn sfen_key_matches_cshogi_format() {
        // 旧TSV（cshogi の出力）にあった局面
        let sfen = "ln1g+R4/1+L5p1/p1p1pp1gp/3p5/1k4P2/2sP1S+B1P/PP1nPPN1n/1K7/5B1+r1 w 2GSL3Pslp 1";
        let pos = Position::from_sfen(sfen).unwrap();
        assert_eq!(sfen_key(&pos, false), "ln1g+R4/1+L5p1/p1p1pp1gp/3p5/1k4P2/2sP1S+B1P/PP1nPPN1n/1K7/5B1+r1 w 2GSL3Pslp");
        assert_eq!(sfen_key(&pos, true), "4+Rg1nl/1p5+L1/pg1pp1p1p/5p3/2P4k1/P1+BS1Ps2/n1NPPn1PP/7K1/1+r1B5 w 2GSL3Pslp");
        let start = Position::startpos();
        assert_eq!(sfen_key(&start, false), "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b -");
    }
}
