//! 定跡（実験BK）。やねうら王の標準の定跡形式（#YANEURAOU-DB2016）を読み、局面のキーで引く。
//! cargo の機能 book のとき、`book/default.db` を実行ファイルに埋め込んで使う。

use crate::position::Position;
use std::collections::HashMap;
use std::sync::OnceLock;

pub struct BookMove {
    pub mv: String,
    pub ponder: Option<String>,
    pub value: i32,
    pub count: u32,
}

pub struct Book {
    map: HashMap<u64, Vec<BookMove>>,
}

impl Book {
    /// 定跡ファイルの文字列を読む（`sfen …` の行と、`指し手 予想応手 評価 深さ 回数` の行）
    pub fn parse(text: &str) -> Book {
        let mut map: HashMap<u64, Vec<BookMove>> = HashMap::new();
        let mut cur: Option<u64> = None;
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(s) = line.strip_prefix("sfen ") {
                cur = Position::from_sfen(s).ok().map(|p| p.key);
                continue;
            }
            let Some(k) = cur else { continue };
            let t: Vec<&str> = line.split_whitespace().collect();
            if t.is_empty() {
                continue;
            }
            map.entry(k).or_default().push(BookMove {
                mv: t[0].to_string(),
                ponder: t.get(1).filter(|&&p| p != "none").map(|s| s.to_string()),
                value: t.get(2).and_then(|v| v.parse().ok()).unwrap_or(0),
                count: t.get(4).and_then(|v| v.parse().ok()).unwrap_or(1),
            });
        }
        Book { map }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// この局面の定跡の手から、回数に比例した確率で 1 つ選ぶ（指せない手は除く）。rnd は乱数
    pub fn pick(&self, pos: &Position, rnd: u64) -> Option<&BookMove> {
        let list = self.map.get(&pos.key)?;
        let mut p = pos.clone();
        let ok: Vec<&BookMove> = list.iter().filter(|b| p.parse_usi_move(&b.mv).is_some()).collect();
        let total: u64 = ok.iter().map(|b| b.count as u64).sum();
        if total == 0 {
            return None;
        }
        let mut r = rnd % total;
        for b in ok {
            if r < b.count as u64 {
                return Some(b);
            }
            r -= b.count as u64;
        }
        None
    }
}

/// 埋め込みの定跡（cargo の機能 book のときだけ中身がある）
pub fn global() -> &'static Book {
    static BOOK: OnceLock<Book> = OnceLock::new();
    BOOK.get_or_init(|| {
        #[cfg(feature = "book")]
        {
            Book::parse(include_str!("../book/default.db"))
        }
        #[cfg(not(feature = "book"))]
        {
            Book::parse("")
        }
    })
}
