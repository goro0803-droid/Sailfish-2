//! 置換表（トランスポジションテーブル）。
//! 複数の探索スレッドから同時に読み書きできるよう、1 エントリを 2 つの AtomicU64 で持つ。
//! 書き込みの途中を読んでしまっても壊れた値を使わないよう、キーとデータの XOR を保存して照合する。

use crate::types::Move;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

pub const BOUND_UPPER: u8 = 1;
pub const BOUND_LOWER: u8 = 2;
pub const BOUND_EXACT: u8 = 3;

#[derive(Clone, Copy, Default)]
pub struct TTEntry {
    pub mv: Move,
    pub score: i16,
    pub depth: i8,
    pub flag: u8,
}

impl TTEntry {
    #[inline]
    fn pack(self) -> u64 {
        self.mv as u64 | (self.score as u16 as u64) << 32 | (self.depth as u8 as u64) << 48 | (self.flag as u64) << 56
    }

    #[inline]
    fn unpack(d: u64) -> TTEntry {
        // 診断 tt_stats・段階2 t_bucket では flag の上の6ビットに世代を入れるので、下の2ビットだけを flag とする
        let flag = if GEN_ON { (d >> 56) as u8 & 3 } else { (d >> 56) as u8 };
        TTEntry { mv: d as u32, score: (d >> 32) as u16 as i16, depth: (d >> 48) as u8 as i8, flag }
    }
}

/// 診断（cargo の機能 tt_stats）：置換表の読み書きを全スレッドの合計で数える。添字は TS_NAMES の順
pub const TS_NAMES: [&str; 22] = [
    "probe", "hit", "store", "same_write", "same_skip", "empty", "replace",
    "rep_old_qs", "rep_old_1_3", "rep_old_4_7", "rep_old_8",
    "rep_prev_gen", "rep_deeper", "rep_deeper_cur4",
    "qs_store", "qs_same_skip", "qs_same_write", "qs_empty", "qs_replace", "qs_rep_main", "qs_rep_main_cur4", "qs_rep_prev_gen",
];
static TS: [AtomicU64; 22] = [const { AtomicU64::new(0) }; 22];

#[inline]
fn tsc(i: usize) {
    if cfg!(feature = "tt_stats") {
        TS[i].fetch_add(1, Ordering::Relaxed);
    }
}

/// 診断 tt_stats：数を読み出して 0 に戻す
pub fn tt_stats_take() -> Vec<u64> {
    TS.iter().map(|a| a.swap(0, Ordering::Relaxed)).collect()
}

/// 世代（何回目の思考か）を置換表に入れるか（診断 tt_stats と、段階2 の t_bucket）
const GEN_ON: bool = cfg!(feature = "tt_stats") || cfg!(feature = "t_bucket");

#[derive(Default)]
struct Slot {
    key_xor: AtomicU64,
    data: AtomicU64,
}

pub struct TT {
    table: Vec<Slot>,
    mask: usize,
    pub size_mb: usize,
    /// 診断 tt_stats：何回目の思考か（下の6ビット）
    generation: AtomicU8,
}

impl TT {
    pub fn new(size_mb: usize) -> TT {
        let n = size_mb.max(1) * 1024 * 1024 / std::mem::size_of::<Slot>();
        let mut size = 1;
        while size * 2 <= n {
            size *= 2;
        }
        let mut table = Vec::with_capacity(size);
        table.resize_with(size, Slot::default);
        TT { table, mask: size - 1, size_mb, generation: AtomicU8::new(0) }
    }

    /// 思考の初めに呼ぶ（診断 tt_stats・段階2 t_bucket のときだけ世代を進める）
    pub fn new_search(&self) {
        if GEN_ON {
            self.generation.store(self.generation.load(Ordering::Relaxed).wrapping_add(1) & 63, Ordering::Relaxed);
        }
    }

    /// 診断 tt_stats・段階2 t_bucket：書き込む値に世代を入れる
    #[inline]
    fn with_gen(&self, data: u64) -> u64 {
        if GEN_ON { data | (self.generation.load(Ordering::Relaxed) as u64) << 58 } else { data }
    }

    /// 診断 tt_stats：別の局面（または空き）に書くときの内訳を数える
    #[inline]
    fn count_replace(&self, old_data: u64, new_depth: i32, qs: bool) {
        if !cfg!(feature = "tt_stats") {
            return;
        }
        let old = TTEntry::unpack(old_data);
        if old.flag == 0 {
            tsc(if qs { 17 } else { 5 });
            return;
        }
        let cur = (old_data >> 58) as u8 == self.generation.load(Ordering::Relaxed);
        let d = old.depth as i32;
        if qs {
            tsc(18);
            if d >= 1 { tsc(19); }
            if cur && d >= 4 { tsc(20); }
            if !cur { tsc(21); }
        } else {
            tsc(6);
            tsc(if d <= 0 { 7 } else if d <= 3 { 8 } else if d <= 7 { 9 } else { 10 });
            if !cur { tsc(11); }
            if d > new_depth { tsc(12); }
            if cur && d >= 4 && d > new_depth { tsc(13); }
        }
    }

    /// 段階2 T1（cargo の機能 t_bucket）：隣り合う2か所を1組にする。key の局面がある方、なければ消す方を選ぶ。
    /// 消すのは、空き → 前の思考の結果 → 浅い結果 の順（同じなら1か所目）。機能が無効なら今までどおり1か所だけ
    #[inline]
    fn slot_for(&self, key: u64) -> &Slot {
        let i = key as usize & self.mask;
        if !cfg!(feature = "t_bucket") {
            return &self.table[i];
        }
        let (a, b) = (&self.table[i & !1], &self.table[i | 1]);
        let (da, db) = (a.data.load(Ordering::Relaxed), b.data.load(Ordering::Relaxed));
        if a.key_xor.load(Ordering::Relaxed) ^ da == key { return a; }
        if b.key_xor.load(Ordering::Relaxed) ^ db == key { return b; }
        let g = self.generation.load(Ordering::Relaxed);
        let worth = |d: u64| -> i32 {
            let e = TTEntry::unpack(d);
            if e.flag == 0 { return -1000; }
            e.depth as i32 - if (d >> 58) as u8 != g { 200 } else { 0 }
        };
        if worth(db) < worth(da) { b } else { a }
    }

    pub fn clear(&self) {
        for s in &self.table {
            s.key_xor.store(0, Ordering::Relaxed);
            s.data.store(0, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn probe(&self, key: u64) -> Option<TTEntry> {
        let s = self.slot_for(key);
        let data = s.data.load(Ordering::Relaxed);
        tsc(0);
        if s.key_xor.load(Ordering::Relaxed) ^ data == key {
            let e = TTEntry::unpack(data);
            if e.flag != 0 {
                tsc(1);
                return Some(e);
            }
        }
        None
    }

    #[inline]
    pub fn store(&self, key: u64, mv: Move, score: i32, depth: i32, flag: u8) {
        let s = self.slot_for(key);
        let old_data = s.data.load(Ordering::Relaxed);
        let same = s.key_xor.load(Ordering::Relaxed) ^ old_data == key;
        let old = TTEntry::unpack(old_data);
        tsc(2);
        if !same || flag == BOUND_EXACT || depth >= old.depth as i32 - 2 {
            if same { tsc(3); } else { self.count_replace(old_data, depth, false); }
            let mv = if mv == 0 && same { old.mv } else { mv };
            let e = TTEntry { mv, score: score as i16, depth: depth.clamp(-1, 127) as i8, flag };
            let data = self.with_gen(e.pack());
            s.data.store(data, Ordering::Relaxed);
            s.key_xor.store(key ^ data, Ordering::Relaxed);
        } else {
            tsc(4);
        }
    }

    /// 静止探索の結果を保存する。同じ局面の通常探索の結果（深さ 1 以上）は上書きしない。
    #[inline]
    pub fn store_qs(&self, key: u64, mv: Move, score: i32, flag: u8) {
        let s = self.slot_for(key);
        let old_data = s.data.load(Ordering::Relaxed);
        let same = s.key_xor.load(Ordering::Relaxed) ^ old_data == key;
        let old = TTEntry::unpack(old_data);
        tsc(14);
        if same && old.depth > 0 {
            tsc(15);
            return;
        }
        if same { tsc(16); } else { self.count_replace(old_data, 0, true); }
        let mv = if mv == 0 && same { old.mv } else { mv };
        let e = TTEntry { mv, score: score as i16, depth: 0, flag };
        let data = self.with_gen(e.pack());
        s.data.store(data, Ordering::Relaxed);
        s.key_xor.store(key ^ data, Ordering::Relaxed);
    }

    /// 使用率（千分率）
    pub fn hashfull(&self) -> usize {
        let n = self.table.len().min(1000);
        self.table[..n].iter().filter(|s| TTEntry::unpack(s.data.load(Ordering::Relaxed)).flag != 0).count() * 1000 / n
    }
}
