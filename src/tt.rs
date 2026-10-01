//! 置換表（トランスポジションテーブル）。
//! 複数の探索スレッドから同時に読み書きできるよう、1 エントリを 2 つの AtomicU64 で持つ。
//! 書き込みの途中を読んでしまっても壊れた値を使わないよう、キーとデータの XOR を保存して照合する。

use crate::types::Move;
use std::sync::atomic::{AtomicU64, Ordering};

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
        TTEntry { mv: d as u32, score: (d >> 32) as u16 as i16, depth: (d >> 48) as u8 as i8, flag: (d >> 56) as u8 }
    }
}

#[derive(Default)]
struct Slot {
    key_xor: AtomicU64,
    data: AtomicU64,
}

pub struct TT {
    table: Vec<Slot>,
    mask: usize,
    pub size_mb: usize,
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
        TT { table, mask: size - 1, size_mb }
    }

    pub fn clear(&self) {
        for s in &self.table {
            s.key_xor.store(0, Ordering::Relaxed);
            s.data.store(0, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn probe(&self, key: u64) -> Option<TTEntry> {
        let s = &self.table[key as usize & self.mask];
        let data = s.data.load(Ordering::Relaxed);
        if s.key_xor.load(Ordering::Relaxed) ^ data == key {
            let e = TTEntry::unpack(data);
            if e.flag != 0 {
                return Some(e);
            }
        }
        None
    }

    #[inline]
    pub fn store(&self, key: u64, mv: Move, score: i32, depth: i32, flag: u8) {
        let s = &self.table[key as usize & self.mask];
        let old_data = s.data.load(Ordering::Relaxed);
        let same = s.key_xor.load(Ordering::Relaxed) ^ old_data == key;
        let old = TTEntry::unpack(old_data);
        if !same || flag == BOUND_EXACT || depth >= old.depth as i32 - 2 {
            let mv = if mv == 0 && same { old.mv } else { mv };
            let e = TTEntry { mv, score: score as i16, depth: depth.clamp(-1, 127) as i8, flag };
            let data = e.pack();
            s.data.store(data, Ordering::Relaxed);
            s.key_xor.store(key ^ data, Ordering::Relaxed);
        }
    }

    /// 静止探索の結果を保存する。同じ局面の通常探索の結果（深さ 1 以上）は上書きしない。
    #[inline]
    pub fn store_qs(&self, key: u64, mv: Move, score: i32, flag: u8) {
        let s = &self.table[key as usize & self.mask];
        let old_data = s.data.load(Ordering::Relaxed);
        let same = s.key_xor.load(Ordering::Relaxed) ^ old_data == key;
        let old = TTEntry::unpack(old_data);
        if same && old.depth > 0 {
            return;
        }
        let mv = if mv == 0 && same { old.mv } else { mv };
        let e = TTEntry { mv, score: score as i16, depth: 0, flag };
        let data = e.pack();
        s.data.store(data, Ordering::Relaxed);
        s.key_xor.store(key ^ data, Ordering::Relaxed);
    }

    /// 使用率（千分率）
    pub fn hashfull(&self) -> usize {
        let n = self.table.len().min(1000);
        self.table[..n].iter().filter(|s| TTEntry::unpack(s.data.load(Ordering::Relaxed)).flag != 0).count() * 1000 / n
    }
}
