//! Clock allocation; fixed movetime and node/depth analysis are handled separately.
#[derive(Clone, Copy, Debug)]
pub struct Budget { pub soft_ms: u64, pub hard_ms: u64 }
pub fn allocate(time: u64, inc: u64, byoyomi: u64, delay: u64, game_ply: usize, balanced: bool) -> Budget {
    let cap = time.saturating_add(byoyomi);
    let legacy_hard = (time / 8).saturating_add(byoyomi).saturating_add(inc).min(cap);
    let legacy_soft = (time / 40).saturating_add(byoyomi).saturating_add(inc.saturating_mul(3) / 4).min(legacy_hard);
    // Opening reserve applies to increment games, not to fixed byoyomi.
    let (soft, hard) = if balanced && inc > 0 && byoyomi == 0 {
        let soft = if game_ply < 20 {
            (time / 100).saturating_add(inc / 2)
        } else if game_ply < 40 {
            (time / 60).saturating_add(inc.saturating_mul(3) / 4)
        } else { legacy_soft };
        let burst = if game_ply < 20 { soft.saturating_mul(3) / 2 }
            else if game_ply < 40 { soft.saturating_mul(2) }
            else { soft.saturating_mul(3) };
        let hard = burst.min(legacy_hard).min(cap);
        (soft.min(hard), hard)
    } else { (legacy_soft, legacy_hard) };
    // For very small clocks, never add a 10ms floor beyond the available clock.
    let hard_ms = hard.saturating_sub(delay).max(1).min(cap.max(1));
    let soft_ms = soft.saturating_sub(delay).max(1).min(hard_ms);
    Budget { soft_ms, hard_ms }
}
pub fn stop_before_next(elapsed: u64, iteration: u64, previous: u64, soft: u64, hard: u64, stable: bool) -> bool {
    let target = if stable { soft } else { soft.saturating_mul(5) / 4 }.min(hard);
    if elapsed >= target { return true; }
    if elapsed < soft / 2 || previous == 0 { return false; }
    let ratio = (iteration as f64 / previous as f64).clamp(1.5, 6.0);
    let next = (iteration as f64 * ratio).ceil() as u64;
    elapsed.saturating_add(next) > target
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn increment_opening_preserves_bank_and_middle_gets_more() {
        let early = allocate(300000,10000,0,120,0,true);
        let middle = allocate(300000,10000,0,120,40,true);
        let old = allocate(300000,10000,0,120,0,false);
        assert_eq!((early.soft_ms,early.hard_ms),(7880,11880));
        assert_eq!((old.soft_ms,old.hard_ms),(14880,47380));
        assert!(middle.soft_ms>early.soft_ms && middle.hard_ms>early.hard_ms);
        let mut bank=300000;
        for ply in (0..20).step_by(2) { bank=bank-allocate(bank,10000,0,120,ply,true).hard_ms+10000; }
        assert!(bank>250000);
    }
    #[test] fn low_time_never_exceeds_available_clock() {
        for time in [0,1,5,100,1000,300000,u64::MAX/2] {
            for ply in [0,20,40] {
                let b=allocate(time,10000,0,120,ply,true);
                assert!(b.soft_ms<=b.hard_ms && b.hard_ms<=time.max(1));
            }
        }
    }
    #[test] fn byoyomi_and_legacy_keep_regular_allocation() {
        for time in [0,1000,300000] {
            let a=allocate(time,0,10000,120,0,true);
            let b=allocate(time,0,10000,120,0,false);
            assert_eq!((a.soft_ms,a.hard_ms),(b.soft_ms,b.hard_ms));
        }
    }
    #[test] fn next_iteration_prediction_avoids_stable_overshoot() {
        assert!(stop_before_next(6000,3000,1000,8000,12000,true));
        assert!(!stop_before_next(1000,500,200,8000,12000,true));
        assert!(stop_before_next(12000,100,100,8000,12000,false));
        assert!(!stop_before_next(8500,100,100,8000,12000,false));
    }
}
