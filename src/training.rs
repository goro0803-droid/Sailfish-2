//! 学習データを選ぶ共通フィルタ。
use crate::position::Position;
use crate::types::*;

#[derive(Clone, Copy, Default)]
pub struct TrainingFilter {
    pub quiet_drops: bool,
    pub skip_opening: bool,
}

pub fn eligible_sample(pos: &mut Position, m: Move, score: f32, ply: usize, open: usize, filter: TrainingFilter) -> bool {
    if (filter.skip_opening && ply < open) || pos.in_check() || !(score.abs() < 2500.0) || mv_is_promo(m) {
        return false;
    }
    if !mv_is_drop(m) {
        return pos.board[mv_to(m)] == 0;
    }
    if !filter.quiet_drops { return false; }
    // 駒打ちは学習に含めるが、王手になる打ちは静かな局面として扱わない。
    pos.do_move(m);
    let gives_check = pos.in_check();
    pos.undo_move();
    !gives_check
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quiet_drops_are_included_without_changing_position() {
        let mut pos = Position::from_sfen("4k4/9/9/9/9/9/9/9/4K4 b G 1").unwrap();
        let quiet = pos.parse_usi_move("G*5e").unwrap();
        let check = pos.parse_usi_move("G*5b").unwrap();
        let before = (pos.board, pos.hand, pos.king, pos.side, pos.key);
        assert!(!eligible_sample(&mut pos, quiet, 0.0, 20, 10, TrainingFilter::default()));
        let filter = TrainingFilter { quiet_drops: true, skip_opening: true };
        assert!(eligible_sample(&mut pos, quiet, 0.0, 20, 10, filter));
        assert!(!eligible_sample(&mut pos, check, 0.0, 20, 10, filter));
        assert_eq!(before, (pos.board, pos.hand, pos.king, pos.side, pos.key));
    }

    #[test]
    fn opening_and_score_filters_preserve_legacy_default() {
        let mut pos = Position::startpos();
        let m = pos.parse_usi_move("7g7f").unwrap();
        let legacy = TrainingFilter::default();
        let filter = TrainingFilter { quiet_drops: true, skip_opening: true };
        assert!(eligible_sample(&mut pos, m, 100.0, 0, 10, legacy));
        assert!(!eligible_sample(&mut pos, m, 100.0, 9, 10, filter));
        assert!(eligible_sample(&mut pos, m, 100.0, 10, 10, filter));
        assert!(!eligible_sample(&mut pos, m, 2500.0, 10, 10, filter));
        assert!(!eligible_sample(&mut pos, m, f32::NAN, 10, 10, filter));
    }
}
