# cshogi で PackedSfenValue を読み、Rust の読み込み（src/psv.rs）を検査するための正解TSVを作る。
# 使い方: python tools/psv-reference.py <教師.bin> <出力.tsv> <件数> [間隔=1]
# 各行: 番号 TAB sfen TAB 指し手(USI) TAB 評価値 TAB 手数 TAB 勝敗（評価値・勝敗は手番側から見た値）
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'data/mekitto_runtime'))
import cshogi, numpy as np
src, out, count = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
step = int(sys.argv[4]) if len(sys.argv) > 4 else 1
total = src.stat().st_size // 40
board = cshogi.Board()
with src.open('rb') as f, out.open('w', encoding='utf-8', newline='\n') as w:
    for k in range(count):
        i = k * step
        if i >= total: break
        f.seek(i * 40)
        r = np.frombuffer(f.read(40), dtype=cshogi.PackedSfenValue)[0]
        board.set_psfen(np.array(r['sfen'], copy=True))
        move = board.move_from_move16(cshogi.move16_from_psv(int(r['move'])))
        w.write(f"{i}\t{board.sfen()}\t{cshogi.move_to_usi(move)}\t{int(r['score'])}\t{int(r['gamePly'])}\t{int(r['game_result'])}\n")
