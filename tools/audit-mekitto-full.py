# めきっと氏の公開教師データ（PackedSfenValue、1局面40バイト）を全件検査する。100万局面ごとに途中経過を保存し、止まっても続きから再開する。
# 使い方: python tools/audit-mekitto-full.py [bin=D:/shogi_data/mekitto/shuffled_20190212.bin] [出力=D:/shogi_data/mekitto/audit_full.json]
# 検査: 盤面の復元、指し手の合法性、勝敗(-1/0/1)、評価値・手数の分布、王手の局面の数。評価値・勝敗は手番側から見た値（反転しない）。
from pathlib import Path
import sys, json, time, collections
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'data/mekitto_runtime'))
import cshogi, numpy as np

BIN = Path(sys.argv[1] if len(sys.argv) > 1 else 'D:/shogi_data/mekitto/shuffled_20190212.bin')
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else 'D:/shogi_data/mekitto/audit_full.json')
CHUNK = 1_000_000
REC = 40
total = BIN.stat().st_size // REC
assert BIN.stat().st_size % REC == 0, 'ファイルの長さが40バイトの倍数ではない'

state = json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists() else {
    'file': str(BIN), 'records': total, 'done': 0, 'counts': {}, 'score_hist': {}, 'ply_hist': {}, 'result': {}, 'examples': []}
counts = collections.Counter(state['counts'])
score_hist = collections.Counter({int(k): v for k, v in state['score_hist'].items()})   # 100点刻み（±3000で頭打ち）
ply_hist = collections.Counter({int(k): v for k, v in state['ply_hist'].items()})       # 10手刻み
result = collections.Counter(state['result'])
board = cshogi.Board()
t0 = time.time()
with BIN.open('rb') as f:
    while state['done'] < total:
        f.seek(state['done'] * REC)
        n = min(CHUNK, total - state['done'])
        data = np.frombuffer(f.read(n * REC), dtype=cshogi.PackedSfenValue)
        for r in data:
            try:
                board.set_psfen(np.array(r['sfen'], copy=True))
            except Exception:
                counts['decode_error'] += 1
                if len(state['examples']) < 20: state['examples'].append({'index': state['done'], 'kind': 'decode_error'})
                continue
            move = board.move_from_move16(cshogi.move16_from_psv(int(r['move'])))
            if not board.is_legal(move):
                counts['illegal_move'] += 1
                if len(state['examples']) < 20: state['examples'].append({'kind': 'illegal_move', 'sfen': board.sfen()})
            res = int(r['game_result'])
            result[str(res)] += 1
            if res not in (-1, 0, 1): counts['invalid_result'] += 1
            s = int(r['score'])
            score_hist[max(-30, min(30, s // 100))] += 1
            ply_hist[min(50, int(r['gamePly']) // 10)] += 1
            if board.is_check(): counts['in_check'] += 1
            counts['read'] += 1
        state['done'] += n
        state.update(counts=dict(counts), score_hist={str(k): v for k, v in sorted(score_hist.items())},
                     ply_hist={str(k): v for k, v in sorted(ply_hist.items())}, result=dict(result))
        OUT.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding='utf-8')
        rate = (state['done'] - 0) / max(1e-9, time.time() - t0)
        print(f"{state['done']:,}/{total:,} 読込 {counts['read']:,} 不正手 {counts['illegal_move']} 復元エラー {counts['decode_error']} 勝敗不正 {counts['invalid_result']}", flush=True)
print('完了', json.dumps({k: state[k] for k in ('records', 'done', 'counts', 'result')}, ensure_ascii=False), flush=True)
bad = counts['illegal_move'] + counts['decode_error'] + counts['invalid_result']
sys.exit(1 if bad else 0)
