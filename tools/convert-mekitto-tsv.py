# めきっと氏の公開教師データ（PackedSfenValue）から、nn_train --positions 用の学習・検証TSVを作る（実験A用、メモリに載る件数まで）。
# 使い方: python tools/convert-mekitto-tsv.py <採用する局面数> <出力フォルダ> [bin=D:/shogi_data/mekitto/shuffled_20190212.bin] [評価値の上限=3000]
# - ファイルは元々シャッフル済みなので、先頭から読む（＝無作為抽出と同等）。
# - 除外: 手数16未満、王手がかかっている局面、|評価値| > 上限、指し手が不正。同じ局面（左右反転を含む）は1つだけ残す。
# - 学習/検証の分割は、左右反転を同一視した局面のSHA256で行う（50分の1を検証）。棋譜IDがないので、同じ対局の別局面が両側に入る可能性は残る。
# - 行の形式は tools/sample-mekitto.py と同じ: P <TAB> sfen <TAB> 評価値 <TAB> 勝敗(1/0.5/0) <TAB> 指し手 <TAB> 手数。評価値・勝敗は手番側から見た値。
from pathlib import Path
import sys, json, hashlib, collections, time
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'data/mekitto_runtime'))
sys.path.insert(0, str(ROOT / 'tools'))
import cshogi, numpy as np
from importlib import import_module
mirror_key = import_module('sample-mekitto').mirror_key

want = int(sys.argv[1]); out = Path(sys.argv[2])
BIN = Path(sys.argv[3] if len(sys.argv) > 3 else 'D:/shogi_data/mekitto/shuffled_20190212.bin')
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
out.mkdir(parents=True, exist_ok=True)
if (out / 'train.tsv').exists(): raise SystemExit('出力が既にあります: ' + str(out))
files = [(out / 'train.tsv').open('w', encoding='utf-8', newline='\n'), (out / 'validation.tsv').open('w', encoding='utf-8', newline='\n')]
counts = collections.Counter(); seen = set(); board = cshogi.Board(); t0 = time.time()
with BIN.open('rb') as f:
    while counts['accepted'] < want:
        raw = f.read(1_000_000 * 40)
        if not raw: break
        for r in np.frombuffer(raw, dtype=cshogi.PackedSfenValue):
            if counts['accepted'] >= want: break
            counts['read'] += 1
            board.set_psfen(np.array(r['sfen'], copy=True))
            move = board.move_from_move16(cshogi.move16_from_psv(int(r['move'])))
            score = int(r['score']); res = int(r['game_result']); ply = int(r['gamePly'])
            if not board.is_legal(move) or res not in (-1, 0, 1): counts['invalid'] += 1; continue
            if ply < 16 or board.is_check() or abs(score) > LIMIT: counts['filtered'] += 1; continue
            sfen = board.sfen()
            key = ' '.join(sfen.split()[:3])
            canonical = min(key, mirror_key(sfen))
            h = hashlib.sha256(canonical.encode()).digest()
            hk = int.from_bytes(h[:8], 'little')
            if hk in seen: counts['duplicate'] += 1; continue
            seen.add(hk)
            val = int.from_bytes(h[8:16], 'little') % 50 == 0
            files[val].write(f"P\t{sfen}\t{score}\t{(res + 1) / 2}\t{cshogi.move_to_usi(move)}\t{ply}\n")
            counts['accepted'] += 1; counts['validation' if val else 'train'] += 1
        print(f"読込 {counts['read']:,} 採用 {counts['accepted']:,}（{counts['read'] / (time.time() - t0):,.0f}件/秒）", flush=True)
for f in files: f.close()
if counts['invalid']: raise SystemExit('不正な局面があります: ' + str(dict(counts)))
(out / 'convert.json').write_text(json.dumps({'source': str(BIN), 'want': want, 'scoreLimit': LIMIT, 'counts': dict(counts),
    'note': '先頭から読み込み（元ファイルはシャッフル済み）。左右反転を同一視して重複除外・分割。'}, ensure_ascii=False, indent=1), encoding='utf-8')
print('完了', dict(counts), flush=True)
