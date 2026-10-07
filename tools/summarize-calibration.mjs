// 浅い読み(深さ1/2/4/8)と深い読み(10秒)のずれの集計。標本ごとに、決着済みの局面（|深い評価| > 1000）と詰み絡みを除いて比べる。
// 入力: baseline/missed_candidates_20261004/eval_calibration.json（負けた局の崩れる直前）、eval_calibration_neutral.json（A 公開棋譜 / B 勝った局）
// 使い方: node tools/summarize-calibration.mjs [|深い評価|の上限=1000]
import { readFileSync } from 'node:fs';

const LIM = +(process.argv[2] || 1000);
const dir = 'baseline/missed_candidates_20261004/';
const lost = JSON.parse(readFileSync(dir + 'eval_calibration.json', 'utf8'));
const neutral = Object.values(JSON.parse(readFileSync(dir + 'eval_calibration_neutral.json', 'utf8')));
const sets = [['A 公開棋譜（強いエンジン同士・中立）', neutral.filter(r => r.src === 'A')], ['B 勝った局の自分の手番', neutral.filter(r => r.src === 'B')], ['負けた局の崩れる直前', lost]];
const q = (a, p) => a[Math.min(a.length - 1, Math.floor(p * a.length))];

console.log(`決着済み(|深い評価|>${LIM})と詰み絡みを除く。差 = 浅い読み − 10秒読み（正なら浅い読みが楽観的）`);
for (const [name, rows] of sets) {
  console.log(`\n## ${name}`);
  for (const d of ['d1', 'd2', 'd4', 'd8']) {
    const v = rows.filter(r => r[d] && r.deep && (r.deep.depth === undefined || r.deep.depth > 1) && Math.abs(r.deep.score) <= LIM && Math.abs(r[d].score) < 20000);
    if (!v.length) { console.log(`  ${d}: （なし）`); continue; }
    const diff = v.map(r => r[d].score - r.deep.score).sort((a, b) => a - b);
    const mean = diff.reduce((a, b) => a + b, 0) / diff.length;
    const sd = Math.sqrt(diff.reduce((a, b) => a + (b - mean) ** 2, 0) / diff.length);
    const abs = diff.map(Math.abs).sort((a, b) => a - b);
    console.log(`  ${d}: n=${v.length} 平均 ${mean.toFixed(0)} 中央値 ${q(diff, 0.5)} 四分位 ${q(diff, 0.25)}〜${q(diff, 0.75)} 標準偏差 ${sd.toFixed(0)} |差|中央値 ${q(abs, 0.5)} 同じ最善手 ${v.filter(r => r[d].best === r.deep.best).length}/${v.length}`);
  }
}
