// NN評価の「浅い読み」が「深い読み」とどれだけずれるかを、first_decline.json の66局面で測る（Threads=1、低優先度）
// 各局面で depth 1/2/4/8 の最善手と評価を取り、保存済みの10秒読み（now.score）との差を集計する。
// 使い方: node tools/eval-calibration.mjs [出力JSON]
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { Usi } from './usi-engine.mjs';

const OUT = resolve(process.argv[2] || 'baseline/missed_candidates_20261004/eval_calibration.json');
const rows = Object.values(JSON.parse(readFileSync('baseline/missed_candidates_20261004/first_decline.json', 'utf8')));
// 詰み絡み・一手しかない局面（深さ1で即答）は除く
const items = rows.filter(r => r.now.depth > 1 && Math.abs(r.now.score) < 20000);
// 局面（sfen）は元の棋譜の分析から引く
const an = JSON.parse(readFileSync('baseline/floodgate_sailfish2_analysis.json', 'utf8'));
const games = [...an.games, ...(an.extraGames || [])];
for (const r of items) r.sfen = (games[r.game - 1].moves.find(m => m.ply === r.ply) || {}).beforeSfen;
const e = new Usi(resolve('baseline/sailfish2_time_candidate.exe')); await e.init();
const out = [];
for (const r of items) {
  const sfen = r.sfen || null;
  if (!sfen) continue;
  const rec = { game: r.game, ply: r.ply, deep: { best: r.now.best, score: r.now.score } };
  for (const d of [1, 2, 4, 8]) {
    const lines = await e.goRaw(`sfen ${sfen}`, `depth ${d}`);
    const m = Usi.multipv(lines)[0];
    rec['d' + d] = m ? { best: m.move, score: m.score } : null;
  }
  out.push(rec);
}
e.quit();
writeFileSync(OUT, JSON.stringify(out, null, 1));

const stat = d => {
  const v = out.filter(r => r[d]).map(r => ({ diff: r[d].score - r.deep.score, same: r[d].best === r.deep.best }));
  const diffs = v.map(x => x.diff).sort((a, b) => a - b);
  const mean = diffs.reduce((a, b) => a + b, 0) / diffs.length;
  const sd = Math.sqrt(diffs.reduce((a, b) => a + (b - mean) ** 2, 0) / diffs.length);
  const absMed = diffs.map(Math.abs).sort((a, b) => a - b)[Math.floor(diffs.length / 2)];
  return `${d}: 平均差 ${mean.toFixed(0)}  標準偏差 ${sd.toFixed(0)}  差の絶対値の中央値 ${absMed}  10秒読みと同じ最善手 ${v.filter(x => x.same).length}/${v.length}`;
};
console.log(`対象 ${out.length} 局面（10秒読みとの差 = 浅い読み − 10秒読み）`);
for (const d of ['d1', 'd2', 'd4', 'd8']) console.log(stat(d));
