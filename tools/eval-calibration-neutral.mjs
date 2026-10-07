// 浅い読みと深い読み(10秒)のずれを、「負けた局の崩れる直前」ではない中立な標本で測る（Threads=1、低優先度）
// 標本 A: floodgate 2025 の公開棋譜（data/floodgate2025_approved_positions.txt）。Sailfish の勝敗と無関係
// 標本 B: 旧IDの floodgate で勝った局の、自分の手番の局面
// 局面の選び方は固定の乱数種（再現できる）。手数は 20〜150。
// 使い方: node tools/eval-calibration-neutral.mjs [--n=30] [--max-minutes=7]   途中で止まっても、保存済みの局面は飛ばして再開する
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { Usi } from './usi-engine.mjs';

const arg = n => (process.argv.find(a => a.startsWith(`--${n}=`)) || '').split('=')[1];
const N = +(arg('n') || 30), MAX_MS = (+(arg('max-minutes') || 1e9)) * 60000, T0 = Date.now();
const OUT = resolve('baseline/missed_candidates_20261004/eval_calibration_neutral.json');
const DEEP_MS = 10000;

let seed = 20261004;
const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
const pickIdx = (len, k) => { const s = new Set(); while (s.size < k) s.add(Math.floor(rnd() * len)); return [...s].sort((a, b) => a - b); };

// 標本 A
const gl = readFileSync('data/floodgate2025_approved_positions.txt', 'utf8').split('\n').filter(l => l.startsWith('R '));
const A = pickIdx(gl.length, N).map(i => {
  const mv = gl[i].split(' ').slice(3).map(t => t.split(':')[0]); // 先頭は「R 結果 序盤で乱した手数」の3項目
  const ply = 20 + Math.floor(rnd() * (Math.min(mv.length, 150) - 20 + 1)); // その手数までを指した局面
  return { src: 'A', id: `A-${i}-${ply}`, position: `startpos moves ${mv.slice(0, ply).join(' ')}` };
});
// 標本 B
const an = JSON.parse(readFileSync('baseline/floodgate_sailfish2_analysis.json', 'utf8'));
const poolB = [];
[...an.games, ...(an.extraGames || [])].forEach((g, gi) => {
  if (g.outcome !== 'win') return;
  for (const m of g.moves) if (m.side === g.side && m.ply >= 20 && m.ply <= 150 && Math.abs(m.scoreOwn ?? 0) < 20000) poolB.push({ gi: gi + 1, m });
});
const B = pickIdx(poolB.length, Math.min(N, poolB.length)).map(i => ({ src: 'B', id: `B-局${poolB[i].gi}-${poolB[i].m.ply}`, position: `sfen ${poolB[i].m.beforeSfen}` }));

const res = existsSync(OUT) ? JSON.parse(readFileSync(OUT, 'utf8')) : {};
const e = new Usi(resolve('baseline/sailfish2_time_candidate.exe')); await e.init();
const stamp = () => new Date().toLocaleTimeString('ja-JP');
let stopped = false;
// A と B を交互に処理する（途中で止まっても両方の標本がそろうように）
const order = []; for (let i = 0; i < Math.max(A.length, B.length); i++) { if (A[i]) order.push(A[i]); if (B[i]) order.push(B[i]); }
for (const it of order) {
  if (res[it.id]) continue;
  if (Date.now() - T0 > MAX_MS) { stopped = true; break; }
  const deep = await e.search(it.position, DEEP_MS);
  const rec = { src: it.src, id: it.id, deep: { best: deep.best, score: deep.score, depth: deep.depth } };
  for (const d of [1, 2, 4, 8]) { const m = Usi.multipv(await e.goRaw(it.position, `depth ${d}`))[0]; rec['d' + d] = m ? { best: m.move, score: m.score } : null; }
  res[it.id] = rec;
  writeFileSync(OUT, JSON.stringify(res, null, 1));
  console.log(stamp(), it.id, `深い ${deep.best}(${deep.score},d${deep.depth})`, `d1 ${rec.d1?.score}`);
}
e.quit();

// 集計
const lost = existsSync(resolve('baseline/missed_candidates_20261004/eval_calibration.json')) ? JSON.parse(readFileSync(resolve('baseline/missed_candidates_20261004/eval_calibration.json'), 'utf8')) : [];
const stat = (rows, d) => {
  // 詰み絡み（|評価|>=20000、浅い読み・深い読みのどちらでも）と、深さ1で即答の局面（一手しかない）は除く
  const v = rows.filter(r => r[d] && r.deep && (r.deep.depth === undefined || r.deep.depth > 1) && Math.abs(r.deep.score) < 20000 && Math.abs(r[d].score) < 20000).map(r => ({ diff: r[d].score - r.deep.score, same: r[d].best === r.deep.best }));
  if (!v.length) return '（なし）';
  const diffs = v.map(x => x.diff), mean = diffs.reduce((a, b) => a + b, 0) / diffs.length;
  const sd = Math.sqrt(diffs.reduce((a, b) => a + (b - mean) ** 2, 0) / diffs.length);
  const med = diffs.map(Math.abs).sort((a, b) => a - b)[Math.floor(diffs.length / 2)];
  return `n=${v.length} 平均差 ${mean.toFixed(0)} 標準偏差 ${sd.toFixed(0)} |差|中央値 ${med} 同じ最善手 ${v.filter(x => x.same).length}/${v.length}`;
};
const all = Object.values(res);
console.log(stopped ? `時間で区切りました（保存 ${all.length} 局面。同じコマンドで続きから）` : `全て完了（${all.length} 局面）`);
for (const [name, rows] of [['A 公開棋譜(中立)', all.filter(r => r.src === 'A')], ['B 勝った局の自分の手番', all.filter(r => r.src === 'B')], ['（参考）負けた局の崩れる直前', lost.map(r => ({ ...r }))]]) {
  console.log(`## ${name}`);
  for (const d of ['d1', 'd2', 'd4', 'd8']) console.log(`  ${d}: ${stat(rows, d)}`);
}
