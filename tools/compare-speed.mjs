// 2つのエンジン設定の速さを比べる：同じ局面を同じ時間ずつ読ませ、到達した深さとノード数を比べる（Threads=1、Hash64、交互に実行）。
// 使い方: node tools/compare-speed.mjs "<exeA>|オプション=値,..." "<exeB>|オプション=値,..." [1局面のms=1000]
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { Usi } from './usi-engine.mjs';

const [specA, specB, msArg = '1000'] = process.argv.slice(2);
const positions = [
  ...JSON.parse(readFileSync('tools/bench-positions.json', 'utf8')).map(b => b.position),
  ...Object.values(JSON.parse(readFileSync('baseline/missed_candidates_20261004/reply_depth.json', 'utf8'))).map(r => `sfen ${r.sfen}`),
];
async function engine(spec) {
  const [exe, opts = ''] = spec.split('|');
  const e = new Usi(resolve(exe)); await e.init();
  for (const kv of opts.split(',').filter(Boolean)) { const [k, v] = kv.split('='); e.setOption(k, v); }
  e.send('isready'); const ready = await e.until(l => l === 'readyok');
  const errors = ready.filter(l => l.startsWith('info string'));
  if (errors.length) throw new Error(`${spec}: ${errors.join(' ')}`);
  return e;
}
const last = lines => {
  const l = lines.filter(x => /^info\b.*\bdepth \d+/.test(x) && / nodes /.test(x)).at(-1);
  if (!l) throw new Error('探索の情報行が読めません');
  return { depth: +l.match(/ depth (\d+)/)[1], nodes: +l.match(/ nodes (\d+)/)[1] };
};
const a = await engine(specA), b = await engine(specB);
const rows = [];
for (const [i, p] of positions.entries()) {
  const order = i % 2 ? [['b', b], ['a', a]] : [['a', a], ['b', b]];
  const r = {};
  for (const [k, e] of order) r[k] = last(await e.goRaw(p, `movetime ${msArg}`));
  rows.push(r);
}
a.quit(); b.quit();
const med = x => { const s = [...x].sort((p, q) => p - q); return s[Math.floor(s.length / 2)]; };
const avg = x => x.reduce((p, q) => p + q, 0) / x.length;
console.log(`${positions.length}局面、1局面 ${msArg}ms`);
for (const k of ['a', 'b']) console.log(`${k === 'a' ? 'A' : 'B'}: 深さ 平均 ${avg(rows.map(r => r[k].depth)).toFixed(2)}（中央値 ${med(rows.map(r => r[k].depth))}）、ノード数 中央値 ${med(rows.map(r => r[k].nodes)).toLocaleString()}`);
console.log(`A/B のノード数の比 中央値 ${med(rows.map(r => r.a.nodes / r.b.nodes)).toFixed(3)}、深さの差（A−B）平均 ${avg(rows.map(r => r.a.depth - r.b.depth)).toFixed(2)}`);
