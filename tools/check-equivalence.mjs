// 2つのエンジン設定が、固定深さで 最善手・評価・ノード数 まで一致するかを確かめる（Threads=1、Hash64）。
// 使い方: node tools/check-equivalence.mjs "<exeA>|オプション=値,..." "<exeB>|オプション=値,..." [深さ=8]
// 局面: tools/bench-positions.json と、「必要な深さ」の測定の23局面。結果が全て一致すれば終了コード0。
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { Usi } from './usi-engine.mjs';

const [specA, specB, depthArg = '8'] = process.argv.slice(2);
const depth = Number(depthArg);
const positions = [
  ...JSON.parse(readFileSync('tools/bench-positions.json', 'utf8')).map(b => ({ id: b.source, pos: b.position })),
  ...Object.values(JSON.parse(readFileSync('baseline/missed_candidates_20261004/reply_depth.json', 'utf8'))).map(r => ({ id: r.id, pos: `sfen ${r.sfen}` })),
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
  const l = lines.filter(x => /^info\b.*\bdepth \d+/.test(x) && / score /.test(x)).at(-1);
  if (!l) throw new Error('探索の情報行が読めません');
  return { score: l.match(/ score (\w+ -?\d+)/)[1], nodes: +l.match(/ nodes (\d+)/)[1], best: lines.at(-1).split(' ')[1] };
};
const a = await engine(specA), b = await engine(specB);
let same = 0;
for (const p of positions) {
  const x = last(await a.goRaw(p.pos, `depth ${depth}`)), y = last(await b.goRaw(p.pos, `depth ${depth}`));
  if (x.best === y.best && x.score === y.score && x.nodes === y.nodes) same++;
  else console.log('不一致', p.id, JSON.stringify(x), JSON.stringify(y));
}
a.quit(); b.quit();
console.log(`深さ${depth}: ${same}/${positions.length} 局面で 最善手・評価・ノード数 が一致`);
process.exit(same === positions.length ? 0 : 1);
