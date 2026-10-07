// 脅威を作る駒打ちの実験（F_THREAT_DROP）の段階0・1の確認（Threads=1、低優先度）
// 段階0: 標準exe と 新exe(機能OFF) が、固定深さで 最善手・評価・ノード数 まで一致するか
// 段階1: 新exe の機能OFF と ON で、固定深さまでのノード数・時間を比べる（ONの増加が中央値で +15% 以内か）
// 局面: tools/bench-positions.json と、「必要な深さ」の測定の23局面（実戦の手を指す前の局面）
// 使い方: node tools/check-threat-drop.mjs [段階0の深さ=9] [段階1の深さ=10]
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { Usi } from './usi-engine.mjs';

const [d0 = 9, d1 = 10] = process.argv.slice(2).map(Number);
const STD = resolve('baseline/sailfish2_time_candidate.exe'); // 標準exeと同じSHA-256
const NEW = resolve('baseline/threat_drop_20261004/sailfish2_threat_drop.exe');
const FEAT_OFF = 131121, FEAT_ON = 131121 + 134217728;
const positions = [
  ...JSON.parse(readFileSync('tools/bench-positions.json', 'utf8')).map(b => ({ id: b.source, pos: b.position })),
  ...Object.values(JSON.parse(readFileSync('baseline/missed_candidates_20261004/reply_depth.json', 'utf8'))).map(r => ({ id: r.id, pos: `sfen ${r.sfen}` })),
];

async function engine(exe, feat) {
  const e = new Usi(exe); await e.init();
  if (feat !== undefined) e.setOption('Feat', feat);
  e.send('isready'); await e.until(l => l === 'readyok');
  return e;
}
const last = lines => {
  const l = lines.filter(x => /^info\b.*\bdepth \d+/.test(x) && / score /.test(x)).at(-1) || '';
  if (!l) throw new Error('探索の情報行が読めません');
  return { depth: +(l.match(/ depth (\d+)/)?.[1] || 0), score: l.match(/ score (\w+ -?\d+)/)?.[1], nodes: +(l.match(/ nodes (\d+)/)?.[1] || 0), time: +(l.match(/ time (\d+)/)?.[1] || 0), best: lines.at(-1).split(' ')[1] };
};

// 段階0（深さに 0 を指定すると省略する）
const std = await engine(STD), off = await engine(NEW, FEAT_OFF);
let same = 0; const diff0 = [];
for (const p of d0 > 0 ? positions : []) {
  const a = last(await std.goRaw(p.pos, `depth ${d0}`)), b = last(await off.goRaw(p.pos, `depth ${d0}`));
  const ok = a.best === b.best && a.score === b.score && a.nodes === b.nodes;
  if (ok) same++; else diff0.push({ id: p.id, std: a, off: b });
}
std.quit();
console.log(`段階0（深さ${d0}）: 標準exe と 新exe(OFF) の一致 ${same}/${positions.length}`);
for (const d of diff0) console.log('  不一致', d.id, JSON.stringify(d.std), JSON.stringify(d.off));

if (d1 <= 0) { off.quit(); writeFileSync(`baseline/threat_drop_20261004/stage0_d${d0}.json`, JSON.stringify({ d0, same, total: positions.length, diff: diff0 }, null, 1)); process.exit(0); }
// 段階1（OFF と ON を交互に実行。深さに 0 を指定すると省略する）
const on = await engine(NEW, FEAT_ON);
const rows = [];
for (const [i, p] of positions.entries()) {
  const order = i % 2 ? [['on', on], ['off', off]] : [['off', off], ['on', on]];
  const r = { id: p.id };
  for (const [k, e] of order) r[k] = last(await e.goRaw(p.pos, `depth ${d1}`));
  rows.push(r);
}
off.quit(); on.quit();
const med = a => { const x = [...a].sort((p, q) => p - q); return x[Math.floor(x.length / 2)]; };
const nodeRatio = rows.map(r => r.on.nodes / r.off.nodes), timeRatio = rows.filter(r => r.off.time > 50).map(r => r.on.time / r.off.time);
console.log(`段階1（深さ${d1}）: ON/OFF のノード数の比 中央値 ${med(nodeRatio).toFixed(3)}（最小 ${Math.min(...nodeRatio).toFixed(2)} 最大 ${Math.max(...nodeRatio).toFixed(2)}）、時間の比 中央値 ${med(timeRatio).toFixed(3)}（${timeRatio.length}局面）`);
console.log(`  最善手が変わった局面 ${rows.filter(r => r.on.best !== r.off.best).length}/${rows.length}`);
writeFileSync(`baseline/threat_drop_20261004/stage${d0 > 0 ? '0_' : ''}1_d${d1}.json`, JSON.stringify({ d0, d1, stage0: { same, total: positions.length, diff: diff0 }, stage1: rows }, null, 1));
