// 対局を一定の局数ごとに区切って実行する（区切りごとに JSON を保存。止まっても再実行で続きから）。最後に合計を表示する。
// 使い方: node tools/run-match-chunks.mjs <出力フォルダ> <エンジンA> <エンジンB> <合計局数> <1区切りの局数> <1手のms> <並列数> <開始局面の番号>
// 区切り k の開始局面の番号 = 開始局面の番号 + (k-1)*区切りの局数/2（先後入替の組ごとに1つ）。
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync, mkdirSync, appendFileSync } from 'node:fs';
import { join } from 'node:path';

const [outDir, specA, specB, totalArg, chunkArg, msArg, parArg, seedArg] = process.argv.slice(2);
if (!seedArg) { console.log('使い方: node tools/run-match-chunks.mjs <出力フォルダ> <エンジンA> <エンジンB> <合計局数> <1区切りの局数> <1手のms> <並列数> <開始局面の番号>'); process.exit(1); }
const [total, chunk, seed] = [+totalArg, +chunkArg, +seedArg];
if (chunk % 2 || total % chunk) throw new Error('区切りの局数は偶数で、合計局数を割り切れること');
mkdirSync(outDir, { recursive: true });
const log = s => { const line = `${new Date().toLocaleString('ja-JP')} ${s}`; console.log(line); appendFileSync(join(outDir, 'chunks.log'), line + '\n'); };

for (let k = 1; k <= total / chunk; k++) {
  const out = join(outDir, `part${k}.json`);
  if (existsSync(out)) continue;
  const r = spawnSync(process.execPath, ['--no-warnings', 'tools/match.mjs', specA, specB, String(chunk), msArg, parArg,
    '--seed-offset', String(seed + (k - 1) * chunk / 2), '--json', out], { encoding: 'utf8', maxBuffer: 1 << 28 });
  if (r.status !== 0 || !existsSync(out)) { log(`part${k} 失敗: ${(r.stderr || '').slice(-500)}`); process.exit(1); }
  const j = JSON.parse(readFileSync(out, 'utf8'));
  log(`part${k}: ${j.results.w}勝${j.results.l}敗${j.results.d}分 ${(j.score * 100).toFixed(1)}%`);
}
// 合計と、先後ペア（同じ開始局面の2局）の平均による95%目安
let W = 0, L = 0, D = 0; const pairs = [];
for (let k = 1; k <= total / chunk; k++) {
  const j = JSON.parse(readFileSync(join(outDir, `part${k}.json`), 'utf8'));
  W += j.results.w; L += j.results.l; D += j.results.d;
  const m = {}; for (const s of j.samples) (m[Math.floor(s.game / 2)] ??= []).push(s.score);
  for (const a of Object.values(m)) if (a.length === 2) pairs.push((a[0] + a[1]) / 2);
}
const n = W + L + D, p = (W + D / 2) / n;
const mu = pairs.reduce((a, b) => a + b, 0) / pairs.length;
const se = Math.sqrt(pairs.reduce((a, b) => a + (b - mu) ** 2, 0) / (pairs.length - 1)) / Math.sqrt(pairs.length);
const summary = { games: n, w: W, l: L, d: D, score: p, paired95: [mu - 1.96 * se, mu + 1.96 * se], elo: -400 * Math.log10(1 / p - 1), specA, specB, moveMs: +msArg };
writeFileSync(join(outDir, 'summary.json'), JSON.stringify(summary, null, 1));
log(`合計 ${n}局 ${W}勝${L}敗${D}分 ${(p * 100).toFixed(1)}%（先後ペアの95%目安 ${(summary.paired95[0] * 100).toFixed(1)}〜${(summary.paired95[1] * 100).toFixed(1)}%）`);
