// 持ち時間つきの対局を一定の局数ごとに区切って実行する（区切りごとに JSON を保存。止まっても再実行で続きから）。最後に合計を表示する。
// 使い方: node tools/run-clock-chunks.mjs <出力フォルダ> <エンジンA> <エンジンB> <合計局数> <1区切りの局数> <持ち時間ms> <加算ms> <並列数> <開始局面の番号>
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync, mkdirSync, appendFileSync } from 'node:fs';
import { join } from 'node:path';

const [outDir, specA, specB, totalArg, chunkArg, bankArg, incArg, parArg, seedArg] = process.argv.slice(2);
if (!seedArg) { console.log('使い方: node tools/run-clock-chunks.mjs <出力フォルダ> <エンジンA> <エンジンB> <合計局数> <1区切りの局数> <持ち時間ms> <加算ms> <並列数> <開始局面の番号>'); process.exit(1); }
const [total, chunk, seed] = [+totalArg, +chunkArg, +seedArg];
if (chunk % 2 || total % chunk) throw new Error('区切りの局数は偶数で、合計局数を割り切れること');
mkdirSync(outDir, { recursive: true });
const log = s => { const line = `${new Date().toLocaleString('ja-JP')} ${s}`; console.log(line); appendFileSync(join(outDir, 'chunks.log'), line + '\n'); };

for (let k = 1; k <= total / chunk; k++) {
  const out = join(outDir, `part${k}.json`);
  if (existsSync(out)) continue;
  const r = spawnSync(process.execPath, ['--no-warnings', 'tools/match-clock.mjs', specA, specB, String(chunk), bankArg, incArg, parArg,
    '--json', out, '--seed-offset', String(seed + (k - 1) * chunk / 2)], { encoding: 'utf8', maxBuffer: 1 << 28 });
  if (r.status !== 0 || !existsSync(out)) { log(`part${k} 失敗: ${(r.stderr || '').slice(-500)}`); process.exit(1); }
  const j = JSON.parse(readFileSync(out, 'utf8'));
  log(`part${k}: ${j.results.w}勝${j.results.l}敗${j.results.d}分 ${(j.score * 100).toFixed(1)}%`);
}
let W = 0, L = 0, D = 0;
for (let k = 1; k <= total / chunk; k++) { const r = JSON.parse(readFileSync(join(outDir, `part${k}.json`), 'utf8')).results; W += r.w; L += r.l; D += r.d; }
const n = W + L + D, p = (W + D / 2) / n;
writeFileSync(join(outDir, 'summary.json'), JSON.stringify({ games: n, w: W, l: L, d: D, score: p, specA, specB, bankMs: +bankArg, incMs: +incArg }, null, 1));
log(`合計 ${n}局 ${W}勝${L}敗${D}分 ${(p * 100).toFixed(1)}%`);
