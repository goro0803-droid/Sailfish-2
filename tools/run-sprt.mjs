// 持ち時間つきの直接対局を、SPRT（逐次確率比検定）で決着がつくまで区切りごとに回す（実験 11/10〜）。
// 対局は tools/match-clock.mjs（ミリ秒で数える・同じ序盤を先後入れ替えて2局ずつ・並列）。区切りごとに partN.json を保存し、止まっても再実行で続きから。
// 判定は同じ序盤の2局の組の平均点（0・0.25・0.5・0.75・1）で行う（5通りの組の分布をそのまま使う正規近似の GSPRT）。
// 使い方: node tools/run-sprt.mjs <設定.json>  ／ --summary で集計だけ
// 設定.json: { "out": "出力フォルダ", "A": "候補 exe|オプション", "B": "基準 exe|オプション", "bankMs": 10000, "incMs": 200, "par": 8,
//              "seed": 20001, "chunk": 64, "elo0": 0, "elo1": 10, "alpha": 0.05, "beta": 0.05, "maxGames": 2000 }
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync, mkdirSync, appendFileSync } from 'node:fs';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const cfg = JSON.parse(readFileSync(process.argv[2], 'utf8'));
// byoMs（11/20 から、省略可）：秒読みのミリ秒。付けると match-clock.mjs に --byoyomi で渡す（incMs は 0、bankMs は 0 でもよい）
const { out, A, B, bankMs, incMs, byoMs = 0, par = 8, seed, chunk = 64, elo0 = 0, elo1 = 10, alpha = 0.05, beta = 0.05, maxGames = 2000 } = cfg;
if (chunk % 2 || maxGames % chunk) throw new Error('chunk は偶数で、maxGames を割り切れること');
mkdirSync(out, { recursive: true });
const log = s => { const line = `${new Date().toLocaleString('ja-JP')} ${s}`; console.log(line); appendFileSync(join(out, 'sprt.log'), line + '\n', 'utf8'); };

const lower = Math.log(beta / (1 - alpha)), upper = Math.log((1 - beta) / alpha);
const score = elo => 1 / (1 + 10 ** (-elo / 400));

/** 保存済みの区切りを読み、組の点・勝敗・時間切れを数える */
function collect() {
  const pairs = [], r = { w: 0, l: 0, d: 0, reasons: {}, timeA: 0, timeB: 0, parts: 0 };
  for (let k = 1; ; k++) {
    const f = join(out, `part${k}.json`);
    if (!existsSync(f)) break;
    const j = JSON.parse(readFileSync(f, 'utf8'));
    r.parts = k;
    for (let i = 0; i + 1 < j.samples.length; i += 2) pairs.push((j.samples[i].score + j.samples[i + 1].score) / 2);
    for (const s of j.samples) {
      if (s.score === 1) r.w++; else if (s.score === 0) r.l++; else r.d++;
      r.reasons[s.reason.split(' ')[0]] = (r.reasons[s.reason.split(' ')[0]] || 0) + 1;
      if (s.reason === '時間切れ') { if (s.score === 0) r.timeA++; else r.timeB++; }
    }
  }
  return { pairs, r };
}

/** 組の点の平均と分散から LLR（正規近似）。分散が 0 のときは 0 */
function llr(pairs) {
  const n = pairs.length;
  if (n < 2) return 0;
  const m = pairs.reduce((a, b) => a + b, 0) / n;
  const v = pairs.reduce((a, b) => a + (b - m) ** 2, 0) / n;
  if (v <= 0) return 0;
  const s0 = score(elo0), s1 = score(elo1);
  return n * (s1 - s0) * (2 * m - s0 - s1) / (2 * v);
}

function report() {
  const { pairs, r } = collect();
  const n = r.w + r.l + r.d, p = (r.w + r.d / 2) / Math.max(1, n);
  const m = pairs.reduce((a, b) => a + b, 0) / Math.max(1, pairs.length);
  const se = pairs.length > 1 ? Math.sqrt(pairs.reduce((a, b) => a + (b - m) ** 2, 0) / (pairs.length - 1) / pairs.length) : 0;
  const elo = x => (x > 0 && x < 1 ? -400 * Math.log10(1 / x - 1) : NaN);
  const L = llr(pairs);
  log(`${n}局 A ${r.w}勝${r.l}敗${r.d}分 ${(p * 100).toFixed(1)}%（約 ${elo(p).toFixed(1)} Elo、組で見た95%の幅 ±${(1.96 * se * 100).toFixed(1)}ポイント）` +
    `  LLR ${L.toFixed(2)}（${lower.toFixed(2)} 〜 ${upper.toFixed(2)}）  時間切れ A ${r.timeA}・B ${r.timeB}  終局 ${JSON.stringify(r.reasons)}`);
  return { n, L, timeA: r.timeA };
}

if (process.argv.includes('--summary')) { report(); process.exit(0); }

log(`開始（pid ${process.pid}）: A=${A}  B=${B}  ${bankMs}ms＋${incMs}ms${byoMs ? `＋秒読み${byoMs}ms` : ''}  ${par}局同時  SPRT [${elo0}, ${elo1}] α=${alpha} β=${beta} 上限${maxGames}局`);
for (let k = 1; k <= maxGames / chunk; k++) {
  const f = join(out, `part${k}.json`);
  if (!existsSync(f)) {
    const rr = spawnSync(process.execPath, ['--no-warnings', 'tools/match-clock.mjs', A, B, String(chunk), String(bankMs), String(incMs), String(par),
      '--json', f, '--seed-offset', String(seed + (k - 1) * chunk / 2), ...(byoMs ? ['--byoyomi', String(byoMs)] : [])], { cwd: ROOT, encoding: 'utf8', maxBuffer: 1 << 28 });
    if (rr.status !== 0 || !existsSync(f)) { log(`part${k} 失敗: ${(rr.stderr || '').slice(-500)}`); process.exit(1); }
  }
  const { n, L } = report();
  if (L >= upper) { log(`判定：H1（A が ${elo1} Elo 強い側）を採る`); break; }
  if (L <= lower) { log(`判定：H0（A は ${elo0} Elo 以下）を採る`); break; }
  if (n >= maxGames) { log(`判定：上限 ${maxGames}局に達した（決着つかず）`); break; }
}
log('SPRT 終了');
