// 保存した floodgate の棋譜から、期間ごとに強さを1局ずつ最尤推定する（相手のレートと勝敗から、ロジスティックの Elo）
// 使い方: node tools/floodgate-mle.mjs <棋譜のフォルダ> <名前> <開始 YYYYMMDDhhmmss> <終了 YYYYMMDDhhmmss> [<開始> <終了> ...]
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

const [dir, me, ...ranges] = process.argv.slice(2);
const all = [];
for (const f of readdirSync(dir).filter(f => f.endsWith('.csa'))) {
  const ts = f.match(/(\d{14})\.csa$/)?.[1];
  const text = readFileSync(join(dir, f), 'utf8');
  const name = s => (text.match(new RegExp(`^N\\${s}(.*)$`, 'm')) || [])[1]?.trim();
  const side = name('+') === me ? 'black' : name('-') === me ? 'white' : null;
  if (!side || !ts) continue;
  const opp = side === 'black' ? 'white' : 'black';
  const m = text.match(new RegExp(`^'${opp}_rate:[^\\n]*:(-?[\\d.]+)$`, 'm'));
  const summary = (text.match(/^'summary:(.*)$/m) || [])[1] || '';
  const score = summary.includes(`${me} win`) ? 1 : summary.includes(`${me} lose`) ? 0 : summary.includes('draw') ? 0.5 : null;
  if (!m || score == null) continue;
  all.push({ ts, opp: +m[1], score, reason: summary.split(':')[0], side });
}
const p = (r, o) => 1 / (1 + 10 ** ((o - r) / 400));
for (let i = 0; i < ranges.length; i += 2) {
  const g = all.filter(x => x.ts >= ranges[i] && x.ts < ranges[i + 1]);
  if (!g.length) { console.log(`${ranges[i]}〜${ranges[i + 1]}: 0局`); continue; }
  // 対数尤度が最大になる R を探す（1点刻み）と、尤度が最大から 1.92 下がる範囲（およそ 95%）
  const ll = r => g.reduce((s, x) => s + (x.score * Math.log(p(r, x.opp)) + (1 - x.score) * Math.log(1 - p(r, x.opp))), 0);
  let best = 0, bl = -Infinity;
  for (let r = 1000; r <= 5000; r++) { const v = ll(r); if (v > bl) { bl = v; best = r; } }
  let lo = best, hi = best;
  while (ll(lo) > bl - 1.92 && lo > 1000) lo--;
  while (ll(hi) > bl - 1.92 && hi < 5000) hi++;
  const w = g.filter(x => x.score === 1).length, l = g.filter(x => x.score === 0).length;
  const avg = Math.round(g.reduce((s, x) => s + x.opp, 0) / g.length);
  console.log(`${ranges[i]}〜${ranges[i + 1]}: ${g.length}局 ${w}勝${l}敗${g.length - w - l}分、相手の平均 ${avg}、推定 ${best}（95%の範囲 ${lo}〜${hi}）`);
}
