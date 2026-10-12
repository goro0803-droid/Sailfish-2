// floodgate の Sailfish の棋譜（URL の一覧）を取り込み、日ごとの勝敗・相手のレート・自分のレート・終わり方をまとめる。
// 使い方: node tools/fetch-floodgate-recent.mjs <URL一覧.txt> <保存フォルダ>
// 棋譜は保存フォルダに置き、2回目からは取り込み済みのものを読み直さない。
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { join, basename } from 'node:path';

const [listPath, outDir] = process.argv.slice(2);
mkdirSync(outDir, { recursive: true });
const urls = readFileSync(listPath, 'utf8').split(/\r?\n/).filter(Boolean);
const games = [];
for (const url of urls) {
  const file = join(outDir, basename(url));
  let text;
  if (existsSync(file)) text = readFileSync(file, 'utf8');
  else {
    const res = await fetch(url);
    if (!res.ok) { console.log(`取得失敗 ${res.status} ${url}`); continue; }
    text = await res.text();
    writeFileSync(file, text);
  }
  const name = s => (text.match(new RegExp(`^N\\${s}(.*)$`, 'm')) || [])[1]?.trim();
  const black = name('+'), white = name('-');
  const side = /^sailfish2/.test(black) ? 'black' : /^sailfish2/.test(white) ? 'white' : null;
  if (!side) continue;
  const me = side === 'black' ? black : white, opp = side === 'black' ? white : black;
  const rate = who => { const m = text.match(new RegExp(`^'${who}_rate:[^\\n]*:(-?[\\d.]+)$`, 'm')); return m ? Number(m[1]) : null; };
  const myRate = rate(side), oppRate = rate(side === 'black' ? 'white' : 'black');
  const summary = (text.match(/^'summary:(.*)$/m) || [])[1] || '';
  const outcome = summary.includes(`${me} win`) ? 'win' : summary.includes(`${me} lose`) ? 'loss' : summary.includes('draw') ? 'draw' : 'other';
  const reason = (summary.split(':')[0] || '').trim();
  const start = (text.match(/^\$START_TIME:(.*)$/m) || [])[1] || '';
  const plies = (text.match(/^[+-]\d{4}[A-Z]{2}$/gm) || []).length;
  games.push({ start, me, opp, side, myRate, oppRate, outcome, reason, plies, url });
}
games.sort((a, b) => a.start.localeCompare(b.start));
writeFileSync(join(outDir, 'summary.json'), JSON.stringify(games, null, 1));
// 日ごと
const byDay = new Map();
for (const g of games) {
  const d = g.start.slice(0, 10);
  const r = byDay.get(d) || { n: 0, w: 0, l: 0, d: 0, opp: [], my: [], reasons: {} };
  r.n++; r[g.outcome === 'win' ? 'w' : g.outcome === 'loss' ? 'l' : 'd']++;
  if (g.oppRate != null) r.opp.push(g.oppRate);
  if (g.myRate != null) r.my.push(g.myRate);
  if (g.outcome === 'loss') r.reasons[g.reason] = (r.reasons[g.reason] || 0) + 1;
  byDay.set(d, r);
}
const avg = a => a.length ? Math.round(a.reduce((x, y) => x + y, 0) / a.length) : '-';
console.log('日付        局数  勝-敗-他  相手の平均レート  自分のレート（最初→最後）  負けの終わり方');
for (const [d, r] of byDay) {
  console.log(`${d}  ${String(r.n).padStart(3)}  ${r.w}-${r.l}-${r.d}  ${String(avg(r.opp)).padStart(6)}  ${r.my[0] ?? '-'} → ${r.my.at(-1) ?? '-'}  ${JSON.stringify(r.reasons)}`);
}
const ids = [...new Set(games.map(g => g.me))];
console.log(`使われた ID: ${ids.join(', ')}`);
