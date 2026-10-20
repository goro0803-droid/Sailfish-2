// floodgate の棋譜から、自分と相手の時間の使い方を集計する（5分＋10秒加算）
// 使い方: node tools/floodgate-time.mjs <棋譜のフォルダ> <名前> <開始 YYYYMMDDhhmmss> <終了 YYYYMMDDhhmmss>
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

const [dir, me, from, to] = process.argv.slice(2);
const TOTAL = 300, INC = 10;
const agg = { me: { n: 0, endRemain: [], phase: [0, 0, 0], phaseMoves: [0, 0, 0], quick: 0, moves: 0, maxMove: [] }, opp: { n: 0, endRemain: [], phase: [0, 0, 0], phaseMoves: [0, 0, 0], quick: 0, moves: 0, maxMove: [] } };
const aroundLoss = []; // 形勢を損ねた手（評価が初めて −300 を下回った手）の前5手の自分の消費時間
const lossRemain = [];
for (const f of readdirSync(dir).filter(f => f.endsWith('.csa'))) {
  const ts = f.match(/(\d{14})\.csa$/)?.[1];
  if (!ts || ts < from || ts >= to) continue;
  const lines = readFileSync(join(dir, f), 'utf8').split(/\r?\n/);
  const name = s => (lines.find(l => l.startsWith('N' + s)) || '').slice(2).trim();
  const side = name('+') === me ? '+' : name('-') === me ? '-' : null;
  if (!side) continue;
  const summary = (lines.find(l => l.startsWith("'summary:")) || '');
  const lost = summary.includes(`${me} lose`);
  const moves = [];
  for (const l of lines) {
    if (/^[+-]\d{4}[A-Z]{2}$/.test(l)) moves.push({ color: l[0], t: 0, score: null });
    else if (/^T\d+/.test(l) && moves.length) moves[moves.length - 1].t = +l.slice(1);
    else if (/^'\*+ /.test(l) && moves.length) { const v = +l.replace(/^'\*+ /, '').split(/\s+/)[0]; moves[moves.length - 1].score = side === '+' ? v : -v; }
  }
  for (const who of ['me', 'opp']) {
    const c = who === 'me' ? side : (side === '+' ? '-' : '+');
    const mine = moves.map((m, i) => ({ ...m, ply: i + 1 })).filter(m => m.color === c);
    const a = agg[who];
    let remain = TOTAL;
    for (const m of mine) {
      remain = Math.max(0, remain - m.t) + INC;
      const ph = m.ply <= 40 ? 0 : m.ply <= 100 ? 1 : 2;
      a.phase[ph] += m.t; a.phaseMoves[ph]++;
      if (m.t <= 1) a.quick++;
      a.moves++;
    }
    a.n++;
    a.endRemain.push(remain);
    a.maxMove.push(Math.max(0, ...mine.map(m => m.t)));
  }
  if (lost) {
    const mine = moves.map((m, i) => ({ ...m, ply: i + 1 })).filter(m => m.color === side);
    const k = mine.findIndex(m => m.score != null && m.score <= -300);
    if (k > 0) {
      const before = mine.slice(Math.max(0, k - 5), k);
      aroundLoss.push(before.reduce((s, m) => s + m.t, 0) / before.length);
      let remain = TOTAL;
      for (const m of mine.slice(0, k)) remain = Math.max(0, remain - m.t) + INC;
      lossRemain.push(remain);
    }
  }
}
const avg = a => a.length ? (a.reduce((s, v) => s + v, 0) / a.length) : NaN;
const med = a => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
for (const who of ['me', 'opp']) {
  const a = agg[who];
  console.log(`[${who === 'me' ? '自分（Sailfish）' : '相手'}] ${a.n}局`);
  console.log(`  1手の平均消費：序盤（〜40手）${(a.phase[0] / a.phaseMoves[0]).toFixed(1)}秒、中盤（41〜100手）${(a.phase[1] / a.phaseMoves[1]).toFixed(1)}秒、終盤（101手〜）${(a.phase[2] / a.phaseMoves[2]).toFixed(1)}秒`);
  console.log(`  1秒以下で指した手 ${(a.quick / a.moves * 100).toFixed(0)}%、1手の最長の平均 ${avg(a.maxMove).toFixed(0)}秒`);
  console.log(`  終局時の残り時間：平均 ${avg(a.endRemain).toFixed(0)}秒、中央値 ${med(a.endRemain)}秒（持ち時間300秒＋1手10秒加算）`);
}
console.log(`[負けた対局] 形勢を損ねる直前5手の自分の平均消費 ${avg(aroundLoss).toFixed(1)}秒、そのときの残り時間 平均 ${avg(lossRemain).toFixed(0)}秒・中央値 ${med(lossRemain)}秒（${aroundLoss.length}局）`);
