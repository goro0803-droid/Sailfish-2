// floodgate の Sailfish の棋譜を1局ずつ振り返る。Sailfish が送った自分の評価値（指し手の後の '* 評価 読み筋）を使う。
// 使い方: node tools/floodgate-review.mjs <棋譜のフォルダ> <名前> <開始 YYYYMMDDhhmmss> <終了 YYYYMMDDhhmmss>
// 1局ごと：先後、相手（レート）、勝敗、終わり方、定跡の手数（自分の手で評価のコメントの読み筋が空、または消費時間0秒が続いた数）、
//   自分の評価の最大・最小、形勢を損ねた手数（評価が初めて −300 を下回った手）、逆転（+300 以上から負け）。
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

const [dir, me, from, to] = process.argv.slice(2);
const rows = [];
for (const f of readdirSync(dir).filter(f => f.endsWith('.csa')).sort((a, b) => a.match(/(\d{14})\.csa$/)?.[1].localeCompare(b.match(/(\d{14})\.csa$/)?.[1]))) {
  const ts = f.match(/(\d{14})\.csa$/)?.[1];
  if (!ts || ts < from || ts >= to) continue;
  const lines = readFileSync(join(dir, f), 'utf8').split(/\r?\n/);
  const name = s => (lines.find(l => l.startsWith('N' + s)) || '').slice(2).trim();
  const side = name('+') === me ? '+' : name('-') === me ? '-' : null;
  if (!side) continue;
  const oppSide = side === '+' ? 'white' : 'black';
  const rateLine = lines.find(l => l.startsWith(`'${oppSide}_rate:`));
  const opp = side === '+' ? name('-') : name('+');
  const summary = (lines.find(l => l.startsWith("'summary:")) || '').slice(9);
  const result = summary.includes(`${me} win`) ? '勝' : summary.includes(`${me} lose`) ? '負' : summary.includes('draw') ? '分' : '？';
  // 指し手・消費時間・コメントを手ごとに集める
  const moves = [];
  for (const l of lines) {
    if (/^[+-]\d{4}[A-Z]{2}$/.test(l)) moves.push({ color: l[0], time: null, score: null, pv: null });
    else if (/^T\d+/.test(l) && moves.length) moves[moves.length - 1].time = +l.slice(1);
    else if (/^'\*+ /.test(l) && moves.length) { // 送ったのは '* …、floodgate は '** … にして保存する
      const t = l.replace(/^'\*+ /, '').trim().split(/\s+/);
      // 将棋アリーナが送る評価値は先手から見た値。自分が後手なら符号を反転して、自分から見た値にする
      moves[moves.length - 1].score = side === '+' ? +t[0] : -t[0];
      moves[moves.length - 1].pv = t.slice(1);
    }
  }
  const mine = moves.map((m, i) => ({ ...m, ply: i + 1 })).filter(m => m.color === side);
  let book = 0;
  for (const m of mine) { if (m.time === 0 || (m.pv && m.pv.length <= 1 && m.time <= 1)) book++; else break; }
  const scores = mine.filter(m => m.score != null && Math.abs(m.score) < 30000);
  const max = scores.length ? Math.max(...scores.map(m => m.score)) : null;
  const min = scores.length ? Math.min(...scores.map(m => m.score)) : null;
  const lost = scores.find(m => m.score <= -300);
  rows.push({
    ts, side: side === '+' ? '先' : '後', opp, rate: rateLine ? Math.round(+rateLine.split(':').pop()) : null, result,
    reason: summary.split(':')[0], plies: moves.length, book, max, min, lostAt: lost ? lost.ply : null,
    comeback: result === '負' && max != null && max >= 300,
  });
}
console.log('日時           先後 結果 相手（レート）                       手数 定跡 評価の最大/最小  −300を下回った手 備考');
for (const r of rows) {
  console.log(`${r.ts} ${r.side}   ${r.result}   ${(r.opp + (r.rate ? `(${r.rate})` : '(-)')).padEnd(36)} ${String(r.plies).padStart(4)} ${String(r.book).padStart(4)} ${String(r.max ?? '-').padStart(6)}/${String(r.min ?? '-').padEnd(6)} ${String(r.lostAt ?? '-').padStart(6)}        ${r.comeback ? '逆転負け' : ''}${r.reason !== 'toryo' ? r.reason : ''}`);
}
const lose = rows.filter(r => r.result === '負');
const phase = p => p == null ? '不明' : p <= 40 ? '序盤（40手まで）' : p <= 100 ? '中盤（41〜100手）' : '終盤（101手〜）';
const byPhase = {};
for (const r of lose) byPhase[phase(r.lostAt)] = (byPhase[phase(r.lostAt)] || 0) + 1;
console.log(`\n${rows.length}局（${rows.filter(r => r.result === '勝').length}勝${lose.length}敗${rows.filter(r => r.result === '分').length}分）、定跡の手数の平均 ${(rows.reduce((s, r) => s + r.book, 0) / Math.max(1, rows.length)).toFixed(1)}`);
console.log(`負けた対局で形勢を損ねた段階: ${Object.entries(byPhase).map(([k, v]) => `${k} ${v}局`).join('、')}`);
console.log(`逆転負け（自分の評価が +300 以上になってから負け）: ${lose.filter(r => r.comeback).length}局`);
