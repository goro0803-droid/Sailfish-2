// floodgate の自分の棋譜から、「見逃し局面」の候補を機械的に抜き出す（読み取り専用の集計）
// 使い方: node tools/extract-missed-candidates.mjs [出力フォルダ=baseline/missed_candidates_20261004] [--analysis=分析JSON]
//
// 入力: baseline/floodgate_sailfish2_analysis.json（tools/analyze-sailfish-floodgate.py の出力）
// 種類:
//   A 見立て崩れ: 自分の連続する2手の間で、自分の評価が DROP 以上下がった（1手目の時点では -FLOOR より上）
//   B 初めて -500 以下: 自分の評価が最初に -500 以下になった局面
// 評価はSailfish自身のコメントだけを自分の手同士で比べる。相手のコメントは尺度が違うので使わない。
// 候補は「そこが敗着だった」という意味ではない。再解析で確かめる入口。
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { resolve } from 'node:path';

const flagA = (process.argv.find(x => x.startsWith('--analysis=')) || '').split('=')[1];
const OUT = resolve(process.argv.slice(2).find(x => !x.startsWith('--')) || 'baseline/missed_candidates_20261004');
const DROP = 300;      // A: 2手の間の評価の下落
const FLOOR = 300;     // A: 1手目の時点で -FLOOR 以下なら、もともと不利なので除く
const MATE = 20000;    // これ以上の絶対値は詰み絡みなので除く

const a = JSON.parse(readFileSync(flagA || 'baseline/floodgate_sailfish2_analysis.json', 'utf8'));
const games = [...a.games, ...(a.extraGames || [])];
const cands = [];

for (const [gi, g] of games.entries()) {
  const own = g.moves.filter(m => m.side === g.side && typeof m.scoreOwn === 'number');
  const label = `${g.start} ${g.opponent}(${g.opponentRate})`;
  for (let i = 0; i + 1 < own.length; i++) {
    const m = own[i], n = own[i + 1];
    if (Math.abs(m.scoreOwn) >= MATE || Math.abs(n.scoreOwn) >= MATE) continue;
    const drop = m.scoreOwn - n.scoreOwn;
    if (m.scoreOwn > -FLOOR && drop >= DROP) {
      const reply = g.moves.find(x => x.ply === m.ply + 1);
      cands.push({
        kind: 'A', game: gi + 1, label, outcome: g.outcome, opponent: g.opponent, rating: g.opponentRate,
        ply: m.ply, sfen: m.beforeSfen, played: m.usi, evalBefore: m.scoreOwn, evalNext: n.scoreOwn, drop,
        opponentReply: reply ? reply.usi : null, thinkSec: m.time, clockBefore: m.clockBeforeIncrement,
      });
    }
  }
  const first = own.find(m => m.scoreOwn <= -500 && Math.abs(m.scoreOwn) < MATE);
  if (first) {
    cands.push({
      kind: 'B', game: gi + 1, label, outcome: g.outcome, opponent: g.opponent, rating: g.opponentRate,
      ply: first.ply, sfen: first.beforeSfen, played: first.usi, evalBefore: first.scoreOwn, evalNext: null, drop: null,
      opponentReply: null, thinkSec: first.time, clockBefore: first.clockBeforeIncrement,
    });
  }
}

mkdirSync(OUT, { recursive: true });
writeFileSync(resolve(OUT, 'floodgate_candidates.json'), JSON.stringify(cands, null, 1));

// 集計
const A = cands.filter(c => c.kind === 'A'), B = cands.filter(c => c.kind === 'B');
const phase = p => (p <= 40 ? '序盤(〜40手)' : p <= 100 ? '中盤(41〜100手)' : '終盤(101手〜)');
const count = (arr, f) => arr.reduce((m, c) => (m[f(c)] = (m[f(c)] || 0) + 1, m), {});
const band = r => (r < 3000 ? '〜2999' : r < 3500 ? '3000〜3499' : '3500〜');
console.log(`対局 ${games.length}（勝${games.filter(g => g.outcome === 'win').length} 負${games.filter(g => g.outcome === 'loss').length}）`);
console.log(`A 見立て崩れ ${A.length}件 / B 初めて-500以下 ${B.length}件`);
console.log('A: 結果別', count(A, c => c.outcome), ' 局面別', count(A, c => phase(c.ply)), ' 相手レート別', count(A, c => band(c.rating)));
console.log('B: 局面別', count(B, c => phase(c.ply)), ' 相手レート別', count(B, c => band(c.rating)));
console.log('--- A の大きい順（上位15）');
for (const c of [...A].sort((x, y) => y.drop - x.drop).slice(0, 15)) {
  console.log(`  局${c.game} ${c.outcome} ${c.opponent}(${c.rating}) ${c.ply}手目 ${c.played}  ${c.evalBefore}→${c.evalNext}（-${c.drop}） 相手の応手 ${c.opponentReply} 考慮${c.thinkSec}秒 残${c.clockBefore}秒`);
}
console.log('--- B 一覧');
for (const c of B) console.log(`  局${c.game} ${c.outcome} ${c.opponent}(${c.rating}) ${c.ply}手目 ${c.played} 評価${c.evalBefore} 残${c.clockBefore}秒`);
