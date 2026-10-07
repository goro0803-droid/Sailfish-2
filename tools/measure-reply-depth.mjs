// 「実際の相手の応手」が、何手読めば最善手として見えるか（必要な深さ d*）を測る（Threads=1、低優先度）
// 対象: first_decline.json（旧ID・新ID）の局面のうち、標準版が予想した応手(expectedReply) と実際の応手(actualReply) が違うもの。
//   見立て崩れ群: 次の自分の手番までに、旧版の自己評価が DROP 以上下がった
//   対照群      : 評価がほとんど動かなかった（|変化| < CALM）
// 測定: 実戦の手を指した後の局面（相手番）を MultiPV=8 で深さ16まで（1局面 CAP_MS で打ち切り）読み、深さごとの最善の応手と、
//       実際の応手の順位・評価を記録する。d* = 実際の応手が1位になる最小の深さ、d*安定 = それ以降の全ての深さで1位。
// 使い方: node tools/measure-reply-depth.mjs [--max-minutes=5]   途中で止まっても、保存済みの局面は飛ばして再開する
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { Usi } from './usi-engine.mjs';

const arg = n => (process.argv.find(a => a.startsWith(`--${n}=`)) || '').split('=')[1];
const MAX_MS = (+(arg('max-minutes') || 1e9)) * 60000, T0 = Date.now();
const OUT = resolve('baseline/missed_candidates_20261004/reply_depth.json');
const DROP = 150, CALM = 50, CAP_MS = 30000, MAXD = 16;
const dir = 'baseline/missed_candidates_20261004';

// 局面の材料（sfen と、次の自分の手番の旧版評価）を、棋譜の分析から引く
const sources = [
  { tag: '旧', fd: `${dir}/first_decline.json`, an: 'baseline/floodgate_sailfish2_analysis.json' },
  { tag: '新', fd: 'baseline/missed_candidates_20261004_newid/first_decline.json', an: 'baseline/floodgate_sailfish2_20261004_analysis.json' },
];
const cases = [];
for (const s of sources) {
  const an = JSON.parse(readFileSync(s.an, 'utf8')), games = [...an.games, ...(an.extraGames || [])];
  for (const r of Object.values(JSON.parse(readFileSync(s.fd, 'utf8')))) {
    if (!r.actualReply || !r.expectedReply || r.actualReply === r.expectedReply || r.now.depth <= 1) continue;
    const g = games[r.game - 1], own = g.moves.filter(m => m.side === g.side && typeof m.scoreOwn === 'number');
    const i = own.findIndex(m => m.ply === r.ply), cur = own[i], next = own[i + 1];
    if (!cur || !next || Math.abs(cur.scoreOwn) >= 20000 || Math.abs(next.scoreOwn) >= 20000) continue;
    const change = next.scoreOwn - cur.scoreOwn; // 次の自分の手番での評価の変化（旧版）
    const group = change <= -DROP ? '見立て崩れ' : Math.abs(change) < CALM ? '対照' : null;
    if (!group) continue;
    cases.push({ id: `${s.tag}局${r.game}-${r.ply}`, group, change, played: cur.usi, actualReply: r.actualReply, expectedReply: r.expectedReply, sfen: cur.beforeSfen, clock: cur.clockBeforeIncrement });
  }
}
// 対照群は多すぎないように、固定間隔で最大 12 件に絞る
const surprise = cases.filter(c => c.group === '見立て崩れ'), calm = cases.filter(c => c.group === '対照');
const pickCalm = calm.filter((_, i) => i % Math.max(1, Math.ceil(calm.length / 12)) === 0).slice(0, 12);
const todo = [...surprise, ...pickCalm];
console.log(`見立て崩れ群 ${surprise.length} 件、対照群 ${pickCalm.length} 件（候補 ${calm.length} 件から抽出）`);

const res = existsSync(OUT) ? JSON.parse(readFileSync(OUT, 'utf8')) : {};
const e = new Usi(resolve('baseline/sailfish2_time_candidate.exe')); await e.init(); e.setOption('MultiPV', 8);
const stamp = () => new Date().toLocaleTimeString('ja-JP');
let stopped = false;

for (const c of todo) {
  if (res[c.id]) continue;
  if (Date.now() - T0 > MAX_MS) { stopped = true; break; }
  const lines = await e.goRaw(`sfen ${c.sfen} moves ${c.played}`, `depth ${MAXD} movetime ${CAP_MS}`);
  // 深さごとの MultiPV 行を集める
  const byDepth = {};
  for (const l of lines) {
    const m = l.match(/^info .*?depth (\d+).*?multipv (\d+) score (cp|mate) (-?\d+).*? pv (\S+)/);
    if (!m) continue;
    const d = +m[1]; (byDepth[d] ??= {})[+m[2]] = { move: m[5], score: m[3] === 'cp' ? +m[4] : (+m[4] > 0 ? 30000 - +m[4] : -30000 - +m[4]) };
  }
  const depths = Object.keys(byDepth).map(Number).sort((a, b) => a - b);
  const rows = depths.map(d => {
    const mv = Object.entries(byDepth[d]).sort((a, b) => a[0] - b[0]).map(([k, v]) => ({ rank: +k, ...v }));
    const r = mv.find(x => x.move === c.actualReply);
    return { depth: d, best: mv[0]?.move, bestScore: mv[0]?.score, actualRank: r ? r.rank : null, actualScore: r ? r.score : null };
  });
  const first1 = rows.find(r => r.actualRank === 1);
  const maxDepth = depths.at(-1);
  // 安定: その深さ以降、到達した全ての深さで実際の応手が1位
  let stable = null;
  for (let i = rows.length - 1; i >= 0 && rows[i].actualRank === 1; i--) stable = rows[i].depth;
  res[c.id] = { ...c, maxDepth, firstBestDepth: first1 ? first1.depth : null, stableDepth: stable, rows };
  writeFileSync(OUT, JSON.stringify(res, null, 1));
  console.log(stamp(), c.id, c.group, `(変化${c.change}) 実際 ${c.actualReply}`, `到達深さ${maxDepth}`, `最初に最善 ${first1 ? 'd' + first1.depth : 'なし'}`, `安定 ${stable ? 'd' + stable : 'なし'}`);
}
e.quit();

// 集計
const all = Object.values(res);
const summarize = g => {
  const v = all.filter(r => r.group === g);
  if (!v.length) return `${g}: （なし）`;
  const fb = v.map(r => r.firstBestDepth), st = v.map(r => r.stableDepth), md = v.map(r => r.maxDepth);
  const med = a => { const x = a.filter(z => z !== null).sort((p, q) => p - q); return x.length ? x[Math.floor(x.length / 2)] : null; };
  return `${g}: n=${v.length} 到達深さ(中央値) ${med(md)} / 最初に最善 中央値 ${med(fb)}（見えた ${fb.filter(z => z !== null).length}/${v.length}） / 安定 中央値 ${med(st)}（見えた ${st.filter(z => z !== null).length}/${v.length}） / 深さ8までに見えた ${v.filter(r => r.firstBestDepth !== null && r.firstBestDepth <= 8).length}/${v.length}`;
};
console.log(stopped ? `時間で区切りました（保存 ${all.length}/${todo.length}。同じコマンドで続きから）` : `全て完了（${all.length} 件）`);
console.log(summarize('見立て崩れ'));
console.log(summarize('対照'));
