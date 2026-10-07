// 負けた局で、評価が「最初に悪化した手」を探す。
// 1. 旧ID の floodgate 棋譜の自分の評価の推移から、確定した悪化点を決める。
//    確定した悪化点: 自分の評価が -150 以下になり、そこから続く6手すべてが -100 以下になる最初の自分の手。
//    詰み絡み(|評価|>=20000)は -100 以下として扱う。
// 2. その直前の自分の手 WINDOW 個（まだ優勢・互角だった側）を、今の標準版(Threads=1)で解析し、
//    実戦の手との違い・指した後の比較・予想した相手の応手と実際の応手の違いを記録する。
// 使い方: node tools/trace-first-decline.mjs [--plan-only] [--analysis=分析JSON] [出力JSON]   途中で止まっても、保存済みの局面は飛ばして再開する。
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { Usi } from './usi-engine.mjs';

const planOnly = process.argv.includes('--plan-only');
// --games=13,9,10 で、その局だけをその順に処理する。--max-minutes=8 で、その時間を過ぎたら現在の局面を終えて区切る（再実行で続きから）
const arg = name => (process.argv.find(a => a.startsWith(`--${name}=`)) || '').split('=')[1];
const ONLY = arg('games') ? arg('games').split(',').map(Number) : null;
const MAX_MS = arg('max-minutes') ? +arg('max-minutes') * 60000 : Infinity;
const T0 = Date.now();
const OUT = resolve(process.argv.slice(2).find(a => !a.startsWith('--')) || 'baseline/missed_candidates_20261004/first_decline.json');
const EXE = resolve('baseline/sailfish2_time_candidate.exe'); // 標準exeと同じSHA-256の凍結コピー
const WINDOW = +(arg('window') || 5), SEARCH_MS = 10000, AFTER_MS = 20000;
const BAD = -150, STAY = -100, STAY_N = 6, MATE = 20000;

const a = JSON.parse(readFileSync(arg('analysis') || 'baseline/floodgate_sailfish2_analysis.json', 'utf8'));
const games = [...a.games, ...(a.extraGames || [])];

const plan = [];
games.forEach((g, gi) => {
  if (g.outcome !== 'loss') return;
  const own = g.moves.filter(m => m.side === g.side && typeof m.scoreOwn === 'number');
  const low = m => Math.abs(m.scoreOwn) >= MATE || m.scoreOwn <= STAY;
  let c = -1;
  for (let i = 0; i < own.length; i++) {
    if (own[i].scoreOwn <= BAD && Math.abs(own[i].scoreOwn) < MATE && own.slice(i, i + STAY_N).every(low)) { c = i; break; }
  }
  if (c < 0) return;
  const from = Math.max(0, c - WINDOW);
  const peak = own.slice(Math.max(0, c - 12), c).reduce((b, m) => (m.scoreOwn > b.scoreOwn ? m : b));
  plan.push({ game: gi + 1, opponent: g.opponent, rating: g.opponentRate, ownColor: g.side ? '後手' : '先手', crossingPly: own[c].ply, crossingEval: own[c].scoreOwn,
    peakPly: peak.ply, peakEval: peak.scoreOwn, clockAtCrossing: own[c].clockBeforeIncrement,
    window: own.slice(from, c + 1).map(m => ({ ply: m.ply, played: m.usi, oldEval: m.scoreOwn, clock: m.clockBeforeIncrement, thinkSec: m.time, sfen: m.beforeSfen,
      actualReply: (g.moves.find(x => x.ply === m.ply + 1) || {}).usi || null })) });
});

if (planOnly) {
  for (const p of plan) console.log(`局${p.game} ${p.opponent}(${p.rating}) ${p.ownColor}: 山 ${p.peakPly}手目(${p.peakEval}) → 確定した悪化点 ${p.crossingPly}手目(${p.crossingEval}) 残${p.clockAtCrossing}秒  窓 ${p.window.map(w => w.ply + ':' + w.oldEval).join(' ')}`);
  process.exit(0);
}

mkdirSync(dirname(OUT), { recursive: true });
const result = existsSync(OUT) ? JSON.parse(readFileSync(OUT, 'utf8')) : {};
const e = new Usi(EXE); await e.init();
const stamp = () => new Date().toLocaleTimeString('ja-JP');
const afterScore = async (pos, mv) => { const s = await e.search(`${pos} moves ${mv}`, AFTER_MS); return { move: mv, ownScore: s.score === null ? null : -s.score, depth: s.depth, reply: s.best }; };

const order = ONLY ? ONLY.map(g => plan.find(p => p.game === g)).filter(Boolean) : plan;
let stopped = false;
for (const p of order) {
  for (const w of p.window) {
    const key = `局${p.game}-${w.ply}手目`;
    if (result[key]) continue;
    if (Date.now() - T0 > MAX_MS) { stopped = true; break; }
    const pos = `sfen ${w.sfen}`;
    const r = { game: p.game, opponent: p.opponent, rating: p.rating, ply: w.ply, played: w.played, oldEval: w.oldEval, clock: w.clock, actualReply: w.actualReply, crossingPly: p.crossingPly };
    const s = await e.search(pos, SEARCH_MS);
    r.now = { best: s.best, score: s.score, depth: s.depth, pv: s.pv };
    r.expectedReply = s.pv[1] || null; // 標準版が予想する相手の応手（最善手の次）
    r.sameMove = s.best === w.played;
    if (!r.sameMove && s.depth > 1) {
      r.afterPlayed = await afterScore(pos, w.played);
      r.afterBest = await afterScore(pos, s.best);
    }
    result[key] = r;
    writeFileSync(OUT, JSON.stringify(result, null, 1));
    console.log(stamp(), key, `実戦 ${w.played}(旧${w.oldEval}) / 今 ${s.best}(${s.score}, 深さ${s.depth})`,
      r.sameMove ? '同じ手' : `違う手: 実戦の手の後 ${r.afterPlayed?.ownScore} / 今の手の後 ${r.afterBest?.ownScore}`,
      `予想応手 ${r.expectedReply} 実際 ${w.actualReply}`);
  }
  if (stopped) break;
}
e.quit();
const left = order.reduce((n, p) => n + p.window.filter(w => !result[`局${p.game}-${w.ply}手目`]).length, 0);
console.log(stamp(), stopped ? `時間で区切りました（残り ${left} 局面。同じコマンドで続きから） →` : '対象の局面は全て完了 →', OUT);
