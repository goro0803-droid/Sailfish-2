// 局面1つを、標準版と旧版・深さごと・時間ごと・評価関数だけ、の4つの面から調べる診断ツール（Threads=1、優先度は低め）
// 使い方: node tools/diagnose-position.mjs <sfen> <実戦の手> <今の標準版の手> <出力JSON> [旧版exe]
// A 版の比較   : 旧版と標準版で 10/18/30 秒の最善手と評価
// B 深さごと   : 標準版で MultiPV=8、深さ 1..16。実戦の手と今の手の評価・順位の推移
// C 時間ごと   : 標準版で MultiPV=8、10/30/60 秒
// D 評価関数だけ: 候補手を指した後の局面の静的評価（eval）。200msの浅い探索も併記
// 評価はエンジン自身の値（手番側から見たcp）。「指した後」の値は手番が変わるので、自分から見るには符号を反転する。
import { resolve, dirname } from 'node:path';
import { writeFileSync, mkdirSync, existsSync, readFileSync } from 'node:fs';
import { Usi } from './usi-engine.mjs';

// --phases=A,B,C,D  実行する面（既定は全部）。--depth-from=15  B の開始深さ。出力JSONが既にあれば読み込んで続きに足す（時間制限で区切って再実行するため）
const flag = n => (process.argv.find(a => a.startsWith(`--${n}=`)) || '').split('=')[1];
const PHASES = new Set((flag('phases') || 'A,B,C,D').split(','));
const DEPTH_FROM = +(flag('depth-from') || 1);
const [sfen, played, alt, outArg, oldArg] = process.argv.slice(2).filter(a => !a.startsWith('--'));
if (!sfen || !played || !alt || !outArg) { console.log('使い方: node tools/diagnose-position.mjs <sfen> <実戦の手> <今の手> <出力JSON> [旧版exe]'); process.exit(1); }
const OUT = resolve(outArg);
const NEW = resolve('baseline/sailfish2_time_candidate.exe');
const OLD = resolve(oldArg || 'baseline/sailfish2_before_time.exe');
const pos = `sfen ${sfen}`;
const stamp = () => new Date().toLocaleTimeString('ja-JP');
const res = existsSync(OUT) ? JSON.parse(readFileSync(OUT, 'utf8')) : { sfen, played, alt, A: {}, B: [], C: {}, D: {} };
const save = () => { mkdirSync(dirname(OUT), { recursive: true }); writeFileSync(OUT, JSON.stringify(res, null, 1)); };
const find = (rows, mv) => rows.find(r => r.move === mv) || null;

// A 版の比較
if (PHASES.has('A')) for (const [name, exe] of [['旧版', OLD], ['標準版', NEW]]) {
  const e = new Usi(exe); await e.init();
  res.A[name] = {};
  for (const ms of [10000, 18000, 30000]) {
    const s = await e.search(pos, ms);
    res.A[name][ms / 1000 + '秒'] = { best: s.best, score: s.score, depth: s.depth };
    console.log(stamp(), `A ${name} ${ms / 1000}秒`, s.best, s.score, `d${s.depth}`);
  }
  e.quit();
  save();
}

const e = new Usi(NEW); await e.init();
e.setOption('MultiPV', 8);

// B 深さごと
if (PHASES.has('B')) for (let d = DEPTH_FROM; d <= 16; d++) {
  const rows = Usi.multipv(await e.goRaw(pos, `depth ${d}`));
  const p = find(rows, played), a = find(rows, alt);
  res.B.push({ depth: d, top: rows.slice(0, 3).map(r => `${r.move}:${r.score}`), played: p && { rank: p.rank, score: p.score }, alt: a && { rank: a.rank, score: a.score } });
  console.log(stamp(), `B 深さ${d}`, rows.slice(0, 3).map(r => `${r.move}:${r.score}`).join(' '), '| 実戦', p ? `${p.rank}位 ${p.score}` : '圏外', '| 今の手', a ? `${a.rank}位 ${a.score}` : '圏外');
}
save();

// C 時間ごと
if (PHASES.has('C')) for (const ms of [10000, 30000, 60000]) {
  const rows = Usi.multipv(await e.goRaw(pos, `movetime ${ms}`));
  res.C[ms / 1000 + '秒'] = rows.map(r => ({ rank: r.rank, move: r.move, score: r.score, depth: r.depth }));
  console.log(stamp(), `C ${ms / 1000}秒`, rows.map(r => `${r.rank}.${r.move}:${r.score}`).join(' '));
}
save();

// D 評価関数だけ
e.setOption('MultiPV', 1);
const cands = [...new Set([played, alt, ...((res.C['10秒'] || []).map(r => r.move))])];
if (PHASES.has('D')) for (const mv of cands) {
  const p2 = `${pos} moves ${mv}`;
  const st = await e.staticEval(p2);
  const d1 = await e.search(p2, 200); // 指した後の局面を浅く読む（相手番の評価）
  res.D[mv] = { staticAfterSideToMove: st, searchAfterSideToMove: d1.score };
  console.log(stamp(), `D ${mv}`, '静的評価(相手番)', st, '/ 200ms探索(相手番)', d1.score);
}
e.quit();
save();
console.log(stamp(), '完了 →', OUT);
