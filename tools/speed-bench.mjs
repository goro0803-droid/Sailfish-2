// 速さの改良の確かめ（実験 V）：同じ局面を決まった局面数だけ読ませ、結果（最善手・評価・局面数）が一致するかと、速さ（nps）を比べる。
// 局面：floodgate の Sailfish の棋譜（D:/shogi_data/floodgate_sailfish_202610 の 10/22〜23）の40・70手目、24局面（診断 diagTree と同じ）。
// 使い方: node tools/speed-bench.mjs [--nodes 2000000] [--repeat 3] "<exe>|オプション=値,..." ["<exe2>|..." ...]
// 1つ目を基準にして、2つ目以降の結果が一致するかを調べる。速さは、繰り返しのうち最も速い回の nps。
import { spawn } from 'node:child_process';
import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';

const args = process.argv.slice(2);
let NODES = 2000000, REPEAT = 3;
const specs = [];
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--nodes') NODES = +args[++i];
  else if (args[i] === '--repeat') REPEAT = +args[++i];
  else specs.push(args[i]);
}
const FG = 'D:/shogi_data/floodgate_sailfish_202610';
const KIND = { FU: 'P', KY: 'L', KE: 'N', GI: 'S', KI: 'G', KA: 'B', HI: 'R', OU: 'K', TO: '+P', NY: '+L', NK: '+N', NG: '+S', UM: '+B', RY: '+R' };
function csaToUsi(text) {
  const board = {};
  const back = ['KY', 'KE', 'GI', 'KI', 'OU', 'KI', 'GI', 'KE', 'KY'];
  for (let f = 1; f <= 9; f++) { board[`${f}1`] = back[9 - f]; board[`${f}9`] = back[9 - f]; board[`${f}3`] = 'FU'; board[`${f}7`] = 'FU'; }
  board['82'] = 'HI'; board['22'] = 'KA'; board['28'] = 'HI'; board['88'] = 'KA';
  const moves = [];
  for (const m of text.matchAll(/^[+-](\d\d)(\d\d)([A-Z]{2})$/gm)) {
    const [, from, to, pc] = m;
    const sq = s => `${s[0]}${'abcdefghi'[+s[1] - 1]}`;
    if (from === '00') moves.push(`${KIND[pc]}*${sq(to)}`);
    else { const promo = !KIND[board[from]].startsWith('+') && KIND[pc].startsWith('+'); moves.push(`${sq(from)}${sq(to)}${promo ? '+' : ''}`); delete board[from]; }
    board[to] = pc;
  }
  return moves;
}
const files = readdirSync(FG).filter(f => /\+(2026102[23]\d{6})\.csa$/.test(f)).sort().slice(-36);
const positions = [];
for (const f of files) {
  const mv = csaToUsi(readFileSync(join(FG, f), 'utf8'));
  for (const ply of [40, 70]) if (mv.length > ply + 2) positions.push(`startpos moves ${mv.slice(0, ply).join(' ')}`);
  if (positions.length >= 24) break;
}

function engine(spec) {
  const [exe, opts = ''] = spec.split('|');
  const p = spawn(resolve(exe), [], { cwd: dirname(resolve(exe)) });
  let buf = '', waiters = [];
  p.stdout.on('data', d => { buf += d; let i; while ((i = buf.indexOf('\n')) >= 0) { const l = buf.slice(0, i).trim(); buf = buf.slice(i + 1); waiters.forEach(w => w(l)); } });
  const send = s => p.stdin.write(s + '\n');
  const until = pred => new Promise(r => { const w = l => { if (pred(l)) { waiters = waiters.filter(x => x !== w); r(l); } }; waiters.push(w); });
  return { send, until, on: w => waiters.push(w), off: w => { waiters = waiters.filter(x => x !== w); },
    async init() {
      send('usi'); await until(l => l === 'usiok');
      for (const o of ['Threads=1', 'USI_Hash=256', 'NetworkDelay=0', 'UseBook=false', ...(opts ? opts.split(',') : [])]) { const [k, v] = o.split('='); send(`setoption name ${k} value ${v}`); }
      send('isready'); await until(l => l === 'readyok');
    } };
}

async function run(spec, reps = REPEAT) {
  const runs = [];
  for (let r = 0; r < reps; r++) {
    // 履歴の表が前の回から引き継がれないよう、回ごとに起動し直す
    const e = engine(spec);
    await e.init();
    const res = []; let totalNodes = 0, totalMs = 0;
    for (const pos of positions) {
      // 置換表を消す時間を測らないよう、消し終わるのを待ってから読ませる
      e.send('usinewgame'); e.send('isready'); await e.until(l => l === 'readyok');
      e.send(`position ${pos}`);
      let last = '';
      const w = l => { if (l.startsWith('info') && / nodes /.test(l) && / pv /.test(l)) last = l; };
      e.on(w);
      const t0 = process.hrtime.bigint();
      e.send(`go nodes ${NODES}`);
      const bm = await e.until(l => l.startsWith('bestmove'));
      const ms = Number(process.hrtime.bigint() - t0) / 1e6;
      e.off(w);
      const g = k => (last.match(new RegExp(` ${k} (\\S+)`)) || [])[1];
      const score = (last.match(/ score (\S+ \S+)/) || [])[1];
      res.push(`${bm.split(' ')[1]} ${score} ${g('depth')}`);
      totalNodes += NODES; totalMs += ms;
    }
    runs.push({ res, nps: totalNodes / (totalMs / 1000) });
    e.send('quit');
  }
  return runs;
}

// CPU の速さの揺れ（ほかの処理、熱など）を打ち消すため、版を1回ずつ交互に読ませる。
// 読ませる順番でも数%変わるので、回ごとに始める版をずらす（繰り返しを版の数の倍数にすると、どの版も同じ回数ずつ各順番になる）
const all = specs.map(() => []);
for (let r = 0; r < REPEAT; r++)
  for (let j = 0; j < specs.length; j++) { const i = (j + r) % specs.length; all[i].push(...(await run(specs[i], 1))); }
const out = specs.map((s, i) => {
  const runs = all[i];
  const stable = runs.every(r => r.res.join('|') === runs[0].res.join('|'));
  return { s, res: runs[0].res, nps: Math.max(...runs.map(r => r.nps)), stable };
});
const base = out[0];
console.log(`局面 ${positions.length}、各 ${NODES} 局面、${REPEAT}回のうち最も速い回`);
for (const o of out) {
  const same = o.res.filter((r, i) => r === base.res[i]).length;
  console.log(`${(o.nps / 1000).toFixed(0)}k nps（基準の ${(o.nps / base.nps).toFixed(3)} 倍）  結果の一致 ${same}/${positions.length}${o.stable ? '' : '（繰り返しで結果が揺れた）'}  ${o.s}`);
}
