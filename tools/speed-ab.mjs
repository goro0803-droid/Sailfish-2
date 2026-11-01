// 速さの比べ方を強くした版（実験 速さ4、11/1）：2本とも起動したままにして、局面ごとに A・B を交互に読ませ（順番も毎回入れ替える）、
// 合計の時間で比べる。speed-bench.mjs の「1本を1分 → もう1本を1分」では、その間に CPU の込み具合が変わると比が ±15% ぶれた。
// 局面は speed-bench.mjs と同じ24局面。結果（最善手・評価・深さ）が一致しない局面があれば知らせる。
// 使い方: node tools/speed-ab.mjs [--nodes 1000000] [--passes 6] <基準の exe> <候補の exe>
import { spawn } from 'node:child_process';
import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';

const args = process.argv.slice(2);
let NODES = 1000000, PASSES = 6;
const exes = [];
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--nodes') NODES = +args[++i];
  else if (args[i] === '--passes') PASSES = +args[++i];
  else exes.push(args[i]);
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

function engine(exe) {
  const p = spawn(resolve(exe), [], { cwd: dirname(resolve(exe)) });
  let buf = '', waiters = [];
  p.stdout.on('data', d => { buf += d; let i; while ((i = buf.indexOf('\n')) >= 0) { const l = buf.slice(0, i).trim(); buf = buf.slice(i + 1); waiters.forEach(w => w(l)); } });
  const send = s => p.stdin.write(s + '\n');
  const until = pred => new Promise(r => { const w = l => { if (pred(l)) { waiters = waiters.filter(x => x !== w); r(l); } }; waiters.push(w); });
  return { send, until, on: w => waiters.push(w), off: w => { waiters = waiters.filter(x => x !== w); },
    async init() {
      send('usi'); await until(l => l === 'usiok');
      for (const o of ['Threads=1', 'USI_Hash=256', 'NetworkDelay=0', 'UseBook=false']) { const [k, v] = o.split('='); send(`setoption name ${k} value ${v}`); }
      send('isready'); await until(l => l === 'readyok');
    },
    // 1局面を読ませて、かかった時間（ミリ秒）と結果を返す
    async think(pos) {
      send('usinewgame'); send('isready'); await until(l => l === 'readyok');
      send(`position ${pos}`);
      let last = '';
      const w = l => { if (l.startsWith('info') && / nodes /.test(l) && / pv /.test(l)) last = l; };
      waiters.push(w);
      const t0 = process.hrtime.bigint();
      send(`go nodes ${NODES}`);
      const bm = await until(l => l.startsWith('bestmove'));
      const ms = Number(process.hrtime.bigint() - t0) / 1e6;
      waiters = waiters.filter(x => x !== w);
      const score = (last.match(/ score (\S+ \S+)/) || [])[1];
      const depth = (last.match(/ depth (\S+)/) || [])[1];
      return { ms, res: `${bm.split(' ')[1]} ${score} ${depth}` };
    },
    quit() { send('quit'); } };
}

const [A, B] = [engine(exes[0]), engine(exes[1])];
await A.init(); await B.init();
const passRatios = [];
let sumA = 0, sumB = 0, diff = 0;
for (let pass = 0; pass < PASSES; pass++) {
  let pa = 0, pb = 0;
  for (let i = 0; i < positions.length; i++) {
    // 局面ごと・回ごとに、先に読む版を入れ替える
    const aFirst = (i + pass) % 2 === 0;
    const r1 = await (aFirst ? A : B).think(positions[i]);
    const r2 = await (aFirst ? B : A).think(positions[i]);
    const [ra, rb] = aFirst ? [r1, r2] : [r2, r1];
    pa += ra.ms; pb += rb.ms;
    if (ra.res !== rb.res) diff++;
  }
  sumA += pa; sumB += pb;
  passRatios.push(+(pa / pb).toFixed(3));
  console.log(`回 ${pass + 1}: 基準 ${(positions.length * NODES / pa).toFixed(0)}k nps、候補 ${(positions.length * NODES / pb).toFixed(0)}k nps、速さの比 ${(pa / pb).toFixed(3)}`);
}
A.quit(); B.quit();
const s = [...passRatios].sort((a, b) => a - b);
const med = (s[(s.length - 1) >> 1] + s[s.length >> 1]) / 2;
console.log(`速さの比（候補 ÷ 基準）：回ごと ${passRatios.join(' ')}、中央値 ${med.toFixed(3)}、合計で ${(sumA / sumB).toFixed(3)}  結果が違った局面 ${diff}`);
