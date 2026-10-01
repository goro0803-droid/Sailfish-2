// 速度の計測: node tools/bench.mjs <エンジン[|オプション=値,...]> [深さ=10]
// いくつかの局面を決まった深さまで読み、合計ノード数・所要時間・NPS を出す。
import { spawn } from 'node:child_process';

const [spec, depthArg = '10'] = process.argv.slice(2);
const [exe, optStr = ''] = spec.split('|');
const opts = optStr ? optStr.split(',').map(kv => kv.split('=')) : [];
const POSITIONS = [
  'startpos',
  'startpos moves 7g7f 3c3d 2g2f 8c8d 2f2e 8d8e 6i7h 4a3b 2e2d 2c2d 2h2d',
  'startpos moves 7g7f 8c8d 7i6h 3c3d 6h7g 7a6b 5g5f 5c5d 3i4h 3a4b 4i5h 4a3b',
];

const p = spawn(exe, [], { windowsHide: true });
let buf = '', waiter = null, lastInfo = '';
p.stdout.on('data', d => {
  buf += d;
  let i;
  while ((i = buf.indexOf('\n')) >= 0) {
    const line = buf.slice(0, i).trim();
    buf = buf.slice(i + 1);
    if (line.startsWith('info') && line.includes(' nodes ')) lastInfo = line;
    if (waiter && line.startsWith(waiter.prefix)) { const w = waiter; waiter = null; w.resolve(line); }
  }
});
const send = l => p.stdin.write(l + '\n');
const wait = prefix => new Promise(resolve => { waiter = { prefix, resolve }; });

send('usi'); await wait('usiok');
for (const [k, v] of opts) send(`setoption name ${k} value ${v}`);
send('isready'); await wait('readyok');
let nodes = 0;
const t0 = Date.now();
for (const pos of POSITIONS) {
  send('usinewgame');
  send(`position ${pos}`);
  send(`go depth ${depthArg}`);
  await wait('bestmove');
  nodes += +(lastInfo.match(/ nodes (\d+)/)?.[1] ?? 0);
}
const ms = Date.now() - t0;
console.log(`${exe}: nodes ${nodes}  time ${ms}ms  nps ${Math.round(nodes / ms * 1000)}`);
send('quit');
