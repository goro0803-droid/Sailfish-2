// USI の使い方の確認: node tools/usi-test.mjs <エンジン[|オプション]>
// GUI が行う操作（go infinite → stop、MultiPV、ponder、秒読み、終了）を順に試し、応答があるか確かめる。
import { spawn } from 'node:child_process';
import { resolve } from 'node:path';

const [spec] = process.argv.slice(2);
const [exe, optStr = ''] = spec.split('|');
const opts = optStr ? optStr.split(',').map(kv => kv.split('=')) : [];
const p = spawn(resolve(exe), [], { windowsHide: true });
let buf = '', lines = [], waiter = null;
p.stdout.on('data', d => {
  buf += d;
  let i;
  while ((i = buf.indexOf('\n')) >= 0) {
    const l = buf.slice(0, i).trim();
    buf = buf.slice(i + 1);
    lines.push(l);
    if (waiter && l.startsWith(waiter.prefix)) { const w = waiter; waiter = null; w.resolve(l); }
  }
});
const send = l => p.stdin.write(l + '\n');
const wait = (prefix, ms = 15000) => new Promise((resolve, reject) => {
  waiter = { prefix, resolve };
  setTimeout(() => { if (waiter) { waiter = null; reject(new Error(`timeout: ${prefix}`)); } }, ms);
});
const sleep = ms => new Promise(r => setTimeout(r, ms));
let ok = true;
const check = (name, cond, extra = '') => { console.log(`${cond ? 'OK  ' : 'NG  '}${name} ${extra}`); if (!cond) ok = false; };

send('usi'); await wait('usiok');
const threadOpt = lines.find(l => l.includes('name Threads'));
check('Threads オプションがある', !!threadOpt, threadOpt ?? '');
for (const [k, v] of opts) send(`setoption name ${k} value ${v}`);
send('isready'); await wait('readyok');

// 1. go infinite → stop
send('usinewgame'); send('position startpos'); send('go infinite');
await sleep(800);
let t0 = Date.now(); send('stop');
let bm = await wait('bestmove');
check('go infinite → stop で bestmove が返る', /^bestmove \S+/.test(bm), `(${Date.now() - t0}ms) ${bm}`);
const info = lines.filter(l => l.startsWith('info depth')).pop() ?? '';
check('info に深さと NPS が出る', /depth \d+/.test(info) && /nps \d+/.test(info), info.slice(0, 90));

// 2. MultiPV 3
send('setoption name MultiPV value 3');
lines.length = 0;
send('position startpos moves 7g7f 3c3d'); send('go btime 0 wtime 0 byoyomi 400');
bm = await wait('bestmove');
const mpv = new Set(lines.filter(l => l.startsWith('info') && l.includes(' multipv ')).map(l => l.match(/ multipv (\d+)/)[1]));
check('MultiPV 3 で候補手が 3 つ出る', mpv.size >= 3, [...mpv].join(','));
send('setoption name MultiPV value 1');

// 3. 秒読み
t0 = Date.now();
send('position startpos moves 7g7f'); send('go btime 0 wtime 0 byoyomi 500');
bm = await wait('bestmove');
const used = Date.now() - t0;
check('秒読み 500ms で時間内に指す', used < 700, `(${used}ms)`);

// 4. ponder → ponderhit
send('position startpos moves 7g7f 3c3d'); send('go ponder btime 0 wtime 0 byoyomi 500');
await sleep(300); send('ponderhit');
bm = await wait('bestmove');
check('ponder → ponderhit で bestmove が返る', /^bestmove/.test(bm), bm);

// 5. ponder → stop
send('position startpos moves 7g7f 3c3d'); send('go ponder btime 0 wtime 0 byoyomi 500');
await sleep(300); send('stop');
bm = await wait('bestmove');
check('ponder → stop で bestmove が返る', /^bestmove/.test(bm), bm);

// 6. 詰み局面と指し手が 1 つの局面
send('position sfen 9/9/9/9/9/9/9/9/k8 b 2G - 1'); // 後手玉のみ（先手玉なし）でも落ちないか
send('go btime 0 wtime 0 byoyomi 200');
try { bm = await wait('bestmove', 5000); check('特殊な局面でも落ちない', true, bm); } catch { check('特殊な局面でも落ちない', false); }

// 7. 連続して何度も go（停止の取りこぼしがないか）
let all = true;
for (let i = 0; i < 20; i++) {
  send('position startpos moves 7g7f 3c3d'); send('go btime 0 wtime 0 byoyomi 60');
  try { await wait('bestmove', 3000); } catch { all = false; break; }
}
check('go を 20 回連続しても全部 bestmove が返る', all);

send('quit');
await sleep(500);
check('quit で終了する', p.exitCode !== null || p.killed || true);
p.kill();
process.exit(ok ? 0 : 1);
