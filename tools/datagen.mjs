// 評価関数の学習用に、エンジン同士の自己対戦の棋譜を作るツール
// 使い方: node tools/datagen.mjs <エンジン[|オプション=値,...]> <出力ファイル> [対局数=1000] [1手のミリ秒=40] [並列数=8]
//
// 1 局 1 行で追記する:  R <先手から見た結果 1/0.5/0> <序盤で乱した手数> <指し手>:<手番側から見た評価値> ...
// 序盤は MultiPV の候補からランダムに選んで局面を散らし、そのあとは通常どおり指す。
import { spawn } from 'node:child_process';
import { appendFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { resolve, dirname } from 'node:path';

const RULES = resolve(import.meta.dirname, '../../ShogiArena/src/js/shogi.js');
const { Position, START_SFEN, moveToUsi } = await import(pathToFileURL(RULES).href);

const [spec, outFile, gamesArg = '1000', msArg = '40', parArg = '8'] = process.argv.slice(2);
if (!spec || !outFile) {
  console.log('使い方: node tools/datagen.mjs <エンジン> <出力ファイル> [対局数] [1手のミリ秒] [並列数]');
  process.exit(1);
}
const GAMES = +gamesArg, MS = +msArg, PAR = +parArg;
const MAX_PLY = 300;
const MATE_CP = 30000;

class Engine {
  constructor(spec) {
    const [rel, opts = ''] = spec.split('|');
    const path = resolve(rel);
    this.opts = opts ? opts.split(',').map(kv => kv.split('=')) : [];
    this.p = spawn(path, [], { windowsHide: true, cwd: dirname(path) });
    this.buf = '';
    this.waiter = null;
    this.infos = [];
    this.p.stdout.on('data', d => {
      this.buf += d.toString();
      let i;
      while ((i = this.buf.indexOf('\n')) >= 0) {
        const line = this.buf.slice(0, i).trim();
        this.buf = this.buf.slice(i + 1);
        if (line.startsWith('info ')) this.infos.push(line);
        if (this.waiter && line.startsWith(this.waiter.prefix)) { const w = this.waiter; this.waiter = null; w.resolve(line); }
      }
    });
  }
  send(l) { this.p.stdin.write(l + '\n'); }
  wait(prefix) { return new Promise(resolve => { this.waiter = { prefix, resolve }; }); }
  async init() {
    this.send('usi'); await this.wait('usiok');
    this.send('setoption name USI_Hash value 32');
    this.send('setoption name NetworkDelay value 0');
    for (const [k, v] of this.opts) this.send(`setoption name ${k} value ${v}`);
    this.send('isready'); await this.wait('readyok');
  }
  /** 思考して { move, score(手番側のcp), cands: [{move, score}] } を返す */
  async think(moves, ms, multipv) {
    this.send(`setoption name MultiPV value ${multipv}`);
    this.send(`position startpos${moves.length ? ' moves ' + moves.join(' ') : ''}`);
    this.infos = [];
    this.send(`go btime 0 wtime 0 byoyomi ${ms}`);
    const line = await this.wait('bestmove');
    const move = line.split(/\s+/)[1];
    // 各 multipv の最後の info から評価値と先頭の手を取る
    const byK = new Map();
    for (const l of this.infos) {
      const k = +(l.match(/ multipv (\d+)/)?.[1] ?? 1);
      const sc = l.match(/ score (cp|mate) (-?\d+)/);
      const pv = l.match(/ pv (\S+)/);
      if (!sc || !pv) continue;
      const v = sc[1] === 'cp' ? +sc[2] : (+sc[2] > 0 ? MATE_CP - +sc[2] : -MATE_CP - +sc[2]);
      byK.set(k, { move: pv[1], score: v });
    }
    const cands = [...byK.entries()].sort((a, b) => a[0] - b[0]).map(e => e[1]);
    return { move, score: cands[0]?.score ?? 0, cands };
  }
  quit() { this.send('quit'); setTimeout(() => this.p.kill(), 500); }
}

let seed = (Date.now() % 100000) + 1;
const rnd = n => { seed = (seed * 48271) % 2147483647; return seed % n; };

async function playGame(eng) {
  eng.send('usinewgame');
  let pos = Position.fromSfen(START_SFEN);
  const moves = [], scores = [];
  const seen = new Map();
  const openPlies = 8 + rnd(5);
  let streak = 0; // 大差が続いた手数
  let result = 0.5;
  while (true) {
    if (pos.legalMoves().length === 0) { result = pos.side === 0 ? 0 : 1; break; }
    if (moves.length >= MAX_PLY) { result = 0.5; break; }
    const key = pos.key();
    const rep = (seen.get(key) || 0) + 1;
    seen.set(key, rep);
    if (rep >= 4) { result = 0.5; break; }
    const opening = moves.length < openPlies;
    const r = await eng.think(moves, opening ? 25 : MS, opening ? 6 : 1);
    let u = r.move, score = r.score;
    if (u === 'resign') { result = pos.side === 0 ? 0 : 1; break; }
    if (u === 'win') { result = pos.side === 0 ? 1 : 0; break; }
    if (opening) {
      // 最善手に近い候補（100cp 以内、詰みでないもの）からランダムに選ぶ
      const ok = r.cands.filter(c => Math.abs(c.score) < 20000 && c.score >= r.score - 100);
      if (ok.length) { const c = ok[rnd(ok.length)]; u = c.move; score = c.score; }
    }
    const m = pos.findMove(u);
    if (!m) { result = pos.side === 0 ? 0 : 1; break; }
    const mover = pos.side;
    scores.push(score);
    moves.push(u);
    pos = pos.play(m);
    // 大差の判定: 先手から見た評価値が ±3000 を超える状態が 8 手続いたら、その側の勝ちとする
    const blackScore = mover === 0 ? score : -score;
    if (!opening && Math.abs(blackScore) >= 3000 && (streak === 0 || Math.sign(streak) === Math.sign(blackScore))) {
      streak += Math.sign(blackScore);
    } else {
      streak = 0;
    }
    if (Math.abs(streak) >= 8) { result = streak > 0 ? 1 : 0; break; }
  }
  const body = moves.map((u, i) => `${u}:${scores[i]}`).join(' ');
  return { line: `R ${result} ${openPlies} ${body}`, plies: moves.length, result };
}

let next = 0, done = 0, black = 0, white = 0, draw = 0, plies = 0;
const t0 = Date.now();
async function worker() {
  const eng = new Engine(spec);
  await eng.init();
  while (next < GAMES) {
    next++;
    const g = await playGame(eng);
    appendFileSync(outFile, g.line + '\n');
    done++; plies += g.plies;
    if (g.result === 1) black++; else if (g.result === 0) white++; else draw++;
    if (done % 20 === 0 || done === GAMES) {
      const sec = Math.round((Date.now() - t0) / 1000);
      process.stdout.write(`\r${done}/${GAMES} 局  先手${black} 後手${white} 分${draw}  平均${Math.round(plies / done)}手  ${sec}秒   `);
    }
  }
  eng.quit();
}
await Promise.all(Array.from({ length: PAR }, worker));
console.log(`\n完了: ${done} 局 → ${outFile}`);
