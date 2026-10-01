// エンジン同士の自動対局で強さを比べるツール
// 使い方: node tools/match.mjs <エンジンA> <エンジンB> [対局数=100] [1手のミリ秒=100] [並列数=6]
// 同じ序盤で先後を入れ替えて 2 局ずつ指し、A から見た勝敗とレート差（Elo）を出す。
import { spawn } from 'node:child_process';
import { pathToFileURL } from 'node:url';
import { resolve, dirname } from 'node:path';

// 指し手の判定には隣のフォルダにある将棋アリーナのルール実装を使う
const RULES = resolve(import.meta.dirname, '../../ShogiArena/src/js/shogi.js');
const { Position, START_SFEN, moveToUsi, PAWN } = await import(pathToFileURL(RULES).href);

const [pathA, pathB, gamesArg = '100', msArg = '100', parArg = '6'] = process.argv.slice(2);
if (!pathA || !pathB) {
  console.log('使い方: node tools/match.mjs <エンジンA> <エンジンB> [対局数] [1手のミリ秒] [並列数]');
  process.exit(1);
}
const GAMES = +gamesArg, MS = +msArg, PAR = +parArg;
const MAX_PLY = 320;

// エンジンの指定は「パス|オプション=値,オプション=値」の形でもよい（例: suisho.exe|Threads=1,NodesLimit=3000）
class Engine {
  constructor(spec) {
    const [rel, opts = ''] = spec.split('|');
    const path = resolve(rel);
    this.opts = opts ? opts.split(',').map(kv => kv.split('=')) : [];
    // 評価関数ファイルなどを探せるよう、エンジンのフォルダで起動する
    this.p = spawn(path, [], { windowsHide: true, cwd: dirname(path) });
    this.buf = '';
    this.waiters = [];
    this.p.stdout.on('data', d => {
      this.buf += d.toString();
      let i;
      while ((i = this.buf.indexOf('\n')) >= 0) {
        const line = this.buf.slice(0, i).trim();
        this.buf = this.buf.slice(i + 1);
        for (const w of [...this.waiters]) {
          if (line.startsWith(w.prefix)) {
            this.waiters = this.waiters.filter(x => x !== w);
            w.resolve(line);
          }
        }
      }
    });
  }
  send(l) { this.p.stdin.write(l + '\n'); }
  wait(prefix) { return new Promise(resolve => this.waiters.push({ prefix, resolve })); }
  async init() {
    this.send('usi');
    await this.wait('usiok');
    this.send('setoption name USI_Hash value 64');
    for (const [k, v] of this.opts) this.send(`setoption name ${k} value ${v}`);
    this.send('isready');
    await this.wait('readyok');
  }
  async think(moves) {
    this.send(`position startpos${moves.length ? ' moves ' + moves.join(' ') : ''}`);
    this.send(`go btime 0 wtime 0 byoyomi ${MS}`);
    const line = await this.wait('bestmove');
    return line.split(/\s+/)[1];
  }
  quit() { this.send('quit'); setTimeout(() => this.p.kill(), 500); }
}

/** 最初の数手をランダムな歩の突きにして、序盤を散らす */
function randomOpening(seed) {
  let s = seed * 9301 + 49297;
  const rnd = n => { s = (s * 9301 + 49297) % 233280; return Math.floor((s / 233280) * n); };
  let pos = Position.fromSfen(START_SFEN);
  const moves = [];
  for (let i = 0; i < 4; i++) {
    const cands = pos.legalMoves().filter(m => !m.drop && pos.movingType(m) === PAWN && !pos.capturedOf(m));
    if (!cands.length) break;
    const m = cands[rnd(cands.length)];
    moves.push(moveToUsi(m));
    pos = pos.play(m);
  }
  return moves;
}

/** 1 局指す。戻り値は先手から見た結果 1 / 0 / 0.5 と理由 */
async function playGame(black, white, opening) {
  let pos = Position.fromSfen(START_SFEN);
  const moves = [];
  const seen = new Map();
  for (const u of opening) { pos = pos.play(pos.findMove(u)); moves.push(u); }
  const eng = [black, white];
  eng.forEach(e => e.send('usinewgame'));
  while (true) {
    const side = pos.side;
    if (pos.legalMoves().length === 0) return [side === 0 ? 0 : 1, '詰み'];
    if (moves.length >= MAX_PLY) return [0.5, '手数上限'];
    const key = pos.key();
    const rep = (seen.get(key) || 0) + 1;
    seen.set(key, rep);
    if (rep >= 4) return [0.5, '千日手'];
    const u = await eng[side].think(moves);
    if (u === 'resign') return [side === 0 ? 0 : 1, '投了'];
    if (u === 'win') return [side === 0 ? 1 : 0, '宣言'];
    const m = pos.findMove(u);
    if (!m) return [side === 0 ? 0 : 1, `反則 ${u}`];
    pos = pos.play(m);
    moves.push(u);
  }
}

const results = { w: 0, l: 0, d: 0 };
const reasons = {};
let next = 0;
const t0 = Date.now();

async function worker() {
  const a = new Engine(pathA), b = new Engine(pathB);
  await Promise.all([a.init(), b.init()]);
  while (true) {
    const g = next++;
    if (g >= GAMES) break;
    const opening = randomOpening(Math.floor(g / 2) + 1);
    const aBlack = g % 2 === 0;
    const [r, why] = await playGame(aBlack ? a : b, aBlack ? b : a, opening);
    const aScore = aBlack ? r : 1 - r;
    if (aScore === 1) results.w++; else if (aScore === 0) results.l++; else results.d++;
    reasons[why.split(' ')[0]] = (reasons[why.split(' ')[0]] || 0) + 1;
    const n = results.w + results.l + results.d;
    process.stdout.write(`\r${n}/${GAMES}  A: ${results.w}勝 ${results.l}敗 ${results.d}分  `);
  }
  a.quit();
  b.quit();
}

await Promise.all(Array.from({ length: PAR }, worker));
const n = results.w + results.l + results.d;
const p = (results.w + results.d / 2) / n;
const elo = p > 0 && p < 1 ? -400 * Math.log10(1 / p - 1) : p >= 1 ? Infinity : -Infinity;
// 勝率の標準誤差から 95% 区間のおおよその幅
const se = Math.sqrt((p * (1 - p)) / n);
const lo = -400 * Math.log10(1 / Math.min(0.999, Math.max(0.001, p - 1.96 * se)) - 1);
const hi = -400 * Math.log10(1 / Math.min(0.999, Math.max(0.001, p + 1.96 * se)) - 1);
console.log(`\n\n対局数 ${n}（1手 ${MS}ms）  A: ${results.w}勝 ${results.l}敗 ${results.d}分  勝率 ${(p * 100).toFixed(1)}%`);
console.log(`レート差（A - B）: ${elo.toFixed(0)}  （95%の目安 ${lo.toFixed(0)} 〜 ${hi.toFixed(0)}）`);
console.log(`終局理由: ${JSON.stringify(reasons)}  所要 ${Math.round((Date.now() - t0) / 1000)}秒`);
