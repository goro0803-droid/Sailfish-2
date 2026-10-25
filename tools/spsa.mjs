// 探索の値を SPSA で自動調整する（実験SPSA）。tune 機能つきの Sailfish どうしで、θ＋c·Δ と θ−c·Δ を先後2局ずつ指し、
// 勝ち越した向きへ θ を寄せる。値の進め方は Stockfish の調整の慣わし（fishtest）にならう：
//   c_k = c_end·N^0.101 / k^0.101、a_k = r_end·c_end²·(A+N)^0.602 / (A+k)^0.602、θ_i += (a_k / c_k²)·c_k·Δ_i·(勝ち − 負け)
// 使い方: node tools/spsa.mjs <設定.json>
// 設定.json: { "engine": "exe", "options": {"Threads":"1",...}, "ms": 100, "workers": 8, "iterations": 15000, "out": "出力フォルダ",
//              "params": [ { "name": "T_0", "label": "RFP の幅", "init": 150, "min": 60, "max": 300, "c": 15, "r": 0.002 }, ... ] }
// 出力フォルダに state.json（いまの θ と回数、止まっても続きから）と history.csv（100回ごとの θ）と spsa.log を書く。
import { spawn } from 'node:child_process';
import { readFileSync, writeFileSync, existsSync, mkdirSync, appendFileSync, renameSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { resolve, dirname, join } from 'node:path';

const RULES = resolve(import.meta.dirname, '../../ShogiArena/src/js/shogi.js');
const { Position, START_SFEN, moveToUsi, PAWN } = await import(pathToFileURL(RULES).href);

const cfg = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const OUT = cfg.out;
mkdirSync(OUT, { recursive: true });
const STATE = join(OUT, 'state.json'), HIST = join(OUT, 'history.csv'), LOG = join(OUT, 'spsa.log');
const log = s => appendFileSync(LOG, `${new Date().toLocaleString('ja-JP')} ${s}\n`, 'utf8');
const P = cfg.params, N = cfg.iterations, A = Math.round(N * 0.1), MAX_PLY = 320;

// いまの θ（小数）と、何回目まで終えたか
let st = existsSync(STATE) ? JSON.parse(readFileSync(STATE, 'utf8'))
  : { theta: P.map(p => p.init), done: 0, w: 0, l: 0, d: 0 };
if (!existsSync(HIST)) writeFileSync(HIST, ['回数', ...P.map(p => p.name)].join(',') + '\n');
const save = () => { writeFileSync(STATE + '.tmp', JSON.stringify(st, null, 1)); renameSync(STATE + '.tmp', STATE); };

const ck = (p, k) => p.c * N ** 0.101 / k ** 0.101;
const Rk = (p, k) => (p.r * p.c ** 2 * (A + N) ** 0.602 / (A + k) ** 0.602) / ck(p, k) ** 2;
const clamp = (v, p) => Math.min(p.max, Math.max(p.min, v));

class Engine {
  constructor() {
    const path = resolve(cfg.engine);
    this.p = spawn(path, [], { windowsHide: true, cwd: dirname(path) });
    this.buf = ''; this.waiters = [];
    this.p.stdout.on('data', d => {
      this.buf += d.toString(); let i;
      while ((i = this.buf.indexOf('\n')) >= 0) {
        const line = this.buf.slice(0, i).trim(); this.buf = this.buf.slice(i + 1);
        for (const w of [...this.waiters]) if (line.startsWith(w.prefix)) { this.waiters = this.waiters.filter(x => x !== w); w.resolve(line); }
      }
    });
  }
  send(l) { this.p.stdin.write(l + '\n'); }
  wait(prefix) { return new Promise(r => this.waiters.push({ prefix, resolve: r })); }
  async init() {
    this.send('usi'); await this.wait('usiok');
    for (const [k, v] of Object.entries(cfg.options || {})) this.send(`setoption name ${k} value ${v}`);
    this.send('isready'); await this.wait('readyok');
  }
  async setParams(vals) {
    P.forEach((p, i) => this.send(`setoption name ${p.name} value ${Math.round(vals[i])}`));
    this.send('isready'); await this.wait('readyok');
  }
  async think(moves) {
    this.send(`position startpos${moves.length ? ' moves ' + moves.join(' ') : ''}`);
    this.send(`go btime 0 wtime 0 byoyomi ${cfg.ms}`);
    return (await this.wait('bestmove')).split(/\s+/)[1];
  }
  quit() { this.send('quit'); setTimeout(() => this.p.kill(), 500); }
}

/** 最初の数手をランダムな歩の突きにして、序盤を散らす（match.mjs と同じ） */
function randomOpening(seed) {
  let s = seed * 9301 + 49297;
  const rnd = n => { s = (s * 9301 + 49297) % 233280; return Math.floor((s / 233280) * n); };
  let pos = Position.fromSfen(START_SFEN); const moves = [];
  for (let i = 0; i < 4; i++) {
    const cands = pos.legalMoves().filter(m => !m.drop && pos.movingType(m) === PAWN && !pos.capturedOf(m));
    if (!cands.length) break;
    const m = cands[rnd(cands.length)]; moves.push(moveToUsi(m)); pos = pos.play(m);
  }
  return moves;
}

/** 1局指す。先手から見た結果 1 / 0 / 0.5 */
async function playGame(black, white, opening) {
  let pos = Position.fromSfen(START_SFEN); const moves = []; const seen = new Map();
  for (const u of opening) { pos = pos.play(pos.findMove(u)); moves.push(u); }
  const eng = [black, white];
  eng.forEach(e => e.send('usinewgame'));
  while (true) {
    const side = pos.side;
    if (pos.legalMoves().length === 0) return side === 0 ? 0 : 1;
    if (moves.length >= MAX_PLY) return 0.5;
    const key = pos.key(); const rep = (seen.get(key) || 0) + 1; seen.set(key, rep);
    if (rep >= 4) return 0.5;
    const u = await eng[side].think(moves);
    if (u === 'resign') return side === 0 ? 0 : 1;
    if (u === 'win') return side === 0 ? 1 : 0;
    const m = pos.findMove(u);
    if (!m) return side === 0 ? 0 : 1;
    pos = pos.play(m); moves.push(u);
  }
}

let next = st.done + 1;
log(`開始（${st.done}回目まで済み、全${N}回、${cfg.workers}組、1手${cfg.ms}ms、値 ${P.length}個）`);
async function worker() {
  const plus = new Engine(), minus = new Engine();
  await Promise.all([plus.init(), minus.init()]);
  while (true) {
    const k = next++;
    if (k > N) break;
    const delta = P.map(() => (Math.random() < 0.5 ? -1 : 1));
    const c = P.map(p => ck(p, k));
    const tp = st.theta.map((t, i) => clamp(t + c[i] * delta[i], P[i]));
    const tm = st.theta.map((t, i) => clamp(t - c[i] * delta[i], P[i]));
    await Promise.all([plus.setParams(tp), minus.setParams(tm)]);
    const op = randomOpening(k);
    const r1 = await playGame(plus, minus, op);       // θ＋ が先手
    const r2 = 1 - (await playGame(minus, plus, op)); // θ＋ が後手
    const res = (r1 + r2) - 1;                         // θ＋ の勝ち越し（−1〜+1、1局 = 0.5）
    const score = res * 2;                             // 勝ち − 負け（−2〜+2）
    st.theta = st.theta.map((t, i) => clamp(t + Rk(P[i], k) * c[i] * delta[i] * score, P[i]));
    for (const r of [r1, r2]) { if (r === 1) st.w++; else if (r === 0) st.l++; else st.d++; }
    st.done = Math.max(st.done, k);
    if (k % 20 === 0) save();
    if (k % 100 === 0) {
      appendFileSync(HIST, [k, ...st.theta.map(v => v.toFixed(2))].join(',') + '\n');
      log(`${k}回：${P.map((p, i) => `${p.label} ${st.theta[i].toFixed(1)}`).join('、')}`);
    }
  }
  plus.quit(); minus.quit();
}
await Promise.all(Array.from({ length: cfg.workers }, worker));
save();
log(`終了：${P.map((p, i) => `${p.name}=${Math.round(st.theta[i])}`).join(',')}`);
