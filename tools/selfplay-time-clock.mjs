// Increment clock: engineA engineB games initialMs incrementMs parallel [--json path] [--seed-offset N]
// エンジン同士の自動対局で強さを比べるツール
// 使い方: node tools/match.mjs <エンジンA> <エンジンB> [対局数=100] [1手のミリ秒=100] [並列数=6]
// 同じ序盤で先後を入れ替えて 2 局ずつ指し、A から見た勝敗とレート差（Elo）を出す。
import { spawn } from 'node:child_process';
import { writeFileSync, appendFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { resolve, dirname } from 'node:path';

// 指し手の判定には隣のフォルダにある将棋アリーナのルール実装を使う
const RULES = resolve(import.meta.dirname, '../../ShogiArena/src/js/shogi.js');
const { Position, START_SFEN, moveToUsi, PAWN } = await import(pathToFileURL(RULES).href);

const [pathA, pathB, gamesArg = '20', bankArg = '300000', incArg = '10000', parArg = '2', ...extraArgs] = process.argv.slice(2);
let jsonPath, seedOffset = 0;
for (let i = 0; i < extraArgs.length; i += 2) {
  const key = extraArgs[i], value = extraArgs[i + 1];
  if (value === undefined) throw new Error('Missing optional argument value');
  if (key === '--json') jsonPath = value;
  else if (key === '--seed-offset' && /^\d+$/.test(value) && Number.isSafeInteger(Number(value))) seedOffset = Number(value);
  else throw new Error('Optional arguments: --json result.json --seed-offset nonnegative_integer');
}
if (!pathA || !pathB) {
  console.log('使い方: node tools/match.mjs <エンジンA> <エンジンB> [対局数] [1手のミリ秒] [並列数]');
  process.exit(1);
}
const GAMES = +gamesArg, BANK = +bankArg, INC = +incArg, PAR = +parArg;
if (![GAMES,BANK,INC,PAR].every(Number.isSafeInteger) || GAMES<=0 || GAMES%2 || BANK<=0 || INC<0 || PAR<=0) throw Error('Invalid clock match arguments');
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
    this.lastInfo = null;
    this.iterationInfos = [];
    this.p.stdout.on('data', d => {
      this.buf += d.toString();
      let i;
      while ((i = this.buf.indexOf('\n')) >= 0) {
        const line = this.buf.slice(0, i).trim();
        if (line.startsWith('info depth') && (!line.includes(' multipv ') || line.includes(' multipv 1 '))) { this.lastInfo = line; this.iterationInfos.push(line); }
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
  async think(moves, clocks) {
    this.send('position startpos'+(moves.length ? ' moves '+moves.join(' ') : ''));
    this.lastInfo = null;
    this.iterationInfos = [];
    const waiting = this.wait('bestmove');
    const start = performance.now();
    this.send('go btime '+Math.floor(clocks[0])+' wtime '+Math.floor(clocks[1])+' binc '+INC+' winc '+INC);
    const line = await waiting;
    return {move:line.split(/\s+/)[1],elapsedMs:performance.now()-start,info:this.lastInfo,infos:[...this.iterationInfos]};
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
async function playGame(black, white, opening, gameId) {
  let pos = Position.fromSfen(START_SFEN);
  const moves = [];
  const seen = new Map();
  for (const u of opening) { pos = pos.play(pos.findMove(u)); moves.push(u); }
  const eng = [black, white];
  const clocks = [BANK,BANK], trace=[];
  eng.forEach(e => e.send('usinewgame'));
  while (true) {
    const side = pos.side;
    if (pos.legalMoves().length === 0) return [side === 0 ? 0 : 1, '詰み', trace];
    if (moves.length >= MAX_PLY) return [0.5, '手数上限', trace];
    const key = pos.key();
    const rep = (seen.get(key) || 0) + 1;
    seen.set(key, rep);
    if (rep >= 4) return [0.5, '千日手', trace];
    const before = clocks[side];
    const {move:u,elapsedMs,info,infos} = await eng[side].think(moves,clocks);
    clocks[side] -= elapsedMs;
    const row={ply:moves.length+1,side,move:u,sfen:pos.toSfen(),beforeMs:before,elapsedMs,afterMs:clocks[side],info,infos};
    trace.push(row);
    active[gameId]={ply:row.ply,side,move:u,bankBefore:before,spentMs:elapsedMs,info}; saveProgress();
    if (clocks[side]<0) return [side===0 ? 0 : 1,'時間切れ',trace];
    if (u === 'resign') return [side === 0 ? 0 : 1, '投了', trace];
    if (u === 'win') return [side === 0 ? 1 : 0, '宣言', trace];
    const m = pos.findMove(u);
    if (!m) return [side === 0 ? 0 : 1, `反則 ${u}`, trace];
    pos = pos.play(m);
    moves.push(u);
    clocks[side] += INC;
    row.afterMs = clocks[side];
    if (jsonPath) appendFileSync(jsonPath+'.moves.jsonl', JSON.stringify({game:gameId,...row})+'\n');
  }
}

const results = { w: 0, l: 0, d: 0 };
const samples = [];
const reasons = {};
let next = 0;
const t0 = Date.now();
const active={};
function saveProgress(){if(jsonPath)writeFileSync(jsonPath+'.progress.json',JSON.stringify({status:'running',targetGames:GAMES,completed:samples.filter(Boolean).length,active,elapsedSeconds:(Date.now()-t0)/1000},null,2));}

async function worker() {
  const a = new Engine(pathA), b = new Engine(pathB);
  await Promise.all([a.init(), b.init()]);
  while (true) {
    const g = next++;
    if (g >= GAMES) break;
    const opening = randomOpening(g + 1 + seedOffset);
    const aBlack = g % 2 === 0;
    const [r, why, trace] = await playGame(aBlack ? a : b, aBlack ? b : a, opening, g);
    const aScore = aBlack ? r : 1 - r;
    samples[g] = { game: g, aBlack, score: aScore, reason: why, opening, trace };
    delete active[g]; saveProgress();
    if(jsonPath)writeFileSync(jsonPath+'.game'+g+'.json',JSON.stringify(samples[g],null,2));
    if (aScore === 1) results.w++; else if (aScore === 0) results.l++; else results.d++;
    reasons[why.split(' ')[0]] = (reasons[why.split(' ')[0]] || 0) + 1;
    const n = results.w + results.l + results.d;
    process.stdout.write(`${process.stdout.isTTY ? "\r" : ""}${n}/${GAMES}  A: ${results.w}勝 ${results.l}敗 ${results.d}分  ${process.stdout.isTTY ? "" : "\n"}`);
  }
  a.quit();
  b.quit();
}

await Promise.all(Array.from({ length: PAR }, worker));
const n = results.w + results.l + results.d;
console.log('Self-play complete: '+n+' games, '+JSON.stringify(reasons));
if(jsonPath){writeFileSync(jsonPath,JSON.stringify({status:'complete',purpose:'current engine self-play collection; no strength/rating comparison',engineA:pathA,engineB:pathB,games:n,initialMs:BANK,incrementMs:INC,parallel:PAR,seedOffset,reasons,elapsedSeconds:(Date.now()-t0)/1000,samples},null,2));writeFileSync(jsonPath+'.progress.json',JSON.stringify({status:'complete',targetGames:GAMES,completed:n,active:{},elapsedSeconds:(Date.now()-t0)/1000},null,2));}
