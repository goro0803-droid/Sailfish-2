// 見逃し局面の候補を、現在の標準版（Threads=1）で再解析する。floodgate の対局に影響しないよう優先度を下げて動かす。
// 使い方: node tools/reanalyze-missed.mjs [出力JSON=baseline/missed_candidates_20261004/reanalysis.json]
// 各局面: 15秒・45秒で最善手と評価を取る。最善手が実戦の手と違えば、実戦の手の後と最善手の後を30秒ずつ比べる。
// 評価はSailfish自身の値（手番側から見た cp、詰みは ±30000 相当）。強い別ソフトによる正解認定ではない。
import { spawn } from 'node:child_process';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';
import os from 'node:os';

const OUT = resolve(process.argv[2] || 'baseline/missed_candidates_20261004/reanalysis.json');
const EXE = resolve('baseline/sailfish2_time_candidate.exe'); // 標準exeと同じSHA-256の凍結コピー
const TARGETS = [[5, 20], [5, 34], [13, 41], [13, 43], [10, 54], [12, 52], [4, 133]]; // [局, 手数]
const cands = JSON.parse(readFileSync('baseline/missed_candidates_20261004/floodgate_candidates.json', 'utf8'));

class Usi {
  constructor() {
    this.p = spawn(EXE, [], { windowsHide: true });
    try { os.setPriority(this.p.pid, os.constants.priority.PRIORITY_BELOW_NORMAL); } catch {}
    this.buf = ''; this.lines = []; this.w = null;
    this.p.stdout.on('data', d => {
      this.buf += d.toString();
      let i;
      while ((i = this.buf.indexOf('\n')) >= 0) { this.lines.push(this.buf.slice(0, i).trim()); this.buf = this.buf.slice(i + 1); }
      if (this.w) this.w();
    });
  }
  send(s) { this.p.stdin.write(s + '\n'); }
  async until(pred) {
    for (;;) {
      const i = this.lines.findIndex(pred);
      if (i >= 0) { const got = this.lines.splice(0, i + 1); return got; }
      await new Promise(r => (this.w = r));
    }
  }
  async init() {
    this.send('usi'); await this.until(l => l === 'usiok');
    for (const o of ['Threads=1', 'USI_Hash=64', 'NetworkDelay=0']) { const [k, v] = o.split('='); this.send(`setoption name ${k} value ${v}`); }
    this.send('isready'); await this.until(l => l === 'readyok');
  }
  // position は "sfen ... moves ..." または "startpos moves ..." の文字列
  async search(position, ms) {
    this.send('usinewgame'); this.send('isready'); await this.until(l => l === 'readyok'); // 置換表をリセットして条件をそろえる
    this.send(`position ${position}`); this.send(`go movetime ${ms}`);
    const got = await this.until(l => l.startsWith('bestmove'));
    const best = got.at(-1).split(' ')[1];
    let score = null, depth = null, pv = '';
    for (const l of got) {
      const m = l.match(/^info .*?depth (\d+).*? score (cp|mate) (-?\d+).*? pv (.*)$/);
      if (m) { depth = +m[1]; score = m[2] === 'cp' ? +m[3] : (+m[3] > 0 ? 30000 - +m[3] : -30000 - +m[3]); pv = m[4].split(' ').slice(0, 8).join(' '); }
    }
    return { best, score, depth, pv };
  }
  quit() { this.send('quit'); }
}

const result = existsSync(OUT) ? JSON.parse(readFileSync(OUT, 'utf8')) : {};
const e = new Usi(); await e.init();
const stamp = () => new Date().toLocaleTimeString('ja-JP');

for (const [game, ply] of TARGETS) {
  const key = `局${game}-${ply}手目`;
  if (result[key]) { console.log(stamp(), key, '（保存済みのためスキップ）'); continue; }
  const c = cands.find(x => x.game === game && x.ply === ply);
  if (!c) { console.log(key, '候補なし'); continue; }
  const pos = `sfen ${c.sfen}`;
  const r = { ...c };
  console.log(stamp(), key, '実戦の手', c.played, '開始');
  r.t15 = await e.search(pos, 15000);
  r.t45 = await e.search(pos, 45000);
  console.log(stamp(), key, '15秒', r.t15.best, r.t15.score, '/ 45秒', r.t45.best, r.t45.score);
  // 指した後の局面は相手番なので、自分から見た評価は符号を反転する
  const after = async (mv) => { const s = await e.search(`${pos} moves ${mv}`, 30000); return { move: mv, ownScore: s.score === null ? null : -s.score, depth: s.depth, reply: s.best }; };
  r.afterPlayed = await after(c.played);
  const alts = [...new Set([r.t15.best, r.t45.best])].filter(m => m && m !== c.played);
  r.afterAlternatives = [];
  for (const m of alts) r.afterAlternatives.push(await after(m));
  result[key] = r;
  writeFileSync(OUT, JSON.stringify(result, null, 1));
  console.log(stamp(), key, '完了 実戦の手の後', r.afterPlayed.ownScore, '代案の後', r.afterAlternatives.map(a => `${a.move}:${a.ownScore}`).join(' ') || '（代案なし）');
}
e.quit();
console.log(stamp(), '全て完了 →', OUT);
