// USI エンジンを1スレッド・低優先度で動かして、局面を指定時間だけ解析する小さな部品（再解析ツールが共有する）
// 評価はエンジン自身の値（手番側から見た cp。詰みは ±30000 相当）。
import { spawn } from 'node:child_process';
import os from 'node:os';

export class Usi {
  constructor(exe) {
    this.p = spawn(exe, [], { windowsHide: true });
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
      if (i >= 0) return this.lines.splice(0, i + 1);
      await new Promise(r => (this.w = r));
    }
  }
  async init() {
    this.send('usi'); await this.until(l => l === 'usiok');
    for (const [k, v] of [['Threads', 1], ['USI_Hash', 64], ['NetworkDelay', 0]]) this.send(`setoption name ${k} value ${v}`);
    this.send('isready'); await this.until(l => l === 'readyok');
  }
  // position: "sfen ... [moves ...]" または "startpos [moves ...]"
  async search(position, ms) {
    this.send('usinewgame'); this.send('isready'); await this.until(l => l === 'readyok'); // 置換表をリセットして条件をそろえる
    this.send(`position ${position}`); this.send(`go movetime ${ms}`);
    const got = await this.until(l => l.startsWith('bestmove'));
    const best = got.at(-1).split(' ')[1];
    let score = null, depth = null, pv = [];
    for (const l of got) {
      const m = l.match(/^info .*?depth (\d+).*? score (cp|mate) (-?\d+).*? pv (.*)$/);
      if (m) { depth = +m[1]; score = m[2] === 'cp' ? +m[3] : (+m[3] > 0 ? 30000 - +m[3] : -30000 - +m[3]); pv = m[4].split(' ').slice(0, 8); }
    }
    return { best, score, depth, pv };
  }
  setOption(name, value) { this.send(`setoption name ${name} value ${value}`); }
  // go の引数（例: "movetime 10000" / "depth 8"）で探索し、info 行をすべて返す。MultiPV の読み取り用
  async goRaw(position, goArgs) {
    this.send('usinewgame'); this.send('isready'); await this.until(l => l === 'readyok');
    this.send(`position ${position}`); this.send(`go ${goArgs}`);
    return await this.until(l => l.startsWith('bestmove'));
  }
  // goRaw の出力から、最後の深さの MultiPV ごとの（手, 評価, 深さ）を取り出す
  static multipv(lines) {
    const last = {};
    for (const l of lines) {
      const m = l.match(/^info .*?depth (\d+).*?multipv (\d+) score (cp|mate) (-?\d+).*? pv (\S+)/) || l.match(/^info .*?depth (\d+).*? score (cp|mate) (-?\d+).*? pv (\S+)/);
      if (!m) continue;
      const hasMpv = m.length === 6;
      const [d, k, t, v, mv] = hasMpv ? [m[1], m[2], m[3], m[4], m[5]] : [m[1], '1', m[2], m[3], m[4]];
      last[k] = { move: mv, score: t === 'cp' ? +v : (+v > 0 ? 30000 - +v : -30000 - +v), depth: +d };
    }
    return Object.keys(last).sort((a, b) => a - b).map(k => ({ rank: +k, ...last[k] }));
  }
  // 局面を設定して eval コマンドの出力（手番側から見た値と推定）を返す
  async staticEval(position) {
    this.send(`position ${position}`); this.send('eval');
    const got = await this.until(l => l.startsWith('eval ='));
    return +got.at(-1).split('=')[1];
  }
  quit() { this.send('quit'); }
}
