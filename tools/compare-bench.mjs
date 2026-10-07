// Repeat fixed-depth searches in alternating engine order.
// Usage: node tools/compare-bench.mjs "old.exe|EvalFile=absolute.nn" "new.exe|EvalFile=absolute.nn" [depth=11] [rounds=7]
// Threads=1, USI_Hash=64. Emits per-position nodes/scores/PVs and median times.
import {spawn,execFileSync} from 'node:child_process';
import {readFileSync} from 'node:fs';
import {resolve,dirname} from 'node:path';
import {performance} from 'node:perf_hooks';

const [specA,specB,depthArg='11',roundsArg='7'] = process.argv.slice(2);
const depth=Number(depthArg), rounds=Number(roundsArg);
const affinityCpu=process.env.SAILFISH_BENCH_CPU === undefined ? null : Number(process.env.SAILFISH_BENCH_CPU);
if (affinityCpu !== null && (!Number.isInteger(affinityCpu) || affinityCpu<0 || affinityCpu>30)) throw Error('Invalid benchmark CPU');
if (!specA || !specB || !Number.isInteger(depth) || depth<1 || !Number.isInteger(rounds) || rounds<1) {
  throw new Error('Usage: compare-bench.mjs engineA engineB [positive depth] [positive rounds]');
}
const positions=JSON.parse(readFileSync(new URL('./bench-positions.json',import.meta.url),'utf8'));

class Engine {
  constructor(spec) {
    const [exe,opts='']=spec.split('|');
    this.opts=opts.split(',').filter(Boolean).map(s=>s.split('='));
    this.child=spawn(resolve(exe),[],{cwd:dirname(resolve(exe)),windowsHide:true});
    if (affinityCpu !== null && process.platform === 'win32' && Number.isInteger(this.child.pid)) {
      const mask=(1n << BigInt(affinityCpu)).toString();
      execFileSync('powershell.exe',['-NoProfile','-NonInteractive','-Command',
        '(Get-Process -Id '+this.child.pid+').ProcessorAffinity=[IntPtr]'+mask],{windowsHide:true});
    }
    this.buffer=''; this.waiter=null; this.info='';
    this.child.stdout.on('data',data=>{
      this.buffer+=data.toString();
      let i;
      while ((i=this.buffer.indexOf('\n'))>=0) {
        const line=this.buffer.slice(0,i).trim(); this.buffer=this.buffer.slice(i+1);
        if (line.startsWith('info depth')) this.info=line;
        if (line.startsWith('info string')) this.fail(new Error(line));
        if (this.waiter && line.startsWith(this.waiter.prefix)) {
          const w=this.waiter; this.waiter=null; clearTimeout(w.timer); w.resolve(line);
        }
      }
    });
    this.child.on('error',e=>this.fail(e));
    this.child.on('exit',code=>this.fail(new Error('Engine exited: '+code)));
    this.child.stderr.on('data',data=>process.stderr.write(data));
  }
  fail(error) {
    this.error=error;
    if(this.waiter) {const w=this.waiter; this.waiter=null;clearTimeout(w.timer);w.reject(error);}
  }
  send(s) { this.child.stdin.write(s+'\n'); }
  command(s,prefix) {
    if (this.error) return Promise.reject(this.error);
    return new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>{this.fail(new Error('Timeout: '+s));this.child.kill();},60000);
      this.waiter={prefix,resolve,reject,timer}; this.send(s);
    });
  }
  async init() {
    await this.command('usi','usiok');
    this.send('setoption name Threads value 1');
    this.send('setoption name USI_Hash value 64');
    for(const [k,v] of this.opts) this.send('setoption name '+k+' value '+v);
    await this.command('isready','readyok');
  }
  async run() {
    let totalMs=0;
    const rows=[];
    for(const {source,position} of positions) {
      // usinewgame only clears TT; the engine retains move-ordering history.
      // Changing hash size recreates Searcher so every run uses identical history.
      this.send('setoption name USI_Hash value 1');
      await this.command('isready','readyok');
      this.send('setoption name USI_Hash value 64');
      await this.command('isready','readyok');
      this.send('position '+position);
      await this.command('isready','readyok');
      this.info='';
      const t=performance.now();
      const bestmove=await this.command('go depth '+depth,'bestmove');
      totalMs+=performance.now()-t;
      const info=this.info;
      if (!info) throw new Error('Missing search information');
      rows.push({
        source,depth:Number(info.match(/depth (\d+)/)?.[1]),
        nodes:Number(info.match(/ nodes (\d+)/)?.[1]),
        score:info.match(/ score (cp|mate) (-?\d+)/)?.slice(1).join(' '),
        pv:info.split(' pv ')[1],bestmove
      });
    }
    return {totalMs,rows};
  }
  quit() { if(!this.child.killed) {this.send('quit');setTimeout(()=>this.child.kill(),500).unref();} }
}
const engines=[new Engine(specA),new Engine(specB)], runs=[[],[]];
try {
  for(const e of engines) await e.init();
  for(const e of engines) await e.run(); // unmeasured warmup
  for(let r=0;r<rounds;r++) {
    for(const index of (r%2 ? [1,0] : [0,1])) runs[index].push(await engines[index].run());
  }
  const signature=run=>JSON.stringify(run.rows);
  const expected=signature(runs[0][0]);
  const identical=runs[0].every((run,i)=>signature(run)===signature(runs[1][i]));
  const stable=runs.flat().every(run=>signature(run)===expected);
  const median=a=>{a=[...a].sort((x,y)=>x-y);const i=Math.floor(a.length/2);return a.length%2?a[i]:(a[i-1]+a[i])/2;};
  const ms=runs.map(rs=>median(rs.map(r=>r.totalMs)));
  console.log(JSON.stringify({
    specs:[specA,specB],depth,rounds,affinityCpu,positions,identical,stable,
    medianMs:ms,speedupPercent:(ms[0]/ms[1]-1)*100,
    runs
  },null,2));
  if(!identical) process.exitCode=2;
} finally {for(const e of engines) e.quit();}
