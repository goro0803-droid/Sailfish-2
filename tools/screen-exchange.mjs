import {spawn} from 'node:child_process';
import {readFileSync,writeFileSync} from 'node:fs';
import {resolve,dirname} from 'node:path';
class Engine{
 constructor(path){this.child=spawn(resolve(path),[],{cwd:dirname(resolve(path)),windowsHide:true});this.buf='';this.info=[];this.waiter=null;this.child.stdout.on('data',d=>{this.buf+=d;let i;while((i=this.buf.indexOf('\n'))>=0){const l=this.buf.slice(0,i).trim();this.buf=this.buf.slice(i+1);if(l.startsWith('info depth'))this.info.push(l);if(this.waiter&&l.startsWith(this.waiter.prefix)){const w=this.waiter;this.waiter=null;clearTimeout(w.timer);w.resolve(l);}}});this.child.on('error',e=>this.fail(e));this.child.on('exit',c=>this.fail(Error('Exit '+c)));}
 fail(e){this.error=e;if(this.waiter){clearTimeout(this.waiter.timer);this.waiter.reject(e);this.waiter=null;}}
 send(s){this.child.stdin.write(s+'\n');}
 command(s,prefix){if(this.error)throw this.error;return new Promise((resolve,reject)=>{this.waiter={prefix,resolve,reject,timer:setTimeout(()=>{this.fail(Error('Timeout '+s));this.child.kill();},60000)};this.send(s);});}
 async init(){await this.command('usi','usiok');this.send('setoption name Threads value 1');this.send('setoption name USI_Hash value 64');this.send('setoption name Hash value 64');this.send('setoption name USI_OwnBook value false');await this.command('isready','readyok');}
 async think(position,nodes,feat=49){this.send('setoption name USI_Hash value 1');this.send('setoption name Hash value 1');await this.command('isready','readyok');this.send('setoption name USI_Hash value 64');this.send('setoption name Hash value 64');this.send('setoption name Feat value '+feat);await this.command('isready','readyok');this.send('usinewgame');this.send('position '+position);this.info=[];const t=Date.now(),l=await this.command('go nodes '+nodes,'bestmove');const info=this.info.at(-1)||'',score=info.match(/score (cp|mate) (-?\d+)/);return {budget:nodes,feat,best:l.split(/\s+/)[1],depth:+(info.match(/\bdepth (\d+)/)?.[1]||0),nodes:+(info.match(/\bnodes (\d+)/)?.[1]||0),scoreType:score?.[1],score:score?+score[2]:null,pv:info.split(' pv ')[1]||'',ms:Date.now()-t};}
 quit(){this.send('quit');setTimeout(()=>this.child.kill(),500).unref();}
}

const samples=JSON.parse(readFileSync('baseline/search_confirmation_positions.json','utf8'));
const e=new Engine('target_exchange/release/sailfish2.exe');
const rows=[];
try{await e.init();for(const sample of samples.filter(s=>[13,16].includes(s.id))){
for(const feat of [49,65585,196657]){
const r=await e.think(sample.position,200000,feat);rows.push({id:sample.id,...r});console.log(rows.at(-1));}}
writeFileSync('baseline/exchange_screen.json',JSON.stringify(rows,null,2));
}finally{e.quit();}
