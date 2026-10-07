// Reproducible diagnostic searches; outputs are measurement only, never training labels.
import {spawn} from 'node:child_process';
import {readFileSync,writeFileSync} from 'node:fs';
import {resolve,dirname} from 'node:path';
const [exe,reference,positions='baseline/floodgate_search_positions.json',out='baseline/search_diagnosis.json']=process.argv.slice(2);
if(!exe||!reference)throw Error('Usage: node tools/diagnose-search.mjs sailfish.exe reference.exe [positions.json] [out.json]');
class Engine{
 constructor(path){this.child=spawn(resolve(path),[],{cwd:dirname(resolve(path)),windowsHide:true});this.buf='';this.info=[];this.waiter=null;this.child.stdout.on('data',d=>{this.buf+=d;let i;while((i=this.buf.indexOf('\n'))>=0){const l=this.buf.slice(0,i).trim();this.buf=this.buf.slice(i+1);if(l.startsWith('info depth'))this.info.push(l);if(this.waiter&&l.startsWith(this.waiter.prefix)){const w=this.waiter;this.waiter=null;clearTimeout(w.timer);w.resolve(l);}}});this.child.on('error',e=>this.fail(e));this.child.on('exit',c=>this.fail(Error('Exit '+c)));}
 fail(e){this.error=e;if(this.waiter){clearTimeout(this.waiter.timer);this.waiter.reject(e);this.waiter=null;}}
 send(s){this.child.stdin.write(s+'\n');}
 command(s,prefix){if(this.error)throw this.error;return new Promise((resolve,reject)=>{this.waiter={prefix,resolve,reject,timer:setTimeout(()=>{this.fail(Error('Timeout '+s));this.child.kill();},60000)};this.send(s);});}
 async init(){await this.command('usi','usiok');this.send('setoption name Threads value 1');this.send('setoption name USI_Hash value 64');this.send('setoption name Hash value 64');this.send('setoption name USI_OwnBook value false');await this.command('isready','readyok');}
 async think(position,nodes,feat=49){this.send('setoption name USI_Hash value 1');this.send('setoption name Hash value 1');await this.command('isready','readyok');this.send('setoption name USI_Hash value 64');this.send('setoption name Hash value 64');this.send('setoption name Feat value '+feat);await this.command('isready','readyok');this.send('usinewgame');this.send('position '+position);this.info=[];const t=Date.now(),l=await this.command('go nodes '+nodes,'bestmove');const info=this.info.at(-1)||'',score=info.match(/score (cp|mate) (-?\d+)/);return {budget:nodes,feat,best:l.split(/\s+/)[1],depth:+(info.match(/\bdepth (\d+)/)?.[1]||0),nodes:+(info.match(/\bnodes (\d+)/)?.[1]||0),scoreType:score?.[1],score:score?+score[2]:null,pv:info.split(' pv ')[1]||'',ms:Date.now()-t};}
 quit(){this.send('quit');setTimeout(()=>this.child.kill(),500).unref();}
}
const sf=new Engine(exe),ref=new Engine(reference),samples=JSON.parse(readFileSync(positions,'utf8')),rows=[];
const ablations={noLMR:64,noNull:128,noRFP:256,noLMP:512,noFutility:1024,noQSEE:2048,noQDelta:4096};
const budgets=(process.env.DIAG_NODES||'2000,20000,200000').split(',').map(Number),refNodes=Number(process.env.DIAG_REF_NODES||200000),ablationNodes=Number(process.env.DIAG_ABLATION_NODES||20000);
const checkpoint=()=>writeFileSync(out,JSON.stringify({measurementOnly:true,exe,reference,positions,budgets,refNodes,ablationNodes,rows},null,2));
try{await sf.init();await ref.init();for(const sample of samples){const searches=[];for(const nodes of budgets)searches.push(await sf.think(sample.position,nodes));const expert=await ref.think(sample.position,refNodes);rows.push({sample,searches,expert});if(rows.length%8===0){console.log('scanned '+rows.length+'/'+samples.length);checkpoint();}}
 const changed=rows.filter(r=>r.searches[0].best!==r.searches[2].best).slice(0,16);
 for(const row of changed){row.ablations={};for(const [name,bit]of Object.entries(ablations))row.ablations[name]=await sf.think(row.sample.position,ablationNodes,49|bit);const moves=[...new Set([row.sample.played,...row.searches.map(s=>s.best),row.expert.best,...Object.values(row.ablations).map(s=>s.best)])].filter(m=>m!=='resign'&&m!=='win');row.branchReference={};for(const m of moves)row.branchReference[m]=await ref.think(row.sample.position+' '+m,refNodes);console.log('diagnosed id='+row.sample.id+' changed '+row.searches[0].best+' -> '+row.searches[2].best);checkpoint();}
 checkpoint();console.log(JSON.stringify({positions:rows.length,shortDeepChanged:rows.filter(r=>r.searches[0].best!==r.searches[2].best).length,shortAgreement:rows.filter(r=>r.searches[0].best===r.expert.best).length,deepAgreement:rows.filter(r=>r.searches[2].best===r.expert.best).length,ablated:changed.length}));
}finally{sf.quit();ref.quit();}

