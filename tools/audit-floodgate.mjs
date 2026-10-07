import {spawn} from 'node:child_process';
import {createInterface} from 'node:readline';
import {writeFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import {resolve} from 'node:path';
const {Position,START_SFEN}=await import(pathToFileURL(resolve('../ShogiArena/src/js/shogi.js')).href);
const [archive,out='baseline/floodgate_quality.json',samplesOut='baseline/floodgate_search_positions.json']=process.argv.slice(2);
if(!archive) throw Error('Usage: node tools/audit-floodgate.mjs archive [report.json] [samples.json]');
const totals={games:0,floodgateGames:0,moves:0,comments:0,numeric:0,malformed:0,zero:0,zeroTime:0,large:0,pv:0,pvMoves:0,postMoveSidePV:0,preMoveSidePV:0,multiple:0};
const players={};let lines=[],seed=20251002,candidates=[],seen=0;
const rnd=n=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed%n;};
function game(ls){
 totals.games++;
 const event=ls.find(l=>l.startsWith('$EVENT:'))?.slice(7)||'';
 if(!event.startsWith('wdoor+floodgate-'))return;
 totals.floodgateGames++;
 const names=[ls.find(l=>l.startsWith('N+'))?.slice(2)||'?',ls.find(l=>l.startsWith('N-'))?.slice(2)||'?'];
 const ratings=[null,null];for(const l of ls){let m=l.match(/^'(black|white)_rate:.*:(-?\d+(?:\.\d+)?)$/);if(m)ratings[m[1]==='black'?0:1]=+m[2];}
 let moves=[],last=null,terminal='',standard=ls.includes('PI')||ls.filter(l=>/^P[1-9]/.test(l)).length===9;
 for(const l of ls){
 if(/^[+-][0-9]{4}[A-Z]{2}$/.test(l)){last={move:l,time:null,comments:[]};moves.push(last);}
 else if(/^T\d+$/.test(l)&&last)last.time=+l.slice(1);
 else if(l.startsWith("'**")&&last)last.comments.push(l.slice(3).trim());
 else if(l.startsWith('%'))terminal=l;
 }
 const finalSide=moves.at(-1)?.move[0]==='+'?1:0;
 const winner=terminal==='%TORYO'?1-finalSide:null;
 const eligible=[];
 for(let i=0;i<moves.length;i++){
 const row=moves[i],side=row.move[0]==='+'?0:1;
 const p=players[names[side]]??={games:0,moves:0,numeric:0,zero:0,zeroTime:0,large:0,tail:0,blackPerspectiveAgree:0,moverPerspectiveAgree:0,ratingMin:null,ratingMax:null};
 if(i===side){p.games++;if(ratings[side]!==null){p.ratingMin=Math.min(p.ratingMin??Infinity,ratings[side]);p.ratingMax=Math.max(p.ratingMax??-Infinity,ratings[side]);}}
 totals.moves++;p.moves++;if(row.comments.length>1)totals.multiple++;
 for(const c of row.comments){totals.comments++;const m=c.match(/^(-?\d+)(?:\s+(.*))?$/);if(!m){totals.malformed++;continue;}
 const score=+m[1],pv=(m[2]||'').match(/[+-][0-9]{4}[A-Z]{2}/g)||[];
 totals.numeric++;p.numeric++;if(score===0){totals.zero++;p.zero++;if(row.time===0){totals.zeroTime++;p.zeroTime++;}}
 if(Math.abs(score)>=2500){totals.large++;p.large++;}
 if(pv.length){totals.pv++;totals.pvMoves+=pv.length;if(pv[0][0]!==row.move[0])totals.postMoveSidePV++;else totals.preMoveSidePV++;}
 if(winner!==null&&i>=moves.length-12&&Math.abs(score)>=500&&Math.abs(score)<1000000){p.tail++;if((score>0?0:1)===winner)p.blackPerspectiveAgree++;if((score>0?side:1-side)===winner)p.moverPerspectiveAgree++;}
 if(standard&&winner!==null&&i>=24&&i<140&&Math.abs(score)>20&&Math.abs(score)<2500&&row.time>0&&pv.length>=4&&pv[0][0]!==row.move[0])eligible.push({i,score,pv,player:names[side],rating:ratings[side],time:row.time});
 }
 }
 if(eligible.length){seen++;const sample={event,moves:moves.map(m=>m.move),...eligible[rnd(eligible.length)]};if(candidates.length<64)candidates.push(sample);else{const j=rnd(seen);if(j<64)candidates[j]=sample;}}
}
const child=spawn('tar',['-xOf',archive],{windowsHide:true,stdio:['ignore','pipe','pipe']});let err='';child.stderr.on('data',d=>err+=d);
const done=new Promise((res,rej)=>{child.on('error',rej);child.on('close',c=>c===0?res():rej(Error(err)));});
for await(const l of createInterface({input:child.stdout,crlfDelay:Infinity})){if(l.startsWith('V2')&&lines.length){game(lines);lines=[];}lines.push(l);}if(lines.length)game(lines);await done;
const codes={FU:1,KY:2,KE:3,GI:4,KA:5,HI:6,KI:7,OU:8,TO:9,NY:10,NK:11,NG:12,UM:13,RY:14},letters={1:'P',2:'L',3:'N',4:'S',5:'B',6:'R',7:'G'};
function convert(p,c){let from=c.slice(1,3),to=c[3]+'abcdefghi'[+c[4]-1],pt=codes[c.slice(5)];if(from==='00')return letters[pt]+'*'+to;const orig=p.board[(+c[1]-1)*9+(+c[2]-1)]&15;return c[1]+'abcdefghi'[+c[2]-1]+to+(orig!==pt?'+':'');}
const samples=[],validation={selected:candidates.length,legalGamePrefixes:0,legalPV:0,invalidPV:0,invalidGame:0};
for(const c of candidates){try{let pos=Position.fromSfen(START_SFEN),prefix=[];for(let i=0;i<c.i;i++){let u=convert(pos,c.moves[i]),m=pos.findMove(u);if(!m)throw Error('game');prefix.push(u);pos=pos.play(m);}validation.legalGamePrefixes++;const played=convert(pos,c.moves[c.i]),m=pos.findMove(played);if(!m)throw Error('game');let q=pos.play(m),pv=[],bad=false;for(const x of c.pv){let u=convert(q,x),m=q.findMove(u);if(!m){bad=true;break;}pv.push(u);q=q.play(m);}if(bad)validation.invalidPV++;else validation.legalPV++;samples.push({id:samples.length+1,event:c.event,ply:c.i+1,player:c.player,rating:c.rating,commentScore:c.score,time:c.time,position:'startpos moves '+prefix.join(' '),played,pv,pvLegal:!bad,sfen:pos.toSfen()});}catch(e){validation.invalidGame++;}}
const report={archive,scope:'All floodgate-prefixed games, no training output; terminal sign correlation is a heuristic, not proof of convention.',totals,players,eligibleGames:seen,validation};
writeFileSync(out,JSON.stringify(report,null,2));writeFileSync(samplesOut,JSON.stringify(samples,null,2));console.log(JSON.stringify({totals,eligibleGames:seen,validation,topPlayers:Object.entries(players).sort((a,b)=>b[1].numeric-a[1].numeric).slice(0,12)},null,2));
