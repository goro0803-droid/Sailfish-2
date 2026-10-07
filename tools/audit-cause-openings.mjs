import {writeFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import {resolve} from 'node:path';
const root=resolve(import.meta.dirname,'..');
const {Position,START_SFEN,moveToUsi,PAWN}=await import(pathToFileURL(resolve(root,'../ShogiArena/src/js/shogi.js')).href);
function opening(seed){let s=seed*9301+49297;const rnd=n=>{s=(s*9301+49297)%233280;return Math.floor(s/233280*n)};let pos=Position.fromSfen(START_SFEN);let moves=[];for(let i=0;i<4;i++){const c=pos.legalMoves().filter(m=>!m.drop&&pos.movingType(m)===PAWN&&!pos.capturedOf(m));let m=c[rnd(c.length)];moves.push(moveToUsi(m));pos=pos.play(m)}return {seed,moves,key:pos.toSfen().split(' ').slice(0,3).join(' ')}}
const groups=[['cal600',80000,100],['R5screen',81000,100]];const out={summary:{},openings:[]};const sets={};for(const [label,offset,n] of groups){let rows=Array.from({length:n},(_,i)=>({label,...opening(offset+i+1)}));out.openings.push(...rows);sets[label]=new Set(rows.map(r=>r.key));out.summary[label]={pairs:n,uniquePositions:sets[label].size}}
out.summary.overlapBetweenComparisons=[...sets.cal600].filter(k=>sets.R5screen.has(k)).length;
writeFileSync(resolve(root,'baseline/cause_opening_audit.json'),JSON.stringify(out,null,2));console.log(JSON.stringify(out.summary));
