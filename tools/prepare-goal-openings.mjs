import {writeFileSync} from 'node:fs';import {pathToFileURL} from 'node:url';import {resolve} from 'node:path';
const root=resolve(import.meta.dirname,'..'),dir=resolve(root,'baseline/goal_shape_20261004');
const {Position,START_SFEN,moveToUsi,PAWN}=await import(pathToFileURL(resolve(root,'../ShogiArena/src/js/shogi.js')).href);
function opening(seed){let s=seed*9301+49297;const rnd=n=>{s=(s*9301+49297)%233280;return Math.floor(s/233280*n)};let pos=Position.fromSfen(START_SFEN),moves=[];for(let i=0;i<4;i++){const c=pos.legalMoves().filter(m=>!m.drop&&pos.movingType(m)===PAWN&&!pos.capturedOf(m));let m=c[rnd(c.length)];moves.push(moveToUsi(m));pos=pos.play(m)}return {seed,moves,key:pos.toSfen().split(' ').slice(0,3).join(' ')}}
const used=new Set(),rows=[];for(let seed=110001;rows.length<300;seed++){const r=opening(seed);if(!used.has(r.key)){used.add(r.key);rows.push(r)}}
writeFileSync(resolve(dir,'screen_openings.json'),JSON.stringify(rows.slice(0,100).map(r=>r.moves)));
writeFileSync(resolve(dir,'confirmation_openings.json'),JSON.stringify(rows.slice(100).map(r=>r.moves)));
writeFileSync(resolve(dir,'opening_manifest.json'),JSON.stringify({screenPairs:100,confirmationPairs:200,uniquePositions:used.size,overlap:0,scope:'4 random pawn moves, screen shared across comparators, confirmation distinct from screen',rows},null,2));console.log('300 legal unique opening positions, no screen-confirmation overlap');
