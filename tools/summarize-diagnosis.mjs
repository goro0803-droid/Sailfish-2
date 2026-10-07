import {readFileSync,writeFileSync} from 'node:fs';
const x=JSON.parse(readFileSync('baseline/search_diagnosis.json','utf8'));
const q=JSON.parse(readFileSync('baseline/floodgate_quality.json','utf8'));
const ablated=x.rows.filter(r=>r.branchReference);
function value(s){return s?.scoreType==='cp'&&Math.abs(s.score)<30000?-s.score:null;}
const delta=(a,b)=>Number.isFinite(a)&&Number.isFinite(b)?a-b:null;
const details=ablated.map(r=>{const branches=Object.fromEntries(Object.entries(r.branchReference).map(([m,s])=>[m,value(s)]));return {id:r.sample.id,ply:r.sample.ply,player:r.sample.player,event:r.sample.event,short:r.searches[0].best,medium:r.searches[1].best,deep:r.searches[2].best,expert:r.expert.best,branches,deepGain:delta(branches[r.searches[2].best],branches[r.searches[0].best]),ablations:Object.fromEntries(Object.entries(r.ablations).map(([n,s])=>[n,{best:s.best,depth:s.depth,ms:s.ms,gain:delta(branches[s.best],branches[r.searches[1].best])}]))};});
const ablationSummary={};for(const name of Object.keys(ablated[0].ablations)){const ds=details.map(d=>d.ablations[name].gain).filter(Number.isFinite);ablationSummary[name]={n:ds.length,improve100:ds.filter(v=>v>=100).length,worse100:ds.filter(v=>v<=-100).length,meanCp:ds.reduce((a,b)=>a+b,0)/ds.length,medianMs:ablated.map(r=>r.ablations[name].ms).sort((a,b)=>a-b).slice(7,9).reduce((a,b)=>a+b,0)/2,matchesDeep:ablated.filter(r=>r.ablations[name].best===r.searches[2].best).length};}
const signs=x.rows.filter(r=>r.expert.scoreType==='cp'&&Math.abs(r.expert.score)>50&&Math.abs(r.sample.commentScore)>50);
const blackSignAgree=signs.filter(r=>Math.sign(r.expert.score)===Math.sign(r.sample.commentScore)*(r.sample.ply%2?1:-1)).length;
const summary={positions:x.rows.length,shortDeepChanged:x.rows.filter(r=>r.searches[0].best!==r.searches[2].best).length,agreements:[0,1,2].map(i=>x.rows.filter(r=>r.searches[i].best===r.expert.best).length),playedAgreement:x.rows.filter(r=>r.sample.played===r.expert.best).length,signs:{n:signs.length,blackSignAgree},ablationSelection:'First 16 of 33 changed positions; exploratory biased subset; child reference evaluations at 200000 nodes.',ablationSummary,details:details.sort((a,b)=>b.deepGain-a.deepGain)};
writeFileSync('baseline/search_diagnosis_summary.json',JSON.stringify(summary,null,2));console.log(JSON.stringify(summary,null,2));

