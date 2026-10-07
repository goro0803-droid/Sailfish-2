from pathlib import Path
import json,runpy
R=Path(__file__).resolve().parent.parent;D=R/'baseline/teacher_coverage_20261004'
def mirror(sfen):
 board,side,hand,*_=sfen.split();ranks=[]
 for rank in board.split('/'):
  cells=[];i=0
  while i<len(rank):
   c=rank[i]
   if c.isdigit():cells+=['.']*int(c)
   elif c=='+':i+=1;cells.append('+'+rank[i])
   else:cells.append(c)
   i+=1
  out='';n=0
  for cell in reversed(cells):
   if cell=='.':n+=1
   else:
    if n:out+=str(n);n=0
    out+=cell
  if n:out+=str(n)
  ranks.append(out)
 return '/'.join(ranks)+' '+side+' '+hand
old=runpy.run_path(str(R/'baseline/sample_mekitto_before_canonical_20261004.py'))['mirror_key']
def canonical(s):return min(' '.join(s.split()[:3]),mirror(s))
sets=[];counts=[];old_bad=0;tested=0
for name in ['sample_train.tsv','sample_validation.tsv']:
 lines=(R/'data/mekitto_public_20190212'/name).read_text(encoding='utf-8').splitlines();keys=set()
 for line in lines:
  s=line.split('\t')[1];keys.add(canonical(s));assert canonical(s)==canonical(mirror(s));tested+=1
  if min(' '.join(s.split()[:3]),old(s))!=min(mirror(s),old(mirror(s))):old_bad+=1
 sets.append(keys);counts.append({'name':name,'records':len(lines),'trueCanonicalUnique':len(keys)})
out={'existingSplit':counts,'trueCanonicalCrossSplitOverlap':len(sets[0]&sets[1]),'oldCanonicalMirrorInvariantFailures':old_bad,'testedPositions':tested,'note':'historical mirror_key formats original SFEN and mirrored expanded cells differently; audit uses compressed SFEN consistently; original datasets preserved'}
windowsets=[]
for offset in [0,1000000,5000000]:
 lines=(D/f'window_{offset}.tsv').read_text(encoding='utf-8').splitlines();keys={canonical(l.split('\t')[1]) for l in lines};windowsets.append(keys)
 out['window'+str(offset)]={'records':len(lines),'trueCanonicalUnique':len(keys)}
out['crossWindowCanonicalOverlap']={'0_1M':len(windowsets[0]&windowsets[1]),'0_5M':len(windowsets[0]&windowsets[2]),'1M_5M':len(windowsets[1]&windowsets[2])}
(D/'split_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
