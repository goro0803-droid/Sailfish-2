from pathlib import Path
import sys,runpy,json,hashlib,numpy as np
R=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(R/'data/mekitto_runtime'))
mirror=runpy.run_path(str(R/'tools/sample-mekitto.py'))['mirror_key']
D=R/'baseline/late_quiet_20261004';D.mkdir(exist_ok=True);assert not (D/'data_manifest.json').exists(),'Preserve previous dataset'
source=R/'baseline/teacher_coverage_20261004';pool=[];seen=set();dup=0
for offset in [0,1000000,5000000]:
 rows=(source/f'window_{offset}.tsv').read_text(encoding='utf-8').splitlines();features=np.loadtxt(source/f'window_{offset}_features.tsv',skiprows=1)
 for row,f in zip(rows,features):
  if not f[7]:continue
  sf=row.split('\t')[1];key=min(' '.join(sf.split()[:3]),mirror(sf))
  if key in seen:dup+=1;continue
  seen.add(key);digest=hashlib.sha256(key.encode()).digest();pool.append((row,key,int(f[3]),int.from_bytes(digest[:8],'little')))
# All canonical groups assigned to validation by a deterministic hash, before balancing.
reserved=[x for x in pool if x[3]%20==0];available=[x for x in pool if x[3]%20!=0]
def order(x):return hashlib.sha256(('latequiet-fixed-order-v1 '+x[1]).encode()).digest()
reserved.sort(key=order);available.sort(key=order);validation=reserved[:2000]
early=[x for x in available if x[2]<121];late=[x for x in available if x[2]>=121];n=21784
control_late=round(n*len(late)/len(available));candidate_late=round(n*.15)
assert len(late)>=candidate_late and len(early)>=n-control_late
control=sorted(early[:n-control_late]+late[:control_late],key=order);candidate=sorted(early[:n-candidate_late]+late[:candidate_late],key=order)
assert len(control)==len(candidate)==n and not ({x[1] for x in available}&{x[1] for x in reserved})
for label,rows in [('control_train',control),('late15_train',candidate),('validation',validation)]:
 (D/(label+'.tsv')).write_text('\n'.join(x[0] for x in rows)+'\n',encoding='utf-8')
manifest={'sourceWindows':[0,1000000,5000000],'legalFilteredPool':len(pool),'canonicalDuplicatesRemoved':dup,'trainPool':len(available),'reservedValidationPool':len(reserved),'reservedValidationLate':sum(x[2]>=121 for x in reserved),'commonValidation':len(validation),'commonValidationLate':sum(x[2]>=121 for x in validation),'trainPositionsEach':n,'controlLate':control_late,'candidateLate':candidate_late,'controlLateRatio':control_late/n,'candidateLateRatio':candidate_late/n,'commonTrainPositions':len({x[1] for x in control}&{x[1] for x in candidate}),'split':'correct mirror canonical SHA256 modulo20; all validation groups withheld before sampling','sampling':'same deterministic priority, nested early/late selection, no repeated records','trainValidationMirrorOverlap':0,'gameDisjointGuaranteed':False}
for label,rows in [('control',control),('late15',candidate)]:
 manifest[label+'Stats']={'meanAbsTeacherScore':sum(abs(float(x[0].split('\t')[2])) for x in rows)/len(rows),'blackRatio':sum(x[0].split('\t')[1].split()[1]=='b' for x in rows)/len(rows),'winRatio':sum(float(x[0].split('\t')[3]) for x in rows)/len(rows),'phaseCounts':[sum(lo<=x[2]<hi for x in rows) for lo,hi in [(0,41),(41,81),(81,121),(121,161),(161,10000)]]}
manifest['fileHashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.glob('*.tsv')}
(D/'data_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(manifest,indent=2))
