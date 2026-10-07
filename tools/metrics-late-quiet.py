from pathlib import Path
import sys,json,numpy as np
R=Path(__file__).resolve().parent.parent;D=R/'baseline/late_quiet_20261004';sys.path.insert(0,str(R/'data/mekitto_runtime'));import cshogi
out={}
for label in ['R5','control','late15']:
 a=np.loadtxt(D/(label+'_validation.tsv'),skiprows=1);assert len(a)==2000;out[label]={}
 for phase,mask in [('all',np.ones(len(a),dtype=bool)),('before121',a[:,5]<121),('121plus',a[:,5]>=121)]:
  b=a[mask];z=b[:,2]/600;y=.9/(1+np.exp(-b[:,0]/600))+.1*b[:,1]
  out[label][phase]={'n':len(b),'teacher_target_ce':float(np.mean(np.logaddexp(0,z)-y*z)),'outcome_ce':float(np.mean(np.logaddexp(0,z)-b[:,1]*z)),'teacher_mae':float(np.mean(abs(b[:,2]-b[:,0])))}
sets=[]
for name in ['screen_openings.json','confirmation_openings.json']:
 rows=json.loads((D/name).read_text(encoding='utf-8'));keys=set()
 for moves in rows:
  b=cshogi.Board()
  for u in moves:m=b.move_from_usi(u);assert b.is_legal(m);b.push(m)
  keys.add(' '.join(b.sfen().split()[:3]))
 assert len(keys)==len(rows);sets.append(keys)
assert not sets[0]&sets[1]
(D/'validation_metrics.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print('validation metrics saved; 300 legal unique positions, overlap0')
