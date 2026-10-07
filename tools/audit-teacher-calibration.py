from pathlib import Path
import numpy as np,json
p=Path(__file__).resolve().parent.parent/'baseline';a=np.loadtxt(p/'cause_teacher_train.tsv',skiprows=1);b=np.loadtxt(p/'cause_teacher_validation.tsv',skiprows=1)
def ce(s,y,k):
 z=s/k;return float(np.mean(np.logaddexp(0,z)-y*z))
out={}
for name,col in [('teacher',0),('nn',2),('handcrafted',3)]:
 k=int(min(range(100,2001,5),key=lambda k:ce(a[:,col],a[:,1],k)))
 out[name]={'train_fitted_k':k,'train_ce600':ce(a[:,col],a[:,1],600),'train_ce_fit':ce(a[:,col],a[:,1],k),'validation_ce600':ce(b[:,col],b[:,1],600),'validation_ce_fit':ce(b[:,col],b[:,1],k)}
out['n']={'train':len(a),'validation':len(b)}
for side in [0,1]:
 x=b[b[:,4]==side];out['side'+str(side)]={'n':len(x),'bias':float(np.mean(x[:,2]-x[:,0])),'mae':float(np.mean(abs(x[:,2]-x[:,0])))}
if (p/'cause_cal465_validation.tsv').exists():
 c=np.loadtxt(p/'cause_cal465_validation.tsv',skiprows=1)
 out['candidate_outcome_ce_at_outputK600']=ce(c[:,2],c[:,1],600)
 out['candidate_raw_teacher_mae']=float(np.mean(abs(c[:,2]-c[:,0])))
 out['standard_raw_teacher_mae']=float(np.mean(abs(b[:,2]-b[:,0])))
(p/'cause_calibration_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
