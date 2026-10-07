from pathlib import Path
import subprocess,json,datetime,hashlib,traceback,os
R=Path(__file__).resolve().parent.parent; S=R/'baseline/cause_cal465_state.json'
resume=S.exists()
if resume:
 previous=json.loads(S.read_text(encoding='utf-8'))
 assert previous['status']=='failed', 'Refuse overwrite completed/running experiment'
os.chdir(R)
net=R/'baseline/nn_teacher_cal465.nn';exe='baseline/sailfish2_time_candidate.exe'
state={'status':'running','teacher':'C:/Users/goro0/OneDrive/Documents/shogi/public_teacher_by_mekitto.7z','hypothesis':'teacher probability calibration only','teacherK':465,'outputK':600,'candidateSha256':hashlib.sha256(net.read_bytes()).hexdigest(),'adoptRule':'independent confirmation paired95 lower > 0.5','matches':[]}
if resume: state['recoveredError']=previous.get('error')
def save():S.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
def match(label,other,games,ms,seed):
 state['stage']=label;save()
 spec=exe+'|Threads=1,NetworkDelay=0,EvalFile='+str(net)
 base=exe+'|Threads=1,NetworkDelay=0'+other
 dest=R/'baseline'/('cause_'+label+'.json')
 if not dest.exists():
  with (R/'baseline'/('cause_'+label+'.log')).open('w',encoding='utf-8') as f:
   subprocess.run(['node','--no-warnings','tools/match.mjs',spec,base,str(games),str(ms),'4','--seed-offset',str(seed),'--json','baseline/cause_'+label+'.json'],stdout=f,stderr=subprocess.STDOUT,check=True,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS,timeout=18000)
 result=json.loads((R/'baseline'/('cause_'+label+'.json')).read_text(encoding='utf-8'));assert result['games']==games
 assert not any('不正' in k or '反則' in k for k in result['reasons'])
 state['matches'].append(dict(label=label,**{k:result[k] for k in ['results','score','paired95','games','moveMs','seedOffset','reasons','elapsedSeconds']}));save();print(label,result['results'],result['score'],flush=True)
 return result
try:
 save()
 match('cal465_vs_cal600_200_50',',EvalFile='+str(R/'baseline/nn_mekitto_sample.nn'),200,50,80000)
 screen=match('cal465_vs_R5_200_50','',200,50,81000)
 if screen['score']>=.5:
  confirmation=match('cal465_vs_R5_400_100','',400,100,82000)
  state['adoptionSupported']=confirmation['paired95'][0]>.5
 else:state['adoptionSupported']=False
 state['status']='validation_complete';state['stage']='complete';save()
except Exception:
 state['status']='failed';state['error']=traceback.format_exc();save();raise
