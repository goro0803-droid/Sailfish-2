from pathlib import Path
import subprocess,json,datetime,traceback
ROOT=Path(__file__).resolve().parent.parent
p=ROOT/'baseline/danger_experiment_state.json'
state=json.loads(p.read_text(encoding='utf-8'))
assert state['status']=='screen_complete'
winner=max(state['results'],key=lambda r:r['score'])
assert winner['label']=='fast_aspiration'
state.update(status='running',stage='confirmation',selected=winner['label'],confirmationStarted=datetime.datetime.now().isoformat())
def save():p.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
try:
 save()
 prefix='danger_fast_confirmation400_100'
 spec='target_danger/release/sailfish2.exe|Threads=1,NetworkDelay=0,Feat=2228273'
 base='baseline/sailfish2_time_candidate.exe|Threads=1,NetworkDelay=0'
 with (ROOT/'baseline'/(prefix+'.log')).open('w',encoding='utf-8') as f:
  subprocess.run(['node','--no-warnings','tools/match.mjs',spec,base,'400','100','2','--seed-offset','60000','--json','baseline/'+prefix+'.json'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS,check=True,timeout=16000)
 result=json.loads((ROOT/'baseline'/(prefix+'.json')).read_text(encoding='utf-8'))
 state['confirmation']=dict(path='baseline/'+prefix+'.json',games=400,ms=100,seedOffset=60000,results=result['results'],score=result['score'],paired95=result['paired95'],reasons=result['reasons'])
 state.update(status='confirmation_complete',confirmationFinished=datetime.datetime.now().isoformat());save()
 print(json.dumps(state['confirmation'],ensure_ascii=False),flush=True)
except Exception:
 state.update(status='failed',error=traceback.format_exc());save();raise
