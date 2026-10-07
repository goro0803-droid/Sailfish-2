from pathlib import Path
import subprocess,json,datetime,traceback
ROOT=Path(__file__).resolve().parent.parent
statepath=ROOT/'baseline/danger_experiment_state.json'
if statepath.exists():raise RuntimeError('Experiment already has saved results; choose a new output name before rerunning')
state=dict(status='running',stage='screen',started=datetime.datetime.now().isoformat(),results=[])
variants=[('reply_order',655409,51000),('reply_both',1703985,52000),('fast_aspiration',2228273,53000)]
def save():statepath.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
def match(label,feat,games,ms,seed):
 prefix='danger_'+label+'_'+str(games)+'_'+str(ms)
 spec='target_danger/release/sailfish2.exe|Threads=1,NetworkDelay=0,Feat='+str(feat)
 base='baseline/sailfish2_time_candidate.exe|Threads=1,NetworkDelay=0'
 with (ROOT/'baseline'/(prefix+'.log')).open('w',encoding='utf-8') as f:
  subprocess.run(['node','--no-warnings','tools/match.mjs',spec,base,str(games),str(ms),'2','--seed-offset',str(seed),'--json','baseline/'+prefix+'.json'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS,check=True,timeout=18000)
 result=json.loads((ROOT/'baseline'/(prefix+'.json')).read_text(encoding='utf-8'))
 state['results'].append(dict(label=label,feat=feat,path='baseline/'+prefix+'.json',games=games,ms=ms,seedOffset=seed,results=result['results'],score=result['score'],paired95=result['paired95'],reasons=result['reasons']))
 save();print(label,json.dumps(state['results'][-1],ensure_ascii=False),flush=True)
 return result
try:
 save()
 for label,feat,seed in variants:
  state['current']=label;save();match(label,feat,100,50,seed)
 state['status']='screen_complete';state['finished']=datetime.datetime.now().isoformat();save()
except Exception:
 state.update(status='failed',error=traceback.format_exc());save();raise
