from pathlib import Path
import subprocess,json,hashlib,datetime,traceback,os
R=Path(__file__).resolve().parent.parent;D=R/'baseline/late_quiet_20261004';S=D/'state.json';os.chdir(R)
assert not S.exists(),'Do not overwrite running or completed experiment'
base=R/'baseline/sailfish2_time_candidate.exe';candidate=D/'late15.nn';control=D/'control.nn'
state={'status':'running','hypothesis':'121+ ply quiet training proportion only:6.09% to15.00%','training':{'positionsEach':21784,'commonValidation':2000,'epochs':24,'lambda':.9,'lr':.00005,'batch':1024,'k':600,'teacherK':600,'threads':2,'initial':'nets/default.nn'},'plan':{'screenGamesEach':200,'screenMs':50,'confirmGamesEach':400,'confirmMs':100,'parallel':4,'confirmRule':'both initial scores >=0.5','supportRule':'both confirmation paired95 lower >0.5','deployment':'candidate only until longer-time verification'},'hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [base,candidate,control]},'matches':[],'started':datetime.datetime.now().astimezone().isoformat()}
def save():S.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
def match(label,opp,games,ms,openings):
 state['stage']=label;save()
 spec=str(base)+'|Threads=1,NetworkDelay=0,EvalFile='+str(candidate)
 other=str(base)+'|Threads=1,NetworkDelay=0'+(',EvalFile='+str(opp) if opp else '')
 with (D/(label+'.log')).open('w',encoding='utf-8') as f:
  subprocess.run(['node','--no-warnings','tools/match.mjs',spec,other,str(games),str(ms),'4','--openings-json',str(D/openings),'--json',str(D/(label+'.json'))],stdout=f,stderr=subprocess.STDOUT,check=True,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS,timeout=18000)
 result=json.loads((D/(label+'.json')).read_text(encoding='utf-8'));assert result['games']==games and sum(result['results'].values())==games
 assert not any('反則' in k or '不正' in k for k in result['reasons'])
 state['matches'].append({'label':label,**{k:result[k] for k in ['games','moveMs','results','score','paired95','reasons','elapsedSeconds','openingSource']}});save();print(label,result['results'],result['score'],flush=True)
 return result
try:
 save();first=match('late15_vs_control_screen200',control,200,50,'screen_openings.json');second=match('late15_vs_R5_screen200',None,200,50,'screen_openings.json')
 if first['score']>=.5 and second['score']>=.5:
  cf=match('late15_vs_control_confirm400',control,400,100,'confirmation_openings.json');cr=match('late15_vs_R5_confirm400',None,400,100,'confirmation_openings.json');state['strengthSupport']=cf['paired95'][0]>.5 and cr['paired95'][0]>.5
 else:state['strengthSupport']=False
 state.update(status='validation_complete',stage='complete',finished=datetime.datetime.now().astimezone().isoformat());save()
except Exception:
 state.update(status='failed',error=traceback.format_exc());save();raise
