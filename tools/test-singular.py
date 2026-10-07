from pathlib import Path
import subprocess, json, datetime, traceback, queue, threading, time, re, sys, hashlib
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'data/mekitto_runtime'))
import cshogi
STATE=ROOT/'baseline/singular_experiment_state.json'
if STATE.exists(): raise RuntimeError('Saved experiment exists: do not overwrite')
FEAT=8519729
BASE=ROOT/'baseline/sailfish2_time_candidate.exe'
CAND=ROOT/'target_singular/release/sailfish2.exe'
def now(): return datetime.datetime.now().astimezone().isoformat()
state=dict(status='running',stage='positions',started=now(),feat=FEAT,plan=dict(screenGames=100,screenMs=50,confirmationGames=400,confirmationMs=100,parallel=4,confirmIfScreenScoreAtLeast=.45),baselineHash=hashlib.sha256(BASE.read_bytes()).hexdigest(),candidateHash=hashlib.sha256(CAND.read_bytes()).hexdigest(),matches=[])
def save(): STATE.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
class Engine:
 def __init__(self,exe,feat):
  self.p=subprocess.Popen([str(exe)],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,encoding='utf-8',creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
  self.q=queue.Queue()
  def reader():
   for line in self.p.stdout:self.q.put(line.strip())
  threading.Thread(target=reader,daemon=True).start()
  self.cmd('usi','usiok');self.send('setoption name Threads value 1');self.send('setoption name NetworkDelay value 0');self.send('setoption name Feat value '+str(feat));self.cmd('isready','readyok')
 def send(self,s):self.p.stdin.write(s+'\n');self.p.stdin.flush()
 def cmd(self,s,prefix):
  self.send(s);lines=[];deadline=time.monotonic()+90
  while True:
   line=self.q.get(timeout=max(.01,deadline-time.monotonic()));lines.append(line)
   if line.startswith(prefix):return line,lines
 def think(self,moves,limit):
  self.send('setoption name USI_Hash value 1');self.cmd('isready','readyok');self.send('setoption name USI_Hash value 64');self.cmd('isready','readyok');self.send('position startpos moves '+' '.join(moves))
  start=time.monotonic();answer,lines=self.cmd('go '+limit,'bestmove');infos=[l for l in lines if l.startswith('info depth') and 'score ' in l];last=infos[-1]
  match=re.search(r'score (cp|mate) (-?\d+)',last)
  board=cshogi.Board()
  for u in moves:
   m=board.move_from_usi(u);assert board.is_legal(m);board.push(m)
  best=answer.split()[1];assert board.is_legal(board.move_from_usi(best))
  for u in last.split(' pv ')[1].split():
   m=board.move_from_usi(u);assert board.is_legal(m),(u,last);board.push(m)
  return dict(best=best,scoreType=match[1],score=int(match[2]),depth=int(re.search(r'\bdepth (\d+)',last)[1]),nodes=int(re.search(r'\bnodes (\d+)',last)[1]),elapsedMs=(time.monotonic()-start)*1000,info=last,lines=infos,legalPV=True)
 def close(self):
  self.send('quit')
  try:self.p.wait(timeout=5)
  except subprocess.TimeoutExpired:self.p.kill()
def positions():
 data=json.loads((ROOT/'baseline/time_loss_analysis/analysis.json').read_text(encoding='utf-8'));rows=[]
 for g in data['games']:
  c=g['longConfirmed'];rows.append(dict(id='loss%d_%d'%(g['game'],c['ply']),moves=g['allMoves'][:c['ply']-1],reference=c['parent']))
 g=next(g for g in data['games'] if g['game']==7)
 for ply in [131,135,137]:rows.append(dict(id='loss7_'+str(ply),moves=g['allMoves'][:ply-1]))
 engines=[Engine(BASE,131121),Engine(CAND,FEAT)]
 out=dict(status='running',conditions=['nodes 1000000','movetime 1000'],positions=[])
 dest=ROOT/'baseline/singular_positions.json'
 try:
  for i,pos in enumerate(rows):
   row=dict(**pos,searches=[]);out['positions'].append(row)
   for limit in out['conditions']:
    for j in ([0,1] if i%2==0 else [1,0]):
     res=engines[j].think(pos['moves'],limit);row['searches'].append(dict(variant=['standard','singular'][j],limit=limit,**res));print(pos['id'],limit,['standard','singular'][j],res['best'],res['score'],res['depth'],flush=True)
   dest.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
  out['status']='complete';dest.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
 finally:
  for e in engines:e.close()
def match(label,games,ms,seed):
 prefix='singular_'+label
 spec='target_singular/release/sailfish2.exe|Threads=1,NetworkDelay=0,Feat='+str(FEAT)
 base='baseline/sailfish2_time_candidate.exe|Threads=1,NetworkDelay=0'
 with (ROOT/'baseline'/(prefix+'.log')).open('w',encoding='utf-8') as f:
  subprocess.run(['node','--no-warnings','tools/match.mjs',spec,base,str(games),str(ms),'4','--seed-offset',str(seed),'--json','baseline/'+prefix+'.json'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS,check=True,timeout=18000)
 result=json.loads((ROOT/'baseline'/(prefix+'.json')).read_text(encoding='utf-8'))
 assert result['games']==games and sum(result['results'].values())==games
 assert not any('反則' in k or '不正' in k for k in result['reasons']),result['reasons']
 summary={k:result[k] for k in ['games','moveMs','seedOffset','results','score','paired95','reasons','elapsedSeconds']};summary['path']='baseline/'+prefix+'.json';state['matches'].append(summary);save();print(label,json.dumps(summary,ensure_ascii=False),flush=True)
 return result
try:
 save();positions();state['stage']='screen';save();screen=match('screen100_50',100,50,71000)
 if screen['score']>=.45:
  state['stage']='confirmation';save();confirmation=match('confirmation400_100',400,100,72000)
  state['adoptionSupported']=confirmation['paired95'][0]>.5
 else:state['adoptionSupported']=False
 state.update(status='validation_complete',stage='complete',finished=now());save()
except Exception:
 state.update(status='failed',error=traceback.format_exc());save();raise
