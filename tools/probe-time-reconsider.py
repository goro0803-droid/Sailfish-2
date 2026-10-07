from pathlib import Path
import sys,json,subprocess,queue,threading,time,re,datetime,traceback
R=Path(__file__).resolve().parent.parent;D=R/'baseline/reconsider_diagnosis_20261004';D.mkdir(exist_ok=True);S=D/'state.json';assert not S.exists(),'Do not overwrite existing diagnosis';sys.path.insert(0,str(R/'data/mekitto_runtime'));import cshogi
source=json.loads((R/'baseline/time_selfplay_pilot_20261004/games.json').read_text(encoding='utf-8'));summary=json.loads((R/'baseline/time_selfplay_pilot_20261004/summary.json').read_text(encoding='utf-8'))
chosen={(x['game'],x['ply']):'competitive_last_iteration_swing' for x in summary['deepIterationSwings'] if max(abs(v) for v in x['scores'])<1800}
changes=[]
for g in source['samples']:
 for row in g['trace']:
  vals=[]
  for line in row['infos']:
   m=re.search(r'info depth (\d+).*score cp (-?\d+).* pv (\S+)',line)
   if m and int(m[1])>=9 and abs(int(m[2]))<1800:vals.append((int(m[1]),int(m[2]),m[3]))
  if row['ply']>=40 and len(vals)>=2 and vals[-1][2]!=vals[-2][2]:changes.append((abs(vals[-1][1]-vals[-2][1]),g['game'],row['ply']))
for _,g,p in sorted(changes,reverse=True):
 if len(chosen)>=6:break
 chosen.setdefault((g,p),'late_best_move_change')
positions=[]
for (gid,ply),why in chosen.items():
 g=next(x for x in source['samples'] if x['game']==gid);r=next(x for x in g['trace'] if x['ply']==ply)
 positions.append({'game':gid,'ply':ply,'reason':why,'sfen':r['sfen'],'actual':r['move'],'actualSpentMs':r['elapsedMs'],'actualInfo':r['info'],'actualIterations':r['infos'],'history':g['opening']+[x['move'] for x in g['trace'] if x['ply']<ply],'searches':[]})
state={'status':'running','started':datetime.datetime.now().astimezone().isoformat(),'stage':'controlled_probes','positions':positions,'conditions':{'coldSearchMs':15000,'baselineReferenceMs':45000,'childCompareMs':30000,'variants':{'standard':131121,'no_lmr':131185,'no_quiet_pruning':132657},'threads':1,'hashMB':64,'source':'selfplay Time+R5, no external teacher or training'},'interpretation':'Longer same-engine cold searches are diagnostic references, not ground truth; evidence required before feature implementation'}
def save():S.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
class Engine:
 def __init__(self):
  self.p=subprocess.Popen([str(R/'baseline/sailfish2_time_candidate.exe')],cwd=R,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,encoding='utf-8',creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS);self.q=queue.Queue()
  def read():
   for line in self.p.stdout:self.q.put(line.strip())
  threading.Thread(target=read,daemon=True).start();self.cmd('usi','usiok');self.send('setoption name Threads value 1');self.send('setoption name NetworkDelay value 0');self.send('setoption name EvalFile value '+str(R/'baseline/time_selfplay_pilot_20261004/R5.nn'));self.cmd('isready','readyok')
 def send(self,s):self.p.stdin.write(s+'\n');self.p.stdin.flush()
 def cmd(self,s,prefix):
  self.send(s);out=[];deadline=time.monotonic()+100
  while True:
   l=self.q.get(timeout=max(.01,deadline-time.monotonic()));out.append(l)
   if l.startswith(prefix):return l,out
 def think(self,moves,ms,feat):
  self.send('setoption name Feat value '+str(feat));self.send('setoption name USI_Hash value 1');self.cmd('isready','readyok');self.send('setoption name USI_Hash value 64');self.cmd('isready','readyok');self.send('usinewgame');self.send('position startpos moves '+' '.join(moves));start=time.monotonic();answer,lines=self.cmd('go movetime '+str(ms),'bestmove');infos=[l for l in lines if l.startswith('info depth') and 'score ' in l];last=infos[-1];m=re.search(r'score (cp|mate) (-?\d+)',last);board=cshogi.Board()
  for u in moves:q=board.move_from_usi(u);assert board.is_legal(q);board.push(q)
  best=answer.split()[1];assert board.is_legal(board.move_from_usi(best))
  for u in last.split(' pv ')[1].split():q=board.move_from_usi(u);assert board.is_legal(q);board.push(q)
  return {'best':best,'scoreType':m[1],'score':int(m[2]),'depth':int(re.search(r'\bdepth (\d+)',last)[1]),'nodes':int(re.search(r'\bnodes (\d+)',last)[1]),'elapsedMs':(time.monotonic()-start)*1000,'info':last,'iterations':infos,'legalPV':True}
 def close(self):
  self.send('quit');self.p.wait(timeout=5)
save();e=Engine()
try:
 for row in positions:
  for label,feat in state['conditions']['variants'].items():
   result=e.think(row['history'],15000,feat);row['searches'].append(dict(variant=label,**result));save();print(row['game'],row['ply'],label,result['best'],result['score'],result['depth'],flush=True)
  row['longBaseline']=e.think(row['history'],45000,131121);save()
 state['stage']='child_confirmation';save()
 for row in positions:
  alternatives={x['best'] for x in row['searches']}|{row['longBaseline']['best']}
  alternatives.discard(row['actual']);row['children']=[]
  if alternatives:
   for move in sorted(alternatives|{row['actual']}):
    result=e.think(row['history']+[move],30000,131121);row['children'].append(dict(move=move,**result));save()
 state.update(status='complete_diagnosis',stage='await_evidence_review',finished=datetime.datetime.now().astimezone().isoformat());save()
except Exception:
 state.update(status='failed',error=traceback.format_exc());save();raise
finally:e.close()
