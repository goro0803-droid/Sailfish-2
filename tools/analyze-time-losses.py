from pathlib import Path
import sys,json,subprocess,queue,threading,time,re,datetime
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'data/mekitto_runtime'))
import cshogi
OUT=ROOT/'baseline/time_loss_analysis'
OUT.mkdir(exist_ok=True)
raw=json.loads((ROOT/'baseline/time_clock20.json').read_text(encoding='utf-8'))
src=(ROOT/'tools/match-clock.mjs').read_text(encoding='utf-8')
fun=src[src.index('function randomOpening(seed)'):src.index('/** 1 局指す')]
node="import {Position,START_SFEN,moveToUsi,PAWN} from '../ShogiArena/src/js/shogi.js';\n"+fun+"\nconsole.log(JSON.stringify(Array.from({length:10},(_,i)=>randomOpening(40001+i))));"
openings=json.loads(subprocess.check_output(['node','--input-type=module','-e',node],cwd=ROOT,text=True,creationflags=subprocess.CREATE_NO_WINDOW))
class Engine:
 def __init__(self):
  self.p=subprocess.Popen([str(ROOT/'baseline/sailfish2_time_candidate.exe')],cwd=ROOT/'baseline',stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,encoding='utf-8',creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
  self.q=queue.Queue()
  def reader():
   for line in self.p.stdout:self.q.put(line.strip())
  threading.Thread(target=reader,daemon=True).start()
  self.cmd('usi','usiok')
  self.send('setoption name Threads value 1')
  self.send('setoption name USI_Hash value 64')
  self.send('setoption name NetworkDelay value 0')
  self.cmd('isready','readyok')
 def send(self,s):self.p.stdin.write(s+'\n');self.p.stdin.flush()
 def cmd(self,s,prefix,timeout=90):
  self.send(s);lines=[];deadline=time.monotonic()+timeout
  while True:
   l=self.q.get(timeout=max(.01,deadline-time.monotonic()));lines.append(l)
   if l.startswith(prefix):return l,lines
 def think(self,moves,ms):
  self.send('setoption name USI_Hash value 1');self.cmd('isready','readyok')
  self.send('setoption name USI_Hash value 64');self.cmd('isready','readyok')
  self.send('usinewgame')
  self.send('position startpos moves '+' '.join(moves))
  start=time.monotonic();answer,lines=self.cmd('go movetime '+str(ms),'bestmove')
  infos=[l for l in lines if l.startswith('info depth') and 'score ' in l]
  last=infos[-1] if infos else ''
  m=re.search(r'score (cp|mate) (-?\d+)',last)
  kind,value=(m[1],int(m[2])) if m else (None,None)
  comparable=value if kind=='cp' else ((30000 if value>0 else -30000) if value is not None else None)
  return dict(best=answer.split()[1],scoreType=kind,score=value,value=comparable,depth=int(re.search(r'\bdepth (\d+)',last)[1]) if last else 0,ms=round((time.monotonic()-start)*1000),info=last,lines=infos)
 def close(self):
  self.send('quit')
  try:self.p.wait(timeout=5)
  except subprocess.TimeoutExpired:self.p.kill()
def now():return datetime.datetime.now().isoformat()
state=dict(status='running',started=now(),scanMs=200,deepMs=5000,confirmMs=15000,threads=1,games=[])
def save(): (OUT/'analysis.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
e=Engine()
try:
 for g in raw['samples']:
  if g['score']!=0:continue
  opening=openings[g['game']//2];moves=opening[:];board=cshogi.Board()
  for u in moves:
   m=board.move_from_usi(u);assert board.is_legal(m);board.push(m)
  own=0 if g['aBlack'] else 1
  game=dict(game=g['game']+1,side=own,opening=opening,reason=g['reason'],rows=[],candidates=[])
  state['games'].append(game)
  for t in g['trace']:
   assert len(moves)+1==t['ply'] and board.turn==t['side']
   m=board.move_from_usi(t['move']);assert board.is_legal(m),(g['game'],t)
   before=e.think(moves,200)
   csa=cshogi.move_to_csa(m)
   game['rows'].append(dict(**t,sfen=board.sfen(),csa=csa,scan=before,ownValue=before['value']*(1 if board.turn==own else -1)))
   board.push(m);moves.append(t['move'])
  game['allMoves']=moves
  game['terminalSfen']=board.sfen()
  game['legalVerified']=True
  rows=game['rows']
  choices=[]
  for i,r in enumerate(rows[:-1]):
   if r['side']!=own:continue
   aft=rows[i+1]['ownValue'];drop=r['ownValue']-aft
   choices.append((drop,i))
  # Prefer competitive positions, exclude moves after the game was already very lost.
  pool=[v for v in choices if rows[v[1]]['ownValue']>-1000] or choices
  selected=sorted(pool,reverse=True)[:3]
  game['candidateIndices']=[i for drop,i in selected]
  save();print('scan game',game['game'],'plies',len(rows),'candidates',[(rows[i]['ply'],round(drop)) for drop,i in selected],flush=True)
 for game in state['games']:
  for idx in game['candidateIndices']:
   r=game['rows'][idx];prefix=game['allMoves'][:r['ply']-1]
   parent=e.think(prefix,5000)
   actual=e.think(prefix+[r['move']],5000)
   alt=e.think(prefix+[parent['best']],5000) if parent['best']!=r['move'] else actual
   item=dict(ply=r['ply'],move=r['move'],csa=r['csa'],remainingSec=r['beforeMs']/1000,spentSec=r['elapsedMs']/1000,scanDrop=r['ownValue']-game['rows'][idx+1]['ownValue'],parent=parent,actualChild=actual,bestChild=alt,actualOwn=-actual['value'],bestOwn=-alt['value'],regret=actual['value']-alt['value'])
   game['candidates'].append(item);save()
   print('deep game',game['game'],'ply',r['ply'],'played',r['move'],'best',parent['best'],'regret',item['regret'],flush=True)
  # Confirm the largest independently evaluated loss with three times the thinking time.
  selected=max(game['candidates'],key=lambda r:r['regret'])
  prefix=game['allMoves'][:selected['ply']-1]
  parent=e.think(prefix,15000)
  actual=e.think(prefix+[selected['move']],15000)
  alt=e.think(prefix+[parent['best']],15000) if parent['best']!=selected['move'] else actual
  game['confirmed']=dict(ply=selected['ply'],move=selected['move'],csa=selected['csa'],remainingSec=selected['remainingSec'],spentSec=selected['spentSec'],parent=parent,actualChild=actual,bestChild=alt,actualOwn=-actual['value'],bestOwn=-alt['value'],regret=actual['value']-alt['value'])
  save();print('confirmed game',game['game'],json.dumps({k:game['confirmed'][k] for k in ['ply','move','remainingSec','spentSec','regret']},ensure_ascii=False),flush=True)
 state.update(status='complete',finished=now());save()
except Exception as err:
 state.update(status='failed',error=repr(err));save();raise
finally:e.close()
