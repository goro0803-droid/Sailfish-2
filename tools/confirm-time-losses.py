from pathlib import Path
import json,math,time
ROOT=Path(__file__).resolve().parent.parent
context={'__file__':str(ROOT/'tools/analyze-time-losses.py')}
exec((ROOT/'tools/analyze-time-losses.py').read_text(encoding='utf-8').split("state=dict(status='running'")[0],context)
Engine=context['Engine']
path=ROOT/'baseline/time_loss_analysis/analysis.json'
data=json.loads(path.read_text(encoding='utf-8'))
chosen={4:56,6:80,7:139,9:143,14:90}
def save():path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
data['longStatus']='running';save()
e=Engine()
try:
 for game in data['games']:
  ply=chosen[game['game']]
  r=next(r for r in game['rows'] if r['ply']==ply)
  prefix=game['allMoves'][:ply-1]
  root_ms=max(45000,math.ceil(r['elapsedMs']*3))
  parent=e.think(prefix,root_ms)
  actual=e.think(prefix+[r['move']],30000)
  alt=e.think(prefix+[parent['best']],30000) if parent['best']!=r['move'] else actual
  game['longConfirmed']=dict(ply=ply,move=r['move'],csa=r['csa'],sfen=r['sfen'],remainingSec=r['beforeMs']/1000,spentSec=r['elapsedMs']/1000,rootBudgetMs=root_ms,childBudgetMs=30000,parent=parent,actualChild=actual,bestChild=alt,actualOwn=-actual['value'],bestOwn=-alt['value'],regret=actual['value']-alt['value'])
  save();print('long game',game['game'],json.dumps({k:game['longConfirmed'][k] for k in ['ply','move','spentSec','actualOwn','bestOwn','regret']},ensure_ascii=False),'best',parent['best'],flush=True)
 data['longStatus']='complete';save()
except Exception as err:
 data['longStatus']='failed';data['longError']=repr(err);save();raise
finally:e.close()
