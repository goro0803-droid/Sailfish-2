from pathlib import Path
import json,time
ROOT=Path(__file__).resolve().parent.parent
source=(ROOT/'tools/analyze-time-losses.py').read_text(encoding='utf-8').split("state=dict(status='running'")[0].replace("ROOT/'baseline/sailfish2_time_candidate.exe'","ROOT/'target_reply/release/sailfish2.exe'").replace("'go movetime '+str(ms)","'go nodes '+str(ms)")
context={'__file__':str(ROOT/'tools/analyze-time-losses.py')};exec(source,context)
Engine=context['Engine'];e=Engine()
data=json.loads((ROOT/'baseline/time_loss_analysis/analysis.json').read_text(encoding='utf-8'))
positions=[]
for g in data['games']:
 c=g['longConfirmed'];positions.append(dict(id='loss%d_%d'%(g['game'],c['ply']),moves=g['allMoves'][:c['ply']-1],reference=c['parent']))
g=next(g for g in data['games'] if g['game']==7)
for ply in [131,135,137]:positions.append(dict(id='loss7_'+str(ply),moves=g['allMoves'][:ply-1]))
variants=[('baseline',131121),('reply_order',655409),('reply_safe',1179697),('reply_both',1703985)]
out=dict(status='running',nodes=1000000,positions=[])
path=ROOT/'baseline/reply_screen.json'
try:
 for pos in positions:
  row=dict(**pos,searches=[]);out['positions'].append(row)
  for label,feat in variants:
   e.send('setoption name Feat value '+str(feat));res=e.think(pos['moves'],1000000)
   row['searches'].append(dict(variant=label,feat=feat,**res))
   path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
  print(pos['id'],[(s['variant'],s['best'],s['score'],s['depth']) for s in row['searches']],flush=True)
 out['status']='complete';path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
finally:e.close()
