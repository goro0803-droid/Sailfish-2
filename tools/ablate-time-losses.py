from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent.parent
source=(ROOT/'tools/analyze-time-losses.py').read_text(encoding='utf-8').split("state=dict(status='running'")[0].replace("'go movetime '+str(ms)","'go nodes '+str(ms)")
context={'__file__':str(ROOT/'tools/analyze-time-losses.py')}
exec(source,context)
Engine=context['Engine']
data=json.loads((ROOT/'baseline/time_loss_analysis/analysis.json').read_text(encoding='utf-8'))
variants=[('standard',131121),('no_lmr',131121|64),('no_null',131121|128),('no_quiet_pruning',131121|512|1024),('no_qs_pruning',131121|2048|4096)]
out=dict(status='running',nodes=1000000,variants=dict(variants),games=[])
path=ROOT/'baseline/time_loss_analysis/ablation.json'
e=Engine()
try:
 for game in data['games']:
  c=game['longConfirmed'];row=dict(game=game['game'],ply=c['ply'],searches=[])
  out['games'].append(row)
  for label,feat in variants:
   e.send('setoption name Feat value '+str(feat))
   result=e.think(game['allMoves'][:c['ply']-1],1000000)
   row['searches'].append(dict(variant=label,feat=feat,**result))
   path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
  print('ablated game',game['game'],[(s['variant'],s['best'],s['score'],s['depth']) for s in row['searches']],flush=True)
 out['status']='complete';path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
finally:e.close()
