from pathlib import Path
import sys,subprocess,json,datetime,collections,runpy
R=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(R/'data/mekitto_runtime'))
import numpy as np,cshogi
D=R/'baseline/teacher_coverage_20261004';D.mkdir(exist_ok=True)
assert not (D/'windows.json').exists(),'Existing complete audit: do not overwrite'
board=cshogi.Board();starts=[0,1000000,5000000];size=40000
proc=subprocess.Popen(['C:/Program Files/7-Zip/7z.exe','x','-so',str(R/'data/mekitto_public_20190212/shuffled_20190212.7z'),'shuffled_20190212.bin'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
def readexact(n):
 parts=[]
 while n:
  x=proc.stdout.read(min(n,1024*1024));assert x,'Unexpected end of stream';parts.append(x);n-=len(x)
 return b''.join(parts)
counts=[];pos=0
try:
 for offset in starts:
  while pos<offset:
   n=min(offset-pos,262144);readexact(n*40);pos+=n
  raw=readexact(size*40);pos+=size
  arr=np.frombuffer(raw,dtype=cshogi.PackedSfenValue);lines=[];scores=[]
  for r in arr:
   board.set_psfen(np.array(r['sfen'],copy=True));m=board.move_from_move16(cshogi.move16_from_psv(int(r['move'])));assert board.is_legal(m)
   result=int(r['game_result']);assert result in [-1,0,1]
   lines.append('P\t'+board.sfen()+'\t'+str(int(r['score']))+'\t'+str((result+1)/2)+'\t'+cshogi.move_to_usi(m)+'\t'+str(int(r['gamePly'])))
  dest=D/f'window_{offset}.tsv';dest.write_text('\n'.join(lines)+'\n',encoding='utf-8');counts.append({'startRecord':offset,'records':size,'illegal':0,'invalid_result':0,'path':str(dest)})
  print('decoded window',offset,size,flush=True)
finally:
 if proc.poll() is None:proc.kill()
 proc.wait()
(D/'windows.json').write_text(json.dumps({'windows':counts,'streamedRecords':pos,'totalRecords':181867500,'fullCRCChecked':False,'representativeFullArchive':False},indent=2),encoding='utf-8')
loss=json.loads((R/'baseline/time_loss_analysis/analysis.json').read_text(encoding='utf-8'));rows=[];mapping=[]
for g in loss['games']:
 focus=g['longConfirmed'];fp=focus['ply']
 for row in g['rows']:
  if row['side']!=g['side'] or abs(row['ply']-fp)>20:continue
  score=focus['parent']['score'] if row['ply']==fp else row['scan']['value']
  board.set_sfen(row['sfen']);m=board.move_from_usi(row['move']);assert board.is_legal(m)
  rows.append('P\t'+row['sfen']+'\t'+str(score)+'\t0\t'+row['move']+'\t'+str(row['ply']))
  mapping.append({'id':len(rows)-1,'game':g['game'],'ply':row['ply'],'critical':row['ply']==fp,'scoreSource':'R5 post-game search, not external teacher','sfen':row['sfen'],'move':row['move']})
(D/'losses.tsv').write_text('\n'.join(rows)+'\n',encoding='utf-8');(D/'loss_map.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2),encoding='utf-8')
for offset in starts:
 subprocess.run([str(R/'target_cause/release/coverage_audit.exe'),str(R/'nets/default.nn'),str(D/f'window_{offset}.tsv'),str(D/f'window_{offset}_features.tsv')],check=True,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
subprocess.run([str(R/'target_cause/release/coverage_audit.exe'),str(R/'nets/default.nn'),str(D/'losses.tsv'),str(D/'loss_features.tsv')],check=True,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
print('coverage feature extraction complete',len(rows),'loss positions',flush=True)
