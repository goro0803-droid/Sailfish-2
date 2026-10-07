from pathlib import Path
import sys,json,collections,numpy as np,hashlib
R=Path(__file__).resolve().parent.parent;D=R/'baseline/teacher_coverage_20261004';sys.path.insert(0,str(R/'data/mekitto_runtime'));import cshogi
names=(D/'window_0_features.tsv').read_text().splitlines()[0].split('\t');idx={n:i for i,n in enumerate(names)}
def load(name):return np.loadtxt(D/name,skiprows=1)
def col(a,k):return a[:,idx[k]]
def summary(a):
 out={'n':len(a),'sideBlack':float(np.mean(col(a,'side')==0)),'resultWin':float(np.mean(col(a,'result')==2)),'plyQuantiles':np.percentile(col(a,'ply'),[10,50,90]).tolist(),'absScoreQuantiles':np.percentile(abs(col(a,'score')),[10,50,90]).tolist()}
 for k in ['in_check','capture_move','promo_move','drop_move','gives_check','loose_major','loose_minor','own_major_hand','opp_major_hand']:
  out[k]=float(np.mean(col(a,k)>0))
 for k in ['legal','captures','checks','own_hand','opp_hand','own_ring','opp_ring','promoted']:
  out[k]={'mean':float(np.mean(col(a,k))),'q50':float(np.median(col(a,k))),'q90':float(np.percentile(col(a,k),90))}
 out['late121plus']=float(np.mean(col(a,'ply')>=121));out['pressure']=float(np.mean((col(a,'own_ring')>=4)|(col(a,'checks')>=4)|(col(a,'captures')>=4)|(col(a,'loose_major')>0)))
 out['nn_teacher_mae']=float(np.mean(abs(col(a,'nn')-col(a,'score'))));return out
windows=[];pool=[]
for offset in [0,1000000,5000000]:
 a=load(f'window_{offset}_features.tsv');raw=a[col(a,'raw_eligible')==1];train=a[col(a,'train_eligible')==1];pool.append(train)
 # Reasons mutually exclusive in the same order as the training filter.
 masks={'raw_excluded':col(a,'raw_eligible')==0,'promo':(col(a,'raw_eligible')==1)&(col(a,'promo_move')==1),'capture':(col(a,'raw_eligible')==1)&(col(a,'promo_move')==0)&(col(a,'capture_move')==1),'checking_drop':(col(a,'raw_eligible')==1)&(col(a,'promo_move')==0)&(col(a,'capture_move')==0)&(col(a,'drop_move')==1)&(col(a,'gives_check')==1),'kept':col(a,'train_eligible')==1}
 assert sum(int(v.sum()) for v in masks.values())==len(a)
 windows.append({'offset':offset,'all':summary(a),'raw':summary(raw),'train':summary(train),'reasons':{k:int(v.sum()) for k,v in masks.items()}})
base=pool[0];allpool=np.concatenate(pool);loss=load('loss_features.tsv');maps=json.loads((D/'loss_map.json').read_text(encoding='utf-8'));critical=[]
# Loose major flags and king-neighborhood attacks are geometric proxies, not proof of a tactical threat.
for row,meta in zip(loss,maps):
 if not meta['critical']:continue
 f={k:int(row[idx[k]]) for k in names if k not in ['id','result']};counts={}
 # A transparent descriptive bucket: ply +/-20; own ring coarse class; opponent major in hand yes/no; promoted-piece count coarse class.
 def ringbin(x):return np.digitize(x,[2,5])
 def promobin(x):return np.digitize(x,[1,3])
 for label,data in [('existing',base),('three_windows',allpool)]:
  mask=(abs(col(data,'ply')-f['ply'])<=20)&(ringbin(col(data,'own_ring'))==ringbin(f['own_ring']))&((col(data,'opp_major_hand')>0)==(f['opp_major_hand']>0))&(promobin(col(data,'promoted'))==promobin(f['promoted']))
  counts[label]={'n':int(mask.sum()),'fraction':float(mask.mean())}
 percentiles={k:float(np.mean(col(base,k)<=f[k])*100) for k in ['ply','legal','captures','checks','own_ring','opp_ring','opp_hand','promoted']}
 critical.append(dict(meta,features=f,similarBuckets=counts,percentileInExistingTrain=percentiles))
# Match original retained count and exact selected positions; independent cshogi feature check.
board=cshogi.Board();lines=(D/'window_0.tsv').read_text(encoding='utf-8').splitlines();check_ids=list(range(0,40000,1999))
for i in check_ids:
 c=lines[i].split('\t');board.set_sfen(c[1]);legal=list(board.legal_moves);r=load('window_0_features.tsv')[i];assert len(legal)==r[idx['legal']]
 captures=sum(not cshogi.move_is_drop(m) and board.piece(cshogi.move_to(m))!=0 for m in legal)
 checks=0
 for m in legal:board.push(m);checks+=board.is_check();board.pop()
 assert captures==r[idx['captures']] and checks==r[idx['checks']]
audit={'windows':windows,'lossRegion':summary(loss),'critical':critical,'checks':{'independentRulePositions':len(check_ids),'allSampleMovesLegal':120000,'existingTrainPlusValidationMatchesExpected':len(base)==22960},'scope':'3 windows total 120000 records, first 5.04 million of 181867500; descriptive audit only, not random full archive sample','bucketDefinition':'ply +/-20; own_ring bins 0-1/2-4/5-8; opponent major hand present yes/no; promoted bins0/1-2/3+; structural bucket not board/strategy equivalence'}
assert len(base)==22960

audit['lossRegionEligibility']={'n':len(loss),'rawEligible':int(col(loss,'raw_eligible').sum()),'trainEligible':int(col(loss,'train_eligible').sum())}
audit['lossPerGame']=[]
for game in [4,6,7,9,14]:
 x=loss[[m['game']==game for m in maps]]
 audit['lossPerGame'].append({'game':game,'n':len(x),'eligible':int(col(x,'train_eligible').sum()),'pressure':float(np.mean((col(x,'own_ring')>=4)|(col(x,'checks')>=4)|(col(x,'captures')>=4)|(col(x,'loose_major')>0)))})
w=load('window_0_features.tsv')
for c in critical:
 f=c['features'];mask=(abs(col(w,'ply')-f['ply'])<=20)&(np.digitize(col(w,'own_ring'),[2,5])==np.digitize(f['own_ring'],[2,5]))&((col(w,'opp_major_hand')>0)==(f['opp_major_hand']>0))&(np.digitize(col(w,'promoted'),[1,3])==np.digitize(f['promoted'],[1,3]))
 c['similarBuckets']['beforeFilters']={'n':int(mask.sum()),'fraction':float(mask.mean())}
audit['lateRetentionFirstWindow']={'allLate':int((col(w,'ply')>=121).sum()),'keptLate':int(((col(w,'ply')>=121)&(col(w,'train_eligible')==1)).sum())}

(D/'coverage_summary.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print('counts',[(w['offset'],w['reasons']) for w in windows]);print('critical',[(c['game'],c['ply'],c['features']['train_eligible'],c['similarBuckets']) for c in critical]);print('late fraction',[(w['offset'],w['all']['late121plus'],w['train']['late121plus']) for w in windows]);print('pressure',[(w['offset'],w['all']['pressure'],w['train']['pressure']) for w in windows]);print('loss pressure',audit['lossRegion']['pressure'])
(D/'teacher_readme_utf8.txt').write_text((R/'data/mekitto_public_20190212/Readme.txt').read_text(encoding='cp932'),encoding='utf-8')
