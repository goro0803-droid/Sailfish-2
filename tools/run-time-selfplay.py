from pathlib import Path
import subprocess,json,datetime,hashlib,traceback,os,re,statistics,sys
R=Path(__file__).resolve().parent.parent;D=R/'baseline/time_selfplay_pilot_20261004';S=D/'state.json';os.chdir(R);state=json.loads(S.read_text(encoding='utf-8'));assert state['status']=='prepared';sys.path.insert(0,str(R/'data/mekitto_runtime'));import cshogi
state.update(status='running',pid=os.getpid());S.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
try:
 spec=str(R/'baseline/sailfish2_time_candidate.exe')+'|Threads=1,NetworkDelay=120,EvalFile='+str(D/'R5.nn')
 with (D/'selfplay.log').open('w',encoding='utf-8') as f:
  subprocess.run(['node','--no-warnings','tools/selfplay-time-clock.mjs',spec,spec,'4','300000','10000','4','--seed-offset','95000','--json',str(D/'games.json')],stdout=f,stderr=subprocess.STDOUT,check=True,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS,timeout=14400)
 data=json.loads((D/'games.json').read_text(encoding='utf-8'));assert data['games']==4;assert not any('反則' in k or '時間切れ' in k for k in data['reasons']),data['reasons']
 rows=[];swings=[];legalPV=0
 for g in data['samples']:
  b=cshogi.Board()
  for u in g['opening']:m=b.move_from_usi(u);assert b.is_legal(m);b.push(m)
  for r in g['trace']:
   assert ' '.join(b.sfen().split()[:3])==' '.join(r['sfen'].split()[:3]);m=b.move_from_usi(r['move']);assert b.is_legal(m)
   if r['info'] and ' pv ' in r['info']:
    check=cshogi.Board(r['sfen'])
    for u in r['info'].split(' pv ')[1].split():q=check.move_from_usi(u);assert check.is_legal(q);check.push(q)
    legalPV+=1
   cp=[]
   for info in r['infos']:
    v=re.search(r'info depth (\d+).*score cp (-?\d+)',info)
    if v and int(v[1])>=7:cp.append((int(v[1]),int(v[2])))
   if len(cp)>=2:
    change=cp[-1][1]-cp[-2][1]
    if abs(change)>=300:swings.append({'game':g['game'],'ply':r['ply'],'depths':[cp[-2][0],cp[-1][0]],'scores':[cp[-2][1],cp[-1][1]],'change':change,'sfen':r['sfen'],'move':r['move']})
   rows.append(r);b.push(m)
 # Self-play is data collection; arbitrary engine A score is not a rating.
 per=[]
 for g in data['samples']:
  for side in [0,1]:
   rr=[r for r in g['trace'] if r['side']==side];first=rr[:10]
   per.append({'game':g['game'],'side':side,'moves':len(rr),'first10SpentSec':sum(r['elapsedMs'] for r in first)/1000,'remainingAfter10Sec':first[-1]['afterMs']/1000,'minBankBeforeSec':min(r['beforeMs'] for r in rr)/1000,'finalBankSec':rr[-1]['afterMs']/1000})
 summary={'games':4,'positions':len(rows),'recordedLegalPVs':legalPV,'late121plus':sum(r['ply']>=121 for r in rows),'perSide':per,'first10SpentMedianSec':statistics.median(x['first10SpentSec'] for x in per),'remainingAfter10MedianSec':statistics.median(x['remainingAfter10Sec'] for x in per),'reasons':data['reasons'],'deepIterationSwings':sorted(swings,key=lambda x:abs(x['change']),reverse=True),'elapsedSeconds':data['elapsedSeconds']}
 (D/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 assert hashlib.sha256((R/'nets/default.nn').read_bytes()).hexdigest()==state['netSha256'];assert hashlib.sha256((R/'target/release/sailfish2.exe').read_bytes()).hexdigest()==state['engineSha256']
 text='# 現在Time版同士の自己対局試行（2026-10-04）\n\n現在のTime版＋R5同士、5分＋毎手10秒、1スレッド、Hash64、NetworkDelay120ms、4局並列。異なるランダム歩4手の開始局面から4局を実施。勝率・レート差の測定ではなく、実際の時間配分と読み筋を記録する試行。\n\n'+f"4局完了、探索して指した{len(rows)}局面、121手以降{summary['late121plus']}局面。全指し手合法、最終読み筋{legalPV}件の合法性確認。終局理由{data['reasons']}。\n各側が探索して指した最初10手の消費時間中央値{summary['first10SpentMedianSec']:.1f}秒、その10手後の残り時間中央値{summary['remainingAfter10MedianSec']:.1f}秒。\n\n"+'深さ7以上の最後の2反復で300点以上変化した局面候補をsummary.jsonへ保存。浅い読みからの変化だけで悪手と断定せず、次の分析候補として扱う。棋譜には毎手SFEN、消費・残り時間、最後の評価/PV、探索中の各反復情報を保存した。\n\n追加学習は未実施。この4局のみでは学習量も局面の多様性も不足し、現在の評価・探索の癖を再生産する可能性もある。まず記録局面を分析し、必要なら長い再探索で教師ラベルの品質を確認してから、対局単位で学習と検証を分離した小規模学習へ進める。標準exe/R5不変。\n\n保存：baseline/time_selfplay_pilot_20261004/games.json、games.json.moves.jsonl、summary.json、state.json。tools/selfplay-time-clock.mjs、tools/run-time-selfplay.pyで再現。\n'
 (R/'REPORT_Time版自己対局試行_20261004.md').write_text(text,encoding='utf-8')
 state.update(status='complete',finished=datetime.datetime.now().astimezone().isoformat(),summary=summary,standardUnchanged=True)
 with (R.parent/'引継ぎ_Sailfish2_Codex.md').open('a',encoding='utf-8') as f:f.write('\n\n## 24. 現在Time版自己対局試行（2026-10-04）\n'+text+'\n')
except Exception:
 state.update(status='failed',error=traceback.format_exc());raise
finally:S.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
