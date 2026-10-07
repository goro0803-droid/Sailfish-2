from pathlib import Path
import sys,json,subprocess,queue,threading,re,time,datetime,hashlib,traceback
R=Path(__file__).resolve().parent.parent;D=R/'baseline/goal_shape_20261004';S=D/'state.json';state=json.loads(S.read_text(encoding='utf-8'));assert state['status']=='building_candidate'
sys.path.insert(0,str(R/'data/mekitto_runtime'));import cshogi
net=R/'baseline/time_selfplay_pilot_20261004/R5.nn';base=R/'baseline/sailfish2_time_candidate.exe';candidate=R/'target_goal/release/sailfish2.exe'
def save():S.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protected=[R/'target/release/sailfish2.exe',R.parent/'ShogiArena/engines/sailfish2.exe',R.parent/'ShogiArena/dist/win-unpacked/resources/engines/sailfish2.exe',R/'nets/default.nn'];before={str(p):sha(p) for p in protected}
class Engine:
 def __init__(self,path,feat):
  self.p=subprocess.Popen([str(path)],cwd=R,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,encoding='utf-8',creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS);self.q=queue.Queue()
  def read():
   for line in self.p.stdout:self.q.put(line.strip())
  threading.Thread(target=read,daemon=True).start();self.cmd('usi','usiok')
  for k,v in [('Threads',1),('NetworkDelay',0),('Feat',feat),('EvalFile',str(net)),('USI_Hash',64)]:self.send('setoption name '+k+' value '+str(v))
  self.cmd('isready','readyok')
 def send(self,s):self.p.stdin.write(s+'\n');self.p.stdin.flush()
 def cmd(self,s,prefix):
  self.send(s);out=[]
  while True:
   l=self.q.get(timeout=60);out.append(l)
   if l.startswith(prefix):return l,out
 def think(self,sfen):
  self.send('usinewgame');self.send('position sfen '+sfen);answer,out=self.cmd('go depth 6','bestmove');info=[x for x in out if x.startswith('info depth')][-1];b=cshogi.Board(sfen);assert b.is_legal(b.move_from_usi(answer.split()[1]))
  for u in info.split(' pv ')[1].split():m=b.move_from_usi(u);assert b.is_legal(m);b.push(m)
  return {'bestmove':answer,'depth':int(re.search(r'\bdepth (\d+)',info)[1]),'nodes':int(re.search(r'\bnodes (\d+)',info)[1]),'score':re.search(r'score (cp|mate) (-?\d+)',info).group(0),'pv':info.split(' pv ')[1]}
 def close(self):self.send('quit');self.p.wait(timeout=5)
def compare(n,ms,name,openings):
 path=D/(name+'.json');assert not path.exists(),'Do not repeat completed experiment';state.update(status='running',stage=name,started=datetime.datetime.now().astimezone().isoformat());save()
 a=str(candidate)+'|Threads=1,NetworkDelay=0,Feat=67239985,EvalFile='+str(net);b=str(base)+'|Threads=1,NetworkDelay=0,EvalFile='+str(net)
 with (D/(name+'.log')).open('w',encoding='utf-8') as log:
  p=subprocess.Popen(['node','tools/match.mjs',a,b,str(n),str(ms),'4','--openings-json',str(D/openings),'--json',str(path)],cwd=R,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS);state['pid']=p.pid;save();assert p.wait(timeout=7200)==0
 q=json.loads(path.read_text(encoding='utf-8'));assert q['games']==n;assert not any(k.startswith(('反則','宣言')) for k in q['reasons']);state[name]=q;save();return q
try:
 positions=json.loads((D/'position_map.json').read_text(encoding='utf-8'));ids=json.loads((D/'independent_validation.json').read_text(encoding='utf-8'))['fixtures'][:6]
 checks=[]
 for i in ids:
  results=[]
  for path,feat in [(base,131121),(candidate,131121),(candidate,67239985)]:
   e=Engine(path,feat)
   try:results.append(e.think(positions[i]['sfen']))
   finally:e.close()
  assert results[0]==results[1],'OFF changed baseline';checks.append({'row':i,'offMatchesStandard':True,'onLegalPV':True,'results':results})
 state['verification']={'depth6Checks':checks,'standardHashesBefore':before,'candidateSha':sha(candidate),'networkSha':sha(net),'independentValidation':'independent_validation.json','libraryTestsPassed':27};save()
 screen=compare(200,50,'screen','screen_openings.json')
 if screen['score']>=.5:
  confirm=compare(400,100,'confirmation','confirmation_openings.json');supported=confirm['paired95'][0]>.5
 else:confirm=None;supported=False
 after={str(p):sha(p) for p in protected};assert before==after
 state.update(status='complete_supported_candidate' if supported else 'complete_not_adopted',stage='reported',decision='longer-time verification required; standard preserved' if supported else 'no statistically supported improvement; standard preserved',finished=datetime.datetime.now().astimezone().isoformat(),standardHashesAfter=after)
 report=['# 勝ちの形から手を選ぶ実験（2026年10月4日）','', '## 何を試したか','相手の玉を追い詰める形の入口として、相手の合法な応じ方が2手以下になる王手を、現在局面の候補の中で先に調べました。駒を取る手と成る手は従来の優先順位を維持し、駒交換で損をしないとSEEが見積もる王手だけが対象です。SEEは安全性の証明ではありません。','', '「こうなれば勝てそう」と仮説を作り、通常の探索で相手の抵抗を読む方法です。応手が少ないだけでは勝ちとは限りません。先の局面を固定して逆算する完全な計画機能ではなく、最初の小さな実験です。','', '## 根拠と確認','Time版＋R5の自己対局4局・537局面を調査。条件に合う王手は58手。2手以下の応手を持つ281手の王手と31手の限定的な1手／3手詰め候補をcshogiで独立検証しました。詰み候補は既に標準版も短時間で発見しており、それを新たな改善実績とは数えていません。','27件のライブラリテストが成功。独立実装に基づく11局面で候補の一致と盤面復元を確認。6局面の深さ6探索で機能OFFの最善手・評価・ノード数・PVが標準版と一致し、ONの指し手とPVも合法でした。','', '## 対局条件','R5を共通に使用。Threads=1、Hash=64MB、NetworkDelay=0、並列4。異なる開始局面で先後を入れ替えます。候補の追加判定時間も持ち時間に含めます。まず200局・1手50msで勝ち点50%以上なら、別の開始局面で400局・1手100msへ進む条件を対局前に保存しました。確認対局の先後ペアを考慮した95%区間の下限が50%を超えることを、改善を支持する条件としました。短時間試験から長時間棋力やfloodgateレートは推定しません。','']
 for label,q in [('初期比較',screen),('確認比較',confirm)]:
  if q:
   z=q['results'];lo,hi=q['paired95'];report+=['## '+label,f"{q['games']}局・1手{q['moveMs']}ms：候補 {z['w']}勝 {z['l']}敗 {z['d']}分。勝ち点 {q['score']*100:.2f}%（引き分けは半勝扱い）。ペア単位の95%目安 {lo*100:.2f}〜{hi*100:.2f}%。",'終局理由：'+json.dumps(q['reasons'],ensure_ascii=False),'']
 report+=['## 判定', '改善を支持する結果が出ました。ただし長い持ち時間での確認が必要なため標準版には採用していません。' if supported else '今回の条件では強くなったという根拠が得られませんでした。候補は不採用とし、標準版とR5を維持します。', '初期比較だけで終了した場合、弱化が確定したという意味ではなく、決めた確認試験への進行条件を満たさなかったという意味です。','', '次に試すなら、実際に標準版が見逃した勝ち筋がある局面を集めてから、逃げ道や守り駒に基づく別の条件を一つだけ検討します。追加実験は今回は自動で続けません。','', '## 保存先','候補：target_goal/release/sailfish2.exe（Feat=67239985、通常の既定値では無効）。記録：baseline/goal_shape_20261004/state.json、screen.json、independent_validation.json。標準exe3か所とnets/default.nnはSHA-256一致で保全を確認しました。']
 name='REPORT_勝ちの形_候補手順序_20261004.md';(R/name).write_text('\n'.join(report)+'\n',encoding='utf-8');state['report']=name;save()
 hand=R.parent/'引継ぎ_Sailfish2_Codex.md';text=hand.read_text(encoding='utf-8-sig');text+='\n\n## 26. 勝ちの形に基づく根局面の手順序（2026-10-04）\n\n'+'\n'.join(report[2:])+'\n';hand.write_text(text,encoding='utf-8')
 plan=R/'baseline/time_selfplay_followup_plan_20261004.json';q=json.loads(plan.read_text(encoding='utf-8'));q.update(status='complete_both_initial_investigations',secondApproachDecision=state['decision'],secondApproachReport=name,stopAutomationRequired=True);plan.write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf-8');print(state['status'],screen['results'], 'confirmation',confirm['results'] if confirm else None,flush=True)
except Exception:
 state.update(status='failed',error=traceback.format_exc());save();raise
