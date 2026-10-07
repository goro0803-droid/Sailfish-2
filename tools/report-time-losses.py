from pathlib import Path
import sys,json,hashlib,datetime
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'data/mekitto_runtime'))
import cshogi
sys.path.insert(0,str(ROOT/'data/plot_runtime'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font=FontProperties(fname='C:/Windows/Fonts/meiryo.ttc')
plt.rcParams['font.family']=font.get_name()
data=json.loads((ROOT/'baseline/time_loss_analysis/analysis.json').read_text(encoding='utf-8'))
assert data['longStatus']=='complete' and len(data['games'])==5
abl=json.loads((ROOT/'baseline/time_loss_analysis/ablation.json').read_text(encoding='utf-8'))
assert abl['status']=='complete'
names={'FU':'歩','KY':'香','KE':'桂','GI':'銀','KI':'金','KA':'角','HI':'飛','OU':'玉','TO':'と','NY':'成香','NK':'成桂','NG':'成銀','UM':'馬','RY':'龍'}
nums='一二三四五六七八九'
def notation(board,usi):
 c=cshogi.move_to_csa(board.move_from_usi(usi));dst=c[2]+nums[int(c[3])-1]+names[c[4:]]
 return dst+('打' if c[:2]=='00' else '（'+c[0]+nums[int(c[1])-1]+'から）')
summary=[]
for g in data['games']:
 board=cshogi.Board();own=[];check_count=0;forced=[]
 for u in g['opening']:board.push_usi(u)
 for r in g['rows']:
  r['legalCount']=len(list(board.legal_moves));r['inCheck']=board.is_check()
  if r['legalCount']==1:forced.append(r['ply'])
  if r['side']==g['side']:
   own.append(r)
   if r['inCheck']:check_count+=1
  board.push_usi(r['move'])
 c=g['longConfirmed'];b=cshogi.Board()
 for u in g['allMoves'][:c['ply']-1]:b.push_usi(u)
 first=next((r for r in own if r['ownValue']<=-500 and r['legalCount']>1),None)
 summary.append(dict(game=g['game'],side=g['side'],plies=len(g['allMoves']),minBankSec=min(r['beforeMs'] for r in own)/1000,finalBankSec=own[-1]['afterMs']/1000,firstScanMinus500=first['ply'] if first else None,scanMinus500BankSec=first['beforeMs']/1000 if first else None,focusPly=c['ply'],played=notation(b,c['move']),alternative=notation(b,c['parent']['best']),spentSec=c['spentSec'],remainingSec=c['remainingSec'],rootMs=c['rootBudgetMs'],rootScore=c['parent']['value'],actualOwn=c['actualOwn'],alternativeOwn=c['bestOwn'],difference=c['regret'],forcedPlies=forced))
 g['forcedPlies']=forced
 (ROOT/'baseline/time_loss_analysis'/('game_%02d.usi.txt'%g['game'])).write_text('position startpos moves '+' '.join(g['allMoves'])+'\n',encoding='utf-8')
path=ROOT/'baseline/time_loss_analysis/analysis.json'
path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'baseline/time_loss_analysis/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
out=ROOT/'output/figures';out.mkdir(exist_ok=True)
fig,ax=plt.subplots(figsize=(10,5.5))
for g in data['games']:
 rows=[r for r in g['rows'] if r['side']==g['side']]
 ax.plot([r['ply'] for r in rows],[r['beforeMs']/1000 for r in rows],label='第%d局'%g['game'],linewidth=1.7)
ax.axhline(60,color='#666666',linestyle='--',alpha=.7,label='残り60秒')
ax.set_title('負けた5局の残り時間：時間切れの負けはなかった',fontproperties=font)
ax.set_xlabel('棋譜の手数（先手・後手を合わせた数）',fontproperties=font)
ax.set_ylabel('自分が考え始める直前の残り秒数',fontproperties=font)
ax.grid(alpha=.2);ax.legend(prop=font,ncol=3);fig.tight_layout()
fig.savefig(out/'Time版_負けた5局の残り時間.png',dpi=160);plt.close(fig)
g=next(g for g in data['games'] if g['game']==7)
r=next(r for r in g['rows'] if r['ply']==139)
cand=next(r for r in g['candidates'] if r['ply']==139)
stages=[('0.2秒',r['scan']),('5秒',cand['parent']),('15秒',g['confirmed']['parent']),('約47秒',g['longConfirmed']['parent'])]
fig,ax=plt.subplots(figsize=(8,4.6))
values=[s['value'] for label,s in stages]
ax.plot(range(4),values,'o-',color='#3267a8',linewidth=2)
ax.set_xticks(range(4),[label for label,s in stages],fontproperties=font)
for i,v in enumerate(values):ax.annotate(str(v),(i,v),xytext=(0,10),textcoords='offset points',ha='center')
ax.set_ylim(-1150,180);ax.axhline(0,color='#777777',linewidth=.8)
ax.set_title('第7局・139手目直前：深く読むと不利が見えてくる',fontproperties=font)
ax.set_xlabel('指定した読み時間（実戦の評価ログではなく、再分析）',fontproperties=font)
ax.set_ylabel('Sailfish2自身の評価：下ほど自分が不利',fontproperties=font)
ax.grid(alpha=.2);fig.tight_layout()
fig.savefig(out/'Time版_第7局の読み時間と評価.png',dpi=160);plt.close(fig)
sha=hashlib.sha256((ROOT/'baseline/sailfish2_time_candidate.exe').read_bytes()).hexdigest()
lines=['# 時間配分版が負けた5局の原因調査（2026-10-04）','',
'## わかったこと','',
'時間切れが直接の敗因だった局はない。5局とも詰みで終わり、最後の自分の手を指した後にも約60〜156秒が残った。序盤の節約を取り消す根拠は今回の調査では見つからない。',
'ただし、時間を残すだけで局面の判断や読み落としが解決するわけではない。第7局では、深く読むと評価が大きく悪化し、実戦とは別の手を選んだ。その手を指しても不利なままで、139手目を唯一の敗着と断定することはできない。',
'第4・6・9・14局の候補局面では、実際の手より明確に改善する代案を長い再分析で確認できなかった。原因候補は中盤から終盤の攻防を十分に見通せないことだが、探索と評価関数のどちらが主因かはまだ切り分け切れていない。','',
'## 対局と調べ方','',
'対象はTime版対旧Delta版の20局（13勝5敗2分）のうち、Time版が負けた第4・6・7・9・14局。floodgateの15局とは別のローカル比較。5分＋毎手10秒、Threads1、Hash64MB、対局時NetworkDelay120ms、ランダムな歩4手の序盤を先後ペアで使用。',
'保存されたseedと対局ツールの同じ序盤生成関数から最初4手を再現し、cshogiで5局全手の合法性を確認した。違法手0。実測時間は保存済みtraceを使用。',
'全759局面を200msで一度ずつ再分析した（117＋157＋156＋206＋123局面）。短い評価で悪化した自分の手を各局3候補選び、親局面・実際の手の後・代案の後を5秒、候補1局面を15秒で確認した。その後、各局の重要局面を実戦の3倍以上・最低45秒で読み直し、実際の手と代案を指した後を同じ30秒で比較した。第9局などの一手しか指せない終盤は原因候補から除外し、もっと前の局面を長時間確認した。',
'再分析も同じTime版＋R5を使用、Threads1/Hash64、固定読み時間、各検索前に置換表と探索状態をリセット。実戦中の評価・PVは対局ツールに記録されていない。実戦時の履歴・置換表を完全再現した実験ではない。',
'評価値は手番側からTime版側へ統一。マイナスはTime版が不利という意味。点数は同じソフト自身の判断であり、強い別のソフトによる正解認定や実際の勝率ではない。詰みの評価を便宜上±30000に置いた行と、指せる手が一手の行は比較の解釈に注意する。','',
'## 5局の重要局面','',
'下表の「残り」は実戦でその手を考え始めた時点。「手の後の評価」は、それぞれの手を指した後に30秒読み直した値。代案との差が小さい場合や、代案の方が悪い場合は、実戦の手を敗着扱いしない。','',
'|対局|局面|実戦の手|実戦消費／残り|長考で選んだ手|実戦の手の後／代案の後|',
'|---|---:|---|---:|---|---:|']
for s in summary:
 lines.append('|第%d局|%d手目|%s|%.1f秒／%.1f秒|%s|%+d／%+d|'%(s['game'],s['focusPly'],s['played'],s['spentSec'],s['remainingSec'],s['alternative'],s['actualOwn'],s['alternativeOwn']))
lines+=['','### 第4局：時間不足では説明しにくい中盤の悪化','',
'56手目に6四角を打った。実戦では約20秒を使い、その直前に約256秒残っていた。約60秒の再分析は2二飛を選ぶが、手の後を30秒ずつ評価すると実戦の角打ち−514、代案−627。代案の方が良いとは確認できない。5秒・15秒・約60秒で候補手が変わるため、短い分析だけで角打ちを敗着としない。長考時には手を指す前から−528であり、もっと前の構想・攻防を調べる余地がある。','',
'### 第6局：長く読んでも同じ歩打ち','',
'80手目の7三歩打ちを45秒の再分析でも選んだ。実戦では約12秒使い、残り約202秒。手の後の30秒評価は−614。後の98手目の7二銀打ちも15秒の再分析で同じ手となった。選んだ局面では、時間の追加だけで救える明確な一手を確認できていない。','',
'### 第7局：深く読むと不利が見える終盤','',
'139手目の6四歩打ちの直前は、200msの再分析では−7とほぼ互角に見えた。しかし5秒では−524、15秒と約47秒では−936で、7八銀（8九から）を選ぶ。実戦では約15.8秒使用、残り約66秒。',
'手の後を同じ30秒で比較すると、歩打ちは−1558、銀を動かす案は−1435。銀の案の方が123良いが、どちらも不利。5秒比較の差221、15秒比較の差414、30秒比較の差123と幅も揺れる。この一手を変えれば勝てたとは言えない。短い読みで危険を軽く見積もる傾向と、複雑な攻防で判断が安定しにくいことを示す診断例として保存する。',
'約47秒検索でも最後に完了した反復は深さ11・約9.9秒時点だった。残りの検索時間では次の深さが完成しなかった。時間を増やすだけでなく、重要な応手を効率よく読む改善も必要な候補となる。','',
'![第7局の読み時間と評価](output/figures/Time版_第7局の読み時間と評価.png)','',
'### 第9局：一手しかない終盤を敗着と誤認しない','',
'終盤の205手目は王手に対して8八玉しか合法手がない。短い検索ではプラス評価が表示されても、その手の後を調べると詰みが見つかるため、プラスから詰みに落ちたという数値だけでこの玉の手を敗着としない。',
'もっと前の143手目の3八飛（3九から）を45秒でも選び、手の後の30秒評価は−427。実戦消費約4.8秒。選んだ局面で、長考によって改善する代案は見つからなかった。長い攻防のどこで勝ち筋を失ったかは追加調査が必要。','',
'### 第14局：飛車打ちの代案は未確認','',
'90手目の6八飛打ちを45秒でも選んだ。実戦消費約5.2秒、手の後の30秒評価は−509。後の104手目の7九銀打ちも15秒で同じ手を選ぶ。末尾の大きな評価変動は既に不利な局面や詰みの評価が混ざるため、そのまま敗着とは扱わない。','',
'## 時間の残り方','',
'![負けた5局の残り時間](output/figures/Time版_負けた5局の残り時間.png)','',
'|対局|対局中の自分の最小残り時間（考え始め）|最後の自分の手の後の残り|',
'|---|---:|---:|']
for s in summary:lines.append('|第%d局|%.1f秒|%.1f秒|'%(s['game'],s['minBankSec'],s['finalBankSec']))
lines+=['','## 探索の省略を変える追加診断','',
'同じ重要局面を各100万ノードで、標準／LMRなし／ヌルムーブなし／静かな手の枝刈りなし／静止探索の枝刈りなしの5条件で比較した。LMRは後で試す手を浅く読む処理、ヌルムーブは仮に一手休んでも有利なら一部の読みを省く処理。どれを外しても同じ深さまで読む費用は変わるので、手が変わっただけで「この省略が原因」とは言えない。',
'全局に効く原因を確定するための試験ではなく、局面ごとの読みの揺れを調べる追加診断。診断設定は標準実行ファイルへ反映していない。','',
'|対局|標準|LMRなし|ヌルムーブなし|静かな手の枝刈りなし|静止探索の枝刈りなし|',
'|---|---|---|---|---|---|']
for g in abl['games']:
 lines.append('|第%d局|'%g['game']+'|'.join(s['best']+' / '+str(s['score'])+' / 深さ'+str(s['depth']) for s in g['searches'])+'|')
lines+=['','## 次に直すなら','',
'まず診断局面を固定して、重要な相手の応手を早く読む指し手の並べ方や、複雑な局面で読みを省略する条件を一つずつ試す。省略を全面的に外すと深く読めなくなるため、一括で弱めるのは避ける。',
'時間配分は序盤の節約を保ちつつ、評価が急変した局面や次の反復が非常に長い局面を観測できるログを加える案が有用。今回の勝敗ログには実戦の評価とPVがなく、原因判定の限界になった。追加ログは今後の実装候補であり、この調査では対局ツールや標準版を変更していない。',
'評価関数の追加学習は、探索改善で救えない局面を分けてから試す。今回の5敗だけから公開教師データの大量学習を優先する根拠はまだない。',
'今回の局面は開発用の診断集合として保存する。これらの局面で良くなることだけを採用条件にせず、未使用の序盤での対局比較でも強さを確かめる。','',
'## 保存先・再現情報','',
'- 元対局：baseline/time_clock20.json',
'- 全局面・候補・長時間PV：baseline/time_loss_analysis/analysis.json',
'- 集計：baseline/time_loss_analysis/summary.json',
'- 省略の切替診断：baseline/time_loss_analysis/ablation.json',
'- 棋譜：baseline/time_loss_analysis/game_04.usi.txtほか5局',
'- スクリプト：tools/analyze-time-losses.py、tools/confirm-time-losses.py、tools/ablate-time-losses.py、tools/report-time-losses.py',
'- 分析exe：baseline/sailfish2_time_candidate.exe / SHA256 '+sha,
'- 標準exe・R5・アプリ同梱版は変更していない。Git公開なし。','']
(ROOT/'REPORT_Time版5敗の原因調査_20261004.md').write_text('\n'.join(lines),encoding='utf-8')
print('Report written',ROOT/'REPORT_Time版5敗の原因調査_20261004.md')
