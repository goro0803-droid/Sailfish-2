from pathlib import Path
import json,hashlib,datetime,sys
R=Path(__file__).resolve().parent.parent;D=R/'baseline/late_quiet_20261004';S=D/'state.json';s=json.loads(S.read_text(encoding='utf-8'));assert s['status']=='validation_complete','Require complete matches; avoid duplicate handover'
m=json.loads((D/'data_manifest.json').read_text(encoding='utf-8'));v=json.loads((D/'validation_metrics.json').read_text(encoding='utf-8'));o=json.loads((D/'opening_manifest.json').read_text(encoding='utf-8'))
paths=[R/'target/release/sailfish2.exe',R.parent/'ShogiArena/engines/sailfish2.exe',R.parent/'ShogiArena/dist/win-unpacked/resources/engines/sailfish2.exe',R/'nets/default.nn'];hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};assert all(hashes[str(p)]=='1129d6fee951d405246745b065724d32c5c908f2c4748b20af7fb660c5457f2c' for p in paths[:3]);assert hashes[str(paths[-1])]=='4e800fc73a799ef2dfe3781aa575df4455b10d6b9dbdbc3de631457fecf006aa'
s['status']='complete_candidate_only' if s['strengthSupport'] else 'complete_not_adopted';s['standardHashes']=hashes;s['standardUnchanged']=True
sys.path.insert(0,str(R/'data/plot_runtime'));import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams['font.family']=FontProperties(fname='C:/Windows/Fonts/meiryo.ttc').get_name()
fig,ax=plt.subplots(figsize=(8,4.5));bars=ax.bar(['自然分布の対照','後半15%の候補'],[m['controlLateRatio']*100,m['candidateLateRatio']*100],color=['#4292c6','#e58b35'])
for b,n in zip(bars,[m['controlLate'],m['candidateLate']]):ax.text(b.get_x()+b.get_width()/2,b.get_height()+.4,f'{b.get_height():.2f}%\n{n:,}局面',ha='center')
ax.set_ylim(0,20);ax.set_ylabel('学習用21,784局面のうち121手目以降の割合（%）');ax.set_title('今回変えたのは、後半の静かな局面の比率');fig.tight_layout();fig.savefig(D/'late_ratio.png',dpi=170);plt.close(fig)
labels={'late15_vs_control_screen200':'後半15% 対 自然分布の対照（初期）','late15_vs_R5_screen200':'後半15% 対 現在のR5（初期）','late15_vs_control_confirm400':'後半15% 対 自然分布の対照（追加）','late15_vs_R5_confirm400':'後半15% 対 現在のR5（追加）'}
rows=[]
for x in s['matches']:
 z=x['results'];lo,hi=x['paired95'];rows.append(f"| {labels[x['label']]} | {x['games']} | {x['moveMs']}ms | {z['w']}勝{z['l']}敗{z['d']}分 | {x['score']*100:.2f}% | {lo*100:.2f}〜{hi*100:.2f}% |")
verdict='追加確認で短時間対局の有望性を確認した。ただし長い持ち時間での検証が必要なため、候補として別保存し標準は維持した。' if s['strengthSupport'] else '今回の条件では強くなったと確認できず、標準への採用を見送った。対照との初期比較が49.25%で事前の次段階条件に届かなかったため、追加400局ずつの確認は行っていない。'
vr=[]
for key,label in [('R5','現在のR5'),('control','自然分布の対照'),('late15','後半15%の候補')]:
 a=v[key]['all'];l=v[key]['121plus'];vr.append(f"| {label} | {a['teacher_target_ce']:.5f} | {l['teacher_target_ce']:.5f} | {l['teacher_mae']:.1f}点 |")
text=f'''# 原因別強化②：後半の静かな局面を増やす実験（2026-10-04）

**{verdict}** 全{sum(x['games'] for x in s['matches'])}局の比較を完了。使用中のSailfish2 TimeとR5ネットは変更していない。

## なぜ試したか

前回の教師12万局面の調査では、学習用の選別によって121手目以降の割合が約9.7%から約6%に減っていた。負け局面に近い後半の攻防も薄めだった。ただし、これが直接の敗因と証明されたわけではない。今回は「後半の静かな局面を増やすと強くなるか」を一つの仮説として試した。

![後半比率の比較]({(D/'late_ratio.png').as_posix()})

## 一つだけ変えた内容

先頭・100万件目・500万件目から抽出済みの教師12万件から、従来と同じ条件の静かな局面を選び、重複と左右反転をまとめた68,616件を元にした。

| 条件 | 自然分布の対照 | 後半15%の候補 |
|---|---:|---:|
| 学習局面の総件数 | 21,784 | 21,784 |
| 121手目以降の件数 | 1,327 | 3,268 |
| 121手目以降の割合 | 6.09% | 15.00% |
| 共通の検証局面 | 2,000 | 同じ2,000 |
| 学習開始ネット | R5 | R5 |
| 学習回数 | 24周 | 24周 |
| 教師評価の混合比率 | 90% | 90% |
| 教師/出力の換算尺度 | K600/K600 | K600/K600 |
| 学習率・バッチ | 0.00005・1024 | 同じ |

静かな局面のフィルタ、左右反転学習、2スレッドも同じ。前回不採用の教師K465は使用せず、探索機能も追加していない。

両方に共通する学習局面は19,843件。後半を増やすため、早い局面1,941件を後半の1,941件へ置き換えた。局面のコピーによる水増しは行っていない。

後半比率を増やすと、点数差が大きい局面などの割合も自然に変わる。平均絶対教師点数は対照471.7点、候補519.0点だった。したがって「手数そのものだけの因果効果」を証明する実験ではなく、後半のデータを多く選ぶ方針の比較である。

## 検証データの分離

正しい圧縮SFENの左右反転キーを作り、SHA256のグループで先に学習と検証を分けた。検証用グループ3,317件はすべて学習から外し、そのうち2,000件を共通検証に使用。同一・左右反転局面の学習/検証の重複は0。後半の検証局面は140件と少なく、数値の安定性には限界がある。

元教師に棋譜IDがないため、同じ対局由来の別局面が両方に含まれないことまでは保証できない。今回の元データも全アーカイブからの均等抽出ではない。

## 教師との一致はどうなったか

実際の整数ネット出力から計算した誤差。小さいほど教師に近いが、これだけで強さは判断しない。

| ネット | 全2,000局面の学習目標誤差 | 後半140局面の学習目標誤差 | 後半の平均教師点数差 |
|---|---:|---:|---:|
{chr(10).join(vr)}

後半140局面では、候補は対照より少し改善した。全体の差は小さい。これを対局での棋力向上へそのまま読み替えない。

## 対局で確認した結果

同じ標準Time実行ファイルを使用し、読み込むネットだけ変更。1スレッド、Hash64、NetworkDelay0、並列4。得点率は引分を半勝として含む。

| 比較 | 局数 | 1手時間 | 後半15%候補の勝敗 | 得点率 | 先後ペア95%の目安 |
|---|---:|---:|---|---:|---:|
{chr(10).join(rows)}

初期比較は同じ100種類の合法開始盤面で先後を交替。候補が対照とR5の両方で得点率50%以上なら、初期と重複しない200種類の開始盤面で、各400局100msの追加確認へ進む手順を対局前に保存した。追加の両比較で95%目安の下限が50%を超えた場合だけ有望判定する。

開始盤面は4手のランダムな歩突きに限られ、多様な定跡・長い持ち時間・floodgateの強さは別途確認が必要。開始盤面300種類は別の将棋ルール実装cshogiでも合法性と重複0を確認した。反則による終局は0。

## 保存と今後

候補：`baseline/late_quiet_20261004/late15.nn`。対照：同フォルダ`control.nn`。学習データ・ログ・対局JSON・条件とハッシュは同フォルダの`data_manifest.json`、`validation_metrics.json`、`opening_manifest.json`、`state.json`へ保存。

実験手順：`tools/prepare-late-quiet.py`、`tools/prepare-late-openings.mjs`、`tools/test-late-quiet.py`。対局ツールに事前指定の開始局面を渡す`--openings-json`を追加し、従来のseed方式は維持した。

今回の初期比較は各200局・50msで、両方の95%目安が50%を含む。差がないことを証明したわけでも、弱くなったと断定できる結果でもない。今回の結果は「後半15%・この規模・この学習設定」の評価であり、後半データや公開教師全体が無意味という結論ではない。次の変更はこの結果を踏まえて一つだけ選ぶ。採用しなかった探索・学習の変更をまとめて有効にしない。
'''
report=R/'REPORT_原因別強化2_後半静かな局面_20261004.md';report.write_text(text,encoding='utf-8')
h=R.parent/'引継ぎ_Sailfish2_Codex.md';body=h.read_text(encoding='utf-8');assert '## 23. 原因別強化②' not in body
note='\n\n## 23. 原因別強化②：後半静かな局面比率（2026-10-04）\n'+verdict+'\n同一12万poolから静かな局面条件を維持。正しいmirror canonicalで重複2件除外pool68616、SHAグループ検証予約3317件全て学習から除外、共通検証2000（後半140）。各21784件、121手以降1327=6.09%対3268=15.00%、共通19843件、早い1941件を後半へ置換。K600/lambda.9/lr.00005/batch1024/24ep/2threads/R5/左右反転同一。局面水増しなし。game-disjoint未保証。\n'+ '\n'.join(rows)+'\n全対局完了、反則0。初期100盤面・追加用200盤面は同一盤面重複0、cshogi合法性独立確認。対局ツール--openings-json追加（旧seed方式維持）。採用条件は両初期>=50%で追加各400局100ms、両追加paired95下限>50%。標準exe3か所/R5既知ハッシュ不変。候補/対照/TSV/JSON/logsはbaseline/late_quiet_20261004へ。状態state.json='+s['status']+'。詳細Sailfish2/'+report.name+'。監視/公開なし。\n'
h.write_text(body+note,encoding='utf-8')
p=R/'README.md';body=p.read_text(encoding='utf-8');p.write_text(body+'\n- [原因別強化②：後半の静かな局面](REPORT_原因別強化2_後半静かな局面_20261004.md)\n',encoding='utf-8');S.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8');print(s['status']);print('report/handover complete')
