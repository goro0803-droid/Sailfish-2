from pathlib import Path
import json,numpy as np,hashlib,datetime
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'data/plot_runtime'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
R=Path(__file__).resolve().parent.parent;D=R/'baseline/teacher_coverage_20261004';s=json.loads((D/'coverage_summary.json').read_text(encoding='utf-8'));split=json.loads((D/'split_audit.json').read_text(encoding='utf-8'))
font=FontProperties(fname='C:/Windows/Fonts/meiryo.ttc');plt.rcParams['font.family']=font.get_name()
a=np.loadtxt(D/'window_0_features.tsv',skiprows=1);names=(D/'window_0_features.tsv').read_text().splitlines()[0].split('\t');ix={k:i for i,k in enumerate(names)};tr=a[a[:,ix['train_eligible']]==1];bins=[0,41,81,121,161,10000];lab=['40手まで','41〜80手','81〜120手','121〜160手','161手以降']
fig,ax=plt.subplots(figsize=(10,4.8));x=np.arange(5)
for j,(data,title,color) in enumerate([(a,'教師サンプル（選別前）','#4292c6'),(tr,'現在の学習条件で残る局面','#e58b35')]):
 y=np.histogram(data[:,ix['ply']],bins=bins)[0]/len(data)*100;bars=ax.bar(x+(j-.5)*.36,y,.36,label=title,color=color)
 for b,v in zip(bars,y):ax.text(b.get_x()+b.get_width()/2,b.get_height()+.4,f'{v:.1f}%',ha='center',fontsize=10)
ax.set_xticks(x,lab);ax.set_ylim(0,55);ax.set_ylabel('各グループの中での割合（%）');ax.set_title('学習用に選別すると、後半の局面が薄くなる');ax.legend();fig.tight_layout();fig.savefig(D/'phase_distribution.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4.6));groups=s['lossPerGame'];bars=ax.bar([f"第{g['game']}局" for g in groups],[g['eligible'] for g in groups],color='#4292c6')
for b,g in zip(bars,groups):ax.text(b.get_x()+b.get_width()/2,b.get_height()+.35,f"{g['eligible']} / 21局面",ha='center')
ax.axhline(21,color='#888',linestyle='--');ax.set_ylim(0,24);ax.set_title('重点局面の前後で、現在の学習条件に残る局面');ax.set_ylabel('局面数（各局、Sailfishの手番21局面）');fig.tight_layout();fig.savefig(D/'loss_region_retention.png',dpi=170);plt.close(fig)
rows=[]
for w in s['windows']:
 rows.append(f"| {w['offset']:,}件目から | {w['all']['n']:,} | {w['train']['n']:,} | {w['all']['plyQuantiles'][1]:.0f}→{w['train']['plyQuantiles'][1]:.0f} | {w['all']['late121plus']*100:.2f}%→{w['train']['late121plus']*100:.2f}% |")
cr=[]
for c in s['critical']:
 f=c['features'];why='条件に残る' if f['train_eligible'] else ('王手の駒打ちで除外' if f['drop_move'] and f['gives_check'] else '駒取りで除外')
 cr.append(f"| 第{c['game']}局 | {c['ply']}手目 | `{c['move']}` | {why} | {c['similarBuckets']['beforeFilters']['n']:,}→{c['similarBuckets']['existing']['n']:,} | {c['similarBuckets']['existing']['fraction']*100:.2f}% |")
report=f'''# 教師サンプルの偏りと、負け局面に近い攻防の調査（2026-10-04）

**今回の範囲では、サンプルの読み取り位置より、学習用の選別によって後半・攻防の局面が薄くなる点が目立った。** ただし、それが負けた直接の原因だと証明したわけではない。追加学習・新たな対局・標準の差し替えは行っていない。

## 何を調べたか

公開教師は `{R.parent/'public_teacher_by_mekitto.7z'}` に移動済み。以前展開した内部アーカイブを読み取り専用で使用し、0件目・100万件目・500万件目から各4万件、合計12万局面を比較した。元データは181,867,500件なので、全体から均等に選んだ調査ではない。範囲は先頭504万件以内で、全体の偏りや完全なCRC検査は未確認。

負けた5局は、前回調査の重点局面と、その前後20手以内のSailfishの手番を比較した。各局21局面、合計105局面。これら5局は、5分＋1手10秒の**ローカルTime版対旧版の比較対局**であり、floodgateの5敗とは別である。同じ対局内の局面は関連するため、105件を独立した105局の証拠とは扱わない。

元データの説明は「複数エンジン・2018年公開定跡を使用、主に深さ12・一部14、中盤多め」と記載されている。棋譜ID・生成エンジンIDが各レコードにはなく、特定の戦型やソフトへの偏りは今回の形式だけでは確定できない。

## 1. 先頭4万件だけが特殊なのか

| 読み取り位置（0開始） | 調査件数 | 現在の学習条件で残る件数 | 手数の中央値：選別前→後 | 121手目以降の割合：選別前→後 |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

3か所の手数・選別率・持ち駒などの分布はよく似ており、この範囲では先頭4万件だけに大きな偏りは見つからなかった。「シャッフル済み」というファイル名だけで全体を代表すると断定はしない。

先頭4万件の先手比率は{s['windows'][0]['all']['sideBlack']*100:.1f}%、手番側が最終的に勝つ比率は{s['windows'][0]['all']['resultWin']*100:.1f}%。既存学習用・検証用には同一局面と左右反転局面の重複は0件。ただし対局単位で完全に分離できているかは、棋譜IDがなく未確認。

## 2. 選別によって何が減るか

既存の4万件は、最初に手数・点数・王手を受けているかで選別され、36,632件となる。そこから教師の手が成り、駒取り、王手の駒打ちであるものを除くと22,960件（学習21,784＋検証1,176）が残る。

| 先頭4万件での除外・採用（重複しない分類） | 件数 |
|---|---:|
| 最初の条件：16手未満・点数範囲外・王手を受けている | 3,368 |
| 教師の手が成り | 2,913 |
| 教師の手が駒取り（成りは上で集計済み） | 9,542 |
| 教師の手が王手の駒打ち | 1,217 |
| 学習・検証対象として残る | 22,960 |

![手数ごとの分布]({(D/'phase_distribution.png').as_posix()})

121手目以降は元の4万件に3,868局面あり、そのうち残るのは1,434局面（37.1%）。全局面の採用率57.4%より低い。元々説明どおり中盤が多いデータを、選別によってさらに早い局面へ寄せている。

重要な区別：**攻防の局面が全部なくなるわけではない。** 選別後でも、手番側に駒取り・王手の選択肢がある局面は残る。教師が選んだ手が駒取り等だと除外される仕組み。また「王手をかける盤上の非捕獲・不成の手」は残り得る。王手をかける手がすべて除外されるわけではない。

## 3. 負けた局面に近い条件はどれだけあるか

![重点局面の前後で残る数]({(D/'loss_region_retention.png').as_posix()})

5重点局面のうち第7局・第9局・第14局は、当時の実際の手を教師の手とみなした場合、現在の学習条件では外れる。周辺105局面も残るのは45件（42.9%）。ただし、実際の手は外部教師の最善手とは限らず、これは選別との関係を調べる診断である。5敗をそのまま教師データとして学習に入れていない。

| 重点局面 | 手数 | 実際の手（USI） | 現行条件での扱い | 近い条件の件数：選別前→後 | 選別後全体に占める割合 |
|---|---:|---|---|---:|---:|
{chr(10).join(cr)}

「近い条件」は、手数±20、手番側の玉に隣接する升への相手の利きの数（0〜1/2〜4/5〜8）、相手の飛角の持ち駒の有無、盤上の成駒数（0/1〜2/3以上）を合わせた分類。盤面・戦型・具体的な手筋が同じという意味ではない。玉周辺の利きも、逃げ道の安全性や詰みを証明しない。

第7局と第9局に近い条件は約1%前後と薄いが、235件・182件存在する。第14局の条件も303件ある。「似た攻防がまったくない」ではなく、**後半の難しい条件は比率が低く、さらに教師の手によって選別される**という結果。

離れた2か所を加えても、選別後の第7局条件は約0.97%、第9局条件は約0.73%。単に先頭以外から同じ件数を読むだけで、この比率が大きく改善するとは考えにくい。

## 4. 調査で見つかった分割処理の問題

以前の抽出スクリプトは、元局面を圧縮SFEN、左右反転局面を別の展開表記として比較していた。そのため「左右反転も同じグループにする」という処理が成立していなかった。以前の説明をここで訂正する。

統一したSFEN表記で既存36,632件を再点検した結果、学習と検証の同一・左右反転局面の重複は0件で、今回調べた既存分割に実際の重複は見つからなかった。元スクリプトを保存して処理を修正し、1,001局面・盤面例で二重反転、グループキー、分割用ハッシュの一致を検証済み。既存の教師データと過去の実験結果は上書きしていない。

## 次に一つだけ試すなら

**静かな手という現行条件を維持し、121手目以降の比率だけを増やす比較**が、今回の結果に直接対応する。まずは既存の約6%から15%を目安にした候補が考えられる。

同じ新規データプールから、総件数・学習回数・教師尺度K600・学習率などを揃えて「元の手数分布の対照」と「後半比率を増やした候補」を作る。正しい左右反転グループで学習/検証を分け、R5へ追加学習し、別の開始局面で対局検証する。15%は効果を確認した値ではなく、次の実験用の仮説である。

王手・駒取り・成りの除外を一度に外すと、駒交換の途中の評価が静的評価へ混ざり、別の問題を起こし得る。まず手数分布だけを変え、改善がなければ棄却する。その後、攻防を静かな局面まで進めて再評価する方法などを、別実験として検討する。

## 記録と検証

- 診断値：`baseline/teacher_coverage_20261004/coverage_summary.json`、`split_audit.json`。
- 12万件の教師の手はすべて合法、勝敗値も有効。既存の選別後22,960件という件数を再現した。
- 合法手数・駒取り・王手の選択肢は、別の将棋ルール実装cshogiで21局面を照合して一致。
- 診断ツール：`coverage_audit`、`tools/audit-teacher-coverage.py`、`tools/summarize-teacher-coverage.py`、`tools/audit-teacher-coverage-split.py`。
- 今回は追加学習・対局を開始していない。現在の標準exe3か所とR5のSHA256不変を確認済み。
'''
paths=[R/'target/release/sailfish2.exe',R.parent/'ShogiArena/engines/sailfish2.exe',R.parent/'ShogiArena/dist/win-unpacked/resources/engines/sailfish2.exe',R/'nets/default.nn'];hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
assert all(hashes[str(p)]=='1129d6fee951d405246745b065724d32c5c908f2c4748b20af7fb660c5457f2c' for p in paths[:3]);assert hashes[str(paths[-1])]=='4e800fc73a799ef2dfe3781aa575df4455b10d6b9dbdbc3de631457fecf006aa'
(R/'REPORT_教師サンプル偏りと攻防不足_20261004.md').write_text(report,encoding='utf-8')
(D/'completion.json').write_text(json.dumps({'status':'complete_audit_no_training','finished':datetime.datetime.now().astimezone().isoformat(),'teacherRecords':120000,'lossPositions':105,'standardHashes':hashes,'nextHypothesis':'late quiet position ratio only, control dataset size and sampling pool','sourceOuter':str(R.parent/'public_teacher_by_mekitto.7z')},ensure_ascii=False,indent=2),encoding='utf-8')
print('report and figures written; standard hashes unchanged')
