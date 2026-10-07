# floodgate棋譜によるSailfish2学習実験（2026-10-02）

ユーザーが東京大学floodgateの公開棋譜を例外として許可したため、wdoor2025.7zを取り込んだ。外部エンジンの評価値コメントは無視し、教師値をSailfish自身の探索で作成した。標準ネットの変更は対局の検証結果で判断する。

## データと許可の範囲

- 元ファイル: C:/Users/goro0/Downloads/wdoor2025.7z（351,630,898 bytes）。元ファイルは変更していない。
- SHA-256: 423504903f211316ec40f9c3d8dcc2e5faf392fc4d1334ede1489242a9d09baa。
- 初回は128,843局を読み、重複5,696、時間切れなどの終局理由2,763、40手未満・512手超9,230、未完了1,338を除外。平手の初期配置と全指し手の合法性を確認した。
- 使用可能109,816局から2.4万局を決定的なreservoir samplingで抽出。全12か月を含む。
- 抽出後、アーカイブ内のfloodgate以外の14局を特定して除外。承認対象の候補は23,986局。変換ツールもEVENT名がwdoor+floodgate-で始まることを必須に修正した。
- 教師作成後にも、除外14局と一致する指し手列の6サンプルを学習対象から取り除いた。承認された教師サンプルは11,994。
- 評価値は取り込み時に仮の0に置き換える。この取り込み直後のファイルをそのまま学習させてはいけない。教師探索済みの approved_labels ファイルを使う。

## 実装

- src/bin/floodgate_import.rs: tarを使って7zをストリーム読み込みする。全解凍は不要。CSAからUSIへ変換し、駒種・成り・先後・合法性・初期配置・終局を検査する。元の評価値は取り込まない。棋譜のEVENT一覧も保存する。
- nn_relabel --uniform: 仮評価値に依存せず、各対局の静かな局面から一様抽出する。最初の20手を除き、180手までを候補にする。1対局1局面をSailfish自身の80,000ノード探索で再評価する。8並列、12,000局面の教師作成は421秒。
- nn_train --dedup: 指し手順ではなく盤面・持ち駒・手番の完全な組を使って、学習データ全体の同一局面を除外する。別の対局・異なる手順でも同じ局面なら重複となる。
- 混合後、同一局面157件を除いた。学習または検証の有効局面が空なら、学習を開始せずエラーで終了する。従来の抽出方法はオプションを指定しない場合に維持する。

## 学習

第5世代を初期値に、以前のg7自己対戦の再ラベル8,000対局とfloodgateの承認済み11,994サンプルを混ぜた。λ=0.7、lr=0.00005、batch=1024、32エポック、左右反転あり、静かな駒打ちを含め、序盤と重複を除外した。

| 項目 | 値 |
|---|---:|
| 有効な全局面 | 18,368 |
| 学習／検証 | 17,481／887 |
| 初期／最良検証loss | 0.62227／0.62058 |
| 初期／最良loss時のcp誤差 | 156.7／162.8 |

検証は対局単位で末尾5%を分けた。末尾にfloodgateデータを置き、その中の順番はシャッフル済み。学習側と検証側に同一局面を入れない。事前学習ネットの全世代に対する未出局面であることを保証する分割ではない。

lossは勝敗と探索評価を混ぜた目標に対する値である。lossは改善した一方でcp誤差は増えたため、これだけで棋力向上とは判断しない。

候補: baseline/nn_floodgate2025_mix.nn。SHA-256: f2c277e76cb5beabc8c848397667cf22db37f0a1046d7ffc69940ed0b1b8cbbb。

## 教師評価を重くした追加候補

最初のλ=0.7の候補はcp誤差が増え、水匠戦でも標準の以前の測定を上回らなかった。対局結果と探索教師の混ぜ方を確認するため、同じデータ・初期値・学習率・32エポックでλだけ0.9に変更した追加候補を一つ作った。

検証lossは0.62691→0.62638、cp誤差は156.7→154.1。9エポック目が最良で、そのネットを保存した。λが異なる二つのlossの絶対値は比較しない。新しい教師データや検証データは追加していない。

候補: baseline/nn_floodgate2025_teacher90.nn。SHA-256: 1ccdef3bf7753f77da889187e52ec7b98317e5842692a9175651284b8b34edc3。

この追加案は最初の対局を見てから選んだ探索的な実験であり、独立した最終認定試験ではない。

## 対局検証

現在の第5世代標準ネットと、同じ最終探索エンジンで比較する。1手150ms、各1スレッド、Hash64MB、8並列。開始局面を先後入替で指す。

| ネット | 勝／敗／分 | 勝率 | 先後ペアの95%目安 |
|---|---:|---:|---:|
| 標準・第5世代（以前の測定） | 277／320／3 | 46.4% | 42.6〜50.3% |
| 自己対戦のみの再ラベル候補（以前の測定） | 295／303／2 | 49.3% | 45.5〜53.2% |
| floodgate混合・教師70% | 272／327／1 | 45.4% | 41.4〜49.5% |
| floodgate混合・教師90% | 249／349／2 | 41.7% | 37.7〜45.6% |

全て水匠5（Threads=1、NodesLimit=2000、NetworkDelay=0、Hash64MB）と600局、Sailfishも1スレッド・1手150ms・Hash64MB、8並列。引分は0.5勝。以前の測定と同じ開始局面番号の先後ペアを対応させて比較した。別の測定回であり、実行時刻や探索履歴のばらつきまで一致させた比較ではない。

標準との差は教師70%が-1.0ポイント（95%目安 -6.6〜4.6ポイント）、教師90%が-4.8ポイント（同 -10.3〜0.8ポイント）。

教師70%候補の標準との直接対局300局は153勝142敗5分、51.8%、先後ペアの95%目安 46.2〜57.4%、Elo +13（通常の95%目安 -27〜+52）。改善を確定できない。教師90%候補の標準との直接対局は未実施。

今回の二案では水匠戦の改善を確認できなかった。どちらも標準ネットには採用しない。nets/default.nn と通常の target/release/sailfish2.exe は第5世代を維持する。候補・教師データ・ツールは次の実験に使えるよう保存した。棋譜利用の可否と、この具体的な学習方法で棋力が向上したかは別に評価する。

50%超の目標は未達。水匠の10,000ノード制限と同時間・無制限の対局は今回未実施。結果の悪い測定も含めて記録した。

## 動作確認

cargo test --release: 合計13件成功（ライブラリ9、取り込み2、再ラベル1、学習の重複除去1）。取り込みでは不正な成り、誤った手番、時間切れ、未完了、許可対象外のEVENT、元の評価値を無視した勝敗を確認。学習では異なる手順から同じ局面になった例を使って、重複が検証側に残らないことを確認した。

## 再現例

新しい出力名を指定する。既存の教師・ネット・対局ログを上書きしない。

```powershell
Set-Location 'C:\Users\goro0\OneDrive\Documents\shogi\Sailfish2'
cargo build --release
.\target\release\floodgate_import.exe 'C:\Users\goro0\Downloads\wdoor2025.7z' data/floodgate_repeat_positions.txt 24000
.\target\release\nn_relabel.exe target/release/sailfish2.exe baseline/nn_r5_new.nn data/floodgate_repeat_positions.txt data/floodgate_repeat_labels.txt 12000 80000 8 --uniform
.\target\release\nn_train.exe baseline/nn_floodgate_repeat.nn 32 0.7 data/codex_relabel_g7_8000.txt data/floodgate_repeat_labels.txt --init baseline/nn_r5_new.nn --lr 0.00005 --batch 1024 --quiet-drops --skip-opening --dedup
```

修正後の取り込みはfloodgate以外を抽出前に除くため、初回の取り込み・事後除外とは選択される対局が変わる。今回そのものを再学習するには data/floodgate2025_approved_labels.txt を使う。

追加案の学習と水匠戦の再現例:

```powershell
.\target\release\nn_train.exe baseline/nn_floodgate_teacher90_repeat.nn 32 0.9 data/codex_relabel_g7_8000.txt data/floodgate2025_approved_labels.txt --init baseline/nn_r5_new.nn --lr 0.00005 --batch 1024 --quiet-drops --skip-opening --dedup
$taskCandidate = (Resolve-Path 'baseline/nn_floodgate_teacher90_repeat.nn').Path
node --no-warnings tools/match.mjs "target/release/sailfish2.exe|NetworkDelay=0,Threads=1,EvalFile=$taskCandidate" "C:\Users\goro0\OneDrive\Documents\shogi\水匠5(211123)\水匠5\Suisho5-AVX2.exe|Threads=1,NodesLimit=2000,NetworkDelay=0,USI_Hash=64" 600 150 8 --json baseline/floodgate_teacher90_repeat.json
```

## 記録と実行先

- 記録: baseline/floodgate2025_import.log、floodgate2025_relabel.log、floodgate2025_train.log、floodgate2025_teacher90_train.log、floodgate2025_audit.json。
- 対局: baseline/floodgate2025_vs_r5_300.log / .json、floodgate2025_suisho2000_600.log / .json、floodgate2025_teacher90_suisho2000_600.log / .json。差の集計: baseline/floodgate2025_comparison.json。
- 承認された元棋譜候補: data/floodgate2025_approved_positions.txt / .manifest.tsv。
- 取り込みツール: target/release/floodgate_import.exe。
- 再ラベル・学習ツール: target/release/nn_relabel.exe、nn_train.exe。
- 初回の未選別候補と教師は名前を区別して保存している。学習には approved_labels のみを使った。
- ソースと候補をローカルに保存。GitHubへのpush、タグ、公開リリースは行っていない。
