# Sailfish 2 Codex ローカル強化記録（2026-10-02）

作業場所: C:/Users/goro0/OneDrive/Documents/shogi/Sailfish2

第5世代 nn_r5_new を標準ネットとして採用し、評価内容を保った計算の高速化を実装した。ネットの幅は H=64 / L2=32、ファイル形式は従来と同じ。

## 実装

- src/nn.rs: 入力の活性化と第2層の4ニューロンをAVXでまとめて計算。積和命令FMAを使わず、8本の加算列と出力の加算順を保つ。AVX非対応環境ではスカラー処理を使う。
- src/search.rs: 手作りの土台評価だけを局面キーで再利用する。NN出力は差分アキュムレータから毎回求め、浮動小数点の丸め方を保つ。キャッシュは探索間でも保持し、評価オプション変更時に無効化する。
- src/position.rs: USIの指し手読み込みは、疑似合法手に含まれることを確認し、その手だけを実際に指して合法性を検証して戻す。全候補を試す処理を減らした。
- nets/default.nn: 旧標準の第3世代から、既存最良の第5世代へ差し替え。独自学習の候補は別ファイルに保存し、標準は第5世代を保持した。
- tools/compare-bench.mjs / bench-positions.json: 6局面の反復速度計測と、指し手・評価値・ノード数の一致確認。3局面はSailfish自身のg7棋譜から抽出した。
- tools/match.mjs: リダイレクト時のログを行単位にし、--jsonで各局の結果と先後入替ペアを考慮した誤差範囲を保存する。

## 速度

Ryzen 7 5700G、1スレッド、Hash=64MB、深さ11。6局面を各7回、A/Bの測定順を交互に変えて実行した。各局面の測定前にSearcherを作り直し、置換表・手の履歴・評価キャッシュを初期化した。初期化と局面読み込みの時間は計測から除外した。

| 項目 | 従来版（第5世代指定） | Codex最終版 |
|---|---:|---:|
| 6局面の合計ノード | 2,406,133 | 2,406,133 |
| 所要時間の中央値 | 5877.0 ms | 5347.9 ms |
| NPS | 409,417 | 449,917 |

探索速度はこのベンチマークで 9.9% 向上。全42回で指し手・評価値・ノード数が一致し、各回の探索内容も一定だった。

開始時の既存AVX変更を含む版との出力計算の単回測定は346.5 ns/回 → 240.1 ns/回。従来lazy実行ファイルのevalbenchは出力計算の値が極端に小さく、出力計算単独の比較には採用していない。

## 対局

初期候補（4ニューロン一括計算＋探索ごとに初期化する土台キャッシュ、第5世代ネット）対 従来第5世代: 600局、309勝278敗13分、52.6%、Elo +18（95%目安 -10〜+46）。最終版はさらに入力のAVX処理と探索間キャッシュ保持を加えているため、この対局結果は初期候補の記録として扱う。

最終高速化版と独自学習候補を、同じ水匠5の2,000ノード制限に各600局で測定した。1手150ms、各1スレッド、Hash=64MB、並列8、NetworkDelay=0。勝率は引分を0.5勝として数える。

| 使用ネット（探索は同じ最終版） | 勝／敗／分 | 勝率 | 先後ペアを考慮した95%目安 |
|---|---:|---:|---:|
| 標準・第5世代 | 277／320／3 | 46.4% | 42.6〜50.3% |
| 独自再ラベル学習候補 | 295／303／2 | 49.3% | 45.5〜53.2% |

候補は標準より2.9ポイント高かった。同じ開始局面の先後ペアで比較した差の95%目安は−2.4〜＋8.3ポイントで、改善は確定できない。50%超の目標も未達。標準ネットは既存第5世代を維持し、候補は実験用として保存した。過去の第5世代48.5%（300局）は別測定の参考値である。

結果: baseline/codex_suisho2000_600.log / .json、baseline/codex_relabel_suisho2000_600.log / .json、差の集計 baseline/codex_relabel_comparison.json。

同じ持ち時間でノード制限なしの水匠5に勝つことや、10,000ノード制限の目標達成を確認した結果ではない。

## 検証

- cargo test --release: 10件成功（ライブラリ9件、再ラベルツール1件）。NNのビット一致、活性化の境界値、差分更新、左右反転、保存読み込み、土台キャッシュの一致と無効化、指し手判定と局面復元、静かな駒打ちの抽出、再ラベルの決定性・1対局1サンプル・元の勝敗維持を検証。
- tools/usi-test.mjs: 最終実行版でThreads=1と4の両方を確認。停止、MultiPV、ponder、秒読み、20回連続go、終了が成功。
- perft: 深さ1〜4は30 / 900 / 25,470 / 719,731。
- 最終版で自己対戦棋譜4局の生成が成功（平均137手）。検証用の棋譜は baseline/codex_smoke_data.txt。
- git diff --check: 成功。既存の未使用mut/importの警告はビルド前から存在する。

## 実行版・控え

- 通常の起動先: target/release/sailfish2.exe。検証用の控え: baseline/sailfish2_codex.exe。USI名は Sailfish 2.1 Codex。第5世代を内蔵し、外部評価ファイルがなくても動作する。
- 開始時に存在した src/nn.rs の未コミットAVX変更を保存: baseline/codex_preexisting_nn.rs。
- その開始時ソースからビルドした控え: baseline/sailfish2_before_codex.exe。
- 旧標準ネット: baseline/default_before_codex.nn（nn_r3.nnとSHA-256一致）。
- 新標準ネットSHA-256: 4e800fc73a799ef2dfe3781aa575df4455b10d6b9dbdbc3de631457fecf006aa。
- 通常のビルドで nn_train.exe / nn_relabel.exe も target/release に配置済み。更新前の通常実行ファイルは baseline/sailfish2_target_before_codex.exe に保存した。
- 実験候補ネット: baseline/nn_codex_relabel.nn。SHA-256: dc1b11b41f8712d4466efc0391b9e2bfdfa56bc9a0b484fabf3fffff585b4565。
- ソース変更は未コミットで保存。GitHubへのpush・タグ付け・公開リリースは行っていない。

## 独自の学習改善

ネットを大型化せず、短い探索の評価が不安定な局面に教師の計算を集中する方法を実装した。

- nn_relabel は既存の自己対戦棋譜から1対局1局面を選ぶ。約半分は静的評価と元の探索値の差が大きい局面、残りは通常の静かな局面から一様抽出する。
- Sailfish自身で80,000ノードの深い探索を行い、教師値を付け直す。USI名を確認し、別エンジンを教師に指定した場合は拒否する。元の対局結果を維持し、選択局面より前の手は学習対象から外す。
- 1対局から1サンプルに限定し、学習・検証の分割で同一対局が混ざることを避ける。ランダム序盤と王手になる駒打ちを除外する。
- nn_train に --quiet-drops（王手にならない駒打ちを含める）と --skip-opening（ランダム序盤を除外）を追加。初期モデルも最良チェックポイントに含め、検証が悪化した場合は元のモデルを保持する。

まずg7+g6の従来ラベルで抽出条件だけを変え、8エポック学習した。3,109,907局面、検証lossは初期0.62031から改善せず、第5世代をそのまま保持した。この試験の検証末尾g6は第5世代の学習データと重なるため、汎化性能の判定には使わない。

次に第5世代がまだ学習していないg7のみから8,000局面を再ラベルした。教師探索は8並列で243秒。詰みなどを除いた7,640局面を学習7,258／検証382に分割し、第5世代を初期値として24エポック、λ=0.7、lr=0.00005で学習した。検証lossは0.59572→0.59041、cp誤差は333.4→323.8になった。検証lossの改善は対局での棋力向上を保証しないため、上記600局で確認した。

教師データ: data/codex_relabel_g7_8000.txt。ログ: baseline/codex_relabel.log / codex_relabel_train.log / codex_filters_train.log。水匠の棋譜・評価値は学習に使っていない。

再現例（出力名は既存ファイルを上書きしない新しい名前にする）:

```powershell
.\target\release\nn_relabel.exe .\target\release\sailfish2.exe .\baseline\nn_r5_new.nn .\data\g7.txt .\data\codex_relabel_repeat.txt 8000 80000 8
.\target\release\nn_train.exe .\baseline\nn_codex_repeat.nn 24 0.7 .\data\codex_relabel_repeat.txt --init .\baseline\nn_r5_new.nn --lr 0.00005 --batch 1024 --quiet-drops --skip-opening
$taskCandidate = (Resolve-Path 'baseline\nn_codex_repeat.nn').Path
node --no-warnings tools/match.mjs "target/release/sailfish2.exe|NetworkDelay=0,Threads=1,EvalFile=$taskCandidate" "C:\Users\goro0\OneDrive\Documents\shogi\水匠5(211123)\水匠5\Suisho5-AVX2.exe|Threads=1,NodesLimit=2000,NetworkDelay=0,USI_Hash=64" 600 150 8 --json baseline/codex_candidate_repeat.json
```

## 再現コマンド

```powershell
Set-Location 'C:\Users\goro0\OneDrive\Documents\shogi\Sailfish2'
$taskEval = (Resolve-Path 'baseline\nn_r5_new.nn').Path
node tools/compare-bench.mjs "baseline/sailfish2_lazy.exe|EvalFile=$taskEval" "baseline/sailfish2_codex.exe" 11 7
node --no-warnings tools/match.mjs "baseline/sailfish2_codex.exe|NetworkDelay=0,Threads=1" "C:\Users\goro0\OneDrive\Documents\shogi\水匠5(211123)\水匠5\Suisho5-AVX2.exe|Threads=1,NodesLimit=2000,NetworkDelay=0,USI_Hash=64" 600 150 8 --json baseline/codex_suisho2000_repeat.json
```

対局条件: 各エンジン1スレッド、1手150ms、並列8。水匠5は測定相手として使用し、その評価値や棋譜を学習に使用していない。

採用した速度計測ログ: baseline/codex_bench_fresh.json。USIログ: baseline/codex_usi_single.log / codex_usi_smp.log。
