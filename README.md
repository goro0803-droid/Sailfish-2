# Sailfish

Rust で書かれた USI プロトコル対応の将棋エンジンです。
[将棋アリーナ (Shogi Arena)](../../../shogi-arena) に同梱されています。ShogiGUI、将棋所、ShogiHome などにも登録して使えます。

評価関数は手作りです（機械学習の評価関数ファイルは使いません）。人が遊んで楽しい、ほどよい強さを目指しています。

## ビルド

```
cargo build --release
```

実行ファイルは `target\release\sailfish.exe` にできます。

## GUI への登録（ShogiGUI の例）

1. ShogiGUI のメニュー「ツール」→「エンジン管理」→「追加」
2. `sailfish.exe` を選ぶ
3. 「対局」→「対局」で先手か後手に Sailfish を選んで開始

### エンジンオプション

| 名前 | 既定値 | 説明 |
|---|---|---|
| USI_Hash | 256 | 置換表のサイズ (MB) |
| NetworkDelay | 120 | 時間切れを防ぐため、思考時間から差し引く余裕 (ms) |
| MultiPV | 1 | 候補手をいくつ読むか (1〜10) |

## 構成

| ファイル | 内容 |
|---|---|
| `src/types.rs` | 駒・指し手の表現、方向テーブル、Zobrist ハッシュ |
| `src/position.rs` | 局面、SFEN 読み込み、指し手生成、合法性判定、SEE（駒の取り合いの損得） |
| `src/eval.rs` | 評価関数（駒割り、玉の危険度、大駒の働き、攻め駒・守り駒の位置、取られそうな駒） |
| `src/search.rs` | 反復深化 PVS、静止探索、置換表、ヌルムーブ、LMR、枝刈り、キラー／ヒストリー／カウンター手、MultiPV、時間管理 |
| `src/tt.rs` | 置換表 |
| `src/main.rs` | USI プロトコル処理（ponder / infinite / stop 対応） |
| `tools/match.mjs` | エンジン同士の自動対局でレート差を測るツール（将棋アリーナのルール実装を使う） |
| `tools/bench.mjs` | 速度の計測 |

## デバッグ用コマンド

エンジンを直接起動して入力できます。

- `d` 盤面表示
- `moves` 合法手一覧
- `eval` 静的評価値
- `perft N` 深さ N までの合法手ノード数（初期局面: 30 / 900 / 25470 / 719731 / 19861490）

## ライセンス

MIT License © 2026 dax（[LICENSE](LICENSE)）
