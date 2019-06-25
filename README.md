# README
作業ディレクトリは全て本`README.md`と同一階層である前提で書いています。

## テスト
```bash
cargo test
```

## 実行方法
```
cargo run --release [タスク名]
```
の形式になっています。以下に各タスク名を記載します。

### 課題1
- `task_01`　$h_G$を計算します。計算するグラフは`src/main.rs`内`task_01()`で定義されています。

### 課題2
- `task_02`　固定されたグラフ(`task_02()`内で定義)を学習します。

### 課題3
いずれも2000のデータのうち1600を学習，400を検定に使っています。

- `task_03_sgd`　SGDで`datasets/train/`内のデータを学習します。
- `task_03_msgd`　Momentum SGDで`datasets/train/`内のデータを学習します。

### 課題4
Adamによるパラメータ更新を実装しました。ハイパーパラメータは論文の推奨値に従っています。

- `task_04`　課題3と同様にtrainを1600/400に分けて更新アルゴリズムをテストするためのタスクです。
- `task_04_test`　すべてのtrainデータを使って学習した後，`datasets/test/`内のグラフを分類し，標準出力に書き出します。

#### テストデータの分類
```bash
cargo run --release task_04_test | tee output_raw.txt
tail -n +102 output_raw.txt > prediction.txt
```

## コードの構成について
### `src/main.rs`
実行起点です。簡単なユニットテストもこのファイル内に書いてあります。

### `gnn::GNN`
GNNの主要な操作はこの中で定義しています。

### `gnn::optimizer::{ SGD, MomentumSGD, Adam }`
更新アルゴリズムのみ別ファイルで定義してします。

### `trait gnn::optimizer::Optimizer`
更新アルゴリズムは全てこのトレイトを実装しています。
