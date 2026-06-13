# sayjp — オフライン日本語 TTS CLI

完全オフラインで動く日本語読み上げ CLI です。テキストを渡すと wav を生成します。
エンジンは Style-Bert-VITS2 (JP-Extra)。声は Style-Bert-VITS2 形式の音声モデルを
変換して使います (特定の音声モデルは同梱しません)。

このリポジトリは**ソースのみ**を配布します。ビルドとモデル生成は手元で行います
(ビルド済みバイナリ・モデルは同梱しません)。

## ビルドとモデル生成

```
mise run build             # cargo build --release
mise run convert-deberta   # deberta.onnx / tokenizer.json を生成 (必須 BERT)

# 音声モデルは Style-Bert-VITS2 形式のものを用意して変換:
MODEL_FILE=voice.safetensors CONFIG_FILE=config.json STYLE_FILE=style_vectors.npy \
  mise run convert-voice   # voice.onnx / style_vectors.json を生成
```

生成物は `models/` に出力されます。モデル生成には Python が必要です
(mise が python 3.11 + uv を用意します)。

## 使い方

```
sayjp "こんにちは、テストです"        # out.wav を生成
sayjp -o greeting.wav "おはよう"      # 出力先を指定
sayjp --play "再生もします"           # 生成後に再生 (wav は残さない)
```

```
sayjp [OPTIONS] "読み上げるテキスト"
  -o FILE          出力 wav (既定: out.wav)
  -s ID            スタイル ID (既定: 0)
  --speed N        話速 (1.0=標準, 大きいほど速い。既定: 1.0)
  --sdp N          抑揚(リズム)の強さ 0.0〜1.0 (大きいほど豊か。既定: 0.3)
  --style-weight N スタイルの強さ (小さいほど無感情・落ち着く。既定: 1.0)
  --model-dir DIR  モデルディレクトリ (既定: 実行ファイル隣の models/)
  --play           再生のみ (wav を残さない。-o 併用時はファイルも残す)
  -h, --help       ヘルプ
```

- モデルは `--model-dir`、環境変数 `SAYJP_MODEL_DIR`、実行ファイル隣の `models/` の順で解決します。
- 再生時の頭切れ対策として、出力 wav の先頭に 0.5 秒の無音を付加します。

## ライセンス

- 本体コード: **AGPL-3.0-or-later** ([LICENSE](LICENSE))
- 移植元コードの帰属: [NOTICE](NOTICE) / [licenses/](licenses)
