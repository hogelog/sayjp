---
name: convert-sbv2-voice
description: Style-Bert-VITS2 (JP-Extra) の音声モデルを sayjp 形式 (voice.onnx + style_vectors.json) に変換する手順。「sayjp に声を追加」「SBV2 モデルを変換」「voice.onnx を作る」「別の声を使いたい」等で発動。
---

# SBV2 音声モデルを sayjp 形式へ変換

sayjp は特定の声を同梱しない。Style-Bert-VITS2 (JP-Extra) 形式の音声モデルを変換して使う。
ここでは zonoko を例に、任意の SBV2 モデルを変換する手順を示す。

## 入力 (SBV2 モデル一式)

変換には次の3ファイルが要る:

- `*.safetensors` — モデル重み
- `config.json` — ハイパーパラメータ
- `style_vectors.npy` — スタイルベクトル

例: zonoko (CC0-1.0 / zgock/style-bert-vits2-zonoko-cc0)

```
base=https://huggingface.co/zgock/style-bert-vits2-zonoko-cc0/resolve/main/zonoko
mkdir -p zk
for f in config.json style_vectors.npy zonoko_e100_s4300.safetensors; do
  curl -sL -o "zk/$f" "$base/$f"
done
```

## 変換

```
# 音声モデル → voice.onnx / style_vectors.json (pyenv は自動で用意される)
MODEL_FILE=zk/zonoko_e100_s4300.safetensors \
CONFIG_FILE=zk/config.json \
STYLE_FILE=zk/style_vectors.npy \
  mise run convert-voice

# BERT (声に依存しない必須コンポーネント) → deberta.onnx / tokenizer.json
mise run convert-deberta
```

いずれも `models/` に出力される。中身は `scripts/convert/convert_model.py` /
`convert_deberta.py`、ONNX 入出力名は `src/engine/` の契約に揃えてある。

## 確認

```
cargo build --release
./target/release/sayjp --model-dir models "こんにちは、テストです"
```

## ライセンス

- 変換出力 (voice.onnx) は**元モデルのライセンスに従う**。zonoko は CC0 なので義務なし。
- 非 CC0 (CC-BY / CC-BY-SA など) の声を使う場合は、配布時にそのモデルの帰属・継承を満たすこと。
- deberta は ku-nlp の CC-BY-SA-4.0 (改変あり)。
