"""Style-Bert-VITS2 (JP-Extra) の safetensors モデルを voice.onnx + style_vectors.json に変換する。

ONNX の入出力名は sayjp の推論エンジン (src/engine/model.rs) の契約に合わせる。
"""

import json
import subprocess
import sys
from argparse import ArgumentParser

import numpy as np
import torch

from style_bert_vits2.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    Languages,
)
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.infer import get_net_g, get_text
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.tts_model import TTSModel

parser = ArgumentParser()
parser.add_argument("--style_file", required=True)
parser.add_argument("--config_file", required=True)
parser.add_argument("--model_file", required=True)
parser.add_argument("--out_dir", required=True, help="voice.onnx / style_vectors.json の出力先")
args = parser.parse_args()

device = "cpu"

bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")

style_vectors = np.load(args.style_file)
hyper_parameters = HyperParameters.load_from_json(args.config_file)
style_json = json.dumps({"data": style_vectors.tolist(), "shape": style_vectors.shape})

# トレース用のサンプル入力を SBV2 の g2p から得る
bert, ja_bert, en_bert, phones, tones, lang_ids = get_text(
    "今日はいい天気ですね。",
    Languages.JP,
    hyper_parameters,
    device,
    assist_text=None,
    assist_text_weight=DEFAULT_ASSIST_TEXT_WEIGHT,
    given_phone=None,
    given_tone=None,
)

tts_model = TTSModel(
    model_path=args.model_file,
    config_path=args.config_file,
    style_vec_path=args.style_file,
    device=device,
)
style_id = tts_model.style2id[DEFAULT_STYLE]
mean = style_vectors[0]
style_vec = mean + (style_vectors[style_id] - mean) * DEFAULT_STYLE_WEIGHT

x_tst = phones.to(device).unsqueeze(0)
tones = tones.to(device).unsqueeze(0)
lang_ids = lang_ids.to(device).unsqueeze(0)
bert = bert.to(device).unsqueeze(0)
x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
style_vec_tensor = torch.from_numpy(style_vec).to(device).unsqueeze(0)

model = get_net_g(args.model_file, hyper_parameters.version, device, hyper_parameters)


def forward(x, x_len, sid, tone, lang, bert, style, length_scale, sdp_ratio, noise_scale, noise_scale_w):
    return model.infer(
        x,
        x_len,
        sid,
        tone,
        lang,
        bert,
        style,
        sdp_ratio=sdp_ratio,
        length_scale=length_scale,
        noise_scale=noise_scale,
        noise_scale_w=noise_scale_w,
    )


model.forward = forward

onnx_path = f"{args.out_dir}/voice.onnx"
torch.onnx.export(
    model,
    (
        x_tst,
        x_tst_lengths,
        torch.LongTensor([0]).to(device),
        tones,
        lang_ids,
        bert,
        style_vec_tensor,
        torch.tensor(1.0),
        torch.tensor(0.0),
        torch.tensor(0.6777),
        torch.tensor(0.8),
    ),
    onnx_path,
    verbose=False,
    dynamic_axes={
        "x_tst": {0: "batch_size", 1: "x_tst_max_length"},
        "x_tst_lengths": {0: "batch_size"},
        "sid": {0: "batch_size"},
        "tones": {0: "batch_size", 1: "x_tst_max_length"},
        "language": {0: "batch_size", 1: "x_tst_max_length"},
        "bert": {0: "batch_size", 2: "x_tst_max_length"},
        "style_vec": {0: "batch_size"},
    },
    input_names=[
        "x_tst",
        "x_tst_lengths",
        "sid",
        "tones",
        "language",
        "bert",
        "style_vec",
        "length_scale",
        "sdp_ratio",
        "noise_scale",
        "noise_scale_w",
    ],
    output_names=["output"],
    # 新しい torch の dynamo exporter は SBV2 で失敗するため legacy exporter を使う
    dynamo=False,
)

# onnxsim は venv の python 経由で呼ぶ (CLI が PATH に無いことがあるため)
subprocess.run([sys.executable, "-m", "onnxsim", onnx_path, onnx_path], check=True)

with open(f"{args.out_dir}/style_vectors.json", "w") as f:
    f.write(style_json)

print(f"wrote {onnx_path} and {args.out_dir}/style_vectors.json")
