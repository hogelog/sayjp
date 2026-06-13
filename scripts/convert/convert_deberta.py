"""deberta.onnx (int8 量子化) と tokenizer.json を Hugging Face transformers だけで生成する。

出力する ONNX は sayjp の推論エンジン
(src/engine/bert.rs) に合わせ、入力を input_ids / attention_mask の 2 つだけにする
(token_type_ids はグラフ内部でゼロ生成する)。BERT 特徴は Style-Bert-VITS2 と同じく
hidden_states[-3:-2] を採用する。
"""

import os
import subprocess
import sys
from argparse import ArgumentParser

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import nn
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers.convert_slow_tokenizer import BertConverter

parser = ArgumentParser()
parser.add_argument("--model", default="ku-nlp/deberta-v2-large-japanese-char-wwm")
parser.add_argument("--out_dir", default="models")
args = parser.parse_args()

# fast tokenizer (tokenizer.json) を BertConverter で生成
slow_tokenizer = AutoTokenizer.from_pretrained(args.model)
BertConverter(slow_tokenizer).converted().save(f"{args.out_dir}/tokenizer.json")


class ORTDeberta(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        # fp32 を強制する。auto 推論だと一部テンソルが float16 になり、後段の int8 動的量子化で
        # DynamicQuantizeLinear が float16 入力を受けて invalid model になる。
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        ).float()

    def forward(self, input_ids, attention_mask):
        # engine は token_type_ids を渡さないため、ここでゼロを生成して隠蔽する。
        token_type_ids = torch.zeros_like(input_ids)
        res = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()


model = ORTDeberta(args.model)
inputs = AutoTokenizer.from_pretrained(args.model)("今日はいい天気ですね", return_tensors="pt")

fp32_path = f"{args.out_dir}/deberta_fp32.onnx"
torch.onnx.export(
    model,
    (inputs["input_ids"], inputs["attention_mask"]),
    fp32_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["output"],
    dynamic_axes={"input_ids": {1: "seq"}, "attention_mask": {1: "seq"}},
    # 新しい torch の dynamo exporter は失敗するため legacy exporter を使う
    dynamo=False,
)

subprocess.run([sys.executable, "-m", "onnxsim", fp32_path, fp32_path], check=True)

# Conv を含めると ConvInteger 非対応で実行時エラー → MatMul のみ量子化 (1.2GB→約460MB)
quantize_dynamic(
    fp32_path,
    f"{args.out_dir}/deberta.onnx",
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul"],
)
os.unlink(fp32_path)
print(f"wrote {args.out_dir}/deberta.onnx and {args.out_dir}/tokenizer.json")
