# -*- coding: utf-8 -*-
"""
Wav2Vec2-large-xlsr-hindi 评估示例代码

在 Common Voice 印地语测试集上评估模型性能，计算词错误率（WER）。
"""

import torch
import torchaudio
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import re

# 加载测试数据集和评估指标
test_dataset = load_dataset("common_voice", "hi", split="test")
wer = load_metric("wer")

# 初始化处理器和模型
processor = Wav2Vec2Processor.from_pretrained("theainerd/Wav2Vec2-large-xlsr-hindi")
model = Wav2Vec2ForCTC.from_pretrained("theainerd/Wav2Vec2-large-xlsr-hindi")
model.to("cuda")

# 创建重采样器
resampler = torchaudio.transforms.Resample(48_000, 16_000)

# 需要忽略的字符
chars_to_ignore_regex = r'[\,\?\.\!\-\;\:\"\\"\"]'


# 预处理函数
def speech_file_to_array_fn(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = resampler(speech_array).squeeze().numpy()
    return batch


# 预处理数据集
test_dataset = test_dataset.map(speech_file_to_array_fn)


# 评估函数
def evaluate(batch):
    inputs = processor(
        batch["speech"],
        sampling_rate=16_000,
        return_tensors="pt",
        padding=True
    )
    with torch.no_grad():
        logits = model(
            inputs.input_values.to("cuda"),
            attention_mask=inputs.attention_mask.to("cuda")
        ).logits
    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_strings"] = processor.batch_decode(pred_ids)
    return batch


# 执行评估
result = test_dataset.map(evaluate, batched=True, batch_size=8)

# 计算词错误率
wer_score = 100 * wer.compute(
    predictions=result["pred_strings"],
    references=result["sentence"]
)

print(f"WER: {wer_score:.2f}%")