# -*- coding: utf-8 -*-
"""
Wav2Vec2-large-xlsr-hindi 使用示例代码

注意：此代码需要安装以下依赖：
- torch
- torchaudio
- datasets
- transformers

实际使用时，请确保音频采样率为 16kHz。
"""

import torch
import torchaudio
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# 加载测试数据集
test_dataset = load_dataset("common_voice", "hi", split="test[:2%]")

# 初始化处理器和模型
processor = Wav2Vec2Processor.from_pretrained("theainerd/Wav2Vec2-large-xlsr-hindi")
model = Wav2Vec2ForCTC.from_pretrained("theainerd/Wav2Vec2-large-xlsr-hindi")

# 创建重采样器（将 48kHz 转换为 16kHz）
resampler = torchaudio.transforms.Resample(48_000, 16_000)


# 预处理函数：将音频文件转换为数组
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = resampler(speech_array).squeeze().numpy()
    return batch


# 预处理数据集
test_dataset = test_dataset.map(speech_file_to_array_fn)

# 进行推理
inputs = processor(
    test_dataset["speech"][:2],
    sampling_rate=16_000,
    return_tensors="pt",
    padding=True
)

with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

predicted_ids = torch.argmax(logits, dim=-1)

# 解码结果
print("Prediction:", processor.batch_decode(predicted_ids))
print("Reference:", test_dataset["sentence"][:2])