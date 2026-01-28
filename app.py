# -*- coding: utf-8 -*-
"""Wav2Vec2-large-xlsr-hindi WebUI 演示界面（不加载真实模型权重）。"""
from __future__ import annotations

import gradio as gr


def fake_load_model():
    """模拟加载模型，实际不下载权重，仅用于界面演示。"""
    return "模型状态：Wav2Vec2-large-xlsr-hindi 已就绪（演示模式，未加载真实权重）"


def fake_transcribe(audio_file) -> str:
    """模拟语音转文字功能，返回演示结果。"""
    if audio_file is None:
        return "请先上传音频文件（支持 WAV、MP3 等格式，建议采样率为 16kHz）。"
    
    return (
        "[演示] 语音识别结果：\n"
        "मैं हिंदी में बोल रहा हूँ। यह एक स्वचालित भाषण पहचान प्रणाली है।\n\n"
        "（中文翻译：我正在用印地语说话。这是一个自动语音识别系统。）\n\n"
        "注意：当前为演示模式，实际使用需加载模型权重进行推理。"
    )


def fake_evaluate_wer(predictions: str, references: str) -> str:
    """模拟词错误率（WER）评估功能。"""
    if not (predictions or "").strip() or not (references or "").strip():
        return "请分别输入预测文本和参考文本。"
    
    return (
        "[演示] 评估结果：\n"
        "词错误率（WER）: 72.62%\n"
        "字符错误率（CER）: 15.34%\n\n"
        "注意：当前为演示模式，实际评估需使用真实模型推理结果。"
    )


def build_ui():
    with gr.Blocks(title="Wav2Vec2-large-xlsr-hindi WebUI") as demo:
        gr.Markdown("## Wav2Vec2-large-xlsr-hindi 印地语语音识别模型 · WebUI 演示")
        gr.Markdown(
            "本界面以交互方式展示 Wav2Vec2-large-xlsr-hindi 模型的典型使用流程，"
            "包括模型加载状态监控、单段语音转文字以及词错误率评估等环节。"
        )

        # 模型加载区
        with gr.Row():
            load_btn = gr.Button("加载模型（演示）", variant="primary")
            status_box = gr.Textbox(label="模型状态", value="尚未加载", interactive=False)
        load_btn.click(fn=fake_load_model, outputs=status_box)

        with gr.Tabs():
            # 单段语音识别
            with gr.Tab("单段语音识别"):
                gr.Markdown(
                    "该功能模拟将一段印地语语音转换为文字，"
                    "支持 16kHz 采样率的音频输入。"
                )
                audio_input = gr.Audio(
                    label="上传音频文件",
                    type="filepath",
                )
                transcribe_btn = gr.Button("开始识别（演示）", variant="primary")
                transcribe_output = gr.Textbox(
                    label="识别结果",
                    lines=8,
                    interactive=False,
                )
                transcribe_btn.click(
                    fn=fake_transcribe,
                    inputs=audio_input,
                    outputs=transcribe_output,
                )

            # 词错误率评估
            with gr.Tab("词错误率评估"):
                gr.Markdown(
                    "该功能模拟在 Common Voice 印地语测试集上评估模型性能，"
                    "计算词错误率（WER）和字符错误率（CER）等指标。"
                )
                with gr.Row():
                    predictions_input = gr.Textbox(
                        label="预测文本",
                        placeholder="输入模型识别的文本结果",
                        lines=4,
                    )
                    references_input = gr.Textbox(
                        label="参考文本",
                        placeholder="输入标准参考文本",
                        lines=4,
                    )
                evaluate_btn = gr.Button("计算 WER（演示）", variant="primary")
                evaluate_output = gr.Textbox(
                    label="评估结果",
                    lines=6,
                    interactive=False,
                )
                evaluate_btn.click(
                    fn=fake_evaluate_wer,
                    inputs=[predictions_input, references_input],
                    outputs=evaluate_output,
                )

        gr.Markdown(
            "---\n"
            "*说明：当前为轻量级演示界面，未实际下载与加载任何大规模模型参数。"
            "实际使用需安装 transformers、torch、torchaudio 等依赖并加载模型权重。*"
        )

    return demo


def main():
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=7863, share=False, show_error=True)


if __name__ == "__main__":
    main()
