# 问题二：MAG + 语音 Mamba 对照实验

在现有 `q2_text_anchor_word_mag` 模型的语音分支中增加一个官方 `mamba_ssm.Mamba` 层。输入为已对齐的语音特征 `50 × 74`，先投影成 `50 × 128`，再经过 Mamba（`d_model=128, d_state=16, d_conv=4, expand=2`），使用可学习的小幅残差和 LayerNorm，最后进行原有的掩码注意力汇总。BERT、词位置 MAG、视觉分支、分类头、分数辅助头与原实验相同。这是针对语音时序的一个受控实验，不是完整 MSAmba 论文模型。

## 一键训练与评估

在 VS Code 点击运行 `train_q2_word_mag_audio_mamba.py`。若当前解释器不是 Mamba 环境，入口会自动调用 `D:/Anaconda_envs/envs/mamba/python.exe`。训练使用 D 盘 `aligned_50.pkl`，最佳验证集检查点保存在 `runs/q2_text_anchor_word_mag_audio_mamba/best.pt`。训练结束后点击运行 `evaluate_q2_word_mag_audio_mamba.py`，生成同目录的 `test_result.json` 和 `test_predictions.csv`。

参数在 `configs/q2_reliability_word_mag_audio_mamba.yaml`。原 MAG 对照实验的最佳验证集宏平均 F1 为 **0.6216**，测试集准确率 **70.01%**、宏平均 F1 **0.6607**。先用新实验的最佳验证集宏平均 F1 比较，确认是否值得进一步分析；不应以单次测试集分数调参。

已用 Mamba 环境的 CUDA 实测官方 Mamba 前向、反向，以及两批数据的完整训练流程。两批数据的 smoke 指标不能当作模型准确率。本实验没有运行全量训练。

参考：[Mamba 原论文](https://arxiv.org/abs/2312.00752)、[MSAmba（AAAI 2025）](https://ojs.aaai.org/index.php/AAAI/article/view/32120)。
