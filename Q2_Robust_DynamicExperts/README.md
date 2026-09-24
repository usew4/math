# 问题二：原双专家模型的局部缺失评估与预测

> GitHub 版本仅包含代码、配置和文档。模型权重、旧实验归档、依赖缓存与生成结果保留在本机。运行前需将原检查点放到 `checkpoints/dynamic_experts_base.pt`，并将 BERT 权重放到 `pretrained/bert/model.safetensors`；BERT 权重也已通过 Git LFS 保存在本仓库的 `e_problem1/bert_model/model.safetensors`。

这里保留在现有实验中表现最稳定的原双专家检查点，用于附件二的缺失实验和附件三的全量预测。当前目录是**评估与推理项目**，主目录不再包含未带来明显改进的整网微调、门控训练、`[MASK]` 和局部重建训练分支。旧运行输出集中放在 `.archive/`；原检查点的来源及 SHA256 见 `provenance.json`。

## 在 VS Code 中点击运行

1. 打开本目录，选择 Python `D:\Anaconda_envs\envs\mamba\python.exe`。
2. 打开 `evaluate.py`，点击运行键。默认对附件二验证集计算完整输入和 7 种局部缺失组合在不同缺失率、位置下的 Accuracy、Macro-F1、MAE、Pearson 和混淆矩阵。结果保存到 `results/`。若要评估测试集，在 VS Code 运行配置中给脚本传 `--split test`，或在终端执行 `python evaluate.py --split test`。
3. 打开 `predict_attachment3.py`，点击运行键。结果为 `results/attachment3_predictions.csv` 和 `results/attachment3_summary.json`，涵盖附件三对齐版本的全部样本。附件三没有标签，因此不计算准确率。

数据路径、评估批大小、人工缺失率和固定随机种子都在 `config.yaml`。数据不复制到本项目，仍从 D 盘读取。`text_bert`、`audio`、`vision` 分别为 `[3,50]`、`[50,74]`、`[50,35]`；三种输入使用附件二 `aligned_50.pkl` 和附件三的对齐版本。输出为负面／中性／正面三分类概率及连续情感强度。

## 模型和缺失实验

`checkpoints/dynamic_experts_base.pt` 是原双专家第 9 轮权重。模型用 BERT 编码文本，对语音和视觉做投影与时间注意力池化，再由文本分类头、融合分类头和决策门得到类别；融合表示另经回归头输出强度。`model.py` 保留了局部缺失时的安全掩码规则，缺失位置不参加池化，完整缺失模态不伪造观测。该项目没有重新训练模型。

`data.py` 在原有有效范围内遮挡连续位置，不移动时间索引，也不把缺失位置当作 padding 之外的新词。`evaluate.py` 对 T、A、V、TA、TV、AV、TAV 七种组合以及 10%、30%、50% 目标缺失率重复固定随机实验，还比较前／中／后缺口与多模态同步缺口。实际移除比例写入结果。缺失率是对齐位置比例，不能直接解释成秒数。

原模型已记录的完整验证集 Macro-F1 为 0.6341、Accuracy 为 0.6552；原始附件二测试集 Macro-F1 为 0.6388、Accuracy 为 0.6781，原始记录见 `checkpoints/base_validation_complete.json` 和 `checkpoints/base_test_result.json`。这些是完整输入指标；缺失条件的指标应以当前脚本重新生成的 `results/` 为准。

## 曾试过的方案

| 方案 | 验证选择结果 | 处理 |
|---|---|---|
| 整网鲁棒微调 | 最佳 epoch 0，未优于初始化 | 训练分支已移除，旧输出归档 |
| 文本可用性门 | 最佳 epoch 8，选择分数从 0.61598 到 0.61657，增益约 0.00059 | 增益很小；附件三文本无缺口，不采用 |
| `[MASK]` 直接替换 | 局部文本缺失表现下降 | 移除 |
| 局部文本重建，两组缺失率 | 最佳 epoch 均为 0 | 训练分支已移除，旧输出归档 |

这些实验的选择分数是固定验证条件上的综合指标，不是测试集准确率。旧实验没有改进的事实可以用于论文中的消融和失败分析；不要把 epoch 0 解释成训练后的提升。此项目保留最简洁、可运行的原模型评估和附件三预测路径。
