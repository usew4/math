# E 题问题二：未对齐特征的鲁棒情感预测

`EBMC-main/` 已加入作者提供的 EBMC 原源码和本题适配入口。若要运行 EBMC 的 MSD、CCE、EMC、IMTD 两阶段模型，请阅读 [EBMC 适配说明](EBMC-main/README_Q2.md)。`DecAlign-main/` 也已由桌面源码包加入；当前默认使用附件 2 的 `aligned_50.pkl` 与现成 `text` 特征训练，见 [DecAlign 适配说明](DecAlign-main/README_Q2.md)。本文件其余部分介绍原先的轻量基线。

本目录是一套已运行的 PyTorch 基线。使用附件 2 的 `unaligned_50.pkl` 训练三分类（Negative / Neutral / Positive）与情感强度（−3 至 3）联合模型，再对附件 3 的**未对齐版本**做推理。论文中的最终模型选择仍应以验证集和消融实验为依据。

## 本机路径与环境

桌面项目位于 `C:\Users\wwh\Desktop\math\e_problem2`。VS Code 使用 `D:\Anaconda_envs\envs\pytorch312\python.exe`。所有数据和 BERT 路径集中在 `paths.py`：附件 2 与附件 3 继续读取 D: 盘的竞赛原数据；BERT 权重及 Transformers 本地依赖读取 `C:\Users\wwh\Documents\ChatGPT\数学建模\e_problem1`。桌面上的 `e_problem1_submission_final` 没有 BERT 权重，因此不作为文本编码器来源。

在 VS Code 终端运行 `& 'D:\Anaconda_envs\envs\pytorch312\python.exe' paths.py` 可检查解释器、PyTorch/CUDA 和各数据路径。日后移动数据时可改 `paths.py`，或设置 `MOSEI_DATA_ROOT`、`MOSEI_UNALIGNED_PKL`、`MOSEI_MISSING_UNALIGNED_DIR`、`MOSEI_BERT_ROOT` 环境变量。

## 直接使用 `text_bert` 的训练入口

附件 2 中的 `text_bert` 形状为 `N×3×50`；三个通道依次是 token ID、attention mask、token type ID。它**不是** `N×50×768` 的 BERT 语义向量，须先送入预训练 BERT。`prepare_text_bert.py` 直接读取这三个通道，使用冻结的本地 BERT 编码并缓存结果。之后仍使用未对齐的音频、视觉和连续缺失增强训练融合模型。

在 VS Code 依次运行“**问题二：从text_bert生成BERT特征**”和“**问题二：训练text_bert未对齐模型（单模态头）**”，或在本目录运行：

```powershell
& 'D:\Anaconda_envs\envs\pytorch312\python.exe' -u prepare_text_bert.py
& 'D:\Anaconda_envs\envs\pytorch312\python.exe' -u train_unaligned.py --text-cache runs\text_bert\features.npz --output runs\text_bert\model_aux --epochs 15 --batch-size 32
```

结果写入 `runs/text_bert/model_aux/`。附件 3 的未对齐文件只提供 `raw_text`、`audio`、`vision`；最终推理时仍需按相同的 BERT tokenizer 将 `raw_text` 转成 token。已核对附件 2 的全部 4850 条样本：这种分词得到的 token ID 和 attention mask 与给定 `text_bert` 完全一致，因此训练和附件 3 推理可以使用一致的编码规则。若直接使用附件 2 已给出的 `text` 向量，则不必运行本节的 BERT 缓存步骤。

## 从原始文本开始的首轮实验

如果要让文本分支从附件 2 的 `raw_text` 出发，而不读取其现成 `text` 向量，先运行 `prepare_raw_text.py`。它用 `paths.py` 指定的本地 `bert-base-uncased` 对原文分词并提取 50×768 的 BERT 最后一层向量，存入 `runs/raw_text_bert/features.npz`。BERT 在此实验中**冻结**；随后 `train_unaligned.py --raw-text-cache ...` 训练文本、音频、视觉三路编码器的融合部分与双任务预测头。音频和视觉仍来自附件 2 的未对齐特征。

在 VS Code 打开 `e_problem2` 文件夹，依次运行“**问题二：原始文本生成BERT特征**”和“**问题二：训练原始文本未对齐模型（单模态头）**”。命令行等价操作：

```powershell
& 'D:\Anaconda_envs\envs\pytorch312\python.exe' -u prepare_raw_text.py
& 'D:\Anaconda_envs\envs\pytorch312\python.exe' -u train_unaligned.py --raw-text-cache runs\raw_text_bert\features.npz --output runs\raw_text_bert\model_aux --epochs 15 --batch-size 32
```

训练曲线写入 `runs/raw_text_bert/model_aux/history.json`，验证集选出的权重写入 `best.pt`，结束后才评估测试集并写入 `result.json`。需要无缺失增强对照时，在训练命令末尾加 `--no-robust`，同时换一个 `--output` 目录。`prepare_raw_text.py --limit 2` 仅用于检查接口，生成的缓存**不能用于完整训练**。已用每个划分 2 条样本检查从 `raw_text` 编码、读取缓存到训练前向和反向传播的链路；尚未替你运行全量原始文本训练。

该方案仍使用预训练 BERT 表示。由于附件 2 的现成 `text` 本身也是同一个 BERT 的表示，从 `raw_text` 重算特征主要验证数据来源和端到端输入接口，**不能预期仅因此获得显著不同的预测表现**。若要微调 BERT 参数，需要单独实现梯度累积或参数高效微调，并重新检查 8 GB 显存占用。

## 数据事实与处理

已核查附件 2：训练 3395 条、验证 728 条、测试 727 条；三个划分的样本 ID 和视频 ID 均无交叉。输入分别为文本 `50×768`、音频 `500×74`、视觉 `500×35`。`audio_lengths` 和 `vision_lengths` 指出原始有效长度；音频和视觉在有效长度内的连续全零段作为局部缺失。文本 `text` 的填充行也有非零 BERT 向量，**不能通过是否全零判断文本有效性**，须使用 `text_bert[:,1,:]` 的 attention mask。

`data_unaligned.py` 把音频、视觉每连续 5 个位置做一次**带可用性掩码的平均池化**，分别保留为 100 步；这仅为降低计算量，**没有强制把三路对齐到同一位置**。音频和视觉的通道均值、标准差只在训练集的有效观测上拟合，再应用到验证、测试和附件 3。分类标签 `0/1/2` 分别对应 Negative/Neutral/Positive，已与回归标签的符号核对。

附件 3 的 30 个未对齐文件只有 `raw_text`、`audio`、`vision`，没有现成文本向量或长度。预测脚本用同一个 `bert-base-uncased` tokenizer 和权重从 `raw_text` 重建 BERT 特征；在附件 2 前 100 条训练数据上，token ID、attention mask 与给定 `text_bert` 完全一致，抽查 BERT 向量与给定 `text` 的有效位置平均绝对差约为 $10^{-6}$。附件 3 的音频和视觉掩码由非零观测行推断。

## 模型与训练

`model_unaligned.py` 用三路独立投影和时间编码器，再以文本序列为查询，分别对语音、视觉序列做跨模态注意力；随后用可用性信息调节三路贡献，联合预测类别和强度。它借鉴了 [MulT 原论文](https://aclanthology.org/P19-1656/)的“跨模态注意力处理未对齐序列”思想，是针对本题长度、缺失掩码和计算资源编写的简化模型，**不是该论文模型的复现**。

融合头的任务损失为加权三分类交叉熵加 $0.5$ 倍 Smooth L1 回归损失。现在每路编码器另接独立的三分类头和强度回归头；默认 `--aux-weight 0.2`，总损失再加上该权重乘以三路单模态任务损失的平均值。单模态损失只对该模态至少有一个有效观测的样本计算，缺失模态不会获得伪标签梯度。推理仍以融合头为最终输出；单模态头只是辅助监督，不是论文 EBMC 全模型的复现。`--aux-weight 0` 关闭辅助头并恢复旧模型结构，可与原始基线公平比较。

为判断单模态头是否有效，两次训练必须使用同一个 `text_bert` 缓存、数据划分和缺失增强设置，只改变辅助损失权重：

```powershell
& 'D:\Anaconda_envs\envs\pytorch312\python.exe' -u train_unaligned.py --text-cache runs\text_bert\features.npz --output runs\text_bert\model_baseline --aux-weight 0 --epochs 15 --batch-size 32
& 'D:\Anaconda_envs\envs\pytorch312\python.exe' -u train_unaligned.py --text-cache runs\text_bert\features.npz --output runs\text_bert\model_aux --aux-weight 0.2 --epochs 15 --batch-size 32
```

`history.json` 的 `valid_unimodal_clean` 和 `result.json` 的 `test_unimodal_clean` 记录每个单模态头在自身有观测的样本上的 Accuracy、Macro-F1、MAE 和 Pearson；融合头仍以 `valid_clean`、缺失场景指标和 `test` 字段为准。此前基于附件 2 现成 `text` 向量的结果不能直接作为 `text_bert` 新实验的严格消融对照。

`train_unaligned.py` 在训练集上随机遮蔽一个模态的连续局部时间段；`--no-robust` 关闭遮蔽，作为对照；`--strong-robust` 另包含整段文本缺失训练，用于研究更难的文本缺失。每轮只用验证集决定是否保存模型，并提前停止。测试集与附件 3 不参与标准化、训练或选模。

## 在 VS Code 中运行

用 VS Code 打开整个 `e_problem2` 文件夹，选择本机 `pytorch312` 解释器。`.vscode/tasks.json` 提供训练、对照、缺失网格评价和附件 3 预测四项任务。命令行等价操作：

```powershell
& 'D:\Anaconda_envs\envs\pytorch312\python.exe' train_unaligned.py --epochs 15 --batch-size 32 --aux-weight 0 --output runs\unaligned_robust
& 'D:\Anaconda_envs\envs\pytorch312\python.exe' train_unaligned.py --epochs 15 --batch-size 32 --no-robust --aux-weight 0 --output runs\unaligned_clean
& 'D:\Anaconda_envs\envs\pytorch312\python.exe' evaluate_missing_grid.py
& 'D:\Anaconda_envs\envs\pytorch312\python.exe' predict_attachment3.py
```

要用新增单模态头训练出的模型预测附件 3，可指定新检查点：

```powershell
& 'D:\Anaconda_envs\envs\pytorch312\python.exe' predict_attachment3.py --checkpoint runs\text_bert\model_aux\best.pt --output runs\text_bert\model_aux\attachment3_predictions.csv
```

`runs/unaligned_robust/best.pt` 是训练好的约 3 MB 权重和训练集统计量。`runs/unaligned_robust/attachment3_predictions.csv` 是 30 条附件 3 样本的**初版预测**，包含类别、强度与类别概率。

## 首轮结果与下一步

按验证集选出的首轮模型，在附件 2 测试集上的结果如下。分类用三分类 Accuracy 和 Macro-F1，回归用 MAE 和 Pearson；结果仅对应当前划分、随机种子和模型设置。

| 训练方式 | Accuracy | Macro-F1 | MAE | Pearson |
|---|---:|---:|---:|---:|
| 连续缺失增强 | 0.666 | 0.632 | 0.631 | 0.664 |
| 无缺失增强对照 | 0.673 | 0.638 | 0.628 | 0.663 |

`runs/robustness_grid.csv` 记录了三种缺失模态、30%/60% 区间长度及起始/中部/末尾缺失的结果。三种位置平均后，文本 60% 缺失的 Macro-F1 为增强模型 **0.612**、对照模型 **0.605**；音频和视觉缺失时未观察到同样的提升。整段遮去文本的验证集 Macro-F1 降至约 **0.367**，提示当前模型明显依赖文本。加入整段文本遮蔽训练的探索模型将该验证集指标提高到约 **0.409**，但完整输入上的验证集表现有所降低。

这些结果说明**已有可运行的未对齐基线和完整评价接口，但尚不能声称所设计的缺失增强全面优于对照**。下一轮应只在验证集上调整缺失训练比例、模态辅助损失和模型选择指标，固定方案后再评价测试集；最终报告要展示每类模态、缺失长度、缺失位置下的性能变化，并对附件 3 全部 30 条给出预测。提交前还须检查类别与强度输出的一致性，并按竞赛的最终文件规范整理 CSV。
