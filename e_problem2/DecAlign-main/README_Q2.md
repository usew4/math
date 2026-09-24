# DecAlign 原源码适配 E 题问题二：已对齐特征

`DecAlign-main/` 来源于桌面的原论文源码包。当前默认训练附件 2 的 **`aligned_50.pkl`**，保留原模型的模态独有/共有特征分解、原型最优传输、分布匹配和跨模态注意力。适配了本题的三分类、情感强度预测及缺失位置掩码；这是竞赛子集上的新实验，不等同于论文原始 MOSEI 基准结果。

## 训练输入

- D 盘附件 2 `aligned_50.pkl`：直接取 `text (50×768)`、`audio (50×74)`、`vision (50×35)`。训练时**不运行 BERT**；`text_bert[:,1,:]` 仅用作文本有效位置掩码。
- 语音、视觉中全零的位置视为缺失或填充。即使已经词级对齐，也要用各自掩码；训练集有 110 条视觉整段为空的样本。
- 语音和视觉仅用训练集有效观测拟合标准化参数。原论文的 5 步局部卷积在这里采用同长度填充，以保留 50 个对齐位置。
- 标签使用 `classification_labels`（Negative、Neutral、Positive）与 `regression_labels`（情感强度）。训练可随机遮蔽一个模态的连续区间；验证集完整输入和 50% 局部缺失都参与选模。

## 在 VS Code 运行

打开 `C:\Users\wwh\Desktop\math\e_problem2`，选择 `D:\Anaconda_envs\envs\pytorch312\python.exe`。在“Tasks: Run Task”里依次选择：

**只点击运行键的方式：**在编辑器中打开 `DecAlign-main/train_q2.py`，点击右上角三角形 **Run Python File**。该文件不传参数时默认读取 D 盘附件 2 的 `aligned_50.pkl`，完整训练最多 40 轮，结果自动写到 `runs/decalign_q2_aligned_final/`。完成训练后，打开 `DecAlign-main/predict_q2_aligned.py` 再点同一个运行键，即可预测附件 3 对齐版。项目 `.vscode/settings.json` 已设置 PyTorch 解释器。

也可以通过任务列表运行：

1. **问题二：DecAlign对齐数据训练验证**：仅训练并看验证集，不评价测试集。
2. **问题二：DecAlign对齐数据最终训练**：训练完成后写出 `runs/decalign_q2_aligned_final/result.json` 和 `best.pt`。
3. **问题二：DecAlign预测附件3对齐版**：读取上述权重，预测附件 3 的 30 条**对齐版本**样本。

在项目根目录使用 PowerShell 的等价命令：

```powershell
& 'D:\Anaconda_envs\envs\pytorch312\python.exe' -u DecAlign-main\train_q2.py --feature-mode aligned --validation-only
& 'D:\Anaconda_envs\envs\pytorch312\python.exe' -u DecAlign-main\train_q2.py --feature-mode aligned --output runs\decalign_q2_aligned_final
& 'D:\Anaconda_envs\envs\pytorch312\python.exe' -u DecAlign-main\predict_q2_aligned.py --checkpoint runs\decalign_q2_aligned_final\best.pt --output runs\decalign_q2_aligned_final\attachment3_predictions.csv
```

默认最多 40 轮，验证指标连续 8 轮无改善时提前停止。每轮输出 `val_acc`、`val_F1` 和局部缺失的 `missing_F1`。可通过 `--epochs`、`--patience`、`--batch-size` 调整；显存不足时设 `--batch-size 4`。

先检查运行接口：

```powershell
& 'D:\Anaconda_envs\envs\pytorch312\python.exe' -u DecAlign-main\train_q2.py --feature-mode aligned --epochs 1 --batch-size 4 --smoke-batches 1 --output runs\decalign_aligned_smoke
```

`--smoke-batches` 只用于检查输入、前向和反向；其权重不可作为论文结果。完整训练后，`result.json` 才包含测试集 Accuracy、Macro-F1、强度 MAE / Pearson 及缺失比例和位置网格。验证集先选模，测试集不参与调参。

附件 3 对齐版本只有 `text_bert`、`audio`、`vision`，没有现成 `text`。预测脚本用本地 BERT 将其 token 编码重建成 `50×768` 文本特征，然后按训练集统计量处理语音、视觉；这一步只发生在推理阶段。

保留此前未对齐实验作为对照：训练时加 `--feature-mode unaligned --output runs\decalign_q2_unaligned`，预测使用 `predict_q2.py`。两种输入和附件 3 版本必须分别配套，不能交叉使用。论文报告的二分类 MOSEI 成绩也不能直接与本题三分类子集比较。

## 如何改进当前 60% 左右的验证准确率

已有验证集诊断：三模态 Macro-F1 为 0.586，只保留文本为 0.575；语音、视觉单独使用约为 0.31。中性类召回率约 40%，是主要短板。当前默认约 70% 的训练批次随机遮蔽一个模态，因此先按顺序检查两个因素，不同时改变模型结构：

1. 在 VS Code 打开 `run_aligned_less_missing.py`，点击右上角 **Run Python File**。它只把随机遮蔽概率从 0.7 降为 0.3，结果存入 `runs/decalign_aligned_less_missing/`。
2. 再打开 `run_aligned_balanced.py` 点击运行。它保持 0.3 的遮蔽概率，并对少数类别施加温和的平方根逆频率权重，结果存入 `runs/decalign_aligned_balanced/`。
3. 最后打开 `compare_q2_validation.py` 点击运行，比较各实验的验证集 Accuracy、Macro-F1、缺失 Macro-F1 和 MAE。新实验都使用 `--validation-only`，不查看测试集；按验证表现确定最终设置。

**两阶段分类头、无模拟缺失实验：**打开 `run_aligned_two_stage.py`，点击右上角 **Run Python File**。第一阶段学习中性/非中性，第二阶段只对非中性学习负面/正面；三个最终概率依次为 `(1−P中性)×P负面|非中性`、`P中性`、`(1−P中性)×P正面|非中性`。情感强度仍作辅助任务，不使用 `±0.5` 阈值。此入口加 `--clean-only`：训练不遮蔽模态，验证和测试也不模拟缺失；验证结果写在 `runs/decalign_aligned_two_stage_clean/`。为隔离“换分类头”的效果，可点击运行 `run_aligned_flat_clean.py`，它也完全不模拟缺失，结果写在 `runs/decalign_aligned_flat_clean/`。`compare_q2_validation.py` 会同时列出这两组。旧 `runs/decalign_aligned_two_stage/` 是先前 30% 缺失增强的未完成运行，不用于这组比较。
**两阶段分类头、无模拟缺失实验：**打开 `run_aligned_two_stage.py`，点击右上角 **Run Python File**。第一阶段学习中性/非中性，第二阶段只对非中性学习负面/正面；三个最终概率依次为 `(1−P中性)×P负面|非中性`、`P中性`、`(1−P中性)×P正面|非中性`。情感强度仍作辅助任务，不使用 `±0.5` 阈值。此入口加 `--clean-only`：训练不遮蔽模态，验证和测试也不模拟缺失。两阶段头现已加上与普通头相同的最终 dropout，新验证结果写在 `runs/decalign_aligned_two_stage_dropout_clean/`；原来的 `runs/decalign_aligned_two_stage_clean/` 保留旧结果，不会覆盖。为隔离“换分类头”的效果，可点击运行 `run_aligned_flat_clean.py`，它也完全不模拟缺失，结果写在 `runs/decalign_aligned_flat_clean/`。`compare_q2_validation.py` 会列出各组验证结果。旧 `runs/decalign_aligned_two_stage/` 是先前 30% 缺失增强的未完成运行，不用于这组比较。

如果两阶段模型的负面召回下降，可再点击运行 `run_aligned_two_stage_weighted_clean.py`。它只在上述无模拟缺失的两阶段方案上加入平方根逆频率类别权重，单独输出到 `runs/decalign_aligned_two_stage_weighted_clean/`；用验证集检查负面和中性是否同时改善，不能预先保证准确率提高。

三个实验使用相同的数据划分与随机种子。类别权重可能提高中性类召回，却不保证提高总体准确率，因此必须看实际验证结果。仅把训练轮数调高通常无法解决第 5 轮后训练损失下降而验证 F1 下降的问题。
