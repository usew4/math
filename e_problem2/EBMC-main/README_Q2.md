# 基于 EBMC 原源码的问题二适配

原始 `EBMC-main.zip` 保留不变。本目录的 `EBMC/ebmc.py` 和 `EBMC/modules/` 延续原论文的两阶段模型：MSD、CCE、EMC、IMTD 与 Soft-MoE 编码器。新增 `EBMC/train_q2.py` 和 `EBMC/predict_q2.py` 对接竞赛附件。这里是**基于 EBMC 源码的任务适配**，不是原论文表格结果的复现。

## 输入和改动

- 数据：D 盘附件 2 的 `unaligned_50.pkl`，沿用同级 `e_problem2/data_unaligned.py` 的读取、训练集标准化、5 帧一组的带掩码池化和固定 train/valid/test 划分。
- 文本：同级 `runs/text_bert/features.npz`，由附件 2 的 `text_bert` token ID、attention mask 通过冻结 BERT 提取。语音为 `100×74`，视觉为 `100×35`，文本为 `50×768`。文本补零到 100 步只为满足 EBMC 张量形状，**不表示与音视频第 100 步逐时刻对齐**。
- 掩码：填充和局部缺失位置不进入对应模态的注意力与样本池化；完全缺失的模态不参与单模态监督、CCE 跨模态补偿或 IMTD 教师加权。EMC 只在三种模态都有观测的样本上约束能量平衡。
- 输出：EBMC 融合头预测 Negative / Neutral / Positive 三分类，新增强度头预测 `[-3, 3]`。分类、强度以及 EBMC 源码四种模块损失共同训练。
- 训练：第一阶段训练单模态分支并保存教师；第二阶段从同一个学生模型继续训练融合分支，加载冻结的第一阶段教师。训练集随机遮蔽一个模态的连续局部时间段。每轮使用验证集的完整输入和 50% 局部缺失选模；最终测试集只在选模完成后评价。

建议先用已经生成的文本缓存。若不存在，在同级 `e_problem2` 文件夹运行 `prepare_text_bert.py`；项目的 [原使用说明](../README.md) 有 BERT 权重和 D 盘数据路径配置。

## 在 VS Code 运行

用 VS Code 打开 `C:\Users\wwh\Desktop\math\e_problem2`，使用 `D:\Anaconda_envs\envs\pytorch312\python.exe` 解释器。任务列表新增“**问题二：EBMC源码训练验证**”“**问题二：EBMC源码最终训练**”和“**问题二：EBMC源码预测附件3**”。也可以在该文件夹的 PowerShell 终端运行：

```powershell
& 'D:\Anaconda_envs\envs\pytorch312\python.exe' -u EBMC-main\EBMC\train_q2.py --validation-only
& 'D:\Anaconda_envs\envs\pytorch312\python.exe' -u EBMC-main\EBMC\train_q2.py --output runs\ebmc_q2_final
& 'D:\Anaconda_envs\envs\pytorch312\python.exe' -u EBMC-main\EBMC\predict_q2.py --checkpoint runs\ebmc_q2_final\best.pt --output runs\ebmc_q2_final\attachment3_predictions.csv
```

默认 `batch-size=8`、`hidden=64`、`depth=1`，用于适配本机 8 GB 显存；需要时用 `--batch-size 4` 降低占用。训练上限为 40 轮：第一阶段 10 轮，第二阶段最多 30 轮；第二阶段验证指标连续 8 轮未改善时提前停止。可以用 `--epochs`、`--stage-epochs` 和 `--patience` 调整；需满足 `1 <= stage-epochs < epochs`。第一次只检查通路可运行：

```powershell
& 'D:\Anaconda_envs\envs\pytorch312\python.exe' -u EBMC-main\EBMC\train_q2.py --epochs 2 --stage-epochs 1 --batch-size 4 --smoke-batches 1 --output runs\ebmc_q2_smoke
```

`--smoke-batches` 只用于验证接口和显存，生成的权重不能作为实验结果。完整训练输出：`stage1_teacher.pt`、`best.pt`、`history.json`、`result.json`；后者包含三分类 Accuracy/Macro-F1、强度 MAE/Pearson，以及文本、语音、视觉在 30%/60% 连续缺失和开头/中段/末尾位置的测试指标。`--validation-only` 不评价测试集；确定设置后再运行最终训练。附件 3 预测 CSV 包含 30 条样本的类别、强度、类别概率和各模态是否有观测。

第二阶段从第 11 轮开始，每轮在终端打印验证集的 `val_acc` 和 50% 局部缺失场景平均的 `missing_acc`（百分比），同时打印 Macro-F1。第一阶段只训练单模态教师，终端显示损失。完整训练结束后，`result.json` 才写入测试集准确率；使用 `--validation-only` 时不评价测试集。

## 比较与解释

与同级 `train_unaligned.py` 的基线比较时，使用同一份 `text_bert` 缓存、相同数据划分和缺失评估规则。论文中的 MOSEI 数值来自另一规模和实验协议，不能直接作为本题 4850 条子集的对照成绩。EBMC 源码原有 `train_ebmc_seq.py` 使用连续帧的随机独立丢弃和回归指标；本适配入口处理**连续局部缺失、三分类加回归**，因此请运行 `train_q2.py`。

原项目版权和使用条件见 [LICENSE](LICENSE)，论文和源代码信息见 [README.md](README.md)。
