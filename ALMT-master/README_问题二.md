# ALMT 用于 E 题问题二：已对齐特征三分类

打开 `C:\Users\wwh\Desktop\math\ALMT-master`，在 VS Code 中选择项目配置的 `D:\Anaconda_envs\envs\pytorch312\python.exe`。打开外层的 `run_q2_aligned.py`，点击右上角 **Run Python File**。脚本会读取 D 盘附件 2 `aligned_50.pkl` 并完整训练；默认最多 40 轮，验证 Macro-F1 连续 8 轮无提升时提前停止，每轮显示验证准确率、Macro-F1 和各类召回。

模型来自内层 `ALMT-master/models/almt.py`，保留预训练 BERT、三模态投影、文本引导的 AHL 模块与跨模态融合。修改最后一层为三个输出，使用交叉熵直接训练 `Negative=0 / Neutral=1 / Positive=2`。这不是用回归值设置阈值。输入为 `text_bert (3×50)`、`audio (50×74)`、`vision (50×35)`；**BERT 会在训练中更新**。本实验不模拟局部模态缺失。

数据和超参数在内层 `ALMT-master/configs/q2_aligned.yaml` 修改。`batch_size: 4` 和 `accumulation_steps: 4` 等效于每 16 条样本更新一次参数；RTX 5060 使用 bfloat16 混合精度。依赖包安装在源码目录的 `.deps`，不会更改现有 PyTorch 环境。BERT 权重使用本机 `e_problem1/bert_model`，不从网络下载。

训练输出保存在内层 `ALMT-master/runs/q2_almt_aligned/`：`history.json` 为每轮验证记录，`best.pt` 为验证 Macro-F1 最优权重，`validation_complete.json` 标记训练完成。训练期间**不查看测试集**。确定使用此模型后，打开外层 `evaluate_q2_aligned.py` 并点击运行，输出 `test_result.json`。

原始 ALMT 的 MOSEI 配置是未对齐输入和情感强度回归。这里是针对竞赛附件 2 子集的适配实验，原论文的指标不能直接与本项目的三分类 Accuracy/Macro-F1 对比。`--smoke-batches 1` 只用于检查代码，产生的指标不能当作实验结果。

## 强度分数再按 ±0.2 分类的独立实验

打开外层 `run_q2_score.py`，点击 **Run Python File**。该实验使用同样的三模态输入，但 ALMT 最后一层只输出一个情感强度分数；用附件 2 的 `regression_labels` 和原 ALMT 的 MSE 损失训练。预测分数小于 −0.2 为负面，处于闭区间 [−0.2, 0.2] 为中性，大于 0.2 为正面。阈值在内层 `ALMT-master/configs/q2_score.yaml` 的 `task.neutral_threshold`，默认已设为 0.2。验证 Macro-F1 用于选择最佳权重。

新训练结果单独保存在内层 `ALMT-master/runs/q2_almt_score_threshold02/`；`valid_predictions.csv` 含每条验证样本的真实分数、预测分数、真实类别和阈值类别。选好模型后可点击外层 `evaluate_q2_score.py`，生成 `test_result_threshold0p2.json` 和 `test_predictions_threshold0p2.csv`；再点击 `predict_q2_score_attachment3.py`，生成 30 条附件 3 对齐样本的 `attachment3_score_predictions_threshold0p2.csv`。文件名中的 `0p2` 表示 0.2。

如果先前的 `±0.5` 回归模型正在 `ALMT-master/runs/q2_almt_score_threshold05/` 训练，**让它继续完成即可**。回归训练损失不依赖分类阈值；训练结束后直接点击外层 `rethreshold_q2_score.py`，它会把已有的验证预测分数按 `±0.2` 重新分类，生成 `valid_predictions_threshold0p2.csv` 和 `validation_threshold0p2.json`，无须重训。测试和附件 3 入口在未找到新模型时会自动使用已有的旧回归权重，但按 `±0.2` 分类。

**阈值与原类别标签仍有轻微差别：**验证集中有 1 条正面样本的真实强度为 +0.1667，会被 `±0.2` 规则划为中性。因此即使回归分数完全准确，按原三分类标签计算的验证准确率上限是 727/728，即 99.86%。实际模型仍可能因为分数预测误差而低于此值。

## 仅训练集提高中性样本抽取概率

打开外层 `run_q2_score_neutral_sampling.py`，在 VS Code 点击 **Run Python File**。它与 `run_q2_score.py` 使用同一 ALMT 回归模型和 `±0.2` 阈值，只在训练集使用有放回加权抽样。配置文件为内层 `ALMT-master/configs/q2_score_neutral_sampling.yaml`，`sampling.neutral_weight: 1.7` 表示每条中性样本的抽取权重是其他样本的 1.7 倍，期望类别占比约为负面 24.6%、中性 32.8%、正面 42.5%。抽样后的每轮样本总数仍是 3395，重复抽取不是生成新样本。

验证集和测试集仍按原始顺序及原始分布读取；与基线比较验证 Macro-F1、中性召回率、准确率和分数 MAE。该实验的权重及记录单独保存在内层 `ALMT-master/runs/q2_almt_score_neutral_sampling/`，不会覆盖基线。若效果不佳，可以在配置中将 `sampling.neutral_weight` 设为 `1.0` 关闭抽样。

## ALMT 双任务：情感分数＋三分类

打开外层 `run_q2_multitask.py`，在 VS Code 点击 **Run Python File**。模型共享原 ALMT 的 BERT、声学与视觉编码、AHL 和融合层，融合表示后分成一个连续分数头与一个三分类头。训练损失为 `MSE(情感分数) + 0.5 × 交叉熵(三分类)`；`0.5` 在内层 `ALMT-master/configs/q2_multitask.yaml` 的 `task.classification_loss_weight` 修改。本实验使用原始训练分布，不叠加中性抽样或模态缺失模拟。

每轮分别打印：分数按 `±0.2` 分类的准确率、Macro-F1、中性召回率，以及分类头直接预测的同三项指标。结果保存在内层 `ALMT-master/runs/q2_almt_multitask/`。`best_score.pt` 按分数分类的验证 Macro-F1 选出，`best_class.pt` 按分类头的验证 Macro-F1 选出；两个验证预测 CSV 都同时包含两条路径的结果。先依据验证集决定采用哪个权重与哪条预测路径，之后才点击外层 `evaluate_q2_multitask.py` 对默认的 `best_score.pt` 做一次独立测试。若要测试 `best_class.pt`，将其路径作为 `--checkpoint` 参数传入内层评估脚本。

## 数据诊断后的文本锚定模型

数据核查与策略依据见外层 `Q2_数据分析与训练策略.md`。打开外层 `run_q2_reliability.py`，点击 **Run Python File**：BERT 仅微调后 3 层，语音与视觉先排除全零填充帧再池化，通过门控为文本补充信息；分类交叉熵是主任务，强度 Huber 损失是辅助任务。训练集有效帧用于计算归一化参数，验证、测试数据不参与拟合。每轮报告分类 Accuracy、Macro-F1、中性召回，以及强度 MAE、Pearson。配置在内层 `ALMT-master/configs/q2_reliability.yaml`，权重输出到 `ALMT-master/runs/q2_text_anchor_reliability/`。仅在比较验证集并决定采用该模型后，点击外层 `evaluate_q2_reliability.py` 做最终测试。当前训练不模拟局部模态缺失，完整问题二仍需另做缺失条件验证与附件 3 推理。
