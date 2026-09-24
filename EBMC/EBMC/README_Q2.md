# 问题二：EBMC连续局部缺失训练

本入口保留 EBMC 的两阶段及 MSD/CCE/EMC/IMTD，使用附件2未对齐特征；增加融合表示上的三分类头。原 `train_EBMC.py`、`train_ebmc_seq.py` 入口保留。本次不包含附件3预测，也不重新生成音视频或文本特征。

## 1. Ubuntu / RTX 3090 环境

在服务器进入本仓库 `EBMC` 根目录（本文件所在目录）。无需安装整个原始环境快照。

```bash
conda create -n ebmc-q2 python=3.10 -y
conda activate ebmc-q2
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-q2.txt
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.ones(1, device='cuda'))"
```

驱动显示 CUDA 13.0 表示驱动支持上限，不要求安装 CUDA 13.0 的 PyTorch。以上采用自带 CUDA 12.1 运行库的 wheel；实际设备操作用于检查驱动兼容性。无需编译 CUDA 扩展。训练脚本请求 CUDA 而不可用时直接报错，不悄悄退回CPU。

## 2. 输入和数据规则

`--data` 指向赛题的 `unaligned_50.pkl`。训练只使用 train（3395条）及 valid（728条）；test（727条）只由独立评价命令使用。读取 pickle 必须来自可信来源。

- 文本：`text [N,50,768]`；长度由 `text_bert[:,1,:]` 求和确定。
- 音频：`audio [N,500,74]`，长度读 `audio_lengths`。
- 视觉：`vision [N,500,35]`，长度读 `vision_lengths`。
- 回归标签在 [-3,3]；分类按负向0／中性1／正向2，加载时核对既有分类标签。
- 有效范围外置零；三模态补齐为500位置仅用于兼容模型形状，并非时间对齐。原始文件只读。
- 标签、形状、长度和有限值异常直接报错，不静默删除样本或修改数值。

训练默认20%完整样本；80%均匀抽取 T/A/V/TA/TV/AV/TAV。被选模态各自独立抽取10%～50%的连续区间；文本保护CLS和SEP。长度按可擦除有效位置计算，至少保留一个可擦除位置；过短模态跳过并写入每轮日志。随机性由种子、轮次、样本编号控制。

批次同时保存 `valid` 和 `observed`。模型沿用序列版的语义：仅屏蔽填充；擦除位置为零但仍处于有效范围。`observed` 用于检查和追踪，不把未知位置删除或压缩。

## 3. 先做小规模验证

```bash
python EBMC/test_q2.py
python EBMC/train_q2.py --data '/path/to/unaligned_50.pkl' --output runs/q2_smoke --smoke
```

`--smoke` 使用训练和验证各前4条，两阶段各1轮、每轮1个批次、批量2；保留500位置和默认256维4层网络，并跑全部22个固定验证场景。此结果不用于评价准确率或选正式模型。输出目录应使用新目录，避免覆盖已有检查点。

## 4. 正式训练（本次未启动）

```bash
python EBMC/train_q2.py --data '/path/to/unaligned_50.pkl' --output runs/q2_seed1111 --device cuda:0
```

默认配置见 `q2_config.json`：FP32，隐藏维256，4层，2个注意力头，批量8，Adam，学习率1e-4，权重衰减1e-5，两阶段各50轮。可使用 `--batch-size`、`--stage1-epochs`、`--stage2-epochs`、`--seed`、`--workers`、`--hidden`、`--depth` 覆盖。其他项通过单独JSON配置并传 `--config` 调整。

显存不够先降低批量，不截短序列。FP32不使用自动混合精度；每轮记录实际GPU峰值显存。默认workers=0避免多进程复制大数组，服务器可在资源允许时提高。

第一阶段以22个验证场景的平均单模态MAE选教师，第二阶段固定该教师并保持eval及无梯度。第一阶段学生在阶段切换时保留末轮参数，教师使用验证集最佳参数。

第二阶段目标为融合回归MSE + 三分类交叉熵（权重1）+ 原辅助损失。第一阶段仍为单模态回归训练。MSD/CCE/EMC/IMTD权重为0.5/0.1/0.1/0.1；预训练教师预测当前缺失输入，未增加完整输入教师路线。

每轮验证22种固定场景：完整 + 7种组合×10%/30%/50%，场景等权平均。第二阶段按三分类Macro-F1选择唯一最佳学生；相同时优先MAE更低。每个场景输出 Accuracy、Macro-F1、三类F1、MAE、Pearson相关系数；常数预测或标签导致相关系数不可定义时保存null。回归头沿用未裁剪输出；分类来自独立三分类头，不使用正负号替代中性预测。

## 5. 检查点、恢复和独立评价

- `teacher.pt`：第一阶段最佳教师，单独存储。
- `best.pt`：验证集选出的唯一最佳学生，连同对应优化器及恢复信息。
- `latest.pt`：最近完整轮次的学生、优化器、下一轮编号、随机状态和配置。
- `metrics.jsonl`：逐轮损失、22个场景指标、跳过擦除次数、峰值显存。
- `environment.json`：环境、设备、数据路径、配置及是否为小规模验证。

恢复时保存的配置优先于命令行超参数。恢复以完整轮次为单位，训练中途退出需从上一完整轮次重新运行。必须保留检查点旁的 `teacher.pt`。小规模恢复仍加 `--smoke`，不可把其检查点当作正式训练起点。

```bash
python EBMC/train_q2.py --data '/path/to/unaligned_50.pkl' --output runs/q2_seed1111 --resume runs/q2_seed1111/latest.pt
python EBMC/train_q2.py --data '/path/to/unaligned_50.pkl' --output runs/q2_valid --evaluate runs/q2_seed1111/best.pt --split valid
python EBMC/train_q2.py --data '/path/to/unaligned_50.pkl' --output runs/q2_stress --evaluate runs/q2_seed1111/best.pt --split valid --stress
python EBMC/train_q2.py --data '/path/to/unaligned_50.pkl' --output runs/q2_test --evaluate runs/q2_seed1111/best.pt --split test
```

`--stress` 包括前／中／后三个位置和10%/30%/50%/70%长度（完整场景另保留）。位置均指各自有效序列的相对位置，不声称跨模态同步。最终测试只在模型和超参数确定后运行，不能依据测试结果回头选模。

评价使用独立预测路径，不需要教师、标签参与预测或辅助损失梯度。标签仅在模型外计算评价指标。GPU恢复数值可能受设备/软件版本影响；严格逐位恢复测试在相同CPU环境完成。

## 6. 边界与待办

当前使用附件2预计算文本特征，未更改其表示。附件3未对齐版本只提供raw_text和音视频，后续需要统一文本编码并处理缺少有效长度的情况，不能保证任意BERT产生的768维特征与训练分布一致。本次不读取附件3拟合任何参数。

这是问题二训练适配版，不声称复现论文数值。改动包含数值稳定性修复、三分类头、连续缺失增强、验证选模与显式教师管理；默认原实验入口的接口保持不变。

本地和服务器验证必须分开记录；本地结果见 `Q2_VALIDATION.md`。未连接用户的3090服务器，服务器环境安装和显存峰值尚需运行上述小规模命令确认。
