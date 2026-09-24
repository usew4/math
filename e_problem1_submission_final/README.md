# E 题问题一完整结果

完整建模、结果与限制见 [问题一完整解答.md](问题一完整解答.md)。最终可提交特征位于 `output_final/`：100 个视频各有一个三模态 `.npz`，另有 100 行汇总、词级时间表、视频帧日志、典型样本图和质量报告。

## 读取结果

```python
import numpy as np
p = r"output_final/features/-3g5yACwYnA__13.npz"
with np.load(p) as d:
    print(d["text"].shape)    # (50, 37)
    print(d["audio"].shape)   # (50, 17)
    print(d["visual"].shape)  # (50, 65)
    print(d["bin_center_sec"])
    print(d["speech_mask"], d["visual_frame_count"], d["mp_face_count"])
```

## 从原视频复现

使用 Python 3.12（Windows）。安装 `requirements.txt` 所列依赖；脚本中的 `vendor` 目录用于部分本地包，`vendor_mp14` 用于 MediaPipe 0.10.14。本机所用 Anaconda 已自带 NumPy、SciPy、scikit-image、Pillow、openpyxl 与 Matplotlib。缺失包可安装到对应目录：

```powershell
python -m pip install --target vendor --no-deps imageio-ffmpeg==0.6.0 vaderSentiment==3.3.2 pocketsphinx==5.0.4 absl-py==2.5.0 protobuf==4.25.8 opencv-contrib-python==4.10.0.84 sentencepiece==0.2.2 sounddevice==0.5.6 msvc-runtime==14.44.35112
python -m pip install --target vendor_mp14 --no-deps mediapipe==0.10.14
```

`face_landmarker.task` 是官方模型，随提交包提供；模型 [来源地址](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)，SHA-256 为 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`。

在本目录执行：

```powershell
python extract.py
python upgrade.py
python add_faces.py
python make_example.py
python analyze_cross_modal.py
python verify_final.py
```

前三个脚本分别生成 `output/`、`output_v2/`、`output_final/`，默认从赛题原路径读取 `E题数据.zip`。如原始 ZIP 移动过，三个脚本均可使用 `--source` 指定新路径。`extract.py` 的 `--output`、`upgrade.py` 的 `--baseline/--output`、`add_faces.py` 的 `--prior/--output` 可调整中间文件位置。`sync_text.py` 仅供现成最终文件需要更新词级对齐时使用，正常从零复现不运行。

`verify_final.py` 已在全部 100 条样本上通过。最终提交包不含原视频、无需打包巨大的 `vendor/` 运行依赖；需要复现时按上面安装。
