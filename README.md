# 项目一：基于 NumPy 的以图搜图 —— ViT 图像编码器复现

用 **NumPy** 复现 OpenCLIP **ViT-B/32** 的 image encoder 前向计算：输入一批预处理后的 RGB 图像 `[B,3,224,224]`，输出每张图一个 512 维单位向量 `[B,512]`。本仓库不训练模型，不使用 PyTorch/OpenCLIP，也不依赖 GPU。

## 结果

- `python check.py --stage all`：**8/8 PASS**
- 输出：`(10,512)`，float32，L2 长度 `0.99999988 – 1.0`
- 最终 `output` stage：`max_abs = 5.662e-07`，`mean_abs = 8.688e-08`
- `batch` stage 用 `[1,0,1]` 打乱并重复图片，仍逐行对应

## 仓库结构

```text
.
├─ image_encoder.py      # 六个 TODO 的实现（本项目唯一需要修改的文件）
├─ helpers.py            # 课程提供：LayerNorm / GELU / softmax / 单位化
├─ preprocess_image.py   # 课程提供：resize / center crop / 归一化
├─ runtime.py            # 课程提供：加载权重和图片
├─ check.py              # 分阶段数值验收
├─ run.py                # 批量输出 image token
├─ transformer_steps.py  # 教学用的小型 Transformer，逐步打印中间结果
├─ config.json           # 模型配置
├─ requirements.txt      # 依赖清单
├─ assets/
│  ├─ images/            # 10 张示例图
│  ├─ images.json
│  └─ manifest.json
└─ weights/
   ├─ manifest.json      # 152 个张量的名称、形状、SHA-256
   └─ README.md          # 权重获取说明
```

## 环境

Python 3.9+，依赖见 `requirements.txt`：

```bash
python -m pip install -r requirements.txt
```

主要依赖：`numpy`、`Pillow`；`imageio` 只用于 `lecture-demos/` 里的课堂示例，不影响本项目。

## 权重与参考数据

课程提供的 `weights/image_encoder.npz` 约 **326 MB**，超过 GitHub 单文件限制，未纳入仓库。请从课程附件/畅课平台下载后放到：

```text
weights/image_encoder.npz
```

`weights/manifest.json` 记录了该文件的 SHA-256 和 152 个张量的校验值；`python check.py --stage assets` 会自动校验。

`assets/reference_outputs.npz` 是课程提供的参考输出，仅用于 `check.py` 验收；为避免答案泄漏，未纳入仓库。如需复现全部 stage，请将其放到 `assets/reference_outputs.npz`。

## 运行

```bash
# 分阶段检查（从第一个不一致的 stage 开始排查）
python check.py --stage assets
python check.py --stage preprocess
python check.py --stage patches
python check.py --stage tokens
python check.py --stage block0
python check.py --stage last_tokens
python check.py --stage output
python check.py --stage batch

# 一次跑完
python check.py --stage all

# 生成输出 token
python run.py --output output_tokens.npz
```

## 实现要点

- `patchify`：`(B,3,7,32,7,32) → transpose(0,2,4,1,3,5) → (B,49,3072)`，patch 内按 C,H,W，patch 间先左后右、先上后下。
- `split_heads`：`(B,50,12,64) → transpose(0,2,1,3) → (B,12,50,64)`；合并时反向。
- `attention`：`scores = QK^T / sqrt(64)`，softmax 后对 V 加权，合并 heads，再做输出投影。
- `TransformerBlock`：两次 pre-LN 残差 + MLP（768→3072→768，GELU）。
- 所有权重按 `[out,in]` 存储，线性层使用 `x @ W.T`；只有 `visual.proj` 是 `[in,out]`，不转置。
- 全程 float32。

## 实验报告

实验报告包含个人信息，单独在畅课平台提交，不纳入本公开仓库。

## 说明

本项目为课程项目一的图像编码器部分；以图搜图的检索流程会在后续 final project 中使用该 encoder。代码实现与调试使用了 AI 编程助手辅助，所有数值结果由本人运行 `check.py` 复核。
