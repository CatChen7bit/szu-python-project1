# weights/

本目录只保留 `manifest.json`（152 个视觉张量的名称、形状、dtype 与 SHA-256）。

课程提供的权重文件 **`image_encoder.npz`（约 326 MB）** 超过 GitHub 单文件限制，未纳入仓库。请从课程附件/畅课平台下载后放到：

```text
weights/image_encoder.npz
```

放置后运行：

```bash
python check.py --stage assets
```

`check.py` 会按 `manifest.json` 校验文件 SHA-256、张量名称、形状和 dtype。校验失败时不要改 `manifest.json`，先确认下载的是原版文件。
