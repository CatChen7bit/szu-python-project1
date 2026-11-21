# assets/

- `images/`：课程提供的 10 张示例图，`images.json` 记录 ID、路径与 SHA-256。
- `manifest.json`：图片清单与参考输出的校验值。
- `reference_outputs.npz`（未纳入仓库）：课程提供的独立 FP32 参考结果，仅供 `check.py` 使用。为避免答案泄漏，不提交到 GitHub；如需复现全部 stage，请从课程附件复制到本目录。
