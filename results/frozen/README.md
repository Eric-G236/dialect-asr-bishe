# Frozen（冻结微调）实验结果

- 实验名：`frozen_bs48000_ep20`
- 基座模型：`paraformer-large`
- 训练配置：冻结编码器，`--batch-size 48000 --num-workers 8 --no-use-fp16`
- 选用 checkpoint：`model.pt.best`
- dev CER：8.2547%
- test CER：8.8128%

## 模型权重

权重文件约为 840MB，不直接放进 Git 仓库。当前存放位置：

```
/root/autodl-tmp/models/paraformer-frozen_bs48000_ep20/model.pt
```

建议后续上传至 ModelScope / Hugging Face 后补上公开下载链接。

## 错误细节

- `dev.errors.tsv`：验证集整体 CER 行 + 逐句错误明细
- `test.errors.tsv`：测试集整体 CER 行 + 逐句错误明细
- `dev.summary.txt` / `dev.summary.json`：验证集汇总
- `test.summary.txt` / `test.summary.json`：测试集汇总
