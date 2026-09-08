# DoRA（权重分解低秩微调）实验结果

- 实验名：`dora_bs48000_ep20`
- 基座模型：`paraformer-large`
- 训练配置：DoRA rank 8、alpha 16、dropout 0.1，作用于 encoder/decoder 的 q/k/v/o；`--batch-size 48000 --num-workers 8 --no-use-fp16`
- 选用 checkpoint：`model.pt.best`
- dev CER：6.6332%
- test CER：6.9267%

> 实现说明：DoRA 的 magnitude 参数按 base 权重的行范数初始化；`adapter_utils.py` 中已修复此初始化问题。

## 模型权重

权重文件不直接放进 Git 仓库。当前存放位置：

```
/root/autodl-tmp/models/paraformer-dora_bs48000_ep20/model.pt
```

建议后续上传至 ModelScope / Hugging Face 后补上公开下载链接。

## 错误细节

- `dev.errors.tsv`：验证集整体 CER 行 + 逐句错误明细
- `test.errors.tsv`：测试集整体 CER 行 + 逐句错误明细
- `dev.summary.txt` / `dev.summary.json`：验证集汇总
- `test.summary.txt` / `test.summary.json`：测试集汇总
