# LoRA（低秩微调）实验结果

- 实验名：`lora_bs48000_ep20`
- 基座模型：`paraformer-large`
- 训练配置：LoRA rank 8、alpha 16、dropout 0.1，作用于 encoder/decoder 的 q/k/v/o；`--batch-size 48000 --num-workers 8 --no-use-fp16`
- 选用 checkpoint：`model.pt.best`
- dev CER：9.0108%
- test CER：9.6442%

> 说明：另有一版 `lora_r32_lr5e-4_bs48000_ep20`（rank 32、lr 5e-4），dev 9.0229%、test 9.6516%，与 rank 8 基本持平。本目录记录表现略优的 rank 8 版本。

## 模型权重

权重文件不直接放进 Git 仓库。当前存放位置：

```
/root/autodl-tmp/models/paraformer-lora_bs48000_ep20/model.pt
```

建议后续上传至 ModelScope / Hugging Face 后补上公开下载链接。

## 错误细节

- `dev.errors.tsv`：验证集整体 CER 行 + 逐句错误明细
- `test.errors.tsv`：测试集整体 CER 行 + 逐句错误明细
- `dev.summary.txt` / `dev.summary.json`：验证集汇总
- `test.summary.txt` / `test.summary.json`：测试集汇总
