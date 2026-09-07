# FFT（全参数微调）实验结果

- 实验名：`fft_bs4000_ep20_lr2e-4`
- 基座模型：`paraformer-large`
- 训练配置：`--max-epoch 20 --batch-size 4000 --lr 0.0002 --no-use-fp16`
- 选用 checkpoint：`model.pt.best`（epoch 1）
- dev CER：6.5645%
- test CER：6.8077%

## 模型权重

权重文件约为 2.5GB，GitHub 单文件限制为 100MB，因此不直接入库。当前存放位置：

```
/root/autodl-tmp/models/paraformer-fft_bs4000_ep20_lr2e-4/model.pt
```

建议后续上传至 ModelScope / Hugging Face 后，把公开下载链接补到这里。

## 错误细节

- `dev.errors.tsv`：验证集整体 CER 行 + 逐句错误明细
- `test.errors.tsv`：测试集整体 CER 行 + 逐句错误明细
- `dev.summary.txt` / `dev.summary.json`：验证集汇总
- `test.summary.txt` / `test.summary.json`：测试集汇总
