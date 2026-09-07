# 实验结果

每组实验包含：错误细节（`*.errors.tsv` 与 `*.summary.*`）以及模型权重说明。权重体积较大，不直接放进 Git 仓库，链接以各组 README 中的托管地址为准。

| 实验 | 方法 | dev CER | test CER | 目录 |
|---|---|---|---|---|
| fft | 全参数微调（epoch 1 最优） | 6.5645% | 6.8077% | [fft](fft/README.md) |
| frozen | 冻结微调 | 待补充 | 待补充 | frozen/ |
| lora | LoRA | 待补充 | 待补充 | lora/ |
| dora | DoRA | 待补充 | 待补充 | dora/ |

基线（未微调 paraformer-large）：dev CER 9.0714%，test CER 9.7167%。
