# 方言识别（毕设）

基于 FunASR / Paraformer-large 的四川方言（成都、重庆）语音识别实验代码。数据来自 KeSpeech 中的川渝子集，按说话人无重叠划分为训练 / 验证 / 测试集，围绕同一套数据开展语音识别微调实验。

> 仓库名说明：GitHub 仓库名需为 ASCII，"方言识别（毕设）" 用作项目标题。

## 目录

```
.
├─ scripts/
│  ├─ filter_dataset.py
│  ├─ make_server_split.py
│  ├─ check_split.py
│  ├─ training/
│  │  ├─ train_fft.py
│  │  ├─ train_frozen.py
│  │  ├─ train_lora.py
│  │  ├─ train_dora.py
│  │  ├─ train_variant_common.py
│  │  ├─ adapter_utils.py
│  │  ├─ dora_train_entry.py
│  │  └─ loss_curve.py
│  └─ evaluation/
│     ├─ run_baseline.py
│     ├─ eval_asr.py
│     └─ cer_utils.py
├─ data/
└─ results/
```

数据清单（train/dev/test 等）由 `scripts/` 下的脚本现场生成，本仓库不包含音频数据。

## License

MIT License
