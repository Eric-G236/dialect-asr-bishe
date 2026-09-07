#!/usr/bin/env python3
"""Training entry point that replaces FunASR LoRA layers with DoRA layers.

This is launched by ``train_dora.py`` under ``torch.distributed.run``.  It must
patch ``funasr.models.lora.layers`` before the Paraformer encoder/decoder are
built, then hand control to FunASR's normal ``train_ds`` Hydra entry point.
"""

from __future__ import annotations

from adapter_utils import patch_lora_for_dora

patch_lora_for_dora()

from funasr.bin.train_ds import main_hydra


if __name__ == "__main__":
    main_hydra()
