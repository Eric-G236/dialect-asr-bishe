#!/usr/bin/env python3
"""LoRA fine-tuning for the Sichuan-dialect Paraformer experiment."""

from train_variant_common import run_variant


if __name__ == "__main__":
    raise SystemExit(run_variant("lora"))
