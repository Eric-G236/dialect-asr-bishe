#!/usr/bin/env python3
"""Standalone frozen-encoder fine-tuning script.

This is the dedicated training script for the frozen experiment.  Compared
with the shared variant entry, it bakes in a configuration tuned to use as
much of the RTX 4090D memory as possible for a single-GPU frozen-encoder run:

* ``--batch-size 16000`` (token-based dynamic batching)
* ``--num-workers 8``
* fp32 by default (``--no-use-fp16``); FunASR 1.4's decoder has a known fp16
  dimension bug, so fp16 is off to keep the frozen run stable.

The encoder is frozen with ``++freeze_param=encoder``.  Logging, real-time
loss curves, checkpointing, and final evaluation follow the same contract as
the FFT experiment: ``train_frozen.log``, ``loss_history.json``,
``loss_curve.png``, and ``*.errors.tsv`` whose first row is the overall CER.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import train_fft as fft


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/root/autodl-tmp/data"))
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("/root/autodl-tmp/models/paraformer-large"),
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=Path("/root/autodl-tmp/data/funasr_prepared"),
    )
    parser.add_argument("--run-name", default="frozen_bs16000_ep20")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/autodl-tmp/experiments"),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/root/autodl-tmp/results"),
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("/root/autodl-tmp/models"),
    )
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--valid-manifest", type=Path, default=None)
    parser.add_argument("--eval-splits", default="dev,test")
    parser.add_argument("--batch-size", type=int, default=16000)
    parser.add_argument("--max-epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--validate-interval", type=int, default=1000)
    parser.add_argument("--save-checkpoint-interval", type=int, default=1000)
    parser.add_argument("--keep-nbest-models", type=int, default=20)
    parser.add_argument("--avg-nbest-model", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--master-port", type=int, default=29501)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--freeze-params", default="encoder")
    parser.add_argument(
        "--use-fp16",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    train_manifest = args.train_manifest or (args.data_dir / "train.jsonl")
    valid_manifest = args.valid_manifest or (args.data_dir / "dev.jsonl")
    prepared_train = args.prepared_dir / "train.funasr.jsonl"
    prepared_valid = args.prepared_dir / "dev.funasr.jsonl"

    if not args.skip_prepare:
        print(f"Preparing training data from {train_manifest}", flush=True)
        fft.prepare_funasr_manifest(train_manifest, prepared_train)
        print(f"Preparing validation data from {valid_manifest}", flush=True)
        fft.prepare_funasr_manifest(valid_manifest, prepared_valid)

    output_dir = args.output_root / args.run_name
    results_dir = args.results_root / args.run_name
    model_out_dir = args.model_root / f"paraformer-{args.run_name}"

    if not args.skip_training:
        fft.run_training(
            train_script=fft.find_funasr_train_script(),
            base_model=args.base_model,
            train_manifest=prepared_train,
            valid_manifest=prepared_valid,
            output_dir=output_dir,
            batch_size=args.batch_size,
            max_epoch=args.max_epoch,
            lr=args.lr,
            validate_interval=args.validate_interval,
            save_checkpoint_interval=args.save_checkpoint_interval,
            keep_nbest_models=args.keep_nbest_models,
            avg_nbest_model=args.avg_nbest_model,
            use_fp16=args.use_fp16,
            num_workers=args.num_workers,
            master_port=args.master_port,
            extra_args=[f"++freeze_param={args.freeze_params}"],
            log_name="train_frozen.log",
        )

    packaged_model = fft.package_model(
        base_model=args.base_model,
        output_dir=output_dir,
        model_out_dir=model_out_dir,
        avg_nbest_model=args.avg_nbest_model,
        adapter_kwargs=None,
    )

    if not args.skip_eval:
        eval_script = (
            Path(__file__).resolve().parent.parent / "evaluation" / "eval_asr.py"
        )
        if not eval_script.exists():
            raise SystemExit(f"Missing evaluation script: {eval_script}")
        fft.run_evaluation(
            eval_script=eval_script,
            model_dir=packaged_model,
            data_dir=args.data_dir,
            splits=args.eval_splits,
            results_dir=results_dir,
            batch_size=args.eval_batch_size,
            limit=args.limit,
        )

    print("Done.", flush=True)
    print(f"training_dir={output_dir}", flush=True)
    print(f"model_dir={model_out_dir}", flush=True)
    print(f"results_dir={results_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
