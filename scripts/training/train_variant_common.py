"""Shared command-line logic for frozen/LoRA/DoRA fine-tuning scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

import train_fft as fft


def add_common_args(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument("--run-name", default=None)
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
    parser.add_argument("--batch-size", type=int, default=4000)
    parser.add_argument("--max-epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--validate-interval", type=int, default=1000)
    parser.add_argument("--save-checkpoint-interval", type=int, default=1000)
    parser.add_argument("--keep-nbest-models", type=int, default=20)
    parser.add_argument("--avg-nbest-model", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--master-port", type=int, default=29501)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--use-fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable mixed-precision training (disable with --no-use-fp16).",
    )
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")


def add_lora_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lora-list", default="q,k,v,o")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)


def add_frozen_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--freeze-params",
        default="encoder",
        help="Comma-separated parameter prefixes passed to FunASR freeze_param.",
    )


def parse_lora_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run_variant(method: str) -> int:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    if method == "frozen":
        add_frozen_args(parser)
    elif method in ("lora", "dora"):
        add_lora_args(parser)
    else:
        raise ValueError(f"Unknown method: {method}")
    args = parser.parse_args()

    run_name = args.run_name or method
    train_manifest = args.train_manifest or (args.data_dir / "train.jsonl")
    valid_manifest = args.valid_manifest or (args.data_dir / "dev.jsonl")
    prepared_train = args.prepared_dir / "train.funasr.jsonl"
    prepared_valid = args.prepared_dir / "dev.funasr.jsonl"

    if not args.skip_prepare:
        print(f"Preparing training data from {train_manifest}", flush=True)
        fft.prepare_funasr_manifest(train_manifest, prepared_train)
        print(f"Preparing validation data from {valid_manifest}", flush=True)
        fft.prepare_funasr_manifest(valid_manifest, prepared_valid)

    output_dir = args.output_root / run_name
    results_dir = args.results_root / run_name
    model_out_dir = args.model_root / f"paraformer-{run_name}"

    extra_args: list[str] = []
    adapter_kwargs: dict | None = None
    train_script = fft.find_funasr_train_script()

    if method == "frozen":
        extra_args.append(f"++freeze_param={args.freeze_params}")
    elif method in ("lora", "dora"):
        lora_list = parse_lora_list(args.lora_list)
        lora_cfg = {
            "lora_list": lora_list,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        }
        list_str = ",".join(lora_list)
        extra_args.extend(
            [
                f"++encoder_conf.lora_list={list_str}",
                f"++encoder_conf.lora_rank={args.lora_rank}",
                f"++encoder_conf.lora_alpha={args.lora_alpha}",
                f"++encoder_conf.lora_dropout={args.lora_dropout}",
                f"++decoder_conf.lora_list={list_str}",
                f"++decoder_conf.lora_rank={args.lora_rank}",
                f"++decoder_conf.lora_alpha={args.lora_alpha}",
                f"++decoder_conf.lora_dropout={args.lora_dropout}",
                "++lora_only=True",
                "++lora_bias=none",
            ]
        )
        adapter_kwargs = {"lora_cfg": lora_cfg, "dora": method == "dora"}
        if method == "dora":
            train_script = Path(__file__).with_name("dora_train_entry.py")
            if not train_script.exists():
                raise SystemExit(f"Missing DoRA training entry: {train_script}")

    if not args.skip_training:
        fft.run_training(
            train_script=train_script,
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
            extra_args=extra_args,
            log_name=f"train_{method}.log",
        )

    packaged_model = fft.package_model(
        base_model=args.base_model,
        output_dir=output_dir,
        model_out_dir=model_out_dir,
        avg_nbest_model=args.avg_nbest_model,
        adapter_kwargs=adapter_kwargs,
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
    print(f"method={method}", flush=True)
    print(f"training_dir={output_dir}", flush=True)
    print(f"model_dir={model_out_dir}", flush=True)
    print(f"results_dir={results_dir}", flush=True)
    return 0
