#!/usr/bin/env python3
"""Full-parameter fine-tuning for the Sichuan-dialect Paraformer experiment.

The script:
  1. Converts the existing ``train.jsonl``/``dev.jsonl`` manifests into the
     FunASR training JSONL format (``key/source/prompt/target/source_len/
     target_len``).
  2. Launches FunASR's official ``train_ds.py`` entry point with
     ``torch.distributed.run`` on one GPU.
  3. Packages the averaged best checkpoint into a normal FunASR model directory.
  4. Evaluates dev/test and writes the same artifacts as the baseline, including
     an overall CER row at the top of every ``*.errors.tsv`` file.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def get_audio_frame_len(path: str) -> int:
    """Return the number of 10 ms fbank frames FunASR will use for a wav."""
    try:
        import soundfile as sf

        info = sf.info(path)
        frames = int(info.frames)
        samplerate = int(info.samplerate)
    except Exception:
        import wave

        with wave.open(path, "rb") as wav:
            frames = wav.getnframes()
            samplerate = wav.getframerate()
    # Same convention as funasr.datasets.audio_datasets.scp2jsonl:
    # int(sample_num * 1000 / 16000 / 10).
    return int(frames * 1000 / samplerate / 10)


def prepare_funasr_manifest(src_path: Path, dst_path: Path) -> None:
    """Convert ``{source,text}`` manifest to FunASR's training JSONL format."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    count = 0
    with src_path.open("r", encoding="utf-8") as fin, dst_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            source = row["source"]
            text = row["text"]
            if not Path(source).exists():
                missing.append(source)
                continue
            key = Path(source).stem
            target = text
            source_len = get_audio_frame_len(source)
            target_len = len(target)
            record = {
                "key": key,
                "source": source,
                "prompt": "<ASR>",
                "target": target,
                "source_len": source_len,
                "target_len": target_len,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"Prepared {dst_path.name}: {count} rows", flush=True)
    if missing:
        print(f"WARNING: {len(missing)} missing audio files", flush=True)
        for item in missing[:10]:
            print(f"  missing: {item}", flush=True)


def find_funasr_train_script() -> Path:
    try:
        import funasr

        funasr_dir = Path(funasr.__file__).resolve().parent
    except Exception as exc:  # pragma: no cover - remote environment only
        raise SystemExit(f"Unable to locate funasr: {exc}") from exc
    script = funasr_dir / "bin" / "train_ds.py"
    if not script.exists():
        raise SystemExit(f"Missing FunASR training script: {script}")
    return script


def run_training(
    train_script: Path,
    base_model: Path,
    train_manifest: Path,
    valid_manifest: Path,
    output_dir: Path,
    batch_size: int,
    max_epoch: int,
    lr: float,
    validate_interval: int,
    save_checkpoint_interval: int,
    keep_nbest_models: int,
    avg_nbest_model: int,
    use_fp16: bool,
    num_workers: int,
    master_port: int,
    extra_args: list[str] | None = None,
    log_name: str = "train_fft.log",
) -> None:
    from loss_curve import load_points_from_log, parse_loss_line, save_loss_curve

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        "1",
        "--master_addr",
        "127.0.0.1",
        "--master_port",
        str(master_port),
        str(train_script),
        f"++model={base_model}",
        f"++train_data_set_list={train_manifest}",
        f"++valid_data_set_list={valid_manifest}",
        f"++dataset_conf.batch_size={batch_size}",
        "++dataset_conf.batch_type=token",
        f"++dataset_conf.num_workers={num_workers}",
        f"++train_conf.max_epoch={max_epoch}",
        f"++train_conf.validate_interval={validate_interval}",
        f"++train_conf.save_checkpoint_interval={save_checkpoint_interval}",
        f"++train_conf.keep_nbest_models={keep_nbest_models}",
        f"++train_conf.avg_nbest_model={avg_nbest_model}",
        f"++optim_conf.lr={lr}",
        f"++output_dir={output_dir}",
    ]
    if use_fp16:
        cmd.append("++train_conf.use_fp16=True")
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "8")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    print("Running training command:", flush=True)
    print(" ".join(cmd), flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / log_name
    points = load_points_from_log(log_path)
    last_plot_step = max((point["step"] for point in points), default=None)
    if last_plot_step is None:
        last_plot_step = 0

    def maybe_plot(force: bool = False) -> None:
        nonlocal last_plot_step
        if not points:
            return
        latest_step = points[-1]["step"]
        should_plot = force or (
            latest_step >= last_plot_step + save_checkpoint_interval
        )
        if should_plot:
            save_loss_curve(
                points,
                output_dir=output_dir,
                title=f"Training loss curve ({output_dir.name})",
            )
            last_plot_step = latest_step

    return_code = None
    process = None
    try:
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
                point = parse_loss_line(line)
                if point is not None:
                    points.append(point)
                    maybe_plot()
            return_code = process.wait()
    except KeyboardInterrupt:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        save_loss_curve(
            points,
            output_dir=output_dir,
            title=f"Training loss curve ({output_dir.name})",
        )
        print("Training interrupted; loss curve and log were saved.", flush=True)
        raise
    finally:
        save_loss_curve(
            points,
            output_dir=output_dir,
            title=f"Training loss curve ({output_dir.name})",
        )

    if return_code != 0:
        raise SystemExit(f"Training failed with return code {return_code}; see {log_path}")
    print(f"Training finished; full log at {log_path}", flush=True)


def select_checkpoint(output_dir: Path, avg_nbest_model: int) -> Path:
    avg = output_dir / f"model.pt.avg{avg_nbest_model}"
    if avg.exists():
        return avg

    best = output_dir / "model.pt.best"
    if best.exists():
        return best

    candidates = sorted(
        (p for p in output_dir.glob("model.pt.ep*") if p.is_file()),
        key=lambda p: (
            int("".join(ch for ch in p.name.split(".")[-1] if ch.isdigit()) or 0)
            if p.name.startswith("model.pt.ep")
            else 0
        ),
        reverse=True,
    )
    if not candidates:
        raise SystemExit(f"No usable checkpoints found in {output_dir}")
    return candidates[0]


def package_model(
    base_model: Path,
    output_dir: Path,
    model_out_dir: Path,
    avg_nbest_model: int,
    adapter_kwargs: dict | None = None,
) -> Path:
    """Copy the base FunASR model and swap in the averaged checkpoint."""
    checkpoint = select_checkpoint(output_dir, avg_nbest_model)
    if adapter_kwargs is not None:
        from adapter_utils import merge_adapter_checkpoint

        return merge_adapter_checkpoint(
            base_model_dir=base_model,
            checkpoint_path=checkpoint,
            model_out_dir=model_out_dir,
            lora_cfg=adapter_kwargs["lora_cfg"],
            dora=adapter_kwargs.get("dora", False),
        )

    model_out_dir.mkdir(parents=True, exist_ok=True)
    if model_out_dir.exists():
        shutil.rmtree(model_out_dir)
    shutil.copytree(base_model, model_out_dir)

    destination = model_out_dir / "model.pt"
    shutil.copy2(checkpoint, destination)

    with (model_out_dir / "training_source.txt").open("w", encoding="utf-8") as f:
        f.write(f"training_output_dir={output_dir}\n")
        f.write(f"checkpoint={checkpoint}\n")
        f.write(f"base_model={base_model}\n")
    print(f"Packaged model: {model_out_dir}", flush=True)
    return model_out_dir


def run_evaluation(
    eval_script: Path,
    model_dir: Path,
    data_dir: Path,
    splits: str,
    results_dir: Path,
    batch_size: int,
    limit: int | None,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(eval_script),
        "--model-dir",
        str(model_dir),
        "--data-dir",
        str(data_dir),
        "--splits",
        splits,
        "--output-dir",
        str(results_dir),
        "--batch-size",
        str(batch_size),
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])
    print("Running evaluation:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--run-name",
        default="fft",
    )
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
    args = parser.parse_args()

    train_manifest = args.train_manifest or (args.data_dir / "train.jsonl")
    valid_manifest = args.valid_manifest or (args.data_dir / "dev.jsonl")
    prepared_train = args.prepared_dir / "train.funasr.jsonl"
    prepared_valid = args.prepared_dir / "dev.funasr.jsonl"

    if not args.skip_prepare:
        print(f"Preparing training data from {train_manifest}", flush=True)
        prepare_funasr_manifest(train_manifest, prepared_train)
        print(f"Preparing validation data from {valid_manifest}", flush=True)
        prepare_funasr_manifest(valid_manifest, prepared_valid)

    output_dir = args.output_root / args.run_name
    results_dir = args.results_root / args.run_name
    model_out_dir = args.model_root / f"paraformer-{args.run_name}"

    if not args.skip_training:
        train_script = find_funasr_train_script()
        run_training(
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
        )

    packaged_model = package_model(
        base_model=args.base_model,
        output_dir=output_dir,
        model_out_dir=model_out_dir,
        avg_nbest_model=args.avg_nbest_model,
    )

    if not args.skip_eval:
        eval_script = (
            Path(__file__).resolve().parent.parent / "evaluation" / "eval_asr.py"
        )
        if not eval_script.exists():
            raise SystemExit(f"Missing evaluation script: {eval_script}")
        run_evaluation(
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
