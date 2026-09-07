#!/usr/bin/env python3
"""Reusable FunASR evaluation for the Sichuan-dialect ASR project.

Compared with the original ``run_baseline.py``, this script additionally writes
the overall CER as the first data row of each ``*.errors.tsv`` file, so every
experiment table starts with the whole-split result.

Outputs, for each split:
  - <split>.predictions.jsonl : model outputs
  - <split>.errors.tsv        : overall CER row followed by per-utterance errors
  - <split>.summary.txt       : human-readable CER summary
  - <split>.summary.json      : machine-readable CER summary
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from funasr import AutoModel

from cer_utils import edit_distance_detail, normalize_text


def load_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate_split(
    model: AutoModel,
    manifest: Path,
    split: str,
    output_dir: Path,
    batch_size: int,
    limit: int | None = None,
) -> dict[str, object]:
    """Evaluate one manifest and write all requested artifacts."""
    rows = load_jsonl(manifest)
    if limit:
        rows = rows[:limit]
    print(f"Evaluating {split}: {len(rows)} utterances", flush=True)

    batch_inputs: list[str] = []
    batch_rows: list[dict[str, str]] = []
    predictions: list[dict[str, str]] = []

    def flush_batch() -> None:
        nonlocal batch_inputs, batch_rows
        if not batch_inputs:
            return
        results = model.generate(input=batch_inputs, batch_size_s=300)
        pred_by_key: dict[str, str] = {}
        for item in results:
            if isinstance(item, dict):
                key = item.get("key") or Path(item.get("input", "")).stem
                pred_by_key[key] = item.get("text", "")
        for idx, row in enumerate(batch_rows):
            utt = Path(row["source"]).stem
            hyp = pred_by_key.get(utt, "")
            if not hyp and idx < len(results):
                hyp = (
                    results[idx].get("text", "")
                    if isinstance(results[idx], dict)
                    else ""
                )
            predictions.append({"utt_id": utt, "ref": row["text"], "hyp": hyp})
        batch_inputs = []
        batch_rows = []

    for row in rows:
        batch_inputs.append(row["source"])
        batch_rows.append(row)
        if len(batch_inputs) >= batch_size:
            flush_batch()
    flush_batch()

    pred_path = output_dir / f"{split}.predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    error_rows = []
    total_ref_len = 0
    total_errors = 0
    total_sub = total_del = total_ins = 0
    for item in predictions:
        ref = normalize_text(item["ref"])
        hyp = normalize_text(item["hyp"])
        dist, sub, dele, ins = edit_distance_detail(ref, hyp)
        ref_len = len(ref)
        cer = (dist / ref_len * 100.0) if ref_len else 0.0
        total_ref_len += ref_len
        total_errors += dist
        total_sub += sub
        total_del += dele
        total_ins += ins
        error_rows.append(
            {
                "utt_id": item["utt_id"],
                "ref": item["ref"],
                "hyp": item["hyp"],
                "ref_len": ref_len,
                "errors": dist,
                "cer": round(cer, 2),
                "substitutions": sub,
                "deletions": dele,
                "insertions": ins,
            }
        )

    overall_cer = (total_errors / total_ref_len * 100.0) if total_ref_len else 0.0
    tsv_path = output_dir / f"{split}.errors.tsv"
    header = [
        "utt_id",
        "cer",
        "errors",
        "ref_len",
        "substitutions",
        "deletions",
        "insertions",
        "ref",
        "hyp",
    ]
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        f.write(
            "\t".join(
                [
                    "[overall]",
                    f"{overall_cer:.2f}",
                    str(total_errors),
                    str(total_ref_len),
                    str(total_sub),
                    str(total_del),
                    str(total_ins),
                    "",
                    "",
                ]
            )
            + "\n"
        )
        for r in sorted(error_rows, key=lambda x: x["cer"], reverse=True):
            f.write(
                "\t".join(
                    [
                        r["utt_id"],
                        f"{r['cer']:.2f}",
                        str(r["errors"]),
                        str(r["ref_len"]),
                        str(r["substitutions"]),
                        str(r["deletions"]),
                        str(r["insertions"]),
                        r["ref"],
                        r["hyp"],
                    ]
                )
                + "\n"
            )

    summary = {
        "split": split,
        "num_utterances": len(predictions),
        "total_ref_chars": total_ref_len,
        "total_errors": total_errors,
        "cer": round(overall_cer, 4),
        "overall_cer": round(overall_cer, 4),
        "substitutions": total_sub,
        "deletions": total_del,
        "insertions": total_ins,
    }
    summary_json = output_dir / f"{split}.summary.json"
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary_txt = output_dir / f"{split}.summary.txt"
    summary_txt.write_text(
        "\n".join(
            [
                f"split={split}",
                f"num_utterances={summary['num_utterances']}",
                f"total_ref_chars={summary['total_ref_chars']}",
                f"total_errors={summary['total_errors']}",
                f"CER={summary['cer']:.4f}%",
                f"substitutions={summary['substitutions']}",
                f"deletions={summary['deletions']}",
                f"insertions={summary['insertions']}",
            ]
        ),
        encoding="utf-8",
    )
    print(f"{split} CER={summary['cer']:.4f}%", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--splits", default="dev,test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "8")
    model = AutoModel(
        model=args.model_dir,
        device=args.device,
        disable_update=True,
    )

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        manifest = data_dir / f"{split}.jsonl"
        if not manifest.exists():
            print(f"SKIP missing manifest: {manifest}")
            continue
        evaluate_split(
            model=model,
            manifest=manifest,
            split=split,
            output_dir=output_dir,
            batch_size=args.batch_size,
            limit=args.limit,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
