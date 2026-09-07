#!/usr/bin/env python3
"""Generate a speaker-disjoint train/dev/test split on the AutoDL data disk.

The source of truth is the extracted flat data directory containing
``*.wav`` plus ``data_sichuan.jsonl``.  The script reads utterance ids/text from
that manifest and writes absolute-path JSONL files for FunASR-style training.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/root/autodl-tmp/data"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    manifest = data_dir / "data_sichuan.jsonl"
    if not manifest.exists():
        raise SystemExit(f"Missing manifest: {manifest}")

    text: dict[str, str] = {}
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            utt_id = Path(row["source"]).stem
            text[utt_id] = row["text"]

    by_speaker: dict[str, list[str]] = defaultdict(list)
    missing: list[str] = []
    for utt_id in sorted(text):
        audio = data_dir / f"{utt_id}.wav"
        if not audio.exists():
            missing.append(utt_id)
            continue
        speaker = utt_id.split("_", 1)[0]
        by_speaker[speaker].append(utt_id)

    if missing:
        raise SystemExit(f"Missing wavs in data dir: {len(missing)}")

    speakers = sorted(by_speaker)
    rng = random.Random(args.seed)
    rng.shuffle(speakers)
    n = len(speakers)
    n_train = round(n * args.train_ratio)
    n_val = round(n * args.val_ratio)
    n_test = n - n_train - n_val

    train_spk = set(speakers[:n_train])
    val_spk = set(speakers[n_train : n_train + n_val])
    test_spk = set(speakers[n_train + n_val :])

    subsets = {
        "train": [u for s in train_spk for u in by_speaker[s]],
        "dev": [u for s in val_spk for u in by_speaker[s]],
        "test": [u for s in test_spk for u in by_speaker[s]],
    }

    for name, utt_ids in subsets.items():
        out_path = data_dir / f"{name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for utt_id in sorted(utt_ids):
                source = f"/root/autodl-tmp/data/{utt_id}.wav"
                f.write(
                    json.dumps(
                        {"source": source, "text": text[utt_id]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    total_utts = len(text)
    print("=========== Server split report ===========")
    for name in ("train", "dev", "test"):
        rows = subsets[name]
        print(f"{name:>5}: rows={len(rows):>6}  speakers={len(set(u.split('_',1)[0] for u in rows)):>5}")
    print(f"total: {total_utts}")
    print("Row ratio:")
    for name in ("train", "dev", "test"):
        print(f"  {name}: {len(subsets[name])/total_utts:.4f}")

    spk_sets = {
        name: {u.split("_", 1)[0] for u in utts}
        for name, utts in subsets.items()
    }
    overlaps = [
        (a, b, len(spk_sets[a] & spk_sets[b]))
        for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))
    ]
    for a, b, count in overlaps:
        print(f"overlap {a}&{b}: {count}")
    print("===========================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
