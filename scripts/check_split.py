#!/usr/bin/env python3
"""Check whether a KeSpeech Sichuan/Chongqing split is speaker-disjoint and balanced."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_kv(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                result[parts[0]] = parts[1]
    return result


def load_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def utt_id_from_source(source: str) -> str:
    return Path(source.replace("\\", "/")).stem


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(r"D:/NEW/sichuan_perfect_data"),
    )
    parser.add_argument(
        "--kepeech-root",
        type=Path,
        default=Path(
            r"D:/SHUJUJI/KeSpeech/KeSpeech.splits/KeSpeech.tar.gz/KeSpeech/KeSpeech"
        ),
    )
    args = parser.parse_args()

    metadata = args.kepeech_root / "Metadata"
    spk2city = parse_kv(metadata / "spk2city")
    spk2gender = parse_kv(metadata / "spk2gender")

    splits: dict[str, list[dict[str, str]]] = {}
    for name in ("train", "dev", "test"):
        path = args.data_dir / f"{name}.jsonl"
        if not path.exists():
            print(f"Missing split file: {path}")
            return 2
        splits[name] = load_jsonl(path)

    print("================ Split quality report ================")
    all_ids = [utt_id_from_source(r["source"]) for rows in splits.values() for r in rows]
    dup_ids = [k for k, v in Counter(all_ids).items() if v > 1]
    print(f"Total rows       : {len(all_ids)}")
    print(f"Duplicate utts   : {len(dup_ids)}")

    spk_by_split: dict[str, set[str]] = {}
    for name, rows in splits.items():
        spks = {utt_id_from_source(r["source"]).split("_", 1)[0] for r in rows}
        spk_by_split[name] = spks
        cities = Counter(spk2city.get(s, "?") for s in spks)
        genders = Counter(spk2gender.get(s, "?") for s in spks)
        print(
            f"{name:>5}: rows={len(rows):>6}  speakers={len(spks):>5}  "
            f"cities={dict(cities)}  gender={dict(genders)}"
        )

    total = len(all_ids)
    print("Row ratio:")
    for name, rows in splits.items():
        print(f"  {name}: {len(rows)/total:.4f}")

    print("Speaker overlap:")
    overlap_found = False
    for a in ("train", "dev", "test"):
        for b in ("dev", "test", "train"):
            if a < b:
                inter = spk_by_split[a] & spk_by_split[b]
                if inter:
                    overlap_found = True
                    print(f"  OVERLAP {a} & {b}: {len(inter)} speakers")
    if not overlap_found:
        print("  None. Speaker isolation is correct.")

    print("=====================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
