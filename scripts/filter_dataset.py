#!/usr/bin/env python3
"""Filter a Sichuan/Chongqing dialect subset from the raw KeSpeech dataset.

The KeSpeech metadata is split into phase1 and phase2.  For the thesis, the
cleanest Sichuan/Chongqing dialect set is obtained by keeping only phase1
utterances whose speaker comes from Chengdu or Chongqing and whose metadata is
labelled ``Southwestern`` + ``Dialect``.

Examples
--------
# Dry-run, print counts only:
python scripts/filter_dataset.py --dry-run

# Generate a portable upload package (hardlinks avoid duplicating the audio):
python scripts/filter_dataset.py --output-dir D:/毕设整理/chuanyu_dataset --stage wavs

# Broader set: all phase1 speech from Chengdu/Chongqing speakers:
python scripts/filter_dataset.py --no-strict
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import wave
from pathlib import Path


def parse_kv(path: Path) -> dict[str, str]:
    """Parse ``key value`` style metadata files."""
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


def parse_text(path: Path) -> dict[str, str]:
    """Parse ``utt_id transcription`` style files."""
    return parse_kv(path)


def split_by_speaker(
    records: list[dict[str, str]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Speaker-disjoint split.

    Splits by speaker rather than by utterance so that no speaker appears in
    more than one subset.
    """
    by_speaker: dict[str, list[dict[str, str]]] = {}
    for rec in records:
        by_speaker.setdefault(rec["speaker"], []).append(rec)

    speakers = sorted(by_speaker.keys())
    rng = random.Random(seed)
    rng.shuffle(speakers)

    n = len(speakers)
    n_train = round(n * train_ratio)
    n_val = round(n * val_ratio)
    # Ensure test always contains at least one speaker.
    n_test = max(1, n - n_train - n_val)
    n_train = n - n_val - n_test

    train_speakers = set(speakers[:n_train])
    val_speakers = set(speakers[n_train : n_train + n_val])
    test_speakers = set(speakers[n_train + n_val :])

    train: list[dict[str, str]] = []
    val: list[dict[str, str]] = []
    test: list[dict[str, str]] = []
    for rec in records:
        spk = rec["speaker"]
        if spk in train_speakers:
            train.append(rec)
        elif spk in val_speakers:
            val.append(rec)
        else:
            test.append(rec)
    return train, val, test


def wav_duration(path: Path) -> float | None:
    """Read duration from a PCM WAV header without decoding the audio."""
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return None


def load_phase1_records(
    kept_root: Path,
    target_cities: set[str],
    strict: bool,
) -> list[dict[str, str]]:
    metadata = kept_root / "Metadata"
    spk2city = parse_kv(metadata / "spk2city")
    text = parse_text(metadata / "phase1.text")
    sub = parse_kv(metadata / "phase1.utt2subdialect")
    style = parse_kv(metadata / "phase1.utt2style")

    records: list[dict[str, str]] = []
    for utt_id, transcription in text.items():
        speaker = utt_id.split("_", 1)[0]
        city = spk2city.get(speaker, "")
        if city not in target_cities:
            continue
        if strict and (sub.get(utt_id) != "Southwestern" or style.get(utt_id) != "Dialect"):
            continue
        records.append(
            {
                "utt_id": utt_id,
                "speaker": speaker,
                "city": city,
                "text": transcription,
                "subdialect": sub.get(utt_id, ""),
                "style": style.get(utt_id, ""),
                "audio_rel": f"Audio/{speaker}/phase1/{utt_id}.wav",
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kepeech-root",
        type=Path,
        default=Path(
            r"D:/SHUJUJI/KeSpeech/KeSpeech.splits/KeSpeech.tar.gz/KeSpeech/KeSpeech"
        ),
        help="Root of the extracted KeSpeech corpus.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"D:/毕设整理/chuanyu_dataset"),
        help="Directory where the filtered manifests are written.",
    )
    parser.add_argument(
        "--cities",
        default="Chengdu,Chongqing",
        help="Comma-separated city names to keep.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only Southwestern + Dialect utterances (default).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the report, do not write files or copy audio.",
    )
    parser.add_argument(
        "--stage",
        choices=["copy", "hardlink"],
        default=None,
        help="Also stage selected wavs into output-dir/wavs (copy or hardlink).",
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Write audio paths as 'wavs/<utt_id>.wav' instead of local absolute paths.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for the speaker-disjoint split.",
    )
    args = parser.parse_args()

    kept_root = args.kepeech_root
    if not kept_root.exists():
        print(f"ERROR: KeSpeech root not found: {kept_root}", file=sys.stderr)
        return 2

    target_cities = {c.strip() for c in args.cities.split(",") if c.strip()}
    records = load_phase1_records(kept_root, target_cities, args.strict)

    missing_audio = 0
    for rec in records:
        if not (kept_root / rec["audio_rel"]).exists():
            missing_audio += 1

    speakers = sorted({r["speaker"] for r in records})
    cities = sorted({r["city"] for r in records})
    print("================ KeSpeech Sichuan/Chongqing filter report ================")
    print(f"KeSpeech root : {kept_root}")
    print(f"Target cities : {', '.join(cities)}")
    print(f"Strict filter : {args.strict}")
    print(f"Selected utts : {len(records)}")
    print(f"Selected spks : {len(speakers)}")
    print(f"Missing audio : {missing_audio}")
    print("==========================================================================")

    if args.dry_run:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = args.output_dir / "wavs"
    if args.stage:
        wav_dir.mkdir(parents=True, exist_ok=True)

    train, val, test = split_by_speaker(
        records, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=args.seed
    )

    def write_manifests(subset: list[dict[str, str]], name: str) -> None:
        with (args.output_dir / f"{name}.text").open("w", encoding="utf-8") as ftext:
            with (args.output_dir / f"{name}.wav.scp").open("w", encoding="utf-8") as fwav:
                with (args.output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as fjson:
                    for rec in sorted(subset, key=lambda x: x["utt_id"]):
                        ftext.write(f"{rec['utt_id']} {rec['text']}\n")

                        if args.stage:
                            src = kept_root / rec["audio_rel"]
                            dst = wav_dir / f"{rec['utt_id']}.wav"
                            if not dst.exists():
                                if args.stage == "hardlink":
                                    try:
                                        os.link(src, dst)
                                    except Exception:
                                        shutil.copy2(src, dst)
                                else:
                                    shutil.copy2(src, dst)
                            rel_audio = f"wavs/{rec['utt_id']}.wav"
                        elif args.relative_paths:
                            rel_audio = f"wavs/{rec['utt_id']}.wav"
                        else:
                            rel_audio = str(kept_root / rec["audio_rel"])

                        fwav.write(f"{rec['utt_id']} {rel_audio}\n")
                        fjson.write(
                            json.dumps(
                                {"source": rel_audio, "text": rec["text"]},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

    write_manifests(train, "train")
    write_manifests(val, "dev")
    write_manifests(test, "test")

    # utt2spk/spk2utt for the whole filtered set.
    with (args.output_dir / "utt2spk").open("w", encoding="utf-8") as fu2s, (
        args.output_dir / "spk2utt"
    ).open("w", encoding="utf-8") as fs2u:
        by_speaker: dict[str, list[str]] = {}
        for rec in sorted(records, key=lambda x: x["utt_id"]):
            fu2s.write(f"{rec['utt_id']} {rec['speaker']}\n")
            by_speaker.setdefault(rec["speaker"], []).append(rec["utt_id"])
        for spk in sorted(by_speaker):
            fs2u.write(f"{spk} {' '.join(sorted(by_speaker[spk]))}\n")

    print(f"Wrote manifests to: {args.output_dir}")
    print(
        "Split counts -> train: {}, dev: {}, test: {}".format(
            len(train), len(val), len(test)
        )
    )
    if args.stage:
        print(f"Staged wavs into: {wav_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
