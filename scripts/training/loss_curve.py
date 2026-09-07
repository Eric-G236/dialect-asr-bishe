"""Parse FunASR training logs and render loss/accuracy curves."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_LOSS_RE = re.compile(
    r"step_in_epoch:\s*(?P<step_epoch>\d+), total step:\s*(?P<step>\d+),"
)
_LOSS_RANK_RE = re.compile(r"\(loss_avg_rank:\s*(?P<value>[-\d.eE+]+)\)")
_LOSS_SLICE_RE = re.compile(r"\(loss_avg_slice:\s*(?P<value>[-\d.eE+]+)\)")
_ACC_SLICE_RE = re.compile(r"\(acc_avg_slice:\s*(?P<value>[-\d.eE+]+)\)")


def parse_loss_line(line: str) -> dict[str, Any] | None:
    step_match = _LOSS_RE.search(line)
    if not step_match:
        return None
    loss_rank_match = _LOSS_RANK_RE.search(line)
    loss_slice_match = _LOSS_SLICE_RE.search(line)
    acc_slice_match = _ACC_SLICE_RE.search(line)
    return {
        "step": int(step_match.group("step")),
        "step_in_epoch": int(step_match.group("step_epoch")),
        "loss_rank": float(loss_rank_match.group("value"))
        if loss_rank_match
        else None,
        "loss_slice": float(loss_slice_match.group("value"))
        if loss_slice_match
        else None,
        "acc_slice": float(acc_slice_match.group("value"))
        if acc_slice_match
        else None,
    }


def load_points_from_log(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    points: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            point = parse_loss_line(line)
            if point is not None:
                points.append(point)
    return points


def save_loss_curve(points: list[dict[str, Any]], output_dir: Path, title: str) -> None:
    if not points:
        print("No training loss points yet; skipping loss-curve output.", flush=True)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "loss_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(points, f, ensure_ascii=False, indent=2)

    steps = [p["step"] for p in points]
    loss_rank = [p["loss_rank"] for p in points]
    loss_slice = [p["loss_slice"] for p in points]
    acc_slice = [p["acc_slice"] for p in points]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax_loss, ax_acc = axes

    ax_loss.plot(steps, loss_rank, label="loss_avg_rank", linewidth=1.4)
    ax_loss.plot(steps, loss_slice, label="loss_avg_slice", linewidth=1.4)
    ax_loss.set_ylabel("loss")
    ax_loss.set_title(title)
    ax_loss.grid(True, alpha=0.25)
    ax_loss.legend()
    if all(value is not None and value > 0 for value in loss_rank):
        ax_loss.set_yscale("log")

    ax_acc.plot(steps, acc_slice, label="acc_avg_slice", linewidth=1.4, color="tab:green")
    ax_acc.set_xlabel("total step")
    ax_acc.set_ylabel("accuracy")
    ax_acc.grid(True, alpha=0.25)
    ax_acc.legend()
    fig.tight_layout()

    curve_path = output_dir / "loss_curve.png"
    fig.savefig(curve_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Updated loss curve: {curve_path}", flush=True)
