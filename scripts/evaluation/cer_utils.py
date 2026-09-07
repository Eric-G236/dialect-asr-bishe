"""Shared Chinese character error rate (CER) helpers."""

from __future__ import annotations


def normalize_text(text: str) -> str:
    """Return a canonical character sequence for CER calculation."""
    return text.replace(" ", "")


def edit_distance_detail(ref: str, hyp: str) -> tuple[int, int, int, int]:
    """Return (distance, substitutions, deletions, insertions).

    Substitution/insertion/deletion counts come from a standard dynamic
    programming traceback.  When multiple optimal paths exist, this
    implementation prefers substitutions over insertions/deletions, which keeps
    the breakdown deterministic.
    """
    ref = list(ref)
    hyp = list(hyp)
    n = len(ref)
    m = len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    i, j = n, m
    sub = dele = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (
            0 if ref[i - 1] == hyp[j - 1] else 1
        ):
            if ref[i - 1] != hyp[j - 1]:
                sub += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1

    return dp[n][m], sub, dele, ins
