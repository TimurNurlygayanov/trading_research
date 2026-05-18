"""
Leakage checks — static (AST scan) and runtime (function-purity test).

Run static_scan() before training; if it returns issues, fix them before proceeding.
Run assert_no_lookahead() in tests to verify feature builders are past-only.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class ScanResult:
    file: str
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


_FORBIDDEN_PATTERNS = [
    (r"\.shift\(\s*-\s*\d+", "shift(-N) — accesses future values"),
    (r"\.iloc\[\s*\w+\s*\+\s*\d+", "iloc[i+N] — future bar access"),
    (r"\.iloc\[\s*\w+\s*\+\s*1\s*:", "iloc[i+1:] — slicing future bars"),
    (r"min_periods\s*=\s*0", "min_periods=0 — unstable early values"),
    (r"\.rolling\([^)]*center\s*=\s*True", "rolling(center=True) — uses future"),
    (r"\.bfill\(", "bfill — fills from future"),
    (r"backfill", "backfill — fills from future"),
    (r"\.tail\(", "tail() inside features — may peek at future"),
]


def static_scan(path: str | Path) -> ScanResult:
    """Regex + AST scan of a Python file for lookahead patterns."""
    p = Path(path)
    src = p.read_text(encoding="utf-8")
    res = ScanResult(file=str(p))

    for pattern, desc in _FORBIDDEN_PATTERNS:
        for m in re.finditer(pattern, src):
            line_no = src[:m.start()].count("\n") + 1
            res.issues.append(f"L{line_no}: {desc} ({m.group(0)!r})")

    try:
        ast.parse(src)
    except SyntaxError as e:
        res.issues.append(f"SyntaxError: {e}")

    return res


def assert_no_lookahead(
    feature_fn: Callable[[pd.DataFrame], pd.Series],
    df: pd.DataFrame,
    sample_indices: list[int] | None = None,
    tol: float = 1e-9,
) -> None:
    """
    Runtime contract test: f(df[:t]) at position t-1 must equal f(df[:t+K]) at position t-1.

    Picks a few random indices; recomputes the feature with strictly less and strictly more
    future data; asserts the value at the cutoff is identical.
    """
    if len(df) < 300:
        return
    if sample_indices is None:
        rng = np.random.default_rng(42)
        sample_indices = sorted(rng.integers(200, len(df) - 50, size=5).tolist())

    for t in sample_indices:
        short = feature_fn(df.iloc[:t])
        long_ = feature_fn(df.iloc[:t + 30])
        a = short.iloc[-1]
        b = long_.iloc[t - 1]
        if pd.isna(a) and pd.isna(b):
            continue
        if abs(float(a) - float(b)) > tol:
            raise AssertionError(
                f"Lookahead detected at t={t}: short_tail={a} != long_at_same_idx={b}"
            )


def assert_tp_sl_invariant(entry: float, sl: float, tp: float, side: str) -> None:
    """Hard guardrail against TP/SL swaps."""
    if side == "long":
        if not (tp > entry > sl):
            raise AssertionError(f"long invariant violated: tp={tp} entry={entry} sl={sl}")
    elif side == "short":
        if not (tp < entry < sl):
            raise AssertionError(f"short invariant violated: tp={tp} entry={entry} sl={sl}")
    else:
        raise ValueError(f"unknown side {side!r}")
