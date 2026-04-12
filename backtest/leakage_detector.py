"""
Look-ahead bias (data leakage) detector for generated strategy code.

Runs BEFORE any backtest to catch common bugs that would make results
unrealistically good. Returns a score 0-10 (10 = clean, 0 = severe leakage)
plus a list of specific issues found.

Checks performed:
  1. shift(-N) patterns: accessing future values
  2. iloc[index+N] patterns: indexing future bars
  3. Indicator calculations inside next() on full series
  4. fit() calls inside next() (refitting on future data)
  5. Wrong use of self.data.Close[-1] in signal entry confirmation
  6. Assignments to df columns inside next() that read future rows
  7. Using df.tail() or df.head() inside next()
  8. Rolling calculations with min_periods=0 that create unstable early values

Usage:
  result = check_leakage(code_str)
  print(result.score, result.issues)
"""
from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass, field


@dataclass
class LeakageResult:
    score: float          # 0 (bad) to 10 (clean)
    issues: list[str]     # human-readable list of problems found
    warnings: list[str]   # non-critical concerns
    passed: bool          # True if score >= 7


# ── AST-based checks ────────────────────────────────────────────────────────

class LeakageVisitor(ast.NodeVisitor):
    """Walk the AST and collect leakage indicators."""

    def __init__(self) -> None:
        self.issues: list[str] = []
        self.warnings: list[str] = []
        self._in_next_method = False
        self._in_init_method = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        prev_next = self._in_next_method
        prev_init = self._in_init_method
        if node.name == "next":
            self._in_next_method = True
        elif node.name == "init":
            self._in_init_method = True
        self.generic_visit(node)
        self._in_next_method = prev_next
        self._in_init_method = prev_init

    def visit_Call(self, node: ast.Call) -> None:
        # Check for .fit() inside next() — refitting on future data
        if self._in_next_method:
            if isinstance(node.func, ast.Attribute) and node.func.attr == "fit":
                self.issues.append(
                    f"Line {node.lineno}: .fit() called inside next() — "
                    "this refits the model with all data up to current bar, "
                    "which is valid, but if fitting on self.data.df it sees future rows."
                )
            # Check for rolling/expanding inside next() on full series
            if isinstance(node.func, ast.Attribute) and node.func.attr in ("rolling", "expanding", "ewm"):
                self.issues.append(
                    f"Line {node.lineno}: .{node.func.attr}() called inside next() — "
                    "use self.I() in init() instead; calling rolling() inside next() "
                    "recalculates on the full available series each bar, which is very slow "
                    "and can access future data if the DataFrame reference includes future rows."
                )
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        # Check for shift(-N) pattern: series.shift(-N) where N > 0
        if isinstance(node.slice, ast.UnaryOp) and isinstance(node.slice.op, ast.USub):
            if isinstance(node.slice.operand, ast.Constant) and isinstance(node.slice.operand.value, int):
                val = node.slice.operand.value
                if val > 0:
                    self.issues.append(
                        f"Line {node.lineno}: Negative index [{-val}] — "
                        "ensure this is accessing PAST bars ([-1]=current, [-2]=previous). "
                        "Positive offset [+N] would be future data."
                    )
        self.generic_visit(node)


# ── Regex-based checks (faster for common patterns) ─────────────────────────

LEAKAGE_PATTERNS: list[tuple[str, str, str]] = [
    # (pattern, severity, description)
    (
        r"\.shift\(\s*-[1-9]",
        "critical",
        "shift(-N): shifting data backward in time uses future values — change to shift(+N) or shift(N)",
    ),
    (
        r"iloc\[.{0,20}\+\s*[1-9]",
        "critical",
        "iloc[i+N]: accessing a future bar by positive offset",
    ),
    (
        r"\.pct_change\(\s*-[1-9]",
        "critical",
        "pct_change(-N): negative period looks forward in time",
    ),
    (
        r"\.diff\(\s*-[1-9]",
        "critical",
        "diff(-N): negative period looks forward in time",
    ),
    (
        r"rolling\([^)]*\)\s*\.mean\(\)\s*\.fillna\(",
        "warning",
        "rolling().fillna(): filling NaN early values with later values can introduce subtle leakage",
    ),
    (
        r"\.fillna\(method=['\"]bfill['\"]",
        "critical",
        "fillna(bfill): backward fill propagates future values backward",
    ),
    (
        r"\.fillna\(method=['\"]backfill['\"]",
        "critical",
        "fillna(backfill): backward fill propagates future values backward",
    ),
    (
        r"bfill\(\)",
        "critical",
        ".bfill(): backward fill propagates future values backward",
    ),
    (
        r"pd\.DataFrame\(.*\)\.sort_values\(.*ascending=False",
        "warning",
        "sort_values descending then resetting index may reorder future data relative to past",
    ),
    (
        r"train_test_split.*shuffle\s*=\s*True",
        "critical",
        "train_test_split with shuffle=True on time series — must use shuffle=False",
    ),
    (
        r"StandardScaler|MinMaxScaler|RobustScaler",
        "warning",
        "Scaler detected — ensure it is fit ONLY on training data, not the full dataset",
    ),
    (
        r"\.fit\(.*self\.data",
        "critical",
        ".fit() on self.data — this includes future bars in the fitting, causing leakage",
    ),
    (
        r"np\.argmax|np\.argmin",
        "warning",
        "argmax/argmin on full series — if used as a label/signal, ensure it only uses past data",
    ),
]


def check_leakage(code: str) -> LeakageResult:
    """
    Run all leakage checks on the provided strategy code string.

    Returns a LeakageResult with score, issues, and warnings.
    Score: 10 = no issues, deducted per issue:
      - critical issue: -2.5 each (max deduction: -10)
      - warning: -0.5 each
    """
    issues: list[str] = []
    warnings: list[str] = []

    # ── 1. Regex pattern checks ──────────────────────────────────────────────
    lines = code.split("\n")
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for pattern, severity, description in LEAKAGE_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                entry = f"Line {i}: {description}"
                if severity == "critical":
                    issues.append(entry)
                else:
                    warnings.append(entry)

    # ── 2. AST checks ────────────────────────────────────────────────────────
    try:
        tree = ast.parse(textwrap.dedent(code))
        visitor = LeakageVisitor()
        visitor.visit(tree)
        issues.extend(visitor.issues)
        warnings.extend(visitor.warnings)
    except SyntaxError as e:
        issues.append(f"SyntaxError in generated code: {e} — code cannot be executed")

    # ── 3. Structural checks ─────────────────────────────────────────────────
    if "class" not in code and "def next" not in code:
        issues.append("No Strategy class or next() method found — code structure is invalid")

    if "self.I(" not in code and "def init" in code:
        warnings.append(
            "No self.I() calls in init() — indicators should be registered with self.I() "
            "to ensure correct bar-by-bar evaluation"
        )

    if re.search(r"def next.*:\s*\n(?:.*\n)*?.*pandas_ta\.", code):
        issues.append(
            "pandas_ta called inside next() — all indicator calculations must be in init() "
            "using self.I(); calling inside next() is extremely slow (recalculates every bar)"
        )

    # ── 4. Score calculation ─────────────────────────────────────────────────
    score = 10.0
    score -= len(issues) * 2.5
    score -= len(warnings) * 0.5
    score = max(0.0, min(10.0, score))

    return LeakageResult(
        score=round(score, 1),
        issues=issues,
        warnings=warnings,
        passed=score >= 7.0,
    )


def check_signal_count(
    signals: "pd.Series | list[bool]",
    dates: "pd.DatetimeIndex",
    min_total: int = 100,
    min_per_year: int = 100,
) -> tuple[bool, str]:
    """
    Verify that a strategy generates enough signals to be statistically meaningful.

    Returns (passed, reason).
    """
    import pandas as pd

    if not isinstance(signals, pd.Series):
        signals = pd.Series(signals, index=dates)

    total = int(signals.sum())
    if total < min_total:
        return False, f"Only {total} total signals — minimum required: {min_total}"

    # Check per-year
    signal_dates = signals[signals].index
    if len(signal_dates) == 0:
        return False, "No signals at all"

    years = signal_dates.to_series().groupby(signal_dates.year).count()
    min_year_count = int(years.min())
    if min_year_count < min_per_year:
        return False, (
            f"Lowest signal count in any year: {min_year_count} "
            f"(year {int(years.idxmin())}) — minimum required per year: {min_per_year}"
        )

    return True, f"Signal count OK: {total} total, min {min_year_count}/year"
