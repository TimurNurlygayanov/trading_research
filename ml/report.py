"""
Report builder — equity curve, summary metrics, per-window table, FTMO compliance.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.walkforward import WalkForwardResult


def _max_drawdown(eq: pd.Series) -> float:
    if eq.empty:
        return 0.0
    peak = eq.cummax()
    dd = (eq - peak) / peak
    return float(dd.min())


def _annualized_sharpe(trades: list, periods_per_year: float = 252.0) -> float:
    if not trades:
        return 0.0
    pnls = np.array([t.pnl for t in trades])
    if pnls.std() == 0:
        return 0.0
    days = (trades[-1].exit_time - trades[0].entry_time).days
    trades_per_year = len(trades) / max(days / 365.0, 1e-9)
    return float(pnls.mean() / pnls.std() * np.sqrt(trades_per_year))


def _worst_daily(daily_pnls: dict, start_equity: float) -> tuple[float, str | None]:
    if not daily_pnls:
        return 0.0, None
    items = sorted(daily_pnls.items(), key=lambda x: x[1])
    worst_day, worst_pnl = items[0]
    return float(worst_pnl / start_equity), str(worst_day.date())


def build_report(wf: WalkForwardResult, out_dir: Path, params: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    eq = wf.equity_curve
    trades = wf.trades

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    win_rate = len(wins) / len(trades) if trades else 0.0
    avg_win = float(np.mean([t.pnl for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([t.pnl for t in losses])) if losses else 0.0
    pf = (sum(t.pnl for t in wins) / -sum(t.pnl for t in losses)) if losses and sum(t.pnl for t in losses) < 0 else float("inf")

    daily_agg: dict = {}
    for w in wf.windows:
        for d, p in w.sim.daily_pnl.items():
            daily_agg[d] = daily_agg.get(d, 0.0) + p

    start_eq = wf.windows[0].sim.start_equity if wf.windows else 100_000.0
    end_eq = float(eq.iloc[-1]) if eq is not None and len(eq) else start_eq
    max_dd = _max_drawdown(eq) if eq is not None else 0.0
    worst_day_pct, worst_day = _worst_daily(daily_agg, start_eq)

    sharpe = _annualized_sharpe(trades)

    summary = {
        "params": params,
        "n_windows": len(wf.windows),
        "n_trades": len(trades),
        "win_rate": round(win_rate, 4),
        "avg_win_usd": round(avg_win, 2),
        "avg_loss_usd": round(avg_loss, 2),
        "profit_factor": round(pf, 3) if pf != float("inf") else None,
        "start_equity": round(start_eq, 2),
        "end_equity": round(end_eq, 2),
        "return_pct": round((end_eq / start_eq - 1) * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "worst_day_pct": round(worst_day_pct * 100, 2),
        "worst_day": worst_day,
        "trade_sharpe_annualized": round(sharpe, 3),
        "halted": wf.halted,
        "halt_reason": wf.halt_reason,
        "ftmo_max_dd_ok": max_dd > -0.08,
        "ftmo_daily_ok": worst_day_pct > -0.04,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # Trades CSV
    if trades:
        trades_df = pd.DataFrame([{
            "entry_time": t.entry_time, "exit_time": t.exit_time, "side": t.side,
            "entry": t.entry, "exit": t.exit, "sl": t.sl, "tp": t.tp,
            "size": t.size, "pnl": t.pnl, "outcome": t.outcome,
            "bars_held": t.bars_held, "proba": t.proba,
        } for t in trades])
        trades_df.to_csv(out_dir / "trades.csv", index=False)

    # Equity curve CSV
    if eq is not None and len(eq):
        eq.to_csv(out_dir / "equity.csv", header=["equity"])

    # Per-window summary
    rows = []
    for w in wf.windows:
        rows.append({
            "trade_start": w.trade_start, "trade_end": w.trade_end,
            "n_train": w.n_train, "n_val": w.n_val, "n_trade": w.n_trade,
            "train_pr_auc": w.model_stats["train_pr_auc"],
            "val_pr_auc": w.model_stats["val_pr_auc"],
            "val_base_rate": w.model_stats["val_base_rate"],
            "threshold": w.model_stats["threshold"],
            "precision": w.model_stats["precision"],
            "recall": w.model_stats["recall"],
            "trades": len(w.sim.trades),
            "tp": sum(1 for t in w.sim.trades if t.outcome == "tp"),
            "sl": sum(1 for t in w.sim.trades if t.outcome == "sl"),
            "end_equity": w.sim.end_equity,
        })
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "windows.csv", index=False)

    # Equity plot
    if eq is not None and len(eq):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(eq.index, eq.values)
            ax.set_title(f"Equity — {params.get('symbol')} {params.get('timeframe')}  "
                         f"RR={params.get('rr')}  return={summary['return_pct']:+.1f}%")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("time")
            ax.set_ylabel("equity ($)")
            fig.tight_layout()
            fig.savefig(out_dir / "equity.png", dpi=110)
            plt.close(fig)
        except Exception as e:
            print(f"[warn] could not draw equity plot: {e}")

    print(f"\n=== SUMMARY ===")
    for k, v in summary.items():
        if k != "params":
            print(f"  {k}: {v}")
    print(f"  output: {out_dir}")

    return summary
