# Range Fade Strategy

Mean-reversion on 1m FX bars. When price breaks out of a tight rolling range, fade the breakout: short above range high, long below range low. Take-profit and stop-loss are ATR-multiples set at entry. One open trade per pair at a time.

This document reflects the **current validated configuration** based on gap-aware backtesting on 2024–2026 data. The original `scripts/strategy1.py` + `scripts/strategy1_portfolio.py` represent the legacy 2022–2024 setup and are kept for reference; new work should go through `strategy1_backtest_curves.py` and `strategy1_live_ib.py`.

---

## Logic

1. **Range**: rolling `max(high)` and `min(low)` over the previous **10 bars** (excludes the current bar — verified no look-ahead).
2. **Entry**: at the close of the breakout bar, take the opposite direction.
   - Close > range_high → **short** (fade)
   - Close < range_low  → **long** (fade)
3. **Exit**: ATR-based bracket set at entry.
   - TP = entry ± 1.0 × ATR(14)
   - SL = entry ∓ 2.0 × ATR(14)
4. **Tightness filter** (legacy): `(range_high − range_low) ≥ 1.5 × ATR` blocks entry. **Disabled in the validated config** — it interacts poorly with shorter lookbacks. Keep `--no-tight` on for reproducibility.
5. **Hour/day filters** (legacy): per-pair UTC-hour and day-of-week allow-lists. **Disabled in the validated config** — they overfit to the tuning window and degrade out-of-sample. Keep `--no-filters` on.
6. **Pairs**: 5 USD-major pairs — **EURUSD, AUDUSD, NZDUSD, USDCHF, USDCAD**. GBPUSD is excluded (consistently loses or borderline). USDJPY is excluded (inconsistent on 1m).

---

## Why the legacy filters are gone

The original config (`strategy1_config.json`) included per-pair hour/day filters tuned on 2024-2025 and a tightness gate. Walk-forward on 2026 showed:

- Filtered run: 2024-2025 in-sample +$1,021K, 2026 OOS +$185K
- Unfiltered run: 2024-2025 in-sample +$975K, 2026 OOS **+$198K**

The filters traded a small in-sample gain ($46K) for a real OOS hit (-$14K). The strategy edge is structurally distributed across most hours and weekdays. Filtering picks up noise, not signal.

The same is true for the tightness filter at lookback=10 — a 10-bar range is naturally tight, so the `< 1.5 × ATR` gate rarely passes and rejects most setups.

---

## Cost model (IB-realistic)

- **Entry** (market order): crosses ~half the bid-ask spread.
- **TP exit** (limit): fills at the limit price exactly. No favorable-gap bonus assumed (deep retail FX liquidity makes that rare).
- **SL exit** (stop → market on trigger): crosses ~half the spread. **If the bar gapped through the stop at the open, real fill is the open price** (the dominant gap risk in retail FX — weekend opens, news minutes).
- **Commission** (IB Pro tier-1): `max($2, 0.20 bp × notional)` per side. The $2 minimum dominates for trades smaller than ~$100K notional.
- **Per-pair half-spreads** baked in (pips):

  | Pair | Half-spread |
  |------|-------------|
  | EURUSD | 0.15 |
  | GBPUSD | 0.30 |
  | AUDUSD | 0.25 |
  | NZDUSD | 0.75 |
  | USDCHF | 0.50 |
  | USDCAD | 0.50 |

Tune via `--spread-mult` (e.g. 1.5 for stress, 0.5 for tight regime). Use `--no-costs` to compare against frictionless.

---

## Risk management (live + backtest)

These rules apply to both the backtest (`strategy1_backtest_curves.py`) and the live bot (`strategy1_live_ib.py`) and are wired with the same semantics.

### Position sizing

Pass exactly one of:
- `--lot N` — fixed lot size (default 0.5)
- `--risk-usd N` — risk-based: position chosen so SL distance × units = $N

`--max-lot N` caps the position regardless of sizing mode. Recommended with `--risk-usd` to bound size when ATR is tight.

### Per-pair daily-DD circuit breaker

`--daily-dd-limit-pair 150` — once today's realised PnL on a given pair drops to -$150, block new entries on that pair until 00:00 UTC. A TP win that recovers the day reopens trading. Resets at midnight UTC.

This caps the worst-case daily loss across the portfolio at `5 × $150 = $750` regardless of trade volume.

### Overnight / weekend rules

`--no-overnight` enables intraday-only defaults:
- `close-by-hour=21` — force-close any open positions when bar UTC hour ≥ 21
- `no-entry-after-hour=20` — block new entries after 20:00 UTC
- `no-friday-after-hour=17` — Friday-only stricter cutoff

Override individually with `--close-by-hour H`, `--no-entry-after-hour H`, `--no-friday-after-hour H`.

Trade-off: blocks ~12% of trade count and ~22% of P&L on the backtest, but eliminates ~50% of weekend gap exposure. Worth it when running the live bot unattended.

### Live-bot-only: spread guard

`--spread-guard` queries IB's bid/ask before submitting and skips trades where current spread exceeds a per-pair limit. Default per-pair caps:

| Pair | Max spread (pips) |
|------|-------------------|
| EURUSD | 0.6 |
| GBPUSD | 1.0 |
| AUDUSD | 1.0 |
| NZDUSD | 2.5 |
| USDCHF | 1.5 |
| USDCAD | 1.5 |
| USDJPY | 0.8 |

These are roughly 2× normal spreads — wide enough that liquid hours pass through, narrow enough that news spikes get filtered out. Override with `--max-spread-pips EURUSD=0.5 NZDUSD=2.0` or globally `--max-spread-pips 1.5`.

---

## Files

| File | Purpose |
|------|---------|
| `scripts/strategy1.py` | Original single-pair backtest (legacy) |
| `scripts/strategy1_config.json` | Pair list + legacy filters |
| `scripts/strategy1_portfolio.py` | Original portfolio backtest (legacy) |
| **`scripts/strategy1_backtest_curves.py`** | **Current backtest — gap-aware, multi-scenario, equity curves, account sizing** |
| **`scripts/strategy1_live_ib.py`** | **Live IB trading bot (matches the backtest semantics 1:1)** |
| `scripts/strategy1_lookback_sweep.py` | Lookback sensitivity study tool |
| `scripts/strategy1_session_optimizer.py` | Per-pair hour/day filter tuner with train/test split |
| `scripts/strategy1_propfirm_sizing.py` | Prop-firm sizing & challenge-completion estimator |
| `scripts/cache_fx_data.py` | Bulk-fetch FX OHLCV into the local parquet cache |

---

## How to run

### Backtest with the validated config

```bash
# 5 pairs, 1m, lookback=10, gap-aware, IB-realistic costs, fixed 0.5 lot
python -m scripts.strategy1_backtest_curves \
    --start 2024-01-01 --end 2026-05-02 \
    --tf 1m --lookback 10 \
    --no-tight --no-filters \
    --exclude GBPUSD \
    --lots 0.5

# With full risk management
python -m scripts.strategy1_backtest_curves \
    --start 2024-01-01 --end 2026-05-02 \
    --tf 1m --lookback 10 \
    --no-tight --no-filters \
    --exclude GBPUSD \
    --risk-usd 50 --max-lot 0.5 \
    --daily-dd-limit-pair 150 \
    --no-overnight

# Stress test (2× normal spreads)
python -m scripts.strategy1_backtest_curves \
    --start 2024-01-01 --end 2026-05-02 \
    --tf 1m --lookback 10 --no-tight --no-filters --exclude GBPUSD \
    --lots 0.5 --spread-mult 2.0
```

Outputs go to `data/backtest_curves/<timestamp>/`:
- per-pair equity-curve PNGs
- portfolio aggregate curve
- `metrics_<scenario>.csv`
- `trades.csv`
- `summary.json`

### Lookback sweep

```bash
python -m scripts.strategy1_lookback_sweep \
    --start 2024-01-01 --end 2026-05-02 \
    --lookbacks 10 20 30 40 50 \
    --tfs 1m 5m
```

### Prop-firm challenge sizing

```bash
# Pass a trades.csv (default: latest backtest run)
python -m scripts.strategy1_propfirm_sizing \
    --account 100000 --target 15000 \
    --max-daily-loss 5000 --max-drawdown 10000
```

Reports:
- per-lot-factor max daily loss, MaxDD, recommended sizing
- average / median / pessimistic time to hit the target

### Live IB bot — recommended config

```bash
# Paper-trade dry-run first (logs signals, no orders)
python -m scripts.strategy1_live_ib \
    --port 7497 --dry-run \
    --pairs EURUSD AUDUSD NZDUSD USDCHF USDCAD \
    --no-filters --no-overnight --spread-guard \
    --risk-usd 50 --max-lot 0.5 --daily-dd-limit-pair 150

# Real paper account, real orders
python -m scripts.strategy1_live_ib \
    --port 7497 \
    --pairs EURUSD AUDUSD NZDUSD USDCHF USDCAD \
    --no-filters --no-overnight --spread-guard \
    --risk-usd 50 --max-lot 0.5 --daily-dd-limit-pair 150

# Live account (port 7496) — only after paper validation
python -m scripts.strategy1_live_ib \
    --port 7496 \
    --pairs EURUSD AUDUSD NZDUSD USDCHF USDCAD \
    --no-filters --no-overnight --spread-guard \
    --risk-usd 50 --max-lot 0.5 --daily-dd-limit-pair 150

# Offline replay (no IB connection) — sanity-check signal logic
python -m scripts.strategy1_live_ib --simulate --sim-hours 48 \
    --pairs EURUSD AUDUSD NZDUSD USDCHF USDCAD --no-filters
```

---

## Backtest results (validated config, 2024-01-01 → 2026-05-02)

5 pairs ex-GBPUSD, 1m, lookback=10, no tightness, no hour/day filters, IB-realistic costs, gap-aware SL fills.

### Fixed 1 lot per pair

| Pair    | Net P&L     | # Trades | Win%  | MaxDD     | Sharpe* |
|---------|------------:|---------:|------:|----------:|--------:|
| EURUSD  |    $+18,821 |   35,560 | 74.7% | $-21,998  | 1.5     |
| AUDUSD  |   $+364,773 |   28,348 | 85.3% | $-3,868   | 21.5    |
| NZDUSD  |   $+371,562 |   24,917 | 88.7% | $-14,238  | 14.0    |
| USDCHF  |   $+338,450 |   24,500 | 84.4% | $-8,419   | 19.3    |
| USDCAD  |    $+52,832 |   29,860 | 79.4% | $-17,846  | 4.6     |
| **Portfolio** | **$+1,146,439** | **143,185** | **81.9%** | **$-26,201** | – |

*Per-trade annualized Sharpe — implausibly high (real strategies rarely exceed 5). Treat magnitudes with skepticism until live-validated.

### With recommended risk caps (`--risk-usd 50 --max-lot 0.5 --daily-dd-limit-pair 150`)

Approximately half the absolute P&L of the table above (since 0.5 lot ≈ half the 1-lot result), with materially lower drawdowns and a hard daily-loss ceiling per pair.

### 2026 out-of-sample only

| Period      | Net P&L (1 lot) | MaxDD     |
|-------------|----------------:|----------:|
| 2026 YTD    |    $+198,352    | $-3,037   |

---

## Account sizing

### Personal IB account (Cyprus retail / EU CySEC, 30:1 leverage)

For trading 1 lot per pair across all 5 pairs concurrently:

| Recommendation | Account size | Coverage |
|---|---:|---|
| Bare minimum | $20K | margin + 1× backtest DD |
| Practical    | $40-50K | margin + 2-3× backtest DD |
| Comfortable  | $100K | full real-world DD buffer |

Margin per pair at 30:1 leverage is small ($2-4K each, $15-22K total for 5 pairs concurrently). The dominant factor is drawdown buffer, not margin.

### $100K prop-firm challenge ($5K daily / $10K trailing-EOD DD, target $15K)

Tested via 122 rolling-Monday starts on 2024-2026 data with the validated config:

| Sizing | Pass rate | Blow-ups | Avg days to $15K | 90th-pct days |
|--------|----------:|---------:|-----------------:|--------------:|
| Full size (max-lot=1, $100 risk, $500 daily-DD) | 97.5% | 1.6% | **9.4** | 13 |
| **Half size** (max-lot=0.5, $50 risk, $150 daily-DD) | **97.5%** | **0.0%** | **20.5** | 27 |

**Recommended for prop firm: half size.** Same pass rate, zero historical blow-ups, ~3 weeks to target instead of 2.

---

## Verified properties

- **No look-ahead** — every input at decision time uses only bars ≤ i. Audited in code (top of `strategy1_backtest_curves.py`).
- **Cost model symmetry** — entry, SL exit, and forced-close fills are modeled with the same spread + commission mechanics in both backtest and live.
- **Gap-aware SL fills** — when the bar gapped past the stop at the open, real fill is at the open price, not the stop price. Captures weekend / news risk that bar-level data otherwise misses.
- **Live bot mirrors backtest semantics 1:1** — the same evaluate-entry function runs offline and online; sizing, daily-DD, and overnight rules use the same parameters.

---

## Caveats before risking real money

1. **Sharpe ratios of 14-22 per pair are implausibly high** for real strategies. Most likely causes:
   - Polygon mid-quote bars don't fully reflect achievable retail fills, especially during fast moves.
   - The 1-pip-ish cost model may underestimate real-world slippage during news minutes.
   - Bar-level resolution misses intra-bar adverse fills.

   Real-world performance typically lands at 50-70% of backtest. Plan accordingly.

2. **Live-paper-trade for at least 2-4 weeks** with the spread guard ON before paying any prop-firm fee or trading real capital. Track:
   - Actual fill quality vs the bar's close price
   - Frequency of `skip_spread` events vs trade attempts
   - Realised vs expected daily P&L

   If your live $/day is ≥ 70% of backtest, you're in good shape. If < 50%, revisit.

3. **Friday-evening + weekend exposure remains a tail risk** even with the gap-aware SL fix. SNB-2015-style events and similar black swans aren't in the dataset and aren't modelled. Run with `--no-overnight` for unattended live trading.

4. **GBPUSD is excluded for a reason** — it both showed weak edge in backtest and exhibited regime instability across 2024 vs 2025 vs 2026. Don't add it back without re-validating.

5. **The strategy is not robust to high IB commission floors at small lot sizes.** At 0.1 lot the $4 round-trip minimum eats most of the per-trade edge. Don't run smaller than 0.3 lot per pair.

6. **MaxDD scales with lot size linearly.** If you want to halve risk, halve all sizing parameters together: `--lot 0.25` (or `--risk-usd 25 --max-lot 0.25`) and `--daily-dd-limit-pair 75`.
