import sys

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import pandas_ta

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


_MT5_TF_MAP = {
    "1m":  1,       # TIMEFRAME_M1
    "5m":  5,       # TIMEFRAME_M5
    "15m": 15,      # TIMEFRAME_M15
    "30m": 30,      # TIMEFRAME_M30
    "1h":  16385,   # TIMEFRAME_H1
    "4h":  16388,   # TIMEFRAME_H4
    "1d":  16408,   # TIMEFRAME_D1
}


def get_data(ticker="EURUSD", timeframe="5m", start="2026-01-01", end="2026-05-01"):
    ts_start = pd.Timestamp(start, tz="UTC")
    ts_end = pd.Timestamp(end, tz="UTC")
    mt5.initialize()

    tf_const  = _MT5_TF_MAP.get(timeframe, 1)
    mt5.symbol_select(ticker, True)
    CHUNK = 50_000

    frames = []
    pos = 0
    while True:
        chunk = mt5.copy_rates_from_pos(ticker, tf_const, pos, CHUNK)
        if chunk is None or len(chunk) < CHUNK:
            break
        frames.append(pd.DataFrame(chunk))
        oldest_broker_ts = int(frames[-1]["time"].min())
        oldest_utc = pd.Timestamp(oldest_broker_ts, unit="s", tz="UTC")
        if oldest_utc <= ts_start:
            break  # reached far enough back
        pos += CHUNK

    df = pd.concat(frames[::-1])  # oldest first
    df = df.drop_duplicates(subset="time").sort_values("time")
    df.index = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[(df.index >= ts_start) & (df.index < ts_end)]
    df = df.rename(columns={
        "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "tick_volume": "Volume",
    })
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    return df


PIP_SIZE    = 0.0001
SPREAD_PIPS = 1.0
LOOKBACK_U  = 5


def add_dema(df: pd.DataFrame) -> pd.DataFrame:
    df.ta.hl2(append=True)
    df.ta.dema(close="HL2", length=20, append=True, col_names=("dema20",))
    return df


def compute_signals(df: pd.DataFrame, strict_monotone: bool = False,
                    lookback_u: int = LOOKBACK_U):
    dema = df["dema20"].values
    high = df["High"].values
    low  = df["Low"].values
    n    = len(df)

    u_long  = np.zeros(n, dtype=bool)
    u_short = np.zeros(n, dtype=bool)

    if strict_monotone:
        for i in range(5, n):
            v = dema[i-5:i+1]
            if not np.all(np.isfinite(v)):
                continue
            diffs = np.diff(v)
            if np.all(diffs > 0):
                u_long[i] = True
            elif np.all(diffs < 0):
                u_short[i] = True
    else:
        for i in range(5, n):
            a, b, c = dema[i], dema[i-2], dema[i-5]
            if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)):
                continue
            if a > b > c:
                u_long[i] = True
            elif a < b < c:
                u_short[i] = True

    touch = np.isfinite(dema) & (low <= dema) & (dema <= high)
    long_rec  = pd.Series(u_long.astype(np.int8) ).rolling(lookback_u, min_periods=1).max().astype(bool).values
    short_rec = pd.Series(u_short.astype(np.int8)).rolling(lookback_u, min_periods=1).max().astype(bool).values
    entry_long  = touch & long_rec
    entry_short = touch & short_rec
    return u_long, u_short, touch, entry_long, entry_short


def backtest(df: pd.DataFrame, direction: int,
             entry_sig: np.ndarray, u_long: np.ndarray, u_short: np.ndarray,
             touch: np.ndarray, symmetric_exit: bool = False,
             sl_pips: float = 0.0, tp_pips: float = 0.0) -> dict:
    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values
    n     = len(df)
    opp_u = u_short if direction == 1 else u_long
    use_signal_exit = (sl_pips == 0 and tp_pips == 0)

    pos      = False
    entry_px = 0.0
    entry_i  = 0
    trades: list[dict] = []

    for i in range(n):
        if pos:
            if sl_pips > 0:
                sl_px = entry_px - direction * sl_pips * PIP_SIZE
                if (direction == 1 and low[i]  <= sl_px) or \
                   (direction == -1 and high[i] >= sl_px):
                    pips = (sl_px - entry_px) * direction / PIP_SIZE
                    trades.append({"pips": pips, "hold": i - entry_i, "exit": "sl"})
                    pos = False
                    continue
            if tp_pips > 0:
                tp_px = entry_px + direction * tp_pips * PIP_SIZE
                if (direction == 1 and high[i] >= tp_px) or \
                   (direction == -1 and low[i]  <= tp_px):
                    pips = (tp_px - entry_px) * direction / PIP_SIZE
                    trades.append({"pips": pips, "hold": i - entry_i, "exit": "tp"})
                    pos = False
                    continue
            if use_signal_exit:
                flip = opp_u[i] and (touch[i] if symmetric_exit else True)
                if flip:
                    pips = (close[i] - entry_px) * direction / PIP_SIZE
                    trades.append({"pips": pips, "hold": i - entry_i, "exit": "flip"})
                    pos = False
                    continue
        if not pos and entry_sig[i]:
            pos      = True
            entry_px = close[i]
            entry_i  = i

    if not trades:
        return {"n": 0}
    pips_gross = np.array([t["pips"] for t in trades])
    pips_net   = pips_gross - SPREAD_PIPS
    cum        = np.cumsum(pips_net)
    max_dd_p   = float((cum - np.maximum.accumulate(cum)).min())
    wins       = pips_net > 0
    days       = max(1, (df.index[-1] - df.index[0]).days)
    sharpe     = (pips_net.mean() / pips_net.std() * np.sqrt(len(trades) / (days / 365.25))
                  if len(trades) > 1 and pips_net.std() > 0 else 0.0)
    return {
        "n":         len(trades),
        "pips_net":  float(pips_net.sum()),
        "win_pct":   float(wins.mean() * 100),
        "avg_win":   float(pips_net[wins].mean())  if wins.any()  else 0.0,
        "avg_loss":  float(pips_net[~wins].mean()) if (~wins).any() else 0.0,
        "max_dd_p":  max_dd_p,
        "avg_hold":  float(np.mean([t["hold"] for t in trades])),
        "sharpe":    float(sharpe),
    }


print("Loading 5m data...")
df_5m = add_dema(get_data(timeframe="5m"))
print(f"  {len(df_5m)} bars")
print("Loading 1h data...")
df_1h = add_dema(get_data(timeframe="1h"))
print(f"  {len(df_1h)} bars")

# (df, strict_monotone, symmetric_exit, sl_pips, tp_pips)
variants = [
    ("Baseline   5m  flip-exit       ", df_5m, False, False,  0,  0),
    ("A: sym-exit   5m                ", df_5m, False, True,   0,  0),
    ("B: 1h         flip-exit         ", df_1h, False, False,  0,  0),
    ("C: 5m  SL10 / TP20              ", df_5m, False, False, 10, 20),
    ("D: 5m  strict-monotone flip     ", df_5m, True,  False,  0,  0),
    ("ALL: 1h strict + sym + SL30/TP60", df_1h, True,  True,  30, 60),
]

hdr = f"{'Variant':<34} {'Dir':<5} {'Trades':>7} {'Net p':>9} {'Sharpe':>7} {'Win%':>6} {'AvgW':>6} {'AvgL':>6} {'Hold':>6}"
print(f"\n{hdr}")
print("-" * len(hdr))

for name, df_v, strict, sym, sl, tp in variants:
    u_l, u_s, tch, e_l, e_s = compute_signals(df_v, strict_monotone=strict)
    for direction, label, e_sig in [(+1, "LONG", e_l), (-1, "SHORT", e_s)]:
        m = backtest(df_v, direction, e_sig, u_l, u_s, tch,
                     symmetric_exit=sym, sl_pips=sl, tp_pips=tp)
        if m.get("n", 0) == 0:
            print(f"{name:<34} {label:<5} {'-':>7}")
            continue
        print(f"{name:<34} {label:<5} {m['n']:>7} {m['pips_net']:>+9.1f} "
              f"{m['sharpe']:>+7.2f} {m['win_pct']:>5.1f}% "
              f"{m['avg_win']:>+6.1f} {m['avg_loss']:>+6.1f} {m['avg_hold']:>6.1f}")

