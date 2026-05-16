"""
Strategy 005 — EURUSD 1h, Random-Forest TP-classifier entries.

Two RF classifiers predict whether price reaches a 2*ATR target in the next 10 bars:
  - model_long  : P(High over bars i+1..i+10 >= Close[i] + 2*ATR[i])
  - model_short : P(Low  over bars i+1..i+10 <= Close[i] - 2*ATR[i])

Entry gates (one open position max, evaluated at each bar's close):
  LONG  if P_long  >= 0.60  AND  P_short < 0.50
  SHORT if P_short >= 0.60  AND  P_long  < 0.50
(EMA50(HL/2) trend filter was tried and removed — too restrictive given low gate-hit rate.)

Exits (first to fire, intra-bar TP/SL using High/Low):
  TP   at Entry ± 2*ATR(entry bar)
  SL   at Entry ∓ 2*ATR(entry bar)           ← 1:1 R:R; not specified by user, default
  Time after 10 bars                         ← matches label horizon

Same-bar TP/SL conflict: SL wins (worst-case fill).
Time exit happens at the bar's Close.

Train: 2025-01-01 .. 2025-12-31  (~12 months, ~6500 bars)
Test : 2026-01-01 .. today        (~4.5 months)

Spread: 1 pip.  ATR length 14.  EMA50 on (High+Low)/2.
"""
import sys

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


_MT5_TF_MAP = {"1h": 16385}


def get_data(ticker, timeframe, start, end):
    ts_start = pd.Timestamp(start, tz="UTC")
    ts_end   = pd.Timestamp(end,   tz="UTC")
    mt5.initialize()
    mt5.symbol_select(ticker, True)

    tf_const = _MT5_TF_MAP[timeframe]
    CHUNK    = 50_000

    frames, pos = [], 0
    while True:
        chunk = mt5.copy_rates_from_pos(ticker, tf_const, pos, CHUNK)
        if chunk is None or len(chunk) == 0:
            break
        frames.append(pd.DataFrame(chunk))
        oldest = pd.Timestamp(int(frames[-1]["time"].min()), unit="s", tz="UTC")
        if oldest <= ts_start or len(chunk) < CHUNK:
            break
        pos += CHUNK

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames[::-1])
    df = df.drop_duplicates(subset="time").sort_values("time")
    df.index = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[(df.index >= ts_start) & (df.index < ts_end)]
    df = df.rename(columns={"open": "Open", "high": "High",
                            "low": "Low", "close": "Close", "tick_volume": "Volume"})
    return df[["Open", "High", "Low", "Close", "Volume"]]


PIP_SIZE    = 0.0001
SPREAD_PIPS = 1.0
ATR_LEN     = 14
EMA50_LEN   = 50
HORIZON     = 10
ATR_MULT    = 2.0
P_HI        = 0.60   # entry threshold for "own-side" model
P_LO        = 0.50   # cap for "opposite-side" model
RF_PARAMS   = dict(n_estimators=300, max_depth=8, min_samples_leaf=30,
                   class_weight="balanced", random_state=42, n_jobs=-1)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]

    df["atr"]   = ta.atr(h, l, c, length=ATR_LEN)
    df["ema20"] = ta.ema(c, length=20)
    df["ema50_hl2"] = ta.ema((h + l) / 2, length=EMA50_LEN)

    df["ret_1"]  = c.pct_change(1)
    df["ret_3"]  = c.pct_change(3)
    df["ret_5"]  = c.pct_change(5)
    df["ret_10"] = c.pct_change(10)

    df["rsi14"]  = ta.rsi(c, length=14)

    macd = ta.macd(c, fast=12, slow=26, signal=9)
    df["macd_n"]  = macd["MACD_12_26_9"]  / df["atr"]
    df["macds_n"] = macd["MACDs_12_26_9"] / df["atr"]
    df["macdh_n"] = macd["MACDh_12_26_9"] / df["atr"]

    df["dist_ema20_atr"] = (c - df["ema20"]) / df["atr"]
    df["dist_ema50_atr"] = (c - df["ema50_hl2"]) / df["atr"]

    bb = ta.bbands(c, length=20, std=2)
    bbu = next(col for col in bb.columns if col.startswith("BBU"))
    bbl = next(col for col in bb.columns if col.startswith("BBL"))
    df["bb_pct"] = (c - bb[bbl]) / (bb[bbu] - bb[bbl])

    hours = df.index.hour
    df["hr_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hr_cos"] = np.cos(2 * np.pi * hours / 24)

    rng = (h - l).replace(0, np.nan)
    df["body_ratio"]  = (c - o).abs() / rng
    df["upper_wick"]  = (h - np.maximum(o, c)) / rng
    df["lower_wick"]  = (np.minimum(o, c) - l) / rng

    adx = ta.adx(h, l, c, length=14)
    df["adx14"] = adx["ADX_14"]

    return df


FEATURE_COLS = [
    "ret_1", "ret_3", "ret_5", "ret_10",
    "rsi14",
    "macd_n", "macds_n", "macdh_n",
    "dist_ema20_atr", "dist_ema50_atr",
    "bb_pct",
    "hr_sin", "hr_cos",
    "body_ratio", "upper_wick", "lower_wick",
    "adx14",
]


def make_labels(df: pd.DataFrame) -> pd.DataFrame:
    """label_long[i]  = 1 if any High in (i+1, i+HORIZON) >= Close[i] + ATR_MULT*ATR[i]
       label_short[i] = 1 if any Low  in (i+1, i+HORIZON) <= Close[i] - ATR_MULT*ATR[i]
    Implemented via rolling future max/min on shifted arrays."""
    high = df["High"].values
    low  = df["Low"].values
    close = df["Close"].values
    atr   = df["atr"].values
    n     = len(df)

    fut_max_high = np.full(n, np.nan)
    fut_min_low  = np.full(n, np.nan)
    for i in range(n - HORIZON):
        sl = slice(i + 1, i + 1 + HORIZON)
        fut_max_high[i] = high[sl].max()
        fut_min_low[i]  = low[sl].min()

    tp_long_px  = close + ATR_MULT * atr
    tp_short_px = close - ATR_MULT * atr
    df["label_long"]  = (fut_max_high >= tp_long_px).astype(int)
    df["label_short"] = (fut_min_low  <= tp_short_px).astype(int)
    # Last HORIZON rows are unlabeled — mark as NaN by setting label to -1 there
    df.iloc[-HORIZON:, df.columns.get_indexer(["label_long", "label_short"])] = -1
    return df


def report_classifier(name, model, X, y):
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    base  = y.mean()
    print(f"  {name}:  base_rate={base:.3f}  "
          f"acc={accuracy_score(y, pred):.3f}  "
          f"prec={precision_score(y, pred, zero_division=0):.3f}  "
          f"rec={recall_score(y, pred, zero_division=0):.3f}  "
          f"auc={roc_auc_score(y, proba):.3f}")


def backtest(df: pd.DataFrame, p_long: np.ndarray, p_short: np.ndarray) -> dict:
    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    atr = df["atr"].values
    ema = df["ema50_hl2"].values
    n   = len(df)

    long_gate  = (p_long  >= P_HI) & (p_short < P_LO) & np.isfinite(atr)
    short_gate = (p_short >= P_HI) & (p_long  < P_LO) & np.isfinite(atr)

    trades = []
    n_open = 0
    open_mtm = 0.0
    i = 0
    while i < n - 1:
        if not (long_gate[i] or short_gate[i]):
            i += 1
            continue
        direction = +1 if long_gate[i] else -1
        entry_px  = c[i]
        atr_e     = atr[i]
        tp_px     = entry_px + direction * ATR_MULT * atr_e
        sl_px     = entry_px - direction * ATR_MULT * atr_e
        entry_i   = i
        deadline  = min(i + HORIZON, n - 1)

        exit_i, exit_px, kind = None, None, None
        for j in range(i + 1, deadline + 1):
            if direction == 1:
                hit_sl = l[j] <= sl_px
                hit_tp = h[j] >= tp_px
            else:
                hit_sl = h[j] >= sl_px
                hit_tp = l[j] <= tp_px
            if hit_sl and hit_tp:
                exit_i, exit_px, kind = j, sl_px, "sl"
                break
            if hit_sl:
                exit_i, exit_px, kind = j, sl_px, "sl"
                break
            if hit_tp:
                exit_i, exit_px, kind = j, tp_px, "tp"
                break
        if exit_i is None:
            exit_i, exit_px, kind = deadline, c[deadline], "time"

        pips_gross = (exit_px - entry_px) * direction / PIP_SIZE
        trades.append({"entry_i": entry_i, "exit_i": exit_i, "dir": direction,
                       "pips": float(pips_gross), "hold": exit_i - entry_i,
                       "exit": kind, "atr_pips": float(atr_e / PIP_SIZE)})
        i = exit_i + 1

    return {"trades": trades, "n_open": n_open, "open_mtm": open_mtm}


def summarize(trades, df, label):
    if not trades:
        print(f"\n{label}: no trades")
        return
    pips_gross = np.array([t["pips"] for t in trades])
    pips_net   = pips_gross - SPREAD_PIPS
    wins       = pips_net > 0
    days       = max(1, (df.index[-1] - df.index[0]).days)
    sharpe     = (pips_net.mean() / pips_net.std()
                  * np.sqrt(len(trades) / (days / 365.25))
                  if len(trades) > 1 and pips_net.std() > 0 else 0.0)
    kinds = [t["exit"] for t in trades]
    n_tp   = sum(k == "tp"   for k in kinds)
    n_sl   = sum(k == "sl"   for k in kinds)
    n_time = sum(k == "time" for k in kinds)
    longs  = sum(1 for t in trades if t["dir"] == 1)
    shorts = len(trades) - longs
    print(f"\n{label}")
    print(f"  trades={len(trades)}  long={longs}  short={shorts}")
    print(f"  net_pips={pips_net.sum():+.1f}  win%={wins.mean()*100:.1f}  "
          f"sharpe={sharpe:+.2f}")
    print(f"  avg_win={pips_net[wins].mean() if wins.any() else 0:+.1f}  "
          f"avg_loss={pips_net[~wins].mean() if (~wins).any() else 0:+.1f}  "
          f"avg_hold={np.mean([t['hold'] for t in trades]):.1f}")
    print(f"  exits: TP={n_tp}  SL={n_sl}  TIME={n_time}")


# ----- run -----
TICKER = "EURUSD"
START  = "2025-01-01"
END    = "2026-05-15"
SPLIT  = "2026-01-01"   # train < SPLIT, test >= SPLIT

print(f"Loading {TICKER} 1h {START}..{END}")
df = get_data(TICKER, "1h", START, END)
if df.empty:
    raise SystemExit(f"No data for {TICKER}")
print(f"  {len(df)} bars")

df = add_features(df)
df = make_labels(df)

# Drop warm-up rows (NaN features) and unlabeled tail rows.
df_lab = df.dropna(subset=FEATURE_COLS + ["atr", "ema50_hl2"]).copy()
df_lab = df_lab[(df_lab["label_long"] != -1) & (df_lab["label_short"] != -1)]
print(f"  labeled rows: {len(df_lab)}")

train = df_lab[df_lab.index < SPLIT]
test  = df_lab[df_lab.index >= SPLIT]
print(f"  train: {len(train)} rows {train.index.min()} .. {train.index.max()}")
print(f"  test : {len(test)}  rows {test.index.min()} .. {test.index.max()}")

X_train = train[FEATURE_COLS].values
X_test  = test[FEATURE_COLS].values
y_train_long  = train["label_long"].values
y_train_short = train["label_short"].values
y_test_long   = test["label_long"].values
y_test_short  = test["label_short"].values

print(f"\nLabel base rates (train):")
print(f"  P(2ATR-long  hit in {HORIZON} bars) = {y_train_long.mean():.3f}")
print(f"  P(2ATR-short hit in {HORIZON} bars) = {y_train_short.mean():.3f}")

print("\nTraining model_long ...")
m_long = RandomForestClassifier(**RF_PARAMS).fit(X_train, y_train_long)
print("Training model_short ...")
m_short = RandomForestClassifier(**RF_PARAMS).fit(X_train, y_train_short)

print("\nClassifier metrics on TEST set:")
report_classifier("model_long ", m_long,  X_test, y_test_long)
report_classifier("model_short", m_short, X_test, y_test_short)

print("\nTop feature importances:")
imp_long = sorted(zip(FEATURE_COLS, m_long.feature_importances_),
                  key=lambda x: -x[1])
imp_short = sorted(zip(FEATURE_COLS, m_short.feature_importances_),
                   key=lambda x: -x[1])
print(f"  long :  " + ", ".join(f"{n}={v:.2f}" for n, v in imp_long[:6]))
print(f"  short:  " + ", ".join(f"{n}={v:.2f}" for n, v in imp_short[:6]))

# Score the test set and run backtest on it.
p_long_test  = m_long.predict_proba(X_test)[:, 1]
p_short_test = m_short.predict_proba(X_test)[:, 1]

n_gate_long  = ((p_long_test  >= P_HI) & (p_short_test < P_LO)).sum()
n_gate_short = ((p_short_test >= P_HI) & (p_long_test  < P_LO)).sum()
print(f"\nGate hits on test (before EMA filter):")
print(f"  long  gate (Plong>={P_HI}, Pshort<{P_LO}): {n_gate_long}")
print(f"  short gate (Pshort>={P_HI}, Plong<{P_LO}): {n_gate_short}")

res = backtest(test, p_long_test, p_short_test)
summarize(res["trades"], test, f"Backtest TEST  {test.index.min().date()} .. "
                                f"{test.index.max().date()}")

# Sanity: also score+backtest on TRAIN to see degree of overfit.
p_long_train  = m_long.predict_proba(X_train)[:, 1]
p_short_train = m_short.predict_proba(X_train)[:, 1]
res_tr = backtest(train, p_long_train, p_short_train)
summarize(res_tr["trades"], train, f"Backtest TRAIN (in-sample, for overfit check)")
