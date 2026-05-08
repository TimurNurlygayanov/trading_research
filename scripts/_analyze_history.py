import pandas as pd, numpy as np

df = pd.read_excel(r'C:\Users\Тимур\Desktop\ReportHistory-2.xlsx', header=None)
cols = ['open_time','ticket','symbol','type','volume','open_price','sl','tp',
        'close_time','close_price','commission','swap','profit']
PAIRS = {'EURUSD','AUDUSD','NZDUSD','USDCHF','USDCAD','GBPUSD','USDJPY'}
trade_rows = [i for i, row in df.iterrows()
              if isinstance(row[2], str) and row[2] in PAIRS]
t = df.iloc[trade_rows][list(range(13))].copy()
t.columns = cols
for c in ['profit','volume','open_price','sl','tp','close_price','commission']:
    t[c] = pd.to_numeric(t[c], errors='coerce')
t['open_time']  = pd.to_datetime(t['open_time'],  format='mixed')
t['close_time'] = pd.to_datetime(t['close_time'], format='mixed')
t = t.dropna(subset=['profit','open_time','close_time'])

# Keep only rows where P&L is plausible for 1-2 lot FX (< $1000 per trade)
# and durations are positive and < 24h
t['dur_s'] = (t['close_time'] - t['open_time']).dt.total_seconds()
t = t[(t['profit'].abs() < 1000) & (t['dur_s'] > 0) & (t['dur_s'] < 86400)]

t['win']    = t['profit'] > 0
t['sl_dist'] = (t['open_price'] - t['sl']).abs()
t['tp_dist'] = (t['tp'] - t['open_price']).abs()
t['rr']     = t['tp_dist'] / t['sl_dist'].replace(0, np.nan)
t['hour']   = t['open_time'].dt.hour
t['minute'] = t['open_time'].dt.floor('min')

print(f'Trades (filtered): {len(t)}')
print(f'Range: {t["open_time"].min()}  to  {t["open_time"].max()}')

print('\n=== OVERALL ===')
print(f'Net P&L:    ${t["profit"].sum():.2f}')
print(f'Commission: ${t["commission"].sum():.2f}')
wr = t['win'].mean()*100
print(f'Win rate:   {wr:.1f}%  ({t["win"].sum()} wins / {(~t["win"]).sum()} losses)')
print(f'Avg win:    ${t.loc[t["win"],"profit"].mean():.2f}')
print(f'Avg loss:   ${t.loc[~t["win"],"profit"].mean():.2f}')
print(f'Avg R:R     {t["rr"].mean():.2f}  (TP/SL distance ratio)')
print(f'Avg dur:    {t["dur_s"].mean()/60:.1f} min  median={t["dur_s"].median()/60:.1f} min')
print(f'< 60s:      {(t["dur_s"]<60).sum()} trades ({(t["dur_s"]<60).mean()*100:.0f}%)')
print(f'< 120s:     {(t["dur_s"]<120).sum()} trades ({(t["dur_s"]<120).mean()*100:.0f}%)')
print(f'Lots:       {dict(t["volume"].value_counts())}')
print(f'Avg SL dist:{t["sl_dist"].mean()*10000:.1f} pips')
print(f'Avg TP dist:{t["tp_dist"].mean()*10000:.1f} pips')

# Outcome: SL hit vs TP hit (within 1 point tolerance)
t['sl_hit'] = (
    ((t['type']=='sell') & (t['close_price'] >= t['sl'] - 0.00003)) |
    ((t['type']=='buy')  & (t['close_price'] <= t['sl'] + 0.00003))
)
t['tp_hit'] = (
    ((t['type']=='sell') & (t['close_price'] <= t['tp'] + 0.00003)) |
    ((t['type']=='buy')  & (t['close_price'] >= t['tp'] - 0.00003))
)
print(f'SL exits:   {t["sl_hit"].sum()}  TP exits: {t["tp_hit"].sum()}  other: {(~t["sl_hit"]&~t["tp_hit"]).sum()}')

print('\n=== BY PAIR ===')
for sym, g in t.groupby('symbol'):
    wins   = g.loc[g['win'], 'profit']
    losses = g.loc[~g['win'], 'profit']
    aw = wins.mean() if len(wins) else 0
    al = losses.mean() if len(losses) else 0
    print(f'  {sym}: n={len(g):3d}  pnl=${g["profit"].sum():7.1f}  '
          f'win={g["win"].mean()*100:.0f}%  '
          f'avg_win=${aw:.1f}  avg_loss=${al:.1f}')

print('\n=== BY HOUR ===')
for hr, g in t.groupby('hour'):
    n = len(g)
    pnl = g['profit'].sum()
    wr  = g['win'].mean()*100
    print(f'  {hr:02d}h: n={n:3d}  pnl=${pnl:7.1f}  win={wr:.0f}%')

print('\n=== TRADE RATE vs BACKTEST EXPECTATION ===')
hours_traded = t['open_time'].max().hour - t['open_time'].min().hour + 1
print(f'Hours traded: ~{hours_traded}h  ({t["open_time"].min().strftime("%H:%M")} to {t["open_time"].max().strftime("%H:%M")})')
print(f'Live rate:    {len(t)/(hours_traded*60/1):.2f} trades/min  ({len(t)} in ~{hours_traded*60:.0f} min)')
pairs = len(t['symbol'].unique())
print(f'  Per pair:   {len(t)/(pairs*hours_traded*60):.4f} entries/bar  ({len(t)/pairs:.0f} trades per pair avg)')
print(f'Backtest 1m: ~{143185/(500*14*60):.4f} entries/bar  (143k trades / 500 days / 14h / 60 min)')

print('\n=== DOUBLE ENTRIES (same pair, same minute) ===')
per_min = t.groupby(['minute','symbol']).size().reset_index(name='n')
doubles = per_min[per_min['n'] > 1]
print(f'Pair-minute slots with >1 entry: {len(doubles)} (out of {len(per_min)} total pair-minute slots)')
print(f'Pair-minute slots with >2 entries: {(per_min["n"]>2).sum()}')
print('\nWorst double-entry minutes:')
worst = doubles.sort_values('n', ascending=False).head(12)
for _, row in worst.iterrows():
    print(f'  {row["minute"]}  {row["symbol"]}  x{row["n"]}')

print('\n=== RAPID RE-ENTRIES (same pair, < 2 min gap) ===')
t_s = t.sort_values('open_time')
for sym, g in t_s.groupby('symbol'):
    g = g.sort_values('open_time')
    fast = (g['open_time'].diff().dt.total_seconds() < 120).sum()
    if fast:
        print(f'  {sym}: {fast} re-entries within 2 min of prior trade')

print('\n=== SHORT-LIVED SL HITS (< 30s) ===')
fast_sl = t[(t['sl_hit']) & (t['dur_s'] < 30)]
print(f'Trades that hit SL in < 30s: {len(fast_sl)} '
      f'({len(fast_sl)/len(t)*100:.0f}% of all trades, '
      f'{len(fast_sl)/t["sl_hit"].sum()*100:.0f}% of SL hits)')
if len(fast_sl):
    print(f'  Avg SL dist of these: {fast_sl["sl_dist"].mean()*10000:.2f} pips')
    print(f'  Hours: {dict(fast_sl["hour"].value_counts().sort_index())}')

print('\n=== CUMULATIVE P&L OVER DAY ===')
t_s2 = t.sort_values('open_time').copy()
t_s2['cum_pnl'] = t_s2['profit'].cumsum()
checkpoints = [50, 100, 200, 300, 400, 500, 600, 700, 800, len(t_s2)-1]
for cp in checkpoints:
    if cp < len(t_s2):
        row = t_s2.iloc[cp]
        print(f'  After trade #{cp+1:4d} ({row["open_time"].strftime("%H:%M")}): cumPnL=${row["cum_pnl"]:.1f}')
