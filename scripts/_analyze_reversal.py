import pandas as pd
import numpy as np

df = pd.read_excel(r'C:\Users\Тимур\Desktop\ReportHistory-2.xlsx', header=None)
cols = ['open_time','ticket','symbol','type','volume','open_price','sl','tp',
        'close_time','close_price','commission','swap','profit']
PAIRS = {'EURUSD','AUDUSD','NZDUSD','USDCHF','USDCAD'}
trade_rows = [i for i,row in df.iterrows()
              if isinstance(row[2],str) and row[2] in PAIRS]
t = df.iloc[trade_rows][list(range(13))].copy()
t.columns = cols
for c in ['profit','volume','open_price','sl','tp','close_price','commission']:
    t[c] = pd.to_numeric(t[c], errors='coerce')
t['open_time']  = pd.to_datetime(t['open_time'],  format='mixed')
t['close_time'] = pd.to_datetime(t['close_time'], format='mixed')
t = t.dropna(subset=['profit','open_time','close_time'])
t['dur_s'] = (t['close_time'] - t['open_time']).dt.total_seconds()
t = t[(t['profit'].abs() < 1000) & (t['dur_s'] > 0) & (t['dur_s'] < 86400)]
t['win'] = t['profit'] > 0

# Simple exact reversal: every trade taken in opposite direction.
# The fill/SL/TP levels are mirrored — exact negation of each P&L.
neg_pnl      = -t['profit'].sum()
commission   = t['commission'].sum()
actual_total = t['profit'].sum() + commission

print("=== ACTUAL ===")
print(f"  Trades:      {len(t)}")
print(f"  Win rate:    {t['win'].mean()*100:.1f}%")
print(f"  Avg win:     ${t.loc[t['win'],'profit'].mean():.2f}")
print(f"  Avg loss:    ${t.loc[~t['win'],'profit'].mean():.2f}")
print(f"  Gross P&L:   ${t['profit'].sum():.2f}")
print(f"  Commission:  ${commission:.2f}")
print(f"  Net P&L:     ${actual_total:.2f}")

print()
print("=== REVERSED (breakout mode — opposite direction on same bars) ===")
rev_wr       = 1.0 - t['win'].mean()
avg_rev_win  = t.loc[~t['win'], 'profit'].abs().mean()   # SL hits become wins
avg_rev_loss = t.loc[t['win'],  'profit'].abs().mean()   # TP hits become losses
ev_rev       = rev_wr * avg_rev_win - (1 - rev_wr) * avg_rev_loss
print(f"  Win rate:    {rev_wr*100:.1f}%")
print(f"  Avg win:     ${avg_rev_win:.2f}  (=current avg_loss magnitude)")
print(f"  Avg loss:    ${avg_rev_loss:.2f}  (=current avg_win magnitude)")
print(f"  EV/trade:    ${ev_rev:.2f}")
print(f"  Gross P&L:   ${neg_pnl:.2f}")
print(f"  Commission:  ${commission:.2f}  (same cost)")
print(f"  Net P&L:     ${neg_pnl + commission:.2f}")

print()
print("=== WHY TODAY WAS A BAD FADE DAY ===")
t['sl_dist_pip'] = (t['open_price'] - t['sl']).abs() * 10000
t['tp_dist_pip'] = (t['tp'] - t['open_price']).abs() * 10000
# How often close was in same direction as the signal (trend confirming signal = bad for fade)
# Measure: did price move from entry toward TP before hitting SL?
sl_hit = (((t['type']=='sell') & (t['close_price'] >= t['sl'] - 0.00003)) |
          ((t['type']=='buy')  & (t['close_price'] <= t['sl'] + 0.00003)))
tp_hit = (((t['type']=='sell') & (t['close_price'] <= t['tp'] + 0.00003)) |
          ((t['type']=='buy')  & (t['close_price'] >= t['tp'] - 0.00003)))
print(f"  SL hits: {sl_hit.sum()} ({sl_hit.mean()*100:.1f}%)")
print(f"  TP hits: {tp_hit.sum()} ({tp_hit.mean()*100:.1f}%)")
print(f"  Avg SL dist: {t['sl_dist_pip'].mean():.1f} pips")
print(f"  Avg TP dist: {t['tp_dist_pip'].mean():.1f} pips")
print()

# Hourly view of reversal value
t['hour'] = t['open_time'].dt.hour
print("  Hour | Actual pnl | Reversed pnl | Actual win% | Trend signal")
print("  -----|------------|--------------|-------------|-------------")
for hr, g in t.groupby('hour'):
    act_pnl = g['profit'].sum()
    rev_pnl_hr = -act_pnl
    wr = g['win'].mean()
    trend = "<<< STRONG TREND" if wr < 0.40 else ("range-bound" if wr > 0.65 else "")
    print(f"  {hr:02d}h  | ${act_pnl:8.1f} | ${rev_pnl_hr:10.1f}   | {wr*100:5.1f}%      | {trend}")
