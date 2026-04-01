# Big Candle Builder Optimization

- Coarse window: `2026-02-17 00:00:00` -> `2026-02-27 00:00:00`
- Train window: `2026-01-27 00:00:00` -> `2026-02-27 00:00:00`
- Unseen window: `2026-02-27 00:00:00` -> `2026-03-30 00:00:00`
- Candidates tested: `96`

## Best Train
- Params: `{'min_candle_pct': 3.0, 'target_fraction': 0.5, 'max_hold_bars': 4, 'min_body_ratio': 0.5, 'volume_spike_multiplier': 0.0, 'volume_lookback': 20, 'cooldown_bars': 2}`
- Return %: `1.755`
- Trades: `70`
- Sharpe: `5.9415`

## Best Unseen
- Params: `{'min_candle_pct': 3.0, 'target_fraction': 0.5, 'max_hold_bars': 4, 'min_body_ratio': 0.5, 'volume_spike_multiplier': 0.0, 'volume_lookback': 20, 'cooldown_bars': 2}`
- Return %: `-0.1132`
- Trades: `16`
- Sharpe: `-3.8766`
