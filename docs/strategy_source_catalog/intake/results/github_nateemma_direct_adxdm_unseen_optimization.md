# ADXDM Unseen-Data Optimization Results

- Strategy: `Strategy_Import_Nateemma_Direct_ADXDM`
- Symbols: `BNBUSDT, ADAUSDT, XRPUSDT, ALGOUSDT, ARBUSDT, ATOMUSDT, DOTUSDT, FILUSDT, NEARUSDT, OPUSDT`
- Training range: `2025-11-29 00:00:00` -> `2026-02-27 00:00:00`
- Unseen range: `2026-02-27 00:00:00` -> `2026-03-30 00:00:00`
- Trial count: `8`
- Wall-clock start: `2026-03-30 07:19:02`
- Wall-clock end: `2026-03-30 09:19:02`
- Wall-clock elapsed seconds: `7200.43`

## Best Training Trial
- Params: `{'buy_adx': 50.0, 'trend_ma_period': 50, 'trend_deadband_pct': 0.1}`
- Return %: `-7.4759`
- Trades: `10`
- Win rate %: `40.0`
- Sharpe: `-6.6614`
- Max DD %: `7.4759`

## Unseen Comparison
- Default params `{'buy_adx': 60.0, 'trend_ma_period': 50, 'trend_deadband_pct': 0.15}` -> return `-2.0271`, trades `10`, sharpe `-5.349`
- Optimized params `{'buy_adx': 50.0, 'trend_ma_period': 50, 'trend_deadband_pct': 0.1}` -> return `-4.0101`, trades `10`, sharpe `-12.4245`

| buy_adx | trend_ma_period | deadband | Train Return % | Train Trades | Train Win Rate % | Train Sharpe | Train Max DD % |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50.0 | 50 | 0.10 | -7.4759 | 10 | 40.0000 | -6.6614 | 7.4759 |
| 50.0 | 50 | 0.15 | -7.4759 | 10 | 40.0000 | -6.6614 | 7.4759 |
| 60.0 | 50 | 0.10 | -9.6690 | 10 | 30.0000 | -9.0303 | 9.6690 |
| 60.0 | 50 | 0.15 | -9.6690 | 10 | 30.0000 | -9.0303 | 9.6690 |
| 50.0 | 34 | 0.10 | -11.1462 | 10 | 30.0000 | -11.2052 | 11.1462 |
| 50.0 | 34 | 0.15 | -11.1462 | 10 | 30.0000 | -11.2052 | 11.1462 |
| 60.0 | 34 | 0.15 | -13.3396 | 10 | 20.0000 | -14.7874 | 13.3396 |
| 60.0 | 34 | 0.10 | -13.3396 | 10 | 20.0000 | -14.7874 | 13.3396 |
