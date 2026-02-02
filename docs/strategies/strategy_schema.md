# Strategy Schema

## Signal Schema
Strategies emit one of the following per symbol/timeframe:
- `OPEN_LONG`
- `OPEN_SHORT`
- `CLOSE_LONG`
- `CLOSE_SHORT`
- `HOLD`

## Strategy File Conventions
- File name starts with `Strategy_` and lives in `simple_strategy/strategies/`.
- Must define `STRATEGY_PARAMETERS` at top (for GUI).
- Must expose `create_strategy(symbols, timeframes, **params)`.
- Optional `simple_test()` helper for quick sanity checks.
- Optional `generate_signals_vectorized(data)` for faster signal generation (used by backtester/paper trader when available).

## Data Contract
Strategies receive:
```python
{symbol: {timeframe: pandas.DataFrame}}
```
DataFrame must include `open`, `high`, `low`, `close`, `volume` and an indexed datetime.
