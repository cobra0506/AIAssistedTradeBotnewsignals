# Indicators Library

## Purpose
Provide standalone technical indicators used across strategies.

## Key File
- `simple_strategy/strategies/indicators_library.py`

## Common Indicators
- Trend: `sma`, `ema`, `wma`, `dema`, `tema`
- Momentum: `rsi`, `stochastic`, `srsi`, `macd`, `cci`, `williams_r`
- Volatility: `atr`

## Usage
```python
from simple_strategy.strategies.indicators_library import rsi, sma
rsi_vals = rsi(df['close'], period=14)
sma_vals = sma(df['close'], period=20)
```

