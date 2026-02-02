# Strategy Builder

## Purpose
Compose strategies from indicators + signal rules with a builder pattern.

## Key File
- `simple_strategy/strategies/strategy_builder.py`

## Typical Usage
```python
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import rsi
from simple_strategy.strategies.signals_library import overbought_oversold

builder = StrategyBuilder(['BTCUSDT'], ['1m'])
builder.add_indicator('rsi', rsi, period=14)
builder.add_signal_rule('rsi_signal', overbought_oversold, indicator='rsi', overbought=70, oversold=30)
strategy = builder.build()
```

## Known Issues (Current)
- Indicator values may not be injected into the DataFrame correctly.
- Signal function parameter passing can be inconsistent.
- Signal return types are inconsistent (string vs numeric) across signal helpers.

## Recommendation
For now, prefer explicit strategy classes in `simple_strategy/strategies/Strategy_*.py` until these issues are resolved.

