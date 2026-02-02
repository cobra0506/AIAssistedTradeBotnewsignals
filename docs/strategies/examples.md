# Strategy Examples

## Simple RSI Strategy (Builder)
```python
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import rsi
from simple_strategy.strategies.signals_library import overbought_oversold

builder = StrategyBuilder(['BTCUSDT'], ['1h'])
builder.add_indicator('rsi', rsi, period=14)
builder.add_signal_rule('rsi_signal', overbought_oversold, indicator='rsi', overbought=70, oversold=30)
strategy = builder.build()
```

## Strategy File Skeleton
```python
STRATEGY_PARAMETERS = {
    'rsi_period': {'type': 'int', 'default': 14, 'min': 5, 'max': 50}
}

def create_strategy(symbols=None, timeframes=None, **params):
    # build strategy here
    return strategy
```

