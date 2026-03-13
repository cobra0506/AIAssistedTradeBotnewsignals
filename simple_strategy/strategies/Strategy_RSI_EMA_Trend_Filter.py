"""Strategy: RSI EMA Trend Filter (Updated for OPEN/CLOSE schema)

Trade logic (simple):
- Determine trend using EMA fast vs EMA slow.
- If uptrend (EMA fast > EMA slow):
  - OPEN_LONG when RSI crosses below oversold.
  - CLOSE_LONG when RSI crosses above overbought.
- If downtrend (EMA fast < EMA slow):
  - OPEN_SHORT when RSI crosses above overbought.
  - CLOSE_SHORT when RSI crosses below oversold.
- HOLD otherwise."""

from simple_strategy.strategies.builder_presets import build_strategy_preset

STRATEGY_PARAMETERS = {
    'rsi_period': {'type': 'int', 'default': 14, 'min': 7, 'max': 21, 'description': 'RSI period'},
    'rsi_overbought': {'type': 'int', 'default': 75, 'min': 70, 'max': 80, 'description': 'RSI overbought level'},
    'rsi_oversold': {'type': 'int', 'default': 25, 'min': 20, 'max': 30, 'description': 'RSI oversold level'},
    'ema_fast_period': {'type': 'int', 'default': 20, 'min': 10, 'max': 50, 'description': 'Fast EMA period for trend'},
    'ema_slow_period': {'type': 'int', 'default': 50, 'min': 20, 'max': 100, 'description': 'Slow EMA period for trend'}
}


def create_strategy(symbols=None, timeframes=None, **params):
    return build_strategy_preset("Strategy_RSI_EMA_Trend_Filter", symbols=symbols, timeframes=timeframes, **params)

if __name__ == "__main__":
    strategy = create_strategy()
    print(strategy.get_strategy_info())
