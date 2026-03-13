"""Strategy 2: Mean Reversion with Trend Filter (Updated for OPEN/CLOSE schema)

Trade logic (simple):
- Calculate RSI plus a fast/slow EMA trend filter.
- If EMA fast > EMA slow (uptrend):
  - OPEN_LONG when RSI crosses below the oversold level.
  - CLOSE_LONG when RSI crosses above the overbought level.
- If EMA fast < EMA slow (downtrend) and bidirectional=True:
  - OPEN_SHORT when RSI crosses above the overbought level.
  - CLOSE_SHORT when RSI crosses below the oversold level.
- HOLD otherwise."""

from simple_strategy.strategies.builder_presets import build_strategy_preset

STRATEGY_PARAMETERS = {
    'rsi_period': {
        'type': 'int',
        'default': 14,
        'min': 1,
        'max': 50,
        'description': 'RSI calculation period',
        'gui_hint': 'Standard values: 14, 21. Lower = more sensitive'
    },
    'rsi_overbought': {
        'type': 'int',
        'default': 70,
        'min': 50,
        'max': 90,
        'description': 'RSI overbought level (signal on crossover)',
        'gui_hint': 'Higher values = more conservative signals'
    },
    'rsi_oversold': {
        'type': 'int',
        'default': 30,
        'min': 10,
        'max': 50,
        'description': 'RSI oversold level (signal on crossover)',
        'gui_hint': 'Lower values = more conservative signals'
    },
    'trend_fast_ema': {
        'type': 'int',
        'default': 20,
        'min': 5,
        'max': 50,
        'description': 'Fast EMA period for trend detection',
        'gui_hint': 'Lower values = more responsive trend signals'
    },
    'trend_slow_ema': {
        'type': 'int',
        'default': 50,
        'min': 20,
        'max': 200,
        'description': 'Slow EMA period for trend detection',
        'gui_hint': 'Higher values = smoother trend signals'
    },
    'bidirectional': {
        'type': 'bool',
        'default': True,
        'description': 'Enable bidirectional trading (long and short)',
        'gui_hint': 'When enabled, trades short positions in downtrends'
    }
}


def create_strategy(symbols=None, timeframes=None, **params):
    return build_strategy_preset("Strategy_2_mean_reversion", symbols=symbols, timeframes=timeframes, **params)

if __name__ == "__main__":
    strategy = create_strategy()
    print(strategy.get_strategy_info())
