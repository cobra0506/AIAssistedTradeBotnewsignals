"""Strategy: Simple MA Crossover (Updated for OPEN/CLOSE schema)

Trade logic (simple):
- Use 5m EMA trend filter: price above trend EMA = bullish, below = bearish.
- On 1m timeframe:
  - If bullish trend and fast EMA > slow EMA and RSI is bullish -> OPEN_LONG.
  - If bearish trend and fast EMA < slow EMA and RSI is bearish -> OPEN_SHORT.
- If in a position and the opposite conditions appear -> CLOSE the position.
- HOLD otherwise."""

from simple_strategy.strategies.builder_presets import build_strategy_preset

STRATEGY_PARAMETERS = {
    'fast_ma_period': {
        'type': 'int',
        'default': 5,
        'min': 3,
        'max': 10,
        'description': 'Fast EMA period for entry signals',
        'gui_hint': 'For 1m timeframe. Lower = more responsive'
    },
    'slow_ma_period': {
        'type': 'int',
        'default': 15,
        'min': 10,
        'max': 30,
        'description': 'Slow EMA period for entry signals',
        'gui_hint': 'For 1m timeframe. Higher = smoother'
    },
    'trend_ma_period': {
        'type': 'int',
        'default': 50,
        'min': 20,
        'max': 100,
        'description': 'Trend EMA period for higher timeframe',
        'gui_hint': 'For 5m timeframe. Higher = longer term trend'
    },
    'rsi_period': {
        'type': 'int',
        'default': 14,
        'min': 7,
        'max': 21,
        'description': 'RSI period for signal confirmation',
        'gui_hint': 'Standard values: 14, 10 for faster signals'
    },
    'rsi_bullish_threshold': {
        'type': 'int',
        'default': 55,
        'min': 50,
        'max': 60,
        'description': 'RSI threshold for bullish confirmation',
        'gui_hint': 'Above this level confirms bullish signals'
    },
    'rsi_bearish_threshold': {
        'type': 'int',
        'default': 45,
        'min': 40,
        'max': 50,
        'description': 'RSI threshold for bearish confirmation',
        'gui_hint': 'Below this level confirms bearish signals'
    },
    'trend_timeframe': {
        'type': 'str',
        'default': '5m',
        'options': ['5m', '15m', '30m'],
        'description': 'Higher timeframe for trend confirmation',
        'gui_hint': 'Default is 5m'
    },
    'entry_timeframe': {
        'type': 'str',
        'default': '1m',
        'options': ['1m', '3m', '5m'],
        'description': 'Entry timeframe for signals',
        'gui_hint': 'Default is 1m'
    }
}


def create_strategy(symbols=None, timeframes=None, **params):
    return build_strategy_preset("Strategy_Simple_MA_Crossover", symbols=symbols, timeframes=timeframes, **params)

if __name__ == "__main__":
    strategy = create_strategy()
    print(strategy.get_strategy_info())
