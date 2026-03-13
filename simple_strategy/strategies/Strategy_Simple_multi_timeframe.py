"""Strategy: Simple Multi-Timeframe (Updated for OPEN/CLOSE schema)

Trade logic (simple):
- Use 5m EMA trend filter: price above trend EMA = bullish, below = bearish.
- On 1m timeframe:
  - If bullish trend and fast EMA > slow EMA -> OPEN_LONG.
  - If bearish trend and fast EMA < slow EMA -> OPEN_SHORT.
- If in a position and the opposite conditions appear -> CLOSE the position.
- HOLD otherwise."""

from simple_strategy.strategies.builder_presets import build_strategy_preset

STRATEGY_PARAMETERS = {
    'fast_ema_period': {
        'type': 'int',
        'default': 9,
        'min': 5,
        'max': 20,
        'description': 'Fast EMA period for entry signals',
        'gui_hint': 'For 1m timeframe. Lower = more sensitive'
    },
    'slow_ema_period': {
        'type': 'int',
        'default': 21,
        'min': 15,
        'max': 50,
        'description': 'Slow EMA period for entry signals',
        'gui_hint': 'For 1m timeframe. Higher = smoother'
    },
    'trend_ema_period': {
        'type': 'int',
        'default': 50,
        'min': 20,
        'max': 100,
        'description': 'Trend EMA period for higher timeframe',
        'gui_hint': 'For 5m timeframe. Higher = longer term trend'
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
    return build_strategy_preset("Strategy_Simple_multi_timeframe", symbols=symbols, timeframes=timeframes, **params)

if __name__ == "__main__":
    strategy = create_strategy()
    print(strategy.get_strategy_info())
