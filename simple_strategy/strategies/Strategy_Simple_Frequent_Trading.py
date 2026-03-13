"""Strategy: Simple Frequent Trading (Updated for OPEN/CLOSE schema)

Trade logic (simple):
- Use fast/slow SMA crossover for direction.
- Use relaxed RSI extremes for extra confirmation.
- Use tiny price change as an additional directional nudge.
- If the combined signal is bullish -> OPEN_LONG.
- If the combined signal is bearish -> OPEN_SHORT.
- If the signal flips against the current position -> CLOSE position.
- HOLD otherwise."""

from simple_strategy.strategies.builder_presets import build_strategy_preset

STRATEGY_PARAMETERS = {
    'fast_ma_period': {
        'type': 'int',
        'default': 3,
        'min': 2,
        'max': 10,
        'description': 'Fast SMA period for entry signals',
        'gui_hint': 'Lower values = more frequent signals'
    },
    'slow_ma_period': {
        'type': 'int',
        'default': 8,
        'min': 5,
        'max': 20,
        'description': 'Slow SMA period for entry signals',
        'gui_hint': 'Higher values = smoother signals'
    },
    'rsi_period': {
        'type': 'int',
        'default': 7,
        'min': 3,
        'max': 14,
        'description': 'RSI period for signal confirmation',
        'gui_hint': 'Lower values = more responsive'
    },
    'rsi_overbought': {
        'type': 'int',
        'default': 60,
        'min': 55,
        'max': 70,
        'description': 'RSI overbought threshold',
        'gui_hint': 'Lower = more sell signals'
    },
    'rsi_oversold': {
        'type': 'int',
        'default': 40,
        'min': 30,
        'max': 45,
        'description': 'RSI oversold threshold',
        'gui_hint': 'Higher = more buy signals'
    },
    'price_change_threshold': {
        'type': 'float',
        'default': 0.001,
        'min': 0.0005,
        'max': 0.005,
        'description': 'Price change threshold for signals',
        'gui_hint': 'Lower = more frequent signals'
    }
}


def create_strategy(symbols=None, timeframes=None, **params):
    return build_strategy_preset("Strategy_Simple_Frequent_Trading", symbols=symbols, timeframes=timeframes, **params)

if __name__ == "__main__":
    strategy = create_strategy()
    print(strategy.get_strategy_info())
