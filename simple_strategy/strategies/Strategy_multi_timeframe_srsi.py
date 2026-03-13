"""Strategy: Multi-Timeframe SRSI (Updated for OPEN/CLOSE schema)

Trade logic (simple):
- Use RSI extremes (overbought/oversold) plus a fast/slow SMA crossover.
- Combine the RSI signal and the MA crossover signal (majority vote style).
- If combined signal is bullish -> OPEN_LONG.
- If combined signal is bearish -> OPEN_SHORT.
- If already in a position and the combined signal flips -> CLOSE the position.
- HOLD otherwise."""

from simple_strategy.strategies.builder_presets import build_strategy_preset

STRATEGY_PARAMETERS = {
    'oversold_threshold': {
        'type': 'int',
        'default': 20,
        'min': 5,
        'max': 30,
        'description': 'RSI oversold level (buy signal)',
        'gui_hint': 'Lower values = more conservative long-entry signals. Recommended: 20'
    },
    'overbought_threshold': {
        'type': 'int',
        'default': 80,
        'min': 70,
        'max': 95,
        'description': 'RSI overbought level (sell signal)',
        'gui_hint': 'Higher values = more conservative short-entry signals. Recommended: 80'
    },
    'rsi_period': {
        'type': 'int',
        'default': 14,
        'min': 7,
        'max': 21,
        'description': 'RSI calculation period',
        'gui_hint': 'Standard values: 14, 21. Lower = more sensitive'
    },
    'sma_fast_period': {
        'type': 'int',
        'default': 12,
        'min': 5,
        'max': 30,
        'description': 'Fast SMA period',
        'gui_hint': 'Lower = more responsive'
    },
    'sma_slow_period': {
        'type': 'int',
        'default': 26,
        'min': 10,
        'max': 50,
        'description': 'Slow SMA period',
        'gui_hint': 'Higher = smoother trend'
    }
}


def create_strategy(symbols=None, timeframes=None, **params):
    return build_strategy_preset("Strategy_multi_timeframe_srsi", symbols=symbols, timeframes=timeframes, **params)

if __name__ == "__main__":
    strategy = create_strategy()
    print(strategy.get_strategy_info())
