"""Strategy 1: Trend Following Strategy (Updated for OPEN/CLOSE schema)
Uses moving average crossovers to identify trend direction and trade accordingly.
Exact trade logic for Strategy_1_Trend_Following
    Indicators: Fast MA and slow MA computed on closing prices (SMA or EMA based on ma_type).
    Crossover detection:
        Cross Up: fast crosses above slow when last_fast > last_slow and prev_fast <= prev_slow.
        Cross Down: fast crosses below slow when last_fast < last_slow and prev_fast >= prev_slow.
    Signal decisions (single signal per bar):
        Cross Up:
            If currently short → CLOSE_SHORT
            If no open position → OPEN_LONG
        Cross Down:
            If currently long → CLOSE_LONG
            If no open position → OPEN_SHORT
        Otherwise → HOLD
    Position gating: Signals are suppressed if they are invalid for the current position state (e.g., trying to open while already in a position, or closing the wrong side)."""

from simple_strategy.strategies.builder_presets import build_strategy_preset

STRATEGY_PARAMETERS = {
    'fast_period': {
        'type': 'int',
        'default': 12,
        'min': 5,
        'max': 50,
        'description': 'Fast moving average period',
        'gui_hint': 'Lower values = more sensitive signals'
    },
    'slow_period': {
        'type': 'int',
        'default': 26,
        'min': 10,
        'max': 100,
        'description': 'Slow moving average period',
        'gui_hint': 'Should be 2-3x the fast period'
    },
    'ma_type': {
        'type': 'str',
        'default': 'ema',
        'options': ['sma', 'ema'],
        'description': 'Moving average type',
        'gui_hint': 'EMA reacts faster to price changes'
    }
}


def create_strategy(symbols=None, timeframes=None, **params):
    return build_strategy_preset("Strategy_1_Trend_Following", symbols=symbols, timeframes=timeframes, **params)

if __name__ == "__main__":
    strategy = create_strategy()
    print(strategy.get_strategy_info())
