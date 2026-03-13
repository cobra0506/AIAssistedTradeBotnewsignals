"""Fast Test Scalper Strategy
==========================

Simple, deterministic strategy intended to open and close positions quickly
so you can verify the paper trader/backtester execution path.

Logic:
- OPEN_LONG when the latest close is above the previous close.
- CLOSE_LONG when the latest close is below the previous close.

This intentionally creates frequent opens/closes to validate execution."""

from simple_strategy.strategies.builder_presets import build_strategy_preset

STRATEGY_PARAMETERS = {
    'min_periods': {
        'type': 'int',
        'default': 2,
        'min': 2,
        'max': 10,
        'description': 'Minimum candles required before emitting signals',
        'gui_hint': 'Keep at 2 for fastest signal generation'
    }
}


def create_strategy(symbols=None, timeframes=None, **params):
    return build_strategy_preset("Strategy_Fast_Test_Scalper", symbols=symbols, timeframes=timeframes, **params)

if __name__ == "__main__":
    strategy = create_strategy()
    print(strategy.get_strategy_info())
