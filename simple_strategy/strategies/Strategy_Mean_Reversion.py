"""Strategy: Mean Reversion (Updated for OPEN/CLOSE schema)

Trade logic (simple):
- Build Bollinger-style bands using SMA and standard deviation.
- OPEN_LONG when price closes below the lower band.
- OPEN_SHORT when price closes above the upper band.
- CLOSE_LONG when price crosses back above the lower exit band.
- CLOSE_SHORT when price crosses back below the upper exit band.
- HOLD otherwise."""

from simple_strategy.strategies.builder_presets import build_strategy_preset

STRATEGY_PARAMETERS = {
    'lookback_period': {
        'type': 'int',
        'default': 20,
        'min': 5,
        'max': 100,
        'description': 'SMA lookback period for Bollinger bands',
        'gui_hint': 'Typical values: 20'
    },
    'entry_threshold': {
        'type': 'float',
        'default': 2.0,
        'min': 1.0,
        'max': 3.0,
        'description': 'Std-dev multiplier for entry bands',
        'gui_hint': 'Typical values: 2.0'
    },
    'exit_threshold': {
        'type': 'float',
        'default': 0.5,
        'min': 0.1,
        'max': 2.0,
        'description': 'Std-dev multiplier for exit bands',
        'gui_hint': 'Typical values: 0.5-1.0'
    }
}


def create_strategy(symbols=None, timeframes=None, **params):
    remapped = dict(params)
    remapped.setdefault("rsi_period", params.get("lookback_period", 20))
    remapped.setdefault("rsi_oversold", 50 - float(params.get("entry_threshold", 2.0)) * 10.0)
    remapped.setdefault("rsi_overbought", 50 + float(params.get("entry_threshold", 2.0)) * 10.0)
    return build_strategy_preset("Strategy_Mean_Reversion", symbols=symbols, timeframes=timeframes, **remapped)

if __name__ == "__main__":
    strategy = create_strategy()
    print(strategy.get_strategy_info())
