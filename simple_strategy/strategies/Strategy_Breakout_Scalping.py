"""Strategy: Breakout Scalping (Updated for OPEN/CLOSE schema)

Trade logic (simple):
- Find consolidation using low ATR and tight range.
- Confirm breakout with strong volume.
- If price breaks ABOVE range and above EMA:
    - OPEN_LONG
    - CLOSE_LONG when stop loss or take profit is hit
- If price breaks BELOW range and below EMA:
    - OPEN_SHORT
    - CLOSE_SHORT when stop loss or take profit is hit
- HOLD otherwise."""

from simple_strategy.strategies.builder_presets import build_strategy_preset

STRATEGY_PARAMETERS = {
    'atr_period': {
        'type': 'int',
        'default': 14,
        'min': 10,
        'max': 20,
        'description': 'ATR period for volatility measurement',
        'gui_hint': 'Standard values: 14, 10 for faster reaction'
    },
    'atr_threshold': {
        'type': 'float',
        'default': 0.15,
        'min': 0.05,
        'max': 0.3,
        'description': 'ATR threshold for consolidation (as % of price)',
        'gui_hint': 'Lower = tighter consolidation, fewer breakouts'
    },
    'range_period': {
        'type': 'int',
        'default': 20,
        'min': 10,
        'max': 50,
        'description': 'Lookback period for range detection',
        'gui_hint': 'Higher = longer consolidation periods'
    },
    'ema_period': {
        'type': 'int',
        'default': 20,
        'min': 10,
        'max': 50,
        'description': 'EMA period for trend filter',
        'gui_hint': 'Higher = smoother trend filter'
    },
    'volume_sma_period': {
        'type': 'int',
        'default': 20,
        'min': 10,
        'max': 50,
        'description': 'Volume SMA period for confirmation',
        'gui_hint': 'Higher = smoother volume trend'
    },
    'volume_multiplier': {
        'type': 'float',
        'default': 2.0,
        'min': 1.5,
        'max': 3.0,
        'description': 'Volume multiplier for breakout confirmation',
        'gui_hint': 'Higher = stronger volume confirmation'
    },
    'atr_stop_loss': {
        'type': 'float',
        'default': 0.8,
        'min': 0.5,
        'max': 1.5,
        'description': 'ATR multiplier for stop loss',
        'gui_hint': 'Lower = tighter stops, more conservative'
    },
    'atr_take_profit': {
        'type': 'float',
        'default': 1.5,
        'min': 1.0,
        'max': 3.0,
        'description': 'ATR multiplier for take profit',
        'gui_hint': 'Higher = larger profit targets'
    },
    'breakout_threshold': {
        'type': 'float',
        'default': 0.1,
        'min': 0.05,
        'max': 0.2,
        'description': 'Breakout threshold as % of range',
        'gui_hint': 'Lower = easier breakouts, more signals'
    },
    'risk_per_trade': {
        'type': 'float',
        'default': 0.5,
        'min': 0.1,
        'max': 1.0,
        'description': 'Risk per trade as % of account balance',
        'gui_hint': 'Lower = more conservative. Recommended: 0.5%'
    }
}


def create_strategy(symbols=None, timeframes=None, **params):
    return build_strategy_preset("Strategy_Breakout_Scalping", symbols=symbols, timeframes=timeframes, **params)

if __name__ == "__main__":
    strategy = create_strategy()
    print(strategy.get_strategy_info())
