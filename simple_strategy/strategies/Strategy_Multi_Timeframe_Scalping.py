"""Strategy: Multi-Timeframe Scalping (Updated for OPEN/CLOSE schema)

Trade logic (simple):
- Use higher timeframe EMA trend as filter.
- On entry timeframe (1m by default):
  - In uptrend: OPEN_LONG when price is above EMAs, RSI not overbought,
    and Bollinger breakout/pullback confirms with volume.
  - In downtrend: OPEN_SHORT when price is below EMAs, RSI not oversold,
    and Bollinger breakout/pullback confirms with volume.
- Exit using ATR-based stop loss or take profit:
  - CLOSE_LONG when stop loss or take profit hit.
  - CLOSE_SHORT when stop loss or take profit hit.
- HOLD otherwise."""

from simple_strategy.strategies.builder_presets import build_strategy_preset

STRATEGY_PARAMETERS = {
    'fast_ema_period': {
        'type': 'int',
        'default': 10,
        'min': 5,
        'max': 20,
        'description': 'Fast EMA period for entry signals',
        'gui_hint': 'Lower values = more sensitive entries. Recommended: 8-12'
    },
    'slow_ema_period': {
        'type': 'int',
        'default': 49,
        'min': 15,
        'max': 50,
        'description': 'Slow EMA period for trend direction',
        'gui_hint': 'Higher values = smoother trend. Recommended: 20-25'
    },
    'rsi_period': {
        'type': 'int',
        'default': 17,
        'min': 7,
        'max': 21,
        'description': 'RSI period for momentum confirmation',
        'gui_hint': 'Standard values: 14, 10 for faster signals'
    },
    'rsi_overbought': {
        'type': 'int',
        'default': 72,
        'min': 60,
        'max': 80,
        'description': 'RSI overbought level for sell signals',
        'gui_hint': 'Higher = more conservative sells'
    },
    'rsi_oversold': {
        'type': 'int',
        'default': 23,
        'min': 20,
        'max': 40,
        'description': 'RSI oversold level for buy signals',
        'gui_hint': 'Lower = more conservative buys'
    },
    'atr_period': {
        'type': 'int',
        'default': 11,
        'min': 10,
        'max': 20,
        'description': 'ATR period for volatility measurement',
        'gui_hint': 'Standard values: 14, 10 for faster reaction'
    },
    'atr_multiplier_sl': {
        'type': 'float',
        'default': 0.65,
        'min': 0.5,
        'max': 3.0,
        'description': 'ATR multiplier for stop-loss distance',
        'gui_hint': 'Higher = wider stops, more conservative'
    },
    'atr_multiplier_tp': {
        'type': 'float',
        'default': 4.46,
        'min': 1.0,
        'max': 5.0,
        'description': 'ATR multiplier for take-profit distance',
        'gui_hint': 'Higher = larger profit targets'
    },
    'bb_period': {
        'type': 'int',
        'default': 24,
        'min': 15,
        'max': 30,
        'description': 'Bollinger Bands period for volatility breakouts',
        'gui_hint': 'Standard values: 20, 15 for faster signals'
    },
    'bb_std_dev': {
        'type': 'float',
        'default': 2.21,
        'min': 1.5,
        'max': 2.5,
        'description': 'Bollinger Bands standard deviation',
        'gui_hint': 'Higher = wider bands, fewer signals'
    },
    'volume_sma_period': {
        'type': 'int',
        'default': 10,
        'min': 10,
        'max': 50,
        'description': 'Volume SMA period for confirmation',
        'gui_hint': 'Higher = smoother volume trend'
    },
    'trend_timeframe': {
        'type': 'str',
        'default': '5m',
        'options': ['5m', '15m', '30m'],
        'description': 'Higher timeframe for trend confirmation',
        'gui_hint': 'Use 5m for scalping, 15m for swing trades'
    },
    'min_atr_threshold': {
        'type': 'float',
        'default': 0.13,
        'min': 0.05,
        'max': 0.5,
        'description': 'Minimum ATR threshold for trading (as % of price)',
        'gui_hint': 'Filter out low volatility periods. 0.1 = 0.1%'
    }
}


def create_strategy(symbols=None, timeframes=None, **params):
    return build_strategy_preset("Strategy_Multi_Timeframe_Scalping", symbols=symbols, timeframes=timeframes, **params)

if __name__ == "__main__":
    strategy = create_strategy()
    print(strategy.get_strategy_info())
