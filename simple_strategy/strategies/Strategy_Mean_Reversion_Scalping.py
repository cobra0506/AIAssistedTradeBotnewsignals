"""Strategy: Mean Reversion Scalping (Updated for OPEN/CLOSE schema)

Trade logic (simple):
- Use RSI + Bollinger Bands to detect overbought/oversold.
- OPEN_SHORT when RSI > overbought AND price > upper band (with volume confirmation).
- OPEN_LONG when RSI < oversold AND price < lower band (with volume confirmation).
- CLOSE_SHORT when RSI drops below exit level OR price returns to middle band.
- CLOSE_LONG when RSI rises above exit level OR price returns to middle band.
- HOLD otherwise."""

from simple_strategy.strategies.builder_presets import build_strategy_preset

STRATEGY_PARAMETERS = {
    'rsi_period': {
        'type': 'int',
        'default': 14,
        'min': 7,
        'max': 21,
        'description': 'RSI period for mean reversion signals',
        'gui_hint': 'Standard values: 14, 10 for faster signals'
    },
    'rsi_overbought': {
        'type': 'int',
        'default': 70,
        'min': 65,
        'max': 80,
        'description': 'RSI overbought level for sell signals',
        'gui_hint': 'Higher = more extreme overbought, fewer signals'
    },
    'rsi_oversold': {
        'type': 'int',
        'default': 30,
        'min': 20,
        'max': 35,
        'description': 'RSI oversold level for buy signals',
        'gui_hint': 'Lower = more extreme oversold, fewer signals'
    },
    'bb_period': {
        'type': 'int',
        'default': 20,
        'min': 15,
        'max': 30,
        'description': 'Bollinger Bands period for mean reversion',
        'gui_hint': 'Standard values: 20, 15 for faster signals'
    },
    'bb_std_dev': {
        'type': 'float',
        'default': 2.0,
        'min': 1.8,
        'max': 2.5,
        'description': 'Bollinger Bands standard deviation',
        'gui_hint': 'Higher = wider bands, more extreme signals'
    },
    'atr_period': {
        'type': 'int',
        'default': 14,
        'min': 10,
        'max': 20,
        'description': 'ATR period for volatility measurement',
        'gui_hint': 'Standard values: 14, 10 for faster reaction'
    },
    'atr_stop_loss': {
        'type': 'float',
        'default': 0.5,
        'min': 0.3,
        'max': 1.0,
        'description': 'ATR multiplier for stop loss (tight for scalping)',
        'gui_hint': 'Lower = tighter stops, more conservative'
    },
    'atr_take_profit': {
        'type': 'float',
        'default': 1.0,
        'min': 0.5,
        'max': 2.0,
        'description': 'ATR multiplier for take profit (quick exits)',
        'gui_hint': 'Higher = larger profit targets'
    },
    'volume_sma_period': {
        'type': 'int',
        'default': 20,
        'min': 10,
        'max': 50,
        'description': 'Volume SMA period for confirmation',
        'gui_hint': 'Higher = smoother volume trend'
    },
    'rsi_exit_level': {
        'type': 'int',
        'default': 50,
        'min': 45,
        'max': 55,
        'description': 'RSI level for exit (mean reversion target)',
        'gui_hint': 'Standard is 50 (middle of RSI range)'
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
    return build_strategy_preset("Strategy_Mean_Reversion_Scalping", symbols=symbols, timeframes=timeframes, **params)

if __name__ == "__main__":
    strategy = create_strategy()
    print(strategy.get_strategy_info())
