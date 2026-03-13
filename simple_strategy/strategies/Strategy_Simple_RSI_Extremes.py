"""Strategy: Simple RSI Extremes (Updated for OPEN/CLOSE schema)

Trade logic (simple):
- Use RSI extremes for mean-reversion signals.
- Use a trend SMA filter to trade with the trend.
- Use volume SMA confirmation to avoid weak signals.
- If oversold + uptrend + volume confirmed -> OPEN_LONG.
- If overbought + downtrend + volume confirmed -> OPEN_SHORT.
- If in a position and the opposite extreme appears -> CLOSE.
- HOLD otherwise."""

from simple_strategy.strategies.builder_presets import build_strategy_preset

STRATEGY_PARAMETERS = {
    'rsi_period': {
        'type': 'int',
        'default': 8,
        'min': 7,
        'max': 21,
        'description': 'RSI period for mean reversion signals',
        'gui_hint': 'Standard values: 14, 10 for faster signals'
    },
    'rsi_overbought': {
        'type': 'int',
        'default': 77,
        'min': 70,
        'max': 80,
        'description': 'RSI overbought level for sell signals',
        'gui_hint': 'Higher = more extreme overbought, fewer but higher quality signals'
    },
    'rsi_oversold': {
        'type': 'int',
        'default': 26,
        'min': 20,
        'max': 30,
        'description': 'RSI oversold level for buy signals',
        'gui_hint': 'Lower = more extreme oversold, fewer but higher quality signals'
    },
    'trend_sma_period': {
        'type': 'int',
        'default': 233,
        'min': 50,
        'max': 300,
        'description': 'SMA period for trend filter',
        'gui_hint': 'Higher = longer term trend. 200 is standard for daily charts'
    },
    'volume_sma_period': {
        'type': 'int',
        'default': 34,
        'min': 10,
        'max': 50,
        'description': 'Volume SMA period for confirmation',
        'gui_hint': 'Higher = smoother volume trend'
    },
    'volume_multiplier': {
        'type': 'float',
        'default': 1.79,
        'min': 1.0,
        'max': 2.0,
        'description': 'Volume multiplier for signal confirmation',
        'gui_hint': 'Higher = stronger volume confirmation required'
    }
}


def create_strategy(symbols=None, timeframes=None, **params):
    return build_strategy_preset("Strategy_Simple_RSI_Extremes", symbols=symbols, timeframes=timeframes, **params)

if __name__ == "__main__":
    strategy = create_strategy()
    print(strategy.get_strategy_info())
