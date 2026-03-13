"""Moving Average Crossover with RSI Confirmation Strategy
=======================================================

A trend-following strategy that:
- Uses fast and slow moving averages to determine trend direction
- Uses RSI to confirm signals and avoid overbought/oversold conditions
- Implements proper OPEN/CLOSE signal logic

Strategy Logic:
1. Trend Detection:
   - UPTREND when Fast MA > Slow MA
   - DOWNTREND when Fast MA < Slow MA

2. Signal Generation:
   - OPEN_LONG when Fast MA crosses above Slow MA AND RSI < 70
   - CLOSE_LONG when Fast MA crosses below Slow MA
   - OPEN_SHORT when Fast MA crosses below Slow MA AND RSI > 30
   - CLOSE_SHORT when Fast MA crosses above Slow MA

Author: AI Assisted TradeBot Team
Date: 2025"""

from simple_strategy.strategies.builder_presets import build_strategy_preset

STRATEGY_PARAMETERS = {
    # Fast MA for trend direction
    'fast_ma_period': {
        'type': 'int',
        'default': 10,
        'min': 5,
        'max': 20,
        'description': 'Fast moving average period',
        'gui_hint': 'Lower values = more sensitive signals. Recommended: 8-12'
    },
    # Slow MA for trend direction
    'slow_ma_period': {
        'type': 'int',
        'default': 30,
        'min': 20,
        'max': 50,
        'description': 'Slow moving average period',
        'gui_hint': 'Higher values = smoother trend. Recommended: 25-35'
    },
    # RSI for signal confirmation
    'rsi_period': {
        'type': 'int',
        'default': 14,
        'min': 7,
        'max': 21,
        'description': 'RSI period for signal confirmation',
        'gui_hint': 'Standard values: 14, 10 for faster signals'
    },
    # RSI levels
    'rsi_overbought': {
        'type': 'int',
        'default': 70,
        'min': 65,
        'max': 80,
        'description': 'RSI overbought level',
        'gui_hint': 'Avoid long entries above this level'
    },
    'rsi_oversold': {
        'type': 'int',
        'default': 30,
        'min': 20,
        'max': 35,
        'description': 'RSI oversold level',
        'gui_hint': 'Avoid short entries below this level'
    }
}


def create_strategy(symbols=None, timeframes=None, **params):
    return build_strategy_preset("Strategy_NEW__MAcrossover_RSIconfirm", symbols=symbols, timeframes=timeframes, **params)

if __name__ == "__main__":
    strategy = create_strategy()
    print(strategy.get_strategy_info())
