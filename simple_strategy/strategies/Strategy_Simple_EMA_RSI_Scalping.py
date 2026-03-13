"""RSI Mean Reversion with EMA Trend Filter Strategy
===============================================

A trend-filtered RSI mean reversion strategy that:
- Uses EMA crossover to determine trend direction
- Uses RSI crossovers for precise entry/exit signals
- Only trades in the direction of the prevailing trend
- Implements proper OPEN/CLOSE signal logic

Strategy Logic:
1. UPTREND (Fast EMA > Slow EMA):
   - OPEN_LONG when RSI crosses UP through oversold level
   - CLOSE_LONG when RSI crosses DOWN through overbought level

2. DOWNTREND (Fast EMA < Slow EMA):
   - OPEN_SHORT when RSI crosses DOWN through overbought level
   - CLOSE_SHORT when RSI crosses UP through oversold level

Author: AI Assisted TradeBot Team
Date: 2025"""

from simple_strategy.strategies.builder_presets import build_strategy_preset

STRATEGY_PARAMETERS = {
    # Fast EMA for trend direction
    'fast_ema_period': {
        'type': 'int',
        'default': 20,
        'min': 5,
        'max': 50,
        'description': 'Fast EMA period for trend direction',
        'gui_hint': 'Lower values = more sensitive trend. Recommended: 15-25'
    },
    # Slow EMA for trend direction
    'slow_ema_period': {
        'type': 'int',
        'default': 50,
        'min': 20,
        'max': 100,
        'description': 'Slow EMA period for trend direction',
        'gui_hint': 'Higher values = smoother trend. Recommended: 40-60'
    },
    # RSI for entry/exit signals
    'rsi_period': {
        'type': 'int',
        'default': 14,
        'min': 7,
        'max': 21,
        'description': 'RSI period for signals',
        'gui_hint': 'Standard values: 14, 10 for faster signals'
    },
    # RSI levels
    'rsi_overbought': {
        'type': 'int',
        'default': 70,
        'min': 65,
        'max': 80,
        'description': 'RSI overbought level',
        'gui_hint': 'Level where RSI is considered overbought'
    },
    'rsi_oversold': {
        'type': 'int',
        'default': 30,
        'min': 20,
        'max': 35,
        'description': 'RSI oversold level',
        'gui_hint': 'Level where RSI is considered oversold'
    }
}


def create_strategy(symbols=None, timeframes=None, **params):
    return build_strategy_preset("Strategy_Simple_EMA_RSI_Scalping", symbols=symbols, timeframes=timeframes, **params)

if __name__ == "__main__":
    strategy = create_strategy()
    print(strategy.get_strategy_info())
