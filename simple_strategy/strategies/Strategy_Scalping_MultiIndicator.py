"""Strategy: Scalping Multi-Indicator (Updated for OPEN/CLOSE schema)

Trade logic (simple):
- Build three indicator signals: RSI (overbought/oversold), EMA crossover, MACD crossover.
- Combine them using weighted voting.
- If the weighted signal is strongly bullish, OPEN_LONG.
- If the weighted signal is strongly bearish, OPEN_SHORT.
- Use ATR-based stop loss and take profit to CLOSE positions.
- Also CLOSE if the weighted signal flips against the open position.
- HOLD otherwise."""

from simple_strategy.strategies.builder_presets import build_strategy_preset

STRATEGY_PARAMETERS = {
    'rsi_period': {
        'type': 'int',
        'default': 9,
        'min': 5,
        'max': 21,
        'description': 'RSI calculation period (shorter for scalping)',
        'gui_hint': 'Scalping: Use 5-9 for faster signals'
    },
    'rsi_overbought': {
        'type': 'int',
        'default': 75,
        'min': 65,
        'max': 85,
        'description': 'RSI overbought level (sell signal)',
        'gui_hint': 'Higher = more conservative sells (75-80 recommended)'
    },
    'rsi_oversold': {
        'type': 'int',
        'default': 25,
        'min': 15,
        'max': 35,
        'description': 'RSI oversold level (buy signal)',
        'gui_hint': 'Lower = more conservative buys (20-25 recommended)'
    },
    'ema_fast': {
        'type': 'int',
        'default': 5,
        'min': 3,
        'max': 10,
        'description': 'Fast EMA period for trend direction',
        'gui_hint': 'Scalping: Use 3-5 for quick trend changes'
    },
    'ema_slow': {
        'type': 'int',
        'default': 13,
        'min': 8,
        'max': 21,
        'description': 'Slow EMA period for trend confirmation',
        'gui_hint': 'Scalping: Use 9-13 for trend confirmation'
    },
    'macd_fast': {
        'type': 'int',
        'default': 5,
        'min': 3,
        'max': 8,
        'description': 'MACD fast EMA period',
        'gui_hint': 'Scalping: Use 3-5 for faster MACD signals'
    },
    'macd_slow': {
        'type': 'int',
        'default': 13,
        'min': 9,
        'max': 21,
        'description': 'MACD slow EMA period',
        'gui_hint': 'Scalping: Use 9-13 for MACD confirmation'
    },
    'macd_signal': {
        'type': 'int',
        'default': 6,
        'min': 3,
        'max': 9,
        'description': 'MACD signal line period',
        'gui_hint': 'Scalping: Use 3-6 for quick MACD signals'
    },
    'stop_loss_atr': {
        'type': 'float',
        'default': 1.5,
        'min': 0.5,
        'max': 3.0,
        'description': 'Stop loss as ATR multiplier',
        'gui_hint': 'Scalping: Use 0.5-1.5 for tight stops'
    },
    'take_profit_atr': {
        'type': 'float',
        'default': 2.5,
        'min': 1.0,
        'max': 5.0,
        'description': 'Take profit as ATR multiplier',
        'gui_hint': 'Scalping: Use 1.5-3.0 for quick profits'
    },
    'atr_period': {
        'type': 'int',
        'default': 14,
        'min': 7,
        'max': 21,
        'description': 'ATR calculation period',
        'gui_hint': 'Use 7-14 for current volatility'
    },
    'rsi_weight': {
        'type': 'float',
        'default': 0.33,
        'min': 0.1,
        'max': 0.6,
        'description': 'RSI signal weight (0.0-1.0)',
        'gui_hint': 'Higher = more importance to RSI signals'
    },
    'ema_weight': {
        'type': 'float',
        'default': 0.33,
        'min': 0.1,
        'max': 0.6,
        'description': 'EMA crossover signal weight (0.0-1.0)',
        'gui_hint': 'Higher = more importance to trend signals'
    },
    'macd_weight': {
        'type': 'float',
        'default': 0.34,
        'min': 0.1,
        'max': 0.6,
        'description': 'MACD signal weight (0.0-1.0)',
        'gui_hint': 'Higher = more importance to momentum signals'
    },
    'signal_threshold': {
        'type': 'float',
        'default': 0.3,
        'min': 0.1,
        'max': 0.9,
        'description': 'Weighted signal threshold for opening trades',
        'gui_hint': 'Lower = more trades, higher = stronger confirmation'
    }
}


def create_strategy(symbols=None, timeframes=None, **params):
    return build_strategy_preset("Strategy_Scalping_MultiIndicator", symbols=symbols, timeframes=timeframes, **params)

if __name__ == "__main__":
    strategy = create_strategy()
    print(strategy.get_strategy_info())
