"""Generated strategy: Strategy_FAST_20260220_101234_04"""
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import bollinger_bands, ema, rsi
from simple_strategy.strategies.signals_library import bollinger_bands_signals, rsi_mean_reversion_with_trend

STRATEGY_PARAMETERS = {}

def create_strategy(symbols=None, timeframes=None, **params):
    symbols = symbols or ['BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT', 'NEARUSDT']
    timeframes = timeframes or ['1', '5']
    builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
    builder.add_indicator('bb_main', bollinger_bands, period=42, std_dev=2.1)
    builder.add_indicator('ema_fast', ema, period=11)
    builder.add_indicator('ema_slow', ema, period=33)
    builder.add_indicator('rsi_main', rsi, period=18)
    builder.add_signal_rule('rsi_trend_combo', rsi_mean_reversion_with_trend, rsi='rsi_main', ema_fast='ema_fast', ema_slow='ema_slow', overbought=77, oversold=40)
    builder.add_signal_rule('bb_reversion', bollinger_bands_signals, price='price', upper_band='upper_band', lower_band='lower_band')
    builder.set_signal_combination('and_signals')
    builder.set_strategy_info('Strategy_FAST_20260220_101234_04', '1.0.0')
    return builder.build()
