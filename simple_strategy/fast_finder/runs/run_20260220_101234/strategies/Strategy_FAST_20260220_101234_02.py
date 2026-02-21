"""Generated strategy: Strategy_FAST_20260220_101234_02"""
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import bollinger_bands, ema, rsi
from simple_strategy.strategies.signals_library import bollinger_bands_signals, overbought_oversold, rsi_mean_reversion_with_trend

STRATEGY_PARAMETERS = {}

def create_strategy(symbols=None, timeframes=None, **params):
    symbols = symbols or ['BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT', 'NEARUSDT']
    timeframes = timeframes or ['1', '5']
    builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
    builder.add_indicator('bb_main', bollinger_bands, period=47, std_dev=1.6)
    builder.add_indicator('ema_fast', ema, period=7)
    builder.add_indicator('ema_slow', ema, period=60)
    builder.add_indicator('rsi_main', rsi, period=14)
    builder.add_signal_rule('rsi_osob', overbought_oversold, indicator='rsi_main', overbought=78, oversold=38)
    builder.add_signal_rule('rsi_trend_combo', rsi_mean_reversion_with_trend, rsi='rsi_main', ema_fast='ema_fast', ema_slow='ema_slow', overbought=72, oversold=36)
    builder.add_signal_rule('bb_reversion', bollinger_bands_signals, price='price', upper_band='upper_band', lower_band='lower_band')
    builder.set_signal_combination('and_signals')
    builder.set_strategy_info('Strategy_FAST_20260220_101234_02', '1.0.0')
    return builder.build()
