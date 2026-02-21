"""Generated strategy: Strategy_FAST_20260220_101234_05"""
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import bollinger_bands, rsi
from simple_strategy.strategies.signals_library import bollinger_bands_signals, divergence_signals, overbought_oversold

STRATEGY_PARAMETERS = {}

def create_strategy(symbols=None, timeframes=None, **params):
    symbols = symbols or ['BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT', 'NEARUSDT']
    timeframes = timeframes or ['1', '5']
    builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
    builder.add_indicator('bb_main', bollinger_bands, period=33, std_dev=2.5)
    builder.add_indicator('rsi_main', rsi, period=12)
    builder.add_signal_rule('rsi_osob', overbought_oversold, indicator='rsi_main', overbought=81, oversold=15)
    builder.add_signal_rule('bb_reversion', bollinger_bands_signals, price='price', upper_band='upper_band', lower_band='lower_band')
    builder.add_signal_rule('rsi_divergence', divergence_signals, price='price', indicator='rsi_main', lookback_period=15)
    builder.set_signal_combination('and_signals')
    builder.set_strategy_info('Strategy_FAST_20260220_101234_05', '1.0.0')
    return builder.build()
