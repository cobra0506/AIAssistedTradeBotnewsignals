"""Generated strategy: Strategy_FAST_20260219_124151_09"""
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import bollinger_bands, rsi, stochastic
from simple_strategy.strategies.signals_library import bollinger_bands_signals, divergence_signals, overbought_oversold, stochastic_signals

STRATEGY_PARAMETERS = {}

def create_strategy(symbols=None, timeframes=None, **params):
    symbols = symbols or ['BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT', 'NEARUSDT']
    timeframes = timeframes or ['1', '5']
    builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
    builder.add_indicator('bb_main', bollinger_bands, period=33, std_dev=2.6)
    builder.add_indicator('rsi_main', rsi, period=21)
    builder.add_indicator('stoch_main', stochastic, k_period=23, d_period=6)
    builder.add_signal_rule('rsi_osob', overbought_oversold, indicator='rsi_main', overbought=78, oversold=38)
    builder.add_signal_rule('bb_reversion', bollinger_bands_signals, price='price', upper_band='upper_band', lower_band='lower_band')
    builder.add_signal_rule('stoch_reversal', stochastic_signals, k_percent='k_percent', d_percent='d_percent', overbought=86, oversold=24)
    builder.add_signal_rule('rsi_divergence', divergence_signals, price='price', indicator='rsi_main', lookback_period=27)
    builder.set_signal_combination('and_signals')
    builder.set_strategy_info('Strategy_FAST_20260219_124151_09', '1.0.0')
    return builder.build()
