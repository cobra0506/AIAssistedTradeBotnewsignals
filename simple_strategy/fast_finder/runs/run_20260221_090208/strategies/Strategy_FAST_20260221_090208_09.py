"""Generated strategy: Strategy_FAST_20260221_090208_09"""
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import bollinger_bands, macd, stochastic
from simple_strategy.strategies.signals_library import bollinger_bands_signals, macd_signals, stochastic_signals

STRATEGY_PARAMETERS = {}

def create_strategy(symbols=None, timeframes=None, **params):
    symbols = symbols or ['BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT', 'NEARUSDT']
    timeframes = timeframes or ['1', '5']
    builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
    builder.add_indicator('bb_main', bollinger_bands, period=39, std_dev=1.8)
    builder.add_indicator('macd_main', macd, fast_period=13, slow_period=49, signal_period=8)
    builder.add_indicator('stoch_main', stochastic, k_period=18, d_period=3)
    builder.add_signal_rule('macd_cross', macd_signals, macd_line='macd_line', signal_line='signal_line')
    builder.add_signal_rule('bb_reversion', bollinger_bands_signals, price='price', upper_band='upper_band', lower_band='lower_band')
    builder.add_signal_rule('stoch_reversal', stochastic_signals, k_percent='k_percent', d_percent='d_percent', overbought=82, oversold=16)
    builder.set_signal_combination('and_signals')
    builder.set_strategy_info('Strategy_FAST_20260221_090208_09', '1.0.0')
    return builder.build()
