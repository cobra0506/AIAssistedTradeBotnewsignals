"""Generated strategy: Strategy_FAST_20260221_090208_08"""
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import macd, rsi, sma, stochastic
from simple_strategy.strategies.signals_library import ma_crossover, macd_signals, overbought_oversold, stochastic_signals

STRATEGY_PARAMETERS = {}

def create_strategy(symbols=None, timeframes=None, **params):
    symbols = symbols or ['BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT', 'NEARUSDT']
    timeframes = timeframes or ['1', '5']
    builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
    builder.add_indicator('macd_main', macd, fast_period=6, slow_period=44, signal_period=10)
    builder.add_indicator('rsi_main', rsi, period=15)
    builder.add_indicator('sma_fast', sma, period=46)
    builder.add_indicator('sma_slow', sma, period=71)
    builder.add_indicator('stoch_main', stochastic, k_period=8, d_period=4)
    builder.add_signal_rule('rsi_osob', overbought_oversold, indicator='rsi_main', overbought=65, oversold=19)
    builder.add_signal_rule('sma_cross', ma_crossover, fast_ma='sma_fast', slow_ma='sma_slow')
    builder.add_signal_rule('macd_cross', macd_signals, macd_line='macd_line', signal_line='signal_line')
    builder.add_signal_rule('stoch_reversal', stochastic_signals, k_percent='k_percent', d_percent='d_percent', overbought=86, oversold=18)
    builder.set_signal_combination('and_signals')
    builder.set_strategy_info('Strategy_FAST_20260221_090208_08', '1.0.0')
    return builder.build()
