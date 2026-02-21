"""Generated strategy: Strategy_FAST_20260219_143656_03"""
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import ema, rsi, sma, stochastic
from simple_strategy.strategies.signals_library import ma_crossover, overbought_oversold, stochastic_signals

STRATEGY_PARAMETERS = {}

def create_strategy(symbols=None, timeframes=None, **params):
    symbols = symbols or ['BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT', 'NEARUSDT']
    timeframes = timeframes or ['1', '5']
    builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
    builder.add_indicator('ema_fast', ema, period=38)
    builder.add_indicator('ema_slow', ema, period=117)
    builder.add_indicator('rsi_main', rsi, period=25)
    builder.add_indicator('sma_fast', sma, period=9)
    builder.add_indicator('sma_slow', sma, period=93)
    builder.add_indicator('stoch_main', stochastic, k_period=15, d_period=4)
    builder.add_signal_rule('rsi_osob', overbought_oversold, indicator='rsi_main', overbought=70, oversold=29)
    builder.add_signal_rule('ema_cross', ma_crossover, fast_ma='ema_fast', slow_ma='ema_slow')
    builder.add_signal_rule('sma_cross', ma_crossover, fast_ma='sma_fast', slow_ma='sma_slow')
    builder.add_signal_rule('stoch_reversal', stochastic_signals, k_percent='k_percent', d_percent='d_percent', overbought=78, oversold=24)
    builder.set_signal_combination('and_signals')
    builder.set_strategy_info('Strategy_FAST_20260219_143656_03', '1.0.0')
    return builder.build()
