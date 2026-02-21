"""Generated strategy: Strategy_FAST_20260221_090208_05"""
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import ema, sma, stochastic
from simple_strategy.strategies.signals_library import ma_crossover, stochastic_signals

STRATEGY_PARAMETERS = {}

def create_strategy(symbols=None, timeframes=None, **params):
    symbols = symbols or ['BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT', 'NEARUSDT']
    timeframes = timeframes or ['1', '5']
    builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
    builder.add_indicator('ema_fast', ema, period=40)
    builder.add_indicator('ema_slow', ema, period=41)
    builder.add_indicator('sma_fast', sma, period=43)
    builder.add_indicator('sma_slow', sma, period=199)
    builder.add_indicator('stoch_main', stochastic, k_period=22, d_period=5)
    builder.add_signal_rule('ema_cross', ma_crossover, fast_ma='ema_fast', slow_ma='ema_slow')
    builder.add_signal_rule('sma_cross', ma_crossover, fast_ma='sma_fast', slow_ma='sma_slow')
    builder.add_signal_rule('stoch_reversal', stochastic_signals, k_percent='k_percent', d_percent='d_percent', overbought=82, oversold=13)
    builder.set_signal_combination('majority_vote')
    builder.set_strategy_info('Strategy_FAST_20260221_090208_05', '1.0.0')
    return builder.build()
