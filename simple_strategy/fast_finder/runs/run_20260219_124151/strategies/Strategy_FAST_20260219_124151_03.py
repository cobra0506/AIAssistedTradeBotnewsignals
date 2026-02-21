"""Generated strategy: Strategy_FAST_20260219_124151_03"""
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import rsi, sma, stochastic
from simple_strategy.strategies.signals_library import divergence_signals, ma_crossover, stochastic_signals

STRATEGY_PARAMETERS = {}

def create_strategy(symbols=None, timeframes=None, **params):
    symbols = symbols or ['BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT', 'NEARUSDT']
    timeframes = timeframes or ['1', '5']
    builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
    builder.add_indicator('rsi_main', rsi, period=13)
    builder.add_indicator('sma_fast', sma, period=49)
    builder.add_indicator('sma_slow', sma, period=71)
    builder.add_indicator('stoch_main', stochastic, k_period=24, d_period=5)
    builder.add_signal_rule('sma_cross', ma_crossover, fast_ma='sma_fast', slow_ma='sma_slow')
    builder.add_signal_rule('stoch_reversal', stochastic_signals, k_percent='k_percent', d_percent='d_percent', overbought=82, oversold=17)
    builder.add_signal_rule('rsi_divergence', divergence_signals, price='price', indicator='rsi_main', lookback_period=12)
    builder.set_signal_combination('and_signals')
    builder.set_strategy_info('Strategy_FAST_20260219_124151_03', '1.0.0')
    return builder.build()
