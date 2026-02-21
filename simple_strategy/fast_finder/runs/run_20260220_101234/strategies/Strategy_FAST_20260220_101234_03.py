"""Generated strategy: Strategy_FAST_20260220_101234_03"""
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import ema, rsi, sma, stochastic
from simple_strategy.strategies.signals_library import ma_crossover, rsi_mean_reversion_with_trend, stochastic_signals

STRATEGY_PARAMETERS = {}

def create_strategy(symbols=None, timeframes=None, **params):
    symbols = symbols or ['BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT', 'NEARUSDT']
    timeframes = timeframes or ['1', '5']
    builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
    builder.add_indicator('ema_fast', ema, period=30)
    builder.add_indicator('ema_slow', ema, period=44)
    builder.add_indicator('rsi_main', rsi, period=28)
    builder.add_indicator('sma_fast', sma, period=21)
    builder.add_indicator('sma_slow', sma, period=191)
    builder.add_indicator('stoch_main', stochastic, k_period=18, d_period=5)
    builder.add_signal_rule('ema_cross', ma_crossover, fast_ma='ema_fast', slow_ma='ema_slow')
    builder.add_signal_rule('sma_cross', ma_crossover, fast_ma='sma_fast', slow_ma='sma_slow')
    builder.add_signal_rule('rsi_trend_combo', rsi_mean_reversion_with_trend, rsi='rsi_main', ema_fast='ema_fast', ema_slow='ema_slow', overbought=83, oversold=22)
    builder.add_signal_rule('stoch_reversal', stochastic_signals, k_percent='k_percent', d_percent='d_percent', overbought=80, oversold=11)
    builder.set_signal_combination('majority_vote')
    builder.set_strategy_info('Strategy_FAST_20260220_101234_03', '1.0.0')
    return builder.build()
