"""Generated strategy: Strategy_FAST_20260221_090208_07"""
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import ema, macd, rsi, sma
from simple_strategy.strategies.signals_library import ma_crossover, macd_signals, rsi_mean_reversion_with_trend

STRATEGY_PARAMETERS = {}

def create_strategy(symbols=None, timeframes=None, **params):
    symbols = symbols or ['BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT', 'NEARUSDT']
    timeframes = timeframes or ['1', '5']
    builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
    builder.add_indicator('ema_fast', ema, period=31)
    builder.add_indicator('ema_slow', ema, period=82)
    builder.add_indicator('macd_main', macd, fast_period=6, slow_period=20, signal_period=6)
    builder.add_indicator('rsi_main', rsi, period=17)
    builder.add_indicator('sma_fast', sma, period=27)
    builder.add_indicator('sma_slow', sma, period=55)
    builder.add_signal_rule('ema_cross', ma_crossover, fast_ma='ema_fast', slow_ma='ema_slow')
    builder.add_signal_rule('sma_cross', ma_crossover, fast_ma='sma_fast', slow_ma='sma_slow')
    builder.add_signal_rule('rsi_trend_combo', rsi_mean_reversion_with_trend, rsi='rsi_main', ema_fast='ema_fast', ema_slow='ema_slow', overbought=66, oversold=20)
    builder.add_signal_rule('macd_cross', macd_signals, macd_line='macd_line', signal_line='signal_line')
    builder.set_signal_combination('and_signals')
    builder.set_strategy_info('Strategy_FAST_20260221_090208_07', '1.0.0')
    return builder.build()
