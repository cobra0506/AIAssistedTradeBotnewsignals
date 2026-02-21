"""Generated strategy: Strategy_FAST_20260220_080627_03"""
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import ema, rsi
from simple_strategy.strategies.signals_library import overbought_oversold, rsi_mean_reversion_with_trend

STRATEGY_PARAMETERS = {}

def create_strategy(symbols=None, timeframes=None, **params):
    symbols = symbols or ['BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT', 'NEARUSDT']
    timeframes = timeframes or ['1', '5']
    builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
    builder.add_indicator('ema_fast', ema, period=35)
    builder.add_indicator('ema_slow', ema, period=66)
    builder.add_indicator('rsi_main', rsi, period=8)
    builder.add_signal_rule('rsi_osob', overbought_oversold, indicator='rsi_main', overbought=79, oversold=29)
    builder.add_signal_rule('rsi_trend_combo', rsi_mean_reversion_with_trend, rsi='rsi_main', ema_fast='ema_fast', ema_slow='ema_slow', overbought=79, oversold=35)
    builder.set_signal_combination('unanimous')
    builder.set_strategy_info('Strategy_FAST_20260220_080627_03', '1.0.0')
    return builder.build()
