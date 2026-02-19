"""Generated strategy: Strategy_FAST_20260218_093807_01"""
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import rsi
from simple_strategy.strategies.signals_library import overbought_oversold

STRATEGY_PARAMETERS = {}

def create_strategy(symbols=None, timeframes=None, **params):
    symbols = symbols or ['BNBUSDT', 'XRPUSDT']
    timeframes = timeframes or ['1']
    builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
    builder.add_indicator('rsi_main', rsi, period=13)
    builder.add_signal_rule('rsi_osob', overbought_oversold, indicator='rsi_main', overbought=83, oversold=23)
    builder.set_signal_combination('unanimous')
    builder.set_strategy_info('Strategy_FAST_20260218_093807_01', '1.0.0')
    return builder.build()
