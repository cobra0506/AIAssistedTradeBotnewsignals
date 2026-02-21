"""Generated strategy: Strategy_FAST_20260219_124151_10"""
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import macd, rsi
from simple_strategy.strategies.signals_library import macd_signals, overbought_oversold

STRATEGY_PARAMETERS = {}

def create_strategy(symbols=None, timeframes=None, **params):
    symbols = symbols or ['BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT', 'NEARUSDT']
    timeframes = timeframes or ['1', '5']
    builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
    builder.add_indicator('macd_main', macd, fast_period=9, slow_period=29, signal_period=7)
    builder.add_indicator('rsi_main', rsi, period=15)
    builder.add_signal_rule('rsi_osob', overbought_oversold, indicator='rsi_main', overbought=69, oversold=23)
    builder.add_signal_rule('macd_cross', macd_signals, macd_line='macd_line', signal_line='signal_line')
    builder.set_signal_combination('and_signals')
    builder.set_strategy_info('Strategy_FAST_20260219_124151_10', '1.0.0')
    return builder.build()
