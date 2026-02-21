"""Generated strategy: Strategy_FAST_20260219_124151_07"""
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import highest, lowest, macd, rsi
from simple_strategy.strategies.signals_library import breakout_signals, divergence_signals, macd_signals, overbought_oversold

STRATEGY_PARAMETERS = {}

def create_strategy(symbols=None, timeframes=None, **params):
    symbols = symbols or ['BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT', 'NEARUSDT']
    timeframes = timeframes or ['1', '5']
    builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
    builder.add_indicator('macd_main', macd, fast_period=9, slow_period=23, signal_period=8)
    builder.add_indicator('resistance_main', highest, period=37)
    builder.add_indicator('rsi_main', rsi, period=13)
    builder.add_indicator('support_main', lowest, period=52)
    builder.add_signal_rule('rsi_osob', overbought_oversold, indicator='rsi_main', overbought=76, oversold=17)
    builder.add_signal_rule('macd_cross', macd_signals, macd_line='macd_line', signal_line='signal_line')
    builder.add_signal_rule('breakout_sr', breakout_signals, price='price', resistance='resistance_main', support='support_main', penetration_pct=0.014000000000000002)
    builder.add_signal_rule('rsi_divergence', divergence_signals, price='price', indicator='rsi_main', lookback_period=39)
    builder.set_signal_combination('and_signals')
    builder.set_strategy_info('Strategy_FAST_20260219_124151_07', '1.0.0')
    return builder.build()
