"""Generated strategy: Strategy_FAST_20260219_143656_10"""
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import ema, highest, lowest, rsi, stochastic
from simple_strategy.strategies.signals_library import breakout_signals, ma_crossover, overbought_oversold, stochastic_signals

STRATEGY_PARAMETERS = {}

def create_strategy(symbols=None, timeframes=None, **params):
    symbols = symbols or ['BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT', 'NEARUSDT']
    timeframes = timeframes or ['1', '5']
    builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
    builder.add_indicator('ema_fast', ema, period=12)
    builder.add_indicator('ema_slow', ema, period=97)
    builder.add_indicator('resistance_main', highest, period=31)
    builder.add_indicator('rsi_main', rsi, period=7)
    builder.add_indicator('stoch_main', stochastic, k_period=23, d_period=3)
    builder.add_indicator('support_main', lowest, period=35)
    builder.add_signal_rule('rsi_osob', overbought_oversold, indicator='rsi_main', overbought=66, oversold=36)
    builder.add_signal_rule('ema_cross', ma_crossover, fast_ma='ema_fast', slow_ma='ema_slow')
    builder.add_signal_rule('stoch_reversal', stochastic_signals, k_percent='k_percent', d_percent='d_percent', overbought=88, oversold=20)
    builder.add_signal_rule('breakout_sr', breakout_signals, price='price', resistance='resistance_main', support='support_main', penetration_pct=0.018000000000000002)
    builder.set_signal_combination('and_signals')
    builder.set_strategy_info('Strategy_FAST_20260219_143656_10', '1.0.0')
    return builder.build()
