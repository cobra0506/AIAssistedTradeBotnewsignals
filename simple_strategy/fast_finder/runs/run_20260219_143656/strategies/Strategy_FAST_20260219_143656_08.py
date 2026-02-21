"""Generated strategy: Strategy_FAST_20260219_143656_08"""
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import highest, lowest, macd, rsi, stochastic
from simple_strategy.strategies.signals_library import breakout_signals, divergence_signals, macd_signals, stochastic_signals

STRATEGY_PARAMETERS = {}

def create_strategy(symbols=None, timeframes=None, **params):
    symbols = symbols or ['BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT', 'NEARUSDT']
    timeframes = timeframes or ['1', '5']
    builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
    builder.add_indicator('macd_main', macd, fast_period=17, slow_period=35, signal_period=4)
    builder.add_indicator('resistance_main', highest, period=37)
    builder.add_indicator('rsi_main', rsi, period=27)
    builder.add_indicator('stoch_main', stochastic, k_period=9, d_period=6)
    builder.add_indicator('support_main', lowest, period=16)
    builder.add_signal_rule('macd_cross', macd_signals, macd_line='macd_line', signal_line='signal_line')
    builder.add_signal_rule('stoch_reversal', stochastic_signals, k_percent='k_percent', d_percent='d_percent', overbought=78, oversold=20)
    builder.add_signal_rule('breakout_sr', breakout_signals, price='price', resistance='resistance_main', support='support_main', penetration_pct=0.014000000000000002)
    builder.add_signal_rule('rsi_divergence', divergence_signals, price='price', indicator='rsi_main', lookback_period=11)
    builder.set_signal_combination('and_signals')
    builder.set_strategy_info('Strategy_FAST_20260219_143656_08', '1.0.0')
    return builder.build()
