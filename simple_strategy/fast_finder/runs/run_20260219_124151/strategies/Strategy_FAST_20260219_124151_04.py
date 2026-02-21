"""Generated strategy: Strategy_FAST_20260219_124151_04"""
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import bollinger_bands, highest, lowest, rsi
from simple_strategy.strategies.signals_library import bollinger_bands_signals, breakout_signals, overbought_oversold

STRATEGY_PARAMETERS = {}

def create_strategy(symbols=None, timeframes=None, **params):
    symbols = symbols or ['BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT', 'NEARUSDT']
    timeframes = timeframes or ['1', '5']
    builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
    builder.add_indicator('bb_main', bollinger_bands, period=20, std_dev=1.6)
    builder.add_indicator('resistance_main', highest, period=58)
    builder.add_indicator('rsi_main', rsi, period=21)
    builder.add_indicator('support_main', lowest, period=25)
    builder.add_signal_rule('rsi_osob', overbought_oversold, indicator='rsi_main', overbought=70, oversold=28)
    builder.add_signal_rule('bb_reversion', bollinger_bands_signals, price='price', upper_band='upper_band', lower_band='lower_band')
    builder.add_signal_rule('breakout_sr', breakout_signals, price='price', resistance='resistance_main', support='support_main', penetration_pct=0.002)
    builder.set_signal_combination('and_signals')
    builder.set_strategy_info('Strategy_FAST_20260219_124151_04', '1.0.0')
    return builder.build()
