"""Generated strategy: Strategy_FAST_20260219_124151_08"""
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import bollinger_bands, ema, rsi, sma
from simple_strategy.strategies.signals_library import bollinger_bands_signals, ma_crossover, rsi_mean_reversion_with_trend

STRATEGY_PARAMETERS = {}

def create_strategy(symbols=None, timeframes=None, **params):
    symbols = symbols or ['BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT', 'NEARUSDT']
    timeframes = timeframes or ['1', '5']
    builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
    builder.add_indicator('bb_main', bollinger_bands, period=29, std_dev=2.9000000000000004)
    builder.add_indicator('ema_fast', ema, period=22)
    builder.add_indicator('ema_slow', ema, period=61)
    builder.add_indicator('rsi_main', rsi, period=20)
    builder.add_indicator('sma_fast', sma, period=20)
    builder.add_indicator('sma_slow', sma, period=42)
    builder.add_signal_rule('ema_cross', ma_crossover, fast_ma='ema_fast', slow_ma='ema_slow')
    builder.add_signal_rule('sma_cross', ma_crossover, fast_ma='sma_fast', slow_ma='sma_slow')
    builder.add_signal_rule('rsi_trend_combo', rsi_mean_reversion_with_trend, rsi='rsi_main', ema_fast='ema_fast', ema_slow='ema_slow', overbought=80, oversold=21)
    builder.add_signal_rule('bb_reversion', bollinger_bands_signals, price='price', upper_band='upper_band', lower_band='lower_band')
    builder.set_signal_combination('and_signals')
    builder.set_strategy_info('Strategy_FAST_20260219_124151_08', '1.0.0')
    return builder.build()
