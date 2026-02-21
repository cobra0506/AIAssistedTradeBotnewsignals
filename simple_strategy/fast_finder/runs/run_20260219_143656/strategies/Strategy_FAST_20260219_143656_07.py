"""Generated strategy: Strategy_FAST_20260219_143656_07"""
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import bollinger_bands, ema, macd, sma
from simple_strategy.strategies.signals_library import bollinger_bands_signals, ma_crossover, macd_signals

STRATEGY_PARAMETERS = {}

def create_strategy(symbols=None, timeframes=None, **params):
    symbols = symbols or ['BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT', 'NEARUSDT']
    timeframes = timeframes or ['1', '5']
    builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
    builder.add_indicator('bb_main', bollinger_bands, period=30, std_dev=2.4)
    builder.add_indicator('ema_fast', ema, period=10)
    builder.add_indicator('ema_slow', ema, period=96)
    builder.add_indicator('macd_main', macd, fast_period=16, slow_period=26, signal_period=12)
    builder.add_indicator('sma_fast', sma, period=19)
    builder.add_indicator('sma_slow', sma, period=152)
    builder.add_signal_rule('ema_cross', ma_crossover, fast_ma='ema_fast', slow_ma='ema_slow')
    builder.add_signal_rule('sma_cross', ma_crossover, fast_ma='sma_fast', slow_ma='sma_slow')
    builder.add_signal_rule('macd_cross', macd_signals, macd_line='macd_line', signal_line='signal_line')
    builder.add_signal_rule('bb_reversion', bollinger_bands_signals, price='price', upper_band='upper_band', lower_band='lower_band')
    builder.set_signal_combination('and_signals')
    builder.set_strategy_info('Strategy_FAST_20260219_143656_07', '1.0.0')
    return builder.build()
