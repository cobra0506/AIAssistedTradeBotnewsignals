"""Generated strategy: Strategy_AE_Published_20260318_191300_01"""
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import sma
from simple_strategy.strategies.signals_library import ma_crossover
from simple_strategy.strategies.builder_presets import price_change_bias_filter

STRATEGY_PARAMETERS = {}
AUTO_EVOLVE_RISK_SETTINGS = {'risk_exit_mode': 'none', 'risk_sl_pct': 0.01, 'risk_atr_period': 17, 'risk_atr_sl_multiplier': 1.1, 'risk_atr_tp_multiplier': 3.2, 'position_size_pct': 0.07}

def create_strategy(symbols=None, timeframes=None, **params):
    symbols = symbols or ['BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT', 'ATOMUSDT', 'ALGOUSDT', 'VETUSDT', 'ICPUSDT', 'FILUSDT', 'AAVEUSDT', 'COMPUSDT', 'CRVUSDT', 'SNXUSDT', 'SUSHIUSDT', 'ARBUSDT', 'OPUSDT', 'NEARUSDT', 'GRTUSDT']
    timeframes = timeframes or ['1m']
    builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
    builder.add_indicator('sma_fast', sma, period=19)
    builder.add_indicator('sma_slow', sma, period=149)
    builder.add_signal_rule('sma_cross', ma_crossover, fast_ma='sma_fast', slow_ma='sma_slow')
    builder.add_filter_rule('price_bias', price_change_bias_filter, entry_timeframe='5m', price_change_threshold=0.011)
    builder.set_signal_combination('majority_vote')
    builder.set_strategy_info('Strategy_AE_Published_20260318_191300_01', '1.0.0')
    return builder.build()
