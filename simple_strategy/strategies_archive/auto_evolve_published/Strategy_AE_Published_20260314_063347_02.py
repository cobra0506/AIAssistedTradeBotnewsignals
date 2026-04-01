"""Generated strategy: Strategy_AE_Published_20260314_063347_02"""
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import rsi, sma
from simple_strategy.strategies.signals_library import ma_crossover, overbought_oversold
from simple_strategy.strategies.builder_presets import entry_timeframe_filter, volume_filter

STRATEGY_PARAMETERS = {}
AUTO_EVOLVE_RISK_SETTINGS = {'risk_exit_mode': 'fixed', 'risk_sl_pct': 0.015, 'risk_atr_period': 19, 'risk_atr_sl_multiplier': 2.0, 'risk_atr_tp_multiplier': 2.7, 'position_size_pct': 0.12000000000000001}

def create_strategy(symbols=None, timeframes=None, **params):
    symbols = symbols or ['BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT', 'ATOMUSDT', 'ALGOUSDT', 'VETUSDT', 'ICPUSDT', 'FILUSDT', 'AAVEUSDT', 'COMPUSDT', 'CRVUSDT', 'SNXUSDT', 'SUSHIUSDT', 'ARBUSDT', 'OPUSDT', 'NEARUSDT', 'GRTUSDT']
    timeframes = timeframes or ['1m']
    builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
    builder.add_indicator('rsi_main', rsi, period=9)
    builder.add_indicator('sma_fast', sma, period=19)
    builder.add_indicator('sma_slow', sma, period=94)
    builder.add_signal_rule('rsi_osob', overbought_oversold, indicator='rsi_main', overbought=67, oversold=22)
    builder.add_signal_rule('sma_cross', ma_crossover, fast_ma='sma_fast', slow_ma='sma_slow')
    builder.add_filter_rule('entry_only', entry_timeframe_filter, entry_timeframe='15m')
    builder.add_filter_rule('volume_confirm', volume_filter, entry_timeframe='15m', volume_sma_period=27, volume_multiplier=1.1)
    builder.set_signal_combination('unanimous')
    builder.set_strategy_info('Strategy_AE_Published_20260314_063347_02', '1.0.0')
    return builder.build()
