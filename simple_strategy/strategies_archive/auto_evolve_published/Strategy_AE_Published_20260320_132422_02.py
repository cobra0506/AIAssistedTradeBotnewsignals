"""Generated strategy: Strategy_AE_Published_20260320_132422_02"""
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import rsi
from simple_strategy.strategies.signals_library import overbought_oversold

STRATEGY_PARAMETERS = {}
AUTO_EVOLVE_RISK_SETTINGS = {'risk_exit_mode': 'trailing', 'risk_sl_pct': 0.030000000000000002, 'risk_atr_period': 26, 'risk_atr_sl_multiplier': 2.7, 'risk_atr_tp_multiplier': 4.0, 'position_size_pct': 0.07}

def create_strategy(symbols=None, timeframes=None, **params):
    symbols = symbols or ['BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT', 'ATOMUSDT', 'ALGOUSDT', 'VETUSDT', 'ICPUSDT', 'FILUSDT', 'AAVEUSDT', 'COMPUSDT', 'CRVUSDT', 'SNXUSDT', 'SUSHIUSDT', 'ARBUSDT', 'OPUSDT', 'NEARUSDT', 'GRTUSDT']
    timeframes = timeframes or ['1m']
    builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
    builder.add_indicator('rsi_main', rsi, period=24)
    builder.add_signal_rule('rsi_osob', overbought_oversold, indicator='rsi_main', overbought=70, oversold=39)
    builder.set_signal_combination('majority_vote')
    builder.set_strategy_info('Strategy_AE_Published_20260320_132422_02', '1.0.0')
    return builder.build()
