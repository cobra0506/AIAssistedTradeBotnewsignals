"""Generated strategy: Strategy_AE_Published_20260315_083757_03"""
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import ema
from simple_strategy.strategies.signals_library import ma_crossover
from simple_strategy.strategies.builder_presets import rsi_gate_filter, volume_filter

STRATEGY_PARAMETERS = {}
AUTO_EVOLVE_RISK_SETTINGS = {'risk_exit_mode': 'none', 'risk_sl_pct': 0.01, 'risk_atr_period': 9, 'risk_atr_sl_multiplier': 2.8, 'risk_atr_tp_multiplier': 2.0, 'position_size_pct': 0.12000000000000001}

def create_strategy(symbols=None, timeframes=None, **params):
    symbols = symbols or ['BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT', 'ATOMUSDT', 'ALGOUSDT', 'VETUSDT', 'ICPUSDT', 'FILUSDT', 'AAVEUSDT', 'COMPUSDT', 'CRVUSDT', 'SNXUSDT', 'SUSHIUSDT', 'ARBUSDT', 'OPUSDT', 'NEARUSDT', 'GRTUSDT']
    timeframes = timeframes or ['1m']
    builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
    builder.add_indicator('ema_fast', ema, period=25)
    builder.add_indicator('ema_slow', ema, period=47)
    builder.add_signal_rule('ema_cross', ma_crossover, fast_ma='ema_fast', slow_ma='ema_slow')
    builder.add_filter_rule('rsi_gate', rsi_gate_filter, entry_timeframe='5m', rsi_period=14, bullish_threshold=57.0, bearish_threshold=58.0)
    builder.add_filter_rule('volume_confirm', volume_filter, entry_timeframe='15m', volume_sma_period=32, volume_multiplier=1.7000000000000002)
    builder.set_signal_combination('majority_vote')
    builder.set_strategy_info('Strategy_AE_Published_20260315_083757_03', '1.0.0')
    return builder.build()
