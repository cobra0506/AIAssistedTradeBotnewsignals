"""Generated strategy: Strategy_AE_Published_publish_smoke_manual_01"""
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import ema
from simple_strategy.strategies.signals_library import ma_crossover
from simple_strategy.strategies.builder_presets import entry_timeframe_filter

STRATEGY_PARAMETERS = {}
AUTO_EVOLVE_RISK_SETTINGS = {'risk_exit_mode': 'none', 'risk_sl_pct': 0.02, 'risk_atr_period': 14, 'risk_atr_sl_multiplier': 1.3, 'risk_atr_tp_multiplier': 1.7, 'position_size_pct': 0.05}

def create_strategy(symbols=None, timeframes=None, **params):
    symbols = symbols or ['BTCUSDT']
    timeframes = timeframes or ['5m']
    builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
    builder.add_indicator('ema_fast', ema, period=9)
    builder.add_indicator('ema_slow', ema, period=21)
    builder.add_signal_rule('ema_cross', ma_crossover, fast_ma='ema_fast', slow_ma='ema_slow')
    builder.add_filter_rule('entry_only', entry_timeframe_filter, entry_timeframe='5m')
    builder.set_signal_combination('majority_vote')
    builder.set_strategy_info('Strategy_AE_Published_publish_smoke_manual_01', '1.0.0')
    return builder.build()
