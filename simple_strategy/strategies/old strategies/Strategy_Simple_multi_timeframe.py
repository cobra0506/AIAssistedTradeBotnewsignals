"""
Simple Multi-Timeframe Strategy
==============================

Uses 1m for entries and 5m for trend confirmation:
- 1m: Fast EMA crossover for entry signals
- 5m: Slow EMA for trend direction
- Simple logic that actually works

Strategy Logic:
1. TREND (5m): Price above 50 EMA = Bullish trend
2. ENTRY (1m): Fast EMA crosses above Slow EMA = BUY
3. EXIT (1m): Fast EMA crosses below Slow EMA = SELL

Best for: Trending markets with multiple timeframe confirmation
Author: AI Assisted TradeBot Team
Date: 2025
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional

# Add parent directories to path for proper imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import required components
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import ema
from simple_strategy.strategies.signals_library import ma_crossover
from simple_strategy.shared.strategy_base import StrategyBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CRITICAL: STRATEGY_PARAMETERS for GUI Configuration
STRATEGY_PARAMETERS = {
    # Fast EMA for entries (1m)
    'fast_ema_period': {
        'type': 'int',
        'default': 9,
        'min': 5,
        'max': 20,
        'description': 'Fast EMA period for entry signals',
        'gui_hint': 'For 1m timeframe. Lower = more sensitive'
    },
    # Slow EMA for entries (1m)
    'slow_ema_period': {
        'type': 'int',
        'default': 21,
        'min': 15,
        'max': 50,
        'description': 'Slow EMA period for entry signals',
        'gui_hint': 'For 1m timeframe. Higher = smoother'
    },
    # Trend EMA for trend confirmation (5m)
    'trend_ema_period': {
        'type': 'int',
        'default': 50,
        'min': 20,
        'max': 100,
        'description': 'Trend EMA period for higher timeframe',
        'gui_hint': 'For 5m timeframe. Higher = longer term trend'
    }
}

def create_strategy(symbols=None, timeframes=None, **params):
    """CREATE STRATEGY FUNCTION - Required by GUI"""
    logger.info(f"🔧 create_strategy called with:")
    logger.info(f" - symbols: {symbols}")
    logger.info(f" - timeframes: {timeframes}")
    logger.info(f" - params: {params}")
    
    if symbols is None or len(symbols) == 0:
        symbols = ['BTCUSDT']
    
    # CRITICAL: Ensure we have both timeframes
    if timeframes is None or len(timeframes) == 0:
        timeframes = ['1m', '5m']
    else:
        # Make sure we have 1m and 5m
        if '1m' not in timeframes:
            timeframes.append('1m')
        if '5m' not in timeframes:
            timeframes.append('5m')
    
    # Get parameters
    fast_ema_period = params.get('fast_ema_period', 9)
    slow_ema_period = params.get('slow_ema_period', 21)
    trend_ema_period = params.get('trend_ema_period', 50)
    
    logger.info(f"🎯 Creating Simple Multi-Timeframe strategy with parameters:")
    logger.info(f" - Symbols: {symbols}")
    logger.info(f" - Timeframes: {timeframes}")
    logger.info(f" - Fast EMA: {fast_ema_period}, Slow EMA: {slow_ema_period}")
    logger.info(f" - Trend EMA: {trend_ema_period}")
    
    try:
        strategy_builder = StrategyBuilder(symbols, timeframes)
        
        # Add indicators for each timeframe
        for timeframe in timeframes:
            if timeframe == '1m':
                # Entry indicators for 1m
                strategy_builder.add_indicator(f'ema_fast_1m', ema, period=fast_ema_period)
                strategy_builder.add_indicator(f'ema_slow_1m', ema, period=slow_ema_period)
            elif timeframe == '5m':
                # Trend indicator for 5m
                strategy_builder.add_indicator(f'ema_trend_5m', ema, period=trend_ema_period)
        
        # Add signal rule for 1m timeframe
        strategy_builder.add_signal_rule('ema_crossover', ma_crossover,
                                       fast_ma='ema_fast_1m',
                                       slow_ma='ema_slow_1m')
        
        strategy_builder.set_signal_combination('majority_vote')
        strategy_builder.set_strategy_info('Simple_Multi_Timeframe', '1.0.0')
        
        strategy = strategy_builder.build()
        
        logger.info(f"✅ Simple Multi-Timeframe strategy created successfully!")
        return strategy
        
    except Exception as e:
        logger.error(f"❌ Error creating Simple Multi-Timeframe strategy: {e}")
        raise

class SimpleMultiTimeframeStrategy(StrategyBase):
    """Simple Multi-Timeframe Strategy Class"""
    
    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        super().__init__(
            name="Simple_Multi_Timeframe",
            symbols=symbols,
            timeframes=timeframes,
            config=config
        )
        
        self.fast_ema_period = config.get('fast_ema_period', 9)
        self.slow_ema_period = config.get('slow_ema_period', 21)
        self.trend_ema_period = config.get('trend_ema_period', 50)
        
        self._validate_parameters()
        logger.info(f"📈 SimpleMultiTimeframeStrategy initialized")
    
    def _validate_parameters(self):
        if self.fast_ema_period >= self.slow_ema_period:
            raise ValueError("Fast EMA period must be less than slow EMA period")
    
    def calculate_position_size(self, symbol: str, current_price: float = None, signal_strength: float = 1.0) -> float:
        return 0.001  # Fixed small position
    
    def generate_signals(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, str]]:
        signals = {}
        
        try:
            for symbol in data:
                signals[symbol] = {}
                
                for timeframe in data[symbol]:
                    signal = self._generate_single_signal(data[symbol][timeframe], symbol, timeframe, data)
                    signals[symbol][timeframe] = signal
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return signals
    
    def _generate_single_signal(self, df: pd.DataFrame, symbol: str, timeframe: str, all_data: Dict) -> str:
        try:
            # Only generate signals for 1m timeframe
            if timeframe != '1m':
                return 'HOLD'
            
            if len(df) < self.slow_ema_period:
                return 'HOLD'
            
            # Get 1m indicators
            ema_fast = df['ema_fast_1m'].iloc[-1]
            ema_slow = df['ema_slow_1m'].iloc[-1]
            
            # Get 5m trend
            if '5m' in all_data.get(symbol, {}) and len(all_data[symbol]['5m']) >= self.trend_ema_period:
                trend_ema = all_data[symbol]['5m']['ema_trend_5m'].iloc[-1]
                current_price_5m = all_data[symbol]['5m']['close'].iloc[-1]
                
                # Determine trend
                bullish_trend = current_price_5m > trend_ema
            else:
                # If no 5m data, no trend filter
                bullish_trend = True
            
            # Simple logic
            if ema_fast > ema_slow and bullish_trend:
                return 'BUY'
            elif ema_fast < ema_slow and not bullish_trend:
                return 'SELL'
            
            return 'HOLD'
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol} {timeframe}: {e}")
            return 'HOLD'

def create_simple_multi_timeframe_instance(symbols=None, timeframes=None, **params):
    try:
        if symbols is None:
            symbols = ['BTCUSDT']
        if timeframes is None:
            timeframes = ['1m', '5m']
        
        strategy = SimpleMultiTimeframeStrategy(symbols, timeframes, params)
        logger.info(f"✅ Simple Multi-Timeframe strategy created successfully")
        return strategy
    except Exception as e:
        logger.error(f"Error creating strategy: {e}")
        raise

def simple_test():
    try:
        strategy = create_strategy(
            symbols=['BTCUSDT'],
            timeframes=['1m', '5m'],
            fast_ema_period=9,
            slow_ema_period=21,
            trend_ema_period=50
        )
        
        print(f"✅ Simple Multi-Timeframe strategy created successfully: {strategy.name}")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Simple Multi-Timeframe strategy: {e}")
        return False

if __name__ == "__main__":
    simple_test()