"""
Simple Frequent Trading Strategy
=================================
A high-frequency trading strategy designed for testing purposes.
Generates frequent buy/sell signals using:
- Fast moving average crossovers
- RSI extremes with relaxed thresholds
- Small price movement detection
- Multiple timeframe confirmation

Strategy Logic:
1. Fast MA (3) crosses above Slow MA (8) = BUY signal
2. Fast MA crosses below Slow MA = SELL signal  
3. RSI < 40 (oversold) = Additional BUY signal
4. RSI > 60 (overbought) = Additional SELL signal
5. Price change > 0.1% = Directional signal

Best for: Testing bot functionality with high trade frequency
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
from simple_strategy.strategies.indicators_library import sma, rsi
from simple_strategy.strategies.signals_library import ma_crossover, overbought_oversold
from simple_strategy.shared.strategy_base import StrategyBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CRITICAL: STRATEGY_PARAMETERS for GUI Configuration
STRATEGY_PARAMETERS = {
    # Fast MA for entries
    'fast_ma_period': {
        'type': 'int',
        'default': 3,
        'min': 2,
        'max': 10,
        'description': 'Fast SMA period for entry signals',
        'gui_hint': 'Lower values = more frequent signals'
    },
    # Slow MA for entries
    'slow_ma_period': {
        'type': 'int', 
        'default': 8,
        'min': 5,
        'max': 20,
        'description': 'Slow SMA period for entry signals',
        'gui_hint': 'Higher values = smoother signals'
    },
    # RSI for confirmation
    'rsi_period': {
        'type': 'int',
        'default': 7,
        'min': 3,
        'max': 14,
        'description': 'RSI period for signal confirmation',
        'gui_hint': 'Lower values = more responsive'
    },
    # RSI levels for frequent signals
    'rsi_overbought': {
        'type': 'int',
        'default': 60,
        'min': 55,
        'max': 70,
        'description': 'RSI overbought threshold',
        'gui_hint': 'Lower = more sell signals'
    },
    'rsi_oversold': {
        'type': 'int',
        'default': 40,
        'min': 30,
        'max': 45,
        'description': 'RSI oversold threshold',
        'gui_hint': 'Higher = more buy signals'
    },
    # Price change threshold
    'price_change_threshold': {
        'type': 'float',
        'default': 0.001,
        'min': 0.0005,
        'max': 0.005,
        'description': 'Price change threshold for signals',
        'gui_hint': 'Lower = more frequent signals'
    }
}

def create_strategy(symbols=None, timeframes=None, **params):
    """CREATE STRATEGY FUNCTION - Required by GUI"""
    logger.info(f"🔧 create_strategy called with:")
    logger.info(f" - symbols: {symbols}")
    logger.info(f" - timeframes: {timeframes}")
    logger.info(f" - params: {params}")
    
    # Handle None values
    if symbols is None or len(symbols) == 0:
        symbols = ['BTCUSDT']
    
    if timeframes is None or len(timeframes) == 0:
        timeframes = ['1m']  # Use 1-minute for maximum frequency
    
    # Get parameters with defaults
    fast_ma_period = params.get('fast_ma_period', 3)
    slow_ma_period = params.get('slow_ma_period', 8)
    rsi_period = params.get('rsi_period', 7)
    rsi_overbought = params.get('rsi_overbought', 60)
    rsi_oversold = params.get('rsi_oversold', 40)
    price_change_threshold = params.get('price_change_threshold', 0.001)
    
    logger.info(f"🎯 Creating Frequent Trading strategy with parameters:")
    logger.info(f" - Symbols: {symbols}")
    logger.info(f" - Timeframes: {timeframes}")
    logger.info(f" - Fast MA: {fast_ma_period}, Slow MA: {slow_ma_period}")
    logger.info(f" - RSI: {rsi_period} (Overbought: {rsi_overbought}, Oversold: {rsi_oversold})")
    logger.info(f" - Price Change Threshold: {price_change_threshold}")
    
    try:
        strategy_builder = StrategyBuilder(symbols, timeframes)
        
        # Add indicators for each timeframe
        for timeframe in timeframes:
            # Moving averages for crossovers
            strategy_builder.add_indicator(f'fast_ma_{timeframe}', sma, period=fast_ma_period)
            strategy_builder.add_indicator(f'slow_ma_{timeframe}', sma, period=slow_ma_period)
            
            # RSI for confirmation
            strategy_builder.add_indicator(f'rsi_{timeframe}', rsi, period=rsi_period)
        
        # Add signal rules
        strategy_builder.add_signal_rule('ma_crossover', ma_crossover,
                                      fast_ma=f'fast_ma_1m',
                                      slow_ma=f'slow_ma_1m')
        
        strategy_builder.add_signal_rule('rsi_confirmation', overbought_oversold,
                                      indicator=f'rsi_1m',
                                      overbought=rsi_overbought,
                                      oversold=rsi_oversold)
        
        # Use majority vote for signal combination (generates more signals)
        strategy_builder.set_signal_combination('majority_vote')
        
        # Set strategy info
        strategy_builder.set_strategy_info('Simple_Frequent_Trading', '1.0.0')
        
        # Build and return strategy
        strategy = strategy_builder.build()
        logger.info(f"✅ Simple Frequent Trading strategy created successfully!")
        return strategy
        
    except Exception as e:
        logger.error(f"❌ Error creating Simple Frequent Trading strategy: {e}")
        raise


class SimpleFrequentTradingStrategy(StrategyBase):
    """Simple Frequent Trading Strategy Class"""
    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        super().__init__(
            name="Simple_Frequent_Trading",
            symbols=symbols,
            timeframes=timeframes,
            config=config
        )
        
        self.fast_ma_period = config.get('fast_ma_period', 3)
        self.slow_ma_period = config.get('slow_ma_period', 8)
        self.rsi_period = config.get('rsi_period', 7)
        self.rsi_overbought = config.get('rsi_overbought', 60)
        self.rsi_oversold = config.get('rsi_oversold', 40)
        self.price_change_threshold = config.get('price_change_threshold', 0.001)
        
        self._validate_parameters()
        logger.info(f"📈 SimpleFrequentTradingStrategy initialized")
    
    def _validate_parameters(self):
        """Validate strategy parameters"""
        if self.fast_ma_period >= self.slow_ma_period:
            raise ValueError("Fast MA period must be less than Slow MA period")
        
        if self.rsi_overbought <= self.rsi_oversold:
            raise ValueError("RSI overbought must be greater than RSI oversold")
        
        if self.price_change_threshold <= 0:
            raise ValueError("Price change threshold must be positive")
    
    def generate_signals(self, data: Dict) -> pd.DataFrame:
        """Generate trading signals"""
        # This method would contain custom signal generation logic
        # For now, we rely on the StrategyBuilder implementation
        pass