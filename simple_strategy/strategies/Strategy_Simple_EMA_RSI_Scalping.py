"""
RSI Mean Reversion with EMA Trend Filter Strategy
===============================================

A trend-filtered RSI mean reversion strategy that:
- Uses EMA crossover to determine trend direction
- Uses RSI crossovers for precise entry/exit signals
- Only trades in the direction of the prevailing trend
- Implements proper OPEN/CLOSE signal logic

Strategy Logic:
1. UPTREND (Fast EMA > Slow EMA):
   - OPEN_LONG when RSI crosses UP through oversold level
   - CLOSE_LONG when RSI crosses DOWN through overbought level

2. DOWNTREND (Fast EMA < Slow EMA):
   - OPEN_SHORT when RSI crosses DOWN through overbought level
   - CLOSE_SHORT when RSI crosses UP through oversold level

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
from simple_strategy.strategies.indicators_library import ema, rsi
from simple_strategy.strategies.signals_library import rsi_mean_reversion_with_trend
from simple_strategy.shared.strategy_base import StrategyBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CRITICAL: STRATEGY_PARAMETERS for GUI Configuration
STRATEGY_PARAMETERS = {
    # Fast EMA for trend direction
    'fast_ema_period': {
        'type': 'int',
        'default': 20,
        'min': 5,
        'max': 50,
        'description': 'Fast EMA period for trend direction',
        'gui_hint': 'Lower values = more sensitive trend. Recommended: 15-25'
    },
    # Slow EMA for trend direction
    'slow_ema_period': {
        'type': 'int',
        'default': 50,
        'min': 20,
        'max': 100,
        'description': 'Slow EMA period for trend direction',
        'gui_hint': 'Higher values = smoother trend. Recommended: 40-60'
    },
    # RSI for entry/exit signals
    'rsi_period': {
        'type': 'int',
        'default': 14,
        'min': 7,
        'max': 21,
        'description': 'RSI period for signals',
        'gui_hint': 'Standard values: 14, 10 for faster signals'
    },
    # RSI levels
    'rsi_overbought': {
        'type': 'int',
        'default': 70,
        'min': 65,
        'max': 80,
        'description': 'RSI overbought level',
        'gui_hint': 'Level where RSI is considered overbought'
    },
    'rsi_oversold': {
        'type': 'int',
        'default': 30,
        'min': 20,
        'max': 35,
        'description': 'RSI oversold level',
        'gui_hint': 'Level where RSI is considered oversold'
    }
}

def create_strategy(symbols=None, timeframes=None, **params):
    """
    CREATE STRATEGY FUNCTION - Required by GUI
    """
    # DEBUG: Log what we receive
    logger.info(f"🔧 create_strategy called with:")
    logger.info(f" - symbols: {symbols}")
    logger.info(f" - timeframes: {timeframes}")
    logger.info(f" - params: {params}")
    
    # CRITICAL: Handle None/empty values with defaults
    if symbols is None or len(symbols) == 0:
        logger.warning("⚠️ No symbols provided, using default: ['BTCUSDT']")
        symbols = ['BTCUSDT']
    
    if timeframes is None or len(timeframes) == 0:
        logger.warning("⚠️ No timeframes provided, using default: ['5m']")
        timeframes = ['5m']
    
    # Get parameters with defaults from STRATEGY_PARAMETERS
    fast_ema_period = params.get('fast_ema_period', 20)
    slow_ema_period = params.get('slow_ema_period', 50)
    rsi_period = params.get('rsi_period', 14)
    rsi_overbought = params.get('rsi_overbought', 70)
    rsi_oversold = params.get('rsi_oversold', 30)
    
    logger.info(f"🎯 Creating RSI Mean Reversion with EMA Trend Filter strategy with parameters:")
    logger.info(f" - Symbols: {symbols}")
    logger.info(f" - Timeframes: {timeframes}")
    logger.info(f" - Fast EMA: {fast_ema_period}, Slow EMA: {slow_ema_period}")
    logger.info(f" - RSI: {rsi_period} (OB: {rsi_overbought}, OS: {rsi_oversold})")
    
    try:
        # Create strategy using StrategyBuilder
        strategy_builder = StrategyBuilder(symbols, timeframes)
        
        # Add indicators for each timeframe
        for timeframe in timeframes:
            # Trend indicators
            strategy_builder.add_indicator(f'ema_fast_{timeframe}', ema, period=fast_ema_period)
            strategy_builder.add_indicator(f'ema_slow_{timeframe}', ema, period=slow_ema_period)
            
            # Momentum indicator
            strategy_builder.add_indicator(f'rsi_{timeframe}', rsi, period=rsi_period)
        
        # Add signal rule using the rsi_mean_reversion_with_trend function
        entry_timeframe = timeframes[0]
        
        strategy_builder.add_signal_rule('rsi_ema_signals', rsi_mean_reversion_with_trend,
                                       rsi=f'rsi_{entry_timeframe}',
                                       ema_fast=f'ema_fast_{entry_timeframe}',
                                       ema_slow=f'ema_slow_{entry_timeframe}',
                                       overbought=rsi_overbought,
                                       oversold=rsi_oversold)
        
        # Set strategy information
        strategy_builder.set_strategy_info('RSI_Mean_Reversion_EMA_Trend_Filter', '1.0.0')
        
        # Build and return the strategy
        strategy = strategy_builder.build()
        
        logger.info(f"✅ RSI Mean Reversion with EMA Trend Filter strategy created successfully!")
        logger.info(f" - Strategy Name: {strategy.name}")
        logger.info(f" - Strategy Symbols: {strategy.symbols}")
        logger.info(f" - Strategy Timeframes: {strategy.timeframes}")
        
        return strategy
        
    except Exception as e:
        logger.error(f"❌ Error creating RSI Mean Reversion with EMA Trend Filter strategy: {e}")
        import traceback
        traceback.print_exc()
        raise

class RSIMeanReversionEMATrendStrategy(StrategyBase):
    """
    RSI Mean Reversion with EMA Trend Filter Strategy Class
    """
    
    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        super().__init__(
            name="RSI_Mean_Reversion_EMA_Trend_Filter",
            symbols=symbols,
            timeframes=timeframes,
            config=config
        )
        
        # Strategy-specific parameters
        self.fast_ema_period = config.get('fast_ema_period', 20)
        self.slow_ema_period = config.get('slow_ema_period', 50)
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        
        # Validate parameters
        self._validate_parameters()
        
        logger.info(f"📈 RSIMeanReversionEMATrendStrategy initialized:")
        logger.info(f" - Symbols: {self.symbols}")
        logger.info(f" - Timeframes: {self.timeframes}")
    
    def _validate_parameters(self):
        """Validate strategy parameters"""
        if self.fast_ema_period >= self.slow_ema_period:
            raise ValueError("Fast EMA period must be less than slow EMA period")
        if self.rsi_oversold >= self.rsi_overbought:
            raise ValueError("RSI oversold level must be less than overbought level")
    
    def generate_signals(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, str]]:
        """Generate trading signals using RSI crossovers with EMA trend filter"""
        signals = {}
        
        try:
            for symbol in data:
                signals[symbol] = {}
                
                for timeframe in data[symbol]:
                    signal = self._generate_single_signal(data[symbol][timeframe], symbol, timeframe)
                    signals[symbol][timeframe] = signal
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return signals
    
    def _generate_single_signal(self, df: pd.DataFrame, symbol: str, timeframe: str) -> str:
        """Generate a single trading signal using RSI crossovers with EMA trend filter"""
        try:
            # Need at least 2 periods for crossover detection
            min_periods = max(self.slow_ema_period, self.rsi_period) + 1
            if len(df) < min_periods:
                return 'HOLD'
            
            # Calculate indicators if not already in DataFrame
            if f'rsi_{timeframe}' not in df.columns:
                df[f'rsi_{timeframe}'] = rsi(df['close'], period=self.rsi_period)
            
            if f'ema_fast_{timeframe}' not in df.columns:
                df[f'ema_fast_{timeframe}'] = ema(df['close'], period=self.fast_ema_period)
            
            if f'ema_slow_{timeframe}' not in df.columns:
                df[f'ema_slow_{timeframe}'] = ema(df['close'], period=self.slow_ema_period)
            
            # Get current and previous values
            current_rsi = df[f'rsi_{timeframe}'].iloc[-1]
            prev_rsi = df[f'rsi_{timeframe}'].iloc[-2]
            
            current_ema_fast = df[f'ema_fast_{timeframe}'].iloc[-1]
            current_ema_slow = df[f'ema_slow_{timeframe}'].iloc[-1]
            
            # Determine trend direction
            is_uptrend = current_ema_fast > current_ema_slow
            is_downtrend = current_ema_fast < current_ema_slow
            
            # Check for RSI crossovers
            rsi_cross_above_oversold = (prev_rsi <= self.rsi_oversold) and (current_rsi > self.rsi_oversold)
            rsi_cross_below_overbought = (prev_rsi >= self.rsi_overbought) and (current_rsi < self.rsi_overbought)
            rsi_cross_above_overbought = (prev_rsi <= self.rsi_overbought) and (current_rsi > self.rsi_overbought)
            rsi_cross_below_oversold = (prev_rsi >= self.rsi_oversold) and (current_rsi < self.rsi_oversold)
            
            # Generate signals based on trend and RSI crossovers
            if is_uptrend:
                # In uptrend, we only go long
                if rsi_cross_above_oversold:
                    return 'OPEN_LONG'  # RSI crossed up through oversold level
                elif rsi_cross_below_overbought:
                    return 'CLOSE_LONG'  # RSI crossed down through overbought level
            
            elif is_downtrend:
                # In downtrend, we only go short
                if rsi_cross_below_overbought:
                    return 'OPEN_SHORT'  # RSI crossed down through overbought level
                elif rsi_cross_above_oversold:  # Fixed: was rsi_cross_below_oversold
                    return 'CLOSE_SHORT'  # RSI crossed up through oversold level
            
            return 'HOLD'
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol} {timeframe}: {e}")
            return 'HOLD'

def create_rsi_ema_trend_strategy_instance(symbols=None, timeframes=None, **params):
    """Create RSI Mean Reversion with EMA Trend Filter strategy instance"""
    try:
        if symbols is None:
            symbols = ['BTCUSDT']
        if timeframes is None:
            timeframes = ['5m']
        
        strategy = RSIMeanReversionEMATrendStrategy(symbols, timeframes, params)
        logger.info(f"✅ RSI Mean Reversion with EMA Trend Filter strategy created successfully")
        return strategy
    except Exception as e:
        logger.error(f"Error creating strategy: {e}")
        raise

def simple_test():
    """Simple test to verify the strategy works"""
    try:
        strategy = create_strategy(
            symbols=['BTCUSDT'],
            timeframes=['5m'],
            fast_ema_period=20,
            slow_ema_period=50,
            rsi_period=14,
            rsi_overbought=70,
            rsi_oversold=30
        )
        
        print(f"✅ RSI Mean Reversion with EMA Trend Filter strategy created successfully: {strategy.name}")
        print(f" - Symbols: {strategy.symbols}")
        print(f" - Timeframes: {strategy.timeframes}")
        return True
        
    except Exception as e:
        print(f"❌ Error testing RSI Mean Reversion with EMA Trend Filter strategy: {e}")
        return False

# For testing
if __name__ == "__main__":
    simple_test()