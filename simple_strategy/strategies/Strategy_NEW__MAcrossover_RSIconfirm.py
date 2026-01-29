"""
Moving Average Crossover with RSI Confirmation Strategy
=======================================================

A trend-following strategy that:
- Uses fast and slow moving averages to determine trend direction
- Uses RSI to confirm signals and avoid overbought/oversold conditions
- Implements proper OPEN/CLOSE signal logic

Strategy Logic:
1. Trend Detection:
   - UPTREND when Fast MA > Slow MA
   - DOWNTREND when Fast MA < Slow MA

2. Signal Generation:
   - OPEN_LONG when Fast MA crosses above Slow MA AND RSI < 70
   - CLOSE_LONG when Fast MA crosses below Slow MA
   - OPEN_SHORT when Fast MA crosses below Slow MA AND RSI > 30
   - CLOSE_SHORT when Fast MA crosses above Slow MA

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
from simple_strategy.strategies.signals_library import ma_crossover
from simple_strategy.shared.strategy_base import StrategyBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CRITICAL: STRATEGY_PARAMETERS for GUI Configuration
STRATEGY_PARAMETERS = {
    # Fast MA for trend direction
    'fast_ma_period': {
        'type': 'int',
        'default': 10,
        'min': 5,
        'max': 20,
        'description': 'Fast moving average period',
        'gui_hint': 'Lower values = more sensitive signals. Recommended: 8-12'
    },
    # Slow MA for trend direction
    'slow_ma_period': {
        'type': 'int',
        'default': 30,
        'min': 20,
        'max': 50,
        'description': 'Slow moving average period',
        'gui_hint': 'Higher values = smoother trend. Recommended: 25-35'
    },
    # RSI for signal confirmation
    'rsi_period': {
        'type': 'int',
        'default': 14,
        'min': 7,
        'max': 21,
        'description': 'RSI period for signal confirmation',
        'gui_hint': 'Standard values: 14, 10 for faster signals'
    },
    # RSI levels
    'rsi_overbought': {
        'type': 'int',
        'default': 70,
        'min': 65,
        'max': 80,
        'description': 'RSI overbought level',
        'gui_hint': 'Avoid long entries above this level'
    },
    'rsi_oversold': {
        'type': 'int',
        'default': 30,
        'min': 20,
        'max': 35,
        'description': 'RSI oversold level',
        'gui_hint': 'Avoid short entries below this level'
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
    fast_ma_period = params.get('fast_ma_period', 10)
    slow_ma_period = params.get('slow_ma_period', 30)
    rsi_period = params.get('rsi_period', 14)
    rsi_overbought = params.get('rsi_overbought', 70)
    rsi_oversold = params.get('rsi_oversold', 30)
    
    logger.info(f"🎯 Creating Moving Average Crossover with RSI Confirmation strategy with parameters:")
    logger.info(f" - Symbols: {symbols}")
    logger.info(f" - Timeframes: {timeframes}")
    logger.info(f" - Fast MA: {fast_ma_period}, Slow MA: {slow_ma_period}")
    logger.info(f" - RSI: {rsi_period} (OB: {rsi_overbought}, OS: {rsi_oversold})")
    
    try:
        # Create strategy using StrategyBuilder
        strategy_builder = StrategyBuilder(symbols, timeframes)
        
        # Add indicators for each timeframe
        for timeframe in timeframes:
            # Trend indicators
            strategy_builder.add_indicator(f'ema_fast_{timeframe}', ema, period=fast_ma_period)
            strategy_builder.add_indicator(f'ema_slow_{timeframe}', ema, period=slow_ma_period)
            
            # Momentum indicator for confirmation
            strategy_builder.add_indicator(f'rsi_{timeframe}', rsi, period=rsi_period)
        
        # Add signal rule using the ma_crossover function
        entry_timeframe = timeframes[0]
        
        strategy_builder.add_signal_rule('ma_crossover_signals', ma_crossover,
                                       fast_ma=f'ema_fast_{entry_timeframe}',
                                       slow_ma=f'ema_slow_{entry_timeframe}')
        
        # Set strategy information
        strategy_builder.set_strategy_info('MA_Crossover_RSI_Confirmation', '1.0.0')
        
        # Build the strategy
        strategy = strategy_builder.build()
        
        # Override the generate_signals method to add RSI confirmation
        original_generate_signals = strategy.generate_signals
        
        def generate_signals_with_rsi_confirmation(data):
            # Get original signals
            original_signals = original_generate_signals(data)
            
            # Create a new signals dictionary with RSI confirmation
            confirmed_signals = {}
            
            for symbol in data:
                confirmed_signals[symbol] = {}
                
                for timeframe in data[symbol]:
                    df = data[symbol][timeframe]
                    
                    # Skip if not enough data
                    if len(df) < slow_ma_period:
                        confirmed_signals[symbol][timeframe] = 'HOLD'
                        continue
                    
                    # Calculate RSI if not already in DataFrame
                    if f'rsi_{timeframe}' not in df.columns:
                        df[f'rsi_{timeframe}'] = rsi(df['close'], period=rsi_period)
                    
                    # Get current RSI value
                    current_rsi = df[f'rsi_{timeframe}'].iloc[-1]
                    
                    # Get original signal
                    original_signal = original_signals[symbol][timeframe]
                    
                    # Apply RSI confirmation
                    if original_signal == 'OPEN_LONG' and current_rsi < rsi_overbought:
                        confirmed_signals[symbol][timeframe] = 'OPEN_LONG'
                    elif original_signal == 'OPEN_SHORT' and current_rsi > rsi_oversold:
                        confirmed_signals[symbol][timeframe] = 'OPEN_SHORT'
                    else:
                        # Keep close signals and hold as is
                        confirmed_signals[symbol][timeframe] = original_signal
            
            return confirmed_signals
        
        # Replace the generate_signals method
        strategy.generate_signals = generate_signals_with_rsi_confirmation
        
        logger.info(f"✅ Moving Average Crossover with RSI Confirmation strategy created successfully!")
        logger.info(f" - Strategy Name: {strategy.name}")
        logger.info(f" - Strategy Symbols: {strategy.symbols}")
        logger.info(f" - Strategy Timeframes: {strategy.timeframes}")
        
        return strategy
        
    except Exception as e:
        logger.error(f"❌ Error creating Moving Average Crossover with RSI Confirmation strategy: {e}")
        import traceback
        traceback.print_exc()
        raise

class MACrossoverRSIConfirmationStrategy(StrategyBase):
    """
    Moving Average Crossover with RSI Confirmation Strategy Class
    """
    
    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        super().__init__(
            name="MA_Crossover_RSI_Confirmation",
            symbols=symbols,
            timeframes=timeframes,
            config=config
        )
        
        # Strategy-specific parameters
        self.fast_ma_period = config.get('fast_ma_period', 10)
        self.slow_ma_period = config.get('slow_ma_period', 30)
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        
        # Validate parameters
        self._validate_parameters()
        
        logger.info(f"📈 MACrossoverRSIConfirmationStrategy initialized:")
        logger.info(f" - Symbols: {self.symbols}")
        logger.info(f" - Timeframes: {self.timeframes}")
    
    def _validate_parameters(self):
        """Validate strategy parameters"""
        if self.fast_ma_period >= self.slow_ma_period:
            raise ValueError("Fast MA period must be less than slow MA period")
        if self.rsi_oversold >= self.rsi_overbought:
            raise ValueError("RSI oversold level must be less than overbought level")
    
    def generate_signals(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, str]]:
        """Generate trading signals using MA crossovers with RSI confirmation"""
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
        """Generate a single trading signal using MA crossovers with RSI confirmation"""
        try:
            # Need at least 2 periods for crossover detection
            min_periods = max(self.slow_ma_period, self.rsi_period) + 1
            if len(df) < min_periods:
                return 'HOLD'
            
            # Calculate indicators if not already in DataFrame
            if f'ema_fast_{timeframe}' not in df.columns:
                df[f'ema_fast_{timeframe}'] = ema(df['close'], period=self.fast_ma_period)
            
            if f'ema_slow_{timeframe}' not in df.columns:
                df[f'ema_slow_{timeframe}'] = ema(df['close'], period=self.slow_ma_period)
            
            if f'rsi_{timeframe}' not in df.columns:
                df[f'rsi_{timeframe}'] = rsi(df['close'], period=self.rsi_period)
            
            # Get current and previous values
            current_fast_ma = df[f'ema_fast_{timeframe}'].iloc[-1]
            prev_fast_ma = df[f'ema_fast_{timeframe}'].iloc[-2]
            
            current_slow_ma = df[f'ema_slow_{timeframe}'].iloc[-1]
            prev_slow_ma = df[f'ema_slow_{timeframe}'].iloc[-2]
            
            current_rsi = df[f'rsi_{timeframe}'].iloc[-1]
            
            # Check for MA crossovers
            ma_cross_above = (prev_fast_ma <= prev_slow_ma) and (current_fast_ma > current_slow_ma)
            ma_cross_below = (prev_fast_ma >= prev_slow_ma) and (current_fast_ma < current_slow_ma)
            
            # Generate signals based on MA crossovers and RSI confirmation
            if ma_cross_above and current_rsi < self.rsi_overbought:
                return 'OPEN_LONG'  # Fast MA crossed above Slow MA with RSI confirmation
            elif ma_cross_below and current_rsi > self.rsi_oversold:
                return 'OPEN_SHORT'  # Fast MA crossed below Slow MA with RSI confirmation
            elif ma_cross_below:
                return 'CLOSE_LONG'  # Fast MA crossed below Slow MA, close long position
            elif ma_cross_above:
                return 'CLOSE_SHORT'  # Fast MA crossed above Slow MA, close short position
            
            return 'HOLD'
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol} {timeframe}: {e}")
            return 'HOLD'

def create_ma_rsi_strategy_instance(symbols=None, timeframes=None, **params):
    """Create Moving Average Crossover with RSI Confirmation strategy instance"""
    try:
        if symbols is None:
            symbols = ['BTCUSDT']
        if timeframes is None:
            timeframes = ['5m']
        
        strategy = MACrossoverRSIConfirmationStrategy(symbols, timeframes, params)
        logger.info(f"✅ Moving Average Crossover with RSI Confirmation strategy created successfully")
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
            fast_ma_period=10,
            slow_ma_period=30,
            rsi_period=14,
            rsi_overbought=70,
            rsi_oversold=30
        )
        
        print(f"✅ Moving Average Crossover with RSI Confirmation strategy created successfully: {strategy.name}")
        print(f" - Symbols: {strategy.symbols}")
        print(f" - Timeframes: {strategy.timeframes}")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Moving Average Crossover with RSI Confirmation strategy: {e}")
        return False

# For testing
if __name__ == "__main__":
    simple_test()