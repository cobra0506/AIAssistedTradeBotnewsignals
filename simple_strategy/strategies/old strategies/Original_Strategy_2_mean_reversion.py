"""
Strategy 2: Mean Reversion Strategy
Uses RSI to identify overbought/oversold conditions for mean reversion trades.
STANDARDIZED VERSION - Clean, consistent structure
"""
import sys
import os
import logging
from typing import Dict, List, Any

# Add parent directories to path for proper imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from .strategy_builder import StrategyBuilder

from .indicators_library import rsi, sma,ema, trend_signal
from .signals_library import overbought_oversold, and_signals

# Configure logging
logger = logging.getLogger(__name__)

# STRATEGY_PARAMETERS - GUI Configuration (AT TOP)
# This defines what parameters appear in the GUI for users to configure
STRATEGY_PARAMETERS = {
    'rsi_period': {
        'type': 'int',
        'default': 2,
        'min': 1,
        'max': 50,
        'description': 'RSI calculation period',
        'gui_hint': 'Standard values: 14, 21. Lower = more sensitive'
    },
    'rsi_overbought': {
        'type': 'int',
        'default': 78,
        'min': 50,
        'max': 90,
        'description': 'RSI overbought level (sell signal)',
        'gui_hint': 'Higher values = more conservative sell signals'
    },
    'rsi_oversold': {
        'type': 'int',
        'default': 22,
        'min': 10,
        'max': 50,
        'description': 'RSI oversold level (buy signal)',
        'gui_hint': 'Lower values = more conservative buy signals'
    }
}

def create_strategy(symbols=None, timeframes=None, **params):
    """
    Create Mean Reversion Strategy
    Uses RSI to identify overbought/oversold conditions for mean reversion trades.
    
    Parameters:
    - symbols: List of trading symbols (default: ['BTCUSDT'])
    - timeframes: List of timeframes (default: ['1m'])
    - rsi_period: RSI calculation period (default: 14)
    - rsi_overbought: RSI overbought level (default: 70)
    - rsi_oversold: RSI oversold level (default: 30)
    """
    # DEBUG: Log what we receive
    logger.info(f"🔧 create_strategy called with:")
    logger.info(f"  - symbols: {symbols}")
    logger.info(f"  - timeframes: {timeframes}")
    logger.info(f"  - params: {params}")
    
    # Handle None/empty values with defaults
    if symbols is None or len(symbols) == 0:
        logger.warning("⚠️ No symbols provided, using default: ['BTCUSDT']")
        symbols = ['BTCUSDT']
    
    if timeframes is None or len(timeframes) == 0:
        logger.warning("⚠️ No timeframes provided, using default: ['1m']")
        timeframes = ['1m']
    
    # Get parameters with defaults from STRATEGY_PARAMETERS
    rsi_period = params.get('rsi_period', 14)
    rsi_overbought = params.get('rsi_overbought', 70)
    rsi_oversold = params.get('rsi_oversold', 30)
    
    logger.info(f"🎯 Creating Mean Reversion strategy with parameters:")
    logger.info(f"  - Symbols: {symbols}")
    logger.info(f"  - Timeframes: {timeframes}")
    logger.info(f"  - RSI Period: {rsi_period}")
    logger.info(f"  - RSI Overbought: {rsi_overbought}")
    logger.info(f"  - RSI Oversold: {rsi_oversold}")
    
    try:
        # Create strategy using StrategyBuilder
        strategy_builder = StrategyBuilder(symbols, timeframes)
        
        # Add RSI indicator
        strategy_builder.add_indicator('rsi', rsi, period=rsi_period)
        
        # Add signal rule for RSI overbought/oversold
        strategy_builder.add_signal_rule('rsi_signal', overbought_oversold,
                                       indicator='rsi',
                                       overbought=rsi_overbought,
                                       oversold=rsi_oversold)

        # Add EMA indicators for trend
        strategy_builder.add_indicator('ema_fast', ema, period=20)
        strategy_builder.add_indicator('ema_slow', ema, period=50)

        # Add trend signal rule
        strategy_builder.add_signal_rule(
            'trend_signal',
            trend_signal,
            ema_fast='ema_fast',
            ema_slow='ema_slow'
        )
        
        # Set signal combination method
        #strategy_builder.set_signal_combination('majority_vote')
        # Combine RSI and trend using AND
        strategy_builder.set_signal_combination('and_signals')

        
        # Set strategy information
        strategy_builder.set_strategy_info('Mean_Reversion', '1.0.0')
        
        # Build and return the strategy
        strategy = strategy_builder.build()
        
        logger.info(f"✅ Mean Reversion strategy created successfully!")
        logger.info(f"  - Strategy Name: {strategy.name}")
        logger.info(f"  - Strategy Symbols: {strategy.symbols}")
        logger.info(f"  - Strategy Timeframes: {strategy.timeframes}")
        
        return strategy
        
    except Exception as e:
        logger.error(f"❌ Error creating Mean Reversion strategy: {e}")
        import traceback
        traceback.print_exc()
        raise

def simple_test():
    """Simple test to verify the strategy works"""
    try:
        # Test strategy creation
        strategy = create_strategy(
            symbols=['BTCUSDT'],
            timeframes=['1m'],
            rsi_period=14,
            rsi_overbought=70,
            rsi_oversold=30
        )
        
        print(f"✅ Mean Reversion strategy created successfully: {strategy.name}")
        print(f"  - Symbols: {strategy.symbols}")
        print(f"  - Timeframes: {strategy.timeframes}")
        return True
    except Exception as e:
        print(f"❌ Error testing Mean Reversion strategy: {e}")
        return False

# For testing
if __name__ == "__main__":
    simple_test()
