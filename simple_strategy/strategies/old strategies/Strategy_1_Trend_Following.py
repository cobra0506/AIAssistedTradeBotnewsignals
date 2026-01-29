"""
Strategy 1: Trend Following Strategy
Uses moving average crossovers to identify trend direction and trade accordingly.
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
from .indicators_library import sma, ema
from .signals_library import ma_crossover

# Configure logging
logger = logging.getLogger(__name__)

# STRATEGY_PARAMETERS - GUI Configuration (AT TOP)
# This defines what parameters appear in the GUI for users to configure
STRATEGY_PARAMETERS = {
    'fast_period': {
        'type': 'int',
        'default': 12,
        'min': 5,
        'max': 50,
        'description': 'Fast moving average period',
        'gui_hint': 'Lower values = more sensitive signals'
    },
    'slow_period': {
        'type': 'int',
        'default': 26,
        'min': 10,
        'max': 100,
        'description': 'Slow moving average period',
        'gui_hint': 'Should be 2-3x the fast period'
    },
    'ma_type': {
        'type': 'str',
        'default': 'ema',
        'options': ['sma', 'ema'],
        'description': 'Moving average type',
        'gui_hint': 'EMA reacts faster to price changes'
    }
}

def create_strategy(symbols=None, timeframes=None, **params):
    """
    Create Trend Following Strategy
    Uses moving average crossovers to identify trend direction
    
    Parameters:
    - symbols: List of trading symbols (default: ['BTCUSDT'])
    - timeframes: List of timeframes (default: ['1m'])
    - fast_period: Fast MA period (default: 12)
    - slow_period: Slow MA period (default: 26)
    - ma_type: Moving average type (default: 'ema')
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
    fast_period = params.get('fast_period', 12)
    slow_period = params.get('slow_period', 26)
    ma_type = params.get('ma_type', 'ema')
    
    logger.info(f"🎯 Creating Trend Following strategy with parameters:")
    logger.info(f"  - Symbols: {symbols}")
    logger.info(f"  - Timeframes: {timeframes}")
    logger.info(f"  - Fast Period: {fast_period}")
    logger.info(f"  - Slow Period: {slow_period}")
    logger.info(f"  - MA Type: {ma_type}")
    
    try:
        # Create strategy using StrategyBuilder
        strategy_builder = StrategyBuilder(symbols, timeframes)
        
        # Add indicators based on MA type
        if ma_type == 'sma':
            strategy_builder.add_indicator('sma_fast', sma, period=fast_period)
            strategy_builder.add_indicator('sma_slow', sma, period=slow_period)
            fast_ma_name = 'sma_fast'
            slow_ma_name = 'sma_slow'
        else:  # ema
            strategy_builder.add_indicator('ema_fast', ema, period=fast_period)
            strategy_builder.add_indicator('ema_slow', ema, period=slow_period)
            fast_ma_name = 'ema_fast'
            slow_ma_name = 'ema_slow'
        
        # Add signal rule for MA crossover
        strategy_builder.add_signal_rule('ma_crossover', ma_crossover,
                                       fast_ma=fast_ma_name,
                                       slow_ma=slow_ma_name)
        
        # Set signal combination method
        strategy_builder.set_signal_combination('majority_vote')
        
        # Set strategy information
        strategy_builder.set_strategy_info('Trend_Following', '1.0.0')
        
        # Build and return the strategy
        strategy = strategy_builder.build()
        
        logger.info(f"✅ Trend Following strategy created successfully!")
        logger.info(f"  - Strategy Name: {strategy.name}")
        logger.info(f"  - Strategy Symbols: {strategy.symbols}")
        logger.info(f"  - Strategy Timeframes: {strategy.timeframes}")
        
        return strategy
        
    except Exception as e:
        logger.error(f"❌ Error creating Trend Following strategy: {e}")
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
            fast_period=12,
            slow_period=26,
            ma_type='ema'
        )
        
        print(f"✅ Trend Following strategy created successfully: {strategy.name}")
        print(f"  - Symbols: {strategy.symbols}")
        print(f"  - Timeframes: {strategy.timeframes}")
        return True
    except Exception as e:
        print(f"❌ Error testing Trend Following strategy: {e}")
        return False

# For testing
if __name__ == "__main__":
    simple_test()

