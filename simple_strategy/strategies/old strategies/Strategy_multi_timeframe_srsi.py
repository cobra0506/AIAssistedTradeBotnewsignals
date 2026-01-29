"""
Multi-Timeframe Stochastic RSI Strategy
Uses Stochastic RSI across multiple timeframes for entry/exit signals.
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
from .indicators_library import sma, ema, rsi
from .signals_library import overbought_oversold, ma_crossover

# Configure logging
logger = logging.getLogger(__name__)

# STRATEGY_PARAMETERS - GUI Configuration (AT TOP)
# This defines what parameters appear in the GUI for users to configure
STRATEGY_PARAMETERS = {
    'oversold_threshold': {
        'type': 'int',
        'default': 20,
        'min': 5,
        'max': 30,
        'description': 'Stochastic RSI oversold level (BUY signal)',
        'gui_hint': 'Lower values = more conservative BUY signals. Recommended: 20'
    },
    'overbought_threshold': {
        'type': 'int',
        'default': 80,
        'min': 70,
        'max': 95,
        'description': 'Stochastic RSI overbought level (SELL signal)',
        'gui_hint': 'Higher values = more conservative SELL signals. Recommended: 80'
    },
    'rsi_period': {
        'type': 'int',
        'default': 14,
        'min': 7,
        'max': 21,
        'description': 'RSI calculation period',
        'gui_hint': 'Standard values: 14, 21. Lower = more sensitive'
    }
}

def create_strategy(symbols=None, timeframes=None, **params):
    """
    Create Multi-Timeframe Stochastic RSI Strategy
    Uses Stochastic RSI across multiple timeframes for entry/exit signals.
    
    Parameters:
    - symbols: List of trading symbols (default: ['BTCUSDT'])
    - timeframes: List of timeframes (default: ['1m', '5m', '15m'])
    - oversold_threshold: Stochastic RSI oversold level (default: 20)
    - overbought_threshold: Stochastic RSI overbought level (default: 80)
    - rsi_period: RSI calculation period (default: 14)
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
        logger.warning("⚠️ No timeframes provided, using default: ['1m', '5m', '15m']")
        timeframes = ['1m', '5m', '15m']
    
    # Get parameters with defaults from STRATEGY_PARAMETERS
    oversold_threshold = params.get('oversold_threshold', 20)
    overbought_threshold = params.get('overbought_threshold', 80)
    rsi_period = params.get('rsi_period', 14)
    
    logger.info(f"🎯 Creating Multi-Timeframe SRSI strategy with parameters:")
    logger.info(f"  - Symbols: {symbols}")
    logger.info(f"  - Timeframes: {timeframes}")
    logger.info(f"  - Oversold Threshold: {oversold_threshold}")
    logger.info(f"  - Overbought Threshold: {overbought_threshold}")
    logger.info(f"  - RSI Period: {rsi_period}")
    
    try:
        # Create strategy using StrategyBuilder
        strategy_builder = StrategyBuilder(symbols, timeframes)
        
        # Add indicators for multi-timeframe analysis
        strategy_builder.add_indicator('rsi', rsi, period=rsi_period)
        strategy_builder.add_indicator('sma_fast', sma, period=12)
        strategy_builder.add_indicator('sma_slow', sma, period=26)
        
        # Add signal rules
        strategy_builder.add_signal_rule('rsi_signal', overbought_oversold,
                                       indicator='rsi',
                                       overbought=overbought_threshold,
                                       oversold=oversold_threshold)
        
        strategy_builder.add_signal_rule('ma_crossover', ma_crossover,
                                       fast_ma='sma_fast',
                                       slow_ma='sma_slow')
        
        # Set signal combination method
        strategy_builder.set_signal_combination('majority_vote')
        
        # Set strategy information
        strategy_builder.set_strategy_info('Multi_Timeframe_SRSI', '1.0.0')
        
        # Build and return the strategy
        strategy = strategy_builder.build()
        
        logger.info(f"✅ Multi-Timeframe SRSI strategy created successfully!")
        logger.info(f"  - Strategy Name: {strategy.name}")
        logger.info(f"  - Strategy Symbols: {strategy.symbols}")
        logger.info(f"  - Strategy Timeframes: {strategy.timeframes}")
        
        return strategy
        
    except Exception as e:
        logger.error(f"❌ Error creating Multi-Timeframe SRSI strategy: {e}")
        import traceback
        traceback.print_exc()
        raise

def simple_test():
    """Simple test to verify the strategy works"""
    try:
        # Test strategy creation
        strategy = create_strategy(
            symbols=['BTCUSDT'],
            timeframes=['1m', '5m', '15m'],
            oversold_threshold=20,
            overbought_threshold=80,
            rsi_period=14
        )
        
        print(f"✅ Multi-Timeframe SRSI strategy created successfully: {strategy.name}")
        print(f"  - Symbols: {strategy.symbols}")
        print(f"  - Timeframes: {strategy.timeframes}")
        return True
    except Exception as e:
        print(f"❌ Error testing Multi-Timeframe SRSI strategy: {e}")
        return False

# For testing
if __name__ == "__main__":
    simple_test()

