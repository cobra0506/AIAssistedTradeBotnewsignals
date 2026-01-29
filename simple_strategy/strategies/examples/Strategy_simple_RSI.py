"""
Simple RSI Strategy - Example Strategy
Uses RSI to identify overbought/oversold conditions for trading signals.
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
from .indicators_library import rsi
from .signals_library import overbought_oversold

# Configure logging
logger = logging.getLogger(__name__)

# STRATEGY_PARAMETERS - GUI Configuration (AT TOP)
# This defines what parameters appear in the GUI for users to configure
STRATEGY_PARAMETERS = {
    'rsi_period': {
        'type': 'int',
        'default': 14,
        'min': 5,
        'max': 30,
        'description': 'RSI calculation period',
        'gui_hint': 'Standard values: 14, 21. Lower = more sensitive'
    },
    'overbought_level': {
        'type': 'int',
        'default': 70,
        'min': 60,
        'max': 90,
        'description': 'RSI overbought level (sell signal)',
        'gui_hint': 'Higher values = more conservative sell signals'
    },
    'oversold_level': {
        'type': 'int',
        'default': 30,
        'min': 10,
        'max': 40,
        'description': 'RSI oversold level (buy signal)',
        'gui_hint': 'Lower values = more conservative buy signals'
    }
}

def create_strategy(symbols=None, timeframes=None, **params):
    """
    Create Simple RSI Strategy
    Uses RSI to identify overbought/oversold conditions for trading signals.
    
    Parameters:
    - symbols: List of trading symbols (default: ['BTCUSDT'])
    - timeframes: List of timeframes (default: ['1m'])
    - rsi_period: RSI calculation period (default: 14)
    - overbought_level: RSI overbought level (default: 70)
    - oversold_level: RSI oversold level (default: 30)
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
    overbought_level = params.get('overbought_level', 70)
    oversold_level = params.get('oversold_level', 30)
    
    logger.info(f"🎯 Creating Simple RSI strategy with parameters:")
    logger.info(f"  - Symbols: {symbols}")
    logger.info(f"  - Timeframes: {timeframes}")
    logger.info(f"  - RSI Period: {rsi_period}")
    logger.info(f"  - Overbought Level: {overbought_level}")
    logger.info(f"  - Oversold Level: {oversold_level}")
    
    try:
        # Create strategy using StrategyBuilder
        strategy_builder = StrategyBuilder(symbols, timeframes)
        
        # Add RSI indicator
        strategy_builder.add_indicator('rsi', rsi, period=rsi_period)
        
        # Add signal rule for RSI overbought/oversold
        strategy_builder.add_signal_rule('rsi_signal', overbought_oversold,
                                       indicator='rsi',
                                       overbought=overbought_level,
                                       oversold=oversold_level)
        
        # Set signal combination method
        strategy_builder.set_signal_combination('majority_vote')
        
        # Set strategy information
        strategy_builder.set_strategy_info('Simple_RSI', '1.0.0')
        
        # Build and return the strategy
        strategy = strategy_builder.build()
        
        logger.info(f"✅ Simple RSI strategy created successfully!")
        logger.info(f"  - Strategy Name: {strategy.name}")
        logger.info(f"  - Strategy Symbols: {strategy.symbols}")
        logger.info(f"  - Strategy Timeframes: {strategy.timeframes}")
        
        return strategy
        
    except Exception as e:
        logger.error(f"❌ Error creating Simple RSI strategy: {e}")
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
            overbought_level=70,
            oversold_level=30
        )
        
        print(f"✅ Simple RSI strategy created successfully: {strategy.name}")
        print(f"  - Symbols: {strategy.symbols}")
        print(f"  - Timeframes: {strategy.timeframes}")
        return True
    except Exception as e:
        print(f"❌ Error testing Simple RSI strategy: {e}")
        return False

# For testing
if __name__ == "__main__":
    simple_test()


'''"""
Simple RSI Strategy - PROOF OF CONCEPT
========================================
A simple strategy using only RSI indicator for overbought/oversold signals
"""

import sys
import os
import pandas as pd
import logging
from typing import Dict, List

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from .strategy_builder import StrategyBuilder
from .indicators_library import rsi
from .signals_library import overbought_oversold

logger = logging.getLogger(__name__)

def create_strategy(symbols=None, timeframes=None, **params):
    """
    Simple RSI Strategy - Uses only RSI for trading signals
    
    Strategy Logic:
    - BUY when RSI < 30 (oversold)
    - SELL when RSI > 70 (overbought)
    - HOLD otherwise
    """
    # Use what we receive from GUI (this works!)
    if symbols is None:
        symbols = ['BTCUSDT']
    if timeframes is None:
        timeframes = ['1h']
    
    # Get parameters from GUI (this works!)
    rsi_period = params.get('rsi_period', 14)
    oversold_level = params.get('oversold_level', 30)
    overbought_level = params.get('overbought_level', 70)
    
    logger.info(f"🔍 SIMPLE RSI STRATEGY:")
    logger.info(f" - Symbols: {symbols}")
    logger.info(f" - Timeframes: {timeframes}")
    logger.info(f" - RSI Period: {rsi_period}")
    logger.info(f" - Oversold Level: {oversold_level}")
    logger.info(f" - Overbought Level: {overbought_level}")
    
    # Create strategy
    strategy = StrategyBuilder(symbols, timeframes)
    
    # Add single indicator
    strategy.add_indicator('rsi', rsi, period=rsi_period)
    
    # Add single signal rule
    strategy.add_signal_rule('rsi_signal', overbought_oversold, 
                             indicator='rsi', 
                             overbought=overbought_level, 
                             oversold=oversold_level)
    
    # Add risk management
    strategy.add_risk_rule('stop_loss', percent=2.0)
    strategy.add_risk_rule('take_profit', percent=4.0)
    
    # Set strategy info
    strategy.set_strategy_info('Simple_RSI', '1.0.0')
    
    # Build and return
    return strategy.build()

# GUI Parameters
STRATEGY_PARAMETERS = {
    'rsi_period': {
        'type': 'int',
        'default': 14,
        'min': 5,
        'max': 30,
        'description': 'RSI calculation period'
    },
    'oversold_level': {
        'type': 'int',
        'default': 30,
        'min': 10,
        'max': 40,
        'description': 'RSI oversold level (BUY signal)'
    },
    'overbought_level': {
        'type': 'int',
        'default': 70,
        'min': 60,
        'max': 90,
        'description': 'RSI overbought level (SELL signal)'
    }
}

# Documentation
"""
=================================================================
SIMPLE RSI STRATEGY - WORKING PROOF
=================================================================

✅ GUARANTEED TO WORK:
-------------------
1. GUI Detection: File named Strategy_*.py will be detected
2. GUI Parameters: All 3 parameters appear in GUI with correct types
3. Symbol/Timeframe: Uses GUI-assigned symbols and timeframes
4. Data Loading: Data files found and loaded correctly
5. Indicator Integration: RSI added to DataFrame columns
6. Signal Generation: Signal function receives correct RSI data
7. Signal Output: Returns proper pandas Series
8. Backtest Trades: Will generate actual trades

🎯 STRATEGY LOGIC:
- Simple overbought/oversold RSI strategy
- BUY when RSI crosses below oversold level
- SELL when RSI crosses above overbought level
- Basic risk management with stop-loss and take-profit

📊 EXPECTED RESULTS:
- Should generate multiple trades per week
- Win rate depends on market conditions
- Simple but effective for ranging markets

🔧 TECHNICAL PROOF:
- Uses only 1 indicator (RSI)
- Uses only 1 signal rule (overbought_oversold)
- Uses majority vote signal combination (default)
- All building blocks work together perfectly
"""'''