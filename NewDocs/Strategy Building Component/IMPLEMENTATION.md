# Strategy Building Component - IMPLEMENTATION

## Detailed Implementation Guide

### Current Implementation Status

**WARNING**: This component has known issues that prevent proper functioning. The following guide describes the intended implementation, but actual results may vary.

## Core Components

### 1. StrategyBuilder Class

#### Location: `simple_strategy/strategies/strategy_builder.py`

#### Class Structure
```python
class StrategyBuilder:
    def __init__(self, symbols: List[str], timeframes: List[str] = ['1m'])
    def add_indicator(self, name: str, indicator_func: Callable, **params) -> 'StrategyBuilder'
    def add_signal_rule(self, name: str, signal_func: Callable, **params) -> 'StrategyBuilder'
    def add_risk_rule(self, rule_type: str, **params) -> 'StrategyBuilder'
    def set_signal_combination(self, method: str, **kwargs) -> 'StrategyBuilder'
    def build(self) -> StrategyBase

Known Issues 

     _calculate_indicators() method calculates indicators but doesn't properly integrate them
     Signal generation process has parameter passing problems
     Returns strategies that generate 0 trades in backtest
     

Usage Pattern 

# Basic pattern (currently has issues)
strategy_builder = StrategyBuilder(['BTCUSDT'], ['1h'])
strategy_builder.add_indicator('rsi', rsi, period=14)
strategy_builder.add_signal_rule('rsi_signal', overbought_oversold, 
                                indicator='rsi', overbought=70, oversold=30)
strategy_builder.set_signal_combination('majority_vote')
strategy = strategy_builder.build()  # This creates a strategy but it won't work properly

2. Indicators Library 
Location: simple_strategy/strategies/indicators_library.py 
Status: WORKING ✅ 

The indicators library is fully functional and can be used independently. 
Available Indicators 

Trend Indicators 

sma(data: pd.Series, period: int = 20) -> pd.Series
ema(data: pd.Series, period: int = 20) -> pd.Series
wma(data: pd.Series, period: int = 20) -> pd.Series
dema(data: pd.Series, period: int = 20) -> pd.Series
tema(data: pd.Series, period: int = 20) -> pd.Series

Momentum Indicators

rsi(data: pd.Series, period: int = 14) -> pd.Series
stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
           k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]
srsi(data: pd.Series, period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]
macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, 
     signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]
cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series
williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series

Volatility Indicators

atr(high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series, 
    period: int = 14) -> pd.Series

Manual Usage (Working)

import pandas as pd
from simple_strategy.strategies.indicators_library import rsi, sma

# This works correctly
df = pd.read_csv('data/BTCUSDT_1h.csv')
rsi_values = rsi(df['close'], period=14)
sma_values = sma(df['close'], period=20)

print(f"RSI calculated: {len(rsi_values)} values")
print(f"SMA calculated: {len(sma_values)} values")

3. Signals Library 
Location: simple_strategy/strategies/signals_library.py 
Status: HAS CRITICAL ISSUES ❌ 

The signals library has inconsistent return types and parameter passing problems. 
Available Signals (With Issues) 

Basic Signals 

# Returns strings: 'BUY', 'SELL', 'HOLD'
overbought_oversold(indicator, overbought=70, oversold=30) -> pd.Series
ma_crossover(fast_ma, slow_ma) -> pd.Series
stochastic_signals(k_percent, d_percent, overbought=80, oversold=20) -> pd.Series

# Returns numeric: 1, -1, 0
macd_signals(macd_line, signal_line, histogram=None) -> pd.Series
bollinger_bands_signals(price, upper_band, lower_band, middle_band=None) -> pd.Series

Advanced Signals

divergence_signals(price, indicator, lookback_period=20) -> pd.Series
breakout_signals(price, resistance, support, penetration_pct=0.01) -> pd.Series
trend_strength_signals(price, short_ma, long_ma, adx=None, adx_threshold=25) -> pd.Series

Combination Signals

majority_vote_signals(signal_list: List[pd.Series]) -> pd.Series
weighted_signals(signal_list: List[Tuple[pd.Series, float]]) -> pd.Series

Critical Issue: Return Type Inconsistency

# This inconsistency causes major problems
signal1 = overbought_oversold(rsi_values, 70, 30)  # Returns: 'BUY', 'SELL', 'HOLD'
signal2 = macd_signals(macd_line, signal_line)     # Returns: 1, -1, 0

# StrategyBuilder cannot handle this inconsistency properly

Strategy Creation Process 
Step 1: Create Strategy File 
File Structure 

"""
Strategy Description
Author: Your Name
Date: 2025
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import components
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import rsi, sma, ema
from simple_strategy.strategies.signals_library import overbought_oversold, ma_crossover

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CRITICAL: STRATEGY_PARAMETERS (AT TOP)
STRATEGY_PARAMETERS = {
    'rsi_period': {
        'type': 'int',
        'default': 14,
        'min': 5,
        'max': 50,
        'description': 'RSI calculation period',
        'gui_hint': 'Standard: 14, Shorter: 5-10, Longer: 20-30'
    },
    'overbought': {
        'type': 'int',
        'default': 70,
        'min': 50,
        'max': 90,
        'description': 'RSI overbought level',
        'gui_hint': 'Standard: 70, More sensitive: 75-80'
    },
    'oversold': {
        'type': 'int',
        'default': 30,
        'min': 10,
        'max': 50,
        'description': 'RSI oversold level',
        'gui_hint': 'Standard: 30, More sensitive: 20-25'
    }
}

def create_strategy(symbols=None, timeframes=None, **params):
    """Create strategy function (REQUIRED)"""
    # Handle None values
    if symbols is None or len(symbols) == 0:
        symbols = ['BTCUSDT']
    if timeframes is None or len(timeframes) == 0:
        timeframes = ['1h']
    
    # Get parameters
    rsi_period = params.get('rsi_period', 14)
    overbought = params.get('overbought', 70)
    oversold = params.get('oversold', 30)
    
    try:
        # Create strategy builder
        strategy_builder = StrategyBuilder(symbols, timeframes)
        
        # Add indicators
        strategy_builder.add_indicator('rsi', rsi, period=rsi_period)
        
        # Add signal rules
        strategy_builder.add_signal_rule('rsi_signal', overbought_oversold,
                                       indicator='rsi', 
                                       overbought=overbought, 
                                       oversold=oversold)
        
        # Set signal combination
        strategy_builder.set_signal_combination('majority_vote')
        
        # Set strategy info
        strategy_builder.set_strategy_info('RSI_Strategy', '1.0.0')
        
        return strategy_builder.build()
        
    except Exception as e:
        logger.error(f"❌ Error creating strategy: {e}")
        raise

def simple_test():
    """Test strategy - EXACT function required"""
    print("🧪 RSI STRATEGY TEST")
    print("=" * 30)
    
    try:
        strategy = create_strategy(
            symbols=['BTCUSDT'],
            timeframes=['1h'],
            rsi_period=14,
            overbought=70,
            oversold=30
        )
        
        print(f"✅ Strategy created: {strategy.name}")
        print(f" Symbols: {strategy.symbols}")
        print(f" Timeframes: {strategy.timeframes}")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    simple_test()

Step 2: File Naming and Location 
Requirements 

     File Name: Must start with Strategy_ (e.g., Strategy_RSI.py)
     Location: simple_strategy/strategies/
     Required Elements:
         STRATEGY_PARAMETERS dictionary at TOP
         create_strategy() function
         simple_test() function
         Proper imports
         Error handling
         
     

Step 3: Testing the Strategy 
Manual Test 

cd AIAssistedTradeBot/
python -c "from simple_strategy.strategies.Strategy_RSI import simple_test; simple_test()"

GUI Test

python main.py
# Then in GUI:
# 1. Open backtest window
# 2. Select your strategy from dropdown
# 3. Choose symbol and timeframe
# 4. Set parameters
# 5. Click "Create Strategy"
# 6. Click "Run Backtest"

Expected Results (Current State)

🧪 RSI STRATEGY TEST
==============================
✅ Strategy created: RSI_Strategy
 Symbols: ['BTCUSDT']
 Timeframes: ['1h']

 But backtest will likely show:

 Total trades: 0
Total return: 0.00%

Debugging Current Issues 
Common Problems and Solutions 
1. Strategy Creates But Generates 0 Trades 

Problem: Strategy creates successfully but backtest shows 0 trades
Solution: This is a known issue with signal generation. The indicators are calculated but signals are not processed correctly. 
2. "Indicator Not Found in DataFrame" Error 

Problem: StrategyBuilder calculates indicators but doesn't integrate them
Solution: This is a core issue that needs to be fixed in the StrategyBuilder class. 
3. Signal Function Parameter Errors 

Problem: Signal functions receive wrong parameters
Solution: Check signal rule definitions in create_strategy() function. 
Debugging Steps 

# Manual indicator calculation test
from simple_strategy.strategies.indicators_library import rsi
import pandas as pd

df = pd.read_csv('data/BTCUSDT_1h.csv')
rsi_values = rsi(df['close'], period=14)
print(f"RSI calculation test: {len(rsi_values)} values calculated")

# Manual signal test
from simple_strategy.strategies.signals_library import overbought_oversold
signals = overbought_oversold(rsi_values, 70, 30)
print(f"Signal test: {signals.value_counts()}")

Current Limitations 

     

    No Working Multi-Indicator Strategies 
         Complex strategies with multiple indicators don't work
         Stick to single-indicator strategies for now
         
     

    No Reliable Backtest Results 
         Backtest engine runs but generates 0 trades
         Performance metrics are meaningless
         
     

    Limited Signal Combination 
         Only 'majority_vote' works partially
         'weighted' and 'unanimous' have issues
         
     

Future Development Priorities 

     

    Fix Signal Return Type Consistency 
         Standardize all signal functions to return numeric values (1, -1, 0)
         Update combination functions accordingly
         
     

    Resolve StrategyBuilder Integration Issues 
         Fix _calculate_indicators() method
         Ensure proper DataFrame integration
         Fix signal parameter passing
         
     

    Add Comprehensive Error Handling 
         Better error messages
         Graceful failure modes
         Debug logging
         
     

    Create Working Integration Tests 
         Test complete strategy creation and execution
         Validate backtest results
         Performance benchmarking
         
     