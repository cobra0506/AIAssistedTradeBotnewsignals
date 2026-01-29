# Strategy Building Component - API REFERENCE

## API Documentation

### Current Status: PARTIALLY IMPLEMENTED

**WARNING**: Many API methods have known issues and may not work as expected. This documentation describes the intended API, but actual behavior may vary.

## StrategyBuilder Class

### Location
`simple_strategy.strategies.strategy_builder.StrategyBuilder`

### Constructor

```python
StrategyBuilder(symbols: List[str], timeframes: List[str] = ['1m'])

Parameters: 

     symbols (List[str]): List of trading symbols
     returns: StrategyBuilder instance
     

Example: 

from simple_strategy.strategies.strategy_builder import StrategyBuilder

# This creates the builder but has integration issues
builder = StrategyBuilder(['BTCUSDT'], ['1h'])

Methods 
add_indicator() 

add_indicator(name: str, indicator_func: Callable, **params) -> 'StrategyBuilder'

Description: Adds an indicator to the strategy 

Parameters: 

     name (str): Unique name for the indicator
     indicator_func (Callable): Indicator function from indicators_library
     **params: Parameters for the indicator function
     

Returns: StrategyBuilder instance for method chaining 

Example: 

from simple_strategy.strategies.indicators_library import rsi

# This adds the indicator but integration has issues
builder.add_indicator('rsi', rsi, period=14)

Known Issues: 

     Indicators are calculated but not properly integrated into DataFrame
     Multi-component indicators (MACD, Bollinger Bands) have special handling requirements
     

add_signal_rule() 

add_signal_rule(name: str, signal_func: Callable, **params) -> 'StrategyBuilder'

Description: Adds a signal rule to the strategy 

Parameters: 

     name (str): Unique name for the signal rule
     signal_func (Callable): Signal function from signals_library
     **params: Parameters including indicator references
     

Returns: StrategyBuilder instance for method chaining 

Example: 

from simple_strategy.strategies.signals_library import overbought_oversold

# This adds the signal rule but has parameter passing issues
builder.add_signal_rule('rsi_signal', overbought_oversold,
                       indicator='rsi', 
                       overbought=70, 
                       oversold=30)

Known Issues: 

     Signal functions called with wrong parameters
     Indicator references not resolved correctly
     Signal output inconsistent (strings vs numeric)
     

add_risk_rule() 

add_risk_rule(rule_type: str, **params) -> 'StrategyBuilder'

Description: Adds a risk management rule 

Parameters: 

     rule_type (str): Type of risk rule ('stop_loss', 'take_profit', 'max_position_size')
     **params: Parameters for the risk rule
     

Returns: StrategyBuilder instance for method chaining 

Example: 

# Risk rules are defined but not fully implemented
builder.add_risk_rule('stop_loss', percentage=2.0)

Status: NOT IMPLEMENTED - Risk rules are defined but not functional 
set_signal_combination() 

Description: Sets how signals should be combined 

Parameters: 

     method (str): Combination method ('majority_vote', 'weighted', 'unanimous')
     **kwargs: Additional parameters (like weights for weighted method)
     

Returns: StrategyBuilder instance for method chaining 

Example: 

# Only 'majority_vote' works partially
builder.set_signal_combination('majority_vote')

# 'weighted' requires weights but has issues
builder.set_signal_combination('weighted', weights={
    'rsi_signal': 0.6,
    'another_signal': 0.4
})

Known Issues: 

     'majority_vote': Works partially but affected by signal type inconsistencies
     'weighted': Has issues with weight validation and application
     'unanimous': Too restrictive, rarely generates signals
     

build() 

build() -> StrategyBase

Description: Builds and returns the strategy 

Returns: StrategyBase instance 

Example: 

# This creates a strategy but it won't work properly
strategy = builder.build()

Known Issues: 

     Returns a strategy object that generates 0 trades
     Integration between indicators and signals is broken
     Strategy appears valid but doesn't execute correctly
     

Indicators Library API 
Location 

simple_strategy.strategies.indicators_library 
Status: WORKING ✅ 

The indicators library is fully functional and can be used independently. 
Trend Indicators 
sma() 

sma(data: pd.Series, period: int = 20) -> pd.Series

Description: Simple Moving Average
Parameters: 

     data (pd.Series): Price series
     period (int): Lookback period (default: 20)
    Returns: pandas Series with SMA values
     

ema() 

ema(data: pd.Series, period: int = 20) -> pd.Series

Description: Exponential Moving Average
Parameters: 

     data (pd.Series): Price series
     period (int): Lookback period (default: 20)
    Returns: pandas Series with EMA values
     

wma() 

wma(data: pd.Series, period: int = 20) -> pd.Series

Description: Weighted Moving Average
Parameters: 

     data (pd.Series): Price series
     period (int): Lookback period (default: 20)
    Returns: pandas Series with WMA values
     

dema() 

dema(data: pd.Series, period: int = 20) -> pd.Series

Description: Double Exponential Moving Average
Parameters: 

     data (pd.Series): Price series
     period (int): Lookback period (default: 20)
    Returns: pandas Series with DEMA values
     

tema() 

tema(data: pd.Series, period: int = 20) -> pd.Series

Description: Triple Exponential Moving Average
Parameters: 

     data (pd.Series): Price series
     period (int): Lookback period (default: 20)
    Returns: pandas Series with TEMA values
     

Momentum Indicators 
rsi() 

rsi(data: pd.Series, period: int = 14) -> pd.Series

Description: Relative Strength Index
Parameters: 

     data (pd.Series): Price series
     period (int): Lookback period (default: 14)
    Returns: pandas Series with RSI values (0-100)
     

stochastic() 

stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
          k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]

Description: Stochastic Oscillator
Parameters: 

     high (pd.Series): High price series
     low (pd.Series): Low price series
     close (pd.Series): Close price series
     k_period (int): %K period (default: 14)
     d_period (int): %D period (default: 3)
    Returns: Tuple of (%K series, %D series)
     

srsi() 

srsi(data: pd.Series, period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]

Description: Stochastic RSI
Parameters: 

     data (pd.Series): Price series
     period (int): RSI period (default: 14)
     d_period (int): %D period (default: 3)
    Returns: Tuple of (SRSI-K series, SRSI-D series)
     

macd() 

macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, 
     signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]

Description: Moving Average Convergence Divergence
Parameters: 

     data (pd.Series): Price series
     fast_period (int): Fast EMA period (default: 12)
     slow_period (int): Slow EMA period (default: 26)
     signal_period (int): Signal line period (default: 9)
    Returns: Tuple of (MACD line, Signal line, Histogram)
     

cci() 

cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series

Description: Commodity Channel Index
Parameters: 

     high (pd.Series): High price series
     low (pd.Series): Low price series
     close (pd.Series): Close price series
     period (int): Lookback period (default: 20)
    Returns: pandas Series with CCI values
     

williams_r() 

williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series

Description: Williams %R Indicator
Parameters: 

     high (pd.Series): High price series
     low (pd.Series): Low price series
     close (pd.Series): Close price series
     period (int): Lookback period (default: 14)
    Returns: pandas Series with Williams %R values (-100 to 0)
     

Volatility Indicators 
atr() 

atr(high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series, 
    period: int = 14) -> pd.Series

Description: Average True Range
Parameters: 

     high_prices (pd.Series): High price series
     low_prices (pd.Series): Low price series
     close_prices (pd.Series): Close price series
     period (int): Lookback period (default: 14)
    Returns: pandas Series with ATR values
     

Note: Contains debug prints that should be removed in production 
Signals Library API 
Location 

simple_strategy.strategies.signals_library 
Status: HAS CRITICAL ISSUES ❌ 

The signals library has inconsistent return types and parameter passing problems. 
Basic Signals 
overbought__oversold()

overbought_oversold(indicator, overbought=70, oversold=30) -> pd.Series

Description: Generate overbought/oversold signals
Parameters: 

     indicator (pd.Series): Indicator series (RSI, Stochastic, etc.)
     overbought (int): Overbought threshold (default: 70)
     oversold (int): Oversold threshold (default: 30)
    Returns: pandas Series with 'BUY', 'SELL', or 'HOLD' signals
     

Known Issues: 

     Returns string values instead of numeric
     Causes inconsistency in signal processing
     

ma_crossover() 

ma_crossover(fast_ma, slow_ma) -> pd.Series

Description: Generate MA crossover signals
Parameters: 

     fast_ma (pd.Series): Fast moving average series
     slow_ma (pd.Series): Slow moving average series
    Returns: pandas Series with 'BUY', 'SELL', or 'HOLD' signals
     

Known Issues: 

     Returns string values instead of numeric
     Crossover detection logic may have edge cases
     

Advanced Signals 
macd_signals() 

macd_signals(macd_line: pd.Series, signal_line: pd.Series, 
             histogram: pd.Series = None) -> pd.Series

Description: Generate MACD-based signals
Parameters: 

     macd_line (pd.Series): MACD line
     signal_line (pd.Series): Signal line
     histogram (pd.Series): MACD histogram (optional)
    Returns: pandas Series with 1=BUY, -1=SELL, 0=HOLD
     

Note: Returns numeric values, which is inconsistent with other signals 
bollinger_bands_signals() 

bollinger_bands_signals(price: pd.Series, upper_band: pd.Series, 
                       lower_band: pd.Series, middle_band: pd.Series = None) -> pd.Series

Description: Generate Bollinger Bands signals
Parameters: 

     price (pd.Series): Price series
     upper_band (pd.Series): Upper Bollinger Band
     lower_band (pd.Series): Lower Bollinger Band
     middle_band (pd.Series): Middle Bollinger Band (optional)
    Returns: pandas Series with 1=BUY, -1=SELL, 0=HOLD
     

Note: Returns numeric values, which is inconsistent with other signals 
stochastic_signals() 

stochastic_signals(k_percent: pd.Series, d_percent: pd.Series, 
                  overbought: float=80, oversold: float=20) -> pd.Series

Description: Generate Stochastic signals
Parameters: 

     k_percent (pd.Series): %K line
     d_percent (pd.Series): %D line
     overbought (float): Overbought threshold (default: 80)
     oversold (float): Oversold threshold (default: 20)
    Returns: pandas Series with 'BUY', 'SELL', or 'HOLD' signals
     

Known Issues: 

     Returns string values instead of numeric
     Inconsistent with macd_signals and bollinger_bands_signals
     

Advanced Signals 
divergence_signals() 

divergence_signals(price: pd.Series, indicator: pd.Series, 
                  lookback_period: int = 20) -> pd.Series

Description: Generate divergence signals
Parameters: 

     price (pd.Series): Price series
     indicator (pd.Series): Indicator series
     lookback_period (int): Period to check for divergence (default: 20)
    Returns: pandas Series with 1=BULLISH_DIVERGENCE, -1=BEARISH_DIVERGENCE, 0=NO_DIVERGENCE
     

Note: Returns numeric values, but complex logic may have issues 
breakout_signals() 

breakout_signals(price: pd.Series, resistance: pd.Series, 
                support: pd.Series, penetration_pct: float = 0.01) -> pd.Series

Description: Generate breakout signals
Parameters: 

     price (pd.Series): Price series
     resistance (pd.Series): Resistance level series
     support (pd.Series): Support level series
     penetration_pct (float): Penetration percentage required (default: 0.01)
    Returns: pandas Series with 1=BUY_BREAKOUT, -1=SELL_BREAKOUT, 0=NO_BREAKOUT
     

Note: Returns numeric values, but breakout detection may be unreliable 
trend_strength_signals() 

trend_strength_signals(price: pd.Series, short_ma: pd.Series, 
                       long_ma: pd.Series, adx: pd.Series = None, 
                       adx_threshold: float = 25) -> pd.Series

Description: Generate trend strength signals
Parameters: 

     price (pd.Series): Price series
     short_ma (pd.Series): Short moving average
     long_ma (pd.Series): Long moving average
     adx (pd.Series): ADX indicator (optional)
     adx_threshold (float): ADX threshold for strong trend (default: 25)
    Returns: pandas Series with 1=STRONG_UPTREND, -1=STRONG_DOWNTREND, 0=WEAK_TREND
     

Note: Returns numeric values, but ADX integration may have issues 
Combination Signals 
majority_vote_signals() 

majority_vote_signals(signal_list: List[pd.Series]) -> pd.Series

Description: Generate majority vote combination of signals
Parameters: 

     signal_list (List[pd.Series]): List of signal series to combine
    Returns: Combined signal series with 1=BUY, -1=SELL, 0=HOLD
     

Known Issues: 

     Tries to handle both string and numeric signals but is fragile
     May not work correctly with mixed signal types
     

weighted_signals() 

weighted_signals(signal_list: List[Tuple[pd.Series, float]]) -> pd.Series

    rectly
     

Error Handling 
Common Exceptions 
StrategyBuilderError 

Base exception for StrategyBuilder errors 
IndicatorIntegrationError 

Raised when indicators cannot be integrated into DataFrame 
SignalProcessingError 

Raised when signal processing fails 
ParameterValidationError 

Raised when parameters are invalid 
Error Messages 
"Indicator not found in DataFrame" 

Cause: StrategyBuilder calculated indicators but didn't integrate them
Solution: This is a known issue that needs to be fixed in the codebase 
"Signal function missing required positional argument" 

Cause: Signal functions called with wrong parameters
Solution: Check signal rule definitions and parameter passing 
"No signal rules defined" 

Cause: StrategyBuilder requires at least one signal rule
Solution: Add at least one signal rule before building 
Usage Examples 
Manual Indicator Calculation (Working) 

import pandas as pd
from simple_strategy.strategies.indicators_library import rsi, sma

# Load data
df = pd.read_csv('data/BTCUSDT_1h.csv')

# Calculate indicators (this works)
rsi_values = rsi(df['close'], period=14)
sma_values = sma(df['close'], period=20)

print(f"RSI: {len(rsi_values)} values")
print(f"SMA: {len(sma_values)} values")

Manual Signal Calculation (Has Issues)

from simple_strategy.strategies.signals_library import overbought_oversold, macd_signals

# This works but returns strings
rsi_signals = overbought_oversold(rsi_values, 70, 30)
print(f"RSI signals: {rsi_signals.value_counts()}")

# This works but returns numeric (inconsistent)
macd_line, signal_line, histogram = macd(df['close'])
macd_signals_result = macd_signals(macd_line, signal_line)
print(f"MACD signals: {macd_signals_result.value_counts()}")

StrategyBuilder Usage (Has Issues)

from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import rsi
from simple_strategy.strategies.signals_library import overbought_oversold

# This creates a strategy but it won't work properly
builder = StrategyBuilder(['BTCUSDT'], ['1h'])
builder.add_indicator('rsi', rsi, period=14)
builder.add_signal_rule('rsi_signal', overbought_oversold, 
                       indicator='rsi', overbought=70, oversold=30)
builder.set_signal_combination('majority_vote')
strategy = builder.build()  # Creates strategy but generates 0 trades

Version Information 
Current Version 

     StrategyBuilder: 1.0.0 (Redesigned with Clear API)
     Indicators Library: 1.0.0 (Complete)
     Signals Library: 1.0.0 (Has Issues)
     

Compatibility 

     Python: 3.8+
     pandas: 1.0+
     numpy: 1.0+
     

Dependencies 

pandas>=1.0.0
numpy>=1.0.0

Future API Changes 
Planned Changes 

     

    Standardize Signal Return Types 
         All signals will return numeric values (1, -1, 0)
         Remove string-based signal returns
         
     

    Fix StrategyBuilder Integration 
         Proper indicator integration into DataFrames
         Correct signal parameter passing
         Improved error handling
         
     

    Add Type Hints 
         Comprehensive type hints for all methods
         Better IDE support and error detection
         
     

    Improve Error Messages 
         More descriptive error messages
         Better debugging information
         
     

Breaking Changes Expected 

     Signal function return types will change
     StrategyBuilder.build() behavior will change
     Error handling will be more strict
     

