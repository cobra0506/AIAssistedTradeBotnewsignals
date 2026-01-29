# Strategy Building Component - OVERVIEW

## Module Purpose and Scope

The Strategy Building Component provides a framework for creating trading strategies using a building-block approach. It enables developers to combine technical indicators with signal processing logic to create customized trading strategies.

## Current Status: PARTIALLY WORKING ⚠️

**IMPORTANT**: This component is NOT fully functional despite some documentation claiming completion. There are known issues that prevent proper strategy execution.

### What's Currently Working ✅

1. **GUI Integration**
   - Strategy detection from files named `Strategy_*.py`
   - Parameter parsing from `STRATEGY_PARAMETERS` dictionary
   - Parameter assignment in GUI (int, float, string, options)
   - Parameter passing to `create_strategy()` function

2. **Data Management**
   - Symbol and timeframe assignment from GUI
   - Data file detection and loading
   - Date range filtering
   - Correct data structure (OHLCV format)

3. **Indicator Functions**
   - Manual indicator calculation works
   - Functions from `indicators_library.py` are functional
   - Example: `sma(df['close'], period=20)` returns valid pandas Series

### What's NOT Working ❌

1. **StrategyBuilder Indicator Integration**
   - Indicators calculated but not properly integrated into DataFrame
   - Error: `test_sma: NOT FOUND in DataFrame`
   - StrategyBuilder calculates indicators but doesn't integrate them correctly

2. **Signal Generation**
   - Signal functions called with wrong parameters
   - Error: `simple_buy_signal() missing 1 required positional argument`
   - Signal functions don't receive expected indicator data

3. **Signal Output**
   - Expected: pandas Series with trading signals
   - Actual: Simple string `'HOLD'`
   - Results in 0 trades in backtest

## Component Architecture

### Core Classes

1. **StrategyBuilder** (`simple_strategy/strategies/strategy_builder.py`)
   - Main class for constructing strategies
   - Handles indicator and signal rule management
   - Provides fluent API for strategy creation

2. **Indicators Library** (`simple_strategy/strategies/indicators_library.py`)
   - Comprehensive technical indicator implementations
   - Trend indicators: SMA, EMA, WMA, DEMA, TEMA
   - Momentum indicators: RSI, Stochastic, SRSI, MACD, CCI, Williams %R
   - Volatility indicators: ATR

3. **Signals Library** (`simple_strategy/strategies/signals_library.py`)
   - Signal processing functions with KNOWN ISSUES
   - Inconsistent return types (some strings, some numeric)
   - Basic signals: overbought_oversold, ma_crossover
   - Advanced signals: macd_signals, bollinger_bands_signals, stochastic_signals

## Key Limitations

1. **Signal Type Inconsistency**
   - Some functions return strings: 'BUY', 'SELL', 'HOLD'
   - Others return numeric: 1, -1, 0
   - This causes processing issues in StrategyBuilder

2. **Multi-Component Indicator Handling**
   - MACD and Bollinger Bands have special handling requirements
   - Component references are complex and error-prone

3. **Testing Status**
   - Manual indicator calculation: WORKING
   - Strategy integration: NOT WORKING
   - Backtest execution: GENERATES 0 TRADES

## Usage Recommendations

### For Current State
1. Use manual indicator calculation for testing
2. Avoid complex multi-indicator strategies
3. Test each component individually before integration
4. Use simple strategies first (RSI or single MA crossover)

### For Future Development
1. Fix signal return type consistency
2. Resolve StrategyBuilder indicator integration issues
3. Improve error handling and debugging
4. Add comprehensive integration tests

## Files and Locations
simple_strategy/strategies/
├── strategy_builder.py          # Main StrategyBuilder class
├── indicators_library.py       # Technical indicators (WORKING)
├── signals_library.py          # Signal functions (HAS ISSUES)
├── shared/strategy_base.py     # Base strategy class
└── examples/                   # Example strategies


## Dependencies

- pandas for data manipulation
- numpy for numerical operations
- Standard Python libraries
- No external trading platform dependencies

## Next Steps for Completion

1. **Priority 1**: Fix signal function return type consistency
2. **Priority 2**: Resolve StrategyBuilder indicator integration
3. **Priority 3**: Add comprehensive error handling
4. **Priority 4**: Create working integration tests

