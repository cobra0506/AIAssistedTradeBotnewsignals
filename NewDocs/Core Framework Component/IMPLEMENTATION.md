# Core Framework Component - IMPLEMENTATION.md

## Detailed Implementation Guide

This document provides a comprehensive implementation guide for the Core Framework component, including architecture details, implementation patterns, and best practices.

## Architecture Overview

### Core Design Principles

1. **Abstract Base Class Pattern**: StrategyBase provides the foundation for all strategy implementations
2. **Data Independence**: CSV files serve as the universal data exchange format
3. **Memory Management**: Intelligent caching and memory monitoring prevent system overload
4. **Loose Coupling**: Components communicate through well-defined interfaces
5. **Extensibility**: Easy to add new features without modifying existing code

### Component Structure
simple_strategy/shared/
├── init.py          # Package initialization
├── strategy_base.py     # Abstract strategy base class
└── data_feeder.py       # Data management and loading


## StrategyBase Implementation

### Class Architecture

```python
class StrategyBase(ABC):
    """
    Abstract base class for all trading strategies.
    Provides common functionality and enforces consistent interface.
    """

Key Implementation Details 
1. Initialization (__init__) 

def __init__(self, name: str, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
    """
    Initialize strategy with configuration.
    
    Args:
        name: Strategy name for identification
        symbols: List of trading symbols (e.g., ['BTCUSDT', 'ETHUSDT'])
        timeframes: List of timeframes (e.g., ['1m', '5m'])
        config: Strategy configuration parameters
    """

Implementation Details: 

     Sets up strategy state including positions, balance, and configuration
     Initializes risk management parameters
     Sets up performance tracking structures
     Configures logging for the strategy instance
     

2. Abstract Signal Generation (generate_signals) 

@abstractmethod
def generate_signals(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, str]]:
    """
    Generate trading signals for all symbols and timeframes.
    Must be implemented by subclasses.
    
    Args:
        data: Nested dictionary {symbol: {timeframe: DataFrame}}
    
    Returns:
        Dictionary {symbol: {timeframe: signal}} where signal is 'BUY', 'SELL', or 'HOLD'
    """
    pass

Implementation Requirements: 

     Must be implemented by all concrete strategy classes
     Should analyze market data and generate trading signals
     Must handle multiple symbols and timeframes
     Should return signals in standardized format
     

3. Position Sizing (calculate_position_size) 

def calculate_position_size(self, symbol: str, current_price: float = None, signal_strength: float = 1.0) -> float:
    """
    Calculate position size based on risk management rules.
    
    Args:
        symbol: Trading symbol
        current_price: Current price of the asset
        signal_strength: Strength of the signal (0.0 to 1.0)
    
    Returns:
        Position size in units of the asset
    """

Implementation Logic: 

     Risk Calculation: Calculates risk amount based on account balance and risk percentage
     Position Size: Converts risk amount to asset units using current price
     Maximum Limits: Ensures position doesn't exceed maximum size limits
     Asset-Specific Rounding: Applies appropriate decimal precision for different assets
     

4. Signal Validation (validate_signal) 

def validate_signal(self, symbol: str, signal: str, data: Dict[str, pd.DataFrame]) -> bool:
    """
    Validate signal against risk management rules.
    
    Args:
        symbol: Trading symbol
        signal: Trading signal ('BUY', 'SELL', 'HOLD')
        data: Current market data
    
    Returns:
        True if signal is valid, False otherwise
    """

Validation Rules: 

     HOLD signals: Always valid
     BUY signals: Checks maximum positions and portfolio risk limits
     SELL signals: Verifies position exists for the symbol
     Portfolio Risk: Ensures total portfolio risk doesn't exceed limits
     

5. State Management (get_strategy_state) 

def get_strategy_state(self) -> Dict[str, Any]:
    """
    Get current strategy state for logging and monitoring.
    
    Returns:
        Dictionary with strategy state information
    """

State Information: 

     Strategy name and configuration
     Current balance and performance metrics
     Open positions count
     Total trades executed
     Risk management status
     

DataFeeder Implementation 
Class Architecture 

class DataFeeder:
    """
    Data Feeder component for loading and managing CSV data from the data collection system.
    Supports multiple symbols, timeframes, and memory management.
    """

Key Implementation Details 
1. Initialization (__init__) 

def __init__(self, data_dir: str = 'data', memory_limit_percent: float = 90):
    """
    Initialize data feeder with memory management.
    
    Args:
        data_dir: Directory containing CSV data files
        memory_limit_percent: Maximum memory usage percentage (0-100)
    """

Implementation Details: 

     Sets up data directory path using pathlib
     Initializes memory management parameters
     Creates data and metadata caches
     Configures logging for data operations
     

2. Memory Management (_check_memory_usage) 

def _check_memory_usage(self) -> bool:
    """
    Check if current memory usage is within limits.
    
    Returns:
        bool: True if memory usage is acceptable, False otherwise
    """

Implementation Logic: 

     Uses psutil library to monitor system memory
     Compares current usage against configured limits
     Provides graceful fallback if memory monitoring fails
     Prevents system overload by blocking data loading when limits are exceeded
     

3. CSV File Loading (_load_csv_file) 

def _load_csv_file(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    """
    Load a single CSV file and return as DataFrame.
    Handles both naming conventions: with and without 'm' suffix.
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        timeframe: Timeframe (e.g., '1m', '5m')
    
    Returns:
        DataFrame with loaded data or None if file doesn't exist
    """

Implementation Details: 

     File Naming: Supports multiple naming conventions (BTCUSDT_1.csv, BTCUSDT_1m.csv)
     Data Processing: Converts timestamp columns to datetime objects
     Index Management: Sets datetime as index for time-based operations
     Data Sorting: Ensures chronological order of data points
     Error Handling: Graceful handling of missing or corrupted files
     

4. Multi-Symbol Data Access (get_data_for_symbols) 

def get_data_for_symbols(self, symbols, timeframes, start_date, end_date):
    """
    Return cached data for multiple symbols/timeframes, filtered by date range.
    
    Args:
        symbols: List of trading symbols
        timeframes: List of timeframes
        start_date: Start date for data filtering
        end_date: End date for data filtering
    
    Returns:
        Dictionary with filtered data for all requested symbols/timeframes
    """

Implementation Logic: 

     Date Conversion: Converts string dates to datetime objects
     Cache Checking: Checks if requested data is already in memory cache
     Data Loading: Loads data from CSV files if not in cache
     Date Filtering: Applies date range filtering to loaded data
     Cache Management: Stores loaded data in cache for future access
     Result Assembly: Returns structured data dictionary
     

Implementation Patterns and Best Practices 
1. Strategy Implementation Pattern 

class MyStrategy(StrategyBase):
    def __init__(self, symbols, timeframes, config):
        super().__init__("MyStrategy", symbols, timeframes, config)
        # Strategy-specific initialization
    
    def generate_signals(self, data):
        signals = {}
        for symbol in self.symbols:
            signals[symbol] = {}
            for timeframe in self.timeframes:
                # Strategy logic here
                signals[symbol][timeframe] = self._analyze_symbol(data[symbol][timeframe])
        return signals
    
    def _analyze_symbol(self, df):
        # Implement specific analysis logic
        return 'BUY'  # or 'SELL' or 'HOLD'

2. Data Access Pattern

# Initialize data feeder
data_feeder = DataFeeder(data_dir='data')

# Get data for backtesting
data = data_feeder.get_data_for_symbols(
    symbols=['BTCUSDT', 'ETHUSDT'],
    timeframes=['1m', '5m'],
    start_date='2023-01-01',
    end_date='2023-12-31'
)

# Use data with strategy
strategy = MyStrategy(['BTCUSDT'], ['1m'], config)
signals = strategy.generate_signals(data)

3. Error Handling Pattern

try:
    data = data_feeder.get_data_for_symbols(symbols, timeframes, start_date, end_date)
    if not data or not all(data.values()):
        logger.warning("Insufficient data for strategy execution")
        return None
    
    signals = strategy.generate_signals(data)
    validated_signals = {}
    
    for symbol in signals:
        validated_signals[symbol] = {}
        for timeframe in signals[symbol]:
            if strategy.validate_signal(symbol, signals[symbol][timeframe], data[symbol]):
                validated_signals[symbol][timeframe] = signals[symbol][timeframe]
    
    return validated_signals
    
except Exception as e:
    logger.error(f"Strategy execution error: {e}")
    return None

4. Memory Management Pattern

# Configure memory limits appropriately
data_feeder = DataFeeder(
    data_dir='data',
    memory_limit_percent=75  # Conservative limit for stability
)

# Monitor memory usage in long-running processes
if not data_feeder._check_memory_usage():
    logger.warning("Memory usage approaching limits, consider clearing cache")
    # Implement cache clearing strategy if needed

Configuration Management 
Strategy Configuration 

strategy_config = {
    'initial_balance': 10000.0,
    'max_risk_per_trade': 0.01,  # 1% of balance
    'max_positions': 3,
    'max_portfolio_risk': 0.10,  # 10% of balance
    # Strategy-specific parameters
    'rsi_period': 14,
    'sma_short': 20,
    'sma_long': 50
}

DataFeeder Configuration

data_feeder_config = {
    'data_dir': 'data',
    'memory_limit_percent': 80,
    'enable_caching': True,
    'cache_size_limit': 1000  # Maximum dataframes to cache
}

Testing Implementation 
Unit Testing StrategyBase 

import unittest
from simple_strategy.shared.strategy_base import StrategyBase

class TestStrategy(StrategyBase):
    def generate_signals(self, data):
        return {'BTCUSDT': {'1m': 'HOLD'}}

class TestStrategyBase(unittest.TestCase):
    def setUp(self):
        self.strategy = TestStrategy(['BTCUSDT'], ['1m'], {})
    
    def test_position_sizing(self):
        position_size = self.strategy.calculate_position_size('BTCUSDT', 50000.0)
        self.assertGreater(position_size, 0)
    
    def test_signal_validation(self):
        # Test various signal validation scenarios
        self.assertTrue(self.strategy.validate_signal('BTCUSDT', 'HOLD', {}))

Unit Testing DataFeeder

import unittest
from simple_strategy.shared.data_feeder import DataFeeder

class TestDataFeeder(unittest.TestCase):
    def setUp(self):
        self.data_feeder = DataFeeder(data_dir='test_data')
    
    def test_memory_check(self):
        result = self.data_feeder._check_memory_usage()
        self.assertIsInstance(result, bool)
    
    def test_data_loading(self):
        # Test with actual test data files
        df = self.data_feeder._load_csv_file('BTCUSDT', '1m')
        if df is not None:
            self.assertGreater(len(df), 0)

Performance Considerations 
1. Memory Optimization 

     Use configurable memory limits to prevent system overload
     Implement intelligent caching strategies
     Clear cache when memory limits are approached
     Use efficient data structures (pandas DataFrames)
     

2. Data Loading Optimization 

     Load data only when needed (lazy loading)
     Cache frequently accessed data
     Use date range filtering to minimize data loaded
     Implement parallel data loading for multiple symbols
     

3. Strategy Execution Optimization 

     Minimize data copying between components
     Use vectorized operations where possible
     Implement efficient signal generation algorithms
     Cache intermediate calculations when beneficial
     

Error Handling and Recovery 
1. Data Loading Errors 

     Handle missing or corrupted data files gracefully
     Provide meaningful error messages
     Implement fallback strategies for missing data
     Log errors for debugging and monitoring
     

2. Memory Management Errors 

     Monitor memory usage continuously
     Implement graceful degradation when limits are approached
     Provide warnings before critical limits are reached
     Implement cache clearing strategies
     

3. Strategy Execution Errors 

     Validate inputs before processing
     Handle exceptions in signal generation
     Provide meaningful error messages
     Implement recovery strategies for failed executions
     

Integration Guidelines 
1. Integration with Strategy Builder 

     Extend StrategyBase for all strategy implementations
     Use consistent parameter naming conventions
     Implement required abstract methods
     Follow established signal generation patterns
     

2. Integration with Backtest Engine 

     Ensure strategies return signals in expected format
     Implement proper position sizing and risk management
     Provide strategy state information for reporting
     Handle edge cases and error conditions appropriately
     

3. Integration with Data Collection 

     Use DataFeeder for all data access operations
     Follow established data format conventions
     Implement proper error handling for data operations
     Use memory management features appropriately
     

Debugging and Troubleshooting 
1. Common Issues and Solutions 

Issue: Strategy not generating signals
Solution:  

     Check that generate_signals method is properly implemented
     Verify data format and structure
     Check for exceptions in signal generation logic
     

Issue: Memory usage too high
Solution: 

     Reduce memory_limit_percent configuration
     Clear data cache manually if needed
     Check for data leaks in custom implementations
     

Issue: Data loading failures
Solution: 

     Verify data file existence and format
     Check file permissions and access rights
     Review data directory configuration
     

2. Debugging Tools 

Logging Configuration: 

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

Memory Monitoring:

import psutil
memory = psutil.virtual_memory()
print(f"Memory usage: {memory.percent}%")

Data Inspection:

# Inspect loaded data structure
print(f"Data keys: {list(data.keys())}")
print(f"Data shape: {data['BTCUSDT']['1m'].shape}")

Future Enhancements 
1. Planned Improvements 

     Additional Data Sources: Support for more exchanges and data providers
     Real-time Data Integration: Enhanced real-time data processing capabilities
     Advanced Risk Management: More sophisticated risk calculation methods
     Performance Monitoring: Enhanced performance tracking and reporting
     

2. Extension Points 

     Custom Data Loaders: Plugin architecture for custom data sources
     Advanced Indicators: Extensible indicator calculation framework
     Strategy Templates: Pre-built strategy templates for common patterns
     Configuration Management: Enhanced configuration system with validation
     

Conclusion 

The Core Framework component provides a solid foundation for the AIAssistedTradeBot system, with well-architected classes, comprehensive error handling, and extensive testing coverage. The implementation follows established software engineering best practices and provides a robust platform for strategy development and execution. 
