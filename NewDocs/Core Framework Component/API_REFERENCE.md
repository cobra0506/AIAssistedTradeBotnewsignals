# Core Framework Component - API_REFERENCE.md

## Complete API Documentation

This document provides comprehensive API documentation for all classes, methods, and interfaces in the Core Framework component.

## Table of Contents

1. [StrategyBase Class](#strategybase-class)
2. [DataFeeder Class](#datafeeder-class)
3. [Utility Functions](#utility-functions)
4. [Data Structures](#data-structures)
5. [Exceptions](#exceptions)
6. [Configuration Parameters](#configuration-parameters)

---

## StrategyBase Class

### Class Overview

```python
class StrategyBase(ABC)

The abstract base class for all trading strategies in the AIAssistedTradeBot system. 
Constructor 

def __init__(self, name: str, symbols: List[str], timeframes: List[str], config: Dict[str, Any])

Parameters: 

     name (str): Unique identifier for the strategy
     symbols (List[str]): List of trading symbols (e.g., ['BTCUSDT', 'ETHUSDT'])
     timeframes (List[str]): List of timeframes (e.g., ['1m', '5m', '1h'])
     config (Dict[str, Any]): Configuration dictionary with strategy parameters
     

Returns: None 

Raises: None 

Example: 

config = {
    'initial_balance': 10000.0,
    'max_risk_per_trade': 0.01,
    'max_positions': 3
}
strategy = MyStrategy('MyStrategy', ['BTCUSDT'], ['1m'], config)

Abstract Methods 
generate_signals 

@abstractmethod
def generate_signals(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, str]]

Generate trading signals for all symbols and timeframes. Must be implemented by subclasses. 

Parameters: 

     data (Dict[str, Dict[str, pd.DataFrame]]): Nested dictionary containing market data
         Structure: {symbol: {timeframe: DataFrame}}
         DataFrame columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
         
     

Returns: 

     Dict[str, Dict[str, str]]: Dictionary containing trading signals
         Structure: {symbol: {timeframe: signal}}
         Signal values: 'BUY', 'SELL', or 'HOLD'
         
     

Raises: None 

Example Implementation: 

def generate_signals(self, data):
    signals = {}
    for symbol in self.symbols:
        signals[symbol] = {}
        for timeframe in self.timeframes:
            df = data[symbol][timeframe]
            # Simple RSI strategy example
            if df['rsi'].iloc[-1] < 30:
                signals[symbol][timeframe] = 'BUY'
            elif df['rsi'].iloc[-1] > 70:
                signals[symbol][timeframe] = 'SELL'
            else:
                signals[symbol][timeframe] = 'HOLD'
    return signals

Public Methods 
calculate_position_size 

def calculate_position_size(self, symbol: str, current_price: float = None, signal_strength: float = 1.0) -> float

Calculate position size based on risk management rules. 

Parameters: 

     symbol (str): Trading symbol (e.g., 'BTCUSDT')
     current_price (float, optional): Current price of the asset. Defaults to None
     signal_strength (float, optional): Strength of the signal (0.0 to 1.0). Defaults to 1.0
     

Returns: 

     float: Position size in units of the asset
     

Raises: None 

Example: 

position_size = strategy.calculate_position_size('BTCUSDT', 50000.0, 0.8)
print(f"Position size: {position_size} BTC")

validate_signal

def validate_signal(self, symbol: str, signal: str, data: Dict[str, pd.DataFrame]) -> bool

Validate signal against risk management rules. 

Parameters: 

     symbol (str): Trading symbol
     signal (str): Trading signal ('BUY', 'SELL', or 'HOLD')
     data (Dict[str, pd.DataFrame]): Current market data
     

Returns: 

     bool: True if signal is valid, False otherwise
     

Raises: None 

Example: 

is_valid = strategy.validate_signal('BTCUSDT', 'BUY', data)
if is_valid:
    # Execute trade
else:
    # Skip trade

get_strategy_state

def get_strategy_state(self) -> Dict[str, Any]

Get current strategy state for logging and monitoring. 

Parameters: None 

Returns: 

     Dict[str, Any]: Dictionary containing strategy state information
         Keys: 'name', 'balance', 'initial_balance', 'total_return', 'open_positions', 'total_trades'
         
     

Raises: None 

Example: 

state = strategy.get_strategy_state()
print(f"Strategy {state['name']} has {state['open_positions']} open positions")
print(f"Total return: {state['total_return']:.2%}")

Protected Methods 
_calculate_portfolio_risk 

def _calculate_portfolio_risk(self) -> float

Calculate current portfolio risk percentage. 

Parameters: None 

Returns: 

     float: Current portfolio risk as a percentage (0.0 to 1.0)
     

Raises: None 

Example: 

portfolio_risk = strategy._calculate_portfolio_risk()
print(f"Current portfolio risk: {portfolio_risk:.2%}")

Properties 
name 

@property
def name(self) -> str

Get the strategy name. 

Returns: str: Strategy name 
symbols 

@property
def symbols(self) -> List[str]

Get the list of trading symbols. 

Returns: List[str]: Trading symbols 
timeframes 

@property
def timeframes(self) -> List[str]

Get the list of timeframes. 

Returns: List[str]: Timeframes 
balance 

@property
def balance(self) -> float

Get current account balance. 

Returns: float: Current balance 
positions 

@property
def positions(self) -> Dict[str, Any]

Get current open positions. 

Returns: Dict[str, Any]: Open positions dictionary 
DataFeeder Class 
Class Overview 

class DataFeeder

Data management and loading system for the AIAssistedTradeBot. 
Constructor 

def __init__(self, data_dir: str = 'data', memory_limit_percent: float = 90)

Parameters: 

     data_dir (str, optional): Directory containing CSV data files. Defaults to 'data'
     memory_limit_percent (float, optional): Maximum memory usage percentage (0-100). Defaults to 90
     

Returns: None 

Raises: None 

Example: 

data_feeder = DataFeeder(data_dir='data', memory_limit_percent=80)

Public Methods 
get_data_for_symbols 

def get_data_for_symbols(self, symbols: List[str], timeframes: List[str], start_date: Union[str, datetime], end_date: Union[str, datetime]) -> Dict[str, Dict[str, pd.DataFrame]]

Return cached data for multiple symbols/timeframes, filtered by date range. 

Parameters: 

     symbols (List[str]): List of trading symbols
     timeframes (List[str]): List of timeframes
     start_date (Union[str, datetime]): Start date for data filtering
     end_date (Union[str, datetime]): End date for data filtering
     

Returns: 

     Dict[str, Dict[str, pd.DataFrame]]: Dictionary with filtered data
         Structure: {symbol: {timeframe: DataFrame}}
         
     

Raises: None 

Example: 

data = data_feeder.get_data_for_symbols(
    symbols=['BTCUSDT', 'ETHUSDT'],
    timeframes=['1m', '5m'],
    start_date='2023-01-01',
    end_date='2023-12-31'
)

clear_cache

def clear_cache(self) -> None

Clear the data cache to free memory. 

Parameters: None 

Returns: None 

Raises: None 

Example: 

data_feeder.clear_cache()

get_cache_info

def get_cache_info(self) -> Dict[str, Any]

Get information about the current cache state. 

Parameters: None 

Returns: 

     Dict[str, Any]: Cache information dictionary
         Keys: 'cached_symbols', 'cached_timeframes', 'memory_usage'
         
     

Raises: None 

Example: 

cache_info = data_feeder.get_cache_info()
print(f"Cached symbols: {cache_info['cached_symbols']}")

Protected Methods 
_check_memory_usage 

def _check_memory_usage(self) -> bool

Check if current memory usage is within limits. 

Parameters: None 

Returns: 

     bool: True if memory usage is acceptable, False otherwise
     

Raises: None 

Example: 

if not data_feeder._check_memory_usage():
    print("Memory usage approaching limits")

_load_csv_file

def _load_csv_file(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]

Load a single CSV file and return as DataFrame. 

Parameters: 

     symbol (str): Trading symbol (e.g., 'BTCUSDT')
     timeframe (str): Timeframe (e.g., '1m', '5m')
     

Returns: 

     Optional[pd.DataFrame]: DataFrame with loaded data or None if file doesn't exist
     

Raises: None 

Example: 

df = data_feeder._load_csv_file('BTCUSDT', '1m')
if df is not None:
    print(f"Loaded {len(df)} data points")

Properties 
data_dir 

@property
def data_dir(self) -> Path

Get the data directory path. 

Returns: Path: Data directory path 
memory_limit_percent 

@property
def memory_limit_percent(self) -> float

Get the memory limit percentage. 

Returns: float: Memory limit percentage 
Utility Functions 
validate_data_format 

def validate_data_format(df: pd.DataFrame) -> bool

Validate that a DataFrame has the expected format for market data. 

Parameters: 

     df (pd.DataFrame): DataFrame to validate
     

Returns: 

     bool: True if format is valid, False otherwise
     

Raises: None 

Example: 

is_valid = validate_data_format(data_frame)
if not is_valid:
    print("Invalid data format")

standardize_symbol

def standardize_symbol(symbol: str) -> str

Standardize symbol format for consistency. 

Parameters: 

     symbol (str): Trading symbol
     

Returns: 

     str: Standardized symbol
     

Raises: None 

Example: 

standardized = standardize_symbol('btcusdt')
print(standardized)  # 'BTCUSDT'

Data Structures 
MarketData Structure 

Dict[str, Dict[str, pd.DataFrame]]

Structure for holding market data: 

     Outer dictionary keys: Trading symbols (e.g., 'BTCUSDT')
     Inner dictionary keys: Timeframes (e.g., '1m', '5m')
     DataFrame columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
     

Example: 

market_data = {
    'BTCUSDT': {
        '1m': pd.DataFrame(...),  # 1-minute data
        '5m': pd.DataFrame(...)   # 5-minute data
    },
    'ETHUSDT': {
        '1m': pd.DataFrame(...),  # 1-minute data
        '5m': pd.DataFrame(...)   # 5-minute data
    }
}

Signal Structure

Dict[str, Dict[str, str]]

Structure for holding trading signals: 

     Outer dictionary keys: Trading symbols (e.g., 'BTCUSDT')
     Inner dictionary keys: Timeframes (e.g., '1m', '5m')
     Signal values: 'BUY', 'SELL', or 'HOLD'
     

Example: 

signals = {
    'BTCUSDT': {
        '1m': 'BUY',
        '5m': 'HOLD'
    },
    'ETHUSDT': {
        '1m': 'SELL',
        '5m': 'HOLD'
    }
}

Position Structure

Dict[str, Any]

Structure for holding position information: 

     'symbol' (str): Trading symbol
     'quantity' (float): Position quantity
     'entry_price' (float): Entry price
     'current_price' (float): Current price
     'pnl' (float): Profit and loss
     'timestamp' (datetime): Position timestamp
     

Example: 

position = {
    'symbol': 'BTCUSDT',
    'quantity': 0.001,
    'entry_price': 50000.0,
    'current_price': 51000.0,
    'pnl': 10.0,
    'timestamp': datetime.now()
}

Exceptions 
DataLoadError 

class DataLoadError(Exception)

Raised when there is an error loading data. 

Example: 

try:
    data = data_feeder.get_data_for_symbols(...)
except DataLoadError as e:
    print(f"Data loading error: {e}")

StrategyError

class StrategyError(Exception)

Raised when there is an error in strategy execution. 

Example: 

try:
    signals = strategy.generate_signals(data)
except StrategyError as e:
    print(f"Strategy error: {e}")

MemoryLimitError

class MemoryLimitError(Exception)

Raised when memory limits are exceeded. 

Example: 

try:
    data = data_feeder.get_data_for_symbols(...)
except MemoryLimitError as e:
    print(f"Memory limit exceeded: {e}")

Configuration Parameters 
Strategy Configuration 

Dict[str, Any]

Configuration parameters for strategies: 

Required Parameters: 

     'initial_balance' (float): Initial account balance
     'max_risk_per_trade' (float): Maximum risk per trade (0.0 to 1.0)
     'max_positions' (int): Maximum number of concurrent positions
     'max_portfolio_risk' (float): Maximum portfolio risk (0.0 to 1.0)
     

Optional Parameters: 

     'stop_loss_percent' (float): Stop loss percentage
     'take_profit_percent' (float): Take profit percentage
     'commission_rate' (float): Trading commission rate
     

Example: 

strategy_config = {
    'initial_balance': 10000.0,
    'max_risk_per_trade': 0.01,      # 1% risk per trade
    'max_positions': 3,               # Maximum 3 concurrent positions
    'max_portfolio_risk': 0.10,      # 10% maximum portfolio risk
    'stop_loss_percent': 0.02,       # 2% stop loss
    'take_profit_percent': 0.04,     # 4% take profit
    'commission_rate': 0.001         # 0.1% commission
}

DataFeeder Configuration

Dict[str, Any]

Configuration parameters for DataFeeder: 

Required Parameters: 

     'data_dir' (str): Directory containing CSV data files
     'memory_limit_percent' (float): Maximum memory usage percentage (0-100)
     

Optional Parameters: 

     'enable_caching' (bool): Enable data caching
     'cache_size_limit' (int): Maximum number of cached DataFrames
     'date_format' (str): Date format for parsing
     

Example: 

data_feeder_config = {
    'data_dir': 'data',
    'memory_limit_percent': 80,
    'enable_caching': True,
    'cache_size_limit': 1000,
    'date_format': '%Y-%m-%d %H:%M:%S'
}

Usage Examples 
Complete Strategy Implementation 

from simple_strategy.shared.strategy_base import StrategyBase
from simple_strategy.shared.data_feeder import DataFeeder
import pandas as pd
from typing import Dict, List, Any

class RSIStrategy(StrategyBase):
    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        super().__init__("RSI Strategy", symbols, timeframes, config)
        self.rsi_period = config.get('rsi_period', 14)
        self.overbought = config.get('overbought', 70)
        self.oversold = config.get('oversold', 30)
    
    def generate_signals(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, str]]:
        signals = {}
        
        for symbol in self.symbols:
            signals[symbol] = {}
            
            for timeframe in self.timeframes:
                df = data[symbol][timeframe]
                
                # Calculate RSI if not already present
                if 'rsi' not in df.columns:
                    df['rsi'] = self._calculate_rsi(df['close'], self.rsi_period)
                
                # Generate signals based on RSI
                rsi_value = df['rsi'].iloc[-1]
                
                if rsi_value <= self.oversold:
                    signals[symbol][timeframe] = 'BUY'
                elif rsi_value >= self.overbought:
                    signals[symbol][timeframe] = 'SELL'
                else:
                    signals[symbol][timeframe] = 'HOLD'
        
        return signals
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

# Usage example
if __name__ == "__main__":
    # Initialize data feeder
    data_feeder = DataFeeder(data_dir='data')
    
    # Strategy configuration
    config = {
        'initial_balance': 10000.0,
        'max_risk_per_trade': 0.01,
        'max_positions': 3,
        'max_portfolio_risk': 0.10,
        'rsi_period': 14,
        'overbought': 70,
        'oversold': 30
    }
    
    # Create strategy
    strategy = RSIStrategy(['BTCUSDT'], ['1m'], config)
    
    # Get data
    data = data_feeder.get_data_for_symbols(
        symbols=['BTCUSDT'],
        timeframes=['1m'],
        start_date='2023-01-01',
        end_date='2023-12-31'
    )
    
    # Generate signals
    signals = strategy.generate_signals(data)
    
    # Validate and execute
    for symbol in signals:
        for timeframe in signals[symbol]:
            signal = signals[symbol][timeframe]
            if strategy.validate_signal(symbol, signal, data[symbol]):
                position_size = strategy.calculate_position_size(symbol)
                print(f"{symbol} {timeframe}: {signal} - Position size: {position_size}")
    
    # Get strategy state
    state = strategy.get_strategy_state()
    print(f"Strategy state: {state}")

Data Management Example

from simple_strategy.shared.data_feeder import DataFeeder
from datetime import datetime, timedelta

# Initialize data feeder with custom configuration
data_feeder = DataFeeder(
    data_dir='data',
    memory_limit_percent=75
)

# Get data for multiple symbols and timeframes
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

data = data_feeder.get_data_for_symbols(
    symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
    timeframes=['1m', '5m', '15m'],
    start_date=start_date,
    end_date=end_date
)

# Check cache information
cache_info = data_feeder.get_cache_info()
print(f"Cache info: {cache_info}")

# Clear cache if needed
if not data_feeder._check_memory_usage():
    print("Memory limit approached, clearing cache")
    data_feeder.clear_cache()

# Access specific data
btc_1m_data = data['BTCUSDT']['1m']
print(f"BTC 1m data shape: {btc_1m_data.shape}")
print(f"Date range: {btc_1m_data.index.min()} to {btc_1m_data.index.max()}")

Error Handling Example

from simple_strategy.shared.strategy_base import StrategyBase, StrategyError
from simple_strategy.shared.data_feeder import DataFeeder, DataLoadError, MemoryLimitError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_strategy_execution():
    try:
        # Initialize components
        data_feeder = DataFeeder(data_dir='data')
        
        config = {
            'initial_balance': 10000.0,
            'max_risk_per_trade': 0.01,
            'max_positions': 3
        }
        
        strategy = MyStrategy(['BTCUSDT'], ['1m'], config)
        
        # Get data with error handling
        try:
            data = data_feeder.get_data_for_symbols(
                symbols=['BTCUSDT'],
                timeframes=['1m'],
                start_date='2023-01-01',
                end_date='2023-12-31'
            )
        except MemoryLimitError as e:
            logger.error(f"Memory limit exceeded: {e}")
            data_feeder.clear_cache()
            return None
        except DataLoadError as e:
            logger.error(f"Data loading error: {e}")
            return None
        
        # Generate signals with error handling
        try:
            signals = strategy.generate_signals(data)
        except StrategyError as e:
            logger.error(f"Strategy error: {e}")
            return None
        
        # Process signals
        processed_signals = {}
        for symbol in signals:
            processed_signals[symbol] = {}
            for timeframe in signals[symbol]:
                signal = signals[symbol][timeframe]
                try:
                    if strategy.validate_signal(symbol, signal, data[symbol]):
                        processed_signals[symbol][timeframe] = signal
                except Exception as e:
                    logger.warning(f"Signal validation error for {symbol} {timeframe}: {e}")
                    processed_signals[symbol][timeframe] = 'HOLD'
        
        return processed_signals
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None

# Execute with error handling
result = safe_strategy_execution()
if result:
    print("Strategy execution completed successfully")
else:
    print("Strategy execution failed")

Version History 
Version 1.0 (Current) 

     Initial release of Core Framework component
     StrategyBase abstract class implementation
     DataFeeder class with memory management
     Comprehensive error handling
     Full test coverage
     

Support and Contact 

For issues, questions, or contributions related to the Core Framework component, please refer to the main project documentation or create an issue in the GitHub repository. 

