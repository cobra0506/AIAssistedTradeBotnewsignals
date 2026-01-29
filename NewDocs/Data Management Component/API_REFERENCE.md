# Data Management Component - API Reference

## Overview

This document provides a comprehensive API reference for the Data Management Component, including all classes, methods, and their parameters, return values, and usage examples.

## Table of Contents

1. [HybridTradingSystem](#hybridtradingsystem)
2. [OptimizedDataFetcher](#optimizeddatafetcher)
3. [WebSocketHandler](#websockethandler)
4. [CSVManager](#csvmanager)
5. [DataIntegrityChecker](#dataintegritychecker)
6. [DataFeeder](#datafeeder)
7. [DataCollectionConfig](#datacollectionconfig)

---

## HybridTradingSystem

### Class: `HybridTradingSystem`

**File**: `shared_modules/data_collection/hybrid_system.py`

**Description**: Core orchestrator that coordinates between historical and real-time data collection systems.

### Constructor

```python
def __init__(self, config: DataCollectionConfig)

Parameters: 

     config (DataCollectionConfig): Configuration object containing system settings
     

Returns: HybridTradingSystem instance 
Methods 
async def initialize(self) 

Description: Initialize both historical and real-time data fetchers 

Returns: None 

Example: 

config = DataCollectionConfig()
system = HybridTradingSystem(config)
await system.initialize()

async def fetch_data_hybrid(self, symbols: List[str] = None, timeframes: List[str] = None, days: int = None, mode: str = "full") 

Description: Fetch data combining historical and real-time sources 

Parameters: 

     symbols (List[str], optional): List of trading symbols. Defaults to config.SYMBOLS
     timeframes (List[str], optional): List of timeframes. Defaults to config.TIMEFRAMES
     days (int, optional): Number of days of historical data. Defaults to config.DAYS_TO_FETCH
     mode (str): Data collection mode. Options:
         "full": Fetch all historical data
         "recent": Fetch only 50 most recent entries
         "live": Fetch only real-time data
         
     

Returns: None 

Example: 

# Fetch full historical data for BTC and ETH
await system.fetch_data_hybrid(
    symbols=['BTCUSDT', 'ETHUSDT'],
    timeframes=['1', '5'],
    days=30,
    mode="full"
)

async def save_to_csv(self, directory: str = "data") 

Description: Save all collected data to CSV files 

Parameters: 

     directory (str): Directory path for CSV files. Defaults to "data"
     

Returns: None 

Example: 

await system.save_to_csv(directory="data")

async def update_csv_with_realtime_data(self, directory: str = "data") 

Description: Update CSV files with real-time data 

Parameters: 

     directory (str): Directory path for CSV files. Defaults to "data"
     

Returns: None 

Example: 

await system.update_csv_with_realtime_data()

def get_data(self, symbol: str, timeframe: str, source: str = "memory") -> List[Dict[str, Any]] 

Description: Get data from specified source 

Parameters: 

     symbol (str): Trading symbol (e.g., "BTCUSDT")
     timeframe (str): Timeframe (e.g., "1", "5", "15")
     source (str): Data source. Options:
         "memory": Data from memory cache
         "websocket": Real-time data only
         "csv": Data from CSV files
         "combined": Combined historical + real-time data
         
     

Returns: List[Dict[str, Any]]: List of candle data dictionaries 

Example: 

# Get combined data for BTC 1-minute timeframe
data = system.get_data("BTCUSDT", "1", source="combined")

async def close(self) 

Description: Clean up resources and close connections 

Returns: None 

Example: 

await system.close()

OptimizedDataFetcher 
Class: OptimizedDataFetcher 

File: shared_modules/data_collection/optimized_data_fetcher.py 

Description: High-performance historical data fetching with concurrent processing and rate limiting. 
Constructor 

def __init__(self, config: DataCollectionConfig)

Parameters: 

     config (DataCollectionConfig): Configuration object
     

Returns: OptimizedDataFetcher instance 
Methods 
async def initialize(self) 

Description: Initialize aiohttp session for HTTP requests 

Returns: None 

Example: 

fetcher = OptimizedDataFetcher(config)
await fetcher.initialize()

async def fetch_historical_data_fast(self, symbols: List[str], timeframes: List[str], days: int, limit_50: bool = False) -> bool 

Description: Ultra-fast historical data fetching with optimized chunking 

Parameters: 

     symbols (List[str]): List of trading symbols
     timeframes (List[str]): List of timeframes
     days (int): Number of days of historical data to fetch
     limit_50 (bool): If True, limit to 50 most recent entries. Defaults to False
     

Returns: bool: True if successful, False otherwise 

Example: 

success = await fetcher.fetch_historical_data_fast(
    symbols=['BTCUSDT', 'ETHUSDT'],
    timeframes=['1', '5'],
    days=30,
    limit_50=False
)

async def _get_all_symbols(self) -> List[str] 

Description: Get all available linear symbols from Bybit exchange 

Returns: List[str]: List of available trading symbols 

Example: 

all_symbols = await fetcher._get_all_symbols()
print(f"Available symbols: {len(all_symbols)}")

def get_memory_data(self) -> Dict[str, List[Dict[str, Any]]] 

Description: Get data currently stored in memory 

Returns: Dict[str, List[Dict[str, Any]]]: Dictionary with symbol_timeframe as keys and data lists as values 

Example: 

memory_data = fetcher.get_memory_data()
btc_1m_data = memory_data.get("BTCUSDT_1", [])

async def close(self) 

Description: Close aiohttp session and clean up resources 

Returns: None 

Example: 

await fetcher.close()

WebSocketHandler 
Class: WebSocketHandler 

File: shared_modules/data_collection/websocket_handler.py 

Description: Real-time data streaming with connection management and message processing. 
Constructor 

def __init__(self, config: DataCollectionConfig, symbols: List[str] = None)

Parameters: 

     config (DataCollectionConfig): Configuration object
     symbols (List[str], optional): List of symbols to subscribe to. Defaults to config.SYMBOLS
     

Returns: WebSocketHandler instance 
Methods 
async def connect(self) 

Description: Connect to WebSocket and start listening for messages 

Returns: None 

Example: 

ws_handler = WebSocketHandler(config, symbols=['BTCUSDT'])
await ws_handler.connect()

async def _listen_for_messages(self, connection) 

Description: Listen for WebSocket messages and process them 

Parameters: 

     connection: WebSocket connection object
     

Returns: None 
async def _subscribe_to_symbols_in_batches(self, connection) 

Description: Subscribe to all symbols and timeframes efficiently in batches 

Parameters: 

     connection: WebSocket connection object
     

Returns: None 
async def _process_message(self, message: str) 

Description: Process incoming WebSocket message 

Parameters: 

     message (str): JSON message string
     

Returns: None 
def get_real_time_data(self, symbol: str, timeframe: str) -> List[Dict[str, Any]] 

Description: Get real-time data for specific symbol and timeframe 

Parameters: 

     symbol (str): Trading symbol
     timeframe (str): Timeframe
     

Returns: List[Dict[str, Any]]: List of real-time candle data 

Example: 

real_time_data = ws_handler.get_real_time_data("BTCUSDT", "1")

def add_callback(self, callback: Callable) 

Description: Add callback function for processing candles 

Parameters: 

     callback (Callable): Function to call for each new candle
     

Returns: None 

Example: 

def process_candle(candle_data):
    print(f"New candle: {candle_data}")

ws_handler.add_callback(process_candle)

CSVManager 
Class: CSVManager 

File: shared_modules/data_collection/csv_manager.py 

Description: Data persistence and file operations with chronological ordering and integrity management. 
Constructor 

def __init__(self, config: DataCollectionConfig)

Parameters: 

     config (DataCollectionConfig): Configuration object
     

Returns: CSVManager instance 
Methods 
def read_csv_data(self, symbol: str, timeframe: str) -> List[Dict[str, Any]] 

Description: Read CSV data and return in chronological order (oldest first) 

Parameters: 

     symbol (str): Trading symbol
     timeframe (str): Timeframe
     

Returns: List[Dict[str, Any]]: List of candle data dictionaries, sorted chronologically 

Example: 

data = csv_manager.read_csv_data("BTCUSDT", "1")
print(f"Read {len(data)} candles from CSV")

def write_csv_data(self, symbol: str, timeframe: str, data: List[Dict[str, Any]]) -> bool 

Description: Write data to CSV in chronological order with 50-entry limit handling 

Parameters: 

     symbol (str): Trading symbol
     timeframe (str): Timeframe
     data (List[Dict[str, Any]]): List of candle data to write
     

Returns: bool: True if successful, False otherwise 

Example: 

candle_data = [
    {
        'timestamp': 1640995200000,
        'datetime': '2022-01-01 00:00:00',
        'open': 47000.0,
        'high': 47100.0,
        'low': 46900.0,
        'close': 47050.0,
        'volume': 1000.0
    }
]

success = csv_manager.write_csv_data("BTCUSDT", "1", candle_data)

def append_new_data(self, symbol: str, timeframe: str, new_candles: List[Dict[str, Any]]) -> bool 

Description: Append new candles to existing CSV data while maintaining chronological order and removing duplicates 

Parameters: 

     symbol (str): Trading symbol
     timeframe (str): Timeframe
     new_candles (List[Dict[str, Any]]): List of new candle data to append
     

Returns: bool: True if successful, False otherwise 

Example: 

new_candles = [get_new_candles_from_websocket()]
success = csv_manager.append_new_data("BTCUSDT", "1", new_candles)

def get_latest_timestamp(self, symbol: str, timeframe: str) -> int 

Description: Get the latest timestamp from CSV file 

Parameters: 

     symbol (str): Trading symbol
     timeframe (str): Timeframe
     

Returns: int: Latest timestamp in milliseconds, or 0 if file doesn't exist 

Example: 

latest_timestamp = csv_manager.get_latest_timestamp("BTCUSDT", "1")
print(f"Latest data: {datetime.fromtimestamp(latest_timestamp/1000)}")

def update_candle(self, symbol: str, timeframe: str, candle_data: Dict) -> bool 

Description: Update or append a candle to CSV file 

Parameters: 

     symbol (str): Trading symbol
     timeframe (str): Timeframe
     candle_data (Dict): Candle data to update or append
     

Returns: bool: True if successful, False otherwise 

Example: 

updated_candle = {
    'timestamp': 1640995200000,
    'open': 47000.0,
    'high': 47100.0,
    'low': 46900.0,
    'close': 47050.0,
    'volume': 1000.0
}

success = csv_manager.update_candle("BTCUSDT", "1", updated_candle)

DataIntegrityChecker 
Class: DataIntegrityChecker 

File: shared_modules/data_collection/data_integrity.py 

Description: Data quality assurance and validation with gap detection and repair capabilities. 
Constructor 

def __init__(self, config: DataCollectionConfig)

Parameters: 

     config (DataCollectionConfig): Configuration object
     

Returns: DataIntegrityChecker instance 
Methods 
def check_all_files(self) -> Dict[str, Any] 

Description: Check integrity of all data files in the data directory 

Returns: Dict[str, Any]: Comprehensive integrity report 

Example: 

checker = DataIntegrityChecker(config)
report = checker.check_all_files()
print(f"Files checked: {report['files_checked']}")
print(f"Files with issues: {report['files_with_issues']}")

def check_single_file(self, filename: str) -> Dict[str, Any] 

Description: Check integrity of a single data file 

Parameters: 

     filename (str): Name of the CSV file to check
     

Returns: Dict[str, Any]: Integrity report for the single file 

Example: 

file_report = checker.check_single_file("BTCUSDT_1.csv")
print(f"Gaps found: {len(file_report['gaps'])}")
print(f"Duplicates: {file_report['duplicate_count']}")

def fix_all_duplicates(self) 

Description: Fix duplicates in all data files 

Returns: None 

Example: 

checker.fix_all_duplicates()

def fix_duplicates(self, filename: str) -> bool 

Description: Remove duplicate entries from a single file 

Parameters: 

     filename (str): Name of the CSV file to fix
     

Returns: bool: True if duplicates were fixed, False otherwise 

Example: 

fixed = checker.fix_duplicates("BTCUSDT_1.csv")

def fill_gaps_in_file(self, filename: str) -> bool 

Description: Fill gaps in a data file with previous candle data 

Parameters: 

     filename (str): Name of the CSV file to fix
     

Returns: bool: True if gaps were filled, False otherwise 

Example: 

filled = checker.fill_gaps_in_file("BTCUSDT_1.csv")

def fill_all_gaps(self) 

Description: Fill gaps in all data files 

Returns: None 

Example: 

checker.fill_all_gaps()

def save_integrity_report(self, results: Dict[str, Any]) -> str 

Description: Save integrity check results to a report file 

Parameters: 

     results (Dict[str, Any]): Integrity check results
     

Returns: str: Path to the saved report file 

Example: 

report_path = checker.save_integrity_report(report)
print(f"Report saved to: {report_path}")

DataFeeder 
Class: DataFeeder 

File: simple_strategy/shared/data_feeder.py 

Description: Data loading and management for backtesting and strategy development with memory management. 
Constructor 

def __init__(self, data_dir: str = 'data', memory_limit_percent: float = 90)

Parameters: 

     data_dir (str): Directory containing CSV data files. Defaults to 'data'
     memory_limit_percent (float): Maximum memory usage percentage (0-100). Defaults to 90
     

Returns: DataFeeder instance 
Methods 
def load_data(self, symbols: List[str], timeframes: List[str], start_date: Optional[Union[str, datetime]] = None, end_date: Optional[Union[str, datetime]] = None) -> bool 

Description: Load data for specified symbols and timeframes with memory management 

Parameters: 

     symbols (List[str]): List of trading symbols
     timeframes (List[str]): List of timeframes
     start_date (Optional[Union[str, datetime]]): Optional start date for filtering
     end_date (Optional[Union[str, datetime]]): Optional end date for filtering
     

Returns: bool: True if loading was successful, False otherwise 

Example: 

data_feeder = DataFeeder(data_dir='data')
success = data_feeder.load_data(
    symbols=['BTCUSDT', 'ETHUSDT'],
    timeframes=['1', '5'],
    start_date='2023-01-01',
    end_date='2023-12-31'
)

def get_data_for_symbols(self, symbols: List[str], timeframes: List[str], start_date, end_date) -> Dict[str, Dict[str, pd.DataFrame]] 

Description: Return cached data for multiple symbols/timeframes, filtered by date range 

Parameters: 

     symbols (List[str]): List of trading symbols
     timeframes (List[str]): List of timeframes
     start_date: Start date for filtering
     end_date: End date for filtering
     

Returns: Dict[str, Dict[str, pd.DataFrame]]: Nested dictionary with data 

Example: 

data = data_feeder.get_data_for_symbols(
    symbols=['BTCUSDT'],
    timeframes=['1', '5'],
    start_date='2023-01-01',
    end_date='2023-01-31'
)
btc_1m_data = data['BTCUSDT']['1']

def get_data_at_timestamp(self, symbol: str, timeframe: str, timestamp: Union[int, datetime, str]) -> Optional[Dict[str, Any]] 

Description: Get data for a specific timestamp 

Parameters: 

     symbol (str): Trading symbol
     timeframe (str): Timeframe
     timestamp (Union[int, datetime, str]): Timestamp (can be int milliseconds, datetime, or string)
     

Returns: Optional[Dict[str, Any]]: Dictionary with OHLCV data or None if not found 

Example: 

candle = data_feeder.get_data_at_timestamp("BTCUSDT", "1", 1640995200000)
if candle:
    print(f"Close price: {candle['close']}")

def get_latest_data(self, symbol: str, timeframe: str, lookback_periods: int = 1) -> Optional[List[Dict[str, Any]]] 

Description: Get latest available data for timeframe 

Parameters: 

     symbol (str): Trading symbol
     timeframe (str): Timeframe
     lookback_periods (int): Number of periods to return. Defaults to 1
     

Returns: Optional[List[Dict[str, Any]]]: List of dictionaries with OHLCV data or None if not found 

Example: 

latest_candles = data_feeder.get_latest_data("BTCUSDT", "1", lookback_periods=5)
for candle in latest_candles:
    print(f"Timestamp: {candle['timestamp_ms']}, Close: {candle['close']}")

def get_multi_timeframe_data(self, symbol: str, timeframes: List[str], timestamp: Union[int, datetime, str]) -> Optional[Dict[str, Dict[str, Any]]] 

Description: Get aligned data across multiple timeframes 

Parameters: 

     symbol (str): Trading symbol
     timeframes (List[str]): List of timeframes to retrieve
     timestamp (Union[int, datetime, str]): Reference timestamp for alignment
     

Returns: Optional[Dict[str, Dict[str, Any]]]: Dictionary with timeframe as key and OHLCV data as value 

Example: 

multi_tf_data = data_feeder.get_multi_timeframe_data(
    "BTCUSDT", 
    ["1", "5", "15"], 
    1640995200000
)
if multi_tf_data:
    print(f"1m close: {multi_tf_data['1']['close']}")
    print(f"5m close: {multi_tf_data['5']['close']}")
    print(f"15m close: {multi_tf_data['15']['close']}")

def get_memory_usage(self) -> Dict[str, Any] 

Description: Get current memory usage statistics 

Returns: Dict[str, Any]: Memory usage information 

Example: 

memory_info = data_feeder.get_memory_usage()
print(f"Memory usage: {memory_info['percent']}%")

DataCollectionConfig 
Class: DataCollectionConfig 

File: shared_modules/data_collection/config.py 

Description: Configuration settings for the data collection system. 
Attributes 
API Settings 

BYBIT_API_KEY: str          # Bybit API key from environment
BYBIT_API_SECRET: str       # Bybit API secret from environment
API_BASE_URL: str           # Bybit API base URL

Data Settings

SYMBOLS: List[str]          # List of trading symbols to collect
TIMEFRAMES: List[str]       # List of timeframes to collect
DATA_DIR: str               # Directory for CSV data storage

Collection Modes

LIMIT_TO_50_ENTRIES: bool   # True=keep only 50 recent entries, False=all data
FETCH_ALL_SYMBOLS: bool     # True=all Bybit symbols, False=config.SYMBOLS only
DAYS_TO_FETCH: int          # Number of days of historical data to fetch

Real-time Data

ENABLE_WEBSOCKET: bool      # Enable real-time WebSocket streaming
RUN_INTEGRITY_CHECK: bool   # Run data validation after collection
RUN_GAP_FILLING: bool       # Auto-fill data gaps after collection

Performance Settings

BULK_BATCH_SIZE: int        # Concurrent requests per batch
BULK_REQUEST_DELAY_MS: int  # Delay between batches (milliseconds)
BULK_MAX_RETRIES: int       # Maximum retry attempts

Usage Example

from shared_modules.data_collection.config import DataCollectionConfig

# Create configuration
config = DataCollectionConfig()

# Modify settings for specific use case
config.SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
config.TIMEFRAMES = ['1', '5', '15']
config.LIMIT_TO_50_ENTRIES = False
config.FETCH_ALL_SYMBOLS = False
config.DAYS_TO_FETCH = 100
config.ENABLE_WEBSOCKET = True

# Use with components
from shared_modules.data_collection.hybrid_system import HybridTradingSystem
system = HybridTradingSystem(config)

Configuration Profiles 
For AI/ML Training 

config.LIMIT_TO_50_ENTRIES = False    # Maximum historical data
config.FETCH_ALL_SYMBOLS = True       # Comprehensive dataset
config.DAYS_TO_FETCH = 100           # Deep historical context
config.ENABLE_WEBSOCKET = True        # Continuous updates

For Strategy Testing

config.LIMIT_TO_50_ENTRIES = False    # Complete historical data
config.FETCH_ALL_SYMBOLS = True       # Full market coverage
config.ENABLE_WEBSOCKET = False       # No real-time needed

Data Structures 
Candle Data Dictionary 

All data operations use a consistent candle data structure: 

{
    'timestamp': int,        # Unix timestamp in milliseconds
    'datetime': str,        # Human-readable datetime (YYYY-MM-DD HH:MM:SS)
    'open': float,          # Opening price
    'high': float,          # Highest price
    'low': float,           # Lowest price
    'close': float,         # Closing price
    'volume': float,        # Trading volume
    'turnover': float,      # Trading turnover (optional)
    'confirm': bool         # Candle confirmation flag (WebSocket only)
}

Integrity Report Structure

{
    'files_checked': int,
    'files_with_issues': int,
    'total_gaps': int,
    'total_duplicates': int,
    'total_invalid_candles': int,
    'issues': {
        'filename.csv': {
            'has_issues': bool,
            'gaps': List[Dict],
            'duplicate_count': int,
            'invalid_candles': int,
            'total_candles': int,
            'first_timestamp': str,
            'last_timestamp': str
        }
    }
}

Gap Information Structure

{
    'position': int,
    'previous_timestamp': str,
    'current_timestamp': str,
    'expected_interval': str,
    'actual_interval': str,
    'gap_duration': str,
    'missing_candles': int,
    'gap_minutes': float
}

Error Handling 
Common Exceptions 

All components implement consistent error handling: 
Connection Errors 

try:
    await system.fetch_data_hybrid()
except ConnectionError as e:
    logger.error(f"Connection failed: {e}")
    # Implement retry logic

Data Validation Errors

try:
    data = csv_manager.read_csv_data("BTCUSDT", "1")
except ValueError as e:
    logger.error(f"Data validation failed: {e}")
    # Handle corrupted data

Memory Errors

try:
    data_feeder.load_data(large_symbol_list, large_timeframe_list)
except MemoryError as e:
    logger.error(f"Memory limit exceeded: {e}")
    # Reduce data load or increase memory limit

Error Recovery Patterns 
Retry with Exponential Backoff 

async def operation_with_retry(max_retries=3):
    for attempt in range(max_retries):
        try:
            return await risky_operation()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = (2 ** attempt) * 1  # 1s, 2s, 4s
            await asyncio.sleep(delay)

Graceful Degradation

def get_data_with_fallback(symbol, timeframe):
    try:
        return data_feeder.get_data_at_timestamp(symbol, timeframe, timestamp)
    except Exception as e:
        logger.warning(f"Primary method failed: {e}")
        # Fallback to CSV direct read
        return csv_manager.read_csv_data(symbol, timeframe)[-1] if csv_manager.read_csv_data(symbol, timeframe) else None

Performance Considerations 
Memory Management 

# Monitor memory usage
memory_info = data_feeder.get_memory_usage()
if memory_info['percent'] > 85:
    logger.warning("High memory usage detected")

# Clear cache if needed
data_feeder.data_cache.clear()

Batch Processing

# Optimal batch sizes for different scenarios
small_batch = 10   # For slow networks or rate-limited APIs
medium_batch = 20  # Default balanced setting
large_batch = 50   # For high-performance environments

config.BULK_BATCH_SIZE = medium_batch

Concurrent Operations

# Process multiple symbols concurrently
async def process_multiple_symbols(symbols, timeframes):
    tasks = []
    for symbol in symbols:
        for timeframe in timeframes:
            task = asyncio.create_task(
                fetcher._fetch_symbol_timeframe(symbol, timeframe, days, False)
            )
            tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

Best Practices 
1. Resource Management 

# Always use context managers or explicit cleanup
async with HybridTradingSystem(config) as system:
    await system.fetch_data_hybrid()
    # System automatically cleaned up

2. Error Handling

# Implement comprehensive error handling
try:
    success = await fetcher.fetch_historical_data_fast(symbols, timeframes, days)
    if not success:
        logger.error("Data fetch failed")
        # Handle failure appropriately
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise

3. Memory Optimization

# Load only necessary data
data_feeder.load_data(
    symbols=['BTCUSDT'],  # Minimal symbol set
    timeframes=['1'],     # Single timeframe
    start_date='2023-01-01',  # Specific date range
    end_date='2023-01-31'
)

4. Configuration Management

# Use configuration profiles for different environments
if environment == "testing":
    config.LIMIT_TO_50_ENTRIES = True
    config.DAYS_TO_FETCH = 7
elif environment == "production":
    config.LIMIT_TO_50_ENTRIES = False
    config.DAYS_TO_FETCH = 365

5. Data Quality Assurance

# Regular integrity checks
checker = DataIntegrityChecker(config)
report = checker.check_all_files()

if report['files_with_issues'] > 0:
    logger.warning(f"Data quality issues detected in {report['files_with_issues']} files")
    # Automatically fix issues
    checker.fix_all_duplicates()
    checker.fill_all_gaps()

