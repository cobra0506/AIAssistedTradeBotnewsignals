# Data Management Component - Implementation Guide

## Implementation Overview

The Data Management Component is implemented as a modular, extensible system with clear separation of concerns. It follows a hybrid architecture combining historical data collection with real-time streaming capabilities.

### Core Design Principles

1. **Modularity**: Each component has a single, well-defined responsibility
2. **Async/Await**: Leverages Python's async capabilities for concurrent operations
3. **Error Resilience**: Comprehensive error handling and recovery mechanisms
4. **Performance**: Optimized for speed and memory efficiency
5. **Configurability**: Flexible settings for different use cases

## Component Architecture

### 1. Hybrid Trading System (`hybrid_system.py`)

#### Purpose
Core orchestrator that coordinates between historical and real-time data collection systems.

#### Key Implementation Details

```python
class HybridTradingSystem:
    def __init__(self, config):
        self.config = config
        self.data_fetcher = OptimizedDataFetcher(config)
        self.websocket_handler = WebSocketHandler(config)
        self.csv_manager = CSVManager(config)
        self.is_initialized = False

Core Methods 

fetch_data_hybrid(): Main data collection method 

async def fetch_data_hybrid(self, symbols=None, timeframes=None, days=None, mode="full"):
    """
    mode: "full" = all historical data
          "recent" = only 50 most recent entries
          "live" = only real-time data
    """

Data Flow: 

    Start WebSocket connection first (if enabled) 
    Fetch historical data (if requested) 
    Combine and coordinate data sources 
    Save to CSV files 

Implementation Features 

     Priority-based Processing: WebSocket starts before historical data to avoid gaps
     Configurable Modes: Support for different data collection strategies
     Error Recovery: Graceful handling of component failures
     Resource Management: Proper cleanup and resource release
     

2. Optimized Data Fetcher (optimized_data_fetcher.py) 
Purpose 

High-performance historical data fetching with concurrent processing and rate limiting. 
Key Implementation Details 

class OptimizedDataFetcher:
    def __init__(self, config):
        self.config = config
        self.memory_data = {}  # symbol -> timeframe -> deque
        self.session = None
        self.csv_manager = CSVManager(config)
        self.fetch_stats = {...}

Core Methods 

fetch_historical_data_fast(): Main historical data fetching method 

async def fetch_historical_data_fast(self, symbols, timeframes, days, limit_50=False):
    """Ultra-fast historical data fetching with optimized chunking and retry logic"""

Chunk Calculation Logic:

def _calculate_chunk_parameters(self, timeframe: str, days: int) -> tuple:
    """Calculate optimal chunk parameters based on timeframe and days"""
    # Convert timeframe to minutes
    timeframe_minutes = int(timeframe)
    # Calculate total candles needed
    total_minutes = days * 24 * 60
    total_candles_needed = total_minutes // timeframe_minutes
    # Use maximum allowed by Bybit (999 candles per chunk)
    candles_per_chunk = 999
    # Calculate number of chunks needed
    num_chunks = (total_candles_needed + candles_per_chunk - 1) // candles_per_chunk

Implementation Features 

Batch Processing: 

batch_size = self.config.BULK_BATCH_SIZE
for i in range(0, len(symbols), batch_size):
    batch_symbols = symbols[i:i+batch_size]
    batch_tasks = []
    for symbol in batch_symbols:
        for timeframe in timeframes:
            task = asyncio.create_task(
                self._fetch_symbol_timeframe(symbol, timeframe, days, limit_50)
            )
            batch_tasks.append(task)
    
    # Execute batch tasks concurrently
    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

Rate Limiting and Retry Logic:

async def _make_chunk_request_with_retry(self, symbol, timeframe, start_time, end_time, limit):
    """Make rate-limited request with exponential backoff retry"""
    for attempt in range(self.config.BULK_MAX_RETRIES):
        try:
            # Make API request
            result = await self._make_rate_limited_request(symbol, timeframe, start_time, end_time, limit)
            if result:
                return result
        except Exception as e:
            if attempt == self.config.BULK_MAX_RETRIES - 1:
                logger.error(f"Max retries reached for {symbol}_{timeframe}")
                return None
            # Exponential backoff
            delay = (2 ** attempt) * 1  # 1s, 2s, 4s, 8s, 16s
            await asyncio.sleep(delay)

3. WebSocket Handler (websocket_handler.py) 
Purpose 

Real-time data streaming with connection management and message processing. 
Key Implementation Details 

class WebSocketHandler:
    def __init__(self, config: DataCollectionConfig, symbols: List[str] = None):
        self.config = config
        self.ws_url = "wss://stream.bybit.com/v5/public/linear"
        self.running = False
        self.real_time_data = {}  # Store real-time data
        self.connection = None
        self.csv_manager = CSVManager(config)

Core Methods 

Connection Management: 

async def connect(self):
    """Connect to WebSocket and start listening"""
    if not self.config.ENABLE_WEBSOCKET:
        return
    
    self.running = True
    connection = await self._connect_with_ssl()
    if connection:
        self.connection = connection
        await self._listen_for_messages(connection)

Batch Subscription:

async def _subscribe_to_symbols_in_batches(self, connection):
    """Subscribe to all symbols and timeframes efficiently in large batches"""
    all_args = []
    for symbol in self.symbols:
        for interval in self.intervals:
            all_args.append(f"kline.{interval}.{symbol}")
    
    batch_size = 300  # Large batch size for efficiency
    for i in range(0, len(all_args), batch_size):
        batch_args = all_args[i:i + batch_size]
        subscription_msg = json.dumps({"op": "subscribe", "args": batch_args})
        
        # Simple retry logic
        for attempt in range(3):
            try:
                await connection.send(subscription_msg)
                break
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(1)

Message Processing:

async def _process_message(self, message):
    """Process incoming WebSocket message"""
    data = json.loads(message)
    
    # Handle subscription confirmation
    if data.get("op") == "subscribe":
        if data.get("success") is True:
            self.subscription_count += 1
        else:
            self.failed_subscriptions.append(data.get('req_id', 'unknown'))
    
    # Handle data messages
    if "topic" in data and "data" in data:
        topic = data["topic"]
        # Parse topic to get symbol and timeframe
        parts = topic.split(".")
        if len(parts) >= 3 and parts[0] == "kline":
            timeframe = parts[1]
            symbol = parts[2]
            # Process candle data
            await self._process_candle(symbol, timeframe, data["data"])

Implementation Features 

Connection Health Management: 

async def _check_connection_health(self, websocket):
    """Check if WebSocket connection is healthy"""
    try:
        if not websocket:
            return False
        
        # Send ping with timeout
        pong_waiter = await websocket.ping()
        await asyncio.wait_for(pong_waiter, timeout=10.0)
        return True
    except asyncio.TimeoutError:
        return False

Heartbeat Mechanism:

async def _heartbeat(self, connection):
    """Maintain connection with periodic pings"""
    while True:
        try:
            await connection.send(json.dumps({"op": "ping"}))
            await asyncio.sleep(20)  # Ping every 20 seconds
        except Exception as e:
            break  # Break to trigger reconnect

4. CSV Manager (csv_manager.py) 
Purpose 

Data persistence and file operations with chronological ordering and integrity management. 
Key Implementation Details 

class CSVManager:
    def __init__(self, config: DataCollectionConfig):
        self.config = config
        self.data_dir = Path(config.DATA_DIR)

Core Methods 

Data Reading: 

def read_csv_data(self, symbol: str, timeframe: str) -> List[Dict[str, Any]]:
    """Read CSV data and return in chronological order (oldest first)"""
    filename = os.path.join(self.data_dir, f"{symbol}_{timeframe}.csv")
    
    with open(filename, 'r', newline='') as f:
        reader = csv.DictReader(f)
        data = list(reader)
        
        # Convert string values to appropriate types
        for row in data:
            row['timestamp'] = int(row['timestamp'])
            row['open'] = float(row['open'])
            row['high'] = float(row['high'])
            row['low'] = float(row['low'])
            row['close'] = float(row['close'])
            row['volume'] = float(row['volume'])
        
        # Sort by timestamp (chronological order)
        data.sort(key=lambda x: x['timestamp'])
        return data

Data Writing:

def write_csv_data(self, symbol: str, timeframe: str, data: List[Dict[str, Any]]) -> bool:
    """Write data to CSV in chronological order with 50-entry limit handling"""
    if not data:
        return False
    
    # Sort data chronologically
    data.sort(key=lambda x: x['timestamp'])
    
    # Apply 50-entry limit if configured
    if self.config.LIMIT_TO_50_ENTRIES and len(data) > 50:
        data = data[-50:]  # Keep most recent 50 entries
    
    # Ensure all required fields are present
    fieldnames = ['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume']
    for row in data:
        if 'datetime' not in row:
            dt = datetime.fromtimestamp(row['timestamp'] / 1000)
            row['datetime'] = dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Write to CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

Data Appending with Deduplication:

def append_new_data(self, symbol: str, timeframe: str, new_candles: List[Dict[str, Any]]) -> bool:
    """Append new candles while maintaining chronological order and removing duplicates"""
    # Read existing data
    existing_data = self.read_csv_data(symbol, timeframe)
    
    # Get existing timestamps for duplicate checking
    existing_timestamps = {row['timestamp'] for row in existing_data}
    
    # Filter out duplicates
    unique_new_candles = []
    for candle in new_candles:
        if candle['timestamp'] not in existing_timestamps:
            unique_new_candles.append(candle)
    
    if unique_new_candles:
        # Combine and write back
        combined_data = existing_data + unique_new_candles
        return self.write_csv_data(symbol, timeframe, combined_data)
    
    return True

5. Data Integrity Checker (data_integrity.py) 
Purpose 

Data quality assurance and validation with gap detection and repair capabilities. 
Key Implementation Details 

class DataIntegrityChecker:
    def __init__(self, config: DataCollectionConfig):
        self.config = config
        self.reports_dir = os.path.join('logs', 'integrity_reports')

Core Methods 

Comprehensive Integrity Checking: 

def check_all_files(self) -> Dict[str, Any]:
    """Check integrity of all data files"""
    results = {
        'files_checked': 0,
        'files_with_issues': 0,
        'total_gaps': 0,
        'total_duplicates': 0,
        'total_invalid_candles': 0,
        'issues': {}
    }
    
    # Get all CSV files
    csv_files = [f for f in os.listdir(self.config.DATA_DIR) if f.endswith('.csv')]
    
    # Check each file
    for filename in csv_files:
        file_issues = self.check_single_file(filename)
        if file_issues['has_issues']:
            results['files_with_issues'] += 1
            results['issues'][filename] = file_issues
            results['total_gaps'] += len(file_issues['gaps'])
            results['total_duplicates'] += file_issues['duplicate_count']
            results['total_invalid_candles'] += file_issues['invalid_candles']
        results['files_checked'] += 1
    
    return results

Candle Validation:

def _validate_candle(self, candle: Dict[str, Any]) -> bool:
    """Validate a single candle"""
    try:
        # Check required fields
        required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
        if not all(field in candle for field in required_fields):
            return False
        
        # Validate timestamp format
        datetime.fromisoformat(candle['timestamp'])
        
        # Validate price values and relationships
        open_price = float(candle['open'])
        high_price = float(candle['high'])
        low_price = float(candle['low'])
        close_price = float(candle['close'])
        
        # Validate price relationships
        if not (low_price <= high_price and
                low_price <= open_price <= high_price and
                low_price <= close_price <= high_price):
            return False
        
        # Validate positive values
        if any(val <= 0 for val in [open_price, high_price, low_price, close_price]):
            return False
        
        return True
    except (ValueError, TypeError):
        return False

Gap Detection:

def _detect_gaps(self, data: List[Dict[str, Any]], timeframe: str) -> List[Dict[str, Any]]:
    """Detect gaps in time series data"""
    gaps = []
    if len(data) < 2:
        return gaps
    
    # Convert timeframe to minutes
    timeframe_minutes = {'1': 1, '5': 5, '15': 15, '60': 60, '240': 240, '1440': 1440}.get(timeframe, 1)
    expected_interval = timedelta(minutes=timeframe_minutes)
    
    for i in range(1, len(data)):
        prev_timestamp = datetime.fromisoformat(data[i-1]['timestamp'])
        curr_timestamp = datetime.fromisoformat(data[i]['timestamp'])
        actual_interval = curr_timestamp - prev_timestamp
        
        # Allow small tolerance (1 second)
        tolerance = timedelta(seconds=1)
        if actual_interval > expected_interval + tolerance:
            gap_duration = actual_interval - expected_interval
            gap_minutes = gap_duration.total_seconds() / 60
            missing_candles = int(gap_minutes / timeframe_minutes)
            
            gaps.append({
                'position': i,
                'previous_timestamp': data[i-1]['timestamp'],
                'current_timestamp': data[i]['timestamp'],
                'expected_interval': str(expected_interval),
                'actual_interval': str(actual_interval),
                'gap_duration': str(gap_duration),
                'missing_candles': missing_candles,
                'gap_minutes': gap_minutes
            })
    
    return gaps

6. Data Feeder (data_feeder.py) 
Purpose 

Data loading and management for backtesting and strategy development with memory management. 
Key Implementation Details 

class DataFeeder:
    def __init__(self, data_dir: str = 'data', memory_limit_percent: float = 90):
        self.data_dir = Path(data_dir)
        self.memory_limit_percent = memory_limit_percent
        self.data_cache: Dict[str, Dict[str, pd.DataFrame]] = {}  # {symbol: {timeframe: DataFrame}}
        self.metadata_cache: Dict[str, Dict[str, Dict]] = {}  # {symbol: {timeframe: metadata}}

Core Methods 

Memory Management: 

def _check_memory_usage(self) -> bool:
    """Check if current memory usage is within limits"""
    try:
        memory = psutil.virtual_memory()
        current_usage_percent = memory.percent
        return current_usage_percent <= self.memory_limit_percent
    except Exception as e:
        logger.warning(f"Could not check memory usage: {e}")
        return True

Data Loading with Caching:

def load_data(self, symbols: List[str], timeframes: List[str],
              start_date: Optional[Union[str, datetime]] = None,
              end_date: Optional[Union[str, datetime]] = None) -> bool:
    """Load data for specified symbols and timeframes with memory management"""
    
    # Check memory usage before loading
    if not self._check_memory_usage():
        logger.error(f"Memory usage exceeds limit of {self.memory_limit_percent}%")
        return False
    
    success_count = 0
    for symbol in symbols:
        if symbol not in self.data_cache:
            self.data_cache[symbol] = {}
            self.metadata_cache[symbol] = {}
        
        for timeframe in timeframes:
            # Load data from CSV
            df = self._load_csv_file(symbol, timeframe)
            if df is None:
                continue
            
            # Apply date filtering if specified
            if start_date is not None:
                df = df[df.index >= start_date]
            if end_date is not None:
                df = df[df.index <= end_date]
            
            if len(df) == 0:
                continue
            
            # Store in cache
            self.data_cache[symbol][timeframe] = df
            self.metadata_cache[symbol][timeframe] = {
                'start_date': df.index.min(),
                'end_date': df.index.max(),
                'row_count': len(df),
                'file_path': str(self.data_dir / f"{symbol}_{timeframe}.csv")
            }
            success_count += 1
    
    return success_count > 0

Data Access Methods:

def get_data_at_timestamp(self, symbol: str, timeframe: str, 
                         timestamp: Union[int, datetime, str]) -> Optional[Dict[str, Any]]:
    """Get data for a specific timestamp"""
    # Convert timestamp to datetime
    if isinstance(timestamp, int):
        dt = pd.to_datetime(timestamp, unit='ms')
    elif isinstance(timestamp, str):
        dt = pd.to_datetime(timestamp)
    else:
        dt = timestamp
    
    # Check if data is loaded
    if symbol not in self.data_cache or timeframe not in self.data_cache[symbol]:
        return None
    
    df = self.data_cache[symbol][timeframe]
    
    # Find the closest timestamp (exact match or previous)
    mask = df.index <= dt
    if not mask.any():
        return None
    
    # Get the most recent data point at or before the timestamp
    result_df = df[mask].tail(1)
    if len(result_df) == 0:
        return None
    
    # Convert to dictionary
    result = result_df.iloc[0].to_dict()
    result['timestamp_ms'] = int(result_df.index[0].timestamp() * 1000)
    return result

Multi-Timeframe Data Alignment:

def get_multi_timeframe_data(self, symbol: str, timeframes: List[str], 
                           timestamp: Union[int, datetime, str]) -> Optional[Dict[str, Dict[str, Any]]]:
    """Get aligned data across multiple timeframes"""
    results = {}
    for timeframe in timeframes:
        data = self.get_data_at_timestamp(symbol, timeframe, timestamp)
        if data is not None:
            results[timeframe] = data
    return results if results else None

Configuration Management 
Data Collection Config (config.py) 

class DataCollectionConfig:
    # API settings
    BYBIT_API_KEY = os.getenv('BYBIT_API_KEY', '')
    BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET', '')
    API_BASE_URL = 'https://api.bybit.com'
    
    # Data settings
    SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', ...]  # Configurable symbol list
    TIMEFRAMES = ['1', '5', '15', '60', '120', '240']  # Multiple timeframes
    DATA_DIR = 'data'
    
    # Data collection mode
    LIMIT_TO_50_ENTRIES = True  # Keep only last 50 entries
    FETCH_ALL_SYMBOLS = True    # Get all symbols from Bybit
    
    # WebSocket settings
    ENABLE_WEBSOCKET = True     # Enable real-time streaming
    
    # Performance settings
    DAYS_TO_FETCH = 200
    BULK_BATCH_SIZE = 20        # Concurrent requests
    BULK_REQUEST_DELAY_MS = 10 # Delay between batches
    BULK_MAX_RETRIES = 5        # Maximum retry attempts

Error Handling and Recovery 
Comprehensive Error Management 

All components implement robust error handling: 

# Example error handling pattern
try:
    result = await risky_operation()
    if result:
        logger.info(f"Operation successful: {result}")
        return result
    else:
        logger.warning("Operation returned empty result")
        return None
except Exception as e:
    logger.error(f"Operation failed: {e}")
    # Implement recovery logic
    return await recovery_operation()

Connection Recovery

async def _handle_connection_recovery(self):
    """Handle connection recovery when health check fails"""
    logger.info("[RECOVERY] Attempting connection recovery...")
    
    # Close existing connection
    if self.connection:
        await self.connection.close()
    
    # Attempt reconnection
    for attempt in range(3):
        try:
            new_connection = await self._connect_with_ssl()
            if new_connection:
                self.connection = new_connection
                logger.info("[RECOVERY] Connection re-established")
                return True
        except Exception as e:
            logger.error(f"[RECOVERY] Reconnection attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    return False

Performance Optimization Techniques 
1. Async/Await Concurrency 

     All I/O operations use async/await for non-blocking execution
     Concurrent processing of multiple symbols and timeframes
     Proper resource cleanup and context management
     

2. Batch Processing 

     Configurable batch sizes for API requests
     Rate limiting between batches to avoid API restrictions
     Efficient memory usage with streaming processing
     

3. Memory Management 

     Configurable memory limits with automatic cleanup
     Intelligent caching with LRU (Least Recently Used) eviction
     Data compression and efficient storage formats
     

4. Connection Pooling 

     Reused HTTP sessions for reduced overhead
     WebSocket connection persistence
     Proper connection lifecycle management
     

Testing and Quality Assurance 
Test Coverage 

     Unit Tests: Individual component testing
     Integration Tests: Component interaction testing
     Performance Tests: Load and stress testing
     Data Quality Tests: Integrity validation testing
     

Test Implementation Example 

async def test_data_fetcher():
    """Test the OptimizedDataFetcher component"""
    config = DataCollectionConfig()
    fetcher = OptimizedDataFetcher(config)
    
    await fetcher.initialize()
    
    # Test with small dataset
    result = await fetcher.fetch_historical_data_fast(
        symbols=['BTCUSDT'], 
        timeframes=['1'], 
        days=1, 
        limit_50=True
    )
    
    assert result is True
    assert fetcher.fetch_stats['successful_requests'] > 0

Monitoring and Logging 
Structured Logging 

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Usage in components
logger.info(f"[DATA] Fetching historical data for {symbol}_{timeframe}")
logger.error(f"[FAIL] Failed to fetch data: {error}")
logger.debug(f"[DEBUG] Processing candle: {candle_data}")

Performance Monitoring 

     Request success/failure rates
     Memory usage tracking
     Data collection speed metrics
     Connection health monitoring
     

Deployment Considerations 
System Requirements 

     Python: 3.8+ with async support
     Memory: Minimum 4GB, recommended 8GB+ for large datasets
     Storage: Sufficient space for CSV data files
     Network: Stable internet connection for API access
     

Environment Configuration 

# Set environment variables
export BYBIT_API_KEY=your_api_key_here
export BYBIT_API_SECRET=your_api_secret_here

# Install dependencies
pip install -r requirements.txt

# Run data collection
python shared_modules/data_collection/launch_data_collection.py

Scaling Considerations 

     Horizontal Scaling: Multiple instances for different symbol groups
     Vertical Scaling: Increased memory and CPU for larger datasets
     Database Integration: Migration from CSV to database for very large datasets
     

