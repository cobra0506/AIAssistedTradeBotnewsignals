# optimized_data_fetcher.py - Fixed chunking logic with proper retry mechanism and text logging
import asyncio
import aiohttp
import csv
import os
import time
import logging
from collections import deque
from typing import Dict, List, Any
from datetime import datetime
from .config import DataCollectionConfig
from .csv_manager import CSVManager
from .logging_utils import setup_logging
from shared_modules.data_collection.config import DataCollectionConfig
MIN_CANDLES = DataCollectionConfig.MIN_CANDLES  # Get from config

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

class OptimizedDataFetcher:
    def __init__(self, config):
        self.config = config
        self.memory_data = {}  # symbol -> timeframe -> deque
        self.session = None
        self.csv_manager = CSVManager(config)
        self.fetch_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'start_time': None,
            'end_time': None
        }

    async def initialize(self):
        """Initialize aiohttp session"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=100)  # High connection limit
        )

    async def cleanup(self):
        """Cleanup aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("[CLEANUP] aiohttp session closed")

    async def _get_all_symbols(self) -> List[str]:
        """Get all available linear symbols from Bybit with pagination"""
        url = f"{self.config.API_BASE_URL}/v5/market/instruments-info"
        all_symbols = []
        cursor = None
        excluded_symbols = ['USDC', 'USDE', 'USTC']
        
        logger.info("Fetching all symbols from Bybit...")
        
        while True:
            params = {
                "category": "linear",
                "limit": 1000
            }
            
            if cursor is not None:
                params["cursor"] = cursor
                
            try:
                logger.debug(f"Making request with params: {params}")
                async with self.session.get(url, params=params, timeout=30) as response:
                    logger.debug(f"Response status: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        logger.debug(f"Response retCode: {data.get('retCode')}")
                        
                        if data.get('retCode') == 0:
                            items = data['result']['list']
                            logger.debug(f"Received {len(items)} items")
                            
                            symbols = [
                                item['symbol'] for item in items
                                if not any(excl in item['symbol'] for excl in excluded_symbols)
                                and "-" not in item['symbol']
                                and item['symbol'].endswith('USDT')
                                and item.get('contractType') == 'LinearPerpetual'
                                and item.get('status') == 'Trading'
                            ]
                            all_symbols.extend(symbols)
                            cursor = data['result'].get('nextPageCursor')
                            logger.debug(f"Next cursor: {cursor}")
                            
                            if not cursor:
                                break
                        else:
                            logger.error(f"Error fetching symbols: {data.get('retMsg')}")
                            break
                    else:
                        logger.error(f"HTTP Error fetching symbols: {response.status}")
                        break
                        
            except Exception as e:
                logger.error(f"Exception fetching symbols: {e}")
                break
                
        logger.info(f"Fetched {len(all_symbols)} perpetual symbols")
        if all_symbols:
            logger.info(f"Examples: {', '.join(all_symbols[:10])}...")
            
        return sorted(all_symbols)

    def _calculate_chunk_parameters(self, timeframe: str, days: int) -> tuple:
        """
        Calculate chunk parameters based on timeframe and days
        Returns: (total_candles_needed, num_chunks, candles_per_chunk, chunk_duration_ms)
        """
        # Convert timeframe to minutes
        timeframe_minutes = int(timeframe)
        
        # Calculate total candles needed
        total_minutes = days * 24 * 60
        total_candles_needed = total_minutes // timeframe_minutes
        
        # Use maximum allowed by Bybit (999 candles per chunk)
        candles_per_chunk = 999
        
        # Calculate number of chunks needed
        num_chunks = (total_candles_needed + candles_per_chunk - 1) // candles_per_chunk
        
        # Calculate chunk duration in milliseconds
        chunk_duration_ms = candles_per_chunk * timeframe_minutes * 60 * 1000
        
        logger.info(f"Calculated chunks for {timeframe}m, {days} days: "
                   f"{total_candles_needed} candles needed, "
                   f"{num_chunks} chunks, "
                   f"{candles_per_chunk} candles per chunk, "
                   f"chunk duration: {chunk_duration_ms/1000/60:.1f} minutes")
        
        return total_candles_needed, num_chunks, candles_per_chunk, chunk_duration_ms

    async def fetch_historical_data_fast(self, symbols: List[str], timeframes: List[str],
                                       days: int, limit_50: bool = False):
        """Ultra-fast historical data fetching with optimized chunking and retry logic"""
        self.fetch_stats['start_time'] = time.time()
        self.fetch_stats['total_requests'] = len(symbols) * len(timeframes)
        self.fetch_stats['successful_requests'] = 0
        self.fetch_stats['failed_requests'] = 0
        
        tasks = []
        batch_size = self.config.BULK_BATCH_SIZE
        
        logger.info(f"[PROCESS] Processing {len(symbols)} symbols in batches of {batch_size}")
        
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(symbols) + batch_size - 1) // batch_size
            
            logger.info(f"[PROCESS] Processing batch {batch_num}/{total_batches} ({len(batch_symbols)} symbols)")
            
            batch_tasks = []
            for symbol in batch_symbols:
                for timeframe in timeframes:
                    task = asyncio.create_task(
                        self._fetch_symbol_timeframe(symbol, timeframe, days, limit_50)
                    )
                    batch_tasks.append(task)
            
            # Execute batch tasks concurrently
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process results
            for result in batch_results:
                if result is True:
                    self.fetch_stats['successful_requests'] += 1
                elif result is False:
                    self.fetch_stats['failed_requests'] += 1
                elif isinstance(result, Exception):
                    logger.error(f"[FAIL] Exception in batch: {result}")
                    self.fetch_stats['failed_requests'] += 1
            
            # Add delay between batches
            if i + batch_size < len(symbols):
                delay_seconds = self.config.BULK_REQUEST_DELAY_MS / 1000 * 2
                logger.info(f"[WAIT] Waiting {delay_seconds} seconds before next batch...")
                await asyncio.sleep(delay_seconds)
        
        self.fetch_stats['end_time'] = time.time()
        
        # Print performance stats
        duration = self.fetch_stats['end_time'] - self.fetch_stats['start_time']
        success_rate = (self.fetch_stats['successful_requests'] / self.fetch_stats['total_requests']) * 100
        
        logger.info(f"[DATA] Historical data fetch completed in {duration:.2f} seconds")
        logger.info(f"[OK] Successful: {self.fetch_stats['successful_requests']}/{self.fetch_stats['total_requests']} ({success_rate:.1f}%)")
        
        return self.fetch_stats['failed_requests'] == 0

    async def _fetch_symbol_timeframe(self, symbol: str, timeframe: str,
                                    days: int, limit_50: bool):
        """Fetch data for single symbol/timeframe with optimized chunking and retry logic"""
        logger.info(f"[PROCESS] Fetching {symbol}_{timeframe}...")
        
        end_time = int(time.time() * 1000)
        
        if limit_50:
            # For limited mode, just fetch the most recent 50 candles
            return await self._fetch_limited_data(symbol, timeframe, end_time)
        else:
            # For full historical data, use optimized chunking
            return await self._fetch_full_historical_data(symbol, timeframe, days, end_time)

    async def _fetch_limited_data(self, symbol: str, timeframe: str, end_time: int) -> bool:
        """Fetch only the most recent MIN_CANDLES candles"""
        timeframe_minutes = int(timeframe)

        # Go back exactly enough time to cover MIN_CANDLES
        start_time = end_time - (MIN_CANDLES * timeframe_minutes * 60 * 1000)

        # Use the rate-limited request method
        result = await self._make_rate_limited_request(symbol, timeframe, start_time, end_time, MIN_CANDLES)

        if result:
            key = f"{symbol}_{timeframe}"
            if key not in self.memory_data:
                self.memory_data[key] = deque(maxlen=MIN_CANDLES)
            
            # Sort by timestamp (newest first) and limit to MIN_CANDLES
            result.sort(key=lambda x: x['timestamp'], reverse=True)
            if len(result) > MIN_CANDLES:
                result = result[:MIN_CANDLES]

            self.memory_data[key].extend(result)
            
            # Save to CSV using CSV manager
            success = self.csv_manager.write_csv_data(symbol, timeframe, result)
            
            if success:
                logger.info(f"[OK] Fetched {len(result)} candles for {symbol}_{timeframe}")
                return True
            else:
                logger.error(f"[FAIL] Failed to save CSV for {symbol}_{timeframe}")
                return False
        else:
            logger.error(f"[FAIL] Failed to fetch {symbol}_{timeframe}")
            return False

    async def _fetch_full_historical_data(self, symbol: str, timeframe: str, days: int, end_time: int) -> bool:
        """Fetch full historical data using optimized chunking"""
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        all_candles = []
        
        # Calculate optimal chunk parameters
        total_candles, num_chunks, candles_per_chunk, chunk_duration = \
            self._calculate_chunk_parameters(timeframe, days)
        
        logger.info(f"[DATA] Fetching {symbol}_{timeframe} using {num_chunks} chunks...")
        logger.info(f"[DATA] Time range: {datetime.fromtimestamp(start_time/1000)} to {datetime.fromtimestamp(end_time/1000)}")
        
        successful_chunks = 0
        failed_chunks = 0
        
        for chunk_index in range(num_chunks):
            # Calculate time range for this chunk
            chunk_end_time = end_time - (chunk_index * chunk_duration)
            chunk_start_time = max(start_time, chunk_end_time - chunk_duration)
            
            # Stop if we've reached the start time
            if chunk_start_time >= chunk_end_time:
                break
                
            logger.info(f"[DATA] Fetching chunk {chunk_index + 1}/{num_chunks}: "
                    f"{datetime.fromtimestamp(chunk_start_time/1000)} to "
                    f"{datetime.fromtimestamp(chunk_end_time/1000)}")
            
            # Make rate-limited request for this chunk with retry mechanism
            chunk_candles = await self._make_chunk_request_with_retry(
                symbol, timeframe, chunk_start_time, chunk_end_time, candles_per_chunk
            )
            
            if chunk_candles:
                all_candles.extend(chunk_candles)
                successful_chunks += 1
                logger.info(f"[OK] Chunk {chunk_index + 1}: Got {len(chunk_candles)} candles")
            else:
                failed_chunks += 1
                logger.error(f"[FAIL] Failed to fetch chunk {chunk_index + 1}")
                
            # If we fail too many chunks, break early
            if failed_chunks >= 3:
                logger.error(f"[FAIL] Too many failed chunks ({failed_chunks}), stopping early")
                break
                
            # Add delay between chunks
            await asyncio.sleep(self.config.BULK_REQUEST_DELAY_MS / 1000)
        
        # Store in memory and CSV - FIXED VERSION
        if all_candles:
            key = f"{symbol}_{timeframe}"
            
            # For memory storage, we still limit to prevent memory issues
            if key not in self.memory_data:
                self.memory_data[key] = deque(maxlen=1000)  # Increased from 50 to 1000
            
            # Sort by timestamp (newest first)
            all_candles.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Add to memory storage (limited)
            memory_candles = all_candles.copy()  # Don't modify the original list
            self.memory_data[key].extend(memory_candles)
            
            # Save ALL candles to CSV (not truncated)
            success = self.csv_manager.write_csv_data(symbol, timeframe, all_candles)
            
            if success:
                logger.info(f"[OK] Fetched {len(all_candles)} candles for {symbol}_{timeframe}")
                logger.info(f"[OK] Saved {len(all_candles)} candles to CSV")
                return True
            else:
                logger.error(f"[FAIL] Failed to save CSV for {symbol}_{timeframe}")
                return False
        else:
            logger.error(f"[FAIL] Failed to fetch {symbol}_{timeframe}")
            return False
        
    async def _make_chunk_request_with_retry(self, symbol: str, timeframe: str,
                                           start_time: int, end_time: int, 
                                           limit: int, max_retries: int = 3) -> List[Dict]:
        """Make a request with exponential backoff retry mechanism"""
        retry_delay = 1.0  # Start with 1 second delay
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info(f"[WAIT] Retry attempt {attempt + 1}/{max_retries} for {symbol}_{timeframe}")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                
                result = await self._make_rate_limited_request(
                    symbol, timeframe, start_time, end_time, limit
                )
                
                if result:
                    return result
                else:
                    logger.warning(f"[FAIL] Empty result for {symbol}_{timeframe} (attempt {attempt + 1}/{max_retries})")
                    
            except Exception as e:
                logger.error(f"[FAIL] Exception in chunk request for {symbol}_{timeframe}: {e} "
                           f"(attempt {attempt + 1}/{max_retries})")
        
        logger.error(f"[FAIL] Max retries reached for {symbol}_{timeframe}")
        return []

    async def _make_rate_limited_request(self, symbol: str, timeframe: str,
                                       start_time: int, end_time: int, limit: int):
        """Make a rate-limited request to Bybit API"""
        url = f"https://api.bybit.com/v5/market/kline"
        max_retries = self.config.BULK_MAX_RETRIES
        
        for attempt in range(max_retries):
            try:
                # Add delay before each request (except the first one)
                if attempt > 0:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                
                params = {
                    "category": "linear",
                    "symbol": symbol,
                    "interval": timeframe,
                    "start": start_time,
                    "end": end_time,
                    "limit": min(limit, 999)  # Ensure we don't exceed Bybit's limit
                }
                
                async with self.session.get(url, params=params) as response:
                    # Check for rate limiting
                    content_type = response.headers.get('Content-Type', '')
                    if 'text/html' in content_type or 'text/plain' in content_type:
                        logger.warning(f"[FAIL] Rate limited for {symbol}_{timeframe} (attempt {attempt + 1}/{max_retries})")
                        if attempt < max_retries - 1:
                            continue
                        else:
                            return None
                    
                    if response.status == 200:
                        data = await response.json()
                        if data.get("retCode") == 0:
                            candles = data["result"]["list"]
                            if candles:
                                processed_candles = []
                                for candle in candles:
                                    processed = {
                                        'timestamp': int(candle[0]),
                                        'open': float(candle[1]),
                                        'high': float(candle[2]),
                                        'low': float(candle[3]),
                                        'close': float(candle[4]),
                                        'volume': float(candle[5])
                                    }
                                    processed_candles.append(processed)
                                return processed_candles
                            else:
                                return []
                        else:
                            logger.error(f"[FAIL] API error for {symbol}_{timeframe}: {data.get('retMsg')}")
                            return None
                    elif response.status == 403:
                        logger.warning(f"[FAIL] 403 Forbidden for {symbol}_{timeframe} (attempt {attempt + 1}/{max_retries})")
                        if attempt < max_retries - 1:
                            continue
                        else:
                            return None
                    else:
                        logger.error(f"[FAIL] HTTP {response.status} for {symbol}_{timeframe}")
                        return None
                        
            except Exception as e:
                logger.error(f"[FAIL] Exception for {symbol}_{timeframe}: {e} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    continue
                else:
                    return None
        
        return None

    def get_memory_data(self):
        """Access to in-memory data"""
        return self.memory_data

    async def save_to_csv(self, directory: str = "data"):
        """Save all data to CSV using CSV manager"""
        logger.info("[SAVE] Saving all data to CSV using CSV manager...")
        
        for key, candles in self.memory_data.items():
            if candles:
                symbol, timeframe = key.split('_')
                success = self.csv_manager.write_csv_data(symbol, timeframe, list(candles))
                
                if not success:
                    logger.error(f"[FAIL] Failed to save {key} to CSV")
        
        logger.info("[OK] CSV save completed")

    async def close(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()

