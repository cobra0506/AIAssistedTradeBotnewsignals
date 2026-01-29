# hybrid_system.py - Updated with fixed WebSocket handling
import asyncio
import json
import os
import logging
from typing import Dict, List, Any
from datetime import datetime
import time
from .config import DataCollectionConfig  # Relative import
from .optimized_data_fetcher import OptimizedDataFetcher
from .websocket_handler import WebSocketHandler
from .csv_manager import CSVManager
from .logging_utils import setup_logging

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

class HybridTradingSystem:
    def __init__(self, config):
        self.config = config
        self.data_fetcher = OptimizedDataFetcher(config)
        self.csv_manager = CSVManager(config)
        self.is_initialized = False
        # Use shared WebSocket manager
        from .shared_websocket_manager import SharedWebSocketManager
        self.shared_ws_manager = SharedWebSocketManager()
        self.websocket_handler = None  # Will be set during initialization

    async def initialize(self):
        """Initialize both fetchers"""
        if not self.is_initialized:
            await self.data_fetcher.initialize()
            # Initialize shared WebSocket
            await self.shared_ws_manager.initialize(self.config)
            self.websocket_handler = self.shared_ws_manager.get_websocket_handler()
            self.is_initialized = True

    async def fetch_data_hybrid(self, symbols: List[str] = None, timeframes: List[str] = None,
                           days: int = None, mode: str = "full"):
        """
        Fetch data in hybrid mode - WebSocket + Historical
        mode: "full" = all historical data
            "recent" = only 50 most recent entries  
            "live" = only real-time data
        """

        # Update status to show data collection is running
        self.update_collection_status(running=True)

        # Use config values if not provided
        if symbols is None:
            symbols = self.config.SYMBOLS
        if timeframes is None:
            timeframes = self.config.TIMEFRAMES
        if days is None:
            days = self.config.DAYS_TO_FETCH
        
        logger.info(f"[DEBUG] Starting data fetch in mode: {mode}")
        
        # Get symbols to use
        if self.config.FETCH_ALL_SYMBOLS:
            logger.info("[DATA] Fetching all symbols from Bybit...")
            all_symbols = await self.data_fetcher._get_all_symbols()
            symbols_to_use = all_symbols
        else:
            symbols_to_use = symbols
            logger.info(f"[DATA] Using {len(symbols_to_use)} symbols from configuration")
        
        # Setup WebSocket if enabled (NON-BLOCKING NOW)
        if self.config.ENABLE_WEBSOCKET and self.websocket_handler:
            logger.info(f"[WS] Setting up WebSocket for real-time updates...")
            
            # Update symbols in the WebSocket handler
            self.websocket_handler.symbols = symbols_to_use
            self.websocket_handler.config.TIMEFRAMES = timeframes
            
            # Start WebSocket connection (this will now return immediately)
            await self.websocket_handler.connect()
            
            # Give WebSocket a moment to establish connection
            await asyncio.sleep(1)
            
            if self.websocket_handler.running:
                logger.info(f"[OK] WebSocket connection established and running")
                logger.info(f"[OK] Monitoring {len(symbols_to_use)} symbols and {len(timeframes)} timeframes")
            else:
                logger.error(f"[FAIL] WebSocket connection failed")
        else:
            logger.info(f"[WS] WebSocket disabled (ENABLE_WEBSOCKET=False) or not available")
        
        # CRITICAL: This will now execute because WebSocket setup is non-blocking
        if mode in ["full", "recent"]:
            logger.info(f"[DATA] Fetching historical data...")
            limit_50 = (mode == "recent")
            
            # Add progress tracking
            total_tasks = len(symbols_to_use) * len(timeframes)
            completed_tasks = 0
            
            # Process in smaller batches to avoid overwhelming the API
            batch_size = 5  # Process 5 symbols at a time
            for i in range(0, len(symbols_to_use), batch_size):
                batch_symbols = symbols_to_use[i:i+batch_size]
                logger.info(f"[DATA] Processing batch {i//batch_size+1}/{(len(symbols_to_use)+batch_size-1)//batch_size}")
                
                # Create tasks for this batch
                tasks = []
                for symbol in batch_symbols:
                    for timeframe in timeframes:
                        task = asyncio.create_task(
                            self.data_fetcher._fetch_symbol_timeframe(symbol, timeframe, days, limit_50)
                        )
                        tasks.append(task)
                
                # Execute batch tasks concurrently
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results and update progress
                for result in batch_results:
                    completed_tasks += 1
                    progress = (completed_tasks / total_tasks) * 100
                    logger.info(f"[PROGRESS] Historical data fetch: {progress:.1f}%")
                    
                    if result is True:
                        logger.debug(f"[OK] Task completed successfully")
                    elif result is False:
                        logger.error(f"[FAIL] Task failed")
                    elif isinstance(result, Exception):
                        logger.error(f"[FAIL] Task exception: {result}")
                
                # Add delay between batches to avoid rate limiting
                if i + batch_size < len(symbols_to_use):
                    logger.info(f"[DATA] Pausing between batches to avoid rate limiting...")
                    await asyncio.sleep(2)
            
            logger.info(f"[COMPLETE] Historical data fetching completed")
            
            # Display sample data for verification
            logger.info(f"[SAMPLE] Displaying sample data for verification:")
            for symbol in symbols_to_use[:2]:  # Show first 2 symbols
                for timeframe in timeframes[:1]:  # Show first timeframe
                    filename = f"{self.config.DATA_DIR}/{symbol}_{timeframe}.csv"
                    if os.path.exists(filename):
                        data = self.csv_manager.load_csv(filename)
                        if len(data) > 0:
                            latest = data[-1]
                            # Convert timestamp to datetime
                            dt = datetime.fromtimestamp(latest['timestamp'] / 1000)
                            datetime_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                            logger.info(f" {symbol}_{timeframe} - Latest: {datetime_str} - O:{latest['open']} H:{latest['high']} L:{latest['low']} C:{latest['close']}")
                        else:
                            logger.info(f" {symbol}_{timeframe} - No data found")
                    else:
                        logger.info(f" {symbol}_{timeframe} - File not found")
        
        logger.info(f"[COMPLETE] Hybrid data fetch completed")
        
        # Update status to show data collection is stopping
        self.update_collection_status(running=False)

        return True

    def _show_sample_data(self, symbols: List[str], timeframes: List[str]):
        """Show sample of fetched data"""
        logger.info("\n[INFO] Sample of fetched historical data:")
        
        for symbol in symbols[:2]:  # Show first 2 symbols
            for timeframe in timeframes[:2]:  # Show first 2 timeframes
                key = f"{symbol}_{timeframe}"
                data = self.data_fetcher.get_memory_data().get(key, [])
                
                if data:
                    logger.info(f" {symbol}_{timeframe}: {len(data)} candles")
                    if len(data) > 0:
                        latest = data[-1]
                        # Convert timestamp to datetime
                        dt = datetime.fromtimestamp(latest['timestamp'] / 1000)
                        datetime_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                        logger.info(f" Latest: {datetime_str} - O:{latest['open']} H:{latest['high']} L:{latest['low']} C:{latest['close']}")
                else:
                    logger.info(f" {symbol}_{timeframe}: No data")

    def get_data(self, symbol: str, timeframe: str, source: str = "memory"):
        """Get data from memory or combine historical + real-time"""
        key = f"{symbol}_{timeframe}"
        
        if source == "memory":
            return list(self.data_fetcher.get_memory_data().get(key, []))
        elif source == "websocket":
            return list(self.websocket_handler.get_real_time_data(symbol, timeframe))
        elif source == "csv":
            # Read from CSV using CSV manager
            return self.csv_manager.read_csv_data(symbol, timeframe)
        else:
            # Combine both memory and real-time
            historical = list(self.data_fetcher.get_memory_data().get(key, []))
            real_time = list(self.websocket_handler.get_real_time_data(symbol, timeframe))
            return historical + real_time
        
    async def cleanup(self):
        """Cleanup resources"""
        if self.is_initialized:
            # Cleanup data fetcher
            if hasattr(self.data_fetcher, 'cleanup'):
                await self.data_fetcher.cleanup()
            
            # Cleanup websocket manager
            if hasattr(self.shared_ws_manager, 'shutdown'):
                await self.shared_ws_manager.shutdown()
            
            self.is_initialized = False
            logger.info("[CLEANUP] HybridTradingSystem resources cleaned up")

    async def save_to_csv(self, directory: str = "data"):
        """Save all data to CSV using CSV manager"""
        logger.info("[SAVE] Saving all data to CSV using CSV manager...")
        
        # Get all data from memory
        memory_data = self.data_fetcher.get_memory_data()
        
        for key, candles in memory_data.items():
            if candles:
                symbol, timeframe = key.split('_')
                success = self.csv_manager.write_csv_data(symbol, timeframe, list(candles))
                
                if not success:
                    logger.error(f"[FAIL] Failed to save {key} to CSV")
        
        logger.info("[OK] CSV save completed")

    async def update_csv_with_realtime_data(self, directory: str = "data"):
        """Update CSV files with real-time data using CSV manager"""
        # Skip if WebSocket is disabled
        if not self.config.ENABLE_WEBSOCKET:
            logger.info("CSV updates skipped (ENABLE_WEBSOCKET=False)")
            return
            
        if not self.config.LIMIT_TO_50_ENTRIES:
            logger.info("CSV updates disabled (LIMIT_TO_50_ENTRIES is False)")
            return
        
        logger.info("[WS] Updating CSV files with real-time data...")
        
        # Get symbols to use
        if self.config.FETCH_ALL_SYMBOLS:
            # For WebSocket, we need to get the symbols that were actually subscribed to
            symbols_to_use = self.websocket_handler.symbols
        else:
            symbols_to_use = self.config.SYMBOLS
        
        for symbol in symbols_to_use:
            for timeframe in self.websocket_handler.config.TIMEFRAMES:
                # Get real-time data
                real_time_data = self.websocket_handler.get_real_time_data(symbol, timeframe)
                
                if real_time_data:
                    # Convert real-time data to CSV format
                    csv_candles = []
                    for candle in real_time_data:
                        csv_candle = {
                            'timestamp': candle['timestamp'],
                            'open': float(candle['open']),
                            'high': float(candle['high']),
                            'low': float(candle['low']),
                            'close': float(candle['close']),
                            'volume': float(candle['volume'])
                        }
                        csv_candles.append(csv_candle)
                    
                    # Use CSV manager to append new data
                    success = self.csv_manager.append_new_data(symbol, timeframe, csv_candles)
                    
                    if success:
                        logger.debug(f"[OK] Updated {symbol}_{timeframe} with real-time data")
                    else:
                        logger.error(f"[FAIL] Failed to update {symbol}_{timeframe} with real-time data")

    async def close(self):
        """Clean up resources"""
        await self.data_fetcher.close()

    # Add this test method to your hybrid_system.py
    async def test_websocket_blocking(self):
        """Test to confirm WebSocket is blocking historical data fetch"""
        logger.info("[TEST] Starting WebSocket blocking test...")
        
        # Test 1: Check if WebSocket setup returns
        logger.info("[TEST] Testing WebSocket setup...")
        start_time = time.time()
        
        if self.config.ENABLE_WEBSOCKET and self.websocket_handler:
            logger.info("[TEST] WebSocket enabled, testing connection...")
            # This should hang if our diagnosis is correct
            try:
                await asyncio.wait_for(self.websocket_handler.connect(), timeout=5.0)
                logger.info("[TEST] WebSocket setup completed (unexpected)")
            except asyncio.TimeoutError:
                logger.info("[TEST] CONFIRMED: WebSocket setup is blocking (timeout after 5 seconds)")
                return True
        
        logger.info("[TEST] WebSocket setup completed normally")
        return False
    
    def update_collection_status(self, running=True):
        """Update the data collection status file"""
        # Create data directory if it doesn't exist
        if not os.path.exists(self.config.DATA_DIR):
            os.makedirs(self.config.DATA_DIR)
        
        status_file = os.path.join(self.config.DATA_DIR, "collection_status.json")
        
        status = {
            'running': running,
            'last_updated': datetime.now().isoformat(),
            'pid': os.getpid()
        }
        
        with open(status_file, 'w') as f:
            json.dump(status, f)

