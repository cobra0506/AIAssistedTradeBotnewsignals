# main.py - Updated with more frequent CSV updates and GUI startup
import os
import csv
import asyncio
import time
import json
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any
from .config import DataCollectionConfig
from .hybrid_system import HybridTradingSystem
from .data_integrity import DataIntegrityChecker

# Global configuration
config = DataCollectionConfig()

def debug_websocket_message(message: str):
    """Debug callback to see all WebSocket messages"""
    try:
        data = json.loads(message)
        if "topic" in data:
            print(f"🔍 Debug: Received message for topic: {data['topic']}")
        elif data.get("op") == "subscribe":
            print(f"🔍 Debug: Subscription response: {data}")
    except:
        pass  # Ignore JSON parsing errors for binary data

def print_memory_usage():
    """Print current memory usage (monitoring only)"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    print(f"💾 Memory usage: {memory_mb:.1f} MB")

async def console_main():
    """Main function using the optimized hybrid system"""
    print("="*60)
    print("OPTIMIZED AI ASSISTED TRADING BOT")
    print("="*60)
    
    # Initialize the hybrid system
    hybrid_system = HybridTradingSystem(config)
    await hybrid_system.initialize()
    
    # Add debug callback to WebSocket
    if config.ENABLE_WEBSOCKET:
        hybrid_system.websocket_handler.add_debug_callback(debug_websocket_message)
    
    try:
        # Determine data collection mode
        if config.LIMIT_TO_50_ENTRIES:
            mode = "recent"
            print("📊 MODE: Recent 50 entries only")
        else:
            mode = "full"
            print("📊 MODE: Full historical data")
        
        if config.ENABLE_WEBSOCKET:
            print("📡 MODE: Live updates enabled")
        else:
            print("📡 MODE: Historical data only")
        
        # Get symbols to process
        if config.FETCH_ALL_SYMBOLS:
            print("🔍 Fetching all available symbols...")
            # Fetch all symbols from the data fetcher
            all_symbols = await hybrid_system.data_fetcher._get_all_symbols()
            
            # Limit to a reasonable number for testing (optional)
            # Remove this line to fetch all symbols
            symbols = all_symbols#[:10]  # Limit to first 10 symbols for testing
            
            print(f"📈 Processing {len(symbols)} symbols (showing first 10): {', '.join(symbols[:10])}")
            if len(symbols) > 10:
                print(f"... and {len(symbols) - 10} more symbols")
        else:
            symbols = config.SYMBOLS
            print(f"📈 Processing {len(symbols)} symbols: {', '.join(symbols)}")
        
        print(f"📈 Processing {len(symbols)} symbols: {', '.join(symbols)}")
        print(f"⏰ Timeframes: {', '.join(config.TIMEFRAMES)}")
        
        # Initial memory check
        print_memory_usage()
        
        # Fetch data with optimized system
        start_time = time.time()
        
        await hybrid_system.fetch_data_hybrid(
            symbols=symbols,
            timeframes=config.TIMEFRAMES,
            days=config.DAYS_TO_FETCH,
            mode=mode
        )
        
        # Performance reporting
        end_time = time.time()
        duration = end_time - start_time
        
        print("="*60)
        print("DATA COLLECTION COMPLETED")
        print("="*60)
        print(f"⏱️  Total time: {duration:.2f} seconds")
        print(f"📊 Mode: {mode}")
        print(f"📡 WebSocket: {'Enabled' if config.ENABLE_WEBSOCKET else 'Disabled'}")
        
        # Save data to CSV if needed
        if hasattr(hybrid_system, 'save_to_csv'):
            print("💾 Saving data to CSV files...")
            await hybrid_system.save_to_csv(config.DATA_DIR)
            print("✅ CSV files saved successfully")
        
        # Display final data status
        print("\n" + "="*60)
        print("FINAL DATA STATUS")
        print("="*60)
        
        for symbol in symbols:
            for timeframe in config.TIMEFRAMES:
                # Get historical data
                hist_data = hybrid_system.get_data(symbol, timeframe, "memory")
                # Get real-time data
                rt_data = hybrid_system.get_data(symbol, timeframe, "websocket")
                
                print(f"\n{symbol}_{timeframe}:")
                print(f"  Historical candles: {len(hist_data)}")
                print(f"  Real-time candles: {len(rt_data)}")
                
                if hist_data:
                    latest_hist = hist_data[-1]
                    dt = datetime.fromtimestamp(latest_hist['timestamp'] / 1000)
                    datetime_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                    print(f"  Latest historical: {datetime_str}")
                
                if rt_data:
                    latest_rt = rt_data[-1]
                    dt = datetime.fromtimestamp(latest_rt['timestamp'] / 1000)
                    datetime_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                    print(f"  Latest real-time: {datetime_str}")
        
        # Final memory check
        print_memory_usage()
        
        # Run integrity check if enabled
        if config.RUN_INTEGRITY_CHECK:
            print("\n" + "="*60)
            print("RUNNING INTEGRITY CHECK")
            print("="*60)
            integrity_checker = DataIntegrityChecker(config)
            results = integrity_checker.check_all_files()
            print(f"Files checked: {results['files_checked']}")
            print(f"Files with issues: {results['files_with_issues']}")
            print(f"Total gaps: {results['total_gaps']}")
            
            # Fill gaps if enabled
            if config.RUN_GAP_FILLING and results['total_gaps'] > 0:
                print("Filling gaps...")
                integrity_checker.fill_all_gaps()
        
        # Keep running for live updates if WebSocket is enabled
        if config.ENABLE_WEBSOCKET:
            print("\n" + "="*60)
            print("LIVE UPDATES MODE - Press Ctrl+C to stop")
            print("="*60)
            print("⏰ CSV updates every 10 seconds")
            print("⏰ Status updates every 10 seconds")
            
            try:
                # Keep the program running for live updates
                live_update_count = 0
                last_csv_update = time.time()
                
                while True:
                    await asyncio.sleep(5)  # Check every 5 seconds instead of 10
                    live_update_count += 1
                    
                    current_time = time.time()
                    
                    # Update CSV files every 10 seconds (more frequent)
                    if current_time - last_csv_update >= 10:
                        print(f"\n📡 Live update #{live_update_count} at {datetime.now().strftime('%H:%M:%S')}:")
                        await hybrid_system.update_csv_with_realtime_data(config.DATA_DIR)
                        last_csv_update = current_time
                        
                        # Display current status
                        for symbol in symbols:
                            for timeframe in config.TIMEFRAMES:
                                rt_data = hybrid_system.get_data(symbol, timeframe, "websocket")
                                if rt_data:
                                    latest = rt_data[-1]
                                    dt = datetime.fromtimestamp(latest['timestamp'] / 1000)
                                    datetime_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                                    print(f"  {symbol}_{timeframe}: {len(rt_data)} candles, latest: {datetime_str}")
                    
                    # Show brief status every 5 seconds (without CSV update)
                    else:
                        # Every other iteration (every 10 seconds) show a brief status
                        if live_update_count % 2 == 0:
                            print(f"⏰ Tick... {datetime.now().strftime('%H:%M:%S')} (WebSocket: {'Connected' if hybrid_system.websocket_handler.running else 'Disconnected'})")
                    
            except KeyboardInterrupt:
                print("\n🛑 Stopping live updates...")
        
        print("\n✅ Program completed successfully")
    
    finally:
        # Clean up resources
        await hybrid_system.close()

async def test_websocket_functionality():
    """Test WebSocket functionality with the hybrid system"""
    print("="*60)
    print("WEBSOCKET FUNCTIONALITY TEST")
    print("="*60)
    
    # Create test configuration
    test_config = DataCollectionConfig()
    test_config.SYMBOLS = ['BTCUSDT']
    test_config.TIMEFRAMES = ['1']
    test_config.DAYS_TO_FETCH = 1
    test_config.ENABLE_WEBSOCKET = True
    test_config.LIMIT_TO_50_ENTRIES = True
    
    # Initialize hybrid system
    hybrid_system = HybridTradingSystem(test_config)
    await hybrid_system.initialize()
    
    # Add debug callback to WebSocket
    hybrid_system.websocket_handler.add_debug_callback(debug_websocket_message)
    
    # Test results tracking
    test_results = {
        'candles_received': 0,
        'start_time': time.time(),
        'last_candle_time': None
    }
    
    def test_callback(symbol: str, timeframe: str, candle: Dict):
        """Test callback to track received candles"""
        test_results['candles_received'] += 1
        test_results['last_candle_time'] = candle['timestamp']
        print(f"📊 TEST: Received candle #{test_results['candles_received']} for {symbol}_{timeframe}")
        print(f"   Timestamp: {candle['timestamp']}")
        print(f"   Confirm: {candle.get('confirm', False)}")
    
    # Add callback to WebSocket handler
    hybrid_system.websocket_handler.add_callback(test_callback)
    
    # Start data collection
    await hybrid_system.fetch_data_hybrid(
        symbols=test_config.SYMBOLS,
        timeframes=test_config.TIMEFRAMES,
        days=test_config.DAYS_TO_FETCH,
        mode="live"
    )
    
    # Wait for test duration (2 minutes)
    print("🧪 Running test for 2 minutes...")
    await asyncio.sleep(120)
    
    # Print test results
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Test duration: 120 seconds")
    print(f"Candles received: {test_results['candles_received']}")
    print(f"Last candle time: {test_results['last_candle_time']}")
    
    if test_results['candles_received'] > 0:
        print("✅ WebSocket test PASSED")
    else:
        print("❌ WebSocket test FAILED")

def main():
    """Main function - Always starts GUI with fallback to console"""
    try:
        # Import and start GUI
        print("Starting AI Assisted TradeBot GUI...")
        import shared_modules.data_collection.gui_monitor as gui_monitor
        gui_monitor.main()
    except ImportError:
        print("GUI not available, running in console mode...")
        asyncio.run(console_main())
    except Exception as e:
        print(f"Failed to start GUI: {e}")
        print("Falling back to console mode...")
        asyncio.run(console_main())

if __name__ == "__main__":

        main()

'''# main.py - Updated with more frequent CSV updates
import os
import csv
import asyncio
import time
import json
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any
from config import DataCollectionConfig
from hybrid_system import HybridTradingSystem
from data_integrity import DataIntegrityChecker  # Keep if you want integrity checks

# Global configuration
config = DataCollectionConfig()

def debug_websocket_message(message: str):
    """Debug callback to see all WebSocket messages"""
    try:
        data = json.loads(message)
        if "topic" in data:
            print(f"🔍 Debug: Received message for topic: {data['topic']}")
        elif data.get("op") == "subscribe":
            print(f"🔍 Debug: Subscription response: {data}")
    except:
        pass  # Ignore JSON parsing errors for binary data

async def main():
    """Main function using the optimized hybrid system"""
    print("="*60)
    print("OPTIMIZED AI ASSISTED TRADING BOT")
    print("="*60)
    
    # Initialize the hybrid system
    hybrid_system = HybridTradingSystem(config)
    await hybrid_system.initialize()
    
    # Add debug callback to WebSocket
    if config.ENABLE_WEBSOCKET:
        hybrid_system.websocket_handler.add_debug_callback(debug_websocket_message)
    
    try:
        # Determine data collection mode
        if config.LIMIT_TO_50_ENTRIES:
            mode = "recent"
            print("📊 MODE: Recent 50 entries only")
        else:
            mode = "full"
            print("📊 MODE: Full historical data")
        
        if config.ENABLE_WEBSOCKET:
            print("📡 MODE: Live updates enabled")
        else:
            print("📡 MODE: Historical data only")
        
        # Get symbols to process
        if config.FETCH_ALL_SYMBOLS:
            print("🔍 Fetching all available symbols...")
            # Fetch all symbols from the data fetcher
            all_symbols = await hybrid_system.data_fetcher._get_all_symbols()
            
            # Limit to a reasonable number for testing (optional)
            # Remove this line to fetch all symbols
            symbols = all_symbols#[:10]  # Limit to first 10 symbols for testing
            
            print(f"📈 Processing {len(symbols)} symbols (showing first 10): {', '.join(symbols[:10])}")
            if len(symbols) > 10:
                print(f"... and {len(symbols) - 10} more symbols")
        else:
            symbols = config.SYMBOLS
            print(f"📈 Processing {len(symbols)} symbols: {', '.join(symbols)}")
        
        print(f"📈 Processing {len(symbols)} symbols: {', '.join(symbols)}")
        print(f"⏰ Timeframes: {', '.join(config.TIMEFRAMES)}")
        
        # Fetch data with optimized system
        start_time = time.time()
        
        await hybrid_system.fetch_data_hybrid(
            symbols=symbols,
            timeframes=config.TIMEFRAMES,
            days=config.DAYS_TO_FETCH,
            mode=mode
        )
        
        # Performance reporting
        end_time = time.time()
        duration = end_time - start_time
        
        print("="*60)
        print("DATA COLLECTION COMPLETED")
        print("="*60)
        print(f"⏱️  Total time: {duration:.2f} seconds")
        print(f"📊 Mode: {mode}")
        print(f"📡 WebSocket: {'Enabled' if config.ENABLE_WEBSOCKET else 'Disabled'}")
        
        # Save data to CSV if needed
        if hasattr(hybrid_system, 'save_to_csv'):
            print("💾 Saving data to CSV files...")
            await hybrid_system.save_to_csv(config.DATA_DIR)
            print("✅ CSV files saved successfully")
        
        # Display final data status
        print("\n" + "="*60)
        print("FINAL DATA STATUS")
        print("="*60)
        
        for symbol in symbols:
            for timeframe in config.TIMEFRAMES:
                # Get historical data
                hist_data = hybrid_system.get_data(symbol, timeframe, "memory")
                # Get real-time data
                rt_data = hybrid_system.get_data(symbol, timeframe, "websocket")
                
                print(f"\n{symbol}_{timeframe}:")
                print(f"  Historical candles: {len(hist_data)}")
                print(f"  Real-time candles: {len(rt_data)}")
                
                if hist_data:
                    latest_hist = hist_data[-1]
                    dt = datetime.fromtimestamp(latest_hist['timestamp'] / 1000)
                    datetime_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                    print(f"  Latest historical: {datetime_str}")
                
                if rt_data:
                    latest_rt = rt_data[-1]
                    dt = datetime.fromtimestamp(latest_rt['timestamp'] / 1000)
                    datetime_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                    print(f"  Latest real-time: {datetime_str}")
        
        # Run integrity check if enabled
        if config.RUN_INTEGRITY_CHECK:
            print("\n" + "="*60)
            print("RUNNING INTEGRITY CHECK")
            print("="*60)
            integrity_checker = DataIntegrityChecker(config)
            results = integrity_checker.check_all_files()
            print(f"Files checked: {results['files_checked']}")
            print(f"Files with issues: {results['files_with_issues']}")
            print(f"Total gaps: {results['total_gaps']}")
        
        # Keep running for live updates if WebSocket is enabled
        if config.ENABLE_WEBSOCKET:
            print("\n" + "="*60)
            print("LIVE UPDATES MODE - Press Ctrl+C to stop")
            print("="*60)
            print("⏰ CSV updates every 10 seconds")
            print("⏰ Status updates every 10 seconds")
            
            try:
                # Keep the program running for live updates
                live_update_count = 0
                last_csv_update = time.time()
                
                while True:
                    await asyncio.sleep(5)  # Check every 5 seconds instead of 10
                    live_update_count += 1
                    
                    current_time = time.time()
                    
                    # Update CSV files every 10 seconds (more frequent)
                    if current_time - last_csv_update >= 10:
                        print(f"\n📡 Live update #{live_update_count} at {datetime.now().strftime('%H:%M:%S')}:")
                        await hybrid_system.update_csv_with_realtime_data(config.DATA_DIR)
                        last_csv_update = current_time
                        
                        # Display current status
                        for symbol in symbols:
                            for timeframe in config.TIMEFRAMES:
                                rt_data = hybrid_system.get_data(symbol, timeframe, "websocket")
                                if rt_data:
                                    latest = rt_data[-1]
                                    dt = datetime.fromtimestamp(latest['timestamp'] / 1000)
                                    datetime_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                                    print(f"  {symbol}_{timeframe}: {len(rt_data)} candles, latest: {datetime_str}")
                    
                    # Show brief status every 5 seconds (without CSV update)
                    else:
                        # Every other iteration (every 10 seconds) show a brief status
                        if live_update_count % 2 == 0:
                            print(f"⏰ Tick... {datetime.now().strftime('%H:%M:%S')} (WebSocket: {'Connected' if hybrid_system.websocket_handler.running else 'Disconnected'})")
                    
            except KeyboardInterrupt:
                print("\n🛑 Stopping live updates...")
        
        print("\n✅ Program completed successfully")
    
    finally:
        # Clean up resources
        await hybrid_system.close()

async def test_websocket_functionality():
    """Test WebSocket functionality with the hybrid system"""
    print("="*60)
    print("WEBSOCKET FUNCTIONALITY TEST")
    print("="*60)
    
    # Create test configuration
    test_config = DataCollectionConfig()
    test_config.SYMBOLS = ['BTCUSDT']
    test_config.TIMEFRAMES = ['1']
    test_config.DAYS_TO_FETCH = 1
    test_config.ENABLE_WEBSOCKET = True
    test_config.LIMIT_TO_50_ENTRIES = True
    
    # Initialize hybrid system
    hybrid_system = HybridTradingSystem(test_config)
    await hybrid_system.initialize()
    
    # Add debug callback to WebSocket
    hybrid_system.websocket_handler.add_debug_callback(debug_websocket_message)
    
    # Test results tracking
    test_results = {
        'candles_received': 0,
        'start_time': time.time(),
        'last_candle_time': None
    }
    
    def test_callback(symbol: str, timeframe: str, candle: Dict):
        """Test callback to track received candles"""
        test_results['candles_received'] += 1
        test_results['last_candle_time'] = candle['timestamp']
        print(f"📊 TEST: Received candle #{test_results['candles_received']} for {symbol}_{timeframe}")
        print(f"   Timestamp: {candle['timestamp']}")
        print(f"   Confirm: {candle.get('confirm', False)}")

    def print_memory_usage():
        """Print current memory usage (monitoring only)"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        print(f"💾 Memory usage: {memory_mb:.1f} MB")
    
    # Add callback to WebSocket handler
    hybrid_system.websocket_handler.add_callback(test_callback)
    
    # Start data collection
    await hybrid_system.fetch_data_hybrid(
        symbols=test_config.SYMBOLS,
        timeframes=test_config.TIMEFRAMES,
        days=test_config.DAYS_TO_FETCH,
        mode="live"
    )
    
    # Wait for test duration (2 minutes)
    print("🧪 Running test for 2 minutes...")
    await asyncio.sleep(120)
    
    # Print test results
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Test duration: 120 seconds")
    print(f"Candles received: {test_results['candles_received']}")
    print(f"Last candle time: {test_results['last_candle_time']}")
    
    if test_results['candles_received'] > 0:
        print("✅ WebSocket test PASSED")
    else:
        print("❌ WebSocket test FAILED")
        

if __name__ == "__main__":
    asyncio.run(main())'''


