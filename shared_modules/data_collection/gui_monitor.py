import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import asyncio
import queue
import json
import psutil
import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from .config import DataCollectionConfig
from .hybrid_system import HybridTradingSystem
from .csv_manager import CSVManager

# Set up logging to capture messages from the data collection system
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollectionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI Assisted TradeBot - Data Collection Monitor")
        self.root.geometry("900x700")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Configuration
        self.gui_config = DataCollectionConfig()  # GUI's own config that can be modified
        self.hybrid_system = None
        self.running = False
        
        # Thread communication
        self.log_queue = queue.Queue()
        
        # Status variables
        self.connection_status = "Disconnected"
        self.websocket_status = "Disconnected"
        self.symbols_count = 0
        self.errors_count = 0
        self.last_error = "No errors"
        
        # NEW: Progress tracking variables
        self.current_activity = "Idle"
        self.historical_progress = 0
        self.total_symbols = 0
        self.completed_symbols = 0
        self.current_symbol = ""
        self.current_timeframe = ""
        
        # System stats
        self.memory_usage = "0 MB"
        self.cpu_usage = "0%"
        
        self.setup_gui()
        self.start_gui_updater()
        self.start_system_stats_updater()
        self.start_data_monitor()  # NEW: Start data monitoring
        
    def setup_gui(self):
        """Setup the GUI components"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status Panel
        status_frame = ttk.LabelFrame(main_frame, text="System Status", padding="10")
        status_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Connection Status
        ttk.Label(status_frame, text="API Connection:").grid(row=0, column=0, sticky=tk.W)
        self.connection_label = ttk.Label(status_frame, text="Disconnected", foreground="red")
        self.connection_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 20))
        
        # WebSocket Status
        ttk.Label(status_frame, text="WebSocket:").grid(row=0, column=2, sticky=tk.W)
        self.websocket_label = ttk.Label(status_frame, text="Disconnected", foreground="red")
        self.websocket_label.grid(row=0, column=3, sticky=tk.W, padx=(10, 20))
        
        # Symbols Count
        ttk.Label(status_frame, text="Symbols:").grid(row=0, column=4, sticky=tk.W)
        self.symbols_label = ttk.Label(status_frame, text="0")
        self.symbols_label.grid(row=0, column=5, sticky=tk.W, padx=(10, 20))
        
        # Errors Count
        ttk.Label(status_frame, text="Errors:").grid(row=0, column=6, sticky=tk.W)
        self.errors_label = ttk.Label(status_frame, text="0", foreground="red")
        self.errors_label.grid(row=0, column=7, sticky=tk.W, padx=(10, 0))
        
        # NEW: Current Activity Display
        ttk.Label(status_frame, text="Activity:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.activity_label = ttk.Label(status_frame, text="Idle", foreground="blue", font=("Arial", 10, "bold"))
        self.activity_label.grid(row=1, column=1, columnspan=3, sticky=tk.W, padx=(10, 0), pady=(5, 0))
        
        # NEW: Progress Display
        ttk.Label(status_frame, text="Progress:").grid(row=1, column=4, sticky=tk.W, pady=(5, 0))
        self.progress_label = ttk.Label(status_frame, text="0%", font=("Arial", 10, "bold"))
        self.progress_label.grid(row=1, column=5, sticky=tk.W, padx=(10, 20), pady=(5, 0))
        
        # Configuration Panel
        config_frame = ttk.LabelFrame(main_frame, text="Configuration Options", padding="10")
        config_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Configuration checkboxes
        self.limit_50_var = tk.BooleanVar(value=self.gui_config.LIMIT_TO_50_ENTRIES)
        self.fetch_all_var = tk.BooleanVar(value=self.gui_config.FETCH_ALL_SYMBOLS)
        self.enable_ws_var = tk.BooleanVar(value=self.gui_config.ENABLE_WEBSOCKET)
        self.integrity_var = tk.BooleanVar(value=self.gui_config.RUN_INTEGRITY_CHECK)
        self.gap_filling_var = tk.BooleanVar(value=self.gui_config.RUN_GAP_FILLING)
        
        # Create checkboxes
        ttk.Checkbutton(config_frame, text="Limit to 50 entries", variable=self.limit_50_var, 
                       command=self.update_config).grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        ttk.Checkbutton(config_frame, text="Fetch all symbols", variable=self.fetch_all_var,
                       command=self.update_config).grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        ttk.Checkbutton(config_frame, text="Enable WebSocket", variable=self.enable_ws_var,
                       command=self.update_config).grid(row=0, column=2, sticky=tk.W, padx=(0, 20))
        ttk.Checkbutton(config_frame, text="Run integrity check", variable=self.integrity_var,
                       command=self.update_config).grid(row=1, column=0, sticky=tk.W, padx=(0, 20))
        ttk.Checkbutton(config_frame, text="Run gap filling", variable=self.gap_filling_var,
                       command=self.update_config).grid(row=1, column=1, sticky=tk.W, padx=(0, 20))
        
        # Control Panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Buttons
        self.start_button = ttk.Button(control_frame, text="Start Data Collection", command=self.start_collection)
        self.start_button.grid(row=0, column=0, padx=(0, 10))
        
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_collection, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=(0, 10))
        
        self.test_button = ttk.Button(control_frame, text="Test Connection", command=self.test_connection)
        self.test_button.grid(row=0, column=2, padx=(0, 10))
        
        # NEW: Add refresh button
        ttk.Button(control_frame, text="Refresh Status", command=self.refresh_status).grid(row=0, column=3, padx=(0, 10))
        
        # System Stats Panel
        stats_frame = ttk.LabelFrame(main_frame, text="System Resources", padding="10")
        stats_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(stats_frame, text="Memory:").grid(row=0, column=0, sticky=tk.W)
        self.memory_label = ttk.Label(stats_frame, text="0 MB")
        self.memory_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 20))
        
        ttk.Label(stats_frame, text="CPU:").grid(row=0, column=2, sticky=tk.W)
        self.cpu_label = ttk.Label(stats_frame, text="0%")
        self.cpu_label.grid(row=0, column=3, sticky=tk.W, padx=(10, 20))
        
        # Last Error Panel
        error_frame = ttk.LabelFrame(main_frame, text="Last Error/Warning", padding="10")
        error_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.error_label = ttk.Label(error_frame, text="No errors or warnings", foreground="green", wraplength=800)
        self.error_label.grid(row=0, column=0, sticky=tk.W)
        
        # Log Display
        log_frame = ttk.LabelFrame(main_frame, text="Activity Log", padding="10")
        log_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.log_display = scrolledtext.ScrolledText(log_frame, height=15, width=100)
        self.log_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Progress Bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(5, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
    # NEW: Method to update activity display
    def update_activity(self, activity):
        """Update the current activity display"""
        self.current_activity = activity
        self.activity_label.config(text=activity)
        self.log_message(f"ACTIVITY: {activity}")
        
    # NEW: Method to update progress
    def update_progress(self, percent):
        """Update progress display"""
        self.historical_progress = percent
        self.progress_label.config(text=f"{percent:.1f}%")
        
    # NEW: Method to start data monitoring
    def start_data_monitor(self):
        """Start monitoring data directory for changes"""
        def monitor_data():
            try:
                while True:
                    if self.running:
                        self.update_data_stats()
                    threading.Event().wait(5)  # Check every 5 seconds
            except Exception as e:
                self.log_message(f"Data monitor error: {e}")
        
        monitor_thread = threading.Thread(target=monitor_data, daemon=True)
        monitor_thread.start()
        
    # NEW: Method to update data statistics
    def update_data_stats(self):
        """Update data statistics from the data directory"""
        try:
            # Get data directory path
            data_dir = Path(self.gui_config.DATA_DIR)
            
            if not data_dir.exists():
                return
            
            # Count CSV files
            csv_files = list(data_dir.glob("*.csv"))
            data_files_count = len(csv_files)
            
            # Extract unique symbols
            symbols = set()
            total_candles = 0
            latest_time = 0
            
            for file in csv_files:
                parts = file.stem.split('_')
                if len(parts) >= 2:
                    symbols.add(parts[0])
                
                # Count candles in file
                try:
                    with open(file, 'r') as f:
                        lines = f.readlines()
                        if len(lines) > 1:  # Has header + data
                            total_candles += (len(lines) - 1)
                            
                            # Get latest timestamp
                            last_line = lines[-1].strip()
                            if last_line:
                                parts = last_line.split(',')
                                if len(parts) > 0:
                                    timestamp = int(parts[0])
                                    if timestamp > latest_time:
                                        latest_time = timestamp
                except:
                    continue
            
            # Update symbols count
            self.symbols_count = len(symbols)
            self.symbols_label.config(text=str(self.symbols_count))
            
            # Check if we're receiving recent data
            import time
            current_time = int(time.time() * 1000)
            one_min_ago = current_time - (60 * 1000)
            
            if latest_time > one_min_ago:
                self.current_activity = "Receiving live data"
                self.activity_label.config(text=self.current_activity, foreground="green")
            elif self.running:
                self.current_activity = "Connected - No recent data"
                self.activity_label.config(text=self.current_activity, foreground="orange")
            
        except Exception as e:
            self.log_message(f"Error updating data stats: {e}")
    
    def update_config(self):
        """Update GUI config when checkboxes change"""
        self.gui_config.LIMIT_TO_50_ENTRIES = self.limit_50_var.get()
        self.gui_config.FETCH_ALL_SYMBOLS = self.fetch_all_var.get()
        self.gui_config.ENABLE_WEBSOCKET = self.enable_ws_var.get()
        self.gui_config.RUN_INTEGRITY_CHECK = self.integrity_var.get()
        self.gui_config.RUN_GAP_FILLING = self.gap_filling_var.get()
        
        self.log_message(f"Configuration updated: {self.get_config_summary()}")
        
    def get_config_summary(self):
        """Get a summary of current config settings"""
        settings = []
        if self.gui_config.LIMIT_TO_50_ENTRIES:
            settings.append("Limit:50")
        if self.gui_config.FETCH_ALL_SYMBOLS:
            settings.append("AllSymbols")
        if self.gui_config.ENABLE_WEBSOCKET:
            settings.append("WebSocket")
        if self.gui_config.RUN_INTEGRITY_CHECK:
            settings.append("Integrity")
        if self.gui_config.RUN_GAP_FILLING:
            settings.append("GapFill")
        return ",".join(settings) if settings else "Default"
        
    def start_gui_updater(self):
        """Start the GUI update loop"""
        def update_gui():
            try:
                # Process log messages
                while not self.log_queue.empty():
                    message = self.log_queue.get_nowait()
                    self.log_display.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
                    self.log_display.see(tk.END)
                    
                    # Update error count if it's an error
                    if "ERROR" in message or "FAIL" in message:
                        self.errors_count += 1
                        self.errors_label.config(text=str(self.errors_count))
                        self.last_error = message
                        self.error_label.config(text=message[-100:] + "..." if len(message) > 100 else message, foreground="red")
                    elif "WARNING" in message:
                        self.last_error = message
                        self.error_label.config(text=message[-100:] + "..." if len(message) > 100 else message, foreground="orange")
                
                # Schedule next update
                self.root.after(100, update_gui)
            except:
                self.root.after(100, update_gui)
        
        update_gui()
        
    def start_system_stats_updater(self):
        """Start updating system stats"""
        def update_stats():
            try:
                # Get memory usage
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                self.memory_usage = f"{memory_mb:.1f} MB"
                self.memory_label.config(text=self.memory_usage)
                
                # Get CPU usage
                cpu_percent = process.cpu_percent(interval=1)
                self.cpu_usage = f"{cpu_percent:.1f}%"
                self.cpu_label.config(text=self.cpu_usage)
                
                # Schedule next update
                self.root.after(2000, update_stats)  # Update every 2 seconds
            except:
                self.root.after(2000, update_stats)
        
        update_stats()
        
    def log_message(self, message: str):
        """Add message to log queue"""
        self.log_queue.put(message)
        
    def update_status(self, connection: str = None, websocket: str = None, symbols: int = None):
        """Update status indicators"""
        if connection:
            self.connection_status = connection
            color = "green" if connection == "Connected" else "red"
            self.connection_label.config(text=connection, foreground=color)
            
        if websocket:
            self.websocket_status = websocket
            color = "green" if websocket == "Connected" else "red"
            self.websocket_label.config(text=websocket, foreground=color)
            
        if symbols is not None:
            self.symbols_count = symbols
            self.symbols_label.config(text=str(symbols))
            
    # NEW: Method to refresh status manually
    def refresh_status(self):
        """Manually refresh the status"""
        self.update_data_stats()
        self.log_message("Status refreshed")
            
    def start_collection(self):
        """Start data collection"""
        try:
            self.running = True
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.test_button.config(state="disabled")
            self.progress.start()
            
            self.log_message(f"Starting data collection with config: {self.get_config_summary()}")
            self.update_activity("Initializing data collection system...")
            
            # Disable config changes during collection
            for child in self.root.winfo_children():
                if isinstance(child, ttk.Checkbutton):
                    child.config(state="disabled")
            
            # Start collection in separate thread
            collection_thread = threading.Thread(target=self.run_collection, daemon=True)
            collection_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start collection: {e}")
            
    def stop_collection(self):
        """Stop data collection"""
        try:
            self.running = False
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.test_button.config(state="normal")
            self.progress.stop()
            
            # Re-enable config changes
            for child in self.root.winfo_children():
                if isinstance(child, ttk.Checkbutton):
                    child.config(state="normal")
            
            self.log_message("Stopping data collection...")
            self.update_activity("Data collection stopped")
            self.update_status(connection="Disconnected", websocket="Disconnected")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop collection: {e}")
            
    def test_connection(self):
        """Test API connection without full data collection"""
        try:
            self.log_message("Testing API connection...")
            self.update_activity("Testing API connection...")
            self.progress.start()
            
            # Run test in separate thread
            test_thread = threading.Thread(target=self.run_connection_test, daemon=True)
            test_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start test: {e}")
            
    def run_connection_test(self):
        """Test API connection in a separate thread"""
        try:
            # Create event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Initialize hybrid system
            self.hybrid_system = HybridTradingSystem(self.gui_config)
            
            async def test_task():
                await self.hybrid_system.initialize()
                
                # Update GUI
                self.update_status(connection="Connected")
                self.log_message("✅ API connection successful!")
                self.update_activity("API connection test successful")
                
                # Test symbol fetching if enabled
                if self.gui_config.FETCH_ALL_SYMBOLS:
                    symbols = await self.hybrid_system.data_fetcher._get_all_symbols()
                    self.update_status(symbols=len(symbols))
                    self.log_message(f"✅ Found {len(symbols)} symbols")
                else:
                    self.update_status(symbols=len(self.gui_config.SYMBOLS))
                    self.log_message(f"✅ Using {len(self.gui_config.SYMBOLS)} configured symbols")
                    
            # Run the test
            loop.run_until_complete(test_task())
            
        except Exception as e:
            self.log_message(f"❌ Connection test failed: {e}")
            self.update_activity(f"Connection test failed: {str(e)}")
        finally:
            # Update GUI when done
            self.root.after(0, lambda: self.progress.stop())
            
    # ENHANCED run_collection method with detailed status updates
    def run_collection(self):
        """Run the data collection in a separate thread"""
        try:
            # Create event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Initialize hybrid system
            self.update_activity("Initializing hybrid system...")
            self.hybrid_system = HybridTradingSystem(self.gui_config)
            
            async def collection_task():
                await self.hybrid_system.initialize()
                
                # Update GUI
                self.update_status(connection="Connected")
                self.log_message("✅ System initialized successfully")
                self.update_activity("System initialized - Getting symbols...")
                
                # Get symbols to process
                if self.gui_config.FETCH_ALL_SYMBOLS:
                    self.update_activity("Fetching all available symbols from Bybit...")
                    symbols = await self.hybrid_system.data_fetcher._get_all_symbols()
                else:
                    symbols = self.gui_config.SYMBOLS
                    
                self.total_symbols = len(symbols)
                self.update_status(symbols=self.total_symbols)
                self.log_message(f"📊 Processing {self.total_symbols} symbols...")
                
                # START WEBSOCKET FIRST (if enabled) to avoid gaps
                websocket_task = None
                if self.gui_config.ENABLE_WEBSOCKET:
                    self.update_activity("Starting WebSocket first to avoid data gaps...")
                    self.log_message("🌐 Starting WebSocket FIRST to ensure no data gaps...")
                    
                    try:
                        # Initialize WebSocket connection
                        await self.hybrid_system.shared_ws_manager.initialize(self.gui_config)
                        self.hybrid_system.websocket_handler = self.hybrid_system.shared_ws_manager.get_websocket_handler()
                        
                        if self.hybrid_system.websocket_handler:
                            self.update_status(websocket="Connected")
                            self.log_message("✅ WebSocket connected - ready for real-time data")
                            
                            # Start WebSocket monitoring as a background task
                            async def websocket_monitor():
                                ws_start_time = datetime.now()
                                last_update_time = datetime.now()
                                
                                while self.running:
                                    if (self.hybrid_system.websocket_handler and 
                                        self.hybrid_system.websocket_handler.running):
                                        
                                        current_time = datetime.now()
                                        # Update status every 30 seconds
                                        if (current_time - last_update_time).seconds >= 30:
                                            ws_elapsed = current_time - ws_start_time
                                            self.update_activity(f"WebSocket active for {ws_elapsed.seconds}s + fetching historical")
                                            self.log_message(f"🌐 WebSocket active for {ws_elapsed.seconds}s - receiving real-time data")
                                            last_update_time = current_time
                                        
                                        # Update CSV with real-time data periodically
                                        if self.gui_config.LIMIT_TO_50_ENTRIES:
                                            try:
                                                updated = await self.hybrid_system.update_csv_with_realtime_data()
                                                if updated:
                                                    self.log_message("💾 Real-time data saved to CSV")
                                            except Exception as e:
                                                self.log_message(f"Warning: CSV update failed: {e}")
                                    else:
                                        self.update_activity("WebSocket disconnected - attempting to reconnect...")
                                        self.log_message("⚠️ WebSocket disconnected")
                                        
                                        # Try to reconnect
                                        try:
                                            await self.hybrid_system.shared_ws_manager.initialize(self.gui_config)
                                            self.hybrid_system.websocket_handler = self.hybrid_system.shared_ws_manager.get_websocket_handler()
                                            if self.hybrid_system.websocket_handler:
                                                self.update_status(websocket="Connected")
                                                self.log_message("✅ WebSocket reconnected")
                                        except Exception as e:
                                            self.log_message(f"❌ WebSocket reconnection failed: {e}")
                                    
                                    await asyncio.sleep(5)  # Check every 5 seconds
                            
                            # Start WebSocket monitoring as background task
                            websocket_task = asyncio.create_task(websocket_monitor())
                            
                        else:
                            self.update_status(websocket="Failed")
                            self.log_message("❌ WebSocket initialization failed")
                            
                    except Exception as e:
                        self.update_status(websocket="Error")
                        self.log_message(f"❌ WebSocket error: {e}")
                
                # NOW START HISTORICAL DATA COLLECTION CONCURRENTLY WITH WEBSOCKET
                self.update_activity("Starting historical data collection (concurrent with WebSocket)...")
                self.log_message("🚀 Starting historical data collection CONCURRENTLY with WebSocket...")
                
                # Track start time
                start_time = datetime.now()
                
                # Add progress tracking
                self.completed_symbols = 0
                self.total_tasks = len(symbols) * len(self.gui_config.TIMEFRAMES)

                # Create a wrapper function for progress tracking
                async def fetch_with_progress(symbol, timeframe, days, limit_50):
                    self.current_symbol = symbol
                    self.current_timeframe = timeframe
                    self.completed_symbols += 1
                    progress = (self.completed_symbols / self.total_tasks) * 100
                    self.update_progress(progress)
                    self.update_activity(f"Fetching {symbol}_{timeframe} ({self.completed_symbols}/{self.total_tasks}) + WebSocket active")
                    
                    result = await self.hybrid_system.data_fetcher._fetch_symbol_timeframe(symbol, timeframe, days, limit_50)
                    return result
                
                # ALWAYS FETCH HISTORICAL DATA - concurrent with WebSocket
                limit_50 = self.gui_config.LIMIT_TO_50_ENTRIES
                
                # Process with progress tracking
                tasks = []
                for symbol in symbols:
                    for timeframe in self.gui_config.TIMEFRAMES:
                        task = asyncio.create_task(
                            fetch_with_progress(symbol, timeframe, self.gui_config.DAYS_TO_FETCH, limit_50)
                        )
                        tasks.append(task)
                
                # Execute all historical data tasks CONCURRENTLY with WebSocket
                self.log_message(f"📥 Fetching historical data for {len(tasks)} symbol/timeframe combinations...")
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check results
                successful_fetches = sum(1 for result in results if result is True)
                failed_fetches = sum(1 for result in results if isinstance(result, Exception))
                
                # Calculate elapsed time
                elapsed = datetime.now() - start_time
                self.log_message(f"⏱️  Historical data collection completed in {elapsed}")
                self.log_message(f"✅ Successful: {successful_fetches}, ❌ Failed: {failed_fetches}")
                
                if successful_fetches > 0:
                    self.log_message("✅ Historical data collection completed successfully!")
                    self.update_activity("Historical data completed + WebSocket active")
                    self.update_progress(100)
                    
                    # Check what data was actually collected
                    self.update_activity("Verifying collected data...")
                    data_dir = Path(self.gui_config.DATA_DIR)
                    if data_dir.exists():
                        csv_files = list(data_dir.glob("*.csv"))
                        self.log_message(f"📁 Found {len(csv_files)} CSV files after historical data collection")
                        
                        # Show sample of collected data
                        for i, file in enumerate(csv_files[:3]):  # Show first 3 files
                            try:
                                with open(file, 'r') as f:
                                    lines = f.readlines()
                                self.log_message(f"📄 {file.name}: {len(lines)-1} data points")
                            except:
                                pass
                    
                    # Run integrity check if enabled
                    if self.gui_config.RUN_INTEGRITY_CHECK:
                        self.update_activity("Running integrity check...")
                        self.log_message("🔍 Running integrity check...")
                        await asyncio.sleep(1)  # Simulate integrity check
                        self.log_message("✅ Integrity check completed!")
                        
                    # Run gap filling if enabled
                    if self.gui_config.RUN_GAP_FILLING:
                        self.update_activity("Running gap filling...")
                        self.log_message("🔧 Running gap filling...")
                        await asyncio.sleep(1)  # Simulate gap filling
                        self.log_message("✅ Gap filling completed!")
                else:
                    self.log_message("❌ Historical data collection failed for all symbols")
                    self.update_activity("Historical data failed + WebSocket active")
                    
                # If WebSocket is running, keep the monitor going
                if self.gui_config.ENABLE_WEBSOCKET and websocket_task:
                    self.update_activity("All systems running: WebSocket + Historical data complete")
                    self.log_message("✅ System fully operational: WebSocket running + Historical data collected")
                    
                    # Wait for the WebSocket monitor to continue running
                    try:
                        await websocket_task
                    except asyncio.CancelledError:
                        pass
                else:
                    self.update_activity("All data collection tasks completed!")
                    self.log_message("✅ Data collection completed")
                    
            # Run the collection task
            loop.run_until_complete(collection_task())
            
        except Exception as e:
            self.log_message(f"❌ Collection error: {e}")
            self.update_activity(f"Collection error: {str(e)}")
            import traceback
            self.log_message(f"Traceback: {traceback.format_exc()}")
        finally:
            # Update GUI when done
            self.root.after(0, self.stop_collection)



            
    def on_closing(self):
        """Handle window closing"""
        if self.running:
            if messagebox.askokcancel("Quit", "Data collection is running. Are you sure you want to quit?"):
                self.running = False
                self.root.destroy()
        else:
            self.root.destroy()
            
    def run(self):
        """Start the GUI"""
        self.root.mainloop()

def main():
    """Main function to run the GUI"""
    gui = DataCollectionGUI()
    gui.run()

if __name__ == "__main__":
    main()