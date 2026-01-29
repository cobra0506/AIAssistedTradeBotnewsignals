"""
CSV Debug Logger for Trading Signals
Writes debug information to a CSV file instead of the console
"""
import os
import csv
import pandas as pd
from datetime import datetime

class CSVDebugLogger:
    def __init__(self, log_dir="logs"):
        """Initialize the CSV debug logger"""
        # Create logs directory if it doesn't exist
        # Fix the path calculation - go up 4 levels from the trading module
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.log_dir = os.path.join(project_root, log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set the CSV file path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = os.path.join(self.log_dir, f'signal_debug_{timestamp}.csv')
        
        # Initialize the CSV file with headers
        self._init_csv()
        
        # Cache for signals to avoid duplicate entries
        self.signal_cache = {}
        
        print(f"✅ CSV debug logger initialized: {self.csv_file}")
    
    def _init_csv(self):
        """Initialize the CSV file with headers"""
        headers = [
            'timestamp', 'symbol', 'signal_type', 'signal_value',
            'rsi_value', 'rsi_signal', 'ema_fast_value', 'ema_slow_value', 'trend_signal',
            'combined_signal', 'notes'
        ]
        
        with open(self.csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
    
    def log_signal(self, symbol, signal_type, signal_value, rsi_value=None, rsi_signal=None,
                   ema_fast_value=None, ema_slow_value=None, trend_signal=None,
                   combined_signal=None, notes=None):
        """Log a signal to the CSV file"""
        timestamp = datetime.now().isoformat()
        
        # Create a unique key for this symbol to avoid duplicates
        cache_key = f"{symbol}_{signal_type}_{signal_value}"
        
        # Check if we've already logged this signal recently
        if cache_key in self.signal_cache:
            # Update the timestamp to keep it recent
            self.signal_cache[cache_key] = timestamp
            return
        
        # Add to cache
        self.signal_cache[cache_key] = timestamp
        
        # Write to CSV
        with open(self.csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp, symbol, signal_type, signal_value,
                rsi_value, rsi_signal, ema_fast_value, ema_slow_value, trend_signal,
                combined_signal, notes
            ])