# csv_manager.py - Centralized CSV operations with chronological ordering
import csv
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path 
from .config import DataCollectionConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSVManager:
    def __init__(self, config: DataCollectionConfig):
        self.config = config
        self.data_dir = Path(config.DATA_DIR)
        
    def ensure_data_directory(self):
        """Ensure data directory exists"""
        os.makedirs(self.data_dir, exist_ok=True)
        
    def read_csv_data(self, symbol: str, timeframe: str) -> List[Dict[str, Any]]:
        """
        Read CSV data and return in chronological order (oldest first)
        Returns empty list if file doesn't exist
        """
        filename = os.path.join(self.data_dir, f"{symbol}_{timeframe}.csv")
        
        if not os.path.exists(filename):
            return []
            
        try:
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
                
            # Sort by timestamp (chronological order - oldest first)
            data.sort(key=lambda x: x['timestamp'])
            
            logger.info(f"Read {len(data)} entries from {filename}")
            return data
            
        except Exception as e:
            logger.error(f"Error reading {filename}: {e}")
            return []
    
    def write_csv_data(self, symbol: str, timeframe: str, data: List[Dict[str, Any]]) -> bool:
        """
        Write data to CSV in chronological order (oldest first)
        Handles 50-entry limit if configured
        """
        if not data:
            logger.warning(f"No data to write for {symbol}_{timeframe}")
            return False
            
        filename = os.path.join(self.data_dir, f"{symbol}_{timeframe}.csv")
        self.ensure_data_directory()
        
        try:
            # Sort data chronologically (oldest first)
            data.sort(key=lambda x: x['timestamp'])
            
            # Apply entry limit if configured
            # New
            if self.config.LIMIT_TO_50_ENTRIES and len(data) > self.config.MIN_CANDLES:
                data = data[-self.config.MIN_CANDLES:]  # Keep most recent MIN_CANDLES entries
                logger.info(f"Limited {symbol}_{timeframe} to {self.config.MIN_CANDLES} most recent entries")

            
            # Ensure all required fields are present
            fieldnames = ['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume']
            for row in data:
                # Generate datetime if not present
                if 'datetime' not in row:
                    dt = datetime.fromtimestamp(row['timestamp'] / 1000)
                    row['datetime'] = dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Ensure all fields exist
                for field in fieldnames:
                    if field not in row:
                        row[field] = 0 if field != 'datetime' else ''
            
            # Write to CSV
            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            
            logger.info(f"Wrote {len(data)} entries to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing {filename}: {e}")
            return False
    
    def append_new_data(self, symbol: str, timeframe: str, new_candles: List[Dict[str, Any]]) -> bool:
        """
        Append new candles to existing CSV data while maintaining chronological order
        Removes duplicates and handles 50-entry limit
        """
        if not new_candles:
            return False
            
        filename = os.path.join(self.data_dir, f"{symbol}_{timeframe}.csv")
        
        try:
            # Read existing data
            existing_data = self.read_csv_data(symbol, timeframe)
            
            # Get existing timestamps for duplicate checking
            existing_timestamps = {row['timestamp'] for row in existing_data}
            
            # Filter out duplicates and add datetime field
            unique_new_candles = []
            for candle in new_candles:
                if candle['timestamp'] not in existing_timestamps:
                    # Ensure datetime field exists
                    if 'datetime' not in candle:
                        dt = datetime.fromtimestamp(candle['timestamp'] / 1000)
                        candle['datetime'] = dt.strftime('%Y-%m-%d %H:%M:%S')
                    unique_new_candles.append(candle)
            
            if not unique_new_candles:
                #logger.info(f"No new unique candles for {symbol}_{timeframe}")
                return True
            
            # Combine existing and new data
            combined_data = existing_data + unique_new_candles
            
            # Write back to CSV (will handle sorting and 50-entry limit)
            success = self.write_csv_data(symbol, timeframe, combined_data)
            
            if success:
                logger.info(f"Appended {len(unique_new_candles)} new candles to {filename}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error appending data to {filename}: {e}")
            return False
    
    def get_latest_timestamp(self, symbol: str, timeframe: str) -> int:
        """Get the latest timestamp from CSV file, returns 0 if file doesn't exist"""
        data = self.read_csv_data(symbol, timeframe)
        if data:
            return data[-1]['timestamp']  # Last entry is newest due to chronological sorting
        return 0
    
    def update_candle(self, symbol: str, timeframe: str, candle_data: Dict):
        """Update or append a candle to CSV file"""
        try:
            filename = os.path.join(self.data_dir, f"{symbol}_{timeframe}.csv")
            
            # Read existing data
            data = []
            if os.path.exists(filename):
                with open(filename, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    data = list(reader)
            
            # Convert new candle timestamp to int for comparison
            new_timestamp = int(candle_data['timestamp'])
            
            # Check if candle with same timestamp exists
            updated = False
            for i, row in enumerate(data):
                if int(row['timestamp']) == new_timestamp:
                    # Update existing candle
                    data[i] = candle_data
                    updated = True
                    logger.info(f"[CSV] Updated existing candle for {symbol}_{timeframe} at timestamp {new_timestamp}")
                    break
            
            # If not found, append new candle
            if not updated:
                data.append(candle_data)
                logger.info(f"[CSV] Added new candle for {symbol}_{timeframe} at timestamp {new_timestamp}")
            
            # Sort by timestamp
            data.sort(key=lambda x: int(x['timestamp']))
            
            # Write back to file
            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume'])
                writer.writeheader()
                writer.writerows(data)
            
            return True
        except Exception as e:
            logger.error(f"Error updating candle in CSV: {e}")
            return False