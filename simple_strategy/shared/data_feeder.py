# data_feeder.py - Data loading and management for backtesting
import os
import csv
import psutil
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataFeeder:
    """
    Data Feeder component for loading and managing CSV data from the data collection system.
    Supports multiple symbols, timeframes, and memory management.
    """
    
    def __init__(self, data_dir: str = 'data', memory_limit_percent: float = 90):
        """
        Initialize data feeder with memory management.
        
        Args:
            data_dir: Directory containing CSV data files
            memory_limit_percent: Maximum memory usage percentage (0-100)
        """
        self.data_dir = Path(data_dir)
        self.memory_limit_percent = memory_limit_percent
        self.data_cache: Dict[str, Dict[str, pd.DataFrame]] = {}  # {symbol: {timeframe: DataFrame}}
        self.metadata_cache: Dict[str, Dict[str, Dict]] = {}  # {symbol: {timeframe: metadata}}
        
        logger.info(f"DataFeeder initialized with data_dir={data_dir}, memory_limit={memory_limit_percent}%")
    
    def _check_memory_usage(self) -> bool:
        """
        Check if current memory usage is within limits.
        
        Returns:
            bool: True if memory usage is acceptable, False otherwise
        """
        try:
            memory = psutil.virtual_memory()
            current_usage_percent = memory.percent
            return current_usage_percent <= self.memory_limit_percent
        except Exception as e:
            logger.warning(f"Could not check memory usage: {e}")
            return True  # Assume it's OK if we can't check
    
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
        # Use only one naming convention: BTCUSDT_1.csv (without 'm')
        possible_filenames = [
            f"{symbol}_{timeframe.rstrip('m')}.csv"  # Without 'm' suffix (e.g., BTCUSDT_5.csv)
        ]
        
        df = None
        loaded_file = None
        
        for filename in possible_filenames:
            file_path = self.data_dir / filename
            
            if file_path.exists():
                try:
                    # Read CSV file
                    df = pd.read_csv(file_path)
                    
                    # Convert timestamp to datetime for easier handling
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df.set_index('datetime', inplace=True)
                    
                    # Sort by timestamp to ensure chronological order
                    df.sort_index(inplace=True)
                    
                    loaded_file = filename
                    logger.info(f"Loaded {len(df)} rows from {file_path}")
                    break
                    
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
                    df = None
        
        if df is None:
            logger.warning(f"Could not find data file for {symbol}_{timeframe}. Tried: {possible_filenames}")
        
        return df
    
    def get_data_for_symbols(self, symbols, timeframes, start_date, end_date):
        """Return cached data for multiple symbols/timeframes, filtered by date range"""
        print(f"🔧 DEBUG: get_data_for_symbols called with symbols={symbols}, timeframes={timeframes}")
        print(f"🔧 DEBUG: Date range: {start_date} to {end_date}")
        
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        result = {}
        for symbol in symbols:
            result[symbol] = {}
            for timeframe in timeframes:
                print(f"🔧 DEBUG: Processing {symbol} {timeframe}")
                
                # Check if data is in cache
                if symbol in self.data_cache and timeframe in self.data_cache[symbol]:
                    print(f"🔧 DEBUG: Found data in cache for {symbol} {timeframe}")
                    df = self.data_cache[symbol][timeframe].copy()
                    print(f"🔧 DEBUG: Original data shape: {df.shape}")
                    print(f"🔧 DEBUG: Original data date range: {df.index.min()} to {df.index.max()}")
                    
                    # Filter by date range
                    mask = (df.index >= start_date) & (df.index <= end_date)
                    filtered_df = df[mask]
                    print(f"🔧 DEBUG: Filtered data shape: {filtered_df.shape}")
                    
                    result[symbol][timeframe] = filtered_df
                else:
                    print(f"🔧 DEBUG: No data in cache for {symbol} {timeframe}")
                    # Try to load data directly if not in cache
                    df = self._load_csv_file(symbol, timeframe)
                    if df is not None:
                        print(f"🔧 DEBUG: Loaded data from file for {symbol} {timeframe}")
                        # Filter by date range
                        mask = (df.index >= start_date) & (df.index <= end_date)
                        filtered_df = df[mask]
                        print(f"🔧 DEBUG: Filtered data shape: {filtered_df.shape}")
                        
                        # Store in cache for future use
                        if symbol not in self.data_cache:
                            self.data_cache[symbol] = {}
                        self.data_cache[symbol][timeframe] = df
                        
                        result[symbol][timeframe] = filtered_df
                    else:
                        print(f"🔧 DEBUG: Could not load data for {symbol} {timeframe}")
                        # Return empty DataFrame with expected columns
                        result[symbol][timeframe] = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        print(f"🔧 DEBUG: Returning data with keys: {list(result.keys())}")
        return result
    
    def load_data(self, symbols: List[str], timeframes: List[str], 
                  start_date: Optional[Union[str, datetime]] = None, 
                  end_date: Optional[Union[str, datetime]] = None) -> bool:
        """
        Load data for specified symbols and timeframes.
        
        Args:
            symbols: List of trading symbols
            timeframes: List of timeframes
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        logger.info(f"Loading data for symbols: {symbols}, timeframes: {timeframes}")
        
        # Check memory usage before loading
        if not self._check_memory_usage():
            logger.error(f"Memory usage exceeds limit of {self.memory_limit_percent}%")
            return False
        
        # Convert date strings to datetime objects
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
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
                    logger.warning(f"No data found for {symbol}_{timeframe} with specified date range")
                    continue
                
                # Store in cache
                self.data_cache[symbol][timeframe] = df
                
                # Store metadata
                self.metadata_cache[symbol][timeframe] = {
                    'start_date': df.index.min(),
                    'end_date': df.index.max(),
                    'row_count': len(df),
                    'file_path': str(self.data_dir / f"{symbol}_{timeframe}.csv")
                }
                
                success_count += 1
                logger.info(f"Successfully loaded {symbol}_{timeframe}: {len(df)} rows")
        
        # Check memory usage after loading
        if not self._check_memory_usage():
            logger.warning(f"Memory usage after loading exceeds limit of {self.memory_limit_percent}%")
        
        logger.info(f"Data loading complete: {success_count}/{len(symbols) * len(timeframes)} files loaded successfully")
        return success_count > 0
    
    def get_data_at_timestamp(self, symbol: str, timeframe: str, timestamp: Union[int, datetime, str]) -> Optional[Dict[str, Any]]:
        """
        Get data for a specific timestamp.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            timestamp: Timestamp (can be int milliseconds, datetime, or string)
            
        Returns:
            Dictionary with OHLCV data or None if not found
        """
        # Convert timestamp to datetime
        if isinstance(timestamp, int):
            # Convert milliseconds to datetime
            dt = pd.to_datetime(timestamp, unit='ms')
        elif isinstance(timestamp, str):
            dt = pd.to_datetime(timestamp)
        else:
            dt = timestamp
        
        # Check if data is loaded
        if symbol not in self.data_cache or timeframe not in self.data_cache[symbol]:
            logger.warning(f"Data not loaded for {symbol}_{timeframe}")
            return None
        
        df = self.data_cache[symbol][timeframe]
        
        # Find the closest timestamp (exact match or previous)
        try:
            # Get the row at or before the requested timestamp
            mask = df.index <= dt
            if not mask.any():
                return None
            
            # Get the most recent data point at or before the timestamp
            result_df = df[mask].tail(1)
            
            if len(result_df) == 0:
                return None
            
            # Convert to dictionary
            result = result_df.iloc[0].to_dict()
            
            # Add timestamp in milliseconds
            result['timestamp_ms'] = int(result_df.index[0].timestamp() * 1000)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting data at timestamp {dt} for {symbol}_{timeframe}: {e}")
            return None
    
    def get_latest_data(self, symbol: str, timeframe: str, lookback_periods: int = 1) -> Optional[List[Dict[str, Any]]]:
        """
        Get latest available data for timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            lookback_periods: Number of periods to return (default: 1)
            
        Returns:
            List of dictionaries with OHLCV data or None if not found
        """
        # Check if data is loaded
        if symbol not in self.data_cache or timeframe not in self.data_cache[symbol]:
            logger.warning(f"Data not loaded for {symbol}_{timeframe}")
            return None
        
        df = self.data_cache[symbol][timeframe]
        
        try:
            # Get the latest N rows
            result_df = df.tail(lookback_periods)
            
            if len(result_df) == 0:
                return None
            
            # Convert to list of dictionaries
            results = []
            for idx, row in result_df.iterrows():
                result = row.to_dict()
                result['timestamp_ms'] = int(idx.timestamp() * 1000)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting latest data for {symbol}_{timeframe}: {e}")
            return None
    
    def get_multi_timeframe_data(self, symbol: str, timeframes: List[str], timestamp: Union[int, datetime, str]) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Get aligned data across multiple timeframes.
        
        Args:
            symbol: Trading symbol
            timeframes: List of timeframes to retrieve
            timestamp: Reference timestamp for alignment
            
        Returns:
            Dictionary with timeframe as key and OHLCV data as value
        """
        results = {}
        
        for timeframe in timeframes:
            data = self.get_data_at_timestamp(symbol, timeframe, timestamp)
            if data is not None:
                results[timeframe] = data
        
        return results if results else None
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get current memory usage statistics.
        
        Returns:
            Dictionary with memory usage information
        """
        try:
            memory = psutil.virtual_memory()
            process = psutil.Process()
            
            # Calculate cache size
            cache_size = 0
            for symbol in self.data_cache:
                for timeframe in self.data_cache[symbol]:
                    cache_size += self.data_cache[symbol][timeframe].memory_usage(deep=True).sum()
            
            return {
                'system_memory_percent': memory.percent,
                'system_memory_available_gb': memory.available / (1024**3),
                'system_memory_total_gb': memory.total / (1024**3),
                'process_memory_mb': process.memory_info().rss / (1024**2),
                'cache_size_mb': cache_size / (1024**2),
                'configured_limit_percent': self.memory_limit_percent,
                'loaded_symbols': list(self.data_cache.keys()),
                'loaded_timeframes': {symbol: list(tfs.keys()) for symbol, tfs in self.data_cache.items()},
                'total_files_loaded': sum(len(tfs) for tfs in self.data_cache.values())
            }
            
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {'error': str(e)}
    
    def clear_cache(self, symbol: Optional[str] = None, timeframe: Optional[str] = None):
        """
        Clear data cache to free memory.
        
        Args:
            symbol: Specific symbol to clear (None for all)
            timeframe: Specific timeframe to clear (None for all)
        """
        if symbol is None:
            # Clear all cache
            self.data_cache.clear()
            self.metadata_cache.clear()
            logger.info("Cleared all data cache")
        else:
            if symbol in self.data_cache:
                if timeframe is None:
                    # Clear all timeframes for symbol
                    del self.data_cache[symbol]
                    del self.metadata_cache[symbol]
                    logger.info(f"Cleared all data cache for symbol: {symbol}")
                else:
                    # Clear specific timeframe for symbol
                    if timeframe in self.data_cache[symbol]:
                        del self.data_cache[symbol][timeframe]
                        del self.metadata_cache[symbol][timeframe]
                        logger.info(f"Cleared data cache for {symbol}_{timeframe}")
    
    def get_data_info(self, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about loaded data.
        
        Args:
            symbol: Specific symbol (None for all)
            timeframe: Specific timeframe (None for all)
            
        Returns:
            Dictionary with data information
        """
        if symbol is None:
            # Return info for all data
            return self.metadata_cache.copy()
        else:
            if symbol in self.metadata_cache:
                if timeframe is None:
                    # Return all timeframes for symbol
                    return self.metadata_cache[symbol].copy()
                else:
                    # Return specific timeframe
                    return self.metadata_cache[symbol].get(timeframe, {})
            else:
                return {}