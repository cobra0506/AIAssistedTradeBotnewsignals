import os
import json
from typing import Dict, List, Any
from .config import DataCollectionConfig
from .csv_manager import CSVManager

class SharedDataAccess:
    """Provides access to data collected by the separate data collection process"""
    
    def __init__(self):
        self.config = DataCollectionConfig()
        self.csv_manager = CSVManager(self.config)
        
    def get_latest_data(self, symbol: str, timeframe: str, limit: int = 50) -> List[Dict]:
        """Get the latest data from CSV files"""
        try:
            filename = f"{self.config.DATA_DIR}/{symbol}_{timeframe}.csv"
            if os.path.exists(filename):
                return self.csv_manager.load_csv(filename)[-limit:]
            return []
        except Exception as e:
            print(f"Error getting latest data: {e}")
            return []
    
    def get_all_symbols(self) -> List[str]:
        """Get list of all symbols with data"""
        try:
            if not os.path.exists(self.config.DATA_DIR):
                return []
            
            symbols = set()
            for filename in os.listdir(self.config.DATA_DIR):
                if filename.endswith('.csv'):
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        symbols.add(parts[0])
            return list(symbols)
        except Exception as e:
            print(f"Error getting all symbols: {e}")
            return []
    
    def is_data_collection_running(self) -> bool:
        """Check if data collection process is running"""
        try:
            # Check for a status file or process ID file
            status_file = os.path.join(self.config.DATA_DIR, "collection_status.json")
            if os.path.exists(status_file):
                with open(status_file, 'r') as f:
                    status = json.load(f)
                    return status.get('running', False)
            return False
        except Exception as e:
            print(f"Error checking data collection status: {e}")
            return False