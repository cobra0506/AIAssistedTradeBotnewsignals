import os
import csv
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from .config import DataCollectionConfig

class DataIntegrityChecker:
    def __init__(self, config: DataCollectionConfig):
        self.config = config
        # Use logs directory instead of data/integrity_reports
        self.reports_dir = os.path.join('logs', 'integrity_reports')
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def check_all_files(self) -> Dict[str, Any]:
        """Check integrity of all data files"""
        print("Starting data integrity check...")
        
        results = {
            'files_checked': 0,
            'files_with_issues': 0,
            'total_gaps': 0,
            'total_duplicates': 0,
            'total_invalid_candles': 0,
            'issues': {}
        }
        
        # Get all CSV files
        csv_files = []
        for filename in os.listdir(self.config.DATA_DIR):
            if filename.endswith('.csv') and not filename.startswith('integrity_'):
                csv_files.append(filename)
        
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
    
    def check_single_file(self, filename: str) -> Dict[str, Any]:
        """Check integrity of a single data file"""
        filepath = os.path.join(self.config.DATA_DIR, filename)
        
        issues = {
            'filename': filename,
            'has_issues': False,
            'gaps': [],
            'duplicate_count': 0,
            'invalid_candles': 0,
            'total_candles': 0,
            'first_timestamp': None,
            'last_timestamp': None
        }
        
        try:
            # Parse symbol and timeframe from filename
            parts = filename.replace('.csv', '').split('_')
            if len(parts) >= 2:
                symbol = parts[0]
                timeframe = parts[1]
            else:
                symbol = 'unknown'
                timeframe = 'unknown'
            
            # Read CSV file
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                data = list(reader)
            
            issues['total_candles'] = len(data)
            
            if not data:
                issues['has_issues'] = True
                return issues
            
            # Check for invalid candles
            valid_data = []
            for i, row in enumerate(data):
                if self._validate_candle(row):
                    valid_data.append(row)
                else:
                    issues['invalid_candles'] += 1
                    issues['has_issues'] = True
            
            # Check for duplicates
            seen_timestamps = set()
            unique_data = []
            
            for candle in valid_data:
                timestamp = candle['timestamp']
                if timestamp in seen_timestamps:
                    issues['duplicate_count'] += 1
                    issues['has_issues'] = True
                else:
                    seen_timestamps.add(timestamp)
                    unique_data.append(candle)
            
            # Sort by timestamp
            unique_data.sort(key=lambda x: self._parse_timestamp_to_datetime(x))
            
            if unique_data:
                issues['first_timestamp'] = unique_data[0]['timestamp']
                issues['last_timestamp'] = unique_data[-1]['timestamp']
                
                # Check for gaps
                gaps = self._detect_gaps(unique_data, timeframe)
                issues['gaps'] = gaps
                
                if gaps:
                    issues['has_issues'] = True
            
        except Exception as e:
            issues['has_issues'] = True
            issues['error'] = str(e)
        
        return issues
    
    def _parse_timestamp_to_datetime(self, candle_or_timestamp: Any) -> datetime:
        """
        Parse timestamp values from multiple supported formats.

        Supported:
        - unix milliseconds (int or numeric string)
        - unix seconds (int or numeric string)
        - ISO datetime string
        """
        raw_value = candle_or_timestamp
        if isinstance(candle_or_timestamp, dict):
            raw_value = candle_or_timestamp.get('timestamp')

        if raw_value is None:
            raise ValueError("Missing timestamp")

        if isinstance(raw_value, (int, float)):
            value = int(raw_value)
            if value > 10**12:
                return datetime.fromtimestamp(value / 1000)
            return datetime.fromtimestamp(value)

        raw_text = str(raw_value).strip()
        if not raw_text:
            raise ValueError("Empty timestamp")

        if raw_text.isdigit():
            value = int(raw_text)
            if value > 10**12:
                return datetime.fromtimestamp(value / 1000)
            return datetime.fromtimestamp(value)

        # Normalize UTC "Z" suffix to fromisoformat-compatible form.
        if raw_text.endswith('Z'):
            raw_text = raw_text[:-1] + '+00:00'

        try:
            return datetime.fromisoformat(raw_text)
        except ValueError:
            # Fallback for common CSV format "YYYY-MM-DD HH:MM:SS"
            return datetime.strptime(raw_text, '%Y-%m-%d %H:%M:%S')

    def _build_filled_candle(self, fieldnames: List[str], base_close: Any, filled_dt: datetime) -> Dict[str, Any]:
        """Build a synthetic candle that matches the file's existing schema."""
        close_value = str(base_close)
        candle = {
            'open': close_value,
            'high': close_value,
            'low': close_value,
            'close': close_value,
            'volume': '0'
        }

        if 'timestamp' in fieldnames:
            if 'datetime' in fieldnames:
                timestamp_ms = int(filled_dt.timestamp() * 1000)
                candle['timestamp'] = str(timestamp_ms)
                candle['datetime'] = filled_dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                candle['timestamp'] = filled_dt.isoformat()

        if 'turnover' in fieldnames:
            candle['turnover'] = '0'

        # Preserve any extra custom columns in existing files.
        for field in fieldnames:
            if field not in candle:
                candle[field] = ''

        return candle

    def _validate_candle(self, candle: Dict[str, Any]) -> bool:
        """Validate a single candle"""
        try:
            # Current collector schema does not always include turnover.
            required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(field in candle for field in required_fields):
                return False
            
            # Validate timestamp format (supports ms/seconds/ISO).
            self._parse_timestamp_to_datetime(candle['timestamp'])
            
            # Validate price values
            open_price = float(candle['open'])
            high_price = float(candle['high'])
            low_price = float(candle['low'])
            close_price = float(candle['close'])
            volume = float(candle['volume'])
            turnover = float(candle.get('turnover', 0.0))
            
            # Validate price relationships
            if not (low_price <= high_price and 
                    low_price <= open_price <= high_price and 
                    low_price <= close_price <= high_price):
                return False
            
            # Validate positive values
            if open_price <= 0 or high_price <= 0 or low_price <= 0 or close_price <= 0 or volume < 0 or turnover < 0:
                return False
            
            return True
            
        except (ValueError, TypeError):
            return False
    
    def _detect_gaps(self, data: List[Dict[str, Any]], timeframe: str) -> List[Dict[str, Any]]:
        """Detect gaps in time series data"""
        gaps = []
        
        if len(data) < 2:
            return gaps
        
        # Convert timeframe to minutes
        timeframe_minutes = {
            '1': 1,
            '5': 5,
            '15': 15,
            '60': 60,
            '120': 120,
            '240': 240,
            '1440': 1440
        }.get(timeframe, 1)
        
        expected_interval = timedelta(minutes=timeframe_minutes)
        
        for i in range(1, len(data)):
            prev_timestamp = self._parse_timestamp_to_datetime(data[i-1]['timestamp'])
            curr_timestamp = self._parse_timestamp_to_datetime(data[i]['timestamp'])
            
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
    
    def fix_all_duplicates(self):
        """Fix duplicates in all data files"""
        print("Fixing duplicates in all files...")
        
        fixed_files = 0
        for filename in os.listdir(self.config.DATA_DIR):
            if filename.endswith('.csv') and not filename.startswith('integrity_'):
                if self.fix_duplicates(filename):
                    fixed_files += 1
        
        print(f"Fixed duplicates in {fixed_files} files")
    
    def fix_duplicates(self, filename: str) -> bool:
        """Remove duplicate entries from a file"""
        filepath = os.path.join(self.config.DATA_DIR, filename)
        
        try:
            # Read data
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                data = list(reader)
            
            # Remove duplicates
            seen_timestamps = set()
            unique_data = []
            duplicates_removed = 0
            
            for candle in data:
                timestamp = candle['timestamp']
                if timestamp in seen_timestamps:
                    duplicates_removed += 1
                else:
                    seen_timestamps.add(timestamp)
                    unique_data.append(candle)
            
            if duplicates_removed > 0:
                # Write back unique data
                with open(filepath, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=unique_data[0].keys())
                    writer.writeheader()
                    writer.writerows(unique_data)
                
                print(f"Fixed {filename}: Removed {duplicates_removed} duplicates")
                return True
            else:
                print(f"No duplicates found in {filename}")
                return False
                
        except Exception as e:
            print(f"Error fixing duplicates in {filename}: {e}")
            return False
        
    def fill_gaps_in_file(self, filename: str) -> bool:
        """Fill gaps in a data file with previous candle data"""
        filepath = os.path.join(self.config.DATA_DIR, filename)
        
        try:
            # Read existing data
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                data = list(reader)
                fieldnames = reader.fieldnames  # Get existing fieldnames
            
            if not data:
                return False
            
            # Parse symbol and timeframe
            parts = filename.replace('.csv', '').split('_')
            if len(parts) >= 2:
                timeframe = parts[1]
            else:
                timeframe = '1'
            
            # Convert timeframe to minutes
            timeframe_minutes = {
                '1': 1, '5': 5, '15': 15, '60': 60, '120': 120, '240': 240, '1440': 1440
            }.get(timeframe, 1)
            
            expected_interval = timedelta(minutes=timeframe_minutes)
            
            # Sort by timestamp
            data.sort(key=lambda x: self._parse_timestamp_to_datetime(x))
            
            # Find and fill gaps
            filled_data = []
            gaps_filled = 0
            
            for i in range(len(data)):
                current_candle = data[i]
                filled_data.append(current_candle)
                
                # Check if there's a gap to next candle
                if i < len(data) - 1:
                    current_timestamp = self._parse_timestamp_to_datetime(current_candle['timestamp'])
                    next_timestamp = self._parse_timestamp_to_datetime(data[i+1]['timestamp'])
                    
                    while next_timestamp - current_timestamp > expected_interval + timedelta(seconds=1):
                        # Create filled candle
                        filled_timestamp = current_timestamp + expected_interval
                        filled_candle = self._build_filled_candle(
                            fieldnames or [],
                            current_candle.get('close', '0'),
                            filled_timestamp
                        )
                        
                        filled_data.append(filled_candle)
                        gaps_filled += 1
                        
                        # Update current timestamp
                        current_timestamp = filled_timestamp
            
            if gaps_filled > 0:
                # Write filled data back to file (using original fieldnames)
                with open(filepath, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(filled_data)
                
                print(f"Filled {gaps_filled} gaps in {filename}")
                return True
            else:
                print(f"No gaps to fill in {filename}")
                return False
                
        except Exception as e:
            print(f"Error filling gaps in {filename}: {e}")
            return False

    def fill_all_gaps(self):
        """Fill gaps in all data files"""
        print("Filling gaps in all files...")
        
        files_filled = 0
        for filename in os.listdir(self.config.DATA_DIR):
            if filename.endswith('.csv') and not filename.startswith('integrity_'):
                if self.fill_gaps_in_file(filename):
                    files_filled += 1
        
        print(f"Filled gaps in {files_filled} files")

    def save_integrity_report(self, results: Dict[str, Any]) -> str:
        """Save integrity check results to a report file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"integrity_report_{timestamp}.txt"
        filepath = os.path.join(self.reports_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write("="*60 + "\n")
            f.write("INTEGRITY CHECK REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            f.write("SUMMARY:\n")
            f.write(f"Files checked: {results['files_checked']}\n")
            f.write(f"Files with issues: {results['files_with_issues']}\n")
            f.write(f"Total gaps: {results['total_gaps']}\n")
            f.write(f"Total duplicates: {results['total_duplicates']}\n")
            f.write(f"Total invalid candles: {results['total_invalid_candles']}\n\n")
            
            if results['files_with_issues'] > 0:
                f.write("DETAILED ISSUES:\n")
                f.write("-"*40 + "\n")
                for filename, issues in results['issues'].items():
                    f.write(f"\nFile: {filename}\n")
                    f.write(f"  Total candles: {issues['total_candles']}\n")
                    f.write(f"  Invalid candles: {issues['invalid_candles']}\n")
                    f.write(f"  Duplicates: {issues['duplicate_count']}\n")
                    f.write(f"  Gaps: {len(issues['gaps'])}\n")
                    if issues['gaps']:
                        f.write("  Gap details:\n")
                        for gap in issues['gaps'][:5]:  # Show first 5 gaps
                            f.write(f"    - {gap}\n")
                        if len(issues['gaps']) > 5:
                            f.write(f"    ... and {len(issues['gaps']) - 5} more gaps\n")
            else:
                f.write("No issues found!\n")
        
        return filepath
