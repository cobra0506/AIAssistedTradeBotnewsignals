"""
Ultra Simple Test Strategy - Direct signal return for testing with better debugging
"""
import sys
import os
import logging
from typing import Dict, List, Any

# Add parent directories to path for proper imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Configure logging
logger = logging.getLogger(__name__)

# STRATEGY_PARAMETERS - GUI Configuration (AT TOP)
STRATEGY_PARAMETERS = {
    'test_mode': {
        'type': 'bool',
        'default': True,
        'description': 'Enable test mode (always generates OPEN_LONG)',
        'gui_hint': 'When enabled, always generates OPEN_LONG signal for testing'
    }
}

def create_strategy(symbols=None, timeframes=None, **params):
    """
    Create Ultra Simple Test Strategy
    Always generates OPEN_LONG for testing purposes.
    """
    # DEBUG: Log what we receive
    logger.info(f"🔧 create_strategy called with:")
    logger.info(f"  - symbols: {symbols}")
    logger.info(f"  - timeframes: {timeframes}")
    logger.info(f"  - params: {params}")
    
    # Handle None/empty values with defaults
    if symbols is None or len(symbols) == 0:
        logger.warning("⚠️ No symbols provided, using default: ['BTCUSDT']")
        symbols = ['BTCUSDT']
    
    if timeframes is None or len(timeframes) == 0:
        logger.warning("⚠️ No timeframes provided, using default: ['1m']")
        timeframes = ['1m']
    
    # Get parameters
    test_mode = params.get('test_mode', True)
    
    logger.info(f"🎯 Creating Ultra Simple Test strategy:")
    logger.info(f"  - Symbols: {symbols}")
    logger.info(f"  - Timeframes: {timeframes}")
    logger.info(f"  - Test Mode: {test_mode}")
    
    try:
        # Create a simple strategy class that bypasses all complex logic
        class UltraSimpleTestStrategy:
            def __init__(self, symbols, timeframes, **params):
                self.name = "Ultra_Simple_Test"
                self.symbols = symbols
                self.timeframes = timeframes
                self.params = params
                
                # Create a simple CSV log file directly
                self._init_csv_logger()
            
            def _init_csv_logger(self):
                """Initialize a simple CSV logger"""
                try:
                    import csv
                    from datetime import datetime
                    
                    # Create logs directory if it doesn't exist
                    # Use a more direct path approach
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
                    log_dir = os.path.join(project_root, "logs")
                    os.makedirs(log_dir, exist_ok=True)
                    
                    # Set the CSV file path with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.csv_file = os.path.join(log_dir, f'signal_debug_{timestamp}.csv')
                    
                    # Initialize the CSV file with headers
                    with open(self.csv_file, 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(['timestamp', 'symbol', 'signal', 'notes'])
                    
                    print(f"✅ CSV logger initialized: {self.csv_file}")
                except Exception as e:
                    print(f"❌ Error initializing CSV logger: {e}")
                    import traceback
                    traceback.print_exc()
                    self.csv_file = None
            
            def generate_signals(self, data):
                """Generate trading signals - always OPEN_LONG for testing"""
                try:
                    print(f"🔍 DEBUG: generate_signals called with data keys: {list(data.keys()) if data else 'None'}")
                    
                    # Log to CSV
                    if self.csv_file:
                        import csv
                        from datetime import datetime
                        
                        with open(self.csv_file, 'a', newline='') as file:
                            writer = csv.writer(file)
                            for symbol in self.symbols:
                                writer.writerow([
                                    datetime.now().isoformat(), 
                                    symbol, 
                                    "OPEN_LONG", 
                                    "Test signal"
                                ])
                    
                    # Return a simple signal structure
                    results = {}
                    for symbol in self.symbols:
                        results[symbol] = {}
                        for timeframe in self.timeframes:
                            results[symbol][timeframe] = "OPEN_LONG"
                    
                    print(f"🔍 DEBUG: Returning signals for {len(results)} symbols")
                    return results
                except Exception as e:
                    print(f"❌ Error in generate_signals: {e}")
                    import traceback
                    traceback.print_exc()
                    return {}
        
        # Create and return the strategy
        strategy = UltraSimpleTestStrategy(symbols, timeframes, **params)
        
        logger.info(f"✅ Ultra Simple Test strategy created successfully!")
        logger.info(f"  - Strategy Name: {strategy.name}")
        logger.info(f"  - Strategy Symbols: {strategy.symbols}")
        logger.info(f"  - Strategy Timeframes: {strategy.timeframes}")
        
        return strategy
        
    except Exception as e:
        logger.error(f"❌ Error creating Ultra Simple Test strategy: {e}")
        import traceback
        traceback.print_exc()
        raise

def simple_test():
    """Simple test to verify the strategy works"""
    try:
        # Test strategy creation
        strategy = create_strategy(
            symbols=['BTCUSDT'],
            timeframes=['1m'],
            test_mode=True
        )
        
        print(f"✅ Ultra Simple Test strategy created successfully: {strategy.name}")
        print(f"  - Symbols: {strategy.symbols}")
        print(f"  - Timeframes: {strategy.timeframes}")
        return True
    except Exception as e:
        print(f"❌ Error testing Ultra Simple Test strategy: {e}")
        return False

# For testing
if __name__ == "__main__":
    simple_test()