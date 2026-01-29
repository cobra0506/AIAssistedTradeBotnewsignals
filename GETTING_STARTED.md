AI Assisted TradeBot - Getting Started Guide 
📋 Overview 

Welcome to the AI Assisted TradeBot! This is a comprehensive cryptocurrency trading bot system that combines traditional technical analysis with advanced AI approaches.  

📚 Primary Documentation: All detailed documentation has been moved to the NewDocs/ folder. This getting started guide provides a quick overview and points you to the detailed documentation. 
🎯 Current System Status 
✅ Fully Operational Components 
Component
 	
Status
 	
Testing
 	
Description
 
Data Collection System ✅ COMPLETE 8/8 tests passing Historical and real-time data fetching from Bybit with **NEW** shared WebSocket architecture
Backtesting Engine	✅ COMPLETE	All tests passing	Comprehensive backtesting with performance analytics 
API Management System	✅ COMPLETE	All tests passing	Secure API credential management 
Parameter Management	✅ COMPLETE	All tests passing	Strategy parameter optimization and storage 
Optimization Engine	✅ COMPLETE	All tests passing	Bayesian optimization for strategy parameters 
GUI Dashboard	✅ COMPLETE	Operational	Central control interface 
 
  
⚠️ Partially Working Components 
Component
 	
Status
 	
Known Issues
 	
Workaround
 
 Strategy Building Component	🔶 PARTIALLY WORKING	Signal integration issues, inconsistent return types	Use manual indicator calculation, avoid complex multi-indicator strategies 
 
  
🚀 Quick Start 
Prerequisites 

     Python 3.8+
     Bybit API credentials (for live trading)
     Windows PC (optimized for Windows deployment)
     Stable internet connection
     

Installation 

# 1. Clone the repository
git clone https://github.com/cobra0506/AIAssistedTradeBot.git
cd AIAssistedTradeBot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables (optional, for live trading)
set BYBIT_API_KEY=your_api_key_here
set BYBIT_API_SECRET=your_api_secret_here

## 🔄 **NEW** Shared WebSocket Architecture

The system now implements a **shared WebSocket architecture** that eliminates duplicate connections and ensures data consistency:

### Key Benefits:
- **Resource Efficiency**: Single WebSocket connection shared between data collection and paper trading
- **Data Consistency**: Both systems receive identical real-time data streams
- **Performance**: Reduced overhead and improved resource management
- **Reliability**: Centralized connection management with auto-recovery

### How It Works:
1. **SharedWebSocketManager**: Singleton pattern ensures only one WebSocket connection exists
2. **Data Collection**: Uses shared connection for historical and real-time data
3. **Paper Trading**: Uses the same shared connection for trading decisions
4. **Resource Management**: Automatic cleanup prevents memory leaks and unclosed sessions

### Testing:
- Comprehensive test suite validates shared WebSocket functionality
- All tests pass (7/7) with proper resource cleanup
- No duplicate WebSocket connections or memory leaks

Running the Application 
Method 1: Using the Dashboard GUI (Recommended) 

python main.py

This opens the control center dashboard where you can: 

     Start/Stop data collection
     Monitor system status
     Access parameter management
     Access API account management
     Open backtesting windows
     

Method 2: Direct Data Collection 

python shared_modules/data_collection/launch_data_collection.py

📊 Working Example: Simple RSI Strategy 

Here's a simple working example that demonstrates current capabilities: 

"""
Simple RSI Strategy Example
This example demonstrates manual indicator calculation (RECOMMENDED APPROACH)
Author: AI Assisted TradeBot Team
Date: 2025
"""

import sys
import os
import pandas as pd
import numpy as np

# Add parent directories to path (REQUIRED)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import required components
from simple_strategy.strategies.indicators_library import rsi
from simple_strategy.shared.data_feeder import DataFeeder
from simple_strategy.backtester.backtester_engine import BacktesterEngine

def simple_rsi_strategy(symbols=None, timeframes=None, **params):
    """
    Simple RSI strategy using manual indicator calculation
    This approach is currently more reliable than StrategyBuilder
    """
    # Handle None values (REQUIRED)
    if symbols is None or len(symbols) == 0:
        symbols = ['BTCUSDT']  # Default symbol
    if timeframes is None or len(timeframes) == 0:
        timeframes = ['5m']  # Default timeframe
    
    # Get strategy parameters
    rsi_period = params.get('rsi_period', 14)
    oversold = params.get('oversold', 30)
    overbought = params.get('overbought', 70)
    
    print(f"Creating RSI strategy for {symbols} on {timeframes}")
    print(f"Parameters: RSI Period={rsi_period}, Oversold={oversold}, Overbought={overbought}")
    
    # Create DataFeeder
    data_feeder = DataFeeder(data_dir='data')
    
    # Load data for backtesting
    data = data_feeder.load_data(
        symbols=symbols,
        timeframes=timeframes,
        start_date='2023-01-01',
        end_date='2023-12-31'
    )
    
    if data is None or len(data) == 0:
        print("❌ No data available for backtesting")
        return None
    
    # Manual indicator calculation (RECOMMENDED)
    signals = []
    for symbol in symbols:
        for timeframe in timeframes:
            df = data.get((symbol, timeframe))
            if df is not None:
                # Calculate RSI manually
                df['rsi'] = rsi(df['close'], period=rsi_period)
                
                # Generate signals manually
                df['signal'] = 0  # Hold by default
                df.loc[df['rsi'] < oversold, 'signal'] = 1   # Buy
                df.loc[df['rsi'] > overbought, 'signal'] = -1  # Sell
                
                signals.append(df)
    
    if not signals:
        print("❌ No signals generated")
        return None
    
    # Combine all signals
    combined_signals = pd.concat(signals)
    
    # Create a simple strategy object
    class SimpleStrategy:
        def __init__(self, signals):
            self.signals = signals
            self.name = "Simple RSI Strategy"
        
        def get_signals(self):
            return self.signals
    
    return SimpleStrategy(combined_signals)

# Example usage
if __name__ == "__main__":
    # Create strategy
    strategy = simple_rsi_strategy(
        symbols=['BTCUSDT'],
        timeframes=['1h'],
        rsi_period=14,
        oversold=30,
        overbought=70
    )
    
    if strategy:
        # Run backtest
        data_feeder = DataFeeder(data_dir='data')
        backtest = BacktesterEngine(data_feeder=data_feeder, strategy=strategy)
        
        results = backtest.run_backtest(
            symbols=['BTCUSDT'],
            timeframes=['1h'],
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        print(f"Backtest Results:")
        print(f"Total Return: {results['performance_metrics']['total_return']:.2f}%")
        print(f"Win Rate: {results['performance_metrics']['win_rate']:.2f}%")

📚 Detailed Documentation 

For comprehensive documentation on each component, please refer to the NewDocs/ folder: 

     NewDocs/ARCHITECTURE.md - System architecture overview
     NewDocs/DEVELOPMENT_STATUS.md - Detailed component status
     NewDocs/PROJECT_OVERVIEW.md - Project goals and scope
     NewDocs/Backtesting Component/ - Backtesting engine documentation
     NewDocs/Strategy Building Component/ - Strategy builder documentation (see known issues)
     NewDocs/Data Management Component/ - Data collection and management
     NewDocs/Optimization Component/ - Parameter optimization
     NewDocs/Trading Interface Component/ - API and trading interface
     NewDocs/Testing Framework Component/ - Testing framework
     

⚠️ Important Known Issues 
Strategy Builder Limitations 

The Strategy Builder component has known issues that prevent reliable strategy creation: 

    Indicator Integration Problems: Indicators are calculated but not properly integrated into DataFrames 
    Signal Generation Issues: Signal functions receive wrong parameters and return inconsistent types 
    Zero Trades Problem: Many strategies result in 0 trades due to signal processing issues 

Current Recommendation: Use manual indicator calculation as shown in the example above until these issues are resolved. 
Testing Status 

     Signal Functions: 13/13 tests passing ✅
     Core System: 40+ tests passing ✅
     Calculation Accuracy: 6/6 tests passing ✅
     Strategy Integration: Known issues, not fully tested ❌
     

🛠️ Development Workflow 

    Start with Data Collection: Use the GUI to collect historical data 
    Test Indicators Manually: Verify indicator calculations work correctly 
    Create Simple Strategies: Use manual calculation approach (as shown above) 
    Backtest Thoroughly: Use the backtesting engine to validate strategies 
    Optimize Parameters: Use the optimization engine for parameter tuning 
    Paper Trade: Test with demo API before live trading 

🆘 Getting Help 

If you encounter issues: 

    Check the detailed documentation in NewDocs/ 
    Review the known issues section above 
    Look at existing strategy files in simple_strategy/strategies/examples/ 
    Check the testing framework in tests/ for working examples 

📝 Next Steps 

    Explore the Documentation: Dive into the NewDocs/ folder for detailed information 
    Run the Example: Try the simple RSI strategy example above 
    Collect Data: Use the GUI to collect some historical data 
    Experiment: Modify the example strategy to test different indicators 
    Check Status: Review NewDocs/DEVELOPMENT_STATUS.md for the latest updates 
    
Note: This project is actively being developed. The Strategy Builder component is being improved to resolve the known issues mentioned above. For the most current status, always check the NewDocs/DEVELOPMENT_STATUS.md file.