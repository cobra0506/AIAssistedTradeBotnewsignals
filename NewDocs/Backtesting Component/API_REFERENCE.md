# Backtesting Component - API REFERENCE

## Complete API Documentation

This document provides comprehensive API documentation for all classes, methods, and functions in the Backtesting Component.

## Table of Contents

1. [BacktesterEngine Class](#backtesterengine-class)
2. [PerformanceTracker Class](#performancetracker-class)
3. [PositionManager Class](#positionmanager-class)
4. [RiskManager Class](#riskmanager-class)
5. [Utility Functions](#utility-functions)
6. [Integration APIs](#integration-apis)
7. [Configuration APIs](#configuration-apis)
8. [Error Handling](#error-handling)

---

## BacktesterEngine Class

### Overview
The main orchestrator class for backtesting operations. Handles time-synchronized processing of multiple symbols and coordinates all backtesting components.

### Constructor

#### `__init__(self, data_feeder, strategy, initial_capital=10000)`
Initialize the BacktesterEngine with required components.

**Parameters:**
- `data_feeder` (DataFeeder): Data feeder instance for historical data access
- `strategy` (Strategy): Built strategy object from StrategyBuilder
- `initial_capital` (float, optional): Starting capital for backtest. Defaults to 10000.

**Returns:**
- None

**Raises:**
- `TypeError`: If data_feeder or strategy is not the correct type
- `ValueError`: If initial_capital is not positive

**Example:**
```python
from simple_strategy.backtester.backtester_engine import BacktesterEngine
from simple_strategy.shared.data_feeder import DataFeeder

data_feeder = DataFeeder(data_dir='data')
backtest = BacktesterEngine(data_feeder=data_feeder, strategy=my_strategy, initial_capital=50000)

Core Methods 
run_backtest(self, symbols, timeframes, start_date, end_date) 

Execute complete backtest for specified parameters. 

Parameters: 

     symbols (list): List of trading symbols (e.g., ['BTCUSDT', 'ETHUSDT'])
     timeframes (list): List of timeframes (e.g., ['1m', '5m'])
     start_date (str): Backtest start date in 'YYYY-MM-DD' format
     end_date (str): Backtest end date in 'YYYY-MM-DD' format
     

Returns: 

     dict: Complete backtest results containing:
         performance_metrics (dict): Performance metrics
         trade_history (list): Complete trade history
         equity_curve (list): Equity curve data
         positions_history (list): Position changes over time
         risk_metrics (dict): Risk analysis metrics
         
     

Raises: 

     ValueError: If date format is invalid or start_date > end_date
     DataError: If historical data is not available
     RuntimeError: If backtest execution fails
     

Example: 

results = backtest.run_backtest(
    symbols=['BTCUSDT'],
    timeframes=['1m'],
    start_date='2023-01-01',
    end_date='2023-12-31'
)

print(f"Total Return: {results['performance_metrics']['total_return']:.2f}%")
print(f"Sharpe Ratio: {results['performance_metrics']['sharpe_ratio']:.2f}")

process_timestamp(self, timestamp) 

Process all symbols at a specific timestamp with time synchronization. 

Parameters: 

     timestamp (datetime): Current timestamp to process
     

Returns: 

     dict: Processing results for the timestamp
     

Raises: 

     ValueError: If timestamp is invalid
     DataError: If market data is not available for timestamp
     

Example: 

from datetime import datetime
timestamp = datetime(2023, 1, 1, 0, 0)
result = backtest.process_timestamp(timestamp)

execute_trade(self, symbol, signal, price, timestamp) 

Execute trade with proper risk management and position sizing. 

Parameters: 

     symbol (str): Trading symbol
     signal (int): Trading signal (1 for long, -1 for short, 0 for neutral)
     price (float): Current market price
     timestamp (datetime): Trade execution timestamp
     

Returns: 

     dict: Trade execution result
     

Raises: 

     ValueError: If parameters are invalid
     RiskError: If trade violates risk management rules
     PositionError: If position cannot be opened/closed
     

Example: 

trade_result = backtest.execute_trade(
    symbol='BTCUSDT',
    signal=1,  # Long signal
    price=45000.0,
    timestamp=datetime.now()
)

get_backtest_results(self) 

Get complete backtest results and performance metrics. 

Parameters: 

     None
     

Returns: 

     dict: Complete backtest results (same format as run_backtest)
     

Raises: 

     RuntimeError: If backtest has not been executed
     

Example: 

results = backtest.get_backtest_results()
performance = results['performance_metrics']

Configuration Methods 
set_initial_capital(self, capital) 

Set initial capital for backtesting. 

Parameters: 

     capital (float): Initial capital amount
     

Returns: 

     None
     

Raises: 

     ValueError: If capital is not positive
     

Example: 

backtest.set_initial_capital(100000)

set_processing_speed(self, speed) 

Set processing speed for backtesting. 

Parameters: 

     speed (str): Processing speed ('slow', 'normal', 'fast')
     

Returns: 

     None
     

Raises: 

     ValueError: If speed is invalid
     

Example: 

backtest.set_processing_speed('fast')

PerformanceTracker Class 
Overview 

Tracks and calculates all performance metrics for the backtest. Maintains equity curves, trade history, and performance analytics. 
Constructor 
__init__(self, initial_balance=10000) 

Initialize the PerformanceTracker. 

Parameters: 

     initial_balance (float, optional): Initial account balance. Defaults to 10000.
     

Returns: 

     None
     

Example: 

from simple_strategy.backtester.performance_tracker import PerformanceTracker
tracker = PerformanceTracker(initial_balance=50000)

Core Methods 
record_trade(self, trade_data) 

Record a completed trade. 

Parameters: 

     trade_data (dict): Trade data containing:
         symbol (str): Trading symbol
         direction (str): Trade direction ('long' or 'short')
         entry_price (float): Entry price
         exit_price (float): Exit price
         entry_timestamp (datetime): Entry timestamp
         exit_timestamp (datetime): Exit timestamp
         size (float): Position size
         pnl (float): Profit and loss
         
     

Returns: 

     None
     

Raises: 

     ValueError: If trade_data is invalid
     

Example: 

trade_data = {
    'symbol': 'BTCUSDT',
    'direction': 'long',
    'entry_price': 45000.0,
    'exit_price': 46000.0,
    'entry_timestamp': datetime(2023, 1, 1),
    'exit_timestamp': datetime(2023, 1, 2),
    'size': 0.1,
    'pnl': 100.0
}
tracker.record_trade(trade_data)

update_equity(self, timestamp, balance, positions_value) 

Update equity curve with current account status. 

Parameters: 

     timestamp (datetime): Current timestamp
     balance (float): Cash balance
     positions_value (float): Total value of open positions
     

Returns: 

     None
     

Raises: 

     ValueError: If parameters are invalid
     

Example: 

tracker.update_equity(
    timestamp=datetime.now(),
    balance=45000.0,
    positions_value=5000.0
)

calculate_metrics(self) 

Calculate all performance metrics. 

Parameters: 

     None
     

Returns: 

     dict: Performance metrics containing:
         total_return (float): Total return percentage
         annualized_return (float): Annualized return percentage
         max_drawdown (float): Maximum drawdown percentage
         sharpe_ratio (float): Sharpe ratio
         sortino_ratio (float): Sortino ratio
         win_rate (float): Win rate percentage
         profit_factor (float): Profit factor
         total_trades (int): Total number of trades
         winning_trades (int): Number of winning trades
         losing_trades (int): Number of losing trades
         
     

Raises: 

     RuntimeError: If insufficient data for calculation
     

Example: 

metrics = tracker.calculate_metrics()
print(f"Total Return: {metrics['total_return']:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

get_equity_curve(self) 

Get equity curve data. 

Parameters: 

     None
     

Returns: 

     list: List of dictionaries containing equity curve data points
     

Example: 

equity_curve = tracker.get_equity_curve()
for point in equity_curve:
    print(f"{point['timestamp']}: {point['equity']}")

get_trade_history(self) 

Get complete trade history. 

Parameters: 

     None
     

Returns: 

     list: List of dictionaries containing trade data
     

Example: 

trade_history = tracker.get_trade_history()
for trade in trade_history:
    print(f"{trade['symbol']}: {trade['pnl']}")

PositionManager Class 
Overview 

Manages all trading positions, account balances, and position limits. Handles position sizing, P&L calculations, and risk enforcement. 
Constructor 
__init__(self, initial_balance, max_positions=10, max_risk_per_trade=0.01) 

Initialize the PositionManager. 

Parameters: 

     initial_balance (float): Initial account balance
     max_positions (int, optional): Maximum number of concurrent positions. Defaults to 10.
     max_risk_per_trade (float, optional): Maximum risk per trade as percentage. Defaults to 0.01.
     

Returns: 

     None
     

Raises: 

     ValueError: If parameters are invalid
     

Example: 

from simple_strategy.backtester.position_manager import PositionManager
position_manager = PositionManager(
    initial_balance=100000,
    max_positions=5,
    max_risk_per_trade=0.02
)

Core Methods 
can_open_position(self, symbol, position_size) 

Check if new position can be opened. 

Parameters: 

     symbol (str): Trading symbol
     position_size (float): Position size to check
     

Returns: 

     bool: True if position can be opened
     

Example: 

can_open = position_manager.can_open_position('BTCUSDT', 0.1)
if can_open:
    print("Can open position")

open_position(self, symbol, direction, size, entry_price, timestamp) 

Open new position. 

Parameters: 

     symbol (str): Trading symbol
     direction (str): Position direction ('long' or 'short')
     size (float): Position size
     entry_price (float): Entry price
     timestamp (datetime): Position open timestamp
     

Returns: 

     bool: True if position opened successfully
     

Raises: 

     ValueError: If parameters are invalid
     PositionError: If position cannot be opened
     RiskError: If position violates risk rules
     

Example: 

success = position_manager.open_position(
    symbol='BTCUSDT',
    direction='long',
    size=0.1,
    entry_price=45000.0,
    timestamp=datetime.now()
)

close_position(self, symbol, exit_price, timestamp) 

Close existing position. 

Parameters: 

     symbol (str): Trading symbol
     exit_price (float): Exit price
     timestamp (datetime): Position close timestamp
     

Returns: 

     dict: Position close result containing P&L information
     

Raises: 

     ValueError: If parameters are invalid
     PositionError: If position does not exist
     

Example: 

result = position_manager.close_position(
    symbol='BTCUSDT',
    exit_price=46000.0,
    timestamp=datetime.now()
)
print(f"P&L: {result['pnl']}")

update_position_value(self, symbol, current_price) 

Update position value with current price. 

Parameters: 

     symbol (str): Trading symbol
     current_price (float): Current market price
     

Returns: 

     None
     

Raises: 

     ValueError: If parameters are invalid
     PositionError: If position does not exist
     

Example: 

position_manager.update_position_value('BTCUSDT', 45500.0)

get_account_summary(self) 

Get account balance and position summary. 

Parameters: 

     None
     

Returns: 

     dict: Account summary containing:
         cash_balance (float): Available cash balance
         positions_value (float): Total value of open positions
         total_balance (float): Total account balance
         unrealized_pnl (float): Total unrealized P&L
         positions_count (int): Number of open positions
         
     

Example: 

summary = position_manager.get_account_summary()
print(f"Total Balance: {summary['total_balance']}")
print(f"Open Positions: {summary['positions_count']}")

RiskManager Class 
Overview 

Implements risk management rules and calculations. Handles position sizing, risk validation, and portfolio-level risk monitoring. 
Constructor 
__init__(self, max_risk_per_trade=0.02, max_portfolio_risk=0.10) 

Initialize the RiskManager. 

Parameters: 

     max_risk_per_trade (float, optional): Maximum risk per trade as percentage. Defaults to 0.02.
     max_portfolio_risk (float, optional): Maximum portfolio risk as percentage. Defaults to 0.10.
     

Returns: 

     None
     

Raises: 

     ValueError: If parameters are invalid
     

Example: 

from simple_strategy.backtester.risk_manager import RiskManager
risk_manager = RiskManager(
    max_risk_per_trade=0.01,
    max_portfolio_risk=0.05
)

Core Methods 
calculate_position_size(self, symbol, price, account_balance, risk_amount=None) 

Calculate safe position size based on risk management rules. 

Parameters: 

     symbol (str): Trading symbol
     price (float): Current market price
     account_balance (float): Total account balance
     risk_amount (float, optional): Specific risk amount. Defaults to None.
     

Returns: 

     float: Safe position size
     

Raises: 

     ValueError: If parameters are invalid
     

Example: 

position_size = risk_manager.calculate_position_size(
    symbol='BTCUSDT',
    price=45000.0,
    account_balance=100000
)
print(f"Safe position size: {position_size}")

validate_trade(self, symbol, direction, size, price, account_summary) 

Validate if a trade complies with all risk management rules. 

Parameters: 

     symbol (str): Trading symbol
     direction (str): Trade direction ('long' or 'short')
     size (float): Position size
     price (float): Trade price
     account_summary (dict): Current account summary
     

Returns: 

     tuple: (is_valid, reason) where is_valid is bool and reason is str
     

Example: 

is_valid, reason = risk_manager.validate_trade(
    symbol='BTCUSDT',
    direction='long',
    size=0.1,
    price=45000.0,
    account_summary=account_summary
)
if is_valid:
    print("Trade is valid")
else:
    print(f"Trade invalid: {reason}")

calculate_portfolio_risk(self, account_summary) 

Calculate current portfolio risk level. 

Parameters: 

     account_summary (dict): Current account summary
     

Returns: 

     float: Portfolio risk as percentage
     

Raises: 

     ValueError: If account_summary is invalid
     

Example: 

portfolio_risk = risk_manager.calculate_portfolio_risk(account_summary)
print(f"Portfolio Risk: {portfolio_risk:.2%}")

set_risk_limits(self, max_risk_per_trade=None, max_portfolio_risk=None) 

Set risk management limits. 

Parameters: 

     max_risk_per_trade (float, optional): Maximum risk per trade
     max_portfolio_risk (float, optional): Maximum portfolio risk
     

Returns: 

     None
     

Raises: 

     ValueError: If parameters are invalid
     

Example: 

risk_manager.set_risk_limits(
    max_risk_per_trade=0.015,
    max_portfolio_risk=0.08
)

Utility Functions 
validate_backtest_parameters(symbols, timeframes, start_date, end_date) 

Validate backtest input parameters. 

Parameters: 

     symbols (list): List of trading symbols
     timeframes (list): List of timeframes
     start_date (str): Start date
     end_date (str): End date
     

Returns: 

     tuple: (is_valid, error_message)
     

Example: 

from simple_strategy.backtester.utils import validate_backtest_parameters
is_valid, error = validate_backtest_parameters(
    ['BTCUSDT'], ['1m'], '2023-01-01', '2023-12-31'
)

calculate_performance_metrics(equity_curve, trade_history) 

Calculate comprehensive performance metrics. 

Parameters: 

     equity_curve (list): Equity curve data
     trade_history (list): Trade history data
     

Returns: 

     dict: Performance metrics
     

Example: 

from simple_strategy.backtester.utils import calculate_performance_metrics
metrics = calculate_performance_metrics(equity_curve, trade_history)

format_backtest_results(results) 

Format backtest results for display. 

Parameters: 

     results (dict): Raw backtest results
     

Returns: 

     dict: Formatted results
     

Example: 

from simple_strategy.backtester.utils import format_backtest_results
formatted = format_backtest_results(results)

Integration APIs 
integrate_with_strategy_builder(strategy) 

Integrate backtester with Strategy Builder output. 

Parameters: 

     strategy (Strategy): Strategy object from StrategyBuilder
     

Returns: 

     bool: True if integration successful
     

Example: 

from simple_strategy.backtester.integration import integrate_with_strategy_builder
success = integrate_with_strategy_builder(my_strategy)

setup_data_feeder(data_feeder) 

Set up data feeder integration. 

Parameters: 

     data_feeder (DataFeeder): DataFeeder instance
     

Returns: 

     bool: True if setup successful
     

Example: 

from simple_strategy.backtester.integration import setup_data_feeder
success = setup_data_feeder(data_feeder)

Configuration APIs 
load_backtest_config(config_file) 

Load backtesting configuration from file. 

Parameters: 

     config_file (str): Path to configuration file
     

Returns: 

     dict: Configuration data
     

Raises: 

     FileNotFoundError: If config file not found
     JSONDecodeError: If config file is invalid JSON
     

Example: 

from simple_strategy.backtester.config import load_backtest_config
config = load_backtest_config('backtest_config.json')

save_backtest_config(config, config_file) 

Save backtesting configuration to file. 

Parameters: 

     config (dict): Configuration data
     config_file (str): Path to configuration file
     

Returns: 

     bool: True if save successful
     

Example: 

from simple_strategy.backtester.config import save_backtest_config
config = {
    'initial_capital': 100000,
    'max_positions': 5,
    'max_risk_per_trade': 0.02
}
success = save_backtest_config(config, 'backtest_config.json')

Error Handling 
Exception Classes 
BacktestError 

Base exception class for backtesting errors. 
DataError 

Raised when data-related errors occur. 
RiskError 

Raised when risk management rules are violated. 
PositionError 

Raised when position-related errors occur. 
ConfigurationError 

Raised when configuration errors occur. 
Error Handling Examples 
Basic Error Handling 

try:
    results = backtest.run_backtest(
        symbols=['BTCUSDT'],
        timeframes=['1m'],
        start_date='2023-01-01',
        end_date='2023-12-31'
    )
except DataError as e:
    print(f"Data error: {e}")
except RiskError as e:
    print(f"Risk error: {e}")
except BacktestError as e:
    print(f"Backtest error: {e}")

Advanced Error Handling

def safe_backtest_execution(backtest, symbols, timeframes, start_date, end_date):
    try:
        # Validate parameters
        if not validate_backtest_parameters(symbols, timeframes, start_date, end_date)[0]:
            raise ValueError("Invalid backtest parameters")
        
        # Execute backtest
        results = backtest.run_backtest(symbols, timeframes, start_date, end_date)
        
        # Validate results
        if not results or 'performance_metrics' not in results:
            raise BacktestError("Invalid backtest results")
        
        return results
        
    except Exception as e:
        print(f"Backtest execution failed: {e}")
        return None
    finally:
        # Cleanup
        backtest.cleanup()

API Usage Examples 
Complete Backtest Workflow 

from simple_strategy.backtester.backtester_engine import BacktesterEngine
from simple_strategy.shared.data_feeder import DataFeeder
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import rsi, sma
from simple_strategy.strategies.signals_library import overbought_oversold

# Create strategy
strategy = StrategyBuilder(['BTCUSDT'], ['1m'])
strategy.add_indicator('rsi', rsi, period=14)
strategy.add_indicator('sma', sma, period=20)
strategy.add_signal_rule('oversold', overbought_oversold, oversold=30)
my_strategy = strategy.build()

# Set up backtester
data_feeder = DataFeeder(data_dir='data')
backtest = BacktesterEngine(data_feeder=data_feeder, strategy=my_strategy)

# Execute backtest
results = backtest.run_backtest(
    symbols=['BTCUSDT'],
    timeframes=['1m'],
    start_date='2023-01-01',
    end_date='2023-12-31'
)

# Analyze results
metrics = results['performance_metrics']
print(f"Total Return: {metrics['total_return']:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")

Advanced Configuration

from simple_strategy.backtester.config import load_backtest_config

# Load configuration
config = load_backtest_config('advanced_config.json')

# Create backtester with custom configuration
backtest = BacktesterEngine(
    data_feeder=data_feeder,
    strategy=my_strategy,
    initial_capital=config['initial_capital']
)

# Configure risk management
backtest.risk_manager.set_risk_limits(
    max_risk_per_trade=config['max_risk_per_trade'],
    max_portfolio_risk=config['max_portfolio_risk']
)

# Configure position management
backtest.position_manager.max_positions = config['max_positions']

# Execute with custom settings
results = backtest.run_backtest(
    symbols=config['symbols'],
    timeframes=config['timeframes'],
    start_date=config['start_date'],
    end_date=config['end_date']
)

Error Handling and Recovery

def robust_backtest_execution(backtest, symbols, timeframes, start_date, end_date):
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Validate inputs
            validation_result = validate_backtest_parameters(symbols, timeframes, start_date, end_date)
            if not validation_result[0]:
                raise ValueError(f"Invalid parameters: {validation_result[1]}")
            
            # Execute backtest
            results = backtest.run_backtest(symbols, timeframes, start_date, end_date)
            
            # Validate results
            if not results or 'performance_metrics' not in results:
                raise BacktestError("Invalid backtest results structure")
            
            return results
            
        except DataError as e:
            print(f"Data error (attempt {retry_count + 1}): {e}")
            retry_count += 1
            if retry_count < max_retries:
                print("Retrying with data refresh...")
                backtest.data_feeder.refresh_data()
                
        except RiskError as e:
            print(f"Risk error: {e}")
            return None
            
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None
    
    print(f"Backtest failed after {max_retries} attempts")
    return None

API Version Information 

     Current Version: 1.0.0
     Last Updated: 2025-06-17
     Compatibility: Compatible with Strategy Builder v1.0+
     Dependencies: pandas, numpy, matplotlib, scikit-learn
     

API Changelog 
Version 1.0.0 (2025-06-17) 

     Initial release
     Complete backtesting functionality
     Integration with Strategy Builder
     Comprehensive performance metrics
     Risk management integration
     Multi-symbol and multi-timeframe support
     

This API reference provides complete documentation for all Backtesting Component functionality. All APIs are production-ready and thoroughly tested. 