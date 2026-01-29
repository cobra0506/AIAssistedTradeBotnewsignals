# Trading Interface Component - API REFERENCE

## Overview
This document provides a comprehensive API reference for the Trading Interface Component, including all classes, methods, and data structures used throughout the system.

## 1. API Management System

### APIManager Class

#### Class Definition
```python
class APIManager:
    def __init__(self)

Methods 
__init__(self) 

Description: Initialize the API Manager with default configuration.
Returns: None
Example: 
from simple_strategy.trading.api_manager import APIManager
api_manager = APIManager()

add_demo_account(self, name, api_key, api_secret, description="") 

Description: Add a new demo trading account to the system.
Parameters: 

     name (str): Unique identifier for the account
     api_key (str): Bybit API key for the demo account
     api_secret (str): Bybit API secret for the demo account
     description (str, optional): Description of the account (default: "")
     

Returns: bool - True if successful, False if account already exists 

Example: 

success = api_manager.add_demo_account(
    "demo_test_1",
    "your_api_key_here",
    "your_api_secret_here",
    "Primary demo testing account"
)

add_live_account(self, name, api_key, api_secret, description="") 

Description: Add a new live trading account to the system.
Parameters: 

     name (str): Unique identifier for the account
     api_key (str): Bybit API key for the live account
     api_secret (str): Bybit API secret for the live account
     description (str, optional): Description of the account (default: "")
     

Returns: bool - True if successful, False if account already exists 

Example: 

success = api_manager.add_live_account(
    "live_main_1",
    "your_live_api_key_here",
    "your_live_api_secret_here",
    "Main live trading account"
)

get_demo_account(self, name) 

Description: Retrieve a specific demo account by name.
Parameters: 

     name (str): Name of the demo account to retrieve
     

Returns: dict or None - Account information if found, None otherwise 

Account Structure: 

{
    "api_key": "string",
    "api_secret": "string", 
    "description": "string",
    "testnet": True
}

Example:

account = api_manager.get_demo_account("demo_test_1")
if account:
    print(f"API Key: {account['api_key']}")
    print(f"Description: {account['description']}")

get_live_account(self, name) 

Description: Retrieve a specific live account by name.
Parameters: 

     name (str): Name of the live account to retrieve
     

Returns: dict or None - Account information if found, None otherwise 

Example: 

account = api_manager.get_live_account("live_main_1")
if account:
    print(f"API Key: {account['api_key']}")
    print(f"Description: {account['description']}")

get_demo_account_names(self) 

Description: Get list of all demo account names.
Parameters: None
Returns: list - List of demo account names 

Example: 

demo_accounts = api_manager.get_demo_account_names()
print(f"Available demo accounts: {demo_accounts}")

get_live_account_names(self) 

Description: Get list of all live account names.
Parameters: None
Returns: list - List of live account names 

Example: 

live_accounts = api_manager.get_live_account_names()
print(f"Available live accounts: {live_accounts}")

update_demo_account(self, name, api_key, api_secret, description="") 

Description: Update an existing demo account.
Parameters: 

     name (str): Name of the account to update
     api_key (str): New API key
     api_secret (str): New API secret
     description (str, optional): New description (default: "")
     

Returns: bool - True if successful, False if account not found 

Example: 

success = api_manager.update_demo_account(
    "demo_test_1",
    "new_api_key",
    "new_api_secret",
    "Updated description"
)

update_live_account(self, name, api_key, api_secret, description="") 

Description: Update an existing live account.
Parameters: 

     name (str): Name of the account to update
     api_key (str): New API key
     api_secret (str): New API secret
     description (str, optional): New description (default: "")
     

Returns: bool - True if successful, False if account not found 

Example: 

success = api_manager.update_live_account(
    "live_main_1",
    "new_live_api_key",
    "new_live_api_secret",
    "Updated live description"
)

delete_demo_account(self, name) 

Description: Delete a demo account.
Parameters: 

     name (str): Name of the account to delete
     

Returns: bool - True if successful, False if account not found 

Example: 

success = api_manager.delete_demo_account("demo_test_1")

delete_live_account(self, name) 

Description: Delete a live account.
Parameters: 

     name (str): Name of the account to delete
     

Returns: bool - True if successful, False if account not found 

Example: 

success = api_manager.delete_live_account("live_main_1")

get_all_accounts(self) 

Description: Get all accounts (both demo and live).
Parameters: None
Returns: dict - Complete accounts structure 

Structure: 

{
    "demo_accounts": {
        "account_name": {
            "api_key": "string",
            "api_secret": "string",
            "description": "string",
            "testnet": True
        }
    },
    "live_accounts": {
        "account_name": {
            "api_key": "string",
            "api_secret": "string",
            "description": "string",
            "testnet": False
        }
    }
}

Example:

all_accounts = api_manager.get_all_accounts()
print(f"Demo accounts: {len(all_accounts['demo_accounts'])}")
print(f"Live accounts: {len(all_accounts['live_accounts'])}")

APIGUI Class 
Class Definition 

class APIGUI:
    def __init__(self, root)

Methods 
__init__(self, root) 

Description: Initialize the API GUI with the provided root window.
Parameters: 

     root (tk.Tk): Root tkinter window
     

Returns: None 

Example: 

import tkinter as tk
from simple_strategy.trading.api_gui import APIGUI

root = tk.Tk()
app = APIGUI(root)
root.mainloop()

create_widgets(self) 

Description: Create and arrange all GUI widgets.
Parameters: None
Returns: None 

GUI Structure: 

     Main container with padding
     Title label
     Notebook with tabs for Demo and Live accounts
     Treeview for account listing
     Action buttons (Add, Edit, Delete)
     Close button
     

create_account_tab(self, parent, account_type) 

Description: Create a tab for managing accounts of specified type.
Parameters: 

     parent (ttk.Frame): Parent frame for the tab
     account_type (str): Type of account ("demo" or "live")
     

Returns: None 
refresh_account_lists(self) 

Description: Refresh the account lists in both tabs.
Parameters: None
Returns: None 
add_account(self, account_type) 

Description: Open dialog to add a new account.
Parameters: 

     account_type (str): Type of account to add
     

Returns: None 
edit_account(self, account_type) 

Description: Open dialog to edit selected account.
Parameters: 

     account_type (str): Type of account to edit
     

Returns: None 
delete_account(self, account_type) 

Description: Delete selected account with confirmation.
Parameters: 

     account_type (str): Type of account to delete
     

Returns: None 
2. Parameter Management System 
ParameterManager Class 
Class Definition 

class ParameterManager:
    def __init__(self)

Methods 
__init__(self) 

Description: Initialize the Parameter Manager.
Returns: None 

Example: 

from simple_strategy.trading.parameter_manager import ParameterManager
param_manager = ParameterManager()

update_parameters(self, strategy_name, params)
Description: Update parameters for a specific strategy.
Parameters:
strategy_name (str): Name of the strategy
params (dict): Parameter values to update
Returns: bool - True if successful 

Example: 

parameters = {
    "rsi_period": 14,
    "sma_short": 20,
    "sma_long": 50,
    "oversold_threshold": 30,
    "overbought_threshold": 70
}
success = param_manager.save_optimized_parameters(
    "rsi_sma_strategy",
    parameters,
    "2025-11-10"
)

get_parameters(self, strategy_name)
Description: Get parameters for a specific strategy.
Parameters:
strategy_name (str): Name of the strategy
Returns: dict or None - Parameter data if found, None otherwise
Structure:
{
"param_name": "value",
"last_optimized": "2025-11-10"
}

Example:

params = param_manager.load_optimized_parameters("rsi_sma_strategy")
if params:
    print(f"Optimized on: {params['optimization_date']}")
    print(f"RSI period: {params['parameters']['rsi_period']}")

get_all_strategies(self)
Description: Get all strategy names that have optimized parameters.
Parameters: None
Returns: list - List of strategy names 

Example: 

strategies = param_manager.get_all_optimized_strategies()
print(f"Optimized strategies: {strategies}")

ParameterGUI Class 
Class Definition 

class ParameterGUI:
    def __init__(self, root)

Methods 
__init__(self, root) 

Description: Initialize the Parameter GUI.
Parameters: 

     root (tk.Tk): Root tkinter window
     

Returns: None 

Example: 

import tkinter as tk
from simple_strategy.trading.parameter_gui import ParameterGUI

root = tk.Tk()
app = ParameterGUI(root)
root.mainloop()

create_widgets(self) 

Description: Create and arrange all GUI widgets.
Parameters: None
Returns: None 

GUI Structure: 

     Main container with padding
     Title label
     Treeview for parameter listing
     Status indicators for optimization
     Action buttons (View, Edit, Delete)
     Close button
     

refresh_parameter_list(self) 

Description: Refresh the parameter list display.
Parameters: None
Returns: None 
view_parameters(self) 

Description: View detailed parameters for selected strategy.
Parameters: None
Returns: None 
edit_parameters(self) 

Description: Edit parameters for selected strategy.
Parameters: None
Returns: None 
delete_parameters(self) 

Description: Delete parameters for selected strategy.
Parameters: None
Returns: None 
3. Paper Trading Engine 
PaperTradingEngine Class 
Class Definition 

class PaperTradingEngine:
    def __init__(self, api_manager, parameter_manager)

Methods 
__init__(self, api_manager, parameter_manager) 

Description: Initialize the Paper Trading Engine.
Parameters: 

     api_manager (APIManager): Instance of API manager
     parameter_manager (ParameterManager): Instance of parameter manager
     

Returns: None 

Example: 

from simple_strategy.trading.paper_trading_engine import PaperTradingEngine
from simple_strategy.trading.api_manager import APIManager
from simple_strategy.trading.parameter_manager import ParameterManager

api_manager = APIManager()
param_manager = ParameterManager()
paper_trader = PaperTradingEngine(api_manager, param_manager)

start_paper_trading(self, account_name, strategy_name, simulated_balance) 

Description: Start paper trading with specified parameters.
Parameters: 

     account_name (str): Name of the demo account to use
     strategy_name (str): Name of the strategy to trade
     simulated_balance (float): Starting balance for paper trading
     

Returns: bool - True if started successfully 

Example: 

success = paper_trader.start_paper_trading(
    "demo_test_1",
    "rsi_sma_strategy",
    1000.0
)

stop_paper_trading(self) 

Description: Stop paper trading and save results.
Parameters: None
Returns: bool - True if stopped successfully 

Example: 

success = paper_trader.stop_paper_trading()

execute_trade(self, signal, symbol, quantity, price) 

Description: Execute a paper trade.
Parameters: 

     signal (str): Trading signal ("buy" or "sell")
     symbol (str): Trading symbol (e.g., "BTCUSDT")
     quantity (float): Quantity to trade
     price (float): Execution price
     

Returns: dict - Trade execution details 

Structure: 

{
    "trade_id": "string",
    "timestamp": "datetime",
    "symbol": "string",
    "signal": "string",
    "quantity": float,
    "price": float,
    "status": "string"
}

Example:

trade_result = paper_trader.execute_trade(
    "buy",
    "BTCUSDT",
    0.001,
    50000.0
)

get_trading_status(self) 

Description: Get current trading status.
Parameters: None
Returns: dict - Trading status information 

Structure: 

{
    "is_running": bool,
    "current_balance": float,
    "total_trades": int,
    "winning_trades": int,
    "total_pnl": float,
    "start_time": "datetime"
}

Example:

status = paper_trader.get_trading_status()
print(f"Running: {status['is_running']}")
print(f"Current balance: {status['current_balance']}")

calculate_performance_metrics(self) 

Description: Calculate trading performance metrics.
Parameters: None
Returns: dict - Performance metrics 

Structure: 

{
    "total_return": float,
    "win_rate": float,
    "sharpe_ratio": float,
    "max_drawdown": float,
    "total_trades": int,
    "profit_factor": float
}

Example:

metrics = paper_trader.calculate_performance_metrics()
print(f"Total return: {metrics['total_return']:.2f}%")
print(f"Win rate: {metrics['win_rate']:.2f}%")

get_trade_history(self) 

Description: Get complete trade history.
Parameters: None
Returns: list - List of all trades 

Trade Structure: 

{
    "trade_id": "string",
    "timestamp": "datetime",
    "symbol": "string",
    "signal": "string",
    "quantity": float,
    "entry_price": float,
    "exit_price": float,
    "pnl": float,
    "status": "string"
}

Example:

history = paper_trader.get_trade_history()
for trade in history:
    print(f"{trade['symbol']}: {trade['pnl']}")

4. Data Structures 
API Account Structure 

{
    "api_key": "string",           # Bybit API key
    "api_secret": "string",        # Bybit API secret
    "description": "string",       # Account description
    "testnet": bool               # True for demo, False for live
}

Optimized Parameters Structure

{
    "parameters": {
        "param_name": "value"      # Strategy-specific parameters
    },
    "optimization_date": "string", # YYYY-MM-DD format
    "performance_metrics": {
        "total_return": float,     # Percentage return
        "win_rate": float,         # Win rate percentage
        "sharpe_ratio": float,     # Sharpe ratio
        "max_drawdown": float,     # Maximum drawdown
        "profit_factor": float     # Profit factor
    }
}

Trade Structure

{
    "trade_id": "string",          # Unique trade identifier
    "timestamp": "datetime",       # Trade execution time
    "symbol": "string",            # Trading symbol
    "signal": "string",            # "buy" or "sell"
    "quantity": float,             # Trade quantity
    "entry_price": float,          # Entry price
    "exit_price": float,           # Exit price (None if open)
    "pnl": float,                  # Profit/loss
    "status": "string"             # "open" or "closed"
}

5. Error Handling 
Common Exceptions 
AccountNotFoundError 

Description: Raised when requested account is not found.
Example: 

try:
    account = api_manager.get_demo_account("nonexistent_account")
except AccountNotFoundError as e:
    print(f"Error: {e}")

InvalidParameterError 

Description: Raised when invalid parameters are provided.
Example: 

try:
    param_manager.save_optimized_parameters("strategy", {}, "invalid_date")
except InvalidParameterError as e:
    print(f"Error: {e}")

TradingEngineError 

Description: Raised when trading engine encounters an error.
Example: 

try:
    paper_trader.start_paper_trading("invalid_account", "strategy", 1000)
except TradingEngineError as e:
    print(f"Error: {e}")

6. Usage Examples 
Complete Trading Setup Example 

import tkinter as tk
from simple_strategy.trading.api_manager import APIManager
from simple_strategy.trading.parameter_manager import ParameterManager
from simple_strategy.trading.paper_trading_engine import PaperTradingEngine

# Initialize managers
api_manager = APIManager()
param_manager = ParameterManager()

# Add demo account
api_manager.add_demo_account(
    "demo_test",
    "your_api_key",
    "your_api_secret",
    "Test demo account"
)

# Save optimized parameters
optimized_params = {
    "rsi_period": 14,
    "sma_short": 20,
    "sma_long": 50,
    "oversold_threshold": 30,
    "overbought_threshold": 70
}
param_manager.save_optimized_parameters(
    "rsi_sma_strategy",
    optimized_params,
    "2025-11-10"
)

# Initialize paper trading engine
paper_trader = PaperTradingEngine(api_manager, param_manager)

# Start paper trading
success = paper_trader.start_paper_trading(
    "demo_test",
    "rsi_sma_strategy",
    1000.0
)

if success:
    print("Paper trading started successfully")
    
    # Monitor trading status
    status = paper_trader.get_trading_status()
    print(f"Current balance: {status['current_balance']}")
    
    # Stop trading after some time
    # paper_trader.stop_paper_trading()
    
    # Get performance metrics
    # metrics = paper_trader.calculate_performance_metrics()
    # print(f"Total return: {metrics['total_return']:.2f}%")

GUI Integration Example

import tkinter as tk
from simple_strategy.trading.api_gui import APIGUI
from simple_strategy.trading.parameter_gui import ParameterGUI

# Create main window
root = tk.Tk()
root.title("AI Assisted TradeBot - Trading Interface")
root.geometry("800x600")

# Create notebook for different interfaces
notebook = tk.ttk.Notebook(root)
notebook.pack(fill=tk.BOTH, expand=True)

# API Management Tab
api_frame = tk.Frame(notebook)
notebook.add(api_frame, text="API Management")
api_gui = APIGUI(api_frame)

# Parameter Management Tab
param_frame = tk.Frame(notebook)
notebook.add(param_frame, text="Parameter Management")
param_gui = ParameterGUI(param_frame)

root.mainloop()


