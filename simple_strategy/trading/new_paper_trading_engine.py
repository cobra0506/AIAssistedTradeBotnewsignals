"""
Paper Trading Engine - Main engine with references to other modules
"""

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from simple_strategy.shared.data_feeder import DataFeeder
from shared_modules.data_collection.config import DataCollectionConfig
MIN_CANDLES = DataCollectionConfig.MIN_CANDLES  # Get from config

# Import our new modules
from .position_manager import PositionManager
from .api_connector import APIConnector

class PaperTradingEngine:
    def __init__(self, api_account, strategy_name, initial_balance=1000, log_callback=None, status_callback=None, performance_callback=None, max_positions=2000, stop_trading_at_percentage=None):
        # Basic initialization
        self.api_account = api_account
        self.strategy_name = strategy_name
        self.initial_balance = float(initial_balance)
        self.working_capital = self.initial_balance
        self.simulated_balance = float(initial_balance)
        self.real_balance = 0.0
        self.balance_offset = 0.0
        
        # Optional trading controls
        self.max_positions = max_positions
        self.stop_trading_at_percentage = stop_trading_at_percentage
        
        # Initialize API connector
        self.api_key = None
        self.api_secret = None
        self.api_connector = None
        
        # Trading state
        self.is_running = False
        self.strategy = None
        
        # Data feeder for strategy integration
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(project_root, 'data')
        self.data_feeder = DataFeeder(data_dir=data_dir)
        
        # GUI callback functions
        self.log_callback = log_callback
        self.status_callback = status_callback
        self.performance_callback = performance_callback
        
        # Shared data access
        self.shared_data_access = None
        self.data_system_initialized = False
        self.trading_loop_active = False
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize API connector and position manager"""
        # Load credentials
        self.load_credentials()
        
        # Initialize API connector
        self.api_connector = APIConnector(
            self.api_key, 
            self.api_secret,
            logger=self.log_message
        )
        
        # Test connection
        self.api_connector.test_connection()
        
        # Get all symbols information
        self.api_connector.get_all_symbols_info()
        
        # Get real balance and calculate offset
        balance_info = self.api_connector.get_real_balance()
        self.real_balance = balance_info['available_balance']
        self.balance_offset = self.real_balance - self.simulated_balance
        
        # Initialize position manager
        self.position_manager = PositionManager(
            self.api_connector,
            self.log_message
        )
        
        # Make sure position manager has access to simulated balance
        self.position_manager.simulated_balance = self.simulated_balance
        
        self.log_message(f"Paper Trading Engine initialized:")
        self.log_message(f"  Account: {self.api_account}")
        self.log_message(f"  Strategy: {self.strategy_name}")
        self.log_message(f"  Initial Simulated Balance: ${self.initial_balance}")
        self.log_message(f"  Real Bybit Available Balance: ${self.real_balance}")
        self.log_message(f"  Balance Offset: ${self.balance_offset}")
        self.log_message(f"  Max Positions: {self.max_positions}")
        if self.stop_trading_at_percentage:
            self.log_message(f"  Stop Trading At: {self.stop_trading_at_percentage}% of initial balance")
    
    # Property to access current_positions from position_manager
    @property
    def current_positions(self):
        """Access current positions from position manager"""
        return self.position_manager.current_positions
    
    # Property to access trades from position_manager
    @property
    def trades(self):
        """Access trades from position manager"""
        return self.position_manager.trades
    
    # Property to access open_trades_count from position_manager
    @property
    def open_trades_count(self):
        """Access open trades count from position manager"""
        return self.position_manager.open_trades_count
    
    # Property to access closed_trades_count from position_manager
    @property
    def closed_trades_count(self):
        """Access closed trades count from position manager"""
        return self.position_manager.closed_trades_count
    
    def load_credentials(self):
        """Load API credentials from file"""
        try:
            api_accounts_file = os.path.join(os.path.dirname(__file__), 'api_accounts.json')
            with open(api_accounts_file, 'r') as f:
                accounts = json.load(f)
            
            # Find the selected account
            for account_type in ['demo_accounts', 'live_accounts']:
                if self.api_account in accounts.get(account_type, {}):
                    account_info = accounts[account_type][self.api_account]
                    self.api_key = account_info['api_key']
                    self.api_secret = account_info['api_secret']
                    self.log_message(f"✅ API credentials loaded for {self.api_account}")
                    return True
            
            self.log_message(f"❌ Account '{self.api_account}' not found")
            return False
            
        except Exception as e:
            self.log_message(f"❌ Error loading API credentials: {e}")
            return False
    
    def initialize_shared_data_access(self):
        """Initialize shared data access safely"""
        try:
            from shared_modules.data_collection.shared_data_access import SharedDataAccess
            self.shared_data_access = SharedDataAccess()
            
            # Check if data collection is running
            if self.shared_data_access.is_data_collection_running():
                self.log_message("✅ Using existing data collection process")
            else:
                self.log_message("⚠️ Data collection not running - will use existing CSV files")
                
            return True
        except Exception as e:
            self.log_message(f"❌ Error initializing shared data access: {e}")
            return False
    
    def log_message(self, message):
        """Log message to both console and GUI if available"""
        print(message)  # Use print instead of self.log_message()
        if self.log_callback:
            self.log_callback(message)
    
    def get_market_data(self, symbol, timeframe, limit=100):
        """Get market data from shared data access"""
        if self.shared_data_access:
            return self.shared_data_access.get_latest_data(symbol, timeframe, limit=limit)
        else:
            # Fallback to empty list if shared data access not available
            self.log_message(f"⚠️ Shared data access not available for {symbol}_{timeframe}")
            return []
    
    def get_balance(self):
        """Get current simulated balance (for compatibility)"""
        return self.get_display_balance()
    
    def get_display_balance(self):
        """Get the simulated balance for display"""
        return self.simulated_balance
    
    def get_all_perpetual_symbols(self):
        """Get all perpetual symbols"""
        try:
            result, error = self.make_request("GET", "/v5/market/instruments-info", params={"category": "linear", "limit": 1000})
            
            if error:
                self.log_message(f"❌ Error getting symbols: {error}")
                return []
            
            symbols = []
            excluded_symbols = ['USDC', 'USDE', 'USTC']
            
            for instrument in result['list']:
                symbol = instrument.get('symbol', '')
                if (not any(excl in symbol for excl in excluded_symbols) and
                    "-" not in symbol and
                    symbol.endswith('USDT') and
                    instrument.get('contractType') == 'LinearPerpetual' and
                    instrument.get('status') == 'Trading'):
                    symbols.append(symbol)
            
            self.log_message(f"✅ Found {len(symbols)} perpetual symbols")
            return sorted(symbols)
                
        except Exception as e:
            self.log_message(f"❌ Error getting symbols: {e}")
            return []
    
    def filter_tradable_symbols(self, all_symbols):
        """Filter symbols to only include tradable ones"""
        tradable_symbols = []
        
        for symbol in all_symbols:
            # Filter out obvious non-tradable symbols
            if any(skip in symbol.upper() for skip in ['1000', '10000', '1000000', 'BABY', 'CHEEMS', 'MOG', 'ELON', 'QUBIC', 'SATS', 'BONK', 'BTT', 'CAT']):
                continue
            
            # Only include major symbols with good liquidity
            if symbol.endswith('USDT') and len(symbol) <= 10:  # Reasonable symbol length
                tradable_symbols.append(symbol)
        
        self.log_message(f"📊 Filtered {len(all_symbols)} symbols down to {len(tradable_symbols)} tradable symbols")
        return tradable_symbols
    
    def should_continue_trading(self):
        """Check if trading should continue based on balance and optional settings"""
        # Check if we've reached the maximum number of open positions
        max_positions = getattr(self, 'max_positions', 20)  # Default to 20 if not set
        if len(self.current_positions) >= max_positions:
            self.log_message(f"🛑 Max positions reached: {len(self.current_positions)} >= {max_positions}")
            return False
        
        # Optional: Check if balance is below a percentage threshold (if enabled)
        if hasattr(self, 'stop_trading_at_percentage') and self.stop_trading_at_percentage:
            min_balance = self.initial_balance * (self.stop_trading_at_percentage / 100)
            if self.simulated_balance < min_balance:
                self.log_message(f"🛑 Balance below threshold: ${self.simulated_balance:.2f} < ${min_balance:.2f} ({self.stop_trading_at_percentage}%)")
                return False
        
        return True
    
    def calculate_position_size(self, symbol):
        """Calculate position size based on working capital"""
        try:
            # Get current price
            current_price = self.get_current_price_from_api(symbol)
            if current_price <= 0:
                return 0.001  # Default fallback
            
            # Use 5% of WORKING capital for position sizing
            position_value = self.get_working_capital() * 0.05
            calculated_quantity = position_value / current_price
            
            # Log what we're doing
            self.log_message(f"📊 Position sizing: 5% of working capital = ${position_value:.2f}, calculated qty = {calculated_quantity:.6f}")
            
            return calculated_quantity
            
        except Exception as e:
            self.log_message(f"❌ Error calculating position size: {e}")
            return 0.001  # Default fallback
    
    def load_strategy(self):
        """Load the selected strategy with optimized parameters"""
        try:
            # First, check for optimized parameters
            from simple_strategy.trading.parameter_manager import ParameterManager
            pm = ParameterManager()
            optimized_params = pm.get_parameters(self.strategy_name)
            
            # Import the strategy registry to get available strategies
            from simple_strategy.strategies.strategy_registry import StrategyRegistry
            registry = StrategyRegistry()
            available_strategies = registry.get_all_strategies()
            
            # Check if the selected strategy exists
            if self.strategy_name not in available_strategies:
                self.log_message(f"Error: Unknown strategy '{self.strategy_name}'")
                self.log_message(f"Available strategies: {list(available_strategies.keys())}")
                return False
            
            # Get strategy info
            strategy_info = available_strategies[self.strategy_name]
            
            # Get default parameters from strategy info
            parameters_def = strategy_info.get('parameters', {})
            default_params = {}
            for param_name, param_info in parameters_def.items():
                default_params[param_name] = param_info.get('default', 0)
            
            # Use optimized parameters if available, otherwise use defaults
            current_params = optimized_params if optimized_params else default_params
            
            # Log which parameters are being used
            if optimized_params:
                self.log_message(f"✅ Using optimized parameters:")
                for param, value in optimized_params.items():
                    if param != 'last_optimized':  # Skip the metadata
                        self.log_message(f"  {param}: {value}")
            else:
                self.log_message("⚠️ Using default parameters (no optimized parameters found)")
            
            # Create the strategy using the create_func from the registry
            if 'create_func' in strategy_info:
                # Get symbols and timeframes - we'll use 1-minute data for paper trading
                symbols = ['BTCUSDT']  # We'll update this per symbol in generate_strategy_signal
                timeframes = ['1m']     # We're using 1-minute data
                
                # Create the strategy
                self.strategy = strategy_info['create_func'](
                    symbols=symbols,
                    timeframes=timeframes,
                    **current_params
                )
                
                return True
            else:
                self.log_message(f"Error: Strategy '{self.strategy_name}' missing create_func")
                return False
                        
        except Exception as e:
            self.log_message(f"Error loading strategy: {e}")
            import traceback
            self.log_message(f"Traceback: {traceback.format_exc()}")
            return False
    
    def generate_trading_signal(self, symbol, current_price):
        """Generate trading signal using the loaded strategy"""
        try:
            # First try to use the loaded strategy
            if self.strategy and hasattr(self.strategy, 'generate_signals'):
                signal = self.generate_strategy_signal(symbol)
                
                # If we get a SELL signal but don't have a position, change to HOLD
                if signal == 'SELL' and symbol not in self.current_positions:
                    self.log_message(f"⚠️ SELL signal for {symbol} but no position open, changing to HOLD")
                    return 'HOLD'
                
                return signal
            
            # Fallback to RSI strategy
            return self.generate_rsi_signal(symbol, current_price)
            
        except Exception as e:
            self.log_message(f"❌ Error generating signal for {symbol}: {e}")
            return 'HOLD'
    
    def generate_strategy_signal(self, symbol):
        """Generate signal using the loaded strategy"""
        try:
            # Get historical data for the symbol
            historical_data = self.get_historical_data_for_symbol(symbol)
            
            # Explicitly check for None or empty DataFrame
            if historical_data is None:
                self.log_message(f"⚠️ No historical data available for {symbol}")
                return 'HOLD'
            
            if isinstance(historical_data, pd.DataFrame) and historical_data.empty:
                self.log_message(f"⚠️ Empty historical data for {symbol}")
                return 'HOLD'
            
            if len(historical_data) < MIN_CANDLES:
                self.log_message(f"⚠️ Not enough historical data for {symbol} (only {len(historical_data)} rows)")
                return 'HOLD'
            
            # Prepare data in the format expected by the strategy
            # The strategy expects: Dict[str, Dict[str, pd.DataFrame]]
            # where the first key is the symbol and the second key is the timeframe
            strategy_data = {
                symbol: {
                    "1m": historical_data  # We're using 1-minute data
                }
            }
            
            # Generate signals using the strategy
            try:
                signals = self.strategy.generate_signals(strategy_data)
                
                # Extract the signal for our symbol and timeframe
                if signals and symbol in signals and "1m" in signals[symbol]:
                    signal = signals[symbol]["1m"]
                    return signal
                else:
                    self.log_message(f"⚠️ No signal returned for {symbol}")
                    return 'HOLD'
                    
            except Exception as e:
                self.log_message(f"❌ Error generating signals: {e}")
                return 'HOLD'
            
        except Exception as e:
            self.log_message(f"❌ Error generating strategy signal for {symbol}: {e}")
            return 'HOLD'
    
    def get_historical_data_for_symbol(self, symbol):
        """Get historical data for a symbol by loading directly from CSV."""
        return self.load_csv_data(symbol)
    
    def load_csv_data(self, symbol):
        """Load data directly from CSV file as fallback"""
        try:
            import pandas as pd
            
            # Construct the correct file path to the project's root 'data' folder
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            csv_file = os.path.join(project_root, 'data', f'{symbol}_1.csv')
            
            if not os.path.exists(csv_file):
                self.log_message(f"⚠️ CSV file not found for {symbol}: {csv_file}")
                return None
            
            # Load CSV file
            df = pd.read_csv(csv_file)
            
            # Ensure we have a DataFrame
            if not isinstance(df, pd.DataFrame):
                self.log_message(f"❌ Loaded data is not a DataFrame for {symbol}")
                return None
            
            # Check if DataFrame is empty
            if df.empty:
                self.log_message(f"⚠️ Empty DataFrame for {symbol}")
                return pd.DataFrame()  # Return empty DataFrame instead of None
            
            # Convert timestamp to datetime if needed
            if 'timestamp' in df.columns and 'datetime' not in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Sort by datetime
            df = df.sort_values('datetime')
            
            # Get last 1000 rows for RSI calculation
            if len(df) > 1000:
                df = df.tail(1000)
            
            self.log_message(f"✅ Loaded {len(df)} rows from CSV for {symbol}")
            return df
            
        except Exception as e:
            self.log_message(f"❌ Error loading CSV for {symbol}: {e}")
            return None
    
    def update_balance_after_trade(self, pnl_amount):
        """Update simulated balance after trade"""
        self.simulated_balance += pnl_amount
        self.log_message(f"💰 Balance updated: ${self.simulated_balance:.2f} (P&L: ${pnl_amount:.2f})")
        # Sync to position manager
        self._sync_balance_to_position_manager()
    
    def get_balance_info(self):
        """Get complete balance information"""
        # Calculate value of open positions
        open_positions_value = 0.0
        
        for symbol, position in self.current_positions.items():
            try:
                current_price = self.get_current_price_from_api(symbol)
                if current_price > 0:
                    open_positions_value += position['quantity'] * current_price
            except Exception as e:
                self.log_message(f"⚠️ Could not get price for {symbol}: {e}")
        
        # Calculate total account value
        account_value = self.simulated_balance + open_positions_value
        
        return {
            'available_balance': self.simulated_balance,  # Cash available for new trades
            'account_value': account_value,  # Cash + open positions
            'open_positions_value': open_positions_value,
            'real_balance': self.real_balance,
            'balance_offset': self.balance_offset,
            'initial_balance': self.initial_balance,
            'total_pnl': self.simulated_balance - self.initial_balance,
            'open_trades_count': self.open_trades_count,
            'closed_trades_count': self.closed_trades_count
        }
    
    # UPDATED: start_trading() with new signal handling
    def start_trading(self):
        """Start paper trading with real API calls"""
        if not self.load_strategy():
            return False
        
        if not self.api_key or not self.api_secret:
            self.log_message("Error: API credentials not loaded")
            return False
        
        # Initialize shared data access
        self.initialize_shared_data_access()
        
        self.is_running = True
        self.log_message(f"Paper trading started for {self.strategy_name}")
        
        # Get symbols and intervals directly from our local data files
        symbols_to_monitor, available_intervals = self.get_symbols_and_intervals_from_data_dir()
        
        # For now, let's just use the 1-minute interval
        if '1' not in available_intervals:
            self.log_message("❌ No 1-minute interval data found. Cannot start.")
            return
            
        self.log_message(f"📈 Monitoring {len(symbols_to_monitor)} symbols on the 1m interval.")
        
        # Main CSV-based trading loop
        loop_count = 0

        while self.is_running:
            loop_count += 1
            self.log_message(f"\n=== Trading Loop #{loop_count} ===")
        
            # Check for position timeouts
            self.position_manager.check_position_timeouts()

            # Process each symbol by reading from the latest CSV data
            for symbol in symbols_to_monitor:
                try:
                    # Get the latest data for the symbol from the CSV file
                    historical_data = self.get_historical_data_for_symbol(symbol)
                    
                    if historical_data is not None and not historical_data.empty:
                        # Generate a trading signal based on this fresh data
                        signal = self.generate_trading_signal(symbol, historical_data.iloc[-1]['close'])
                        
                        # Add this debug logging
                        if signal != 'HOLD':
                            self.log_message(f"🔍 Signal for {symbol}: {signal}")

                        # Handle different signal types
                        if signal == 'OPEN_LONG':
                            # Check if we can open a new long position
                            if self.should_continue_trading():
                                # Check if we don't already have a position for this symbol
                                if symbol not in self.current_positions:
                                    self.position_manager.execute_open_long(symbol)
                                    time.sleep(0.1)
                                else:
                                    self.log_message(f"⚠️ OPEN_LONG signal for {symbol} but position already open, skipping")
                            else:
                                # Log why we're not opening a position
                                if len(self.current_positions) >= self.max_positions:
                                    self.log_message(f"⚠️ Max positions reached ({len(self.current_positions)} >= {self.max_positions}), skipping {symbol}")
                                elif hasattr(self, 'stop_trading_at_percentage') and self.stop_trading_at_percentage:
                                    min_balance = self.initial_balance * (self.stop_trading_at_percentage / 100)
                                    if self.simulated_balance < min_balance:
                                        self.log_message(f"⚠️ Balance below threshold ({self.simulated_balance:.2f} < {min_balance:.2f}), skipping {symbol}")
                        
                        elif signal == 'CLOSE_LONG':
                            # Check if we have a long position to close
                            if symbol in self.current_positions and self.current_positions[symbol]['type'] == 'LONG':
                                self.position_manager.execute_close_long(symbol)
                                time.sleep(0.1)
                            else:
                                self.log_message(f"⚠️ CLOSE_LONG signal for {symbol} but no long position open, skipping")
                        
                        elif signal == 'OPEN_SHORT':
                            # Check if we can open a new short position
                            if self.should_continue_trading():
                                # Check if we don't already have a position for this symbol
                                if symbol not in self.current_positions:
                                    self.position_manager.execute_open_short(symbol)
                                    time.sleep(0.1)
                                else:
                                    self.log_message(f"⚠️ OPEN_SHORT signal for {symbol} but position already open, skipping")
                            else:
                                # Log why we're not opening a position
                                if len(self.current_positions) >= self.max_positions:
                                    self.log_message(f"⚠️ Max positions reached ({len(self.current_positions)} >= {self.max_positions}), skipping {symbol}")
                                elif hasattr(self, 'stop_trading_at_percentage') and self.stop_trading_at_percentage:
                                    min_balance = self.initial_balance * (self.stop_trading_at_percentage / 100)
                                    if self.simulated_balance < min_balance:
                                        self.log_message(f"⚠️ Balance below threshold ({self.simulated_balance:.2f} < {min_balance:.2f}), skipping {symbol}")
                        
                        elif signal == 'CLOSE_SHORT':
                            # Check if we have a short position to close
                            if symbol in self.current_positions and self.current_positions[symbol]['type'] == 'SHORT':
                                self.position_manager.execute_close_short(symbol)
                                time.sleep(0.1)
                            else:
                                self.log_message(f"⚠️ CLOSE_SHORT signal for {symbol} but no short position open, skipping")
                        
                        # No need to log 'HOLD' to keep the log cleaner

                except Exception as e:
                    self.log_message(f"❌ Error processing {symbol}: {e}")
                    continue
            
            # Update performance and wait for the next minute
            self.update_performance_display()
            if self.is_running:
                self.log_message("✅ Cycle complete. Waiting for the next minute...")
                time.sleep(60) # Wait 60 seconds before the next cycle

        self.log_message("🛑 Trading loop ended")
    
    def update_performance(self):
        """Update performance display"""
        try:
            if hasattr(self, 'trading_engine') and self.trading_engine:
                # Get real performance data from trading engine
                performance = self.trading_engine.get_performance_summary()
                
                # Check if performance data is valid
                if performance and 'error' not in performance:
                    initial_balance = performance.get('initial_balance', self.simulated_balance)
                    current_balance = performance.get('current_balance', self.simulated_balance)
                    total_trades = performance.get('total_trades', 0)
                    open_positions = performance.get('open_positions', 0)
                    win_rate = performance.get('win_rate', 0.0)
                    pnl = performance.get('total_pnl', 0.0)
                    
                    # Calculate closed trades (total trades / 2, assuming each complete trade has a buy and sell)
                    closed_trades = total_trades // 2 if total_trades > 0 else 0
                    
                    perf_text = f"""Initial Balance: ${initial_balance:.2f}
                    Current Balance: ${current_balance:.2f}
                    Open Positions: {open_positions}
                    Closed Trades: {closed_trades}
                    Total Trades: {total_trades}
                    Win Rate: {win_rate:.1f}%
                    Profit/Loss: ${pnl:.2f}"""
                else:
                    # Fallback to dummy data if engine not available or error
                    perf_text = f"""Initial Balance: ${self.simulated_balance:.2f}
                    Current Balance: ${self.simulated_balance:.2f}
                    Open Positions: 0
                    Closed Trades: 0
                    Total Trades: 0
                    Win Rate: 0.0%
                    Profit/Loss: $0.00"""
            else:
                # Fallback to dummy data if engine not available
                perf_text = f"""Initial Balance: ${self.simulated_balance:.2f}
                Current Balance: ${self.simulated_balance:.2f}
                Open Positions: 0
                Closed Trades: 0
                Total Trades: 0
                Win Rate: 0.0%
                Profit/Loss: $0.00"""
            
            self.perf_text.delete(1.0, "end")
            self.perf_text.insert(1.0, perf_text)
            
        except Exception as e:
            print(f"Error updating performance: {e}")
            # Show error in performance display
            error_text = f"""Error updating performance: {str(e)}
    Please check the trading log for details."""
            
            self.perf_text.delete(1.0, "end")
            self.perf_text.insert(1.0, error_text)
    
    def get_performance_summary(self):
        """Get a summary of trading performance"""
        try:
            balance_info = self.get_balance_info()
            
            # Calculate win rate from completed (closed) trades
            winning_trades = 0
            for trade in self.trades:
                if trade['type'] == 'SELL' and 'pnl' in trade:
                    if trade['pnl'] > 0:
                        winning_trades += 1
            
            win_rate = (winning_trades / self.closed_trades_count * 100) if self.closed_trades_count > 0 else 0
            
            summary = {
                'initial_balance': balance_info['initial_balance'],
                'current_balance': balance_info['simulated_balance'],
                'available_balance': balance_info['available_balance'],
                'account_value': balance_info['account_value'],
                'open_positions_value': balance_info['open_positions_value'],
                'total_pnl': balance_info['total_pnl'],
                'total_trades': self.closed_trades_count,  # Use closed trades count
                'open_trades': self.open_trades_count,
                'closed_trades': self.closed_trades_count,
                'win_rate': win_rate,
                'open_positions': len(self.current_positions),
                'status': 'Running' if self.is_running else 'Stopped'
            }
            
            return summary
        except Exception as e:
            self.log_message(f"Error generating performance summary: {e}")
            return {
                'error': str(e),
                'status': 'Error'
            }
    
    def stop_trading(self):
        """Stop paper trading"""
        self.is_running = False
        self.trading_loop_active = False
        self.log_message("🛑 Paper trading stopped by user")
    
    # UPDATED: process_symbol() with new signal handling
    def process_symbol(self, symbol):
        """Process a single symbol for trading signals"""
        try:
            # Get current price
            current_price = self.api_connector.get_current_price_from_api(symbol)
            if not current_price or current_price == 0:
                return
            
            self.log_message(f"💰 {symbol}: ${current_price}")
            
            # Get strategy signal
            signal = self.generate_trading_signal(symbol, current_price)
            
            if signal == 'OPEN_LONG':
                # Check if we can open a new long position
                if self.should_continue_trading():
                    # Check if we don't already have a position for this symbol
                    if symbol not in self.current_positions:
                        self.position_manager.execute_open_long(symbol)
                    else:
                        self.log_message(f"⚠️ OPEN_LONG signal for {symbol} but position already open, skipping")
                else:
                    self.log_message(f"⚠️ Cannot open long position for {symbol}: Trading conditions not met")
            
            elif signal == 'CLOSE_LONG':
                # Check if we have a long position to close
                if symbol in self.current_positions and self.current_positions[symbol]['type'] == 'LONG':
                    self.position_manager.execute_close_long(symbol)
                else:
                    self.log_message(f"⚠️ CLOSE_LONG signal for {symbol} but no long position open, skipping")
            
            elif signal == 'OPEN_SHORT':
                # Check if we can open a new short position
                if self.should_continue_trading():
                    # Check if we don't already have a position for this symbol
                    if symbol not in self.current_positions:
                        self.position_manager.execute_open_short(symbol)
                    else:
                        self.log_message(f"⚠️ OPEN_SHORT signal for {symbol} but position already open, skipping")
                else:
                    self.log_message(f"⚠️ Cannot open short position for {symbol}: Trading conditions not met")
            
            elif signal == 'CLOSE_SHORT':
                # Check if we have a short position to close
                if symbol in self.current_positions and self.current_positions[symbol]['type'] == 'SHORT':
                    self.position_manager.execute_close_short(symbol)
                else:
                    self.log_message(f"⚠️ CLOSE_SHORT signal for {symbol} but no short position open, skipping")
            
            else:
                self.log_message(f"📊 Signal for {symbol}: HOLD")
                
        except Exception as e:
            self.log_message(f"❌ Error processing {symbol}: {e}")
    
    def update_performance_display(self):
        """Update performance display with new metrics"""
        try:
            # 1. Get current real-time balances from Bybit
            real_balances = self.api_connector.get_real_balance()  # CHANGED: Added self.api_connector.
            real_available_balance = real_balances['available_balance']
            real_margin_balance = real_balances['margin_balance']

            # 2. Calculate the simulated metrics
            # Available Balance is the spendable cash
            simulated_available_balance = self.simulated_balance
            
            # Account Value is the total worth (cash + unrealized P&L)
            simulated_account_value = real_margin_balance - self.balance_offset

            # Total Trades is the number of completed trades (each SELL is a completed trade)
            total_trades = sum(1 for trade in self.trades if trade['type'] == 'SELL')

            # 3. Calculate Win Rate (based on completed trades)
            winning_trades = sum(1 for trade in self.trades if trade['type'] == 'SELL' and trade.get('pnl', 0) > 0)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
            
            # 4. Calculate Profit/Loss based on the new Account Value
            pnl = simulated_account_value - self.initial_balance

            # Log performance for debugging
            self.log_message(f"Performance Update: Account Value=${simulated_account_value:.2f}, Available Balance=${simulated_available_balance:.2f}, P&L=${pnl:.2f}")
            self.log_message(f"Open positions: {len(self.current_positions)}, Total Trades: {total_trades}")
            
            # 5. Update GUI if available
            if self.performance_callback:
                performance_data = {
                    'account_value': simulated_account_value,  # New metric
                    'available_balance': simulated_available_balance, # Renamed from 'balance'
                    'pnl': pnl, # P&L calculated from the new account value
                    'win_rate': win_rate,
                    'open_positions': len(self.current_positions),
                    'total_trades': total_trades, # Renamed from 'closed_trades'
                    'closed_trades': total_trades # Keep for compatibility for now
                }
                self.performance_callback(performance_data)
                
        except Exception as e:
            self.log_message(f"❌ Error updating performance: {e}")
    
    def get_symbols_and_intervals_from_data_dir(self):
        """
        Scans the data directory to find all available symbols and intervals.
        This is a fast, reliable way to know what data we have locally.
        """
        start_time = time.time()
        
        # Get the absolute path to the project's root 'data' folder
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(project_root, 'data')

        symbols = set()
        intervals = set()
        file_count = 0

        self.log_message(f"🔍 Scanning for symbols in {data_dir}...")

        try:
            for filename in os.listdir(data_dir):
                if filename.endswith('.csv'):
                    # Expected format: SYMBOL_INTERVAL.csv (e.g., BTCUSDT_1.csv)
                    parts = filename[:-4].split('_') # Remove .csv and split by _
                    if len(parts) == 2:
                        symbol = parts[0]
                        interval = parts[1]
                        symbols.add(symbol)
                        intervals.add(interval)
                        file_count += 1
        except FileNotFoundError:
            self.log_message(f"❌ Error: Data directory not found at {data_dir}")
            return [], []

        end_time = time.time()
        duration = end_time - start_time

        # Log the statistics
        self.log_message(f"✅ Scan complete in {duration:.2f} seconds.")
        self.log_message(f"📊 Found {file_count} data files.")
        self.log_message(f"📈 Found {len(symbols)} unique symbols: {', '.join(list(symbols)[:10])}...")
        self.log_message(f"⏱️ Found {len(intervals)} unique intervals: {', '.join(sorted(list(intervals)))}")

        return sorted(list(symbols)), sorted(list(intervals))
    
    def _sync_balance_to_position_manager(self):
        """Sync the current simulated balance to the position manager"""
        if hasattr(self, 'position_manager'):
            self.position_manager.update_simulated_balance(self.simulated_balance)