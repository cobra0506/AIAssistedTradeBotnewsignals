#paper_trading_engine.py

import os
import json
import time
import requests
import hmac
import hashlib
import numpy as np
import pandas as pd
import time
from urllib.parse import urlencode
from datetime import datetime
from datetime import timedelta
from simple_strategy.shared.data_feeder import DataFeeder
from shared_modules.data_collection.config import DataCollectionConfig
MIN_CANDLES = DataCollectionConfig.MIN_CANDLES  # Get from config
# Removed unused imports to avoid circular dependencies
import asyncio
#from shared_modules.data_collection.hybrid_system import HybridTradingSystem
#from shared_modules.data_collection.config import DataCollectionConfig

class PaperTradingEngine:
    def __init__(self, api_account, strategy_name, initial_balance=1000, log_callback=None, status_callback=None, performance_callback=None, max_positions=2000, stop_trading_at_percentage=None):
        self.api_account = api_account
        self.strategy_name = strategy_name
        self.initial_balance = float(initial_balance)  # This is your working capital
        self.working_capital = self.initial_balance  # Track working capital separately
        self.simulated_balance = float(initial_balance)
        self.real_balance = 0.0
        self.balance_offset = 0.0
        
        # Optional trading controls
        self.max_positions = max_positions
        self.stop_trading_at_percentage = stop_trading_at_percentage  # None or percentage (e.g., 10 for 10%)
        
        # API configuration - using the EXACT working configuration
        self.api_key = None
        self.api_secret = None
        self.base_url = "https://api-demo.bybit.com"
        self.recv_window = "5000"
        
       # Trading state
        self.is_running = False
        self.trades = []
        self.current_positions = {}
        self.open_trades_count = 0  # Count of open trades
        self.closed_trades_count = 0  # Count of closed trades
        self.strategy = None
        
        # Data feeder for strategy integration (keep for compatibility)
        # Get the absolute path to the project's root 'data' folder
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(project_root, 'data')
        self.data_feeder = DataFeeder(data_dir=data_dir)
        
        # GUI callback functions
        self.log_callback = log_callback
        self.status_callback = status_callback
        self.performance_callback = performance_callback
        
        # NEW: Use shared data access instead of creating new data collection
        # Initialize this later to avoid recursion issues
        self.shared_data_access = None

        # Add these missing attributes
        self.data_system_initialized = False
        self.is_running = False
        self.trading_loop_active = False

        # Initialize symbol information cache
        self.symbol_info_cache = {}
        
        # Load credentials and test connection
        self.load_credentials()
        self.test_connection()
        
        # Get all symbols information
        self.get_all_symbols_info()
        
        # Sync any open positions from exchange (in case of restart)
        self._sync_positions_from_exchange()
        
        # Get real balance and calculate offset AFTER connecting
        balance_info = self.get_real_balance() # This now returns a dictionary
        self.real_balance = balance_info['available_balance'] # Use the available balance for the offset
        self.balance_offset = self.real_balance - self.simulated_balance
        
        self.log_message(f"Paper Trading Engine initialized:")
        self.log_message(f"  Account: {api_account}")
        self.log_message(f"  Strategy: {strategy_name}")
        self.log_message(f"  Initial Simulated Balance: ${self.initial_balance}")
        self.log_message(f"  Real Bybit Available Balance: ${self.real_balance}")
        self.log_message(f"  Balance Offset: ${self.balance_offset}")
        self.log_message(f"  Max Positions: {self.max_positions}")
        if self.stop_trading_at_percentage:
            self.log_message(f"  Stop Trading At: {self.stop_trading_at_percentage}% of initial balance")
            
    def get_all_symbols_info(self):
        """Get information for all symbols at once and cache it"""
        try:
            result, error = self.make_request("GET", "/v5/market/instruments-info", 
                                            params={"category": "linear", "limit": 1000})
            
            if error:
                self.log_message(f"❌ Error getting symbols info: {error}")
                return False
            
            if result and 'list' in result and result['list']:
                # Cache all symbol information
                self.symbol_info_cache = {}
                for symbol_info in result['list']:
                    symbol = symbol_info.get('symbol', '')
                    if symbol:
                        self.symbol_info_cache[symbol] = symbol_info
                
                self.log_message(f"✅ Cached information for {len(self.symbol_info_cache)} symbols")
                return True
            else:
                self.log_message("❌ No symbol information found")
                return False
                
        except Exception as e:
            self.log_message(f"❌ Error getting symbols info: {e}")
            return False

    def get_trading_rules(self, symbol):
        """Get trading rules for a specific symbol"""
        if symbol not in self.symbol_info_cache:
            self.log_message(f"❌ Symbol {symbol} not found in cache")
            return None
        
        symbol_info = self.symbol_info_cache[symbol]
        
        # Extract trading rules from lotSizeFilter and priceFilter
        lot_size_filter = symbol_info.get('lotSizeFilter', {})
        price_filter = symbol_info.get('priceFilter', {})
        
        rules = {
            'min_order_qty': float(lot_size_filter.get('minOrderQty', '0')),
            'max_order_qty': float(lot_size_filter.get('maxOrderQty', '0')),
            'qty_step': float(lot_size_filter.get('qtyStep', '0')),
            'min_notional_value': float(lot_size_filter.get('minNotionalValue', '0')),
            'max_mkt_order_qty': float(lot_size_filter.get('maxMktOrderQty', '0')),
            'price_tick': float(price_filter.get('tickSize', '0'))
        }
        
        return rules

    def format_quantity(self, quantity, qty_step):
        """Format quantity according to the symbol's requirements"""
        if qty_step <= 0:
            return quantity
        
        # Calculate the number of decimal places needed
        decimal_places = 0
        step_str = f"{qty_step:.10f}".rstrip('0').rstrip('.')
        if '.' in step_str:
            decimal_places = len(step_str.split('.')[1])
        
        # Round to the correct number of decimal places
        return round(quantity, decimal_places)

    def calculate_valid_quantity(self, symbol, current_price, position_value):
        """Calculate a valid quantity for a symbol based on trading rules"""
        # Get trading rules for the symbol
        rules = self.get_trading_rules(symbol)
        if not rules:
            self.log_message(f"❌ No trading rules found for {symbol}")
            return None
        
        # Calculate the raw quantity
        raw_quantity = position_value / current_price
        
        # Ensure the quantity meets the minimum notional value
        min_qty_for_value = rules['min_notional_value'] / current_price
        if raw_quantity < min_qty_for_value:
            raw_quantity = min_qty_for_value
        
        # Round to the correct quantity step
        steps = raw_quantity / rules['qty_step']
        rounded_quantity = round(steps) * rules['qty_step']
        
        # Ensure the quantity is within the valid range
        if rounded_quantity < rules['min_order_qty']:
            rounded_quantity = rules['min_order_qty']
        
        if rounded_quantity > rules['max_order_qty']:
            rounded_quantity = rules['max_order_qty']
        
        # Format the quantity correctly
        formatted_quantity = self.format_quantity(rounded_quantity, rules['qty_step'])
        
        self.log_message(f"📊 Position sizing for {symbol}:")
        self.log_message(f"   Desired value: ${position_value:.2f}")
        self.log_message(f"   Current price: ${current_price:.6f}")
        self.log_message(f"   Raw quantity: {raw_quantity:.6f}")
        self.log_message(f"   Min qty for value: {min_qty_for_value:.6f}")
        self.log_message(f"   Qty step: {rules['qty_step']}")
        self.log_message(f"   Rounded quantity: {rounded_quantity:.6f}")
        self.log_message(f"   Formatted quantity: {formatted_quantity:.6f}")
        self.log_message(f"   Final order value: ${formatted_quantity * current_price:.2f}")
        
        return formatted_quantity

    def set_leverage(self, symbol):
        """Set appropriate leverage for a symbol"""
        try:
            # Get trading rules for the symbol
            rules = self.get_trading_rules(symbol)
            if not rules:
                return False
            
            # Get max leverage from leverage filter
            symbol_info = self.symbol_info_cache.get(symbol, {})
            leverage_filter = symbol_info.get('leverageFilter', {})
            max_leverage = float(leverage_filter.get('maxLeverage', 20))
            
            # Use a reasonable leverage (e.g., 5x or max allowed, whichever is lower)
            leverage = min(5.0, max_leverage)
            
            # Set leverage
            leverage_data = {
                "category": "linear",
                "symbol": symbol,
                "buyLeverage": str(leverage),
                "sellLeverage": str(leverage)
            }
            
            result, error = self.make_request("POST", "/v5/position/set-leverage", data=leverage_data)
            
            if error:
                self.log_message(f"⚠️ Could not set leverage for {symbol}: {error}")
                return False
            
            self.log_message(f"✅ Leverage set to {leverage}x for {symbol}")
            return True
            
        except Exception as e:
            self.log_message(f"❌ Error setting leverage for {symbol}: {e}")
            return False

    def get_working_capital(self):
        """Get current working capital for position sizing"""
        # Calculate the value of all open positions
        open_positions_value = 0.0
        for symbol, position in self.current_positions.items():
            try:
                current_price = self.get_current_price_from_api(symbol)
                if current_price > 0:
                    position_value = position['quantity'] * current_price
                    open_positions_value += position_value
            except Exception as e:
                self.log_message(f"❌ Error calculating position value for {symbol}: {e}")
        
        # Available capital = simulated balance - value of open positions
        available_capital = self.simulated_balance - open_positions_value
        
        self.log_message(f"💰 Capital Analysis:")
        self.log_message(f"   Simulated Balance: ${self.simulated_balance:.2f}")
        self.log_message(f"   Open Positions Value: ${open_positions_value:.2f}")
        self.log_message(f"   Available Capital: ${available_capital:.2f}")
        
        return available_capital
    
    def check_position_timeouts(self):
        """Check for positions that have been open too long and close them"""
        if not self.current_positions:
            return
        
        # Set timeout duration (e.g., 4 hours in seconds)
        timeout_duration = 4 * 60 * 60  # 4 hours
        
        current_time = datetime.now()
        positions_to_close = []
        
        for symbol, position in self.current_positions.items():
            entry_time = datetime.fromisoformat(position['entry_time'])
            time_open = (current_time - entry_time).total_seconds()
            
            if time_open > timeout_duration:
                positions_to_close.append(symbol)
                self.log_message(f"⏰ Position {symbol} timed out after {time_open/3600:.1f} hours")
        
        # Close timed out positions based on their type
        for symbol in positions_to_close:
            if self.current_positions[symbol]['type'] == 'LONG':
                self.execute_close_long(symbol)
            elif self.current_positions[symbol]['type'] == 'SHORT':
                self.execute_close_short(symbol)

    def update_working_capital_after_trade(self, trade_pnl):
        """Update working capital after a trade"""
        # Adjust working capital by the P&L of the trade
        self.working_capital += trade_pnl

    def log_message(self, message):
        """Log message to both console and GUI if available"""
        print(message)  # Use print instead of self.log_message()
        if self.log_callback:
            self.log_callback(message)

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
    
    def generate_signature(self, timestamp, method, path, body='', params=None):
        """Generate HMAC-SHA256 signature - EXACT working method"""
        if method == "GET" and params:
            sorted_params = sorted(params.items())
            query_string = urlencode(sorted_params)
            param_str = timestamp + self.api_key + self.recv_window + query_string
        elif method == "POST" and body:
            if isinstance(body, dict):
                import json
                body_str = json.dumps(body, separators=(',', ':'), sort_keys=True)
                param_str = timestamp + self.api_key + self.recv_window + body_str
            else:
                param_str = timestamp + self.api_key + self.recv_window + str(body)
        else:
            param_str = timestamp + self.api_key + self.recv_window + str(body)
        
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def make_request(self, method, path, params=None, data=None):
        """Make authenticated request - EXACT working method"""
        try:
            # Handle query parameters
            if params:
                query_string = urlencode(params)
                url = f"{self.base_url}{path}?{query_string}"
            else:
                url = f"{self.base_url}{path}"
            
            headers = {"Content-Type": "application/json"}
            
            # Add authentication
            timestamp = str(int(time.time() * 1000))
            signature = self.generate_signature(timestamp, method, path, body=data, params=params)
            
            headers.update({
                "X-BAPI-API-KEY": self.api_key,
                "X-BAPI-TIMESTAMP": timestamp,
                "X-BAPI-RECV-WINDOW": self.recv_window,
                "X-BAPI-SIGN": signature
            })
            
            # Make request
            if method == "GET":
                response = requests.get(url, headers=headers)
            elif method == "POST":
                if isinstance(data, dict):
                    import json
                    body_str = json.dumps(data, separators=(',', ':'), sort_keys=True)
                    response = requests.post(url, headers=headers, data=body_str)
                else:
                    response = requests.post(url, headers=headers, json=data)
            
            result = response.json()
            
            if response.status_code == 200 and result.get('retCode') == 0:
                return result['result'], None
            else:
                error_msg = result.get('retMsg', 'Unknown error')
                return None, error_msg
                
        except Exception as e:
            return None, str(e)
        
    def get_market_data(self, symbol, timeframe, limit=100):
        """Get market data from shared data access"""
        if self.shared_data_access:
            return self.shared_data_access.get_latest_data(symbol, timeframe, limit=limit)
        else:
            # Fallback to empty list if shared data access not available
            self.log_message(f"⚠️ Shared data access not available for {symbol}_{timeframe}")
            return []
    
    def test_connection(self):
        """Test the connection - EXACT working method"""
        try:
            self.log_message("Testing connection...")
            result, error = self.make_request("GET", "/v5/account/wallet-balance", params={"accountType": "UNIFIED"})
            
            if error:
                self.log_message(f"❌ Connection test failed: {error}")
                return False
            
            if result and 'list' in result and result['list']:
                wallet_data = result['list'][0]
                balance = float(wallet_data.get('totalAvailableBalance', '0'))
                self.log_message(f"✅ Connection successful! Balance: ${balance}")
                return True
            else:
                self.log_message("❌ Connection test failed: Invalid response format")
                return False
                
        except Exception as e:
            self.log_message(f"❌ Connection test error: {e}")
            return False
    
    def get_balance(self):
        """Get current simulated balance (for compatibility)"""
        return self.get_display_balance()

    def get_real_balance(self):
        """Get actual available and margin balances from Bybit"""
        try:
            result, error = self.make_request("GET", "/v5/account/wallet-balance", params={"accountType": "UNIFIED"})
            if error:
                self.log_message(f"❌ Error getting real balance: {error}")
                return {'available_balance': 0.0, 'margin_balance': 0.0}
            
            if result and 'list' in result and result['list']:
                wallet_data = result['list'][0]
                # Bybit API provides these fields
                available_balance = float(wallet_data.get('totalAvailableBalance', '0'))
                margin_balance = float(wallet_data.get('totalMarginBalance', '0'))
                
                return {
                    'available_balance': available_balance,
                    'margin_balance': margin_balance
                }
            else:
                return {'available_balance': 0.0, 'margin_balance': 0.0}
                
        except Exception as e:
            self.log_message(f"❌ Error getting real balance: {e}")
            return {'available_balance': 0.0, 'margin_balance': 0.0}

    def get_display_balance(self):
        """Get the simulated balance for display"""
        return self.simulated_balance
    
    def _sync_positions_from_exchange(self):
        """
        Syncs self.current_positions with the real exchange to handle restarts.
        If the bot crashed and restarted, this prevents opening duplicate positions.
        """
        self.log_message("🔄 Syncing positions from exchange...")
        
        try:
            result, error = self.make_request("GET", "/v5/position/list", params={"category": "linear"})
            
            if error:
                self.log_message(f"⚠️ Could not sync positions: {error}")
                return

            if result and 'list' in result:
                synced_count = 0
                for pos in result['list']:
                    # Only process positions that actually have size
                    size = float(pos.get('size', 0))
                    if size <= 0:
                        continue

                    symbol = pos.get('symbol')
                    side = pos.get('side') # 'Buy' or 'Sell'
                    entry_price = float(pos.get('avgPrice', 0))
                    
                    if entry_price == 0:
                        continue

                    # Determine position type based on side
                    pos_type = 'LONG' if side == 'Buy' else 'SHORT'
                    
                    # Estimate cost and margin (We don't have the exact history from startup)
                    cost = size * entry_price
                    
                    # Get leverage from cached info to estimate margin
                    symbol_info = self.symbol_info_cache.get(symbol, {})
                    leverage_filter = symbol_info.get('leverageFilter', {})
                    leverage = float(leverage_filter.get('maxLeverage', 5))
                    margin_used = cost / leverage

                    # Update internal state
                    self.current_positions[symbol] = {
                        'type': pos_type,
                        'quantity': size,
                        'order_id': 'Recovered', # No order ID available
                        'entry_time': datetime.now().isoformat(),
                        'entry_price': entry_price,
                        'cost': cost,
                        'margin_used': margin_used
                    }
                    
                    synced_count += 1
                    self.log_message(f"✅ Recovered {pos_type} position for {symbol} (Size: {size})")

                if synced_count > 0:
                    self.open_trades_count = len(self.current_positions)
                    self.log_message(f"✅ Sync complete. Recovered {synced_count} open positions.")
                else:
                    self.log_message("✅ Sync complete. No open positions found on exchange.")
                    
        except Exception as e:
            self.log_message(f"❌ Error syncing positions: {e}")
    
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
    
    def execute_open_long(self, symbol, quantity=None):
        """Execute a long position opening"""
        try:
            # Add this logging at the beginning
            self.log_message(f"🔍 DEBUG: Current simulated balance before opening LONG: ${self.simulated_balance:.2f}")
            
            # Set leverage before placing the order
            self.set_leverage(symbol)

            # Get current price
            current_price = self.get_current_price_from_api(symbol)
            if current_price <= 0:
                self.log_message(f"❌ Could not get current price for {symbol}")
                return None
            
            # Calculate minimum position value based on trading rules
            rules = self.get_trading_rules(symbol)
            if not rules:
                self.log_message(f"❌ No trading rules found for {symbol}")
                return None
            
            # Calculate minimum required capital for this symbol
            min_qty = rules['min_order_qty']
            min_position_value = min_qty * current_price
            
            # Add this logging
            self.log_message(f"🔍 DEBUG: Min position value: ${min_position_value:.2f}")
            
            # Check if we have enough simulated balance for this position
            if self.simulated_balance < min_position_value:
                self.log_message(f"❌ Insufficient balance for {symbol}: Need ${min_position_value:.2f}, Have ${self.simulated_balance:.2f}")
                return None
            
            # Use 5% of simulated balance for position sizing
            position_value = self.simulated_balance * 0.05
            
            # Add this logging
            self.log_message(f"🔍 DEBUG: Calculated position value (5% of simulated balance): ${position_value:.2f}")
            
            # Ensure we meet minimum position requirements
            position_value = max(position_value, min_position_value)
            
            # Add this logging
            self.log_message(f"🔍 DEBUG: Final position value after min check: ${position_value:.2f}")
            
            # Calculate a valid quantity based on trading rules
            final_quantity = self.calculate_valid_quantity(symbol, current_price, position_value)
            if final_quantity is None:
                self.log_message(f"❌ Could not calculate valid quantity for {symbol}")
                return None
            
            # Final check: ensure we can afford this position
            actual_position_cost = final_quantity * current_price
            
            # Add this logging
            self.log_message(f"🔍 DEBUG: Final position cost: ${actual_position_cost:.2f}")
            
            # Get leverage for this symbol
            symbol_info = self.symbol_info_cache.get(symbol, {})
            leverage_filter = symbol_info.get('leverageFilter', {})
            leverage = float(leverage_filter.get('maxLeverage', 5))
            
            # Calculate margin needed (position value ÷ leverage)
            margin_needed = actual_position_cost / leverage
            
            # Add this logging
            self.log_message(f"🔍 DEBUG: Position value: ${actual_position_cost:.2f}, Leverage: {leverage}x, Margin needed: ${margin_needed:.2f}")
            
            if margin_needed > self.simulated_balance:
                self.log_message(f"❌ Insufficient margin for {symbol}: Need ${margin_needed:.2f}, Have ${self.simulated_balance:.2f}")
                return None
            
            # Create order data with properly formatted quantity
            order_data = {
                "category": "linear",
                "symbol": symbol,
                "side": "Buy",
                "orderType": "Market",
                "qty": str(final_quantity),
                "timeInForce": "GTC"
            }
            
            self.log_message(f"📈 Placing BUY order for {final_quantity} {symbol} (value: ${actual_position_cost:.2f}, margin: ${margin_needed:.2f})...")
            result, error = self.make_request("POST", "/v5/order/create", data=order_data)
            
            if error:
                self.log_message(f"❌ Buy order failed: {error}")
                return None
            
            # FIX: Update simulated balance directly with margin cost, not based on real balance
            old_simulated_balance = self.simulated_balance
            self.simulated_balance -= margin_needed
            
            # Add this logging
            self.log_message(f"🔍 DEBUG: Old simulated balance: ${old_simulated_balance:.2f}")
            self.log_message(f"🔍 DEBUG: New simulated balance after buy: ${self.simulated_balance:.2f}")
            self.log_message(f"🔍 DEBUG: Margin deducted: ${margin_needed:.2f}")
            
            self.log_message(f"💰 Simulated balance after buy: ${self.simulated_balance:.2f} (margin used: ${margin_needed:.2f})")
            
            # Record the trade
            trade = {
                'timestamp': datetime.now().isoformat(),
                'type': 'OPEN_LONG',
                'symbol': symbol,
                'quantity': final_quantity,
                'order_id': result.get('orderId'),
                'status': result.get('orderStatus', 'Unknown'),
                'balance_before': old_simulated_balance,
                'balance_after': self.simulated_balance,
                'pnl': -margin_needed,  # Negative because it's a cost
                'position_value': actual_position_cost,
                'margin_used': margin_needed
            }

            self.log_trade_to_csv(trade)
            
            self.trades.append(trade)
            self.current_positions[symbol] = {
                'type': 'LONG',  # New field
                'quantity': final_quantity,
                'order_id': result.get('orderId'),
                'entry_time': datetime.now().isoformat(),
                'entry_price': current_price,
                'cost': actual_position_cost,
                'margin_used': margin_needed
            }
            
            # Update working capital after the trade
            self.update_working_capital_after_trade(-margin_needed)
            
            # Update GUI if available
            if self.performance_callback:
                self.update_performance_display()
            
            # Update counters
            self.open_trades_count += 1
            
            self.log_message(f"✅ Buy order successful! Order ID: {result.get('orderId')}")
            return trade
            
        except Exception as e:
            self.log_message(f"❌ Error executing buy order: {e}")
            return None

    def execute_close_long(self, symbol, quantity=None):
        """Execute a long position closing"""
        if symbol not in self.current_positions:
            self.log_message(f"❌ No position found for {symbol}")
            return None
        
        # Check if it's a long position
        if self.current_positions[symbol]['type'] != 'LONG':
            self.log_message(f"❌ Cannot close LONG position for {symbol}: Current position is SHORT")
            return None
        
        if quantity is None:
            quantity = self.current_positions[symbol]['quantity']
        
        try:
            # Get current price
            current_price = self.get_current_price_from_api(symbol)
            if current_price <= 0:
                self.log_message(f"❌ Could not get current price for {symbol}")
                return None
            
            # Get trading rules for symbol
            rules = self.get_trading_rules(symbol)
            if not rules:
                self.log_message(f"❌ No trading rules found for {symbol}")
                return None
            
            # Format the quantity according to the symbol's requirements
            final_quantity = self.format_quantity(quantity, rules['qty_step'])
            
            # Create order data with properly formatted quantity
            order_data = {
                "category": "linear",
                "symbol": symbol,
                "side": "Sell",
                "orderType": "Market",
                "qty": str(final_quantity),
                "timeInForce": "GTC"
            }
            
            self.log_message(f"📉 Placing SELL order for {final_quantity} {symbol}...")
            result, error = self.make_request("POST", "/v5/order/create", data=order_data)
            
            if error:
                self.log_message(f"❌ Sell order failed: {error}")
                return None
            
            # Get current real balance BEFORE closing position
            old_real_balance = self.real_balance
            
            # Calculate position P&L
            entry_price = self.current_positions[symbol]['entry_price']
            position_quantity = self.current_positions[symbol]['quantity']
            position_value = position_quantity * current_price
            original_cost = self.current_positions[symbol]['cost']
            margin_used = self.current_positions[symbol]['margin_used']
            
            # Calculate P&L based on position value change (for logging only)
            trade_pnl = position_value - original_cost
            
            # Add this logging
            self.log_message(f"🔍 DEBUG: Position value: ${position_value:.2f}, Original cost: ${original_cost:.2f}")
            self.log_message(f"🔍 DEBUG: Calculated P&L: ${trade_pnl:.2f}")
            
            # FIX: Get REAL Bybit balance AFTER closing position
            new_balance_info = self.get_real_balance()
            new_real_balance = new_balance_info['available_balance']
            
            # Calculate REAL P&L from Bybit (includes all fees, slippage, funding rates)
            real_pnl = new_real_balance - old_real_balance
            
            # Update simulated balance with REAL P&L
            old_simulated_balance = self.simulated_balance
            self.simulated_balance += real_pnl
            self.real_balance = new_real_balance
            
            # Add this logging
            self.log_message(f"🔍 DEBUG: Old simulated balance: ${old_simulated_balance:.2f}")
            self.log_message(f"🔍 DEBUG: New simulated balance after sell: ${self.simulated_balance:.2f}")
            self.log_message(f"🔍 DEBUG: REAL P&L from Bybit: ${real_pnl:.2f}")
            self.log_message(f"🔍 DEBUG: Calculated vs Real P&L diff: ${abs(trade_pnl - real_pnl):.2f}")
            
            # Calculate position duration in minutes
            entry_time = datetime.fromisoformat(self.current_positions[symbol]['entry_time'])
            exit_time = datetime.now()
            position_duration = (exit_time - entry_time).total_seconds() / 60  # Convert to minutes
            
            # Record the trade with P&L and position duration
            trade = {
                'timestamp': datetime.now().isoformat(),
                'type': 'CLOSE_LONG',
                'symbol': symbol,
                'quantity': final_quantity,
                'order_id': result.get('orderId'),
                'status': result.get('orderStatus', 'Unknown'),
                'balance_before': old_simulated_balance,
                'balance_after': self.simulated_balance,
                'pnl': trade_pnl,
                'position_duration': position_duration,
                'position_value': position_value,
                'original_cost': original_cost,
                'margin_returned': margin_used
            }

            self.log_trade_to_csv(trade)
            
            self.trades.append(trade)
            del self.current_positions[symbol]
            
            # Update working capital after the trade
            self.update_working_capital_after_trade(trade_pnl + margin_used)
            
            # Update GUI if available
            if self.performance_callback:
                self.update_performance_display()
            
            # Update counters
            self.open_trades_count -= 1
            self.closed_trades_count += 1
            
            self.log_message(f"💰 Simulated balance after sell: ${self.simulated_balance:.2f} (P&L: ${trade_pnl:.2f}, margin returned: ${margin_used:.2f})")
            self.log_message(f"✅ Sell order successful! Order ID: {result.get('orderId')}")
            return trade
            
        except Exception as e:
            self.log_message(f"❌ Error executing sell order: {e}")
            return None

    def execute_open_short(self, symbol, quantity=None):
        """Execute a short position opening"""
        try:
            # Add this logging at the beginning
            self.log_message(f"🔍 DEBUG: Current simulated balance before opening SHORT: ${self.simulated_balance:.2f}")
            
            # Set leverage before placing the order
            self.set_leverage(symbol)

            # Get current price
            current_price = self.get_current_price_from_api(symbol)
            if current_price <= 0:
                self.log_message(f"❌ Could not get current price for {symbol}")
                return None
            
            # Calculate minimum position value based on trading rules
            rules = self.get_trading_rules(symbol)
            if not rules:
                self.log_message(f"❌ No trading rules found for {symbol}")
                return None
            
            # Calculate minimum required capital for this symbol
            min_qty = rules['min_order_qty']
            min_position_value = min_qty * current_price
            
            # Add this logging
            self.log_message(f"🔍 DEBUG: Min position value: ${min_position_value:.2f}")
            
            # Check if we have enough simulated balance for this position
            if self.simulated_balance < min_position_value:
                self.log_message(f"❌ Insufficient balance for {symbol}: Need ${min_position_value:.2f}, Have ${self.simulated_balance:.2f}")
                return None
            
            # Use 5% of simulated balance for position sizing
            position_value = self.simulated_balance * 0.05
            
            # Add this logging
            self.log_message(f"🔍 DEBUG: Calculated position value (5% of simulated balance): ${position_value:.2f}")
            
            # Ensure we meet minimum position requirements
            position_value = max(position_value, min_position_value)
            
            # Add this logging
            self.log_message(f"🔍 DEBUG: Final position value after min check: ${position_value:.2f}")
            
            # Calculate a valid quantity based on trading rules
            final_quantity = self.calculate_valid_quantity(symbol, current_price, position_value)
            if final_quantity is None:
                self.log_message(f"❌ Could not calculate valid quantity for {symbol}")
                return None
            
            # Final check: ensure we can afford this position
            actual_position_cost = final_quantity * current_price
            
            # Add this logging
            self.log_message(f"🔍 DEBUG: Final position cost: ${actual_position_cost:.2f}")
            
            # Get leverage for this symbol
            symbol_info = self.symbol_info_cache.get(symbol, {})
            leverage_filter = symbol_info.get('leverageFilter', {})
            leverage = float(leverage_filter.get('maxLeverage', 5))
            
            # Calculate margin needed (position value ÷ leverage)
            margin_needed = actual_position_cost / leverage
            
            # Add this logging
            self.log_message(f"🔍 DEBUG: Position value: ${actual_position_cost:.2f}, Leverage: {leverage}x, Margin needed: ${margin_needed:.2f}")
            
            if margin_needed > self.simulated_balance:
                self.log_message(f"❌ Insufficient margin for {symbol}: Need ${margin_needed:.2f}, Have ${self.simulated_balance:.2f}")
                return None
            
            # Create order data with properly formatted quantity
            order_data = {
                "category": "linear",
                "symbol": symbol,
                "side": "Sell",
                "orderType": "Market",
                "qty": str(final_quantity),
                "timeInForce": "GTC"
            }
            
            self.log_message(f"📉 Placing SELL order for {final_quantity} {symbol} (value: ${actual_position_cost:.2f}, margin: ${margin_needed:.2f})...")
            result, error = self.make_request("POST", "/v5/order/create", data=order_data)
            
            if error:
                self.log_message(f"❌ Sell order failed: {error}")
                return None
            
            # FIX: Update simulated balance directly with margin cost, not based on real balance
            old_simulated_balance = self.simulated_balance
            self.simulated_balance -= margin_needed
            
            # Add this logging
            self.log_message(f"🔍 DEBUG: Old simulated balance: ${old_simulated_balance:.2f}")
            self.log_message(f"🔍 DEBUG: New simulated balance after sell: ${self.simulated_balance:.2f}")
            self.log_message(f"🔍 DEBUG: Margin deducted: ${margin_needed:.2f}")
            
            self.log_message(f"💰 Simulated balance after sell: ${self.simulated_balance:.2f} (margin used: ${margin_needed:.2f})")
            
            # Record the trade
            trade = {
                'timestamp': datetime.now().isoformat(),
                'type': 'OPEN_SHORT',
                'symbol': symbol,
                'quantity': final_quantity,
                'order_id': result.get('orderId'),
                'status': result.get('orderStatus', 'Unknown'),
                'balance_before': old_simulated_balance,
                'balance_after': self.simulated_balance,
                'pnl': -margin_needed,  # Negative because it's a cost
                'position_value': actual_position_cost,
                'margin_used': margin_needed
            }

            self.log_trade_to_csv(trade)
            
            self.trades.append(trade)
            self.current_positions[symbol] = {
                'type': 'SHORT',  # New field
                'quantity': final_quantity,
                'order_id': result.get('orderId'),
                'entry_time': datetime.now().isoformat(),
                'entry_price': current_price,
                'cost': actual_position_cost,
                'margin_used': margin_needed
            }
            
            # Update working capital after the trade
            self.update_working_capital_after_trade(-margin_needed)
            
            # Update GUI if available
            if self.performance_callback:
                self.update_performance_display()
            
            # Update counters
            self.open_trades_count += 1
            
            self.log_message(f"✅ Sell order successful! Order ID: {result.get('orderId')}")
            return trade
            
        except Exception as e:
            self.log_message(f"❌ Error executing sell order: {e}")
            return None

    def execute_close_short(self, symbol, quantity=None):
        """Execute a short position closing"""
        if symbol not in self.current_positions:
            self.log_message(f"❌ No position found for {symbol}")
            return None
        
        # Check if it's a short position
        if self.current_positions[symbol]['type'] != 'SHORT':
            self.log_message(f"❌ Cannot close SHORT position for {symbol}: Current position is LONG")
            return None
        
        if quantity is None:
            quantity = self.current_positions[symbol]['quantity']
        
        try:
            # Get current price
            current_price = self.get_current_price_from_api(symbol)
            if current_price <= 0:
                self.log_message(f"❌ Could not get current price for {symbol}")
                return None
            
            # Get trading rules for symbol
            rules = self.get_trading_rules(symbol)
            if not rules:
                self.log_message(f"❌ No trading rules found for {symbol}")
                return None
            
            # Format the quantity according to the symbol's requirements
            final_quantity = self.format_quantity(quantity, rules['qty_step'])
            
            # Create order data with properly formatted quantity
            order_data = {
                "category": "linear",
                "symbol": symbol,
                "side": "Buy",
                "orderType": "Market",
                "qty": str(final_quantity),
                "timeInForce": "GTC"
            }
            
            self.log_message(f"📈 Placing BUY order for {final_quantity} {symbol}...")
            result, error = self.make_request("POST", "/v5/order/create", data=order_data)
            
            if error:
                self.log_message(f"❌ Buy order failed: {error}")
                return None
            
            # Get current real balance BEFORE closing position
            old_real_balance = self.real_balance
            
            # Calculate position P&L (for short positions, profit when price goes down)
            entry_price = self.current_positions[symbol]['entry_price']
            position_quantity = self.current_positions[symbol]['quantity']
            position_value = position_quantity * current_price
            original_cost = self.current_positions[symbol]['cost']
            margin_used = self.current_positions[symbol]['margin_used']
            
            # For short positions: profit when price goes down
            trade_pnl = original_cost - position_value
            
            # Add this logging
            self.log_message(f"🔍 DEBUG: Position value: ${position_value:.2f}, Original cost: ${original_cost:.2f}")
            self.log_message(f"🔍 DEBUG: Calculated P&L: ${trade_pnl:.2f}")
            
            # FIX: Get REAL Bybit balance AFTER closing position
            new_balance_info = self.get_real_balance()
            new_real_balance = new_balance_info['available_balance']
            
            # Calculate REAL P&L from Bybit (includes all fees, slippage, funding rates)
            real_pnl = new_real_balance - old_real_balance
            
            # Update simulated balance with REAL P&L
            old_simulated_balance = self.simulated_balance
            self.simulated_balance += real_pnl
            self.real_balance = new_real_balance
            
            # Add this logging
            self.log_message(f"🔍 DEBUG: Old simulated balance: ${old_simulated_balance:.2f}")
            self.log_message(f"🔍 DEBUG: New simulated balance after buy: ${self.simulated_balance:.2f}")
            self.log_message(f"🔍 DEBUG: REAL P&L from Bybit: ${real_pnl:.2f}")
            self.log_message(f"🔍 DEBUG: Calculated vs Real P&L diff: ${abs(trade_pnl - real_pnl):.2f}")
            
            # Calculate position duration in minutes
            entry_time = datetime.fromisoformat(self.current_positions[symbol]['entry_time'])
            exit_time = datetime.now()
            position_duration = (exit_time - entry_time).total_seconds() / 60  # Convert to minutes
            
            # Record the trade with P&L and position duration
            trade = {
                'timestamp': datetime.now().isoformat(),
                'type': 'CLOSE_SHORT',
                'symbol': symbol,
                'quantity': final_quantity,
                'order_id': result.get('orderId'),
                'status': result.get('orderStatus', 'Unknown'),
                'balance_before': old_simulated_balance,
                'balance_after': self.simulated_balance,
                'pnl': trade_pnl,
                'position_duration': position_duration,
                'position_value': position_value,
                'original_cost': original_cost,
                'margin_returned': margin_used
            }

            self.log_trade_to_csv(trade)
            
            self.trades.append(trade)
            del self.current_positions[symbol]
            
            # Update working capital after the trade
            self.update_working_capital_after_trade(trade_pnl + margin_used)
            
            # Update GUI if available
            if self.performance_callback:
                self.update_performance_display()
            
            # Update counters
            self.open_trades_count -= 1
            self.closed_trades_count += 1
            
            self.log_message(f"💰 Simulated balance after buy: ${self.simulated_balance:.2f} (P&L: ${trade_pnl:.2f}, margin returned: ${margin_used:.2f})")
            self.log_message(f"✅ Buy order successful! Order ID: {result.get('orderId')}")
            return trade
            
        except Exception as e:
            self.log_message(f"❌ Error executing buy order: {e}")
            return None
        
    def log_trade_to_csv(self, trade):
        """Log trade details to CSV for verification - Updated for new signal schema"""
        import csv
        import os
        
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # CSV file path
        csv_file = os.path.join(logs_dir, 'trading_log.csv')
        
        # Check if file exists to determine if we need to write headers
        file_exists = os.path.exists(csv_file)
        
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            
            # Write headers if file is new
            if not file_exists:
                headers = ['timestamp', 'type', 'symbol', 'quantity', 'entry_price', 'exit_price', 
                        'position_value', 'margin_used', 'pnl', 'position_duration', 
                        'balance_before', 'balance_after']
                writer.writerow(headers)
            
            # Extract data based on trade type - Updated for new signal schema
            if trade['type'] in ['OPEN_LONG', 'OPEN_SHORT']:
                # Opening positions
                writer.writerow([
                    trade['timestamp'], trade['type'], trade['symbol'], trade['quantity'],
                    trade.get('entry_price', ''), '', trade.get('position_value', ''),
                    trade.get('margin_used', ''), trade.get('pnl', ''), '',
                    trade['balance_before'], trade['balance_after']
                ])
            elif trade['type'] in ['CLOSE_LONG', 'CLOSE_SHORT']:
                # Closing positions
                # Get exit price from current price
                exit_price = self.get_current_price_from_api(trade['symbol'])
                writer.writerow([
                    trade['timestamp'], trade['type'], trade['symbol'], trade['quantity'],
                    trade.get('entry_price', ''), exit_price, trade.get('position_value', ''),
                    trade.get('margin_returned', ''), trade.get('pnl', ''), 
                    trade.get('position_duration', ''), trade['balance_before'], trade['balance_after']
                ])

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
        """Generate signal using the loaded strategy - Updated for new signal schema"""
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
            strategy_data = {
                symbol: {
                    "1m": historical_data  # We're using 1-minute data
                }
            }
            
            # Generate signals using the strategy
            try:
                self.log_message(f"🔍 DEBUG: Calling strategy.generate_signals for {symbol}")
                signals = self.strategy.generate_signals(strategy_data)
                
                self.log_message(f"🔍 DEBUG: Raw signals from strategy for {symbol}: {signals}")
                
                # Extract the signal for our symbol and timeframe
                if signals and symbol in signals and "1m" in signals[symbol]:
                    signal = signals[symbol]["1m"]
                    self.log_message(f"🔍 DEBUG: Extracted signal for {symbol}: {signal}")
                    
                    # Validate that the signal is one of the expected types
                    valid_signals = ['OPEN_LONG', 'OPEN_SHORT', 'CLOSE_LONG', 'CLOSE_SHORT', 'HOLD']
                    if signal in valid_signals:
                        if signal != 'HOLD':
                            self.log_message(f"🚨 SIGNAL DETECTED: {symbol} -> {signal}")
                        return signal
                    else:
                        self.log_message(f"⚠️ Invalid signal type for {symbol}: {signal}. Expected one of: {valid_signals}")
                        return 'HOLD'
                else:
                    self.log_message(f"⚠️ No signal returned for {symbol}")
                    return 'HOLD'
                    
            except Exception as e:
                self.log_message(f"❌ Error generating signals: {e}")
                import traceback
                self.log_message(f"Traceback: {traceback.format_exc()}")
                return 'HOLD'
            
        except Exception as e:
            self.log_message(f"❌ Error generating strategy signal for {symbol}: {e}")
            import traceback
            self.log_message(f"Traceback: {traceback.format_exc()}")
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

    def get_balance_info(self):
        """Get complete balance information"""
        # Calculate value of open positions
        open_positions_value = 0.0
        liquidation_value = self.simulated_balance
        unrealized_pnl_total = 0.0

        for symbol, position in self.current_positions.items():
            try:
                current_price = self.get_current_price_from_api(symbol)
                if current_price > 0:
                    open_positions_value += position['quantity'] * current_price
                    entry_price = position.get('entry_price', current_price)
                    qty = position.get('quantity', 0)
                    margin_used = position.get('margin_used', 0.0)
                    if position.get('type') == 'SHORT':
                        unrealized_pnl = (entry_price - current_price) * qty
                    else:
                        unrealized_pnl = (current_price - entry_price) * qty
                    liquidation_value += margin_used + unrealized_pnl
                    unrealized_pnl_total += unrealized_pnl

            except Exception as e:
                self.log_message(f"⚠️ Could not get price for {symbol}: {e}")

        
        # Calculate total account value
        account_value = self.simulated_balance + open_positions_value

        # Calculate realized PnL from closed trades
        realized_pnl_total = 0.0
        for trade in self.trades:
            if trade.get('type') in ('CLOSE_LONG', 'CLOSE_SHORT') and 'pnl' in trade:
                realized_pnl_total += trade['pnl']

        
        return {
            'simulated_balance': self.simulated_balance,
            'available_balance': self.simulated_balance,  # Cash available for new trades
            'account_value': account_value,  # Cash + open positions
            'liquidation_value': liquidation_value,  # Cash if all positions closed now
            'open_positions_value': open_positions_value,
            'realized_pnl': realized_pnl_total,
            'unrealized_pnl': unrealized_pnl_total,
            'real_balance': self.real_balance,
            'balance_offset': self.balance_offset,
            'initial_balance': self.initial_balance,
            'total_pnl': self.simulated_balance - self.initial_balance,
            'open_trades_count': self.open_trades_count,
            'closed_trades_count': self.closed_trades_count
        }


    
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
        #symbols_to_monitor = ['BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT', 'ATOMUSDT', 'ALGOUSDT', 'VETUSDT', 'ICPUSDT', 'FILUSDT', 'AAVEUSDT', 'COMPUSDT', 'CRVUSDT', 'SNXUSDT', 'SUSHIUSDT', 'ARBUSDT', 'OPUSDT', 'NEARUSDT', 'GRTUSDT']
        #available_intervals = ['1', '5', '15', '60']  # We'll use 1-minute for trading

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
            self.check_position_timeouts()

            # Process each symbol by reading from the latest CSV data
            for symbol in symbols_to_monitor:
                try:
                    # Get the latest data for the symbol from the CSV file
                    historical_data = self.get_historical_data_for_symbol(symbol)
                    
                    if historical_data is not None and not historical_data.empty:
                        # Generate a trading signal based on this fresh data
                        signal = self.generate_trading_signal(symbol, historical_data.iloc[-1]['close'])
                        
                        # Handle different signal types
                        if signal == 'OPEN_LONG':
                            # Check if we can open a new long position
                            if self.should_continue_trading():
                                # Check if we don't already have a position for this symbol
                                if symbol not in self.current_positions:
                                    self.execute_open_long(symbol)
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
                                self.execute_close_long(symbol)
                                time.sleep(0.1)
                            else:
                                self.log_message(f"⚠️ CLOSE_LONG signal for {symbol} but no long position open, skipping")
                        
                        elif signal == 'OPEN_SHORT':
                            # Check if we can open a new short position
                            if self.should_continue_trading():
                                # Check if we don't already have a position for this symbol
                                if symbol not in self.current_positions:
                                    self.execute_open_short(symbol)
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
                                self.execute_close_short(symbol)
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

    def get_current_price_from_api(self, symbol):
        """Fallback method to get price from API"""
        try:
            result, error = self.make_request("GET", "/v5/market/tickers", 
                                            params={"category": "linear", "symbol": symbol})
            if result and 'list' in result and result['list']:
                return float(result['list'][0].get('lastPrice', 0))
            return 0
        except Exception as e:
            self.log_message(f"❌ Error getting price from API: {e}")
            return 0
        
    def get_performance_summary(self):
        """Get a summary of trading performance"""
        try:
            balance_info = self.get_balance_info()
            
            # Calculate win rate from completed (closed) trades
            winning_trades = 0
            for trade in self.trades:
                if trade.get('type') in ('CLOSE_LONG', 'CLOSE_SHORT') and 'pnl' in trade:
                    if trade['pnl'] > 0:
                        winning_trades += 1
            
            win_rate = (winning_trades / self.closed_trades_count * 100) if self.closed_trades_count > 0 else 0
            
            summary = {
                'initial_balance': balance_info['initial_balance'],
                'current_balance': balance_info['simulated_balance'],
                'available_balance': balance_info['available_balance'],
                'account_value': balance_info['account_value'],
                'liquidation_value': balance_info.get('liquidation_value', balance_info['simulated_balance']),
                'open_positions_value': balance_info['open_positions_value'],
                'total_pnl': balance_info['total_pnl'],
                'realized_pnl': balance_info.get('realized_pnl', 0.0),
                'unrealized_pnl': balance_info.get('unrealized_pnl', 0.0),
                'total_trades': self.closed_trades_count,

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

    def update_performance_display(self):
        """
        Update performance display by delegating to get_performance_summary.
        This ensures a single source of truth for all metrics.
        """
        try:
            # Get the authoritative summary
            performance = self.get_performance_summary()
            
            # Update GUI if available
            if self.performance_callback:
                self.performance_callback(performance)
                
            # Optional: Log a summary to console
            self.log_message(
                f"💰 Perf: ${performance['account_value']:.2f} | "
                f"If Close Now: ${performance['liquidation_value']:.2f} | "
                f"P&L: ${performance['total_pnl']:.2f} | "
                f"Realized: ${performance.get('realized_pnl', 0.0):.2f} | "
                f"Unrealized: ${performance.get('unrealized_pnl', 0.0):.2f} | "
                f"WinRate: {performance['win_rate']:.1f}% | "

                f"Trades: {performance['total_trades']}"
            )
                
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

