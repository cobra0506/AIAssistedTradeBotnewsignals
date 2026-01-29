"""
API Connector - Handles all API-related operations
"""

import os
import time
import requests
import hmac
import hashlib
import json
from urllib.parse import urlencode

class APIConnector:
    """Handles all API communication with the exchange"""
    
    def __init__(self, api_key, api_secret, base_url="https://api-demo.bybit.com", recv_window="5000", logger=None):
        """Initialize API connector with credentials"""
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.recv_window = recv_window
        self.log_message = logger if logger else print
        self.symbol_info_cache = {}
    
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