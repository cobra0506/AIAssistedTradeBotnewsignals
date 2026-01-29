"""
Position Manager - Handles all position-related operations
"""

import os
import time
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional

class PositionManager:
    """Manages opening and closing positions for both long and short trades"""
    
    def __init__(self, api_connector, logger):
        """Initialize with API connector and logger"""
        self.api_connector = api_connector
        self.log_message = logger
        self.current_positions = {}
        self.trades = []
        self.open_trades_count = 0
        self.closed_trades_count = 0
    
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
        """Log trade details to CSV for verification"""
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
            
            # Extract data based on trade type
            if trade['type'] == 'BUY':
                writer.writerow([
                    trade['timestamp'], trade['type'], trade['symbol'], trade['quantity'],
                    trade.get('entry_price', ''), '', trade.get('position_value', ''),
                    trade.get('margin_used', ''), trade.get('pnl', ''), '',
                    trade['balance_before'], trade['balance_after']
                ])
            else:  # SELL
                # Get exit price from current price
                exit_price = self.get_current_price_from_api(trade['symbol'])
                writer.writerow([
                    trade['timestamp'], trade['type'], trade['symbol'], trade['quantity'],
                    trade.get('entry_price', ''), exit_price, trade.get('position_value', ''),
                    trade.get('margin_returned', ''), trade.get('pnl', ''), 
                    trade.get('position_duration', ''), trade['balance_before'], trade['balance_after']
                ])
    
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

    def update_simulated_balance(self, new_balance):
        """Update the simulated balance from the main engine"""
        self.simulated_balance = new_balance