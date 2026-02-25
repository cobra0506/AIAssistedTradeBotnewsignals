#paper_trading_engine.py

import os
import json
import time
import requests
import hmac
import hashlib
import math
import numpy as np
import pandas as pd
import time
from urllib.parse import urlencode
from datetime import datetime
from datetime import timedelta
from simple_strategy.shared.data_feeder import DataFeeder
from simple_strategy.shared.global_rules import resolve_global_rules
from shared_modules.data_collection.config import DataCollectionConfig
MIN_CANDLES = DataCollectionConfig.MIN_CANDLES  # Get from config
# Removed unused imports to avoid circular dependencies
import asyncio
#from shared_modules.data_collection.hybrid_system import HybridTradingSystem
#from shared_modules.data_collection.config import DataCollectionConfig

class PaperTradingEngine:
    def __init__(
        self,
        api_account,
        strategy_name,
        initial_balance=1000,
        log_callback=None,
        status_callback=None,
        performance_callback=None,
        max_positions=2000,
        stop_trading_at_percentage=None,
        enable_engine_risk_guard=True,
        execution_max_drawdown_pct=0.20,
        execution_max_position_notional_pct=0.10,
        shadow_mode=True,
        shadow_only=False,
        global_rules_config=None,
        enable_exchange_stop_loss=False,
        enable_exchange_trailing_stop=False,
        exchange_stop_loss_pct=0.02,
        risk_exit_mode=None,
        risk_sl_pct=None,
        risk_atr_period=14,
        risk_atr_sl_multiplier=1.3,
        risk_atr_tp_multiplier=1.7,
        strategy_params_override=None,
    ):
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
        self.enable_engine_risk_guard = bool(enable_engine_risk_guard)
        self.execution_max_drawdown_pct = max(0.0, float(execution_max_drawdown_pct))
        self.execution_max_position_notional_pct = max(
            0.0, min(1.0, float(execution_max_position_notional_pct))
        )
        self.shadow_mode = bool(shadow_mode)
        self.shadow_only = bool(shadow_only)
        global_rules_payload = {}
        if isinstance(global_rules_config, dict):
            if 'global_rules' in global_rules_config or 'enable_global_rules' in global_rules_config:
                global_rules_payload = dict(global_rules_config)
            else:
                global_rules_payload = {
                    'enable_global_rules': bool(global_rules_config.get('enabled', False)),
                    'global_rules_profile': global_rules_config.get('profile', 'balanced'),
                    'global_rules': dict(global_rules_config),
                }
        self.global_rules = resolve_global_rules(global_rules_payload)
        self.enable_exchange_stop_loss = bool(enable_exchange_stop_loss)
        self.enable_exchange_trailing_stop = bool(enable_exchange_trailing_stop)
        try:
            sl_pct = float(exchange_stop_loss_pct)
        except Exception:
            sl_pct = 0.02
        if sl_pct > 1.0:
            sl_pct = sl_pct / 100.0
        self.exchange_stop_loss_pct = max(0.0001, min(0.99, sl_pct))
        self.risk_exit_mode = self._normalize_risk_exit_mode(risk_exit_mode)
        if self.risk_exit_mode == 'legacy':
            self.risk_exit_mode = (
                'trailing' if self.enable_exchange_stop_loss and self.enable_exchange_trailing_stop
                else 'fixed' if self.enable_exchange_stop_loss
                else 'none'
            )
        self.risk_stop_loss_pct = self._normalize_risk_pct(
            risk_sl_pct if risk_sl_pct is not None else self.exchange_stop_loss_pct
        )
        self.risk_atr_period = max(2, int(self._safe_float(risk_atr_period, 14)))
        self.risk_atr_sl_multiplier = max(0.1, self._safe_float(risk_atr_sl_multiplier, 1.3))
        self.risk_atr_tp_multiplier = max(0.1, self._safe_float(risk_atr_tp_multiplier, 1.7))
        if self.risk_exit_mode == 'none':
            self.enable_exchange_stop_loss = False
            self.enable_exchange_trailing_stop = False
        elif self.risk_exit_mode == 'trailing':
            self.enable_exchange_stop_loss = True
            self.enable_exchange_trailing_stop = True
        elif self.risk_exit_mode in {'fixed', 'atr'}:
            self.enable_exchange_stop_loss = True
            self.enable_exchange_trailing_stop = False
        self.strategy_params_override = (
            dict(strategy_params_override) if isinstance(strategy_params_override, dict) else {}
        )
        self.global_rule_stats = {
            'blocked_signals': 0,
            'blocked_reasons': {},
            'emergency_closes': 0,
        }
        self._market_snapshot_cache = {}
        self._market_snapshot_ttl_sec = 30
        
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
        self.graceful_stop_requested = False
        self.stop_new_entries = False
        self.force_close_all_requested = False
        self.strategy = None
        
        # Data feeder for strategy integration (keep for compatibility)
        # Get the absolute path to the project's root 'data' folder
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(project_root, 'data')
        self.data_feeder = DataFeeder(data_dir=data_dir)
        self.data_dir = data_dir
        self.shadow_dashboard_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'logs',
            'shadow_dashboard.html',
        )
        self.shadow_state_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'logs',
            'shadow_state.json',
        )
        self._shadow_events = []
        
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
        self.log_message(f"  Engine Risk Guard: {'ON' if self.enable_engine_risk_guard else 'OFF'}")
        self.log_message(
            f"  Engine Max Drawdown %: {self.execution_max_drawdown_pct * 100:.2f}%"
        )
        self.log_message(
            f"  Engine Max Position Notional %: {self.execution_max_position_notional_pct * 100:.2f}%"
        )
        self.log_message(
            f"  Global Rules: {'ON' if self.global_rules.enabled else 'OFF'} (profile={self.global_rules.profile})"
        )
        self.log_message(
            f"  Exchange Stop Loss: {'ON' if self.enable_exchange_stop_loss else 'OFF'} "
            f"({self.exchange_stop_loss_pct * 100:.2f}%)"
        )
        if self.enable_exchange_stop_loss:
            self.log_message(
                f"  Exchange Stop Mode: {'TRAILING' if self.enable_exchange_trailing_stop else 'FIXED'}"
            )
        self.log_message(
            f"  Risk Exit Mode: {self.risk_exit_mode.upper()} "
            f"(SL {self.risk_stop_loss_pct * 100:.2f}%, ATR {self.risk_atr_period}, "
            f"SLx{self.risk_atr_sl_multiplier:.2f}, TPx{self.risk_atr_tp_multiplier:.2f})"
        )
        if self.global_rules.enabled:
            self.log_message(
                f"  Global Liquidity Floor: ${self.global_rules.min_24h_notional_usdt:,.0f} 24h notional"
            )
        self.log_message(f"  Shadow Mode: {'ON' if self.shadow_mode else 'OFF'}")
        if self.shadow_only:
            self.log_message("  Shadow Only: ON (signals logged, orders not executed)")
        if self.stop_trading_at_percentage:
            self.log_message(f"  Stop Trading At: {self.stop_trading_at_percentage}% of initial balance")
            
    def get_all_symbols_info(self):
        """Get information for all symbols at once and cache it"""
        try:
            result, error = self.make_request("GET", "/v5/market/instruments-info", 
                                            params={"category": "linear", "limit": 1000})
            
            if error:
                self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Error getting symbols info: {error}")
                return False
            
            if result and 'list' in result and result['list']:
                # Cache all symbol information
                self.symbol_info_cache = {}
                for symbol_info in result['list']:
                    symbol = symbol_info.get('symbol', '')
                    if symbol:
                        self.symbol_info_cache[symbol] = symbol_info
                
                self.log_message(f"ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Cached information for {len(self.symbol_info_cache)} symbols")
                return True
            else:
                self.log_message("ÃƒÂ¢Ã‚ÂÃ…â€™ No symbol information found")
                return False
                
        except Exception as e:
            self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Error getting symbols info: {e}")
            return False

    def get_trading_rules(self, symbol):
        """Get trading rules for a specific symbol"""
        if symbol not in self.symbol_info_cache:
            self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Symbol {symbol} not found in cache")
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
            self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ No trading rules found for {symbol}")
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
        
        self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  Position sizing for {symbol}:")
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
                self.log_message(f"ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â Could not set leverage for {symbol}: {error}")
                return False
            
            self.log_message(f"ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Leverage set to {leverage}x for {symbol}")
            return True
            
        except Exception as e:
            self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Error setting leverage for {symbol}: {e}")
            return False

    def _format_stop_price(self, symbol, raw_price, position_type):
        """Format stop price using the symbol's tick size."""
        try:
            price = float(raw_price)
            if price <= 0:
                return None
            rules = self.get_trading_rules(symbol) or {}
            tick_size = float(rules.get('price_tick', 0) or 0)
            if tick_size > 0:
                steps = price / tick_size
                if str(position_type).upper() == 'LONG':
                    steps = math.floor(steps)
                else:
                    steps = math.ceil(steps)
                price = max(tick_size, steps * tick_size)
                tick_str = f"{tick_size:.10f}".rstrip('0').rstrip('.')
                decimals = len(tick_str.split('.')[1]) if '.' in tick_str else 0
                return f"{price:.{decimals}f}"
            return f"{price:.8f}".rstrip('0').rstrip('.')
        except Exception:
            return None

    def _format_price_distance(self, symbol, raw_distance):
        """Format trailing-stop distance using the symbol's tick size."""
        try:
            distance = float(raw_distance)
            if distance <= 0:
                return None
            rules = self.get_trading_rules(symbol) or {}
            tick_size = float(rules.get('price_tick', 0) or 0)
            if tick_size > 0:
                steps = max(1, int(round(distance / tick_size)))
                distance = steps * tick_size
                tick_str = f"{tick_size:.10f}".rstrip('0').rstrip('.')
                decimals = len(tick_str.split('.')[1]) if '.' in tick_str else 0
                return f"{distance:.{decimals}f}"
            return f"{distance:.8f}".rstrip('0').rstrip('.')
        except Exception:
            return None

    def _get_position_idx(self, symbol, position_type):
        """Get position index for the active symbol and side."""
        side = 'Buy' if str(position_type).upper() == 'LONG' else 'Sell'
        try:
            result, error = self.make_request(
                "GET",
                "/v5/position/list",
                params={"category": "linear", "symbol": symbol},
            )
            if error or not result:
                return None
            for pos in result.get('list', []):
                if str(pos.get('symbol', '')).upper() != str(symbol).upper():
                    continue
                if str(pos.get('side', '')).capitalize() != side:
                    continue
                size = float(pos.get('size', 0) or 0)
                if size <= 0:
                    continue
                try:
                    return int(pos.get('positionIdx', 0))
                except Exception:
                    return 0
        except Exception:
            return None
        return None

    def _set_exchange_stop_loss(self, symbol, position_type, entry_price, position_meta=None):
        """Attach exchange-side protective orders (SL / trailing / ATR TP)."""
        if not self.enable_exchange_stop_loss:
            return True

        try:
            entry = float(entry_price)
            if entry <= 0:
                self.log_message(f"[SL] Cannot set exchange SL for {symbol}: invalid entry price")
                return False

            metadata = position_meta if isinstance(position_meta, dict) else {}
            mode = str(metadata.get('risk_exit_mode', self.risk_exit_mode)).lower()
            if mode not in {'fixed', 'trailing', 'atr'}:
                mode = 'trailing' if self.enable_exchange_trailing_stop else 'fixed'

            sl_pct = self._normalize_risk_pct(metadata.get('risk_sl_pct', self.risk_stop_loss_pct))
            trailing_mode = mode == 'trailing'
            stop_loss = None
            take_profit = None
            if trailing_mode:
                trailing_distance = self._format_price_distance(symbol, entry * sl_pct)
                active_price = self._format_stop_price(symbol, entry, position_type)
                if not trailing_distance or not active_price:
                    self.log_message(
                        f"[SL] Cannot set trailing stop for {symbol}: invalid distance/active price"
                    )
                    return False
            else:
                raw_stop = metadata.get('risk_stop_price')
                if raw_stop is None:
                    if str(position_type).upper() == 'LONG':
                        raw_stop = entry * (1.0 - sl_pct)
                    else:
                        raw_stop = entry * (1.0 + sl_pct)
                stop_loss = self._format_stop_price(symbol, raw_stop, position_type)
                if not stop_loss:
                    self.log_message(f"[SL] Cannot set exchange SL for {symbol}: invalid stop price")
                    return False
                raw_tp = metadata.get('risk_take_profit_price')
                if raw_tp is not None and self._safe_float(raw_tp, 0.0) > 0:
                    take_profit = self._format_stop_price(symbol, raw_tp, position_type)

            payload = {
                "category": "linear",
                "symbol": symbol,
                "tpslMode": "Full",
                "positionIdx": 0,
            }
            if trailing_mode:
                payload["trailingStop"] = trailing_distance
                payload["activePrice"] = active_price
            else:
                payload["stopLoss"] = stop_loss
                payload["slTriggerBy"] = "LastPrice"
                if take_profit:
                    payload["takeProfit"] = take_profit
                    payload["tpTriggerBy"] = "LastPrice"
            position_idx = self._get_position_idx(symbol, position_type)
            if position_idx is not None:
                payload["positionIdx"] = int(position_idx)

            last_error = None
            for attempt in range(1, 4):
                result, error = self.make_request("POST", "/v5/position/trading-stop", data=payload)
                if not error:
                    if trailing_mode:
                        self.log_message(
                            f"[SL] Exchange trailing stop set for {symbol} ({position_type}) "
                            f"distance={trailing_distance} ({sl_pct * 100:.2f}%)"
                        )
                    else:
                        if take_profit:
                            self.log_message(
                                f"[SL] Exchange ATR exits set for {symbol} ({position_type}) "
                                f"SL={stop_loss}, TP={take_profit}"
                            )
                        else:
                            self.log_message(
                                f"[SL] Exchange stop loss set for {symbol} ({position_type}) at {stop_loss} "
                                f"({sl_pct * 100:.2f}%)"
                            )
                    return True

                last_error = str(error)
                if "position idx not match position mode" in last_error.lower():
                    payload["positionIdx"] = 0
                else:
                    refreshed_idx = self._get_position_idx(symbol, position_type)
                    if refreshed_idx is not None:
                        payload["positionIdx"] = int(refreshed_idx)
                if attempt < 3:
                    time.sleep(0.2)

            self.log_message(f"[SL] Failed to set exchange stop loss for {symbol}: {last_error}")
            return False
        except Exception as e:
            self.log_message(f"[SL] Error setting exchange stop loss for {symbol}: {e}")
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
                self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Error calculating position value for {symbol}: {e}")
        
        # Available capital = simulated balance - value of open positions
        available_capital = self.simulated_balance - open_positions_value
        
        self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬â„¢Ã‚Â° Capital Analysis:")
        self.log_message(f"   Simulated Balance: ${self.simulated_balance:.2f}")
        self.log_message(f"   Open Positions Value: ${open_positions_value:.2f}")
        self.log_message(f"   Available Capital: ${available_capital:.2f}")
        
        return available_capital
    
    def check_position_timeouts(self):
        """Check for positions that have been open too long and close them."""
        if not self.current_positions:
            return

        timeout_minutes = 240
        if self.global_rules.enabled and self.global_rules.emergency_close_enabled:
            timeout_minutes = max(1, int(self.global_rules.max_hold_minutes))
        timeout_duration = timeout_minutes * 60

        current_time = datetime.now()
        positions_to_close = []

        for symbol, position in self.current_positions.items():
            entry_time = datetime.fromisoformat(position['entry_time'])
            time_open = (current_time - entry_time).total_seconds()
            if time_open > timeout_duration:
                positions_to_close.append(symbol)
                self.log_message(
                    f"[EMERGENCY] Position {symbol} timed out after {time_open/60:.1f} minutes "
                    f"(limit {timeout_minutes}m)"
                )

        for symbol in positions_to_close:
            if self.current_positions[symbol]['type'] == 'LONG':
                self.execute_close_long(symbol)
                self.global_rule_stats['emergency_closes'] = int(self.global_rule_stats.get('emergency_closes', 0)) + 1
            elif self.current_positions[symbol]['type'] == 'SHORT':
                self.execute_close_short(symbol)
                self.global_rule_stats['emergency_closes'] = int(self.global_rule_stats.get('emergency_closes', 0)) + 1

    def _check_global_emergency_exits(self):
        # Apply configured per-position exits first (fixed / trailing / ATR).
        self._check_configured_risk_exits()

        if not self.global_rules.enabled or not self.global_rules.emergency_close_enabled:
            return
        if not self.current_positions:
            return

        stop_loss_pct = max(0.0001, self._safe_float(self.global_rules.position_stop_loss_pct, 0.02))
        positions_to_close = []
        for symbol, position in self.current_positions.items():
            entry_price = self._safe_float(position.get('entry_price', 0.0))
            if entry_price <= 0.0:
                continue
            current_price = self.get_current_price_from_api(symbol)
            if current_price <= 0.0:
                continue
            position_type = str(position.get('type', '')).upper()
            if position_type == 'LONG':
                stop_price = entry_price * (1.0 - stop_loss_pct)
                if current_price <= stop_price:
                    positions_to_close.append((symbol, 'LONG', current_price, stop_price))
            elif position_type == 'SHORT':
                stop_price = entry_price * (1.0 + stop_loss_pct)
                if current_price >= stop_price:
                    positions_to_close.append((symbol, 'SHORT', current_price, stop_price))

        for symbol, position_type, current_price, stop_price in positions_to_close:
            self.log_message(
                f"[EMERGENCY] Stop loss hit for {symbol} ({position_type}) "
                f"price={current_price:.6f}, stop={stop_price:.6f}"
            )
            if position_type == 'LONG':
                self.execute_close_long(symbol)
            else:
                self.execute_close_short(symbol)
            self.global_rule_stats['emergency_closes'] = int(self.global_rule_stats.get('emergency_closes', 0)) + 1

    def update_working_capital_after_trade(self, trade_pnl):
        """Update working capital after a trade"""
        # Adjust working capital by the P&L of the trade
        self.working_capital += trade_pnl

    def log_message(self, message):
        """Log message to both console and GUI if available"""
        text = str(message)
        try:
            print(text)  # Use print instead of recursive logger calls
        except UnicodeEncodeError:
            # Some Windows consoles cannot print certain symbols; keep logging alive.
            safe_text = text.encode('ascii', errors='replace').decode('ascii')
            print(safe_text)
        if self.log_callback:
            self.log_callback(text)

    def _normalize_risk_exit_mode(self, mode):
        text = str(mode).strip().lower()
        if not text:
            return 'legacy'
        if text in {'none', 'fixed', 'trailing', 'atr'}:
            return text
        return 'legacy'

    def _normalize_risk_pct(self, value):
        pct = self._safe_float(value, 0.02)
        if pct > 1.0:
            pct = pct / 100.0
        return max(0.0001, min(0.99, pct))

    def _calculate_atr(self, historical_data, period):
        try:
            if historical_data is None or historical_data.empty:
                return None
            p = max(2, int(period))
            if len(historical_data) < (p + 1):
                return None

            high = pd.to_numeric(historical_data['high'], errors='coerce')
            low = pd.to_numeric(historical_data['low'], errors='coerce')
            close = pd.to_numeric(historical_data['close'], errors='coerce')
            prev_close = close.shift(1)
            true_range = pd.concat(
                [
                    (high - low).abs(),
                    (high - prev_close).abs(),
                    (low - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr_series = true_range.rolling(window=p, min_periods=p).mean()
            atr_value = self._safe_float(atr_series.iloc[-1], 0.0)
            if atr_value <= 0:
                return None
            return atr_value
        except Exception:
            return None

    def _get_symbol_atr(self, symbol, timeframe, period):
        data = self.get_historical_data_for_symbol(symbol, timeframe)
        return self._calculate_atr(data, period)

    def _build_position_risk_config(self, symbol, position_type, entry_price):
        position_side = str(position_type).upper()
        mode = str(self.risk_exit_mode).lower()
        config = {
            'risk_exit_mode': mode,
            'risk_sl_pct': float(self.risk_stop_loss_pct),
            'risk_atr_period': int(self.risk_atr_period),
            'risk_atr_sl_multiplier': float(self.risk_atr_sl_multiplier),
            'risk_atr_tp_multiplier': float(self.risk_atr_tp_multiplier),
        }

        entry = self._safe_float(entry_price, 0.0)
        if entry <= 0:
            config['risk_exit_mode'] = 'none'
            return config

        if mode == 'none':
            return config

        if mode in {'fixed', 'trailing'}:
            sl_pct = float(self.risk_stop_loss_pct)
            if position_side == 'LONG':
                config['risk_stop_price'] = entry * (1.0 - sl_pct)
                if mode == 'trailing':
                    config['risk_high_watermark'] = entry
            else:
                config['risk_stop_price'] = entry * (1.0 + sl_pct)
                if mode == 'trailing':
                    config['risk_low_watermark'] = entry
            config['stop_loss_pct'] = sl_pct
            return config

        if mode == 'atr':
            timeframe = self._resolve_entry_timeframe()
            atr_value = self._get_symbol_atr(symbol, timeframe, self.risk_atr_period)
            if atr_value is None:
                fallback_sl_pct = float(self.risk_stop_loss_pct)
                config['risk_exit_mode'] = 'fixed'
                if position_side == 'LONG':
                    config['risk_stop_price'] = entry * (1.0 - fallback_sl_pct)
                else:
                    config['risk_stop_price'] = entry * (1.0 + fallback_sl_pct)
                config['stop_loss_pct'] = fallback_sl_pct
                self.log_message(
                    f"[RISK] ATR unavailable for {symbol}; fallback to FIXED SL {fallback_sl_pct * 100:.2f}%"
                )
                return config

            config['risk_atr_value'] = float(atr_value)
            sl_dist = atr_value * float(self.risk_atr_sl_multiplier)
            tp_dist = atr_value * float(self.risk_atr_tp_multiplier)
            if position_side == 'LONG':
                config['risk_stop_price'] = entry - sl_dist
                config['risk_take_profit_price'] = entry + tp_dist
            else:
                config['risk_stop_price'] = entry + sl_dist
                config['risk_take_profit_price'] = entry - tp_dist

            stop_loss_pct = abs(entry - float(config['risk_stop_price'])) / max(entry, 1e-12)
            config['stop_loss_pct'] = max(0.0001, stop_loss_pct)
            return config

        config['risk_exit_mode'] = 'none'
        return config

    def _check_configured_risk_exits(self):
        if not self.current_positions:
            return

        positions_to_close = []
        for symbol, position in list(self.current_positions.items()):
            mode = str(position.get('risk_exit_mode', self.risk_exit_mode)).lower()
            if mode not in {'fixed', 'trailing', 'atr'}:
                continue

            current_price = self.get_current_price_from_api(symbol)
            if current_price <= 0:
                continue

            position_type = str(position.get('type', '')).upper()
            if position_type not in {'LONG', 'SHORT'}:
                continue

            if mode == 'trailing':
                sl_pct = self._normalize_risk_pct(position.get('risk_sl_pct', self.risk_stop_loss_pct))
                if position_type == 'LONG':
                    high_watermark = max(
                        self._safe_float(position.get('risk_high_watermark', 0.0), 0.0),
                        current_price,
                    )
                    position['risk_high_watermark'] = high_watermark
                    stop_price = high_watermark * (1.0 - sl_pct)
                else:
                    low_watermark = min(
                        self._safe_float(position.get('risk_low_watermark', current_price), current_price),
                        current_price,
                    )
                    position['risk_low_watermark'] = low_watermark
                    stop_price = low_watermark * (1.0 + sl_pct)
                position['risk_stop_price'] = stop_price
                if position_type == 'LONG' and current_price <= stop_price:
                    positions_to_close.append((symbol, position_type, current_price, stop_price, 'trailing_stop_hit'))
                if position_type == 'SHORT' and current_price >= stop_price:
                    positions_to_close.append((symbol, position_type, current_price, stop_price, 'trailing_stop_hit'))
                continue

            stop_price = self._safe_float(position.get('risk_stop_price', 0.0), 0.0)
            take_profit_price = self._safe_float(position.get('risk_take_profit_price', 0.0), 0.0)

            if stop_price > 0:
                if position_type == 'LONG' and current_price <= stop_price:
                    reason = 'atr_stop_loss_hit' if mode == 'atr' else 'fixed_stop_loss_hit'
                    positions_to_close.append((symbol, position_type, current_price, stop_price, reason))
                    continue
                if position_type == 'SHORT' and current_price >= stop_price:
                    reason = 'atr_stop_loss_hit' if mode == 'atr' else 'fixed_stop_loss_hit'
                    positions_to_close.append((symbol, position_type, current_price, stop_price, reason))
                    continue

            if mode == 'atr' and take_profit_price > 0:
                if position_type == 'LONG' and current_price >= take_profit_price:
                    positions_to_close.append(
                        (symbol, position_type, current_price, take_profit_price, 'atr_take_profit_hit')
                    )
                if position_type == 'SHORT' and current_price <= take_profit_price:
                    positions_to_close.append(
                        (symbol, position_type, current_price, take_profit_price, 'atr_take_profit_hit')
                    )

        for symbol, position_type, current_price, trigger_price, reason in positions_to_close:
            self.log_message(
                f"[RISK] {reason} for {symbol} ({position_type}) "
                f"price={current_price:.6f}, trigger={trigger_price:.6f}"
            )
            if position_type == 'LONG':
                self.execute_close_long(symbol)
            else:
                self.execute_close_short(symbol)
            self.global_rule_stats['emergency_closes'] = int(self.global_rule_stats.get('emergency_closes', 0)) + 1

    def _normalize_timeframe(self, timeframe):
        text = str(timeframe).strip()
        return text.rstrip('m')


    def _timeframe_with_suffix(self, timeframe):
        normalized = self._normalize_timeframe(timeframe)
        return f"{normalized}m"


    def _collect_strategy_timeframes(self, fallback_timeframes=None):
        timeframes = []
        if fallback_timeframes:
            timeframes.extend(list(fallback_timeframes))

        if self.strategy is not None:
            strategy_timeframes = getattr(self.strategy, 'timeframes', None)
            if strategy_timeframes:
                timeframes.extend(list(strategy_timeframes))

            entry_tf = getattr(self.strategy, 'entry_timeframe', None)
            if entry_tf:
                timeframes.append(entry_tf)

            entry_tfs = getattr(self.strategy, 'entry_timeframes', None)
            if isinstance(entry_tfs, (list, tuple)):
                timeframes.extend(list(entry_tfs))

        cleaned = []
        seen = set()
        for tf in timeframes:
            tf_text = self._timeframe_with_suffix(tf)
            key = self._normalize_timeframe(tf_text)
            if key not in seen:
                seen.add(key)
                cleaned.append(tf_text)

        if not cleaned:
            cleaned = ['1m']
        return cleaned


    def _extract_timeframes_from_params(self, parameters_def, current_params):
        timeframes = []
        if not isinstance(parameters_def, dict) or not isinstance(current_params, dict):
            return timeframes

        for param_name, param_info in parameters_def.items():
            if 'timeframe' not in str(param_name).lower():
                continue
            value = current_params.get(param_name)
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                for item in value:
                    timeframes.append(self._timeframe_with_suffix(item))
            else:
                timeframes.append(self._timeframe_with_suffix(value))
        return timeframes


    def _runtime_strategy_params(self):
        if not isinstance(self.strategy_params_override, dict):
            return {}

        sanitized = {}
        for key, value in self.strategy_params_override.items():
            if value is None:
                continue
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    continue
            sanitized[str(key)] = value
        return sanitized


    def _resolve_entry_timeframe(self):
        if self.strategy is None:
            return '1m'

        entry_tfs = getattr(self.strategy, 'entry_timeframes', None)
        if isinstance(entry_tfs, (list, tuple)) and len(entry_tfs) > 0:
            return self._timeframe_with_suffix(entry_tfs[0])

        entry_tf = getattr(self.strategy, 'entry_timeframe', None)
        if entry_tf:
            return self._timeframe_with_suffix(entry_tf)

        strategy_timeframes = getattr(self.strategy, 'timeframes', None)
        if isinstance(strategy_timeframes, (list, tuple)) and len(strategy_timeframes) > 0:
            return self._timeframe_with_suffix(strategy_timeframes[0])

        return '1m'


    def _get_strategy_state_safe(self):
        if self.strategy is None:
            return {}
        if not hasattr(self.strategy, 'get_strategy_state'):
            return {}
        try:
            state = self.strategy.get_strategy_state()
            if isinstance(state, dict):
                return state
        except Exception as e:
            self.log_message(f"ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â Could not read strategy state: {e}")
        return {}


    def _estimate_open_notional(self, symbol, current_price):
        if current_price is None or current_price <= 0:
            current_price = self.get_current_price_from_api(symbol)
        if current_price <= 0:
            return 0.0

        target = max(self.simulated_balance, 0.0) * 0.05
        rules = self.get_trading_rules(symbol)
        if rules:
            min_qty = float(rules.get('min_order_qty', 0.0))
            min_position_value = min_qty * current_price
            target = max(target, min_position_value)
        return float(target)

    def _record_global_block(self, reason):
        if not hasattr(self, 'global_rule_stats'):
            self.global_rule_stats = {
                'blocked_signals': 0,
                'blocked_reasons': {},
                'emergency_closes': 0,
            }
        self.global_rule_stats['blocked_signals'] = int(self.global_rule_stats.get('blocked_signals', 0)) + 1
        blocked_reasons = self.global_rule_stats.setdefault('blocked_reasons', {})
        blocked_reasons[reason] = int(blocked_reasons.get(reason, 0)) + 1

    def _expected_move_pct_from_strategy(self):
        if self.strategy is None:
            return max(0.1, self.global_rules.position_stop_loss_pct * 200.0)

        for attr in ('expected_move_pct', 'take_profit_pct', 'target_profit_pct', 'tp_pct'):
            value = getattr(self.strategy, attr, None)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if numeric <= 0.0:
                continue
            if numeric > 1.0:
                return numeric
            return numeric * 100.0
        return max(0.1, self.global_rules.position_stop_loss_pct * 200.0)

    def _get_symbol_market_snapshot(self, symbol):
        now_ts = time.time()
        cached = self._market_snapshot_cache.get(symbol)
        if cached and now_ts - cached.get('timestamp', 0.0) <= self._market_snapshot_ttl_sec:
            return cached.get('data', {})

        result, error = self.make_request(
            "GET",
            "/v5/market/tickers",
            params={"category": "linear", "symbol": symbol},
        )
        if error:
            return {}
        if not result or 'list' not in result or not result['list']:
            return {}

        snapshot = result['list'][0]
        self._market_snapshot_cache[symbol] = {'timestamp': now_ts, 'data': snapshot}
        return snapshot

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _safe_float_or_none(value):
        """Return float(value) or None for empty/invalid values."""
        try:
            if value is None:
                return None
            if isinstance(value, str) and not value.strip():
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _parse_wallet_balances(self, wallet_data):
        """
        Parse Bybit wallet balances with fallbacks.
        Some accounts return empty strings for totalAvailableBalance/totalMarginBalance.
        """
        if not isinstance(wallet_data, dict):
            return {'available_balance': 0.0, 'margin_balance': 0.0}

        available_balance = self._safe_float_or_none(wallet_data.get('totalAvailableBalance'))
        margin_balance = self._safe_float_or_none(wallet_data.get('totalMarginBalance'))
        wallet_balance = self._safe_float_or_none(wallet_data.get('totalWalletBalance'))

        if margin_balance is None:
            margin_balance = wallet_balance if wallet_balance is not None else 0.0
        if available_balance is None:
            available_balance = margin_balance if margin_balance is not None else 0.0

        return {
            'available_balance': float(available_balance),
            'margin_balance': float(margin_balance),
        }

    def _estimate_spread_bps(self, symbol, current_price=None, snapshot=None):
        snap = snapshot if isinstance(snapshot, dict) else self._get_symbol_market_snapshot(symbol)
        bid = self._safe_float(snap.get('bid1Price', 0.0))
        ask = self._safe_float(snap.get('ask1Price', 0.0))
        mid = self._safe_float(current_price, 0.0)
        if mid <= 0.0 and bid > 0.0 and ask > 0.0:
            mid = (bid + ask) / 2.0
        if bid <= 0.0 or ask <= 0.0 or mid <= 0.0:
            return 0.0
        return max(0.0, ((ask - bid) / mid) * 10000.0)

    def _estimate_24h_notional(self, symbol, current_price=None, snapshot=None):
        snap = snapshot if isinstance(snapshot, dict) else self._get_symbol_market_snapshot(symbol)
        turnover24h = self._safe_float(snap.get('turnover24h', 0.0))
        if turnover24h > 0.0:
            return turnover24h

        volume24h = self._safe_float(snap.get('volume24h', 0.0))
        if volume24h <= 0.0:
            return 0.0

        price = self._safe_float(current_price, 0.0)
        if price <= 0.0:
            price = self._safe_float(snap.get('lastPrice', 0.0))
        if price <= 0.0:
            return 0.0
        return volume24h * price

    def _current_open_notional(self):
        total = 0.0
        for pos in self.current_positions.values():
            cost = self._safe_float(pos.get('cost', 0.0))
            if cost <= 0.0:
                quantity = self._safe_float(pos.get('quantity', 0.0))
                entry_price = self._safe_float(pos.get('entry_price', 0.0))
                cost = quantity * entry_price
            total += abs(cost)
        return total

    def _current_open_risk(self):
        stop_loss_pct = max(0.0, self._safe_float(self.global_rules.position_stop_loss_pct, 0.02))
        return self._current_open_notional() * stop_loss_pct

    def _apply_global_entry_rules(self, symbol, current_price, planned_notional, account_value):
        rules = getattr(self, 'global_rules', None)
        if rules is None:
            rules = resolve_global_rules({})
            self.global_rules = rules
        if not hasattr(self, 'global_rule_stats'):
            self.global_rule_stats = {
                'blocked_signals': 0,
                'blocked_reasons': {},
                'emergency_closes': 0,
            }
        if not hasattr(self, '_market_snapshot_cache'):
            self._market_snapshot_cache = {}
        if not hasattr(self, '_market_snapshot_ttl_sec'):
            self._market_snapshot_ttl_sec = 30

        if not rules.enabled:
            return True, 'global_rules_disabled'

        if rules.one_position_per_symbol and symbol in self.current_positions:
            return False, 'global_position_already_open'

        snapshot = self._get_symbol_market_snapshot(symbol)

        min_notional = self._safe_float(rules.min_24h_notional_usdt, 0.0)
        if min_notional > 0.0:
            estimated_24h_notional = self._estimate_24h_notional(
                symbol=symbol, current_price=current_price, snapshot=snapshot
            )
            if estimated_24h_notional <= 0.0 or estimated_24h_notional < min_notional:
                return False, 'liquidity_below_min_notional'

        spread_bps = self._estimate_spread_bps(symbol=symbol, current_price=current_price, snapshot=snapshot)
        if self._safe_float(rules.max_spread_bps, 0.0) > 0.0 and spread_bps > float(rules.max_spread_bps):
            return False, f'spread_above_limit({spread_bps:.2f}bps)'

        # Use conservative fixed slippage for cost filter.
        slippage_pct = 0.0003
        spread_pct = spread_bps / 10000.0
        fee_pct = 0.00055
        expected_move_pct = self._expected_move_pct_from_strategy()
        round_trip_cost_pct = rules.round_trip_cost_pct(
            fee_pct=fee_pct,
            spread_pct=spread_pct,
            slippage_pct=slippage_pct,
        )
        if round_trip_cost_pct > 0.0 and self._safe_float(rules.min_expected_move_to_cost_ratio, 0.0) > 0.0:
            ratio = expected_move_pct / round_trip_cost_pct
            if ratio < float(rules.min_expected_move_to_cost_ratio):
                return False, 'expected_move_below_cost_ratio'

        max_notional_cap = max(0.0, float(account_value) * self._safe_float(rules.max_notional_exposure_pct, 0.0))
        if max_notional_cap > 0.0:
            if (self._current_open_notional() + planned_notional) > (max_notional_cap + 1e-9):
                return False, 'max_notional_exposure_exceeded'

        max_open_risk_cap = max(0.0, float(account_value) * self._safe_float(rules.max_total_open_risk_pct, 0.0))
        if max_open_risk_cap > 0.0:
            planned_risk = planned_notional * self._safe_float(rules.position_stop_loss_pct, 0.02)
            if (self._current_open_risk() + planned_risk) > (max_open_risk_cap + 1e-9):
                return False, 'max_total_open_risk_exceeded'

        return True, 'global_rules_allowed'


    def _engine_risk_guard(self, symbol, signal, current_price=None):
        if signal not in ('OPEN_LONG', 'OPEN_SHORT'):
            return True, 'non_open_signal'

        if self.shadow_only:
            return False, 'shadow_only_mode'

        if not self.enable_engine_risk_guard:
            return True, 'guard_disabled'

        if symbol in self.current_positions:
            return False, 'position_already_open'

        strategy_state = self._get_strategy_state_safe()
        if bool(strategy_state.get('drawdown_lock', False)):
            return False, 'strategy_drawdown_lock'

        virtual_drawdown = float(strategy_state.get('virtual_drawdown_pct', 0.0) or 0.0)
        strategy_max_dd = float(
            strategy_state.get('max_drawdown_pct', self.execution_max_drawdown_pct) or self.execution_max_drawdown_pct
        )
        effective_max_dd = min(max(strategy_max_dd, 0.0), max(self.execution_max_drawdown_pct, 0.0))
        if effective_max_dd > 0 and virtual_drawdown >= effective_max_dd:
            return False, 'drawdown_limit_breached'

        cap_pct = self.execution_max_position_notional_pct
        strategy_cap = strategy_state.get('effective_position_size_pct')
        if strategy_cap is not None:
            try:
                cap_pct = min(cap_pct, max(0.0, float(strategy_cap)))
            except (TypeError, ValueError):
                pass

        if cap_pct <= 0.0:
            return False, 'position_cap_zero'

        balance_info = self.get_balance_info()
        account_value = float(balance_info.get('account_value', self.simulated_balance) or self.simulated_balance)
        max_allowed_notional = max(account_value, 0.0) * cap_pct
        planned_notional = self._estimate_open_notional(symbol, current_price=current_price)

        if planned_notional <= 0.0:
            return False, 'invalid_planned_notional'
        if planned_notional > max_allowed_notional + 1e-9:
            return False, (
                f'position_cap_exceeded(planned={planned_notional:.2f},'
                f'max={max_allowed_notional:.2f})'
            )

        global_allowed, global_reason = self._apply_global_entry_rules(
            symbol=symbol,
            current_price=current_price,
            planned_notional=planned_notional,
            account_value=account_value,
        )
        if not global_allowed:
            self._record_global_block(global_reason)
            return False, global_reason

        return True, 'allowed'


    def _record_shadow_event(self, symbol, signal, allowed, reason):
        if not self.shadow_mode:
            return
        event = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'signal': signal,
            'allowed': bool(allowed),
            'reason': str(reason),
            'open_positions': len(self.current_positions),
            'simulated_balance': float(self.simulated_balance),
            'strategy_state': self._get_strategy_state_safe(),
        }
        self._shadow_events.append(event)
        if len(self._shadow_events) > 500:
            self._shadow_events = self._shadow_events[-500:]
        self._write_shadow_dashboard()


    def _write_shadow_dashboard(self):
        if not self.shadow_mode:
            return

        try:
            os.makedirs(os.path.dirname(self.shadow_dashboard_path), exist_ok=True)
            payload = {
                'generated_at': datetime.now().isoformat(),
                'event_count': len(self._shadow_events),
                'events': self._shadow_events[-200:],
            }
            with open(self.shadow_state_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            last_rows = self._shadow_events[-100:]
            rows_html = []
            for row in reversed(last_rows):
                color = '#059669' if row.get('allowed') else '#dc2626'
                rows_html.append(
                    '<tr>'
                    f"<td>{row.get('timestamp','')}</td>"
                    f"<td>{row.get('symbol','')}</td>"
                    f"<td>{row.get('signal','')}</td>"
                    f"<td style='color:{color};font-weight:600'>{'ALLOW' if row.get('allowed') else 'BLOCK'}</td>"
                    f"<td>{row.get('reason','')}</td>"
                    f"<td>{row.get('open_positions',0)}</td>"
                    f"<td>${float(row.get('simulated_balance',0.0)):.2f}</td>"
                    '</tr>'
                )

            state = self._get_strategy_state_safe()
            html = (
                "<!doctype html><html><head><meta charset='utf-8'/>"
                "<meta http-equiv='refresh' content='10'/>"
                "<title>Shadow Dashboard</title>"
                "<style>"
                "body{font-family:Segoe UI,Arial,sans-serif;background:#f8fafc;color:#0f172a;margin:16px;}"
                ".card{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px;margin-bottom:12px;}"
                "table{width:100%;border-collapse:collapse;}th,td{border:1px solid #e2e8f0;padding:6px;font-size:12px;text-align:left;}"
                "th{background:#f1f5f9;}"
                "code{background:#eef2ff;padding:1px 4px;border-radius:4px;}"
                "</style></head><body>"
                "<h2>RL Shadow Dashboard</h2>"
                f"<div class='card'><b>Generated:</b> {datetime.now().isoformat()}<br/>"
                f"<b>Events:</b> {len(self._shadow_events)}<br/>"
                f"<b>Shadow Only:</b> {self.shadow_only}<br/>"
                f"<b>Risk Guard:</b> {self.enable_engine_risk_guard}</div>"
                "<div class='card'><h3>Strategy State</h3><pre>"
                f"{json.dumps(state, indent=2, default=str)[:5000]}"
                "</pre></div>"
                "<div class='card'><h3>Recent Decisions</h3><table>"
                "<tr><th>Timestamp</th><th>Symbol</th><th>Signal</th><th>Result</th><th>Reason</th><th>Open Pos</th><th>Balance</th></tr>"
                + "".join(rows_html)
                + "</table></div></body></html>"
            )
            with open(self.shadow_dashboard_path, 'w', encoding='utf-8') as f:
                f.write(html)
        except Exception as e:
            self.log_message(f"ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â Could not write shadow dashboard: {e}")

    def initialize_shared_data_access(self):
        """Initialize shared data access safely"""
        try:
            from shared_modules.data_collection.shared_data_access import SharedDataAccess
            self.shared_data_access = SharedDataAccess()
            
            # Check if data collection is running
            if self.shared_data_access.is_data_collection_running():
                self.log_message("ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Using existing data collection process")
            else:
                self.log_message("ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â Data collection not running - will use existing CSV files")
                
            return True
        except Exception as e:
            self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Error initializing shared data access: {e}")
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
                    self.log_message(f"ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ API credentials loaded for {self.api_account}")
                    return True
            
            self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Account '{self.api_account}' not found")
            return False
            
        except Exception as e:
            self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Error loading API credentials: {e}")
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
            self.log_message(f"ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â Shared data access not available for {symbol}_{timeframe}")
            return []
    
    def test_connection(self):
        """Test the connection - EXACT working method"""
        try:
            self.log_message("Testing connection...")
            result, error = self.make_request("GET", "/v5/account/wallet-balance", params={"accountType": "UNIFIED"})
            
            if error:
                self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Connection test failed: {error}")
                return False
            
            if result and 'list' in result and result['list']:
                wallet_data = result['list'][0]
                balance = self._parse_wallet_balances(wallet_data)['available_balance']
                self.log_message(f"ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Connection successful! Balance: ${balance}")
                return True
            else:
                self.log_message("ÃƒÂ¢Ã‚ÂÃ…â€™ Connection test failed: Invalid response format")
                return False
                
        except Exception as e:
            self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Connection test error: {e}")
            return False
    
    def get_balance(self):
        """Get current simulated balance (for compatibility)"""
        return self.get_display_balance()

    def get_real_balance(self):
        """Get actual available and margin balances from Bybit"""
        try:
            result, error = self.make_request("GET", "/v5/account/wallet-balance", params={"accountType": "UNIFIED"})
            if error:
                self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Error getting real balance: {error}")
                return {'available_balance': 0.0, 'margin_balance': 0.0}
            
            if result and 'list' in result and result['list']:
                wallet_data = result['list'][0]
                return self._parse_wallet_balances(wallet_data)
            else:
                return {'available_balance': 0.0, 'margin_balance': 0.0}
                
        except Exception as e:
            self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Error getting real balance: {e}")
            return {'available_balance': 0.0, 'margin_balance': 0.0}

    def get_display_balance(self):
        """Get the simulated balance for display"""
        return self.simulated_balance
    
    def _sync_positions_from_exchange(self):
        """
        Syncs self.current_positions with the real exchange to handle restarts.
        If the bot crashed and restarted, this prevents opening duplicate positions.
        """
        self.log_message("ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ¢â‚¬Å¾ Syncing positions from exchange...")
        
        try:
            all_positions = []
            sync_errors = []
            for settle_coin in ("USDT", "USDC"):
                result, error = self.make_request(
                    "GET",
                    "/v5/position/list",
                    params={"category": "linear", "settleCoin": settle_coin},
                )
                if error:
                    sync_errors.append(f"{settle_coin}: {error}")
                    continue
                if result and 'list' in result:
                    all_positions.extend(result['list'])
            
            if not all_positions and sync_errors:
                self.log_message("Could not sync positions: " + " | ".join(sync_errors))
                return

            if all_positions:
                synced_count = 0
                for pos in all_positions:
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
                    self.log_message(f"ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Recovered {pos_type} position for {symbol} (Size: {size})")

                if synced_count > 0:
                    self.open_trades_count = len(self.current_positions)
                    self.log_message(f"ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Sync complete. Recovered {synced_count} open positions.")
                else:
                    self.log_message("ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Sync complete. No open positions found on exchange.")
                    
        except Exception as e:
            self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Error syncing positions: {e}")
    
    def get_all_perpetual_symbols(self):
        """Get all perpetual symbols"""
        try:
            result, error = self.make_request("GET", "/v5/market/instruments-info", params={"category": "linear", "limit": 1000})
            
            if error:
                self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Error getting symbols: {error}")
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
            
            self.log_message(f"ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Found {len(symbols)} perpetual symbols")
            return sorted(symbols)
                
        except Exception as e:
            self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Error getting symbols: {e}")
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
        
        self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  Filtered {len(all_symbols)} symbols down to {len(tradable_symbols)} tradable symbols")
        return tradable_symbols
    
    def execute_open_long(self, symbol, quantity=None):
        """Execute a long position opening"""
        try:
            # Add this logging at the beginning
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â DEBUG: Current simulated balance before opening LONG: ${self.simulated_balance:.2f}")
            
            # Set leverage before placing the order
            self.set_leverage(symbol)
            # Get current price
            current_price = self.get_current_price_from_api(symbol)
            if current_price <= 0:
                self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Could not get current price for {symbol}")
                return None

            can_open, block_reason = self._engine_risk_guard(symbol, 'OPEN_LONG', current_price=current_price)
            if not can_open:
                self.log_message(f"[BLOCKED] Engine risk guard blocked OPEN_LONG for {symbol}: {block_reason}")
                self._record_shadow_event(symbol, 'OPEN_LONG', False, block_reason)
                return None
            
            # Calculate minimum position value based on trading rules
            rules = self.get_trading_rules(symbol)
            if not rules:
                self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ No trading rules found for {symbol}")
                return None
            
            # Calculate minimum required capital for this symbol
            min_qty = rules['min_order_qty']
            min_position_value = min_qty * current_price
            
            # Add this logging
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â DEBUG: Min position value: ${min_position_value:.2f}")
            
            # Check if we have enough simulated balance for this position
            if self.simulated_balance < min_position_value:
                self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Insufficient balance for {symbol}: Need ${min_position_value:.2f}, Have ${self.simulated_balance:.2f}")
                return None
            
            # Use 5% of simulated balance for position sizing
            position_value = self.simulated_balance * 0.05
            
            # Add this logging
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â DEBUG: Calculated position value (5% of simulated balance): ${position_value:.2f}")
            
            # Ensure we meet minimum position requirements
            position_value = max(position_value, min_position_value)
            
            # Add this logging
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â DEBUG: Final position value after min check: ${position_value:.2f}")
            
            # Calculate a valid quantity based on trading rules
            final_quantity = self.calculate_valid_quantity(symbol, current_price, position_value)
            if final_quantity is None:
                self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Could not calculate valid quantity for {symbol}")
                return None
            
            # Final check: ensure we can afford this position
            actual_position_cost = final_quantity * current_price
            
            # Add this logging
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â DEBUG: Final position cost: ${actual_position_cost:.2f}")
            
            # Get leverage for this symbol
            symbol_info = self.symbol_info_cache.get(symbol, {})
            leverage_filter = symbol_info.get('leverageFilter', {})
            leverage = float(leverage_filter.get('maxLeverage', 5))
            
            # Calculate margin needed (position value ÃƒÆ’Ã‚Â· leverage)
            margin_needed = actual_position_cost / leverage
            
            # Add this logging
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â DEBUG: Position value: ${actual_position_cost:.2f}, Leverage: {leverage}x, Margin needed: ${margin_needed:.2f}")
            
            if margin_needed > self.simulated_balance:
                self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Insufficient margin for {symbol}: Need ${margin_needed:.2f}, Have ${self.simulated_balance:.2f}")
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
            
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã‹â€  Placing OPEN_LONG order for {final_quantity} {symbol} (value: ${actual_position_cost:.2f}, margin: ${margin_needed:.2f})...")
            result, error = self.make_request("POST", "/v5/order/create", data=order_data)
            
            if error:
                self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ OPEN_LONG order failed: {error}")
                return None
            
            # FIX: Update simulated balance directly with margin cost, not based on real balance
            old_simulated_balance = self.simulated_balance
            self.simulated_balance -= margin_needed
            
            # Add this logging
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â DEBUG: Old simulated balance: ${old_simulated_balance:.2f}")
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â DEBUG: New simulated balance after open long: ${self.simulated_balance:.2f}")
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â DEBUG: Margin deducted: ${margin_needed:.2f}")
            
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬â„¢Ã‚Â° Simulated balance after open long: ${self.simulated_balance:.2f} (margin used: ${margin_needed:.2f})")
            
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
            risk_config = self._build_position_risk_config(symbol, 'LONG', current_price)
            self.current_positions[symbol].update(risk_config)
            if self.enable_exchange_stop_loss:
                sl_set = self._set_exchange_stop_loss(
                    symbol,
                    'LONG',
                    current_price,
                    position_meta=self.current_positions[symbol],
                )
                self.current_positions[symbol]['exchange_sl_set'] = bool(sl_set)
                self.current_positions[symbol]['exchange_sl_pct'] = float(
                    self.current_positions[symbol].get('risk_sl_pct', self.risk_stop_loss_pct)
                )
                self.current_positions[symbol]['exchange_sl_mode'] = str(
                    self.current_positions[symbol].get('risk_exit_mode', self.risk_exit_mode)
                ).upper()
            
            # Update working capital after the trade
            self.update_working_capital_after_trade(-margin_needed)
            
            # Update GUI if available
            if self.performance_callback:
                self.update_performance_display()
            
            # Update counters
            self.open_trades_count += 1
            
            self.log_message(f"ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Buy order successful! Order ID: {result.get('orderId')}")
            return trade
            
        except Exception as e:
            self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Error executing buy order: {e}")
            return None
    def execute_close_long(self, symbol, quantity=None):
        """Execute a long position closing"""
        if symbol not in self.current_positions:
            self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ No position found for {symbol}")
            return None
        
        # Check if it's a long position
        if self.current_positions[symbol]['type'] != 'LONG':
            self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Cannot close LONG position for {symbol}: Current position is SHORT")
            return None
        
        if quantity is None:
            quantity = self.current_positions[symbol]['quantity']
        
        try:
            # Get current price
            current_price = self.get_current_price_from_api(symbol)
            if current_price <= 0:
                self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Could not get current price for {symbol}")
                return None
            
            # Get trading rules for symbol
            rules = self.get_trading_rules(symbol)
            if not rules:
                self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ No trading rules found for {symbol}")
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
            
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã¢â‚¬Â° Placing CLOSE_LONG order for {final_quantity} {symbol}...")
            result, error = self.make_request("POST", "/v5/order/create", data=order_data)
            
            if error:
                self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Sell order failed: {error}")
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
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â DEBUG: Position value: ${position_value:.2f}, Original cost: ${original_cost:.2f}")
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â DEBUG: Calculated P&L: ${trade_pnl:.2f}")
            
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
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â DEBUG: Old simulated balance: ${old_simulated_balance:.2f}")
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â DEBUG: New simulated balance after sell: ${self.simulated_balance:.2f}")
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â DEBUG: REAL P&L from Bybit: ${real_pnl:.2f}")
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â DEBUG: Calculated vs Real P&L diff: ${abs(trade_pnl - real_pnl):.2f}")
            
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
            
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬â„¢Ã‚Â° Simulated balance after sell: ${self.simulated_balance:.2f} (P&L: ${trade_pnl:.2f}, margin returned: ${margin_used:.2f})")
            self.log_message(f"ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Sell order successful! Order ID: {result.get('orderId')}")
            return trade
            
        except Exception as e:
            self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Error executing sell order: {e}")
            return None

    def execute_open_short(self, symbol, quantity=None):
        """Execute a short position opening"""
        try:
            # Add this logging at the beginning
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â DEBUG: Current simulated balance before opening SHORT: ${self.simulated_balance:.2f}")
            
            # Set leverage before placing the order
            self.set_leverage(symbol)
            # Get current price
            current_price = self.get_current_price_from_api(symbol)
            if current_price <= 0:
                self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Could not get current price for {symbol}")
                return None

            can_open, block_reason = self._engine_risk_guard(symbol, 'OPEN_SHORT', current_price=current_price)
            if not can_open:
                self.log_message(f"[BLOCKED] Engine risk guard blocked OPEN_SHORT for {symbol}: {block_reason}")
                self._record_shadow_event(symbol, 'OPEN_SHORT', False, block_reason)
                return None
            
            # Calculate minimum position value based on trading rules
            rules = self.get_trading_rules(symbol)
            if not rules:
                self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ No trading rules found for {symbol}")
                return None
            
            # Calculate minimum required capital for this symbol
            min_qty = rules['min_order_qty']
            min_position_value = min_qty * current_price
            
            # Add this logging
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â DEBUG: Min position value: ${min_position_value:.2f}")
            
            # Check if we have enough simulated balance for this position
            if self.simulated_balance < min_position_value:
                self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Insufficient balance for {symbol}: Need ${min_position_value:.2f}, Have ${self.simulated_balance:.2f}")
                return None
            
            # Use 5% of simulated balance for position sizing
            position_value = self.simulated_balance * 0.05
            
            # Add this logging
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â DEBUG: Calculated position value (5% of simulated balance): ${position_value:.2f}")
            
            # Ensure we meet minimum position requirements
            position_value = max(position_value, min_position_value)
            
            # Add this logging
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â DEBUG: Final position value after min check: ${position_value:.2f}")
            
            # Calculate a valid quantity based on trading rules
            final_quantity = self.calculate_valid_quantity(symbol, current_price, position_value)
            if final_quantity is None:
                self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Could not calculate valid quantity for {symbol}")
                return None
            
            # Final check: ensure we can afford this position
            actual_position_cost = final_quantity * current_price
            
            # Add this logging
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â DEBUG: Final position cost: ${actual_position_cost:.2f}")
            
            # Get leverage for this symbol
            symbol_info = self.symbol_info_cache.get(symbol, {})
            leverage_filter = symbol_info.get('leverageFilter', {})
            leverage = float(leverage_filter.get('maxLeverage', 5))
            
            # Calculate margin needed (position value ÃƒÆ’Ã‚Â· leverage)
            margin_needed = actual_position_cost / leverage
            
            # Add this logging
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â DEBUG: Position value: ${actual_position_cost:.2f}, Leverage: {leverage}x, Margin needed: ${margin_needed:.2f}")
            
            if margin_needed > self.simulated_balance:
                self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Insufficient margin for {symbol}: Need ${margin_needed:.2f}, Have ${self.simulated_balance:.2f}")
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
            
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã¢â‚¬Â° Placing OPEN_SHORT order for {final_quantity} {symbol} (value: ${actual_position_cost:.2f}, margin: ${margin_needed:.2f})...")
            result, error = self.make_request("POST", "/v5/order/create", data=order_data)
            
            if error:
                self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ OPEN_SHORT order failed: {error}")
                return None
            
            # FIX: Update simulated balance directly with margin cost, not based on real balance
            old_simulated_balance = self.simulated_balance
            self.simulated_balance -= margin_needed
            
            # Add this logging
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â DEBUG: Old simulated balance: ${old_simulated_balance:.2f}")
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â DEBUG: New simulated balance after open short: ${self.simulated_balance:.2f}")
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â DEBUG: Margin deducted: ${margin_needed:.2f}")
            
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬â„¢Ã‚Â° Simulated balance after open short: ${self.simulated_balance:.2f} (margin used: ${margin_needed:.2f})")
            
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
            risk_config = self._build_position_risk_config(symbol, 'SHORT', current_price)
            self.current_positions[symbol].update(risk_config)
            if self.enable_exchange_stop_loss:
                sl_set = self._set_exchange_stop_loss(
                    symbol,
                    'SHORT',
                    current_price,
                    position_meta=self.current_positions[symbol],
                )
                self.current_positions[symbol]['exchange_sl_set'] = bool(sl_set)
                self.current_positions[symbol]['exchange_sl_pct'] = float(
                    self.current_positions[symbol].get('risk_sl_pct', self.risk_stop_loss_pct)
                )
                self.current_positions[symbol]['exchange_sl_mode'] = str(
                    self.current_positions[symbol].get('risk_exit_mode', self.risk_exit_mode)
                ).upper()
            
            # Update working capital after the trade
            self.update_working_capital_after_trade(-margin_needed)
            
            # Update GUI if available
            if self.performance_callback:
                self.update_performance_display()
            
            # Update counters
            self.open_trades_count += 1
            
            self.log_message(f"ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Sell order successful! Order ID: {result.get('orderId')}")
            return trade
            
        except Exception as e:
            self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Error executing sell order: {e}")
            return None
    def execute_close_short(self, symbol, quantity=None):
        """Execute a short position closing"""
        if symbol not in self.current_positions:
            self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ No position found for {symbol}")
            return None
        
        # Check if it's a short position
        if self.current_positions[symbol]['type'] != 'SHORT':
            self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Cannot close SHORT position for {symbol}: Current position is LONG")
            return None
        
        if quantity is None:
            quantity = self.current_positions[symbol]['quantity']
        
        try:
            # Get current price
            current_price = self.get_current_price_from_api(symbol)
            if current_price <= 0:
                self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Could not get current price for {symbol}")
                return None
            
            # Get trading rules for symbol
            rules = self.get_trading_rules(symbol)
            if not rules:
                self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ No trading rules found for {symbol}")
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
            
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã‹â€  Placing CLOSE_SHORT order for {final_quantity} {symbol}...")
            result, error = self.make_request("POST", "/v5/order/create", data=order_data)
            
            if error:
                self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Buy order failed: {error}")
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
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â DEBUG: Position value: ${position_value:.2f}, Original cost: ${original_cost:.2f}")
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â DEBUG: Calculated P&L: ${trade_pnl:.2f}")
            
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
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â DEBUG: Old simulated balance: ${old_simulated_balance:.2f}")
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â DEBUG: New simulated balance after buy: ${self.simulated_balance:.2f}")
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â DEBUG: REAL P&L from Bybit: ${real_pnl:.2f}")
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â DEBUG: Calculated vs Real P&L diff: ${abs(trade_pnl - real_pnl):.2f}")
            
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
            
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬â„¢Ã‚Â° Simulated balance after buy: ${self.simulated_balance:.2f} (P&L: ${trade_pnl:.2f}, margin returned: ${margin_used:.2f})")
            self.log_message(f"ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Buy order successful! Order ID: {result.get('orderId')}")
            return trade
            
        except Exception as e:
            self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Error executing buy order: {e}")
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
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂºÃ¢â‚¬Ëœ Max positions reached: {len(self.current_positions)} >= {max_positions}")
            return False
        
        # Optional: Check if balance is below a percentage threshold (if enabled)
        if hasattr(self, 'stop_trading_at_percentage') and self.stop_trading_at_percentage:
            min_balance = self.initial_balance * (self.stop_trading_at_percentage / 100)
            if self.simulated_balance < min_balance:
                self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂºÃ¢â‚¬Ëœ Balance below threshold: ${self.simulated_balance:.2f} < ${min_balance:.2f} ({self.stop_trading_at_percentage}%)")
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
            self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  Position sizing: 5% of working capital = ${position_value:.2f}, calculated qty = {calculated_quantity:.6f}")
            
            return calculated_quantity
            
        except Exception as e:
            self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Error calculating position size: {e}")
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

            runtime_params = self._runtime_strategy_params()
            if runtime_params:
                current_params = dict(current_params or {})
                current_params.update(runtime_params)
                self.log_message("[INFO] Using runtime strategy parameter overrides:")
                for param, value in runtime_params.items():
                    self.log_message(f"  {param}: {value}")
            
            # Log which parameters are being used
            if optimized_params:
                self.log_message(f"ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Using optimized parameters:")
                for param, value in optimized_params.items():
                    if param != 'last_optimized':  # Skip the metadata
                        self.log_message(f"  {param}: {value}")
            else:
                self.log_message("ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â Using default parameters (no optimized parameters found)")
            
            # Create the strategy using the create_func from the registry
            if 'create_func' in strategy_info:
                # Resolve symbols and timeframes (default fallback is 1m).
                symbols = ['BTCUSDT']  # Updated per symbol in generate_strategy_signal
                timeframes = ['1m']
                param_timeframes = self._extract_timeframes_from_params(parameters_def, current_params)
                if param_timeframes:
                    timeframes.extend(param_timeframes)
                timeframes = self._collect_strategy_timeframes(timeframes)
                
                # Create the strategy
                self.strategy = strategy_info['create_func'](
                    symbols=symbols,
                    timeframes=timeframes,
                    **current_params
                )
                self.log_message(
                    f"[INFO] Strategy loaded with timeframes={self._collect_strategy_timeframes(['1m'])} "
                    f"entry={self._resolve_entry_timeframe()}"
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
        """Generate trading signal using the loaded strategy."""
        try:
            if self.strategy and hasattr(self.strategy, 'generate_signals'):
                signal = self.generate_strategy_signal(symbol)

                if signal in ('OPEN_LONG', 'OPEN_SHORT'):
                    can_open, block_reason = self._engine_risk_guard(
                        symbol,
                        signal,
                        current_price=current_price,
                    )
                    if not can_open:
                        self.log_message(
                            f"[BLOCKED] Engine risk guard blocked {signal} for {symbol}: {block_reason}"
                        )
                        self._record_shadow_event(symbol, signal, False, block_reason)
                        return 'HOLD'

                if signal in ('CLOSE_LONG', 'CLOSE_SHORT'):
                    position = self.current_positions.get(symbol)
                    if not position:
                        self.log_message(f"?? {signal} for {symbol} but no position open, changing to HOLD")
                        return 'HOLD'
                    if signal == 'CLOSE_LONG' and position.get('type') == 'SHORT':
                        self.log_message(f"?? CLOSE_LONG for {symbol} but position is SHORT, changing to HOLD")
                        return 'HOLD'
                    if signal == 'CLOSE_SHORT' and position.get('type') == 'LONG':
                        self.log_message(f"?? CLOSE_SHORT for {symbol} but position is LONG, changing to HOLD")
                        return 'HOLD'

                self._record_shadow_event(symbol, signal, True, 'signal_forwarded')
                return signal

            fallback = self.generate_rsi_signal(symbol, current_price)
            self._record_shadow_event(symbol, fallback, True, 'fallback_rsi')
            return fallback

        except Exception as e:
            self.log_message(f"? Error generating signal for {symbol}: {e}")
            self._record_shadow_event(symbol, 'HOLD', False, f'generate_signal_error:{e}')
            return 'HOLD'
    def generate_strategy_signal(self, symbol):
        """Generate signal using the loaded strategy with multi-timeframe support."""
        try:
            timeframes = self._collect_strategy_timeframes(['1m'])
            strategy_timeframe_data = {}
            for timeframe in timeframes:
                df = self.get_historical_data_for_symbol(symbol, timeframe)
                if df is not None and not df.empty:
                    strategy_timeframe_data[timeframe] = df

            if not strategy_timeframe_data:
                self.log_message(f"?? No historical data available for {symbol} on requested timeframes")
                return 'HOLD'

            entry_timeframe = self._resolve_entry_timeframe()
            entry_candidates = [
                entry_timeframe,
                self._normalize_timeframe(entry_timeframe),
                self._timeframe_with_suffix(entry_timeframe),
            ]
            entry_df = None
            for key in entry_candidates:
                if key in strategy_timeframe_data:
                    entry_df = strategy_timeframe_data[key]
                    break

            if entry_df is None:
                first_key = next(iter(strategy_timeframe_data))
                entry_df = strategy_timeframe_data[first_key]
                entry_timeframe = first_key

            if len(entry_df) < MIN_CANDLES:
                self.log_message(
                    f"?? Not enough historical data for {symbol} {entry_timeframe} "
                    f"(only {len(entry_df)} rows)"
                )
                return 'HOLD'

            strategy_data = {
                symbol: strategy_timeframe_data
            }

            try:
                if hasattr(self.strategy, 'generate_signals_vectorized'):
                    self.log_message(f"?? DEBUG: Calling strategy.generate_signals_vectorized for {symbol}")
                    signals = self.strategy.generate_signals_vectorized(strategy_data)
                else:
                    self.log_message(f"?? DEBUG: Calling strategy.generate_signals for {symbol}")
                    signals = self.strategy.generate_signals(strategy_data)

                self.log_message(f"?? DEBUG: Raw signals from strategy for {symbol}: {signals}")

                if signals and symbol in signals:
                    symbol_signals = signals[symbol]
                    signal = None
                    for key in (
                        entry_timeframe,
                        self._normalize_timeframe(entry_timeframe),
                        self._timeframe_with_suffix(entry_timeframe),
                    ):
                        if key in symbol_signals:
                            signal = symbol_signals[key]
                            break

                    if signal is None and symbol_signals:
                        first_tf = next(iter(symbol_signals))
                        signal = symbol_signals[first_tf]

                    if signal is None:
                        self.log_message(f"?? No signal returned for {symbol}")
                        return 'HOLD'

                    if hasattr(signal, 'iloc'):
                        signal = signal.iloc[-1]
                    self.log_message(f"?? DEBUG: Extracted signal for {symbol}: {signal}")

                    valid_signals = ['OPEN_LONG', 'OPEN_SHORT', 'CLOSE_LONG', 'CLOSE_SHORT', 'HOLD']
                    if signal in valid_signals:
                        if signal != 'HOLD':
                            self.log_message(f"?? SIGNAL DETECTED: {symbol} -> {signal}")
                        return signal

                    self.log_message(
                        f"?? Invalid signal type for {symbol}: {signal}. Expected one of: {valid_signals}"
                    )
                    return 'HOLD'

                self.log_message(f"?? No signal returned for {symbol}")
                return 'HOLD'

            except Exception as e:
                self.log_message(f"? Error generating signals: {e}")
                import traceback
                self.log_message(f"Traceback: {traceback.format_exc()}")
                return 'HOLD'

        except Exception as e:
            self.log_message(f"? Error generating strategy signal for {symbol}: {e}")
            import traceback
            self.log_message(f"Traceback: {traceback.format_exc()}")
            return 'HOLD'

    def get_historical_data_for_symbol(self, symbol, timeframe='1m'):
        """Get historical data for a symbol/timeframe by loading directly from CSV."""
        return self.load_csv_data(symbol, timeframe=timeframe)

    def load_csv_data(self, symbol, timeframe='1m'):
        """Load data directly from CSV file as fallback."""
        try:
            import pandas as pd

            tf = self._normalize_timeframe(timeframe)
            csv_file = os.path.join(self.data_dir, f'{symbol}_{tf}.csv')

            if not os.path.exists(csv_file):
                self.log_message(f"?? CSV file not found for {symbol} {timeframe}: {csv_file}")
                return None

            df = pd.read_csv(csv_file)

            if not isinstance(df, pd.DataFrame):
                self.log_message(f"? Loaded data is not a DataFrame for {symbol} {timeframe}")
                return None

            if df.empty:
                self.log_message(f"?? Empty DataFrame for {symbol} {timeframe}")
                return pd.DataFrame()

            if 'timestamp' in df.columns and 'datetime' not in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

            if 'datetime' in df.columns:
                df = df.sort_values('datetime')

            if len(df) > 1000:
                df = df.tail(1000)

            self.log_message(f"? Loaded {len(df)} rows from CSV for {symbol} {timeframe}")
            return df

        except Exception as e:
            self.log_message(f"? Error loading CSV for {symbol} {timeframe}: {e}")
            return None

    def _freshness_threshold_seconds(self, timeframe: str) -> int:
        """Get max acceptable age for entry data before marking a symbol stale."""
        tf_text = self._normalize_timeframe(timeframe)
        try:
            tf_minutes = int(tf_text)
        except (TypeError, ValueError):
            tf_minutes = 1
        return max(180, tf_minutes * 120)

    def _is_historical_data_fresh(self, historical_data, timeframe: str):
        """Return (is_fresh, age_seconds) for the entry timeframe dataset."""
        if historical_data is None or historical_data.empty:
            return False, None

        try:
            latest_dt = None
            if 'datetime' in historical_data.columns:
                latest_dt = pd.to_datetime(historical_data['datetime'].iloc[-1], errors='coerce')
            elif 'timestamp' in historical_data.columns:
                latest_dt = pd.to_datetime(historical_data['timestamp'].iloc[-1], unit='ms', errors='coerce')

            if latest_dt is None or pd.isna(latest_dt):
                return False, None

            if getattr(latest_dt, "tzinfo", None) is not None:
                latest_dt = latest_dt.tz_localize(None)

            now_dt = datetime.now()
            age_seconds = max(0.0, (now_dt - latest_dt.to_pydatetime()).total_seconds())
            return age_seconds <= float(self._freshness_threshold_seconds(timeframe)), age_seconds
        except Exception:
            return False, None

    def update_balance_after_trade(self, pnl_amount):
        """Update simulated balance after trade"""
        self.simulated_balance += pnl_amount
        self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬â„¢Ã‚Â° Balance updated: ${self.simulated_balance:.2f} (P&L: ${pnl_amount:.2f})")

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
                self.log_message(f"ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â Could not get price for {symbol}: {e}")

        
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

    def request_graceful_stop(self):
        """Stop opening new positions, keep closing until all positions are flat."""
        self.graceful_stop_requested = True
        self.stop_new_entries = True
        self.force_close_all_requested = False
        self.log_message(
            f"[STOP] Graceful stop requested. New entries disabled. Open positions: {len(self.current_positions)}"
        )

    def request_force_close_all(self):
        """Request immediate close for all currently open positions."""
        self.graceful_stop_requested = True
        self.stop_new_entries = True
        self.force_close_all_requested = True
        self.log_message(
            f"[STOP] Force-close requested for {len(self.current_positions)} open position(s)."
        )

    def _force_close_all_positions(self):
        """Close all currently open positions using market close orders."""
        if not self.current_positions:
            return 0

        closed_count = 0
        for symbol, position in list(self.current_positions.items()):
            if symbol not in self.current_positions:
                continue
            pos_type = str(position.get('type', '')).upper()
            if pos_type == 'LONG':
                result = self.execute_close_long(symbol)
            elif pos_type == 'SHORT':
                result = self.execute_close_short(symbol)
            else:
                continue
            if result is not None:
                closed_count += 1
        return closed_count

    def get_shutdown_state(self):
        """Expose shutdown state for launcher UI polling."""
        return {
            'is_running': bool(self.is_running),
            'graceful_stop_requested': bool(self.graceful_stop_requested),
            'stop_new_entries': bool(self.stop_new_entries),
            'force_close_all_requested': bool(self.force_close_all_requested),
            'open_positions': len(self.current_positions),
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
        
        self.graceful_stop_requested = False
        self.stop_new_entries = False
        self.force_close_all_requested = False
        self.is_running = True
        self.trading_loop_active = True
        self.log_message(f"Paper trading started for {self.strategy_name}")
        
        # Get symbols and intervals directly from our local data files
        symbols_to_monitor, available_intervals = self.get_symbols_and_intervals_from_data_dir()
        #symbols_to_monitor = ['BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT', 'ATOMUSDT', 'ALGOUSDT', 'VETUSDT', 'ICPUSDT', 'FILUSDT', 'AAVEUSDT', 'COMPUSDT', 'CRVUSDT', 'SNXUSDT', 'SUSHIUSDT', 'ARBUSDT', 'OPUSDT', 'NEARUSDT', 'GRTUSDT']
        #available_intervals = ['1', '5', '15', '60']  # We'll use 1-minute for trading
        entry_timeframe = self._resolve_entry_timeframe()
        entry_interval = self._normalize_timeframe(entry_timeframe)
        if entry_interval not in available_intervals:
            self.log_message(f"[ERROR] No {entry_timeframe} interval data found. Cannot start.")
            return

        self.log_message(
            f"[INFO] Monitoring {len(symbols_to_monitor)} symbols on entry timeframe {entry_timeframe}.",
        )
        
        # Main CSV-based trading loop
        loop_count = 0
        while self.is_running:
            if self.force_close_all_requested:
                if self.current_positions:
                    open_before = len(self.current_positions)
                    self.log_message(f"[STOP] Force-closing {open_before} position(s)...")
                    closed_count = self._force_close_all_positions()
                    self.log_message(
                        f"[STOP] Force-close sweep done. Closed {closed_count}, remaining {len(self.current_positions)}."
                    )
                self.force_close_all_requested = False

            if self.graceful_stop_requested and not self.current_positions:
                self.log_message("[STOP] All positions closed. Safe shutdown complete.")
                self.is_running = False
                break

            loop_count += 1
            self.log_message(f"\n=== Trading Loop #{loop_count} ===")
            stale_symbols_cycle = []
            blocked_open_signals_cycle = 0
        
            # Check for position timeouts
            self.check_position_timeouts()
            self._check_global_emergency_exits()
            # Process each symbol by reading from the latest CSV data
            for symbol in symbols_to_monitor:
                try:
                    entry_timeframe = self._resolve_entry_timeframe()
                    historical_data = self.get_historical_data_for_symbol(symbol, entry_timeframe)

                    if historical_data is not None and not historical_data.empty:
                        is_fresh, _age_seconds = self._is_historical_data_fresh(historical_data, entry_timeframe)
                        if not is_fresh:
                            stale_symbols_cycle.append(symbol)
                            continue

                        signal = self.generate_trading_signal(symbol, historical_data.iloc[-1]['close'])

                        if self.shadow_only and signal in ('OPEN_LONG', 'OPEN_SHORT', 'CLOSE_LONG', 'CLOSE_SHORT'):
                            self.log_message(
                                f"[SHADOW] {symbol} {signal} blocked in shadow_only mode"
                            )
                            self._record_shadow_event(symbol, signal, False, 'shadow_only_mode')
                            continue

                        if signal == 'OPEN_LONG':
                            if self.stop_new_entries:
                                blocked_open_signals_cycle += 1
                                continue
                            if self.should_continue_trading():
                                if symbol not in self.current_positions:
                                    self.execute_open_long(symbol)
                                    time.sleep(0.1)
                                else:
                                    self.log_message(f"?? OPEN_LONG signal for {symbol} but position already open, skipping")
                            else:
                                if len(self.current_positions) >= self.max_positions:
                                    self.log_message(f"?? Max positions reached ({len(self.current_positions)} >= {self.max_positions}), skipping {symbol}")
                                elif hasattr(self, 'stop_trading_at_percentage') and self.stop_trading_at_percentage:
                                    min_balance = self.initial_balance * (self.stop_trading_at_percentage / 100)
                                    if self.simulated_balance < min_balance:
                                        self.log_message(f"?? Balance below threshold ({self.simulated_balance:.2f} < {min_balance:.2f}), skipping {symbol}")

                        elif signal == 'CLOSE_LONG':
                            if symbol in self.current_positions and self.current_positions[symbol]['type'] == 'LONG':
                                self.execute_close_long(symbol)
                                time.sleep(0.1)
                            else:
                                self.log_message(f"?? CLOSE_LONG signal for {symbol} but no long position open, skipping")

                        elif signal == 'OPEN_SHORT':
                            if self.stop_new_entries:
                                blocked_open_signals_cycle += 1
                                continue
                            if self.should_continue_trading():
                                if symbol not in self.current_positions:
                                    self.execute_open_short(symbol)
                                    time.sleep(0.1)
                                else:
                                    self.log_message(f"?? OPEN_SHORT signal for {symbol} but position already open, skipping")
                            else:
                                if len(self.current_positions) >= self.max_positions:
                                    self.log_message(f"?? Max positions reached ({len(self.current_positions)} >= {self.max_positions}), skipping {symbol}")
                                elif hasattr(self, 'stop_trading_at_percentage') and self.stop_trading_at_percentage:
                                    min_balance = self.initial_balance * (self.stop_trading_at_percentage / 100)
                                    if self.simulated_balance < min_balance:
                                        self.log_message(f"?? Balance below threshold ({self.simulated_balance:.2f} < {min_balance:.2f}), skipping {symbol}")

                        elif signal == 'CLOSE_SHORT':
                            if symbol in self.current_positions and self.current_positions[symbol]['type'] == 'SHORT':
                                self.execute_close_short(symbol)
                                time.sleep(0.1)
                            else:
                                self.log_message(f"?? CLOSE_SHORT signal for {symbol} but no short position open, skipping")
                except Exception as e:
                    self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Error processing {symbol}: {e}")
                    continue
            
            if stale_symbols_cycle:
                stale_sample = ", ".join(stale_symbols_cycle[:12])
                more_count = len(stale_symbols_cycle) - 12
                more_text = f" (+{more_count} more)" if more_count > 0 else ""
                self.log_message(
                    f"[STALE] Skipped {len(stale_symbols_cycle)}/{len(symbols_to_monitor)} symbols due to stale "
                    f"{entry_timeframe} data: {stale_sample}{more_text}"
                )
            if blocked_open_signals_cycle > 0:
                self.log_message(
                    f"[STOP] Graceful stop active: blocked {blocked_open_signals_cycle} new OPEN signal(s)."
                )

            # Update performance and wait for the next minute
            self.update_performance_display()
            if self.is_running:
                self.log_message("Cycle complete. Waiting for the next minute...")
                sleep_seconds = 5 if self.graceful_stop_requested else 60
                for _ in range(sleep_seconds):
                    if not self.is_running or self.force_close_all_requested:
                        break
                    time.sleep(1)
        self.trading_loop_active = False
        self.log_message("Trading loop ended")
    
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
            self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Error getting price from API: {e}")
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
                'status': (
                    'Stopping (graceful)'
                    if self.is_running and self.graceful_stop_requested
                    else 'Running' if self.is_running else 'Stopped'
                ),
                'global_rules': {
                    'enabled': bool(getattr(getattr(self, 'global_rules', None), 'enabled', False)),
                    'profile': str(getattr(getattr(self, 'global_rules', None), 'profile', 'balanced')),
                    'blocked_signals': int(getattr(self, 'global_rule_stats', {}).get('blocked_signals', 0)),
                    'blocked_reasons': dict(getattr(self, 'global_rule_stats', {}).get('blocked_reasons', {})),
                    'emergency_closes': int(getattr(self, 'global_rule_stats', {}).get('emergency_closes', 0)),
                },
            }
            
            return summary
        except Exception as e:
            self.log_message(f"Error generating performance summary: {e}")
            return {
                'error': str(e),
                'status': 'Error'
            }
        
    def stop_trading(self, immediate=True):
        """Stop paper trading immediately, or request graceful stop."""
        if not immediate:
            self.request_graceful_stop()
            return
        self.is_running = False
        self.trading_loop_active = False
        self.graceful_stop_requested = False
        self.stop_new_entries = False
        self.force_close_all_requested = False
        self.log_message("Paper trading stopped by user")

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
                f"ÃƒÂ°Ã…Â¸Ã¢â‚¬â„¢Ã‚Â° Perf: ${performance['account_value']:.2f} | "
                f"If Close Now: ${performance['liquidation_value']:.2f} | "
                f"P&L: ${performance['total_pnl']:.2f} | "
                f"Realized: ${performance.get('realized_pnl', 0.0):.2f} | "
                f"Unrealized: ${performance.get('unrealized_pnl', 0.0):.2f} | "
                f"WinRate: {performance['win_rate']:.1f}% | "

                f"Trades: {performance['total_trades']}"
            )
                
        except Exception as e:
            self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Error updating performance: {e}")
            
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

        self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â Scanning for symbols in {data_dir}...")

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
            self.log_message(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Error: Data directory not found at {data_dir}")
            return [], []

        end_time = time.time()
        duration = end_time - start_time

        # Log the statistics
        self.log_message(f"ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Scan complete in {duration:.2f} seconds.")
        self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  Found {file_count} data files.")
        self.log_message(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã‹â€  Found {len(symbols)} unique symbols: {', '.join(list(symbols)[:10])}...")
        self.log_message(f"ÃƒÂ¢Ã‚ÂÃ‚Â±ÃƒÂ¯Ã‚Â¸Ã‚Â Found {len(intervals)} unique intervals: {', '.join(sorted(list(intervals)))}")

        return sorted(list(symbols)), sorted(list(intervals))

