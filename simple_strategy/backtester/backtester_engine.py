"""
Backtester Engine Component - Phase 1.1
Core backtesting logic that processes data and executes strategies
Integrates with DataFeeder for data access and StrategyBase for signal generation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging
import heapq
from concurrent.futures import ThreadPoolExecutor, as_completed
from simple_strategy.backtester.risk_manager import RiskManager
from simple_strategy.backtester.performance_tracker import PerformanceTracker
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys
# Fix import paths - shared is a sibling directory, not a subdirectory
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.data_feeder import DataFeeder
from shared.strategy_base import StrategyBase
from simple_strategy.shared.global_rules import GlobalRules, resolve_global_rules
from simple_strategy.shared.risk_exit_engine import (
    build_position_risk_config,
    calculate_atr_from_dataframe,
    evaluate_risk_exit,
)

# Configure logging to reduce debug output
logging.basicConfig(
    level=logging.WARNING,  # Change from INFO to WARNING to reduce output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Specifically set debug level for backtester to WARNING
logging.getLogger('simple_strategy.backtester.backtester_engine').setLevel(logging.WARNING)
logging.getLogger('simple_strategy.strategies.strategy_builder').setLevel(logging.WARNING)

# Create logger instance for this module
logger = logging.getLogger(__name__)

class BacktesterEngine:
    """
    Core backtesting engine that processes historical data and executes strategies
    """
    def __init__(self, data_feeder: DataFeeder, strategy: StrategyBase,
                 risk_manager: Optional[RiskManager] = None, config: Dict[str, Any] = None):
        """
        Initialize backtester engine with risk management integration
        Args:
            data_feeder: DataFeeder instance for data access
            strategy: StrategyBase instance for signal generation
            risk_manager: RiskManager instance for risk management (optional)
            config: Backtester configuration
        """
        self.data_feeder = data_feeder
        self.strategy = strategy
        self.risk_manager = risk_manager or RiskManager()  # Use default if not provided
        self.config = config or {}
        self.global_rules: GlobalRules = resolve_global_rules(self.config)
        self._global_rule_stats = {
            'blocked_signals': 0,
            'blocked_reasons': {},
            'emergency_closes': 0,
        }
        
        # Backtester state
        self.is_running = False
        self.current_timestamp = None
        self.processed_data = {}
        self.results = {}
        
        # Performance tracking
        self.start_time = None
        self.end_time = None
        self.processing_stats = {
            'total_rows_processed': 0,
            'total_signals_generated': 0,
            'total_trades_executed': 0,
            'processing_speed_rows_per_sec': 0
        }

        self.equity = 0.0
        self.peak_equity = 0.0
        self.max_drawdown = 0.0
        self._debug_count = []
        self.debug_signals = self.config.get('debug_signals', False)
        
        # Configuration
        self.processing_mode = self.config.get('processing_mode', 'sequential')  # 'sequential' or 'parallel'
        self.batch_size = self.config.get('batch_size', 1000)
        self.memory_limit_percent = self.config.get('memory_limit_percent', 70)
        self.enable_parallel_processing = self.config.get('enable_parallel_processing', False)
        
        # Initialize performance tracker
        self.performance_tracker = PerformanceTracker(initial_balance=10000)

        # Position tracking
        self.positions = {}  # Track open positions by symbol

        logger.info(f"BacktesterEngine initialized with strategy: {strategy.name}")
        logger.info(f"Risk management {'enabled'if self.risk_manager else'disabled'}")

    def get_order_size(self, symbol, entry_price, stop_price=None):
        """
        Default to paper-like balance percentage sizing, with an opt-in
        risk-based mode preserved for older runs.
        """
        try:
            entry = float(entry_price)
        except (TypeError, ValueError):
            return 0.0
        if entry <= 0.0:
            return 0.0

        sizing_mode = str(self.config.get('position_sizing_mode', 'balance_pct')).strip().lower()
        if sizing_mode == 'risk_based':
            risk_per_trade = self.config.get('risk_per_trade', 0.02)
            account_risk = self.balance * risk_per_trade

            if stop_price is None:
                stop_price = entry * 0.99

            stop_distance = abs(entry - float(stop_price))
            if stop_distance <= 0:
                return 0.0
            return account_risk / stop_distance

        position_value = max(float(getattr(self, 'balance', 0.0)), 0.0) * self._resolve_position_size_pct()
        if position_value <= 0.0:
            return 0.0
        return position_value / entry
    
    def _calculate_unrealized_pnl(self, current_prices):
        unrealized = 0.0
        for symbol, position in self.positions.items():
            if symbol not in current_prices:
                continue

            price = current_prices[symbol]
            entry = position['entry_price']
            qty = position['quantity']

            if position.get('is_short', False):
                unrealized += (entry - price) * qty
            else:
                unrealized += (price - entry) * qty

        return unrealized

    def _refresh_runtime_config(self, runtime_config: Optional[Dict[str, Any]] = None) -> None:
        if runtime_config:
            merged = dict(self.config)
            merged.update(runtime_config)
            self.config = merged
        self.global_rules = resolve_global_rules(self.config)
        self._global_rule_stats = {
            'blocked_signals': 0,
            'blocked_reasons': {},
            'emergency_closes': 0,
        }

    @staticmethod
    def _timeframe_to_minutes(timeframe: str) -> int:
        text = str(timeframe).strip().lower()
        if not text:
            return 1

        suffix = text[-1]
        try:
            value = int(text[:-1]) if suffix in {'m', 'h', 'd'} else int(text)
        except ValueError:
            return 1

        if suffix == 'h':
            return max(1, value * 60)
        if suffix == 'd':
            return max(1, value * 1440)
        return max(1, value)

    @staticmethod
    def _normalize_timeframe(timeframe: Any) -> str:
        text = str(timeframe).strip().lower()
        if not text:
            return '1'
        if text.endswith('m'):
            return text[:-1] or '1'
        if text.endswith('h'):
            try:
                return str(max(1, int(text[:-1])) * 60)
            except ValueError:
                return '1'
        if text.endswith('d'):
            try:
                return str(max(1, int(text[:-1])) * 1440)
            except ValueError:
                return '1'
        return text

    def _timeframe_with_suffix(self, timeframe: Any) -> str:
        return f"{self._normalize_timeframe(timeframe)}m"

    def _collect_strategy_timeframes(
        self,
        strategy: StrategyBase,
        fallback_timeframes: Optional[List[str]] = None,
    ) -> List[str]:
        timeframes: List[str] = []
        if fallback_timeframes:
            timeframes.extend(list(fallback_timeframes))

        if strategy is not None:
            strategy_timeframes = getattr(strategy, 'timeframes', None)
            if strategy_timeframes:
                timeframes.extend(list(strategy_timeframes))

            entry_tf = getattr(strategy, 'entry_timeframe', None)
            if entry_tf:
                timeframes.append(entry_tf)

            entry_tfs = getattr(strategy, 'entry_timeframes', None)
            if isinstance(entry_tfs, (list, tuple)):
                timeframes.extend(list(entry_tfs))

        cleaned: List[str] = []
        seen = set()
        for tf in timeframes:
            tf_text = self._timeframe_with_suffix(tf)
            key = self._normalize_timeframe(tf_text)
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(tf_text)

        if not cleaned:
            cleaned = ['1m']
        return cleaned

    def _resolve_entry_timeframe(self, strategy: Optional[StrategyBase] = None) -> str:
        target_strategy = strategy or self.strategy
        if target_strategy is None:
            return '1m'

        entry_tfs = getattr(target_strategy, 'entry_timeframes', None)
        if isinstance(entry_tfs, (list, tuple)) and len(entry_tfs) > 0:
            return self._timeframe_with_suffix(entry_tfs[0])

        entry_tf = getattr(target_strategy, 'entry_timeframe', None)
        if entry_tf:
            return self._timeframe_with_suffix(entry_tf)

        strategy_timeframes = getattr(target_strategy, 'timeframes', None)
        if isinstance(strategy_timeframes, (list, tuple)) and len(strategy_timeframes) > 0:
            return self._timeframe_with_suffix(strategy_timeframes[0])

        return '1m'

    def _resolve_timeframe_key(self, mapping: Dict[str, Any], preferred: Any) -> Optional[str]:
        if not isinstance(mapping, dict) or not mapping:
            return None

        target = self._normalize_timeframe(preferred)
        for key in mapping.keys():
            if self._normalize_timeframe(key) == target:
                return key
        return None

    def _strategy_requires_full_signal_path(self, strategy: StrategyBase) -> bool:
        if bool(getattr(strategy, 'force_strategy_generate_signals', False)):
            return True

        builder = getattr(strategy, 'builder', None)
        if builder is not None and getattr(builder, 'filter_rules', None):
            if len(getattr(builder, 'filter_rules', {})) > 0:
                return True

        entry_tfs = getattr(strategy, 'entry_timeframes', None)
        if isinstance(entry_tfs, (list, tuple)) and len(entry_tfs) > 0:
            return True

        entry_tf = getattr(strategy, 'entry_timeframe', None)
        if entry_tf:
            return True

        strategy_timeframes = getattr(strategy, 'timeframes', None)
        if isinstance(strategy_timeframes, (list, tuple)) and len(strategy_timeframes) > 1:
            return True

        return False

    def _build_strategy_snapshot(
        self,
        symbol_timeframe_data: Dict[str, pd.DataFrame],
        strategy_timeframes: List[str],
        timestamp: Any,
    ) -> Dict[str, pd.DataFrame]:
        snapshot: Dict[str, pd.DataFrame] = {}
        target_ts = pd.Timestamp(timestamp)

        for requested_tf in strategy_timeframes:
            actual_key = self._resolve_timeframe_key(symbol_timeframe_data, requested_tf)
            if actual_key is None:
                continue

            tf_df = symbol_timeframe_data.get(actual_key)
            if tf_df is None or tf_df.empty:
                continue

            try:
                cutoff = int(tf_df.index.searchsorted(target_ts, side='right'))
            except Exception:
                cutoff = len(tf_df.loc[:target_ts])

            if cutoff <= 0:
                continue

            snapshot[actual_key] = tf_df.iloc[:cutoff]

        return snapshot

    def _extract_signal_value(
        self,
        signals: Dict[str, Dict[str, Any]],
        *,
        symbol: str,
        entry_timeframe: str,
    ) -> str:
        if not isinstance(signals, dict) or symbol not in signals:
            return 'HOLD'

        symbol_signals = signals.get(symbol, {})
        if not isinstance(symbol_signals, dict) or not symbol_signals:
            return 'HOLD'

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
            signal = next(iter(symbol_signals.values()))

        if hasattr(signal, 'iloc'):
            try:
                signal = signal.iloc[-1]
            except Exception:
                signal = None

        if isinstance(signal, (int, float, np.integer, np.floating)):
            if signal >= 2:
                return 'OPEN_LONG'
            if signal == 1:
                return 'CLOSE_SHORT'
            if signal == -1:
                return 'CLOSE_LONG'
            if signal <= -2:
                return 'OPEN_SHORT'
            return 'HOLD'

        if signal in {'OPEN_LONG', 'CLOSE_LONG', 'OPEN_SHORT', 'CLOSE_SHORT', 'HOLD'}:
            return str(signal)
        return 'HOLD'

    def _resolve_position_size_pct(self) -> float:
        raw_pct = self.config.get('position_size_pct', self.config.get('risk_per_trade', 0.05))
        try:
            pct = float(raw_pct)
        except (TypeError, ValueError):
            pct = 0.05
        if pct > 1.0:
            pct = pct / 100.0

        if hasattr(self.strategy, 'get_strategy_state'):
            try:
                strategy_state = self.strategy.get_strategy_state()
            except Exception:
                strategy_state = {}
            if isinstance(strategy_state, dict):
                strategy_cap = strategy_state.get('effective_position_size_pct')
                if strategy_cap is not None:
                    try:
                        pct = min(pct, max(0.0, float(strategy_cap)))
                    except (TypeError, ValueError):
                        pass

        return max(0.0001, min(1.0, pct))

    def _estimate_24h_notional(self, df: pd.DataFrame, index: int, timeframe: str) -> float:
        if df is None or df.empty or index < 0:
            return 0.0

        timeframe_minutes = self._timeframe_to_minutes(timeframe)
        bars = max(1, int((24 * 60) / timeframe_minutes))
        start_idx = max(0, index - bars + 1)
        window = df.iloc[start_idx:index + 1]
        if window.empty:
            return 0.0

        for col in ('quote_volume', 'quoteVolume', 'turnover', 'notional', 'value'):
            if col in window.columns:
                values = pd.to_numeric(window[col], errors='coerce').fillna(0.0)
                return float(values.sum())

        if 'volume' in window.columns and 'close' in window.columns:
            volume = pd.to_numeric(window['volume'], errors='coerce').fillna(0.0)
            close = pd.to_numeric(window['close'], errors='coerce').fillna(0.0)
            return float((volume * close).sum())

        return 0.0

    def _total_open_notional(self) -> float:
        return float(
            sum(
                abs(float(pos.get('entry_price', 0.0)) * float(pos.get('quantity', 0.0)))
                for pos in self.positions.values()
            )
        )

    def _total_open_risk(self) -> float:
        total = 0.0
        for pos in self.positions.values():
            entry_price = float(pos.get('entry_price', 0.0))
            quantity = float(pos.get('quantity', 0.0))
            stop_loss_pct = float(pos.get('stop_loss_pct', self.global_rules.position_stop_loss_pct))
            total += abs(entry_price * quantity) * max(0.0, stop_loss_pct)
        return float(total)

    def _cap_quantity_with_global_rules(
        self,
        entry_price: float,
        quantity: float,
    ) -> Tuple[float, Optional[str]]:
        """
        Reduce order size to fit global exposure/risk caps.
        Returns: (adjusted_quantity, block_reason_if_any)
        """
        if entry_price <= 0.0 or quantity <= 0.0:
            return 0.0, 'invalid_order_size'

        rules = self.global_rules
        if not rules.enabled:
            return quantity, None

        base_balance = float(getattr(self, 'initial_balance', getattr(self, 'balance', 0.0)))
        if base_balance <= 0.0:
            return 0.0, 'invalid_account_balance'

        planned_notional = abs(entry_price * quantity)
        adjusted_notional = planned_notional

        max_notional_allowed = max(0.0, base_balance * float(rules.max_notional_exposure_pct))
        if max_notional_allowed > 0.0:
            remaining_notional = max(0.0, max_notional_allowed - self._total_open_notional())
            if remaining_notional <= 0.0:
                return 0.0, 'max_notional_exposure_exceeded'
            adjusted_notional = min(adjusted_notional, remaining_notional)

        stop_loss_pct = max(0.0, float(rules.position_stop_loss_pct))
        max_total_open_risk = max(0.0, base_balance * float(rules.max_total_open_risk_pct))
        if max_total_open_risk > 0.0 and stop_loss_pct > 0.0:
            remaining_risk = max(0.0, max_total_open_risk - self._total_open_risk())
            if remaining_risk <= 0.0:
                return 0.0, 'max_total_open_risk_exceeded'
            risk_capped_notional = remaining_risk / stop_loss_pct
            adjusted_notional = min(adjusted_notional, max(0.0, risk_capped_notional))

        if adjusted_notional <= 0.0:
            return 0.0, 'invalid_order_size'

        return float(adjusted_notional / entry_price), None

    def _expected_move_pct(self) -> float:
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

    def _record_global_block(self, reason: str) -> None:
        if not reason:
            return
        self._global_rule_stats['blocked_signals'] += 1
        blocked = self._global_rule_stats.setdefault('blocked_reasons', {})
        blocked[reason] = int(blocked.get(reason, 0)) + 1

    def _can_open_with_global_rules(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        row_index: int,
        entry_price: float,
        quantity: float,
    ) -> Tuple[bool, str]:
        rules = self.global_rules
        if not rules.enabled:
            return True, 'rules_disabled'

        if quantity <= 0.0 or entry_price <= 0.0:
            return False, 'invalid_order_size'

        if rules.one_position_per_symbol and symbol in self.positions:
            return False, 'position_already_open'

        min_notional = float(rules.min_24h_notional_usdt)
        if min_notional > 0.0:
            estimated_24h_notional = self._estimate_24h_notional(df=df, index=row_index, timeframe=timeframe)
            if estimated_24h_notional <= 0.0 or estimated_24h_notional < min_notional:
                return False, 'liquidity_below_min_notional'

        spread_pct = float(self.config.get('spread_pct', 0.0004))
        slippage_pct = float(self.config.get('slippage_pct', 0.0003))
        spread_bps = (max(0.0, spread_pct) + max(0.0, slippage_pct)) * 10000.0
        if rules.max_spread_bps > 0.0 and spread_bps > float(rules.max_spread_bps):
            return False, 'spread_above_limit'

        fee_pct = float(self.config.get('fee_pct', 0.00055))
        round_trip_cost_pct = rules.round_trip_cost_pct(fee_pct=fee_pct, spread_pct=spread_pct, slippage_pct=slippage_pct)
        expected_move_pct = self._expected_move_pct()
        if round_trip_cost_pct > 0.0 and rules.min_expected_move_to_cost_ratio > 0.0:
            ratio = expected_move_pct / round_trip_cost_pct
            if ratio < float(rules.min_expected_move_to_cost_ratio):
                return False, 'expected_move_below_cost_ratio'

        planned_notional = abs(entry_price * quantity)
        max_notional_allowed = max(0.0, float(self.initial_balance) * float(rules.max_notional_exposure_pct))
        if max_notional_allowed > 0.0:
            if (self._total_open_notional() + planned_notional) > (max_notional_allowed + 1e-9):
                return False, 'max_notional_exposure_exceeded'

        planned_risk = planned_notional * max(0.0, float(rules.position_stop_loss_pct))
        max_total_open_risk = max(0.0, float(self.initial_balance) * float(rules.max_total_open_risk_pct))
        if max_total_open_risk > 0.0:
            if (self._total_open_risk() + planned_risk) > (max_total_open_risk + 1e-9):
                return False, 'max_total_open_risk_exceeded'

        return True, 'allowed'

    def _check_emergency_close_reason(
        self,
        symbol: str,
        current_price: float,
        timestamp: Any,
    ) -> Optional[str]:
        rules = self.global_rules
        if not rules.enabled or not rules.emergency_close_enabled:
            return None

        position = self.positions.get(symbol)
        if position is None:
            return None

        entry_ts = position.get('entry_timestamp')
        if entry_ts is not None:
            try:
                hold_minutes = (pd.Timestamp(timestamp) - pd.Timestamp(entry_ts)).total_seconds() / 60.0
                if hold_minutes >= float(rules.max_hold_minutes):
                    return 'max_hold_minutes'
            except Exception:
                pass

        entry_price = float(position.get('entry_price', 0.0))
        if entry_price <= 0.0 or current_price <= 0.0:
            return None

        stop_loss_pct = max(0.0, float(position.get('stop_loss_pct', rules.position_stop_loss_pct)))
        is_short = bool(position.get('is_short', False))

        if is_short:
            stop_price = entry_price * (1.0 + stop_loss_pct)
            if current_price >= stop_price:
                return 'stop_loss_hit'
        else:
            stop_price = entry_price * (1.0 - stop_loss_pct)
            if current_price <= stop_price:
                return 'stop_loss_hit'

        return None

    def _close_position(self, symbol: str, current_price: float, timestamp: Any, reason: str) -> bool:
        position = self.positions.get(symbol)
        if position is None:
            return False

        is_short = bool(position.get('is_short', False))
        entry_price = float(position.get('entry_price', 0.0))
        quantity = float(position.get('quantity', 0.0))
        entry_timestamp = position.get('entry_timestamp')
        if quantity <= 0.0:
            self.positions.pop(symbol, None)
            return False

        execution_side = 'price_up' if is_short else 'price_down'
        exit_price = self.get_execution_price(float(current_price), execution_side)
        gross_pnl = (entry_price - exit_price) * quantity if is_short else (exit_price - entry_price) * quantity
        fee = self.apply_fees(exit_price * quantity)
        net_pnl = gross_pnl - fee
        self.balance += net_pnl

        direction = 'CLOSE_SHORT' if is_short else 'CLOSE_LONG'
        trade_record = {
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': quantity,
            'entry_timestamp': entry_timestamp,
            'exit_timestamp': timestamp,
            'pnl': net_pnl,
            'exit_reason': reason,
        }
        self.performance_tracker.record_trade(trade_record)
        self.positions.pop(symbol, None)
        print(f"CLOSE {direction}: {symbol} at {exit_price} (PnL: {net_pnl:.4f}, reason: {reason})")
        return True

    def _build_entry_risk_plan(
        self,
        *,
        position_type: str,
        entry_price: float,
        entry_window: pd.DataFrame,
    ) -> Tuple[Dict[str, Any], float, float]:
        default_stop_loss_pct = (
            self.global_rules.position_stop_loss_pct
            if self.global_rules.enabled
            else 0.01
        )
        risk_exit_mode = self.config.get('risk_exit_mode', 'none')
        atr_value = None
        if str(risk_exit_mode).strip().lower() == 'atr':
            atr_value = calculate_atr_from_dataframe(
                entry_window,
                int(self.config.get('risk_atr_period', 14)),
            )

        risk_config = build_position_risk_config(
            entry_price=entry_price,
            position_type=position_type,
            risk_exit_mode=risk_exit_mode,
            risk_sl_pct=self.config.get('risk_sl_pct', default_stop_loss_pct),
            risk_atr_period=int(self.config.get('risk_atr_period', 14)),
            risk_atr_sl_multiplier=self.config.get('risk_atr_sl_multiplier', 1.3),
            risk_atr_tp_multiplier=self.config.get('risk_atr_tp_multiplier', 1.7),
            atr_value=atr_value,
        )

        stop_loss_pct = float(risk_config.get('stop_loss_pct', default_stop_loss_pct))
        if str(position_type).upper() == 'SHORT':
            fallback_stop = entry_price * (1.0 + stop_loss_pct)
        else:
            fallback_stop = entry_price * (1.0 - stop_loss_pct)
        stop_price = float(risk_config.get('risk_stop_price', fallback_stop))
        return risk_config, stop_loss_pct, stop_price

    def _build_backtest_symbol_states(
        self,
        strategy: StrategyBase,
        data_with_indicators: Dict[str, Dict[str, pd.DataFrame]],
        timeframes: Optional[List[str]],
    ) -> Tuple[Dict[str, Dict[str, List[Any]]], Dict[str, Dict[str, Any]], List[Tuple[pd.Timestamp, str]], int, str, List[str]]:
        entry_timeframe = self._resolve_entry_timeframe(strategy)
        strategy_timeframes = self._collect_strategy_timeframes(strategy, fallback_timeframes=timeframes)
        signal_history = {
            symbol: {entry_timeframe: []}
            for symbol in data_with_indicators.keys()
        }

        use_full_signal_path = self._strategy_requires_full_signal_path(strategy)
        symbol_states: Dict[str, Dict[str, Any]] = {}
        event_heap: List[Tuple[pd.Timestamp, str]] = []
        total_rows = 0

        for symbol, symbol_timeframe_data in data_with_indicators.items():
            entry_key = self._resolve_timeframe_key(symbol_timeframe_data, entry_timeframe)
            if entry_key is None and symbol_timeframe_data:
                entry_key = next(iter(symbol_timeframe_data))
            if entry_key is None:
                continue

            entry_df = symbol_timeframe_data.get(entry_key)
            if entry_df is None or entry_df.empty:
                continue

            valid_rows = entry_df['indicators_valid'] if 'indicators_valid' in entry_df.columns else pd.Series(False, index=entry_df.index)
            valid_indices = np.flatnonzero(valid_rows.to_numpy())
            if len(valid_indices) == 0:
                continue

            first_valid_pos = int(valid_indices[0])
            signals_series = None
            if not use_full_signal_path:
                if hasattr(strategy, 'builder'):
                    try:
                        builder_df = entry_df.copy()
                        strategy.builder._calculate_indicators(builder_df)
                        signals_series = strategy.builder._execute_signal_rules(builder_df)
                    except Exception:
                        signals_series = None
                elif hasattr(strategy, 'generate_signals_vectorized'):
                    try:
                        signals_map = strategy.generate_signals_vectorized({symbol: {entry_key: entry_df}})
                        symbol_signals = signals_map.get(symbol, {})
                        signals_series = symbol_signals.get(entry_key)
                        if signals_series is None:
                            signals_series = symbol_signals.get(self._normalize_timeframe(entry_key))
                        if signals_series is None:
                            signals_series = symbol_signals.get(self._timeframe_with_suffix(entry_key))
                    except Exception:
                        signals_series = None

            symbol_states[symbol] = {
                'symbol': symbol,
                'entry_timeframe': entry_key,
                'entry_df': entry_df,
                'symbol_timeframe_data': symbol_timeframe_data,
                'row_index': first_valid_pos,
                'signals_series': signals_series,
            }
            total_rows += max(0, len(entry_df) - first_valid_pos)
            heapq.heappush(event_heap, (pd.Timestamp(entry_df.index[first_valid_pos]), symbol))

        return signal_history, symbol_states, event_heap, total_rows, entry_timeframe, strategy_timeframes

    def _queue_next_backtest_event(
        self,
        event_heap: List[Tuple[pd.Timestamp, str]],
        state: Dict[str, Any],
    ) -> None:
        state['row_index'] = int(state.get('row_index', 0)) + 1
        entry_df = state.get('entry_df')
        symbol = state.get('symbol')
        row_index = int(state.get('row_index', 0))
        if entry_df is None or symbol is None:
            return
        if row_index < len(entry_df):
            heapq.heappush(event_heap, (pd.Timestamp(entry_df.index[row_index]), symbol))

    def _open_backtest_position(
        self,
        *,
        strategy: StrategyBase,
        symbol: str,
        signal: str,
        current_price: float,
        timestamp: Any,
        timeframe: str,
        entry_df: pd.DataFrame,
        row_index: int,
        entry_window: pd.DataFrame,
    ) -> bool:
        max_positions = self.config.get('max_positions', 3)
        if len(self.positions) >= max_positions or symbol in self.positions:
            return False

        is_short = signal == 'OPEN_SHORT'
        position_type = 'SHORT' if is_short else 'LONG'
        execution_side = 'price_down' if is_short else 'price_up'
        entry_price = self.get_execution_price(current_price, execution_side)
        risk_config, stop_loss_pct, stop_price = self._build_entry_risk_plan(
            position_type=position_type,
            entry_price=entry_price,
            entry_window=entry_window,
        )

        quantity = self.get_order_size(symbol, entry_price, stop_price)
        quantity, cap_block_reason = self._cap_quantity_with_global_rules(
            entry_price=entry_price,
            quantity=quantity,
        )
        if quantity <= 0:
            self._record_global_block(cap_block_reason or 'invalid_order_size')
            return False

        allowed, reason = self._can_open_with_global_rules(
            symbol=symbol,
            timeframe=timeframe,
            df=entry_df,
            row_index=row_index,
            entry_price=entry_price,
            quantity=quantity,
        )
        if not allowed:
            self._record_global_block(reason)
            return False

        fee = self.apply_fees(entry_price * quantity)
        self.balance -= fee
        position = {
            'entry_price': entry_price,
            'quantity': quantity,
            'entry_timestamp': timestamp,
            'is_short': is_short,
            'type': position_type,
            'stop_loss_pct': stop_loss_pct,
            'owner_strategy': strategy.name if hasattr(strategy, 'name') else strategy.__class__.__name__,
            'entry_timeframe': timeframe,
        }
        position.update(risk_config)
        self.positions[symbol] = position
        print(f"{signal}: {symbol} at {entry_price} (qty: {quantity})")
        return True

    def _run_entry_timeframe_backtest_loop(
        self,
        *,
        strategy: StrategyBase,
        signal_history: Dict[str, Dict[str, List[Any]]],
        symbol_states: Dict[str, Dict[str, Any]],
        event_heap: List[Tuple[pd.Timestamp, str]],
        total_rows: int,
        strategy_timeframes: List[str],
        progress_callback=None,
    ) -> None:
        processed_rows = 0
        last_progress_update = 0

        while event_heap:
            timestamp, symbol = heapq.heappop(event_heap)
            state = symbol_states.get(symbol)
            if not state:
                continue

            timeframe = state['entry_timeframe']
            entry_df = state['entry_df']
            row_index = int(state['row_index'])
            if row_index >= len(entry_df):
                continue

            row = entry_df.iloc[row_index]
            processed_rows += 1
            progress_percent = (processed_rows / total_rows) * 100 if total_rows > 0 else 100.0
            if progress_callback and (progress_percent - last_progress_update >= 1 or progress_percent == 100):
                progress_callback(progress_percent)
                last_progress_update = progress_percent

            entry_window = entry_df.iloc[:row_index + 1]
            signal = 'HOLD'
            signals_series = state.get('signals_series')
            if signals_series is not None and len(signals_series) > row_index:
                signal = self._extract_signal_value(
                    {symbol: {timeframe: signals_series.iloc[row_index]}},
                    symbol=symbol,
                    entry_timeframe=timeframe,
                )
            else:
                current_data = self._build_strategy_snapshot(
                    state['symbol_timeframe_data'],
                    strategy_timeframes,
                    timestamp,
                )
                if current_data:
                    signals = strategy.generate_signals({symbol: current_data})
                    signal = self._extract_signal_value(
                        signals,
                        symbol=symbol,
                        entry_timeframe=timeframe,
                    )

            if symbol not in signal_history:
                signal_history[symbol] = {}
            if timeframe not in signal_history[symbol]:
                signal_history[symbol][timeframe] = []
            signal_history[symbol][timeframe].append((timestamp, signal))

            if self.debug_signals:
                self._debug_signal(symbol, timeframe, timestamp, signal, row, signal_history[symbol][timeframe][-5:])

            current_price = float(row.get('close', 0.0) or 0.0)
            if current_price <= 0.0:
                self._queue_next_backtest_event(event_heap, state)
                continue

            high_price = float(row.get('high', current_price) or current_price)
            low_price = float(row.get('low', current_price) or current_price)

            position = self.positions.get(symbol)
            if position is not None:
                risk_exit = evaluate_risk_exit(
                    position,
                    high=high_price,
                    low=low_price,
                    current_price=current_price,
                )
                if risk_exit:
                    if self._close_position(
                        symbol=symbol,
                        current_price=float(risk_exit.get('trigger_price', current_price) or current_price),
                        timestamp=timestamp,
                        reason=str(risk_exit.get('reason', 'configured_risk_exit')),
                    ):
                        self._global_rule_stats['emergency_closes'] += 1

            emergency_reason = self._check_emergency_close_reason(
                symbol=symbol,
                current_price=current_price,
                timestamp=timestamp,
            )
            if emergency_reason:
                if self._close_position(
                    symbol=symbol,
                    current_price=current_price,
                    timestamp=timestamp,
                    reason=f"emergency_{emergency_reason}",
                ):
                    self._global_rule_stats['emergency_closes'] += 1

            if signal in ['CLOSE_LONG', 'CLOSE_SHORT']:
                if symbol not in self.positions:
                    self._queue_next_backtest_event(event_heap, state)
                    continue
                if signal == 'CLOSE_LONG' and self.positions[symbol].get('is_short', False):
                    self._queue_next_backtest_event(event_heap, state)
                    continue
                if signal == 'CLOSE_SHORT' and not self.positions[symbol].get('is_short', False):
                    self._queue_next_backtest_event(event_heap, state)
                    continue

            if signal == 'OPEN_LONG':
                self._open_backtest_position(
                    strategy=strategy,
                    symbol=symbol,
                    signal=signal,
                    current_price=current_price,
                    timestamp=timestamp,
                    timeframe=timeframe,
                    entry_df=entry_df,
                    row_index=row_index,
                    entry_window=entry_window,
                )
            elif signal == 'OPEN_SHORT':
                self._open_backtest_position(
                    strategy=strategy,
                    symbol=symbol,
                    signal=signal,
                    current_price=current_price,
                    timestamp=timestamp,
                    timeframe=timeframe,
                    entry_df=entry_df,
                    row_index=row_index,
                    entry_window=entry_window,
                )
            elif signal == 'CLOSE_LONG':
                self._close_position(
                    symbol=symbol,
                    current_price=current_price,
                    timestamp=timestamp,
                    reason='strategy_signal',
                )
            elif signal == 'CLOSE_SHORT':
                self._close_position(
                    symbol=symbol,
                    current_price=current_price,
                    timestamp=timestamp,
                    reason='strategy_signal',
                )

            self._queue_next_backtest_event(event_heap, state)


    def run_backtest(self, symbols, timeframes, start_date, end_date, config=None, strategy=None, data=None, initial_balance=None, progress_callback=None):
        """
        Run backtest with optimized parameters loading and progress callback
        """
        import time
        from datetime import datetime, timedelta
        
        # Add timing at the very beginning
        start_time = time.time()
        start_dt = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
        
        # Load optimized parameters if available
        try:
            from simple_strategy.trading.parameter_manager import ParameterManager
            pm = ParameterManager()
            strategy_name = self.strategy.name if hasattr(self.strategy, 'name') else self.strategy.__class__.__name__
            optimized_params = pm.get_parameters(strategy_name)
            
            if optimized_params and 'last_optimized' in optimized_params:
                strategy_params = {k: v for k, v in optimized_params.items() if k != 'last_optimized'}
                for param, value in strategy_params.items():
                    if hasattr(self.strategy, param):
                        setattr(self.strategy, param, value)
                    elif hasattr(self.strategy, 'params'):
                        self.strategy.params[param] = value
        except Exception as e:
            pass # Silently fail parameter loading to keep flow going
        
        try:
            self._refresh_runtime_config(config)
            # Initial balance
            initial_balance = initial_balance or (config['initial_balance'] if config and 'initial_balance' in config else 10000.0)
            
            # Strategy selection
            if strategy is None:
                strategy = self.strategy
            else:
                pass # Using provided strategy
            
            # Load data
            data_load_start = time.time()
            if data is None:
                data = self.data_feeder.get_data_for_symbols(
                    symbols or strategy.symbols, 
                    timeframes or strategy.timeframes,
                    start_date or datetime.now() - timedelta(days=30),
                    end_date or datetime.now()
                )
            
            # Pre-calculate all indicators for all symbols/timeframes
            print("Pre-calculating indicators for all symbols/timeframes...")
            data_with_indicators = self._precalculate_indicators(data)
            print("Indicator calculation completed.")
            
            # Count total data points
            total_data_points = sum(len(data[s][t]) for s in data for t in data[s])
            
            # Initialize variables
            self.trades = []
            self.balance = initial_balance
            self.initial_balance = initial_balance
            self.equity = initial_balance
            self.peak_equity = initial_balance
            self.max_drawdown = 0.0
            self.performance_tracker = PerformanceTracker(initial_balance=initial_balance)
            self.positions = {}
            self._debug_count = []
            self._global_rule_stats['blocked_signals'] = 0
            self._global_rule_stats['blocked_reasons'] = {}
            self._global_rule_stats['emergency_closes'] = 0
            
            signal_history, symbol_states, event_heap, total_rows, entry_timeframe, strategy_timeframes = (
                self._build_backtest_symbol_states(
                    strategy,
                    data_with_indicators,
                    timeframes,
                )
            )

            self._run_entry_timeframe_backtest_loop(
                strategy=strategy,
                signal_history=signal_history,
                symbol_states=symbol_states,
                event_heap=event_heap,
                total_rows=total_rows,
                strategy_timeframes=strategy_timeframes,
                progress_callback=progress_callback,
            )

            for symbol in list(self.positions.keys()):
                state = symbol_states.get(symbol)
                if not state:
                    continue
                entry_df = state['entry_df']
                current_price = float(entry_df.iloc[-1]['close'])
                self._close_position(
                    symbol=symbol,
                    current_price=current_price,
                    timestamp=entry_df.index[-1],
                    reason='end_of_backtest',
                )
            
            # === FINAL EQUITY & DRAWDOWN UPDATE ===
            self.equity = self.balance
            if self.equity > self.peak_equity:
                self.peak_equity = self.equity
            final_drawdown = (self.peak_equity - self.equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, final_drawdown)
            
            # === CALCULATE PERFORMANCE METRICS ===
            metrics = self._calculate_performance_metrics()
            
            # === TOTAL TIMING ===
            total_time = time.time() - start_time
            end_dt = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            total_seconds = total_time
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = total_seconds % 60
            duration_str = f"{hours}h {minutes}m {seconds:.2f}s"
            
            # === SAVE BACKTEST RESULTS ===
            try:
                strategy_name = strategy.name if hasattr(strategy, 'name') else strategy.__class__.__name__
                self._save_backtest_results(
                    strategy_name=strategy_name,
                    symbols=symbols or strategy.symbols,
                    timeframes=timeframes or strategy.timeframes,
                    start_date=start_date,
                    end_date=end_date,
                    metrics=metrics,
                    duration=duration_str
                )
            except Exception:
                pass

            # === RETURN METRICS ===
            return {
                'win_rate': metrics['win_rate_pct'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown_pct'],
                'total_return': metrics['total_return_pct'],
                'total_trades': metrics['total_trades'],
                'start_time': start_dt,
                'end_time': end_dt,
                'duration': duration_str,
                'global_rules': {
                    'enabled': bool(self.global_rules.enabled),
                    'profile': str(self.global_rules.profile),
                    'blocked_signals': int(self._global_rule_stats.get('blocked_signals', 0)),
                    'blocked_reasons': dict(self._global_rule_stats.get('blocked_reasons', {})),
                    'emergency_closes': int(self._global_rule_stats.get('emergency_closes', 0)),
                },
            }

        except Exception as e:
            print(f"Error in backtest: {e}")
            import traceback
            traceback.print_exc()
            return {
                'win_rate': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_return': 0.0,
                'total_trades': 0,
                'global_rules': {
                    'enabled': bool(self.global_rules.enabled),
                    'profile': str(self.global_rules.profile),
                    'blocked_signals': int(self._global_rule_stats.get('blocked_signals', 0)),
                    'blocked_reasons': dict(self._global_rule_stats.get('blocked_reasons', {})),
                    'emergency_closes': int(self._global_rule_stats.get('emergency_closes', 0)),
                },
            }

    
    def _validate_data(self, data: Dict[str, Dict[str, pd.DataFrame]], 
                      symbols: List[str], timeframes: List[str]) -> bool:
        """
        Validate loaded data structure and content
        
        Args:
            data: Data dictionary from DataFeeder
            symbols: Expected symbols
            timeframes: Expected timeframes
            
        Returns:
            True if data is valid, False otherwise
        """
        logger.info("Validating data structure...")
        
        # Check all symbols are present
        for symbol in symbols:
            if symbol not in data:
                logger.error(f"Missing data for symbol: {symbol}")
                return False
            
            # Check all timeframes are present for each symbol
            for timeframe in timeframes:
                if timeframe not in data[symbol]:
                    logger.error(f"Missing data for {symbol} timeframe: {timeframe}")
                    return False
                
                # Check DataFrame is not empty
                df = data[symbol][timeframe]
                if df.empty:
                    logger.error(f"Empty DataFrame for {symbol} {timeframe}")
                    return False
                
                # Check required columns
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    logger.error(f"Missing columns for {symbol} {timeframe}: {missing_columns}")
                    return False
        
        logger.info("Data validation passed")
        return True
    
    
        
    def _can_execute_trade(self, symbol, signal, timestamp, current_data):
        """
        Determine if a trade can be executed.
        """
        has_position = symbol in self.positions

        if signal == 'OPEN_LONG' and has_position and not self.positions[symbol]['is_short']:
            return False
        if signal == 'OPEN_SHORT' and has_position and self.positions[symbol]['is_short']:
            return False
        if signal == 'CLOSE_LONG' and (not has_position or self.positions[symbol]['is_short']):
            return False
        if signal == 'CLOSE_SHORT' and (not has_position or not self.positions[symbol]['is_short']):
            return False

        return True

    
    def _get_data_for_timestamp(self, data: Dict[str, Dict[str, pd.DataFrame]],
                           symbols: List[str], timeframes: List[str],
                           timestamp: datetime) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Get data for a specific timestamp across all symbols and timeframes
        Args:
            data: Full data dictionary
            symbols: Trading symbols
            timeframes: Timeframes
            timestamp: Target timestamp
        Returns:
            Data dictionary for the specific timestamp with historical data
        """
        timestamp_data = {}
        
        for symbol in symbols:
            timestamp_data[symbol] = {}
            
            for timeframe in timeframes:
                df = data[symbol][timeframe]
                
                # Get all data up to and including the target timestamp
                # This is crucial for indicators that need historical data
                mask = df.index <= timestamp
                
                if mask.any():
                    # Get all historical data up to this timestamp
                    historical_data = df[mask].copy()
                    timestamp_data[symbol][timeframe] = historical_data
                else:
                    # No data before this timestamp
                    timestamp_data[symbol][timeframe] = df.iloc[0:0]  # Empty DataFrame
        
        return timestamp_data
    
    def _get_current_price(self, symbol: str, current_data: Dict[str, Dict[str, pd.DataFrame]]) -> float:
        """Get current price for a symbol from the current data"""
        try:
            # Check if current_data is the expected structure
            if not isinstance(current_data, dict) or symbol not in current_data:
                logger.warning(f" Invalid data structure for {symbol}")
                return 50000.0  # Return a reasonable default price
            
            # Get the first timeframe data for this symbol
            if not current_data[symbol]:
                logger.warning(f" No timeframe data for {symbol}")
                return 50000.0
            
            # Get the first timeframe DataFrame
            timeframe_data = list(current_data[symbol].values())[0]
            
            if timeframe_data.empty:
                logger.warning(f" Empty DataFrame for {symbol}")
                return 50000.0
            
            # Get the last close price
            current_price = timeframe_data['close'].iloc[-1]
            
            # Ensure it's a valid price
            if pd.isna(current_price) or current_price <= 0:
                logger.warning(f" Invalid price {current_price} for {symbol}")
                return 50000.0
            
            return float(current_price)
            
        except Exception as e:
            logger.error(f" Error getting current price for {symbol}: {e}")
            # Return a reasonable default instead of 0
            return 50000.0
    
    
    def stop_backtest(self):
        """Stop the currently running backtest"""
        logger.info("Stopping backtest...")
        self.is_running = False

    def calculate_performance_metrics(self, trades, initial_balance, final_balance):
        """
        Calculate comprehensive performance metrics
        """
        if not trades:
            return {
                'win_rate': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_return': 0.0
            }
        
        # Calculate win rate
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0.0
        
        # Calculate total return
        total_return = ((final_balance - initial_balance) / initial_balance) * 100
        
        # Calculate Sharpe ratio (simplified)
        if len(trades) > 1:
            pnl_list = [t.get('pnl', 0) for t in trades]
            avg_return = sum(pnl_list) / len(pnl_list)
            std_return = (sum((x - avg_return) ** 2 for x in pnl_list) / len(pnl_list)) ** 0.5
            sharpe_ratio = (avg_return / std_return) * (252 ** 0.5) if std_return > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # Calculate max drawdown (simplified)
        max_drawdown = 0.0
        peak_balance = initial_balance
        
        for trade in trades:
            current_balance = trade.get('balance_after', initial_balance)
            if current_balance > peak_balance:
                peak_balance = current_balance
            
            drawdown = ((peak_balance - current_balance) / peak_balance) * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return {
            'win_rate': win_rate * 100,  # Convert to percentage
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_return': total_return
        }

    def display_results(self, performance_metrics):
        """
        Display backtest results in a formatted way
        """
        print("\n" + "="*50)
        print(" BACKTEST RESULTS")
        print("="*50)
        print(f" Total Return: {performance_metrics['total_return']:.2f}%")
        print(f" Win Rate: {performance_metrics['win_rate']:.2f}%")
        print(f" Sharpe Ratio: {performance_metrics['sharpe_ratio']:.2f}")
        print(f" Max Drawdown: {performance_metrics['max_drawdown']:.2f}%")
        print(f" Total Trades: {len(self.trades)}")
        print("="*50)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current backtest statusrun_backtest
        
        Returns:
            Status dictionary
        """
        return {
            'is_running': self.is_running,
            'current_timestamp': self.current_timestamp,
            'processing_stats': self.processing_stats,
            'strategy_state': self.strategy.get_strategy_state() if hasattr(self.strategy, 'get_strategy_state') else None
        }
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics from trades"""
        logger.info(f"DEBUG: Calculating metrics with {len(self.performance_tracker.trades)} trades")
        
        if not self.performance_tracker.trades:
            logger.info("DEBUG: No trades found in performance tracker")
            return {
                'total_return_pct': 0.0,
                'win_rate_pct': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown_pct': self.max_drawdown * 100,
                'total_trades': 0,
                'global_rules': {
                    'enabled': bool(self.global_rules.enabled),
                    'profile': str(self.global_rules.profile),
                    'blocked_signals': int(self._global_rule_stats.get('blocked_signals', 0)),
                    'blocked_reasons': dict(self._global_rule_stats.get('blocked_reasons', {})),
                    'emergency_closes': int(self._global_rule_stats.get('emergency_closes', 0)),
                },
            }
        
        trades = self.performance_tracker.trades
        total_trades = len(trades)
        logger.info(f"DEBUG: Processing {total_trades} trades")
        
        # Debug: Print first few trades
        for i, trade in enumerate(trades[:3]):
            logger.info(f"DEBUG: Trade {i}: {trade.direction} {trade.symbol} pnl={trade.pnl}")
        
        winning_trades = len([t for t in trades if t.pnl > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        total_pnl = sum(t.pnl for t in trades)
        logger.info(f"DEBUG: Total PnL: {total_pnl}")
        total_return_pct = (total_pnl / self.initial_balance * 100) if self.initial_balance > 0 else 0.0
        
        # Simple Sharpe ratio calculation (assuming risk-free rate = 0)
        pnl_list = [t.pnl for t in trades]
        sharpe_ratio = (np.mean(pnl_list) / np.std(pnl_list)) * np.sqrt(252) if len(pnl_list) > 1 and np.std(pnl_list) > 0 else 0.0
        
        metrics = {
            'total_return_pct': total_return_pct,
            'win_rate_pct': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': self.max_drawdown * 100,
            'total_trades': total_trades,
            'global_rules': {
                'enabled': bool(self.global_rules.enabled),
                'profile': str(self.global_rules.profile),
                'blocked_signals': int(self._global_rule_stats.get('blocked_signals', 0)),
                'blocked_reasons': dict(self._global_rule_stats.get('blocked_reasons', {})),
                'emergency_closes': int(self._global_rule_stats.get('emergency_closes', 0)),
            },
        }
        
        logger.info(f"DEBUG: Calculated metrics: {metrics}")
        return metrics

    def _save_backtest_results(self, strategy_name, symbols, timeframes, start_date, end_date, metrics, duration):
        results_path = os.path.join(os.path.dirname(__file__), '..', 'optimization_results', 'backtest_results.json')
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        def _as_str(value):
            try:
                return value.strftime('%Y-%m-%d')
            except Exception:
                return str(value)

        try:
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    data = json.load(f)
            else:
                data = {}
        except Exception:
            data = {}

        total_return_pct = metrics.get('total_return_pct', 0.0)
        initial_balance = float(getattr(self, 'initial_balance', 0.0))
        net_profit_loss = (initial_balance * total_return_pct / 100.0) if initial_balance else 0.0

        data[strategy_name] = {
            'last_backtest': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbols': list(symbols) if symbols is not None else [],
            'timeframes': list(timeframes) if timeframes is not None else [],
            'start_date': _as_str(start_date),
            'end_date': _as_str(end_date),
            'duration': duration,
            'initial_balance': initial_balance,
            'net_profit_loss': net_profit_loss,
            'metrics': {
                'win_rate': metrics.get('win_rate_pct', 0.0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
                'max_drawdown': metrics.get('max_drawdown_pct', 0.0),
                'total_return': metrics.get('total_return_pct', 0.0),
                'total_trades': metrics.get('total_trades', 0)
            }
        }

        with open(results_path, 'w') as f:
            json.dump(data, f, indent=4)

    def _display_results(self, metrics):
        """Display backtest results"""
        print("=" * 50)
        print(" BACKTEST RESULTS")
        print("=" * 50)
        print(f" Total Return: {metrics['total_return_pct']:.2f}%")
        print(f" Win Rate: {metrics['win_rate_pct']:.2f}%")
        print(f" Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f" Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        print(f" Total Trades: {metrics['total_trades']}")
        print("=" * 50)

    def get_execution_price(self, price: float, side: str) -> float:
        """
        Simulate realistic execution price with spread + slippage.
        side: 'price_up' or 'price_down'
        """
        # Conservative defaults for crypto perp trading
        spread_pct = self.config.get('spread_pct', 0.0004)      # 0.04%
        slippage_pct = self.config.get('slippage_pct', 0.0003)  # 0.03%

        if side == 'price_up':
            return price * (1 + spread_pct + slippage_pct)
        else:
            return price * (1 - spread_pct - slippage_pct)
 
    def apply_fees(self, notional_value: float) -> float:
        """
        Apply exchange trading fees.
        """
        fee_pct = self.config.get('fee_pct', 0.00055)  # 0.055% taker fee
        return notional_value * fee_pct
    
    def _precalculate_indicators(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Pre-calculate all indicators needed by the strategy with optimized approach
        """
        from simple_strategy.strategies.indicators_library import ema, rsi
        import numpy as np
        
        # Get strategy parameters
        fast_ma_period = getattr(self.strategy, 'fast_ema_period', 20)
        slow_ma_period = getattr(self.strategy, 'slow_ema_period', 50)
        rsi_period = getattr(self.strategy, 'rsi_period', 14)
        
        enhanced_data = {}
        
        for symbol in data:
            enhanced_data[symbol] = {}
            for timeframe in data[symbol]:
                df = data[symbol][timeframe].copy()
                
                if df.empty or 'close' not in df.columns:
                    # Keep expected columns so downstream code can skip cleanly.
                    df[f'ema_fast_{timeframe}'] = np.nan
                    df[f'ema_slow_{timeframe}'] = np.nan
                    df[f'rsi_{timeframe}'] = np.nan
                    df['indicators_valid'] = False
                    enhanced_data[symbol][timeframe] = df
                    continue

                # Calculate EMAs and RSI
                df[f'ema_fast_{timeframe}'] = ema(df['close'], period=fast_ma_period)
                df[f'ema_slow_{timeframe}'] = ema(df['close'], period=slow_ma_period)
                df[f'rsi_{timeframe}'] = rsi(df['close'], period=rsi_period)
                
                # Mark valid indicator rows. Some windows may not have enough candles.
                valid_candidates = [
                    df[f'ema_fast_{timeframe}'].first_valid_index(),
                    df[f'ema_slow_{timeframe}'].first_valid_index(),
                    df[f'rsi_{timeframe}'].first_valid_index(),
                ]
                valid_candidates = [idx for idx in valid_candidates if idx is not None]
                if valid_candidates:
                    min_valid_idx = max(valid_candidates)
                    df['indicators_valid'] = df.index >= min_valid_idx
                else:
                    df['indicators_valid'] = False
                
                enhanced_data[symbol][timeframe] = df
        
        return enhanced_data

    def _debug_signal(self, symbol, timeframe, timestamp, signal, row, recent_signals=None):
        """Debug method to check what signals are being generated - optimized version"""
        # Only debug for the first symbol and only trading signals
        if symbol != 'BNBUSDT' or signal == 'HOLD':
            return
        
        # Debug all trading signals with recent history
        if f'ema_fast_{timeframe}' in row.index:
            fast_ema = row.get(f'ema_fast_{timeframe}')
            slow_ema = row.get(f'ema_slow_{timeframe}')
            rsi = row.get(f'rsi_{timeframe}')
            
            print(f"\n=== TRADING SIGNAL: {symbol} {timeframe} at {timestamp} ===")
            print(f"Signal: {signal}")
            print(f"Fast EMA: {fast_ema:.2f}, Slow EMA: {slow_ema:.2f}, RSI: {rsi:.2f}")
            print(f"Trend: {'UP' if fast_ema > slow_ema else 'DOWN'}")
            
            if recent_signals:
                print(f"Recent signals: {recent_signals}")
