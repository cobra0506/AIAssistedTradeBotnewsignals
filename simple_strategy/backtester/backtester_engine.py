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







        self.positions = {}  # Track open positions by (symbol, timeframe)







        logger.info(f"BacktesterEngine initialized with strategy: {strategy.name}")







        logger.info(f"Risk management {'enabled'if self.risk_manager else'disabled'}")







    def get_order_size(self, symbol, entry_price, stop_price=None):







        """







        True risk-based position sizing.







        stop_price is OPTIONAL so legacy/backtest paths never crash.







        """







        risk_per_trade = self.config.get('risk_per_trade', 0.02)







        account_risk = self.balance * risk_per_trade







        # Fallback stop for backtester / safety







        if stop_price is None:







            stop_price = entry_price * 0.99  # 1% default stop







        stop_distance = abs(entry_price - stop_price)







        if stop_distance <= 0:







            return 0







        quantity = account_risk / stop_distance







        return quantity







    







    def _calculate_unrealized_pnl(self, current_prices):







        unrealized = 0.0







        for key, position in self.positions.items():







            symbol = position.get('symbol')







            if not symbol:







                symbol = key[0] if isinstance(key, tuple) else key







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







    def run_backtest(self, symbols, timeframes, start_date, end_date, config=None, strategy=None, data=None, initial_balance=None, progress_callback=None):







        """







        Run backtest with optimized parameters loading and progress callback







        """







        import time







        from datetime import datetime, timedelta







        







        # Add timing at the very beginning







        start_time = time.time()







        start_dt = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')







        







        try:







            _original_config = self.config.copy() if isinstance(self.config, dict) else {}







            if isinstance(config, dict):







                merged = _original_config.copy()







                merged.update(config)







                self.config = merged







            use_saved_params = self.config.get('use_saved_params', True)







            # Initial balance







            initial_balance = initial_balance or self.config.get('initial_balance', 10000.0)







            # execution_delay_bars is counted in bars of the SAME timeframe



            execution_delay_bars = int(self.config.get('execution_delay_bars', 1))



            if execution_delay_bars < 0:







                execution_delay_bars = 0







            use_intrabar_stops = self.config.get('use_intrabar_stops', True)







            stop_loss_pct = self.config.get('stop_loss_pct', None)







            take_profit_pct = self.config.get('take_profit_pct', None)







            if stop_loss_pct is not None and stop_loss_pct <= 0:







                stop_loss_pct = None







            if take_profit_pct is not None and take_profit_pct <= 0:







                take_profit_pct = None







            enable_liquidation = self.config.get('enable_liquidation', True)







            leverage = float(self.config.get('leverage', 5.0))







            if leverage <= 0:







                leverage = 1.0







            maintenance_margin_pct = float(self.config.get('maintenance_margin_pct', 0.005))



            if maintenance_margin_pct < 0:



                maintenance_margin_pct = 0.0



            strict_signal_schema = self.config.get('strict_signal_schema', True)



            







            # Strategy selection







            if strategy is None:







                strategy = self.strategy







            else:







                pass # Using provided strategy







            self.strategy = strategy







            







            # Load optimized parameters if enabled







            if use_saved_params:







                try:







                    from simple_strategy.trading.parameter_manager import ParameterManager







                    pm = ParameterManager()







                    strategy_name = strategy.name if hasattr(strategy, 'name') else strategy.__class__.__name__







                    optimized_params = pm.get_parameters(strategy_name)







                    







                    if optimized_params and 'last_optimized' in optimized_params:







                        strategy_params = {k: v for k, v in optimized_params.items() if k != 'last_optimized'}







                        for param, value in strategy_params.items():







                            if hasattr(strategy, param):







                                setattr(strategy, param, value)







                            elif hasattr(strategy, 'params'):







                                strategy.params[param] = value







                except Exception:







                    pass # Silently fail parameter loading to keep flow going







            







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







            







            def _make_position_key(symbol, timeframe):







                return (symbol, timeframe)















            def _slice_until(df, ts):







                if df is None:







                    return pd.DataFrame()







                if df.empty:







                    return df.iloc[0:0]







                pos = df.index.searchsorted(ts, side='right')







                if pos <= 0:







                    return df.iloc[0:0]







                return df.iloc[:pos]















            def _risk_check(symbol, timeframe, signal, price, timestamp):



                if not self.risk_manager:



                    return True



                account_state = {







                    'balance': self.balance,







                    'positions': self.positions







                }







                signal_payload = {







                    'symbol': symbol,







                    'timeframe': timeframe,







                    'signal_type': signal,







                    'price': price,







                    'timestamp': timestamp







                }







                try:







                    result = self.risk_manager.validate_trade_signal(signal_payload, account_state)







                    return bool(result.get('valid', False))







                except Exception:



                    return True



            



            valid_signals = {'OPEN_LONG', 'CLOSE_LONG', 'OPEN_SHORT', 'CLOSE_SHORT', 'HOLD'}



            



            def _normalize_signal(raw_signal, symbol, timeframe, timestamp):



                if raw_signal is None:



                    if strict_signal_schema:



                        raise ValueError(f"Invalid signal None for {symbol} {timeframe} at {timestamp}. Use only {sorted(valid_signals)}.")



                    return 'HOLD'



                if isinstance(raw_signal, (int, float, np.integer, np.floating, np.number)):



                    if strict_signal_schema:



                        raise ValueError(f"Numeric signal {raw_signal} for {symbol} {timeframe} at {timestamp}. Use only {sorted(valid_signals)}.")



                    if self.debug_signals:



                        print(f"WARNING: Numeric signal {raw_signal} for {symbol} {timeframe}, treating as HOLD.")



                    return 'HOLD'



                if isinstance(raw_signal, str):



                    raw_signal = raw_signal.strip()



                if raw_signal not in valid_signals:



                    if strict_signal_schema:



                        raise ValueError(f"Invalid signal {raw_signal} for {symbol} {timeframe} at {timestamp}. Use only {sorted(valid_signals)}.")



                    if self.debug_signals:



                        print(f"WARNING: Invalid signal {raw_signal} for {symbol} {timeframe}, treating as HOLD.")



                    return 'HOLD'



                return raw_signal











            def _execute_open(symbol, timeframe, timestamp, base_price, is_short):







                position_key = _make_position_key(symbol, timeframe)







                max_positions = self.config.get('max_positions', 3)







                if len(self.positions) >= max_positions:







                    return







                if position_key in self.positions:







                    return







                action = 'OPEN_SHORT' if is_short else 'OPEN_LONG'







                entry_price = self.get_execution_price(base_price, action)







                if not _risk_check(symbol, timeframe, action, entry_price, timestamp):







                    return







                stop_price = None







                take_price = None







                if stop_loss_pct is not None:







                    if is_short:







                        stop_price = entry_price * (1 + stop_loss_pct)







                    else:







                        stop_price = entry_price * (1 - stop_loss_pct)







                if take_profit_pct is not None:







                    if is_short:







                        take_price = entry_price * (1 - take_profit_pct)







                    else:







                        take_price = entry_price * (1 + take_profit_pct)







                liquidation_price = None







                if enable_liquidation and leverage > 1:







                    if is_short:







                        liquidation_price = entry_price * (1 + (1 / leverage) - maintenance_margin_pct)







                        if liquidation_price <= entry_price:







                            liquidation_price = None







                    else:







                        liquidation_price = entry_price * (1 - (1 / leverage) + maintenance_margin_pct)







                        if liquidation_price >= entry_price or liquidation_price <= 0:







                            liquidation_price = None







                quantity = self.get_order_size(symbol, entry_price, stop_price)







                if quantity <= 0:







                    return







                fee = self.apply_fees(entry_price * quantity)







                available_balance = self.balance - fee







                if available_balance <= 0:







                    return







                # Reserve margin for the position







                required_margin = (entry_price * quantity) / leverage







                if required_margin > available_balance:







                    quantity = (available_balance * leverage) / entry_price







                    if quantity <= 0:







                        return







                    fee = self.apply_fees(entry_price * quantity)







                    available_balance = self.balance - fee







                    required_margin = (entry_price * quantity) / leverage







                    if required_margin > available_balance:







                        return







                self.balance -= (fee + required_margin)







                self.positions[position_key] = {







                    'symbol': symbol,







                    'timeframe': timeframe,







                    'entry_price': entry_price,







                    'quantity': quantity,







                    'entry_timestamp': timestamp,







                    'is_short': is_short,







                    'stop_price': stop_price,







                    'take_profit_price': take_price,







                    'liquidation_price': liquidation_price,







                    'leverage': leverage,







                    'margin_used': required_margin







                }







                if is_short:







                    if self.debug_signals:
                        print(f"OPEN_SHORT: {symbol} {timeframe} at {entry_price} (qty: {quantity})")







                else:







                    if self.debug_signals:
                        print(f"OPEN_LONG: {symbol} {timeframe} at {entry_price} (qty: {quantity})")















            def _execute_close(symbol, timeframe, timestamp, base_price, reason_label):







                position_key = _make_position_key(symbol, timeframe)







                position = self.positions.get(position_key)







                if not position:







                    return







                entry_price = position['entry_price']







                quantity = position['quantity']







                is_short = position.get('is_short', False)







                action = 'CLOSE_SHORT' if is_short else 'CLOSE_LONG'







                exit_price = self.get_execution_price(base_price, action)







                if not _risk_check(symbol, timeframe, action, exit_price, timestamp):







                    return







                position = self.positions.pop(position_key)







                if is_short:







                    gross_pnl = (entry_price - exit_price) * quantity







                    direction = 'CLOSE_SHORT'







                else:







                    gross_pnl = (exit_price - entry_price) * quantity







                    direction = 'CLOSE_LONG'







                fee = self.apply_fees(exit_price * quantity)







                net_pnl = gross_pnl - fee







                self.balance += net_pnl + position.get('margin_used', 0.0)







                self.performance_tracker.record_trade({



                    'symbol': symbol, 'timeframe': timeframe, 'direction': direction,



                    'entry_price': entry_price, 'exit_price': exit_price,



                    'size': quantity, 'entry_timestamp': position['entry_timestamp'],



                    'exit_timestamp': timestamp, 'pnl': net_pnl



                })



                if self.debug_signals:
                    print(f"{reason_label}: {symbol} {timeframe} at {exit_price} (PnL: {net_pnl})")















            # Track signal history for debugging







            signal_history = None
            if self.debug_signals:
                signal_history = {symbol: {timeframe: [] for timeframe in timeframes} for symbol in symbols}














            use_builder_signals = hasattr(strategy, 'builder')







            signals_series_map = {}







            signals_map = None







            if use_builder_signals:







                for symbol in data_with_indicators:







                    signals_series_map[symbol] = {}







                    for timeframe in data_with_indicators[symbol]:







                        df = data_with_indicators[symbol][timeframe]







                        try:







                            signals_series_map[symbol][timeframe] = strategy.builder._execute_signal_rules(df)







                        except Exception:







                            signals_series_map[symbol][timeframe] = None







            elif hasattr(strategy, 'generate_signals_vectorized'):







                try:







                    signals_map = strategy.generate_signals_vectorized(data_with_indicators)







                except Exception:







                    signals_map = None















            # Build global time-ordered events







            events = []







            for symbol in data_with_indicators:







                for timeframe in data_with_indicators[symbol]:







                    df = data_with_indicators[symbol][timeframe]







                    if df is None or df.empty:







                        continue







                    if 'indicators_valid' in df.columns and df['indicators_valid'].any():







                        first_valid_ts = df[df['indicators_valid']].index[0]







                        start_idx = df.index.get_loc(first_valid_ts)







                    else:







                        start_idx = 0







                    for i in range(start_idx, len(df)):







                        events.append((df.index[i], symbol, timeframe, i))







            events.sort(key=lambda x: x[0])















            total_rows = len(events)







            processed_rows = 0







            last_progress_update = 0







            last_prices = {}







            pending_orders = {}















            for timestamp, symbol, timeframe, i in events:







                df = data_with_indicators[symbol][timeframe]







                row = df.iloc[i]







                open_price = row['open']







                high_price = row['high']







                low_price = row['low']







                close_price = row['close']







                processed_rows += 1







                progress_percent = (processed_rows / total_rows) * 100 if total_rows > 0 else 100.0















                # UPDATE: Call the progress callback if provided







                if progress_callback and (progress_percent - last_progress_update >= 1 or progress_percent == 100):







                    progress_callback(progress_percent)







                    last_progress_update = progress_percent















                position_key = _make_position_key(symbol, timeframe)







                pending_list = pending_orders.get(position_key, [])







                if pending_list:



                    due_orders = [o for o in pending_list if o['due_index'] <= i]



                    pending_orders[position_key] = [o for o in pending_list if o['due_index'] > i]



                    for order in due_orders:



                        if order['signal'] == 'OPEN_LONG':



                            _execute_open(symbol, timeframe, timestamp, open_price, is_short=False)



                        elif order['signal'] == 'OPEN_SHORT':



                            _execute_open(symbol, timeframe, timestamp, open_price, is_short=True)



                        elif order['signal'] == 'CLOSE_LONG':



                            _execute_close(symbol, timeframe, timestamp, open_price, 'CLOSE_LONG (DELAYED)')







                        elif order['signal'] == 'CLOSE_SHORT':







                            _execute_close(symbol, timeframe, timestamp, open_price, 'CLOSE_SHORT (DELAYED)')















                # Intrabar liquidation check (uses high/low)







                position = self.positions.get(position_key)







                if enable_liquidation and position:







                    liquidation_price = position.get('liquidation_price')







                    if liquidation_price is not None:







                        is_short = position.get('is_short', False)







                        if not is_short and low_price <= liquidation_price:







                            _execute_close(symbol, timeframe, timestamp, liquidation_price, 'LIQUIDATION')







                        elif is_short and high_price >= liquidation_price:







                            _execute_close(symbol, timeframe, timestamp, liquidation_price, 'LIQUIDATION')















                # Intrabar stop/target checks (uses high/low for realism)







                if use_intrabar_stops and position:







                    stop_price = position.get('stop_price')







                    take_price = position.get('take_profit_price')







                    if stop_price is not None or take_price is not None:







                        is_short = position.get('is_short', False)







                        if not is_short:







                            stop_hit = stop_price is not None and low_price <= stop_price







                            take_hit = take_price is not None and high_price >= take_price







                            if stop_hit:







                                _execute_close(symbol, timeframe, timestamp, stop_price, 'STOP_LOSS')







                            elif take_hit:







                                _execute_close(symbol, timeframe, timestamp, take_price, 'TAKE_PROFIT')







                        else:







                            stop_hit = stop_price is not None and high_price >= stop_price







                            take_hit = take_price is not None and low_price <= take_price







                            if stop_hit:







                                _execute_close(symbol, timeframe, timestamp, stop_price, 'STOP_LOSS')







                            elif take_hit:







                                _execute_close(symbol, timeframe, timestamp, take_price, 'TAKE_PROFIT')















                signal = None







                signals_series = None







                if use_builder_signals:







                    signals_series = signals_series_map.get(symbol, {}).get(timeframe)







                elif signals_map is not None:







                    signals_series = signals_map.get(symbol, {}).get(timeframe)















                force_full_context = len(timeframes) > 1







                if not force_full_context and signals_series is not None:



                    if hasattr(signals_series, 'iloc') and i < len(signals_series):



                        signal = signals_series.iloc[i]



                    else:



                        signal = 'HOLD'



                    signal = _normalize_signal(signal, symbol, timeframe, timestamp)



                else:



                    current_data = {}



                    for sym in symbols:



                        current_data[sym] = {}



                        for tf in timeframes:



                            df_full = data_with_indicators.get(sym, {}).get(tf)







                            if df_full is None:







                                continue







                            current_data[sym][tf] = _slice_until(df_full, timestamp)







                    signals = strategy.generate_signals(current_data)



                    signal = signals.get(symbol, {}).get(timeframe, 'HOLD')



                    if hasattr(signal, 'iloc'):



                        signal = signal.iloc[-1]



                    signal = _normalize_signal(signal, symbol, timeframe, timestamp)







                if self.debug_signals and signal_history is not None:
                    signal_history[symbol][timeframe].append((timestamp, signal))
                    self._debug_signal(symbol, timeframe, timestamp, signal, row, signal_history[symbol][timeframe][-5:])















                if signal in ['OPEN_LONG', 'CLOSE_LONG', 'OPEN_SHORT', 'CLOSE_SHORT']:



                    if signal in ['CLOSE_LONG', 'CLOSE_SHORT']:



                        position = self.positions.get(position_key)



                        if not position:



                            continue



                        if signal == 'CLOSE_LONG' and position.get('is_short', False):







                            continue







                        if signal == 'CLOSE_SHORT' and not position.get('is_short', False):







                            continue







                    if execution_delay_bars == 0:







                        if signal == 'OPEN_LONG':







                            _execute_open(symbol, timeframe, timestamp, close_price, is_short=False)







                        elif signal == 'OPEN_SHORT':







                            _execute_open(symbol, timeframe, timestamp, close_price, is_short=True)







                        elif signal == 'CLOSE_LONG':







                            _execute_close(symbol, timeframe, timestamp, close_price, 'CLOSE_LONG (SAME BAR)')







                        elif signal == 'CLOSE_SHORT':







                            _execute_close(symbol, timeframe, timestamp, close_price, 'CLOSE_SHORT (SAME BAR)')







                    else:







                        pending_orders.setdefault(position_key, []).append({







                            'signal': signal,







                            'due_index': i + execution_delay_bars







                        })















                # Update equity and drawdown each bar







                last_prices[symbol] = close_price







                unrealized = self._calculate_unrealized_pnl(last_prices)







                # balance is free cash; add reserved margin to get total equity



                reserved_margin = sum(pos.get('margin_used', 0.0) for pos in self.positions.values())



                positions_value = reserved_margin + unrealized



                self.performance_tracker.update_equity(timestamp, self.balance, positions_value)



                self.equity = self.balance + positions_value







                if self.equity > self.peak_equity:







                    self.peak_equity = self.equity







                if self.peak_equity > 0:







                    current_drawdown = (self.peak_equity - self.equity) / self.peak_equity







                    self.max_drawdown = max(self.max_drawdown, current_drawdown)







            # === CLOSE REMAINING POSITIONS ===







            for position_key, position in list(self.positions.items()):







                if isinstance(position_key, tuple):







                    symbol, timeframe = position_key







                else:







                    symbol = position_key







                    timeframe = position.get('timeframe')







                last_symbol_data = None







                if symbol in data_with_indicators:







                    if timeframe and timeframe in data_with_indicators[symbol]:







                        last_symbol_data = data_with_indicators[symbol][timeframe]







                    elif data_with_indicators[symbol]:







                        last_symbol_data = max(







                            [data_with_indicators[symbol][tf] for tf in data_with_indicators[symbol]],







                            key=lambda x: x.index[-1]







                        )







                if last_symbol_data is None or last_symbol_data.empty:







                    del self.positions[position_key]







                    continue







                current_price = last_symbol_data.iloc[-1]['close']







                entry_price = position['entry_price']







                quantity = position['quantity']







                is_short = position.get('is_short', False)







                entry_timestamp = position['entry_timestamp']







                







                if is_short:







                    exit_price = self.get_execution_price(current_price, 'CLOSE_SHORT')







                    gross_pnl = (entry_price - exit_price) * quantity







                else:







                    exit_price = self.get_execution_price(current_price, 'CLOSE_LONG')







                    gross_pnl = (exit_price - entry_price) * quantity







                







                fee = self.apply_fees(exit_price * quantity)







                net_pnl = gross_pnl - fee







                self.balance += net_pnl + position.get('margin_used', 0.0)







                direction = 'CLOSE_SHORT' if is_short else 'CLOSE_LONG'







                exit_timestamp = last_symbol_data.index[-1]







                







                self.performance_tracker.record_trade({



                    'symbol': symbol, 'timeframe': timeframe, 'direction': direction,



                    'entry_price': entry_price, 'exit_price': exit_price,



                    'size': quantity, 'entry_timestamp': entry_timestamp,



                    'exit_timestamp': exit_timestamp, 'pnl': net_pnl



                })



                print(f"CLOSE {direction}: {symbol} {timeframe} at {exit_price} (PnL: {net_pnl})")







                del self.positions[position_key]







            # === FINAL EQUITY & DRAWDOWN UPDATE ===







            self.equity = self.balance







            if self.equity > self.peak_equity:







                self.peak_equity = self.equity







            final_drawdown = (self.peak_equity - self.equity) / self.peak_equity







            self.max_drawdown = max(self.max_drawdown, final_drawdown)







            







            # === CALCULATE PERFORMANCE METRICS ===



            metrics = self._calculate_performance_metrics()



            strategy_name = strategy.name if hasattr(strategy, 'name') else strategy.__class__.__name__



            chart_path = None



            if self.config.get('export_chart', False):



                try:



                    chart_path = self._export_interactive_chart(



                        data_with_indicators=data_with_indicators,



                        trades=self.performance_tracker.trades,



                        symbols=symbols or strategy.symbols,



                        timeframes=timeframes or strategy.timeframes,



                        strategy_name=strategy_name



                    )



                except Exception:



                    chart_path = None



            







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



                'chart_path': chart_path,



                'start_time': start_dt,



                'end_time': end_dt,



                'duration': duration_str



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







                'total_trades': 0







            }







        finally:



            if '_original_config' in locals():



                self.config = _original_config







    def _export_interactive_chart(self, data_with_indicators, trades, symbols, timeframes, strategy_name):



        """



        Export an interactive HTML chart with OHLC, indicators, and trade markers.



        """



        try:



            import plotly.graph_objects as go



            import plotly.io as pio



        except Exception:



            return None



        if not data_with_indicators:



            return None



        output_dir = self.config.get(



            'chart_output_dir',



            os.path.join(os.path.dirname(__file__), '..', 'optimization_results', 'backtest_charts')



        )



        Path(output_dir).mkdir(parents=True, exist_ok=True)



        safe_name = ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in str(strategy_name))



        chart_path = os.path.join(output_dir, f"{safe_name}_backtest_chart.html")



        fig = go.Figure()



        group_names = []



        group_trace_indices = []



        base_cols = {'open', 'high', 'low', 'close', 'volume', 'indicators_valid'}



        for symbol in symbols:



            for timeframe in timeframes:



                df = data_with_indicators.get(symbol, {}).get(timeframe)



                if df is None or df.empty:



                    continue



                group_name = f"{symbol} {timeframe}"



                start_idx = len(fig.data)



                fig.add_trace(go.Candlestick(



                    x=df.index,



                    open=df['open'],



                    high=df['high'],



                    low=df['low'],



                    close=df['close'],



                    name=f"{group_name} OHLC"



                ))



                indicator_cols = []



                for col in df.columns:



                    if col in base_cols:



                        continue



                    if pd.api.types.is_numeric_dtype(df[col]):



                        indicator_cols.append(col)



                for col in indicator_cols:



                    fig.add_trace(go.Scatter(



                        x=df.index,



                        y=df[col],



                        mode='lines',



                        name=f"{group_name} {col}"



                    ))



                open_long_x, open_long_y, open_long_text = [], [], []



                open_short_x, open_short_y, open_short_text = [], [], []



                close_long_x, close_long_y, close_long_text = [], [], []



                close_short_x, close_short_y, close_short_text = [], [], []



                for t in trades:



                    if getattr(t, 'symbol', None) != symbol:



                        continue



                    trade_tf = getattr(t, 'timeframe', None)



                    if trade_tf is not None and trade_tf != timeframe:



                        continue



                    if trade_tf is None and len(timeframes) > 1:



                        continue



                    direction = getattr(t, 'direction', '')



                    if direction == 'CLOSE_LONG':



                        open_long_x.append(t.entry_timestamp)



                        open_long_y.append(t.entry_price)



                        open_long_text.append('OPEN_LONG')



                        close_long_x.append(t.exit_timestamp)



                        close_long_y.append(t.exit_price)



                        close_long_text.append('CLOSE_LONG')



                    elif direction == 'CLOSE_SHORT':



                        open_short_x.append(t.entry_timestamp)



                        open_short_y.append(t.entry_price)



                        open_short_text.append('OPEN_SHORT')



                        close_short_x.append(t.exit_timestamp)



                        close_short_y.append(t.exit_price)



                        close_short_text.append('CLOSE_SHORT')



                if open_long_x:



                    fig.add_trace(go.Scatter(



                        x=open_long_x,



                        y=open_long_y,



                        mode='markers',



                        name=f"{group_name} OPEN_LONG",



                        marker=dict(symbol='triangle-up', size=10, color='green'),



                        text=open_long_text,



                        hoverinfo='text+x+y'



                    ))



                if open_short_x:



                    fig.add_trace(go.Scatter(



                        x=open_short_x,



                        y=open_short_y,



                        mode='markers',



                        name=f"{group_name} OPEN_SHORT",



                        marker=dict(symbol='triangle-up', size=10, color='red'),



                        text=open_short_text,



                        hoverinfo='text+x+y'



                    ))



                if close_long_x:



                    fig.add_trace(go.Scatter(



                        x=close_long_x,



                        y=close_long_y,



                        mode='markers',



                        name=f"{group_name} CLOSE_LONG",



                        marker=dict(symbol='triangle-down', size=10, color='blue'),



                        text=close_long_text,



                        hoverinfo='text+x+y'



                    ))



                if close_short_x:



                    fig.add_trace(go.Scatter(



                        x=close_short_x,



                        y=close_short_y,



                        mode='markers',



                        name=f"{group_name} CLOSE_SHORT",



                        marker=dict(symbol='triangle-down', size=10, color='orange'),



                        text=close_short_text,



                        hoverinfo='text+x+y'



                    ))



                end_idx = len(fig.data)



                if end_idx > start_idx:



                    group_names.append(group_name)



                    group_trace_indices.append(list(range(start_idx, end_idx)))



        if not group_names:



            return None



        total_traces = len(fig.data)



        buttons = []



        for idx, group in enumerate(group_names):



            visible = [False] * total_traces



            for trace_idx in group_trace_indices[idx]:



                visible[trace_idx] = True



            buttons.append(dict(



                label=group,



                method='update',



                args=[{'visible': visible}, {'title': f"{strategy_name} - {group}"}]



            ))



        # Default visibility to first group



        default_visible = [False] * total_traces



        for trace_idx in group_trace_indices[0]:



            default_visible[trace_idx] = True



        for i, v in enumerate(default_visible):



            fig.data[i].visible = v



        fig.update_layout(



            title=f"{strategy_name} - {group_names[0]}",



            xaxis_rangeslider_visible=False,



            hovermode='x unified',



            updatemenus=[dict(



                active=0,



                buttons=buttons,



                x=1.02,



                y=1,



                xanchor='left',



                yanchor='top'



            )],



            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)



        )



        pio.write_html(fig, file=chart_path, include_plotlyjs='cdn', full_html=True, config={'responsive': True})



        return chart_path



    







    def run_walk_forward(self, symbols, timeframes, start_date, end_date,







                         window_days: int = 90, step_days: Optional[int] = None,







                         config: Optional[Dict[str, Any]] = None,







                         strategy: Optional[StrategyBase] = None,







                         initial_balance: Optional[float] = None,







                         progress_callback=None):







        """







        Run backtest in rolling windows across the date range.







        """







        if step_days is None:







            step_days = window_days







        if step_days <= 0:







            step_days = 1







        start_dt = pd.to_datetime(start_date)







        end_dt = pd.to_datetime(end_date)







        results = []







        window_start = start_dt







        window_index = 1







        while window_start + timedelta(days=window_days) <= end_dt:







            window_end = window_start + timedelta(days=window_days)







            window_result = self.run_backtest(







                symbols=symbols,







                timeframes=timeframes,







                start_date=window_start.strftime('%Y-%m-%d'),







                end_date=window_end.strftime('%Y-%m-%d'),







                config=config,







                strategy=strategy,







                initial_balance=initial_balance,







                progress_callback=progress_callback







            )







            results.append({







                'window': window_index,







                'start_date': window_start.strftime('%Y-%m-%d'),







                'end_date': window_end.strftime('%Y-%m-%d'),







                'results': window_result







            })







            window_start = window_start + timedelta(days=step_days)







            window_index += 1







        return results







    







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







        position = None







        if isinstance(symbol, tuple):







            position = self.positions.get(symbol)







        else:







            position = self.positions.get(symbol)







            if position is None:







                for key, pos in self.positions.items():







                    if isinstance(key, tuple) and key[0] == symbol:







                        position = pos







                        break







        has_position = position is not None







        if signal == 'OPEN_LONG' and has_position and not position.get('is_short', False):







            return False







        if signal == 'OPEN_SHORT' and has_position and position.get('is_short', False):







            return False







        if signal == 'CLOSE_LONG' and (not has_position or position.get('is_short', False)):







            return False







        if signal == 'CLOSE_SHORT' and (not has_position or not position.get('is_short', False)):







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







                logger.warning(f"⚠️ Invalid data structure for {symbol}")







                return 50000.0  # Return a reasonable default price







            







            # Get the first timeframe data for this symbol







            if not current_data[symbol]:







                logger.warning(f"⚠️ No timeframe data for {symbol}")







                return 50000.0







            







            # Get the first timeframe DataFrame







            timeframe_data = list(current_data[symbol].values())[0]







            







            if timeframe_data.empty:







                logger.warning(f"⚠️ Empty DataFrame for {symbol}")







                return 50000.0







            







            # Get the last close price







            current_price = timeframe_data['close'].iloc[-1]







            







            # Ensure it's a valid price







            if pd.isna(current_price) or current_price <= 0:







                logger.warning(f"⚠️ Invalid price {current_price} for {symbol}")







                return 50000.0







            







            return float(current_price)







            







        except Exception as e:







            logger.error(f"❌ Error getting current price for {symbol}: {e}")







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







        print("📊 BACKTEST RESULTS")







        print("="*50)







        print(f"💰 Total Return: {performance_metrics['total_return']:.2f}%")







        print(f"🎯 Win Rate: {performance_metrics['win_rate']:.2f}%")







        print(f"📈 Sharpe Ratio: {performance_metrics['sharpe_ratio']:.2f}")







        print(f"📉 Max Drawdown: {performance_metrics['max_drawdown']:.2f}%")







        print(f"🔄 Total Trades: {len(self.trades)}")







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







                'total_trades': 0







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







            'total_trades': total_trades







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







        print("📊 BACKTEST RESULTS")







        print("=" * 50)







        print(f"💰 Total Return: {metrics['total_return_pct']:.2f}%")







        print(f"🎯 Win Rate: {metrics['win_rate_pct']:.2f}%")







        print(f"📈 Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")







        print(f"📉 Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")







        print(f"🔄 Total Trades: {metrics['total_trades']}")







        print("=" * 50)







    def get_execution_price(self, price: float, action: str) -> float:







        """







        Simulate realistic execution price with spread + slippage.







        action: 'OPEN_LONG', 'CLOSE_LONG', 'OPEN_SHORT', or 'CLOSE_SHORT'







        """







        # Conservative defaults for crypto perp trading







        spread_pct = self.config.get('spread_pct', 0.0004)      # 0.04%







        slippage_pct = self.config.get('slippage_pct', 0.0003)  # 0.03%







        is_buy = action in ('OPEN_LONG', 'CLOSE_SHORT')







        if is_buy:







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







                







                # Calculate EMAs and RSI







                df[f'ema_fast_{timeframe}'] = ema(df['close'], period=fast_ma_period)







                df[f'ema_slow_{timeframe}'] = ema(df['close'], period=slow_ma_period)







                df[f'rsi_{timeframe}'] = rsi(df['close'], period=rsi_period)







                







                # Mark valid indicator rows







                min_valid_idx = max(







                    df[f'ema_fast_{timeframe}'].first_valid_index(),







                    df[f'ema_slow_{timeframe}'].first_valid_index(),







                    df[f'rsi_{timeframe}'].first_valid_index()







                )







                df['indicators_valid'] = df.index >= min_valid_idx







                







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







