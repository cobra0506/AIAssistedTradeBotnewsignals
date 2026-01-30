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
            
            # Track signal history for debugging
            signal_history = {symbol: {timeframe: [] for timeframe in timeframes} for symbol in symbols}
            
            # Main loop - optimized version
            total_rows = sum(len(data[s][t]) for s in data for t in data[s])
            processed_rows = 0
            last_progress_update = 0
            
            # Process each symbol/timeframe
            for symbol in data_with_indicators:
                for timeframe in data_with_indicators[symbol]:
                    df = data_with_indicators[symbol][timeframe]

                    use_builder_signals = hasattr(strategy, 'builder')
                    signals_series = None
                    if use_builder_signals:
                        try:
                            signals_series = strategy.builder._execute_signal_rules(df)
                        except Exception:
                            signals_series = None

                    
                    # Find the first row where indicators are valid
                    first_valid_idx = df[df['indicators_valid']].index[0] if df['indicators_valid'].any() else len(df)
                    
                    # Process every row to catch all signals
                    for i in range(df.index.get_loc(first_valid_idx), len(df)):
                        timestamp = df.index[i]
                        processed_rows += 1
                        progress_percent = (processed_rows / total_rows) * 100
                        
                        # UPDATE: Call the progress callback if provided
                        if progress_callback and (progress_percent - last_progress_update >= 1 or progress_percent == 100):
                            progress_callback(progress_percent)
                            last_progress_update = progress_percent
                        
                        if signals_series is not None:
                            signal = signals_series.iloc[i]
                            if isinstance(signal, (int, float)):
                                if signal >= 2:
                                    signal = 'OPEN_LONG'
                                elif signal == 1:
                                    signal = 'CLOSE_SHORT'
                                elif signal == -1:
                                    signal = 'CLOSE_LONG'
                                elif signal <= -2:
                                    signal = 'OPEN_SHORT'
                                else:
                                    signal = 'HOLD'
                        else:
                            # Get data up to this timestamp
                            current_data = {symbol: {timeframe: df.iloc[:i+1].copy()}}
                            # Generate signal
                            signals = strategy.generate_signals(current_data)
                            signal = signals[symbol][timeframe]

                        
                        # Track signal history
                        signal_history[symbol][timeframe].append((timestamp, signal))
                        
                        # Debug all trading signals
                        self._debug_signal(symbol, timeframe, timestamp, signal, df.iloc[:i+1], signal_history[symbol][timeframe][-5:]) 

                        current_price = df.iloc[i]['close']
                        
                        # === TRADE EXECUTION ===
                        if signal in ['OPEN_LONG', 'CLOSE_LONG', 'OPEN_SHORT', 'CLOSE_SHORT']:
                            # Add validation for CLOSE signals
                            if signal in ['CLOSE_LONG', 'CLOSE_SHORT']:
                                if symbol not in self.positions:
                                    #print(f"⚠️ WARNING: {signal} signal for {symbol} but no open position. Skipping.")
                                    # Check if we ever had an OPEN signal for this symbol
                                    recent_signals = [s[1] for s in signal_history[symbol][timeframe][-10:]]
                                    has_open_signal = any(s in ['OPEN_LONG', 'OPEN_SHORT'] for s in recent_signals)
                                    print(f"  Recent signals: {recent_signals}")
                                    print(f"  Had OPEN signal recently: {has_open_signal}")
                                    continue
                                if signal == 'CLOSE_LONG' and self.positions[symbol].get('is_short', False):
                                    print(f"⚠️ WARNING: CLOSE_LONG signal for {symbol} but position is SHORT. Skipping.")
                                    continue
                                if signal == 'CLOSE_SHORT' and not self.positions[symbol].get('is_short', False):
                                    print(f"⚠️ WARNING: CLOSE_SHORT signal for {symbol} but position is LONG. Skipping.")
                                    continue
                            
                            if signal == 'OPEN_LONG':
                                max_positions = self.config.get('max_positions', 3)
                                if len(self.positions) >= max_positions:
                                    continue
                                if symbol not in self.positions:
                                    entry_price = self.get_execution_price(current_price, 'BUY')
                                    stop_price = entry_price * 0.99
                                    quantity = self.get_order_size(symbol, entry_price, stop_price)
                                    fee = self.apply_fees(entry_price * quantity)
                                    self.balance -= fee
                                    self.positions[symbol] = {'entry_price': entry_price, 'quantity': quantity, 'entry_timestamp': timestamp, 'is_short': False}
                                    print(f"🟢 OPEN_LONG: {symbol} at {entry_price} (qty: {quantity})")
                            elif signal == 'CLOSE_LONG':
                                # Only process if we have a long position
                                if symbol in self.positions and not self.positions[symbol].get('is_short', False):
                                    position = self.positions.pop(symbol)
                                    entry_price = position['entry_price']
                                    quantity = position['quantity']
                                    exit_price = self.get_execution_price(current_price, 'SELL')
                                    gross_pnl = (exit_price - entry_price) * quantity
                                    fee = self.apply_fees(exit_price * quantity)
                                    net_pnl = gross_pnl - fee
                                    self.balance += net_pnl
                                    self.performance_tracker.record_trade({
                                        'symbol': symbol, 'direction': 'CLOSE_LONG',
                                        'entry_price': entry_price, 'exit_price': exit_price,
                                        'size': quantity, 'entry_timestamp': position['entry_timestamp'],
                                        'exit_timestamp': timestamp, 'pnl': net_pnl
                                    })
                                    print(f"🔴 CLOSE_LONG: {symbol} at {exit_price} (PnL: {net_pnl})")
                            elif signal == 'OPEN_SHORT':
                                max_positions = self.config.get('max_positions', 3)
                                if len(self.positions) >= max_positions:
                                    continue
                                if symbol not in self.positions:
                                    entry_price = self.get_execution_price(current_price, 'SELL')
                                    stop_price = entry_price * 1.01
                                    quantity = self.get_order_size(symbol, entry_price, stop_price)
                                    fee = self.apply_fees(entry_price * quantity)
                                    self.balance -= fee
                                    self.positions[symbol] = {'entry_price': entry_price, 'quantity': quantity, 'entry_timestamp': timestamp, 'is_short': True}
                                    print(f"🟢 OPEN_SHORT: {symbol} at {entry_price} (qty: {quantity})")
                            elif signal == 'CLOSE_SHORT':
                                # Only process if we have a short position
                                if symbol in self.positions and self.positions[symbol].get('is_short', False):
                                    position = self.positions.pop(symbol)
                                    entry_price = position['entry_price']
                                    quantity = position['quantity']
                                    exit_price = self.get_execution_price(current_price, 'BUY')
                                    gross_pnl = (entry_price - exit_price) * quantity
                                    fee = self.apply_fees(exit_price * quantity)
                                    net_pnl = gross_pnl - fee
                                    self.balance += net_pnl
                                    self.performance_tracker.record_trade({
                                        'symbol': symbol, 'direction': 'CLOSE_SHORT',
                                        'entry_price': entry_price, 'exit_price': exit_price,
                                        'size': quantity, 'entry_timestamp': position['entry_timestamp'],
                                        'exit_timestamp': timestamp, 'pnl': net_pnl
                                    })
                                    print(f"🔴 CLOSE_SHORT: {symbol} at {exit_price} (PnL: {net_pnl})")


            # === CLOSE REMAINING POSITIONS ===
            for symbol, position in list(self.positions.items()):
                last_symbol_data = max([data_with_indicators[symbol][tf] for tf in data_with_indicators[symbol]], key=lambda x: x.index[-1])
                current_price = last_symbol_data.iloc[-1]['close']
                entry_price = position['entry_price']
                quantity = position['quantity']
                is_short = position.get('is_short', False)
                entry_timestamp = position['entry_timestamp']
                
                if is_short:
                    exit_price = self.get_execution_price(current_price, 'BUY')
                    gross_pnl = (entry_price - exit_price) * quantity
                else:
                    exit_price = self.get_execution_price(current_price, 'SELL')
                    gross_pnl = (exit_price - entry_price) * quantity
                
                fee = self.apply_fees(exit_price * quantity)
                net_pnl = gross_pnl - fee
                self.balance += net_pnl
                direction = 'CLOSE_SHORT' if is_short else 'CLOSE_LONG'
                
                self.performance_tracker.record_trade({
                    'symbol': symbol, 'direction': direction,
                    'entry_price': entry_price, 'exit_price': current_price,
                    'size': quantity, 'entry_timestamp': entry_timestamp,
                    'exit_timestamp': last_symbol_data.index[-1], 'pnl': net_pnl
                })
                print(f"🔴 CLOSE {direction}: {symbol} at {exit_price} (PnL: {net_pnl})")
                del self.positions[symbol]
            
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
            
            # === RETURN METRICS ===
            return {
                'win_rate': metrics['win_rate_pct'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown_pct'],
                'total_return': metrics['total_return_pct'],
                'total_trades': metrics['total_trades'],
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

    def get_execution_price(self, price: float, side: str) -> float:
        """
        Simulate realistic execution price with spread + slippage.
        side: 'BUY' or 'SELL'
        """
        # Conservative defaults for crypto perp trading
        spread_pct = self.config.get('spread_pct', 0.0004)      # 0.04%
        slippage_pct = self.config.get('slippage_pct', 0.0003)  # 0.03%

        if side == 'BUY':
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

    def _debug_signal(self, symbol, timeframe, timestamp, signal, df, recent_signals=None):
        """Debug method to check what signals are being generated - optimized version"""
        # Only debug for the first symbol and only trading signals
        if symbol != 'BNBUSDT' or signal == 'HOLD':
            return
        
        # Debug all trading signals with recent history
        if f'ema_fast_{timeframe}' in df.columns:
            fast_ema = df[f'ema_fast_{timeframe}'].iloc[-1]
            slow_ema = df[f'ema_slow_{timeframe}'].iloc[-1]
            rsi = df[f'rsi_{timeframe}'].iloc[-1]
            
            print(f"\n=== TRADING SIGNAL: {symbol} {timeframe} at {timestamp} ===")
            print(f"Signal: {signal}")
            print(f"Fast EMA: {fast_ema:.2f}, Slow EMA: {slow_ema:.2f}, RSI: {rsi:.2f}")
            print(f"Trend: {'UP' if fast_ema > slow_ema else 'DOWN'}")
            
            if recent_signals:
                print(f"Recent signals: {recent_signals}")
