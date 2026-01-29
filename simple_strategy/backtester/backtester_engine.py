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
                                    print(f"⚠️ WARNING: {signal} signal for {symbol} but no open position. Skipping.")
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

'''"""
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
            
            # Main loop
            total_rows = sum(len(data[s][t]) for s in data for t in data[s])
            processed_rows = 0
            last_progress_update = 0
            
            if not hasattr(self, '_signal_cache'):
                self._signal_cache = {}
            
            for symbol in data:
                for timeframe in data[symbol]:
                    df = data[symbol][timeframe].copy()
                    
                    for timestamp, row in df.iterrows():
                        processed_rows += 1
                        progress_percent = (processed_rows / total_rows) * 100
                        
                        # UPDATE: Call the progress callback if provided (update every 1% for smoothness)
                        if progress_callback and (progress_percent - last_progress_update >= 1 or progress_percent == 100):
                            progress_callback(progress_percent)
                            last_progress_update = progress_percent
                        
                        current_data = {symbol: {timeframe: df.loc[:timestamp]}}
                        signal_cache_key = f"{symbol}_{timeframe}_{timestamp}"
                        
                        if signal_cache_key in self._signal_cache:
                            signals = self._signal_cache[signal_cache_key]
                        else:
                            signals = strategy.generate_signals(current_data)
                            self._signal_cache[signal_cache_key] = signals
                        
                        signal = signals[symbol][timeframe]
                        current_price = row['close']
                        
                        # === TRADE EXECUTION ===
                        if signal in ['OPEN_LONG', 'CLOSE_LONG', 'OPEN_SHORT', 'CLOSE_SHORT']:
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
                            elif signal == 'CLOSE_LONG':
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
                            elif signal == 'OPEN_SHORT':
                                pass
                            elif signal == 'CLOSE_SHORT':
                                pass
            
            # === CLOSE REMAINING POSITIONS ===
            for symbol, position in list(self.positions.items()):
                last_symbol_data = max([data[symbol][tf] for tf in data[symbol]], key=lambda x: x.index[-1])
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
    
    def _generate_signals_batch(self, data: Dict[str, Dict[str, pd.DataFrame]], 
                           symbol: str, timeframe: str, 
                           start_idx: int = 0, batch_size: int = 100) -> Dict:
        """
        Generate signals for a batch of rows at once
        
        Args:
            data: Full data dictionary
            symbol: Current symbol
            timeframe: Current timeframe
            start_idx: Starting index position
            batch_size: Number of rows to process in this batch
            
        Returns:
            Dictionary mapping timestamps to signals
        """
        df = data[symbol][timeframe]
        end_idx = min(start_idx + batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        
        signals = {}
        
        for timestamp, row in batch_df.iterrows():
            # Get data up to this timestamp
            historical_data = df.iloc[:df.index.get_loc(timestamp)+1]
            current_data = {symbol: {timeframe: historical_data}}
            
            # Generate signal for this timestamp
            signal_result = self.strategy.generate_signals(current_data)
            signals[timestamp] = signal_result[symbol][timeframe]
        
        return signals'''


'''        
BacktesterEngine Architecture Summary 

Core Purpose: An event-driven backtesting engine that iterates through historical OHLCV data row-by-row, generates signals via a strategy object, and executes trades while tracking performance and positions. 

Key Dependencies: 

     DataFeeder: Supplies historical data (dict[symbol][timeframe] -> DataFrame).
     StrategyBase: User-defined strategy implementing generate_signals().
     RiskManager: Handles risk validation (optional).
     PerformanceTracker: Records trades and calculates equity curve (external class).
     

State Variables: 

     self.positions: Dictionary tracking open positions ({symbol: {'entry_price', 'quantity', 'is_short', ...}}).
     self.balance: Current account balance.
     self.performance_tracker: Instance of PerformanceTracker.
     

Critical Architecture Note (Dual Execution Paths):
The codebase contains two separate logic flows for executing trades. The primary entry point run_backtest contains inline logic for handling trades (updating self.positions and calculating PnL manually inside the loop). Separately, there are helper methods (_execute_trade, _open_position, _close_position) which appear to be intended for a cleaner execution flow but are not currently called by the main run_backtest loop. Investigation should focus on which path is actually active. 
Function Reference 

Class: BacktesterEngine 
Initialization & Config 

     __init__(self, data_feeder: DataFeeder, strategy: StrategyBase, risk_manager: Optional[RiskManager] = None, config: Dict[str, Any] = None)
         Desc: Initializes the engine, sets processing mode (sequential/parallel), and initializes PerformanceTracker.
         Sets: self.positions, self.balance (delayed to run), self.config.
         
     

Main Execution Loop 

     run_backtest(self, symbols, timeframes, start_date, end_date, config=None, strategy=None, data=None, initial_balance=None) -> Dict
         Desc: Primary Entry Point. Orchestrates the entire backtest.
         Flow:
            Loads optimized parameters if available. 
            Fetches data via data_feeder (if not provided). 
            Iterates Symbol -> Timeframe -> Row (Timestamp). 
            Constructs current_data slice (history up to current timestamp). 
            Calls strategy.generate_signals(current_data). 
            Inline Execution: Checks for OPEN_LONG, OPEN_SHORT, CLOSE_LONG, CLOSE_SHORT. Manually updates self.positions, calculates PnL, and calls self.performance_tracker.record_trade. 
            Closes any remaining open positions at the end of data. 
            Calculates metrics and returns summary dict. 
         
     

Position Sizing 

     get_order_size(self, symbol, current_price) -> float
         Desc: Calculates trade quantity based on self.balance and risk_per_trade (default 2%) from config.
         Formula: (Balance * Risk%) / Current_Price.
         
     

Data Processing Helpers (Secondary/Alternative Flow) 

     _process_data_chronologically(self, data, symbols, timeframes) -> Dict
         Desc: An alternative processing loop that processes all timestamps globally. It calls _process_signals and _execute_trade. (Note: This is distinct from the inline logic in run_backtest).
         
     _get_data_for_timestamp(self, data, symbols, timeframes, timestamp) -> Dict
         Desc: Slices the full dataframes to return only data <= timestamp. Crucial for indicator calculation in strategies.
         
     _validate_data(self, data, symbols, timeframes) -> bool
         Desc: Checks if DataFrames exist, are non-empty, and contain required columns (open, high, low, close, volume).
         
     

Signal & Trade Execution (Helper Methods) 

     _process_signals(self, signals, current_data, timestamp) -> List
         Desc: Iterates through generated signals and calls _execute_trade if valid.
         
     _can_execute_trade(self, symbol, signal, timestamp, current_data) -> bool
         Desc: Logic guard preventing opening a position if one exists, or closing if none exists.
         
     _execute_trade(self, symbol, signal, timestamp, current_data)
         Desc: Dispatcher that routes to _open_position or _close_position based on signal type.
         
     _open_position(self, symbol, quantity, price, timestamp, is_short=False)
         Desc: Updates self.positions dict. Handles reversing positions (closing opposite then opening new).
         
     _close_position(self, symbol, price, timestamp, is_short=False)
         Desc: Calculates PnL ((Exit - Entry) * Qty for Long, inverse for Short), records trade via performance_tracker, and removes entry from self.positions.
         
     _get_current_price(self, symbol, current_data) -> float
         Desc: Safely extracts the 'close' price from the current data structure.
         
     

Results & Metrics 

     _calculate_performance_metrics(self) -> Dict
         Desc: Computes Win Rate, Total Return, Sharpe Ratio, and Max Drawdown from self.performance_tracker.trades.
         
     _display_results(self, metrics)
         Desc: Prints formatted metrics to console.
         
     calculate_performance_metrics(self, trades, initial_balance, final_balance) -> Dict
         Desc: Public static-like method for calculating metrics (distinct from the private method used by the engine).
         
     _update_results(self, results, signals, trade_results, timestamp)
         Desc: Updates the equity curve and history lists (used in secondary flow).
         
     

Utility 

     stop_backtest(self)
         Desc: Sets self.is_running = False to halt loops.
         
     get_status(self) -> Dict
         Desc: Returns current state, including is_running and processing_stats.
         
     '''

'''def _process_data_chronologically(self, data: Dict[str, Dict[str, pd.DataFrame]],
                                symbols: List[str], timeframes: List[str]) -> Dict[str, Any]:
        """Process data chronologically and execute strategy signals"""
        try:
            # Get all unique timestamps across all symbols and timeframes
            all_timestamps = set()
            for symbol in symbols:
                for timeframe in timeframes:
                    if symbol in data and timeframe in data[symbol]:
                        all_timestamps.update(data[symbol][timeframe].index)
            
            # Sort timestamps
            sorted_timestamps = sorted(all_timestamps)
            
            # Initialize results tracking
            results = {
                'equity_curve': [],
                'trades': [],
                'signals': [],
                'timestamps': [],
                'portfolio_values': []
            }
            
            # Track first few signals for debugging
            signal_count = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            
            # Process each timestamp
            for i, timestamp in enumerate(sorted_timestamps):
                if not self.is_running:
                    break
                
                # Get data for current timestamp across all symbols and timeframes
                current_data = self._get_data_for_timestamp(data, symbols, timeframes, timestamp)
                
                # Generate signals using strategy
                signals = self.strategy.generate_signals(current_data)
                
                # Count signals for debugging
                for symbol, timeframe_signals in signals.items():
                    for timeframe, signal in timeframe_signals.items():
                        signal_count[signal] += 1
                
                # Process signals and execute trades
                trade_results = self._process_signals(signals, current_data, timestamp)
                
                # Update results
                self._update_results(results, signals, trade_results, timestamp)
                
                # Update processing stats
                self.processing_stats['total_rows_processed'] += len(symbols) * len(timeframes)
                self.processing_stats['total_signals_generated'] += len([s for s in signals.values() if s != 'HOLD'])
                
                # Log progress every 1000 timestamps
                if i % 1000 == 0:
                    progress = (i / len(sorted_timestamps)) * 100
                    print(f"🔧 Processing progress: {progress:.1f}%")
            
            # Print signal summary for debugging
            print(f"🔧 Signal summary: BUY={signal_count['BUY']}, SELL={signal_count['SELL']}, HOLD={signal_count['HOLD']}")
            print(f"🔧 Processing complete. Generated {len(results['trades'])} trades")
            
            return results
        except Exception as e:
            print(f"🔧 Error in _process_data_chronologically: {e}")
            import traceback
            print(f"🔧 Full traceback: {traceback.format_exc()}")
            return {'error': str(e)}'''

'''def _process_signals(self, signals: Dict[str, Dict[str, str]], current_data: Dict[str, Dict[str, pd.DataFrame]], timestamp: datetime) -> List[Dict[str, Any]]:
        """Process signals and execute trades"""
        trade_results = []
        
        try:
            for symbol, timeframe_signals in signals.items():
                for timeframe, signal in timeframe_signals.items():
                    if signal in ['OPEN_LONG', 'OPEN_SHORT', 'CLOSE_LONG', 'CLOSE_SHORT']:
                        if self._can_execute_trade(symbol, signal, timestamp, current_data):
                            trade_result = self._execute_trade(symbol, signal, timestamp, current_data)
                            trade_results.append(trade_result)
                        else:
                            logger.info(f"⚠️ Cannot execute {signal} trade for {symbol} at {timestamp}")

            return trade_results
            
        except Exception as e:
            logger.error(f"❌ Error processing signals: {e}")
            return []'''
    
'''def _execute_trade(self, symbol, signal, timestamp, current_data):
            """
            Execute a trade based on signal:
            Handles OPEN/CLOSE for LONG and SHORT positions.
            """
            price = current_data['close']  # Or whatever field you use for current price
            quantity = self.get_order_size(symbol, price)  # Existing function or adjust as needed

            if signal == 'OPEN_LONG':
                self._open_position(symbol, quantity, price, timestamp, is_short=False)

            elif signal == 'CLOSE_LONG':
                self._close_position(symbol, price, timestamp, is_short=False)

            elif signal == 'OPEN_SHORT':
                self._open_position(symbol, quantity, price, timestamp, is_short=True)

            elif signal == 'CLOSE_SHORT':
                self._close_position(symbol, price, timestamp, is_short=True)
            
            else:
                logger.warning(f"Unknown signal {signal} for {symbol} at {timestamp}")
            
            return {
                'symbol': symbol,
                'signal': signal,
                'price': price,
                'timestamp': timestamp
            }'''

'''def _open_position(self, symbol, quantity, price, timestamp, is_short=False):
            """
            Open or add to a long/short position.
            """
            if symbol in self.positions:
                pos = self.positions[symbol]
                if pos['is_short'] == is_short:
                    # Add to existing position
                    old_qty = pos['quantity']
                    old_price = pos['entry_price']
                    new_qty = old_qty + quantity
                    new_entry = ((old_price * old_qty) + (price * quantity)) / new_qty
                    pos['entry_price'] = new_entry
                    pos['quantity'] = new_qty
                    logger.info(f"Updated {'SHORT' if is_short else 'LONG'} position for {symbol}: price={new_entry}, qty={new_qty}")
                else:
                    # Opposite position exists: close it first
                    self._close_position(symbol, price, timestamp, is_short=not is_short)
                    self.positions[symbol] = {'entry_price': price, 'quantity': quantity, 'entry_timestamp': timestamp, 'is_short': is_short}
                    logger.info(f"Opened {'SHORT' if is_short else 'LONG'} position for {symbol}: price={price}, qty={quantity}")
            else:
                # New position
                self.positions[symbol] = {'entry_price': price, 'quantity': quantity, 'entry_timestamp': timestamp, 'is_short': is_short}
                logger.info(f"Opened {'SHORT' if is_short else 'LONG'} position for {symbol}: price={price}, qty={quantity}")'''

'''def _close_position(self, symbol, price, timestamp, is_short=False):
            """
            Close a long/short position and calculate PnL.
            """
            if symbol not in self.positions:
                logger.warning(f"No position to close for {symbol}")
                return
            
            pos = self.positions[symbol]
            if pos['is_short'] != is_short:
                logger.warning(f"Position type mismatch for {symbol}, cannot close")
                return

            entry_price = pos['entry_price']
            qty = pos['quantity']

            # Calculate PnL
            pnl = (entry_price - price) * qty if is_short else (price - entry_price) * qty
            direction = 'CLOSE_SHORT' if is_short else 'CLOSE_LONG'

            # Record trade
            self.performance_tracker.record_trade({
                'symbol': symbol,
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': price,
                'size': qty,
                'entry_timestamp': pos['entry_timestamp'],
                'exit_timestamp': timestamp,
                'pnl': pnl
            })

            logger.info(f"Closed {'SHORT' if is_short else 'LONG'} position for {symbol}: entry={entry_price}, exit={price}, qty={qty}, pnl={pnl}")
            del self.positions[symbol]'''

        
'''def _update_results(self, results: Dict[str, Any], signals: Dict[str, Dict[str, str]], trade_results: List[Dict[str, Any]], timestamp: datetime):
            """Update results with new signals and trades"""
            try:
                # Update signals
                results['signals'].append({
                    'timestamp': timestamp,
                    'signals': signals
                })
                
                # Update trades
                results['trades'].extend(trade_results)
                
                # Update timestamps
                results['timestamps'].append(timestamp)
                
                # Calculate current portfolio value
                current_value = self.strategy.balance
                
                # Add value of open positions
                for symbol, position in self.strategy.positions.items():
                    if position.get('size', 0) > 0:
                        # Get current price (this is a simplified approach)
                        current_price = position.get('current_price', position.get('entry_price', 0))
                        position_value = position['size'] * current_price
                        current_value += position_value
                
                # Update portfolio values
                results['portfolio_values'].append({
                    'timestamp': timestamp,
                    'value': current_value
                })
                
                # Update equity curve
                results['equity_curve'].append({
                    'timestamp': timestamp,
                    'value': current_value
                })
                
            except Exception as e:
                logger.error(f"❌ Error updating results: {e}")'''
        
'''def _calculate_final_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
            """
            Calculate final backtest results from intermediate results
            Args:
                results: Intermediate results from _process_data_chronologically
            Returns:
                Final results dictionary
            """
            try:
                logger.debug("🔧 DEBUG: _calculate_final_results called")
                
                # Check if results contains an error
                if 'error' in results:
                    logger.error(f"❌ Error in backtest processing: {results['error']}")
                    return {'error': results['error']}
                
                # Calculate performance metrics
                final_results = {
                    'equity_curve': results.get('equity_curve', []),
                    'trades': results.get('trades', []),
                    'signals': results.get('signals', []),
                    'timestamps': results.get('timestamps', []),
                    'portfolio_values': results.get('portfolio_values', []),
                    'performance_metrics': {}
                }
                
                # Calculate basic metrics
                if final_results['trades']:
                    total_trades = len(final_results['trades'])
                    winning_trades = len([t for t in final_results['trades'] if t.get('pnl', 0) > 0])
                    losing_trades = total_trades - winning_trades
                    
                    final_results['performance_metrics']['total_trades'] = total_trades
                    final_results['performance_metrics']['winning_trades'] = winning_trades
                    final_results['performance_metrics']['losing_trades'] = losing_trades
                    
                    if total_trades > 0:
                        win_rate = winning_trades / total_trades
                        final_results['performance_metrics']['win_rate'] = win_rate
                
                # Calculate equity-based metrics
                if final_results['equity_curve']:
                    # Handle the case where equity curve contains dictionaries
                    if isinstance(final_results['equity_curve'][0], dict):
                        # Extract values from dictionaries
                        initial_equity = final_results['equity_curve'][0].get('value', 10000) if final_results['equity_curve'] else 10000
                        final_equity = final_results['equity_curve'][-1].get('value', 10000) if final_results['equity_curve'] else initial_equity
                    else:
                        # Direct numeric values
                        initial_equity = final_results['equity_curve'][0] if final_results['equity_curve'] else 10000
                        final_equity = final_results['equity_curve'][-1] if final_results['equity_curve'] else initial_equity
                    
                    total_return = (final_equity - initial_equity) / initial_equity if initial_equity != 0 else 0
                    final_results['performance_metrics']['total_return'] = total_return
                    final_results['performance_metrics']['initial_equity'] = initial_equity
                    final_results['performance_metrics']['final_equity'] = final_equity
                
                # Add processing stats
                final_results['processing_stats'] = self.processing_stats
                
                # Add execution time
                if self.start_time and self.end_time:
                    execution_time = self.end_time - self.start_time
                    final_results['execution_time'] = execution_time
                
                logger.debug("✅ Final results calculated successfully")
                return final_results
            
            except Exception as e:
                logger.error(f"❌ Error calculating final results: {e}")
                import traceback
                logger.error(f"❌ Full traceback: {traceback.format_exc()}")
                return {'error': str(e)}'''
'''
        ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        1. Core Architecture & How It Works

Your paper trader:

Uses the exact same strategy file as your backtester and future live trading. ✅

Executes real orders on Bybit demo account (not simulated).

All trade execution details are handled by Bybit:

Partial fills ✅

Slippage ✅

Fees ✅

Funding rates ✅

Your code only tracks simulated balance internally, for reporting purposes. This is purely for limiting positions, GUI display, and logging — it does not affect order execution.

Logs trades and updates balances after Bybit confirms execution, using real P&L and margins returned.

Implication: This is as close to live trading as possible without using real funds. It is not a simulation, except for your internal balance adjustments.

2. Can You Trust the Results?

Yes — with a few important clarifications:

Strategy correctness:

If the backtester shows the correct logic, and the paper trader executes trades correctly on demo, the strategy file itself works correctly. ✅

Trade execution vs real account:

Because Bybit handles fills, slippage, and fees:

The price you enter/exit at in paper trading will match real trading very closely, assuming similar market conditions.

So yes, if your paper trader earns $100 on a demo trade, a real account should earn around the same $100, minus extremely rare liquidity edge cases. ✅

Limitations to note:

Liquidity effects on very large trades: Demo may not perfectly reflect extreme market depth.

Latency & timing: Network latency in live trading could cause tiny differences.

Strategy profit ≠ guaranteed money: Correct execution doesn’t ensure profitable trades — the strategy could still generate losses.

Verdict: For validating strategy logic and execution, you can trust the results. It represents real trading closely enough that your P&L is realistic.

3. Code Review – Observations
Redundant / Funny / Suboptimal Code

Multiple performance calculation paths

update_performance_display calculates account value and P&L from real balances + simulated balance.

get_performance_summary also calculates P&L and win rate.

Effect: Minor inefficiency and slightly confusing reporting, but not harmful.

Recommendation: Keep one authoritative source (e.g., get_performance_summary) and call it from display function.

Duplicate signal handling

Signals are handled both in start_trading loop and process_symbol.

process_symbol is never called inside start_trading in your code (I see only inline loop).

Effect: Small duplication; could remove process_symbol or use it consistently.

Balance updates

You update simulated balance in multiple places: inside execute_close_short, execute_close_long, and update_balance_after_trade.

Effect: Redundant logging, but balances are consistent.

Sleep calls (time.sleep(0.1))

Minor, used to avoid API rate limits. Fine.

Fallback methods

Functions like get_current_price_from_api are used in logging and CSV, which might be slower for large loops.

CSV logging

Multiple ways of logging trades (headers, type checking). Works fine; could be simplified, but harmless.

Summary of Redundant Code

Not harmful, mostly inefficiency / duplicate logging / multiple ways to calculate performance.

If you want a cleaner codebase:

Consolidate performance calculations.

Use process_symbol uniformly or remove it.

Reduce duplicate balance updates.

4. Key Strengths

Uses real Bybit demo execution, so you cannot simulate yourself out of fees/slippage.

Uses exact same strategy file as real trading, so you validate the logic fully.

Logs all trades with timestamps, P&L, and position duration.

Calculates win rate, total trades, open positions correctly.

Handles edge cases like:

Signals without open positions

Max positions

Stop trading thresholds

Invalid or empty data

✅ This is extremely robust for testing strategy logic and live execution behavior.

5. Can You Treat P&L as Real?

Yes, within limits:

If demo trade shows $100 profit, real account should get roughly the same $100, since execution is by Bybit.

Minor differences to expect:

Slight timing / latency / extreme liquidity effects

Demo vs live depth edge cases

Important: This doesn’t guarantee profitability; strategy could still lose. You are validating execution, not guaranteeing gains.

6. Recommendations / Possible Improvements

Clean up redundancy:

Consolidate performance calculation into a single function, call it everywhere.

Simplify signal handling:

Decide whether to use inline signal processing in start_trading or process_symbol, not both.

Balance logging:

Remove duplicate simulated balance updates for clarity.

Performance display vs GUI callback:

Only calculate metrics once per loop, pass them to GUI.

Optional: Add a dry-run flag to skip real API calls if needed for faster debugging.

7. Overall Assessment

✅ Trustworthiness: High. Execution on Bybit demo is almost identical to live trading.

✅ Can you say $100 in paper ≈ $100 in real trading? Yes, for small/medium trades that don’t impact market depth.

✅ Strategy validation: Backtester + paper trader = 100% confidence in strategy logic correctness.

⚠️ Profit guarantee: Not guaranteed; strategy can still lose.

⚠️ Code cleanup: Some redundant calculations and logging, but functionally correct.

Bottom Line

This paper trader is as close to live trading as a developer can get without risking real money.

You can trust the results for strategy validation.

Minor code inefficiencies exist, but they do not affect execution or P&L.

Combining backtester + paper trader gives complete confidence that the strategy file works correctly in a live scenario.
        
        ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++       
        '''