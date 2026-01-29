"""
Performance Tracker Component - Phase 1.2
Tracks and calculates performance metrics for backtesting results
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradeRecord:
    """Data class representing a completed trade"""
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    size: float
    entry_timestamp: datetime
    exit_timestamp: datetime
    pnl: float
    trade_id: str
    duration: timedelta = None
    return_pct: float = 0.0
    
    def __post_init__(self):
        if self.duration is None:
            self.duration = self.exit_timestamp - self.entry_timestamp
        if self.return_pct == 0.0 and self.entry_price > 0:
            if self.direction == 'long':
                self.return_pct = ((self.exit_price - self.entry_price) / self.entry_price) * 100
            else:  # short
                self.return_pct = ((self.entry_price - self.exit_price) / self.entry_price) * 100

@dataclass
class PerformanceMetrics:
    """Data class containing all performance metrics"""
    # Basic metrics
    total_return: float
    total_return_pct: float
    final_balance: float
    initial_balance: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    win_rate_pct: float
    
    # Profit metrics
    gross_profit: float
    gross_loss: float
    net_profit: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Risk metrics
    max_drawdown: float
    max_drawdown_pct: float
    avg_drawdown: float
    recovery_factor: float
    
    # Efficiency metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    avg_trade_duration: timedelta
    
    # Additional metrics
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    expectancy: float

class PerformanceTracker:
    """
    Tracks and calculates comprehensive performance metrics for backtesting
    """
    
    def __init__(self, initial_balance: float = 10000.0):
        """
        Initialize performance tracker
        
        Args:
            initial_balance: Starting account balance
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        
        # Data storage
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self.balance_history: List[Dict[str, Any]] = []
        self.drawdown_history: List[Dict[str, Any]] = []
        
        # Performance cache
        self._metrics_cache: Optional[PerformanceMetrics] = None
        self._last_calculation_time: Optional[datetime] = None
        
        logger.info(f"PerformanceTracker initialized with balance: ${initial_balance:.2f}")
    
    def record_trade(self, trade_data: Dict[str, Any]) -> bool:
        """
        Record a completed trade
        
        Args:
            trade_data: Dictionary containing trade information
            
        Returns:
            True if trade was recorded successfully
        """
        try:
            # Validate required fields
            required_fields = ['symbol', 'direction', 'entry_price', 'exit_price', 
                             'size', 'entry_timestamp', 'exit_timestamp', 'pnl']
            
            for field in required_fields:
                if field not in trade_data:
                    logger.error(f"Missing required field in trade data: {field}")
                    return False
            
            # Create TradeRecord
            trade = TradeRecord(
                symbol=trade_data['symbol'],
                direction=trade_data['direction'],
                entry_price=trade_data['entry_price'],
                exit_price=trade_data['exit_price'],
                size=trade_data['size'],
                entry_timestamp=trade_data['entry_timestamp'],
                exit_timestamp=trade_data['exit_timestamp'],
                pnl=trade_data['pnl'],
                trade_id=trade_data.get('trade_id', f"{trade_data['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            )
            
            # Add to trades list
            self.trades.append(trade)
            # Update current balance
            self.current_balance+=trade.pnl
            # Invalidate cache
            self._metrics_cache = None
            self._last_calculation_time = None

            # Debug output
            logger.info(f"DEBUG: Recorded trade {trade.trade_id}: {trade.direction} {trade.symbol} pnl={trade.pnl}")
            logger.info(f"DEBUG: Total trades in tracker: {len(self.trades)}")

            return True
            
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
            return False
    
    def update_equity(self, timestamp: datetime, balance: float, positions_value: float = 0.0):
        """
        Update equity curve with current balance and positions value
        
        Args:
            timestamp: Current timestamp
            balance: Current account balance
            positions_value: Total value of open positions
        """
        total_equity = balance + positions_value
        
        equity_point = {
            'timestamp': timestamp,
            'balance': balance,
            'positions_value': positions_value,
            'total_equity': total_equity
        }
        
        self.equity_curve.append(equity_point)
        
        # Invalidate cache
        self._metrics_cache = None
        self._last_calculation_time = None
    
    def calculate_metrics(self, risk_free_rate: float = 0.02) -> PerformanceMetrics:
        """
        Calculate all performance metrics
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino ratios
            
        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        # Return cached metrics if available and recent
        if (self._metrics_cache is not None and 
            self._last_calculation_time is not None and
            (datetime.now() - self._last_calculation_time).seconds < 60):
            return self._metrics_cache
        
        # Calculate basic metrics
        total_return = (self.current_balance - self.initial_balance) / self.initial_balance
        total_return_pct = total_return * 100
        
        # Calculate trade statistics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.pnl > 0])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Calculate profit metrics
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        net_profit = gross_profit - gross_loss
        
        avg_win = gross_profit / winning_trades if winning_trades > 0 else 0.0
        avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        # Calculate drawdown metrics
        max_drawdown, max_drawdown_pct, avg_drawdown = self._calculate_drawdown_metrics()
        
        # Calculate risk-adjusted returns
        sharpe_ratio, sortino_ratio = self._calculate_risk_metrics(risk_free_rate)
        
        # Calculate additional metrics
        largest_win = max([t.pnl for t in self.trades]) if self.trades else 0.0
        largest_loss = min([t.pnl for t in self.trades]) if self.trades else 0.0
        
        consecutive_wins, consecutive_losses = self._calculate_consecutive_trades()
        expectancy = self._calculate_expectancy()
        
        # Calculate average trade duration
        avg_duration = self._calculate_avg_trade_duration()
        
        # Calculate recovery factor
        recovery_factor = net_profit / max_drawdown if max_drawdown > 0 else 0.0
        
        # Calculate Calmar ratio
        if max_drawdown_pct > 0:
            calmar_ratio = abs(total_return_pct / max_drawdown_pct)
        else:
            calmar_ratio = 0.0
        
        # Create PerformanceMetrics object
        metrics = PerformanceMetrics(
            total_return=total_return,
            total_return_pct=total_return_pct,
            final_balance=self.current_balance,
            initial_balance=self.initial_balance,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            win_rate_pct=win_rate * 100,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            net_profit=net_profit,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            avg_drawdown=avg_drawdown,
            recovery_factor=recovery_factor,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            avg_trade_duration=avg_duration,
            largest_win=largest_win,
            largest_loss=largest_loss,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            expectancy=expectancy
        )
        
        # Cache the results
        self._metrics_cache = metrics
        self._last_calculation_time = datetime.now()
        
        return metrics
    
    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get equity curve as pandas DataFrame
        
        Returns:
            DataFrame with equity curve data
        """
        if not self.equity_curve:
            return pd.DataFrame()
        
        return pd.DataFrame(self.equity_curve).set_index('timestamp')
    
    def get_trade_history(self) -> pd.DataFrame:
        """
        Get complete trade history as pandas DataFrame
        
        Returns:
            DataFrame with all trade records
        """
        if not self.trades:
            return pd.DataFrame()
        
        # Convert trade records to dictionaries
        trades_data = []
        for trade in self.trades:
            trade_dict = asdict(trade)
            trade_dict['duration_minutes'] = trade.duration.total_seconds() / 60
            trades_data.append(trade_dict)
        
        return pd.DataFrame(trades_data)
    
    def get_drawdown_periods(self) -> pd.DataFrame:
        """
        Get drawdown periods as pandas DataFrame
        
        Returns:
            DataFrame with drawdown period information
        """
        if not self.equity_curve:
            return pd.DataFrame()
        
        equity_df = self.get_equity_curve()
        equity_df = equity_df.sort_index()  # Ensure chronological order
        
        # Calculate drawdown series
        equity_df['drawdown'] = self._calculate_drawdown_series(equity_df['total_equity'])
        equity_df['running_max'] = equity_df['total_equity'].expanding().max()
        
        # Identify drawdown periods
        drawdown_periods = []
        in_drawdown = False
        drawdown_start = None
        peak_value = 0
        trough_value = float('inf')
        
        for i in range(len(equity_df)):
            current_equity = equity_df['total_equity'].iloc[i]
            current_drawdown = equity_df['drawdown'].iloc[i]
            current_running_max = equity_df['running_max'].iloc[i]
            timestamp = equity_df.index[i]
            
            # Check if we're in a drawdown (drawdown > 0.1% to avoid noise)
            if current_drawdown > 0.001:
                if not in_drawdown:
                    # Start of new drawdown period
                    in_drawdown = True
                    drawdown_start = timestamp
                    peak_value = current_running_max
                    trough_value = current_equity
                else:
                    # Continue drawdown, update trough if we've gone lower
                    trough_value = min(trough_value, current_equity)
            else:
                if in_drawdown:
                    # End of drawdown period
                    drawdown_end = timestamp
                    drawdown_depth = peak_value - trough_value
                    drawdown_pct = (drawdown_depth / peak_value) * 100
                    duration = drawdown_end - drawdown_start
                    
                    drawdown_periods.append({
                        'start_time': drawdown_start,
                        'end_time': drawdown_end,
                        'duration': duration,
                        'peak_value': peak_value,
                        'trough_value': trough_value,
                        'drawdown_amount': drawdown_depth,
                        'drawdown_pct': drawdown_pct
                    })
                    
                    in_drawdown = False
        
        # Handle case where we're still in drawdown at the end
        if in_drawdown:
            drawdown_end = equity_df.index[-1]
            drawdown_depth = peak_value - trough_value
            drawdown_pct = (drawdown_depth / peak_value) * 100
            duration = drawdown_end - drawdown_start
            
            drawdown_periods.append({
                'start_time': drawdown_start,
                'end_time': drawdown_end,
                'duration': duration,
                'peak_value': peak_value,
                'trough_value': trough_value,
                'drawdown_amount': drawdown_depth,
                'drawdown_pct': drawdown_pct
            })
        
        return pd.DataFrame(drawdown_periods)
    
    def get_symbol_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance breakdown by symbol
        
        Returns:
            Dictionary with performance metrics for each symbol
        """
        symbol_stats = {}
        
        # Group trades by symbol
        for symbol in set(trade.symbol for trade in self.trades):
            symbol_trades = [t for t in self.trades if t.symbol == symbol]
            
            if not symbol_trades:
                continue
            
            # Calculate metrics for this symbol
            total_trades = len(symbol_trades)
            winning_trades = len([t for t in symbol_trades if t.pnl > 0])
            total_pnl = sum(t.pnl for t in symbol_trades)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            avg_win = np.mean([t.pnl for t in symbol_trades if t.pnl > 0]) if winning_trades > 0 else 0.0
            avg_loss = np.mean([t.pnl for t in symbol_trades if t.pnl < 0]) if (total_trades - winning_trades) > 0 else 0.0
            
            symbol_stats[symbol] = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
            }
        
        return symbol_stats
    
    def export_results(self, filepath: str, include_charts: bool = False) -> bool:
        """
        Export performance results to file
        
        Args:
            filepath: Path to save results
            include_charts: Whether to include chart data
            
        Returns:
            True if export was successful
        """
        try:
            # Calculate metrics
            metrics = self.calculate_metrics()
            
            # Prepare export data
            export_data = {
                'metrics': asdict(metrics),
                'trades': [asdict(trade) for trade in self.trades],
                'equity_curve': self.equity_curve,
                'export_timestamp': datetime.now().isoformat()
            }
            
            # Save to file
            file_path = Path(filepath)
            
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                # Try to export as Excel, but handle missing openpyxl gracefully
                try:
                    import openpyxl  # Check if openpyxl is available
                    
                    metrics_df = pd.DataFrame([asdict(metrics)])
                    trades_df = self.get_trade_history()
                    
                    with pd.ExcelWriter(file_path) as writer:
                        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                        trades_df.to_excel(writer, sheet_name='Trades', index=False)
                        
                except ImportError:
                    logger.warning("openpyxl not available. Falling back to CSV format.")
                    # Fall back to CSV format
                    base_path = str(file_path).replace(file_path.suffix, '')
                    metrics_df = pd.DataFrame([asdict(metrics)])
                    trades_df = self.get_trade_history()
                    
                    metrics_df.to_csv(f"{base_path}_metrics.csv", index=False)
                    trades_df.to_csv(f"{base_path}_trades.csv", index=False)
                    
                    # Update filepath to reflect the change
                    logger.info(f"Results exported to CSV files: {base_path}_metrics.csv and {base_path}_trades.csv")
                    return True
            elif file_path.suffix.lower() == '.csv':
                # Export as CSV (metrics and trades separately)
                metrics_df = pd.DataFrame([asdict(metrics)])
                trades_df = self.get_trade_history()
                
                # Create CSV files with different names
                base_path = str(file_path).replace('.csv', '')
                metrics_df.to_csv(f"{base_path}_metrics.csv", index=False)
                trades_df.to_csv(f"{base_path}_trades.csv", index=False)
                
                logger.info(f"Results exported to CSV files: {base_path}_metrics.csv and {base_path}_trades.csv")
            else:
                logger.error(f"Unsupported file format: {file_path.suffix}")
                return False
            
            logger.info(f"Results exported to: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return False
    
    def _calculate_drawdown_metrics(self) -> Tuple[float, float, float]:
        """
        Calculate drawdown-related metrics
        
        Returns:
            Tuple of (max_drawdown, max_drawdown_pct, avg_drawdown)
        """
        if not self.equity_curve:
            return 0.0, 0.0, 0.0
        
        equity_df = self.get_equity_curve()
        drawdown_series = self._calculate_drawdown_series(equity_df['total_equity'])
        
        max_drawdown = drawdown_series.max()
        max_drawdown_pct = max_drawdown * 100
        avg_drawdown = drawdown_series.mean()
        
        return max_drawdown, max_drawdown_pct, avg_drawdown
    
    def _calculate_drawdown_series(self, equity_series: pd.Series) -> pd.Series:
        """
        Calculate drawdown series from equity series
        
        Args:
            equity_series: Series of equity values
            
        Returns:
            Series of drawdown values (as decimal)
        """
        running_max = equity_series.expanding().max()
        drawdown = (running_max - equity_series) / running_max
        return drawdown.fillna(0)
    
    def _calculate_risk_metrics(self, risk_free_rate: float) -> Tuple[float, float]:
        """
        Calculate risk-adjusted return metrics
        
        Args:
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Tuple of (sharpe_ratio, sortino_ratio)
        """
        if not self.equity_curve or len(self.equity_curve) < 2:
            return 0.0, 0.0
        
        equity_df = self.get_equity_curve()
        
        # Calculate returns
        returns = equity_df['total_equity'].pct_change().dropna()
        
        if len(returns) == 0:
            return 0.0, 0.0
        
        # Annualize returns and volatility
        trading_days_per_year = 252
        periods_per_day = len(returns) / len(equity_df.index.normalize().unique())
        periods_per_year = trading_days_per_year * periods_per_day
        
        annual_return = returns.mean() * periods_per_year
        annual_volatility = returns.std() * np.sqrt(periods_per_year)
        
        # Calculate Sharpe ratio
        if annual_volatility > 0:
            sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        else:
            sharpe_ratio = 0.0
        
        # Calculate Sortino ratio (only consider negative returns)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_volatility = negative_returns.std() * np.sqrt(periods_per_year)
            if downside_volatility > 0:
                sortino_ratio = (annual_return - risk_free_rate) / downside_volatility
            else:
                sortino_ratio = 0.0
        else:
            sortino_ratio = 0.0
        
        return sharpe_ratio, sortino_ratio
    
    def _calculate_consecutive_trades(self) -> Tuple[int, int]:
        """
        Calculate consecutive winning and losing trades
        
        Returns:
            Tuple of (max_consecutive_wins, max_consecutive_losses)
        """
        if not self.trades:
            return 0, 0
        
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in self.trades:
            if trade.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
        
        return max_consecutive_wins, max_consecutive_losses
    
    def _calculate_expectancy(self) -> float:
        """
        Calculate trade expectancy
        
        Returns:
            Expectancy value
        """
        if not self.trades:
            return 0.0
        
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        if not winning_trades or not losing_trades:
            return 0.0
        
        win_rate = len(winning_trades) / len(self.trades)
        avg_win = np.mean([t.pnl for t in winning_trades])
        avg_loss = abs(np.mean([t.pnl for t in losing_trades]))
        
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        return expectancy
    
    def _calculate_avg_trade_duration(self) -> timedelta:
        """
        Calculate average trade duration
        
        Returns:
            Average duration as timedelta
        """
        if not self.trades:
            return timedelta(0)
        
        total_duration = sum(
            (t.exit_timestamp - t.entry_timestamp for t in self.trades),
            timedelta(0)
        )
        avg_duration = total_duration / len(self.trades)
        
        return avg_duration
    
    def reset(self):
        """Reset all performance tracking data"""
        self.trades.clear()
        self.equity_curve.clear()
        self.balance_history.clear()
        self.drawdown_history.clear()
        self.current_balance = self.initial_balance
        self._metrics_cache = None
        self._last_calculation_time = None
        
        logger.info("PerformanceTracker reset")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of key performance metrics
        
        Returns:
            Dictionary with summary metrics
        """
        metrics = self.calculate_metrics()
        
        return {
            'total_return_pct': f"{metrics.total_return_pct:.2f}%",
            'final_balance': f"${metrics.final_balance:.2f}",
            'total_trades': metrics.total_trades,
            'win_rate_pct': f"{metrics.win_rate_pct:.1f}%",
            'profit_factor': f"{metrics.profit_factor:.2f}",
            'max_drawdown_pct': f"{metrics.max_drawdown_pct:.2f}%",
            'sharpe_ratio': f"{metrics.sharpe_ratio:.2f}",
            'expectancy': f"${metrics.expectancy:.2f}"
        }