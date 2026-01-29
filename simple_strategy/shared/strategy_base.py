# strategy_base.py - Abstract base class and building blocks for strategies
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrategyBase(ABC):
    """
    Abstract base class for all trading strategies.
    Provides common functionality and enforces consistent interface.
    """
    def __init__(self, name: str, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        """
        Initialize strategy with configuration.
        Args:
            name: Strategy name
            symbols: List of trading symbols
            timeframes: List of timeframes to analyze
            config: Strategy configuration parameters
        """
        self.name = name
        self.symbols = symbols
        self.timeframes = timeframes
        self.config = config
        # Strategy state
        self.positions = {}  # symbol -> position info
        self.balance = config.get('initial_balance', 10000.0)
        self.initial_balance = self.balance
        # Risk management parameters
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.01)  # 1% of balance
        self.max_positions = config.get('max_positions', 3)
        self.max_portfolio_risk = config.get('max_portfolio_risk', 0.10)  # 10% of balance
        # Performance tracking
        self.trades = []
        self.equity_curve = []
        logger.info(f"Strategy {name} initialized with symbols: {symbols}, timeframes: {timeframes}")

    @abstractmethod
    def generate_signals(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, str]]:
        """
        Generate trading signals for all symbols and timeframes.
        Must be implemented by subclasses.
        Args:
            data: Nested dictionary {symbol: {timeframe: DataFrame}}
        Returns:
            Dictionary {symbol: {timeframe: signal}} where signal is 'BUY', 'SELL', or 'HOLD'
        """
        pass

    def calculate_position_size(self, symbol: str, current_price: float = None, signal_strength: float = 1.0) -> float:
        """
        Calculate position size based on risk management rules.
        Args:
            symbol: Trading symbol
            current_price: Current price of the asset
            signal_strength: Strength of the signal (0.0 to 1.0)
        Returns:
            Position size in units of the asset
        """
        if current_price is None:
            # We don't have the current price, so we can't calculate the position size accurately
            # Return a small fixed size as a fallback
            if symbol.startswith('BTC'):
                return 0.001  # 0.001 BTC
            elif symbol.startswith('ETH'):
                return 0.01   # 0.01 ETH
            else:
                return 1.0    # 1 unit of other assets
        
        # Calculate risk amount for this trade
        risk_amount = self.balance * self.max_risk_per_trade * signal_strength
        
        # Calculate position size in units of the asset
        position_size = risk_amount / current_price
        
        # Ensure we don't exceed maximum position size
        # Max position size is a fraction of the account balance
        max_position_value = self.balance * self.max_positions / 10  # Distribute among max positions
        max_position_size = max_position_value / current_price
        
        position_size = min(position_size, max_position_size)
        
        # For crypto assets, we might want to round to a reasonable number of decimal places
        if symbol.startswith('BTC'):
            position_size = round(position_size, 6)  # Bitcoin can be divided to 8 decimal places, but 6 is reasonable for trading
        elif symbol.startswith('ETH'):
            position_size = round(position_size, 4)  # Ethereum can be divided to 18 decimal places, but 4 is reasonable
        else:
            position_size = round(position_size, 2)  # Other assets
        
        return position_size

    def validate_signal(self, symbol: str, signal: str, data: Dict[str, pd.DataFrame]) -> bool:
        """
        Validate signal against risk management rules.
        Args:
            symbol: Trading symbol
            signal: Trading signal ('BUY', 'SELL', 'HOLD')
            data: Current market data
        Returns:
            True if signal is valid, False otherwise
        """
        if signal == 'HOLD':
            return True
        # Check if we already have maximum positions
        if signal == 'BUY' and len(self.positions) >= self.max_positions:
            logger.warning(f"Signal validation failed: Maximum positions reached ({len(self.positions)}/{self.max_positions})")
            return False
        # Check portfolio risk
        if signal == 'BUY':
            portfolio_risk = self._calculate_portfolio_risk()
            if portfolio_risk >= self.max_portfolio_risk:
                logger.warning(f"Signal validation failed: Maximum portfolio risk reached ({portfolio_risk:.2%})")
                return False
        # Check if we have position to sell
        if signal == 'SELL' and symbol not in self.positions:
            logger.warning(f"Signal validation failed: No position to sell for {symbol}")
            return False
        return True

    def get_strategy_state(self) -> Dict[str, Any]:
        """
        Get current strategy state for logging and monitoring.
        Returns:
            Dictionary with strategy state information
        """
        return {
            'name': self.name,
            'balance': self.balance,
            'initial_balance': self.initial_balance,
            'total_return': (self.balance - self.initial_balance) / self.initial_balance,
            'open_positions': len(self.positions),
            'total_trades': len(self.trades),
            'symbols': self.symbols,
            'timeframes': self.timeframes,
            'config': self.config
        }

    def _get_current_price(self, symbol: str) -> float:
        """
        Get current price for a symbol.
        Placeholder method - should be overridden or data should be passed in.
        Args:
            symbol: Trading symbol
        Returns:
            Current price
        """
        # This is a placeholder - in practice, this would come from the data feeder
        logger.warning(f"Using placeholder price for {symbol}")
        return 50000.0  # Placeholder price

    def _calculate_portfolio_risk(self) -> float:
        """
        Calculate current portfolio risk.
        Returns:
            Portfolio risk as percentage of balance
        """
        # Simple calculation - in practice, this would be more sophisticated
        total_position_value = sum(pos.get('value', 0) for pos in self.positions.values())
        return total_position_value / self.balance if self.balance > 0 else 0

    # ============================================================================
    # INDICATOR METHODS (make functions accessible as methods)
    # ============================================================================
    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI - wrapper for the standalone function"""
        return calculate_rsi_func(data, period)

    def calculate_sma(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate SMA - wrapper for the standalone function"""
        return calculate_sma_func(data, period)

    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate EMA - wrapper for the standalone function"""
        return calculate_ema_func(data, period)

    def calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> tuple:
        """Calculate Stochastic - wrapper for the standalone function"""
        return calculate_stochastic_func(data, k_period, d_period)

    def calculate_srsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate SRSI - wrapper for the standalone function"""
        return calculate_srsi_func(data, period)

# ============================================================================
# INDICATOR BUILDING BLOCKS (renamed to avoid naming conflicts)
# ============================================================================
def calculate_rsi_func(data: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    Args:
        data: Price series (typically closing prices)
        period: RSI period (default: 14)
    Returns:
        RSI values as pandas Series
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_sma_func(data: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).
    Args:
        data: Price series
        period: SMA period
    Returns:
        SMA values as pandas Series
    """
    return data.rolling(window=period).mean()

def calculate_ema_func(data: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).
    Args:
        data: Price series
        period: EMA period
    Returns:
        EMA values as pandas Series
    """
    return data.ewm(span=period, adjust=False).mean()

def calculate_stochastic_func(data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> tuple:
    """
    Calculate Stochastic Oscillator.
    Args:
        data: DataFrame with 'high', 'low', 'close' columns
        k_period: %K period (default: 14)
        d_period: %D period (default: 3)
    Returns:
        Tuple of (%K, %D) as pandas Series
    """
    low_min = data['low'].rolling(window=k_period).min()
    high_max = data['high'].rolling(window=k_period).max()
    k_percent = 100 * ((data['close'] - low_min) / (high_max - low_min))
    d_percent = k_percent.rolling(window=d_period).mean()
    return k_percent, d_percent

def calculate_srsi_func(data: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Stochastic RSI.
    Args:
        data: Price series
        period: SRSI period (default: 14)
    Returns:
        SRSI values as pandas Series
    """
    rsi = calculate_rsi_func(data, period)
    # Create DataFrame for stochastic calculation
    stochastic_data = pd.DataFrame({
        'high': rsi,
        'low': rsi,
        'close': rsi
    })
    # FIXED: Use k_period instead of period
    k_percent, d_percent = calculate_stochastic_func(stochastic_data, k_period=period, d_period=3)
    return k_percent

# ============================================================================
# SIGNAL BUILDING BLOCKS
# ============================================================================
def check_oversold(indicator_value: pd.Series, threshold: float = 20) -> pd.Series:
    """
    Check if indicator is in oversold territory.
    Args:
        indicator_value: Indicator values
        threshold: Oversold threshold (default: 20)
    Returns:
        Boolean series indicating oversold condition
    """
    return indicator_value <= threshold

def check_overbought(indicator_value: pd.Series, threshold: float = 80) -> pd.Series:
    """
    Check if indicator is in overbought territory.
    Args:
        indicator_value: Indicator values
        threshold: Overbought threshold (default: 80)
    Returns:
        Boolean series indicating overbought condition
    """
    return indicator_value >= threshold

def check_crossover(fast_ma: pd.Series, slow_ma: pd.Series) -> pd.Series:
    """
    Check for moving average crossover.
    Args:
        fast_ma: Fast moving average series
        slow_ma: Slow moving average series
    Returns:
        Boolean series indicating crossover (fast crosses above slow)
    """
    # Create shifted series for comparison
    fast_prev = fast_ma.shift(1)
    slow_prev = slow_ma.shift(1)
    # Crossover condition: (current fast > current slow) AND (previous fast <= previous slow)
    crossover = (fast_ma > slow_ma) & (fast_prev <= slow_prev)
    # First value can never be a crossover (no previous data)
    crossover.iloc[0] = False
    return crossover

def check_crossunder(fast_ma: pd.Series, slow_ma: pd.Series) -> pd.Series:
    """
    Check for moving average crossunder.
    Args:
        fast_ma: Fast moving average series
        slow_ma: Slow moving average series
    Returns:
        Boolean series indicating crossunder (fast crosses below slow)
    """
    # Create shifted series for comparison
    fast_prev = fast_ma.shift(1)
    slow_prev = slow_ma.shift(1)
    # Crossunder condition: (current fast < current slow) AND (previous fast >= previous slow)
    crossunder = (fast_ma < slow_ma) & (fast_prev >= slow_prev)
    # First value can never be a crossunder (no previous data)
    crossunder.iloc[0] = False
    return crossunder

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def validate_data_format(data: pd.DataFrame) -> bool:
    """
    Validate that data has required columns and format.
    Args:
        data: DataFrame to validate
    Returns:
        True if data format is valid
    """
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    return all(col in data.columns for col in required_columns)

def align_multi_timeframe_data(data_dict: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Align multi-timeframe data to ensure consistent timestamps.
    Args:
        data_dict: Nested dictionary {symbol: {timeframe: DataFrame}}
    Returns:
        Aligned data dictionary
    """
    aligned_data = {}
    for symbol, timeframe_data in data_dict.items():
        aligned_data[symbol] = {}
        for timeframe, df in timeframe_data.items():
            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            aligned_data[symbol][timeframe] = df
    return aligned_data

