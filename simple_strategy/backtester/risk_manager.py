"""
Risk Manager Component - Phase 2.2
Implements risk management rules and calculations for the backtesting system
Handles position sizing, signal validation, portfolio risk, and stop-loss mechanisms
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskManager:
    """
    Risk Manager component that implements risk management rules and calculations
    """
    
    def __init__(self, max_risk_per_trade: float = 0.02, max_portfolio_risk: float = 0.10,
                 max_positions: int = 10, default_stop_loss_pct: float = 0.02):
        """
        Initialize risk manager with risk parameters
        
        Args:
            max_risk_per_trade: Maximum risk per trade as percentage of account balance (default: 2%)
            max_portfolio_risk: Maximum total portfolio risk (default: 10%)
            max_positions: Maximum number of concurrent positions (default: 10)
            default_stop_loss_pct: Default stop loss percentage (default: 2%)
        """
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.max_positions = max_positions
        self.default_stop_loss_pct = default_stop_loss_pct
        
        # Risk management strategies
        self.risk_strategies = {
            'fixed_percentage': self._fixed_percentage_sizing,
            'volatility_based': self._volatility_based_sizing,
            'kelly_criterion': self._kelly_criterion_sizing
        }
        
        logger.info(f"RiskManager initialized with max_risk_per_trade={max_risk_per_trade}, "
                   f"max_portfolio_risk={max_portfolio_risk}, max_positions={max_positions}")
    
    def calculate_position_size(self, symbol: str, price: float, account_balance: float,
                              risk_amount: Optional[float] = None, 
                              strategy: str = 'fixed_percentage',
                              volatility: Optional[float] = None,
                              win_rate: Optional[float] = None,
                              avg_win: Optional[float] = None,
                              avg_loss: Optional[float] = None) -> float:
        """
        Calculate safe position size based on risk management rules
        
        Args:
            symbol: Trading symbol
            price: Current price of the asset
            account_balance: Total account balance
            risk_amount: Specific risk amount (optional, uses max_risk_per_trade if None)
            strategy: Position sizing strategy ('fixed_percentage', 'volatility_based', 'kelly_criterion')
            volatility: Asset volatility (for volatility-based sizing)
            win_rate: Historical win rate (for Kelly criterion)
            avg_win: Average win amount (for Kelly criterion)
            avg_loss: Average loss amount (for Kelly criterion)
            
        Returns:
            Safe position size in base currency
        """
        try:
            # Use provided risk amount or default to max risk per trade
            if risk_amount is None:
                risk_amount = account_balance * self.max_risk_per_trade
            
            # Validate inputs
            if price <= 0:
                logger.warning(f"Invalid price for {symbol}: {price}")
                return 0.0
            
            if account_balance <= 0:
                logger.warning("Invalid account balance")
                return 0.0
            
            # Use selected strategy
            if strategy in self.risk_strategies:
                position_size = self.risk_strategies[strategy](
                    symbol, price, account_balance, risk_amount,
                    volatility, win_rate, avg_win, avg_loss
                )
            else:
                logger.warning(f"Unknown risk strategy: {strategy}, using fixed_percentage")
                position_size = self._fixed_percentage_sizing(
                    symbol, price, account_balance, risk_amount,
                    volatility, win_rate, avg_win, avg_loss
                )
            
            # Apply maximum position size limit (never risk more than account balance)
            max_position_size = account_balance / price
            position_size = min(position_size, max_position_size)
            
            logger.debug(f"Calculated position size for {symbol}: {position_size} at price {price}")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0
    
    def _fixed_percentage_sizing(self, symbol: str, price: float, account_balance: float,
                               risk_amount: float, volatility: Optional[float] = None,
                               win_rate: Optional[float] = None, avg_win: Optional[float] = None,
                               avg_loss: Optional[float] = None) -> float:
        """Fixed percentage position sizing strategy"""
        # Fixed percentage sizing: position_size = risk_amount / (price * stop_loss_pct)
        # This calculates the position size based on the risk amount and stop loss distance
        stop_loss_distance = price * self.default_stop_loss_pct
        position_size = risk_amount / stop_loss_distance
        return position_size
    
    def _volatility_based_sizing(self, symbol: str, price: float, account_balance: float,
                               risk_amount: float, volatility: Optional[float] = None,
                               win_rate: Optional[float] = None, avg_win: Optional[float] = None,
                               avg_loss: Optional[float] = None) -> float:
        """Volatility-based position sizing strategy"""
        if volatility is None:
            logger.warning("Volatility not provided, using fixed percentage sizing")
            return self._fixed_percentage_sizing(symbol, price, account_balance, risk_amount)
        
        # Adjust position size based on volatility (higher volatility = smaller position)
        volatility_factor = 1.0 / (1.0 + volatility)
        adjusted_risk = risk_amount * volatility_factor
        stop_loss_distance = price * self.default_stop_loss_pct
        position_size = adjusted_risk / stop_loss_distance
        
        return position_size
    
    def _kelly_criterion_sizing(self, symbol: str, price: float, account_balance: float,
                              risk_amount: float, volatility: Optional[float] = None,
                              win_rate: Optional[float] = None, avg_win: Optional[float] = None,
                              avg_loss: Optional[float] = None) -> float:
        """Kelly criterion position sizing strategy"""
        if win_rate is None or avg_win is None or avg_loss is None:
            logger.warning("Kelly criterion parameters not provided, using fixed percentage sizing")
            return self._fixed_percentage_sizing(symbol, price, account_balance, risk_amount)
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win / avg_loss, p = win_rate, q = 1 - win_rate
        if avg_loss == 0:
            logger.warning("Average loss is zero, cannot use Kelly criterion")
            return 0.0
        
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Limit Kelly fraction to avoid over-leveraging
        kelly_fraction = max(0.0, min(kelly_fraction, 0.25))  # Max 25% of account
        
        position_size = (account_balance * kelly_fraction) / price
        
        return position_size
    
    def validate_trade_signal(self, signal: Dict[str, Any], account_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate trading signal against risk management rules
        
        Args:
            signal: Trading signal dictionary
            account_state: Current account state including positions and balance
            
        Returns:
            Validation result with status and reason if rejected
        """
        try:
            result = {
                'valid': True,
                'reason': None,
                'adjusted_position_size': None
            }
            
            # Check if signal has required fields
            required_fields = ['symbol', 'signal_type', 'price', 'timestamp']
            for field in required_fields:
                if field not in signal:
                    result['valid'] = False
                    result['reason'] = f"Missing required field: {field}"
                    return result
            
            symbol = signal['symbol']
            signal_type = signal['signal_type']
            price = signal['price']
            
            # Check account state
            if 'balance' not in account_state:
                result['valid'] = False
                result['reason'] = "Missing account balance"
                return result
            
            balance = account_state['balance']
            positions = account_state.get('positions', {})
            
            # Validate signal type
            if signal_type not in ['BUY', 'SELL']:
                result['valid'] = False
                result['reason'] = f"Invalid signal type: {signal_type}"
                return result
            
            # For BUY signals, check if we can open a new position
            if signal_type == 'BUY':
                # Check maximum positions limit
                if len(positions) >= self.max_positions:
                    result['valid'] = False
                    result['reason'] = f"Maximum positions limit reached: {self.max_positions}"
                    return result
                
                # Check if we already have a position for this symbol
                if symbol in positions:
                    result['valid'] = False
                    result['reason'] = f"Already have position for {symbol}"
                    return result
                
                # Calculate position size
                position_size = self.calculate_position_size(
                    symbol, price, balance
                )
                
                if position_size <= 0:
                    result['valid'] = False
                    result['reason'] = "Calculated position size is zero or negative"
                    return result
                
                result['adjusted_position_size'] = position_size
            
            # For SELL signals, check if we have a position to close
            elif signal_type == 'SELL':
                if symbol not in positions:
                    result['valid'] = False
                    result['reason'] = f"No position to close for {symbol}"
                    return result
            
            # Check portfolio risk
            portfolio_risk = self.calculate_portfolio_risk(positions, balance)
            if portfolio_risk > self.max_portfolio_risk:
                result['valid'] = False
                result['reason'] = f"Portfolio risk too high: {portfolio_risk:.2%}"
                return result
            
            logger.debug(f"Signal validation result for {symbol}: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error validating trade signal: {e}")
            return {
                'valid': False,
                'reason': f"Validation error: {str(e)}",
                'adjusted_position_size': None
            }
    
    def calculate_portfolio_risk(self, positions: Dict[str, Dict[str, Any]], account_balance: float = 10000.0) -> float:
        """
        Calculate current portfolio risk
        
        Args:
            positions: Dictionary of current positions
            account_balance: Current account balance
            
        Returns:
            Portfolio risk as percentage of account balance
        """
        try:
            if not positions:
                return 0.0
            
            total_position_value = 0.0
            
            for symbol, position in positions.items():
                if 'size' in position and 'current_price' in position:
                    position_value = position['size'] * position['current_price']
                    total_position_value += position_value
            
            # Calculate portfolio risk as percentage of account balance
            if account_balance > 0:
                portfolio_risk = total_position_value / account_balance
            else:
                portfolio_risk = 0.0
            
            # Cap at 100% (1.0)
            portfolio_risk = min(portfolio_risk, 1.0)
            
            logger.debug(f"Portfolio risk calculated: {portfolio_risk:.2%}")
            return portfolio_risk
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return 0.0
    
    def check_stop_loss(self, position: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """
        Check if stop-loss should be triggered for a position
        
        Args:
            position: Position dictionary
            current_price: Current market price
            
        Returns:
            Stop-loss check result
        """
        try:
            result = {
                'triggered': False,
                'reason': None,
                'stop_price': None
            }
            
            # Check if position has required fields
            required_fields = ['symbol', 'direction', 'entry_price', 'size']
            for field in required_fields:
                if field not in position:
                    result['triggered'] = False
                    result['reason'] = f"Missing required field: {field}"
                    return result
            
            symbol = position['symbol']
            direction = position['direction']
            entry_price = position['entry_price']
            
            # Get stop loss percentage (use position-specific or default)
            stop_loss_pct = position.get('stop_loss_pct', self.default_stop_loss_pct)
            
            # Calculate stop price based on position direction
            stop_price = None
            if direction == 'long':
                stop_price = entry_price * (1 - stop_loss_pct)
                if current_price <= stop_price:
                    result['triggered'] = True
                    result['reason'] = f"Stop-loss triggered for {symbol}: {current_price} <= {stop_price}"
                    result['stop_price'] = stop_price
            elif direction == 'short':
                stop_price = entry_price * (1 + stop_loss_pct)
                if current_price >= stop_price:
                    result['triggered'] = True
                    result['reason'] = f"Stop-loss triggered for {symbol}: {current_price} >= {stop_price}"
                    result['stop_price'] = stop_price
            else:
                result['triggered'] = False
                result['reason'] = f"Unknown position direction: {direction}"
            
            logger.debug(f"Stop-loss check for {symbol}: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error checking stop-loss: {e}")
            return {
                'triggered': False,
                'reason': f"Stop-loss check error: {str(e)}",
                'stop_price': None
            }
    
    def set_risk_parameters(self, max_risk_per_trade: Optional[float] = None,
                           max_portfolio_risk: Optional[float] = None,
                           max_positions: Optional[int] = None,
                           default_stop_loss_pct: Optional[float] = None):
        """
        Update risk management parameters
        
        Args:
            max_risk_per_trade: New maximum risk per trade
            max_portfolio_risk: New maximum portfolio risk
            max_positions: New maximum positions
            default_stop_loss_pct: New default stop loss percentage
        """
        if max_risk_per_trade is not None:
            self.max_risk_per_trade = max_risk_per_trade
        if max_portfolio_risk is not None:
            self.max_portfolio_risk = max_portfolio_risk
        if max_positions is not None:
            self.max_positions = max_positions
        if default_stop_loss_pct is not None:
            self.default_stop_loss_pct = default_stop_loss_pct
        
        logger.info(f"Risk parameters updated: max_risk_per_trade={self.max_risk_per_trade}, "
                   f"max_portfolio_risk={self.max_portfolio_risk}, max_positions={self.max_positions}")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """
        Get current risk management configuration summary
        
        Returns:
            Dictionary with current risk parameters
        """
        return {
            'max_risk_per_trade': self.max_risk_per_trade,
            'max_portfolio_risk': self.max_portfolio_risk,
            'max_positions': self.max_positions,
            'default_stop_loss_pct': self.default_stop_loss_pct,
            'available_strategies': list(self.risk_strategies.keys())
        }

