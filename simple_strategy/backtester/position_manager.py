"""
Position Manager Component - Phase 1.3
Manages positions, balances, and trading limits within the backtester
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import uuid
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Data class representing a trading position"""
    symbol: str
    direction: str  # 'long' or 'short'
    size: float
    entry_price: float
    current_price: float
    entry_timestamp: datetime
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    position_id: str = None
    
    def __post_init__(self):
        if self.position_id is None:
            self.position_id = str(uuid.uuid4())
        self.update_unrealized_pnl()
    
    def update_unrealized_pnl(self):
        """Calculate unrealized P&L based on current price"""
        if self.direction == 'long':
            self.unrealized_pnl = (self.current_price - self.entry_price) * self.size
        elif self.direction == 'short':
            self.unrealized_pnl = (self.entry_price - self.current_price) * self.size
        else:
            self.unrealized_pnl = 0.0

@dataclass
class Trade:
    """Data class representing a completed trade"""
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    size: float
    entry_timestamp: datetime
    exit_timestamp: datetime
    pnl: float
    trade_id: str = None
    
    def __post_init__(self):
        if self.trade_id is None:
            self.trade_id = str(uuid.uuid4())
        # Calculate P&L
        if self.direction == 'long':
            self.pnl = (self.exit_price - self.entry_price) * self.size
        elif self.direction == 'short':
            self.pnl = (self.entry_price - self.exit_price) * self.size

class PositionManager:
    """
    Manages all trading positions, account balance, and trading limits
    """
    
    def __init__(self, initial_balance: float = 10000.0, max_positions: int = 3, 
                 max_risk_per_trade: float = 0.02):
        """
        Initialize position manager
        
        Args:
            initial_balance: Starting account balance
            max_positions: Maximum number of concurrent positions
            max_risk_per_trade: Maximum risk per trade as fraction of balance
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.max_positions = max_positions
        self.max_risk_per_trade = max_risk_per_trade
        
        # Storage
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.completed_trades: List[Trade] = []
        
        # Logging
        logger.info(f"PositionManager initialized with balance: ${initial_balance:.2f}")
    
    def can_open_position(self, symbol: str, position_size: float, 
                         entry_price: float) -> Tuple[bool, str]:
        """
        Check if a new position can be opened
        
        Args:
            symbol: Trading symbol
            position_size: Size of the position
            entry_price: Entry price
            
        Returns:
            Tuple of (can_open, reason)
        """
        # Check if symbol already has an open position
        if symbol in self.positions:
            return False, f"Position already open for {symbol}"
        
        # Check maximum positions limit
        if len(self.positions) >= self.max_positions:
            return False, f"Maximum positions ({self.max_positions}) reached"
        
        # Check if enough balance
        required_margin = position_size * entry_price
        if required_margin > self.current_balance:
            return False, f"Insufficient balance. Required: ${required_margin:.2f}, Available: ${self.current_balance:.2f}"
        
        # Check risk per trade
        risk_amount = self.current_balance * self.max_risk_per_trade
        if required_margin > risk_amount:
            return False, f"Position size exceeds risk limit. Max risk: ${risk_amount:.2f}, Required: ${required_margin:.2f}"
        
        return True, "Can open position"
    
    def open_position(self, symbol: str, direction: str, size: float, 
                     entry_price: float, timestamp: datetime) -> bool:
        """
        Open a new position
        
        Args:
            symbol: Trading symbol
            direction: 'long' or 'short'
            size: Position size
            entry_price: Entry price
            timestamp: Entry timestamp
            
        Returns:
            True if position opened successfully
        """
        # Validate direction
        if direction not in ['long', 'short']:
            logger.error(f"Invalid direction: {direction}")
            return False
        
        # Check if position can be opened
        can_open, reason = self.can_open_position(symbol, size, entry_price)
        if not can_open:
            logger.warning(f"Cannot open position for {symbol}: {reason}")
            return False
        
        # Create new position
        position = Position(
            symbol=symbol,
            direction=direction,
            size=size,
            entry_price=entry_price,
            current_price=entry_price,
            entry_timestamp=timestamp
        )
        
        # Store position
        self.positions[symbol] = position
        
        # Deduct from balance (margin)
        margin_required = size * entry_price
        self.current_balance -= margin_required
        
        logger.info(f"Opened {direction} position for {symbol}: size={size}, price=${entry_price:.2f}")
        return True
    
    def close_position(self, symbol: str, exit_price: float, 
                      timestamp: datetime) -> Optional[Trade]:
        """
        Close an existing position
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price
            timestamp: Exit timestamp
            
        Returns:
            Trade object if position closed successfully, None otherwise
        """
        if symbol not in self.positions:
            logger.warning(f"No open position for {symbol}")
            return None
        
        position = self.positions[symbol]
        
        # Calculate P&L
        if position.direction == 'long':
            pnl = (exit_price - position.entry_price) * position.size
        else:  # short
            pnl = (position.entry_price - exit_price) * position.size
        
        # Create trade record
        trade = Trade(
            symbol=position.symbol,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=exit_price,
            size=position.size,
            entry_timestamp=position.entry_timestamp,
            exit_timestamp=timestamp,
            pnl=pnl
        )
        
        # FIX: Proper balance management
        # Return the original margin
        original_margin = position.size * position.entry_price
        self.current_balance += original_margin
        
        # Add the profit/loss
        self.current_balance += pnl
        
        # Add trade to completed trades
        self.completed_trades.append(trade)
        
        # Remove position
        del self.positions[symbol]
        
        logger.info(f"Closed {position.direction} position for {symbol}: "
                    f"entry=${position.entry_price:.2f}, exit=${exit_price:.2f}, "
                    f"pnl=${pnl:.2f}")
        
        return trade
    
    def update_position_value(self, symbol: str, current_price: float):
        """
        Update position value with current price
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
        """
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        position.current_price = current_price
        position.update_unrealized_pnl()
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position details for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position object if exists, None otherwise
        """
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """
        Get all open positions
        
        Returns:
            Dictionary of all open positions
        """
        return self.positions.copy()
    
    def get_account_summary(self) -> Dict:
        """
        Get account balance and position summary
        
        Returns:
            Dictionary with account summary
        """
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_realized_pnl = sum(trade.pnl for trade in self.completed_trades)
        
        # Calculate total portfolio value
        total_margin_used = sum(pos.size * pos.entry_price for pos in self.positions.values())
        total_portfolio_value = self.current_balance + total_margin_used + total_unrealized_pnl
        
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'total_margin_used': total_margin_used,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_realized_pnl': total_realized_pnl,
            'total_portfolio_value': total_portfolio_value,
            'open_positions_count': len(self.positions),
            'completed_trades_count': len(self.completed_trades),
            'available_balance': self.current_balance,
            'max_positions': self.max_positions,
            'max_risk_per_trade': self.max_risk_per_trade
        }
    
    def get_trade_history(self) -> List[Trade]:
        """
        Get complete trade history
        
        Returns:
            List of all completed trades
        """
        return self.completed_trades.copy()
    
    def calculate_position_size(self, symbol: str, price: float,
                           risk_fraction: float = None) -> float:
        """
        Calculate safe position size based on risk - FIXED VERSION
        """
        if risk_fraction is None:
            risk_fraction = self.max_risk_per_trade
        
        # Calculate risk amount in dollars
        risk_amount = self.current_balance * risk_fraction
        
        # Calculate position size
        position_size = risk_amount / price
        
        # Apply minimum position sizes based on asset type
        if symbol.startswith('BTC'):
            min_position = 0.001  # Minimum 0.001 BTC
        elif symbol.startswith('ETH'):
            min_position = 0.01   # Minimum 0.01 ETH
        elif symbol.startswith('SOL'):
            min_position = 1.0    # Minimum 1 SOL
        else:
            min_position = 1.0    # Minimum 1 unit for other assets
        
        # Ensure minimum position size
        position_size = max(position_size, min_position)
        
        # Ensure we don't exceed available balance
        max_position = (self.current_balance * 0.95) / price  # Use 95% of balance
        position_size = min(position_size, max_position)
        
        print(f"📏 Position size calculation:")
        print(f"  - Symbol: {symbol}")
        print(f"  - Price: ${price:.2f}")
        print(f"  - Risk amount: ${risk_amount:.2f}")
        print(f"  - Position size: {position_size}")
        
        return position_size
    
    def get_positions_by_direction(self, direction: str) -> Dict[str, Position]:
        """
        Get positions filtered by direction
        
        Args:
            direction: 'long' or 'short'
            
        Returns:
            Dictionary of positions with specified direction
        """
        return {symbol: pos for symbol, pos in self.positions.items() 
                if pos.direction == direction}
    
    def force_close_all_positions(self, current_prices: Dict[str, float], 
                                timestamp: datetime) -> List[Trade]:
        """
        Force close all open positions (emergency function)
        
        Args:
            current_prices: Dictionary of symbol -> current price
            timestamp: Current timestamp
            
        Returns:
            List of closed trades
        """
        closed_trades = []
        
        for symbol in list(self.positions.keys()):
            if symbol in current_prices:
                trade = self.close_position(symbol, current_prices[symbol], timestamp)
                if trade:
                    closed_trades.append(trade)
        
        return closed_trades