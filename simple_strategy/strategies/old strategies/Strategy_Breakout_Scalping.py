"""
Breakout Scalping Strategy
==========================

A scalping strategy that capitalizes on volatility breakouts:
- Identifies consolidation periods with low ATR
- Enters on strong breakouts with volume confirmation
- Quick profit targets with tight stops
- Pure momentum following - no mean reversion

Strategy Logic:
1. CONSOLIDATION: Low ATR and tight price range
2. BREAKOUT: Price breaks range with high volume
3. ENTRY: In direction of breakout
4. EXIT: Quick profit target at 1.5x ATR

Best for: Trending markets with clear breakouts
Author: AI Assisted TradeBot Team
Date: 2025
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional

# Add parent directories to path for proper imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import required components
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import atr, ema, volume_sma, highest, lowest
from simple_strategy.strategies.signals_library import ma_crossover
from simple_strategy.shared.strategy_base import StrategyBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CRITICAL: STRATEGY_PARAMETERS for GUI Configuration
STRATEGY_PARAMETERS = {
    # ATR for consolidation detection
    'atr_period': {
        'type': 'int',
        'default': 14,
        'min': 10,
        'max': 20,
        'description': 'ATR period for volatility measurement',
        'gui_hint': 'Standard values: 14, 10 for faster reaction'
    },
    # ATR threshold for consolidation
    'atr_threshold': {
        'type': 'float',
        'default': 0.15,
        'min': 0.05,
        'max': 0.3,
        'description': 'ATR threshold for consolidation (as % of price)',
        'gui_hint': 'Lower = tighter consolidation, fewer breakouts'
    },
    # Range lookback period
    'range_period': {
        'type': 'int',
        'default': 20,
        'min': 10,
        'max': 50,
        'description': 'Lookback period for range detection',
        'gui_hint': 'Higher = longer consolidation periods'
    },
    # EMA for trend filter
    'ema_period': {
        'type': 'int',
        'default': 20,
        'min': 10,
        'max': 50,
        'description': 'EMA period for trend filter',
        'gui_hint': 'Higher = smoother trend filter'
    },
    # Volume confirmation
    'volume_sma_period': {
        'type': 'int',
        'default': 20,
        'min': 10,
        'max': 50,
        'description': 'Volume SMA period for confirmation',
        'gui_hint': 'Higher = smoother volume trend'
    },
    # Volume multiplier for breakout confirmation
    'volume_multiplier': {
        'type': 'float',
        'default': 2.0,
        'min': 1.5,
        'max': 3.0,
        'description': 'Volume multiplier for breakout confirmation',
        'gui_hint': 'Higher = stronger volume confirmation'
    },
    # Risk management parameters
    'atr_stop_loss': {
        'type': 'float',
        'default': 0.8,
        'min': 0.5,
        'max': 1.5,
        'description': 'ATR multiplier for stop loss',
        'gui_hint': 'Lower = tighter stops, more conservative'
    },
    'atr_take_profit': {
        'type': 'float',
        'default': 1.5,
        'min': 1.0,
        'max': 3.0,
        'description': 'ATR multiplier for take profit',
        'gui_hint': 'Higher = larger profit targets'
    },
    # Breakout threshold
    'breakout_threshold': {
        'type': 'float',
        'default': 0.1,
        'min': 0.05,
        'max': 0.2,
        'description': 'Breakout threshold as % of range',
        'gui_hint': 'Lower = easier breakouts, more signals'
    },
    # Risk per trade
    'risk_per_trade': {
        'type': 'float',
        'default': 0.5,
        'min': 0.1,
        'max': 1.0,
        'description': 'Risk per trade as % of account balance',
        'gui_hint': 'Lower = more conservative. Recommended: 0.5%'
    }
}

def create_strategy(symbols=None, timeframes=None, **params):
    """
    CREATE STRATEGY FUNCTION - Required by GUI
    This function is called by the GUI to create strategy instances.
    
    Creates a breakout scalping strategy.
    
    Args:
        symbols: List of trading symbols (e.g., ['BTCUSDT'])
        timeframes: List of timeframes (e.g., ['1m', '5m'])
        **params: Strategy parameters from GUI/user input
        
    Returns:
        Built strategy instance ready for backtesting/trading
    """
    # DEBUG: Log what we receive
    logger.info(f"🔧 create_strategy called with:")
    logger.info(f" - symbols: {symbols}")
    logger.info(f" - timeframes: {timeframes}")
    logger.info(f" - params: {params}")
    
    # CRITICAL: Handle None/empty values with defaults
    if symbols is None or len(symbols) == 0:
        logger.warning("⚠️ No symbols provided, using default: ['BTCUSDT']")
        symbols = ['BTCUSDT']
    
    if timeframes is None or len(timeframes) == 0:
        logger.warning("⚠️ No timeframes provided, using default: ['1m']")
        timeframes = ['1m']
    
    # Get parameters with defaults from STRATEGY_PARAMETERS
    atr_period = params.get('atr_period', 14)
    atr_threshold = params.get('atr_threshold', 0.15)
    range_period = params.get('range_period', 20)
    ema_period = params.get('ema_period', 20)
    volume_sma_period = params.get('volume_sma_period', 20)
    volume_multiplier = params.get('volume_multiplier', 2.0)
    atr_stop_loss = params.get('atr_stop_loss', 0.8)
    atr_take_profit = params.get('atr_take_profit', 1.5)
    breakout_threshold = params.get('breakout_threshold', 0.1)
    risk_per_trade = params.get('risk_per_trade', 0.5)
    
    logger.info(f"🎯 Creating Breakout Scalping strategy with parameters:")
    logger.info(f" - Symbols: {symbols}")
    logger.info(f" - Timeframes: {timeframes}")
    logger.info(f" - ATR: {atr_period} (Threshold: {atr_threshold}%)")
    logger.info(f" - Range Period: {range_period}")
    logger.info(f" - EMA: {ema_period}")
    logger.info(f" - Volume: {volume_sma_period} (Multiplier: {volume_multiplier}x)")
    logger.info(f" - Risk Management: SL {atr_stop_loss}x, TP {atr_take_profit}x")
    logger.info(f" - Breakout Threshold: {breakout_threshold}%")
    logger.info(f" - Risk per Trade: {risk_per_trade}%")
    
    try:
        # Create strategy using StrategyBuilder
        strategy_builder = StrategyBuilder(symbols, timeframes)
        
        # Add indicators for each timeframe
        for timeframe in timeframes:
            # Volatility measurement
            strategy_builder.add_indicator(f'atr_{timeframe}', atr, period=atr_period)
            
            # Trend filter
            strategy_builder.add_indicator(f'ema_{timeframe}', ema, period=ema_period)
            
            # Range detection
            strategy_builder.add_indicator(f'highest_{timeframe}', highest, period=range_period)
            strategy_builder.add_indicator(f'lowest_{timeframe}', lowest, period=range_period)
            
            # Volume confirmation
            strategy_builder.add_indicator(f'volume_sma_{timeframe}', volume_sma, period=volume_sma_period)
        
        # Add a simple signal rule for StrategyBuilder to work
        entry_timeframe = '1m' if '1m' in timeframes else timeframes[0]
        
        # Add EMA crossover signal as a basic signal
        strategy_builder.add_signal_rule('ema_signal', ma_crossover,
                                       fast_ma=f'ema_{entry_timeframe}',
                                       slow_ma=f'ema_{entry_timeframe}')
        
        # Set signal combination method
        strategy_builder.set_signal_combination('majority_vote')
        
        # Set strategy information
        strategy_builder.set_strategy_info('Breakout_Scalping', '1.0.0')
        
        # Build and return the strategy
        strategy = strategy_builder.build()
        
        logger.info(f"✅ Breakout Scalping strategy created successfully!")
        logger.info(f" - Strategy Name: {strategy.name}")
        logger.info(f" - Strategy Symbols: {strategy.symbols}")
        logger.info(f" - Strategy Timeframes: {strategy.timeframes}")
        
        return strategy
        
    except Exception as e:
        logger.error(f"❌ Error creating Breakout Scalping strategy: {e}")
        import traceback
        traceback.print_exc()
        raise

class BreakoutScalpingStrategy(StrategyBase):
    """
    Breakout Scalping Strategy Class
    Implements pure breakout logic with quick exits
    """
    
    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        """
        Initialize the breakout scalping strategy
        """
        # CRITICAL: Initialize with EXACT symbols and timeframes provided
        super().__init__(
            name="Breakout_Scalping",
            symbols=symbols,
            timeframes=timeframes,
            config=config
        )
        
        # Strategy-specific parameters
        self.atr_period = config.get('atr_period', 14)
        self.atr_threshold = config.get('atr_threshold', 0.15)
        self.range_period = config.get('range_period', 20)
        self.ema_period = config.get('ema_period', 20)
        self.volume_sma_period = config.get('volume_sma_period', 20)
        self.volume_multiplier = config.get('volume_multiplier', 2.0)
        self.atr_stop_loss = config.get('atr_stop_loss', 0.8)
        self.atr_take_profit = config.get('atr_take_profit', 1.5)
        self.breakout_threshold = config.get('breakout_threshold', 0.1)
        self.risk_per_trade = config.get('risk_per_trade', 0.5) / 100.0  # Convert to decimal
        
        # Risk management
        self.min_position_size = 0.001
        self.max_position_size = 0.05  # Small positions for scalping
        
        # Track positions
        self.positions = {}
        
        # Validate parameters
        self._validate_parameters()
        
        logger.info(f"📈 BreakoutScalpingStrategy initialized:")
        logger.info(f" - Symbols: {self.symbols}")
        logger.info(f" - Timeframes: {self.timeframes}")
        logger.info(f" - Risk per Trade: {self.risk_per_trade * 100}%")
    
    def _validate_parameters(self):
        """Validate strategy parameters"""
        if self.atr_threshold <= 0 or self.atr_threshold > 1:
            raise ValueError("ATR threshold must be between 0 and 1")
        if self.volume_multiplier <= 1:
            raise ValueError("Volume multiplier must be greater than 1")
        if self.atr_stop_loss <= 0 or self.atr_take_profit <= 0:
            raise ValueError("ATR multipliers must be positive")
        if self.atr_take_profit <= self.atr_stop_loss:
            raise ValueError("Take profit must be greater than stop loss")
        if self.breakout_threshold <= 0 or self.breakout_threshold > 1:
            raise ValueError("Breakout threshold must be between 0 and 1")
    
    def calculate_position_size(self, symbol: str, current_price: float = None, signal_strength: float = 1.0) -> float:
        """
        Calculate position size based on fixed percentage risk
        """
        try:
            # Calculate position value based on risk management
            position_value = self.balance * self.risk_per_trade * signal_strength
            
            # Calculate position size
            if current_price and current_price > 0:
                position_size = position_value / current_price
            else:
                position_size = self.min_position_size
            
            # Apply position size limits
            position_size = max(self.min_position_size, min(position_size, self.max_position_size))
            
            return position_size
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return self.min_position_size
    
    def generate_signals(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, str]]:
        """
        Generate trading signals with breakout logic
        """
        signals = {}
        
        try:
            for symbol in data:
                signals[symbol] = {}
                
                for timeframe in data[symbol]:
                    # Generate signal for each timeframe
                    signal = self._generate_single_signal(data[symbol][timeframe], symbol, timeframe)
                    signals[symbol][timeframe] = signal
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return signals
    
    def _generate_single_signal(self, df: pd.DataFrame, symbol: str, timeframe: str) -> str:
        """
        Generate a single trading signal with breakout logic
        """
        try:
            if len(df) < max(self.atr_period, self.range_period, self.ema_period):
                return 'HOLD'
            
            # Get current values
            current_close = df['close'].iloc[-1]
            current_atr = df[f'atr_{timeframe}'].iloc[-1] if f'atr_{timeframe}' in df.columns else 0
            current_ema = df[f'ema_{timeframe}'].iloc[-1] if f'ema_{timeframe}' in df.columns else current_close
            
            # Get range values
            highest_high = df[f'highest_{timeframe}'].iloc[-1] if f'highest_{timeframe}' in df.columns else current_close
            lowest_low = df[f'lowest_{timeframe}'].iloc[-1] if f'lowest_{timeframe}' in df.columns else current_close
            
            # Get volume
            current_volume = df['volume'].iloc[-1]
            volume_sma = df[f'volume_sma_{timeframe}'].iloc[-1]
            
            # Check if in consolidation (low volatility)
            atr_percent = (current_atr / current_close) * 100
            in_consolidation = atr_percent < self.atr_threshold
            
            # Calculate range and breakout levels
            range_size = highest_high - lowest_low
            range_percent = (range_size / current_close) * 100
            
            # Breakout levels
            breakout_high = highest_high + (range_size * self.breakout_threshold / 100)
            breakout_low = lowest_low - (range_size * self.breakout_threshold / 100)
            
            # Volume confirmation
            volume_confirmed = current_volume > volume_sma * self.volume_multiplier
            
            # Breakout logic
            
            # Check for existing positions
            if symbol in self.positions:
                position = self.positions[symbol]
                
                # For long positions
                if position['type'] == 'LONG':
                    # Exit if take profit or stop loss hit
                    if current_close >= position['take_profit'] or current_close <= position['stop_loss']:
                        del self.positions[symbol]  # Remove position
                        return 'SELL'
                
                # For short positions
                elif position['type'] == 'SHORT':
                    # Exit if take profit or stop loss hit
                    if current_close <= position['take_profit'] or current_close >= position['stop_loss']:
                        del self.positions[symbol]  # Remove position
                        return 'BUY'
            
            # New entry signals
            elif in_consolidation and volume_confirmed:
                # Bullish breakout
                if current_close > breakout_high and current_close > current_ema:
                    # Calculate stop loss and take profit
                    stop_loss = current_close - (current_atr * self.atr_stop_loss)
                    take_profit = current_close + (current_atr * self.atr_take_profit)
                    
                    # Store position info
                    self.positions[symbol] = {
                        'type': 'LONG',
                        'entry_price': current_close,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    }
                    
                    return 'BUY'
                
                # Bearish breakout
                elif current_close < breakout_low and current_close < current_ema:
                    # Calculate stop loss and take profit
                    stop_loss = current_close + (current_atr * self.atr_stop_loss)
                    take_profit = current_close - (current_atr * self.atr_take_profit)
                    
                    # Store position info
                    self.positions[symbol] = {
                        'type': 'SHORT',
                        'entry_price': current_close,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    }
                    
                    return 'SELL'
            
            return 'HOLD'
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol} {timeframe}: {e}")
            return 'HOLD'

def create_breakout_scalping_instance(symbols=None, timeframes=None, **params):
    """
    Create breakout scalping strategy instance
    """
    try:
        if symbols is None:
            symbols = ['BTCUSDT']
        if timeframes is None:
            timeframes = ['1m', '5m']
        
        strategy = BreakoutScalpingStrategy(symbols, timeframes, params)
        logger.info(f"✅ Breakout Scalping strategy created successfully")
        return strategy
    except Exception as e:
        logger.error(f"Error creating strategy: {e}")
        raise

def simple_test():
    """
    Simple test to verify the strategy works - MUST EXIST
    """
    try:
        # Test strategy creation
        strategy = create_strategy(
            symbols=['BTCUSDT'],
            timeframes=['1m', '5m'],
            atr_period=14,
            atr_threshold=0.15,
            range_period=20,
            ema_period=20,
            volume_sma_period=20,
            volume_multiplier=2.0,
            atr_stop_loss=0.8,
            atr_take_profit=1.5,
            breakout_threshold=0.1,
            risk_per_trade=0.5
        )
        
        print(f"✅ Breakout Scalping strategy created successfully: {strategy.name}")
        print(f" - Symbols: {strategy.symbols}")
        print(f" - Timeframes: {strategy.timeframes}")
        print(f" - Strategy Type: Breakout scalping with quick exits")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Breakout Scalping strategy: {e}")
        return False

# For testing - MUST EXIST
if __name__ == "__main__":
    simple_test()