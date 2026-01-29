"""
Mean Reversion Scalping Strategy
================================

A scalping strategy that capitalizes on price deviations from the mean:
- Uses RSI and Bollinger Bands to identify overbought/oversold conditions
- Quick entries when price extremes are reached
- Fast exits with tight stops
- No trend following - pure mean reversion

Strategy Logic:
1. OVERBOUGHT: RSI > 70 AND price > upper Bollinger Band = SELL
2. OVERSOLD: RSI < 30 AND price < lower Bollinger Band = BUY
3. EXIT: When price returns to middle Bollinger Band or RSI crosses 50

Best for: Ranging markets with clear support/resistance levels
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
from simple_strategy.strategies.indicators_library import rsi, bollinger_bands, atr, volume_sma
from simple_strategy.strategies.signals_library import overbought_oversold, bollinger_bands_signals
from simple_strategy.shared.strategy_base import StrategyBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CRITICAL: STRATEGY_PARAMETERS for GUI Configuration
STRATEGY_PARAMETERS = {
    # RSI for mean reversion signals
    'rsi_period': {
        'type': 'int',
        'default': 14,
        'min': 7,
        'max': 21,
        'description': 'RSI period for mean reversion signals',
        'gui_hint': 'Standard values: 14, 10 for faster signals'
    },
    # RSI levels - more extreme for better mean reversion
    'rsi_overbought': {
        'type': 'int',
        'default': 70,
        'min': 65,
        'max': 80,
        'description': 'RSI overbought level for sell signals',
        'gui_hint': 'Higher = more extreme overbought, fewer signals'
    },
    'rsi_oversold': {
        'type': 'int',
        'default': 30,
        'min': 20,
        'max': 35,
        'description': 'RSI oversold level for buy signals',
        'gui_hint': 'Lower = more extreme oversold, fewer signals'
    },
    # Bollinger Bands for volatility bands
    'bb_period': {
        'type': 'int',
        'default': 20,
        'min': 15,
        'max': 30,
        'description': 'Bollinger Bands period for mean reversion',
        'gui_hint': 'Standard values: 20, 15 for faster signals'
    },
    'bb_std_dev': {
        'type': 'float',
        'default': 2.0,
        'min': 1.8,
        'max': 2.5,
        'description': 'Bollinger Bands standard deviation',
        'gui_hint': 'Higher = wider bands, more extreme signals'
    },
    # ATR for risk management
    'atr_period': {
        'type': 'int',
        'default': 14,
        'min': 10,
        'max': 20,
        'description': 'ATR period for volatility measurement',
        'gui_hint': 'Standard values: 14, 10 for faster reaction'
    },
    # Risk management parameters - tight stops for scalping
    'atr_stop_loss': {
        'type': 'float',
        'default': 0.5,
        'min': 0.3,
        'max': 1.0,
        'description': 'ATR multiplier for stop loss (tight for scalping)',
        'gui_hint': 'Lower = tighter stops, more conservative'
    },
    'atr_take_profit': {
        'type': 'float',
        'default': 1.0,
        'min': 0.5,
        'max': 2.0,
        'description': 'ATR multiplier for take profit (quick exits)',
        'gui_hint': 'Higher = larger profit targets'
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
    # RSI exit level
    'rsi_exit_level': {
        'type': 'int',
        'default': 50,
        'min': 45,
        'max': 55,
        'description': 'RSI level for exit (mean reversion target)',
        'gui_hint': 'Standard is 50 (middle of RSI range)'
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
    
    Creates a mean reversion scalping strategy.
    
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
    rsi_period = params.get('rsi_period', 14)
    rsi_overbought = params.get('rsi_overbought', 70)
    rsi_oversold = params.get('rsi_oversold', 30)
    bb_period = params.get('bb_period', 20)
    bb_std_dev = params.get('bb_std_dev', 2.0)
    atr_period = params.get('atr_period', 14)
    atr_stop_loss = params.get('atr_stop_loss', 0.5)
    atr_take_profit = params.get('atr_take_profit', 1.0)
    volume_sma_period = params.get('volume_sma_period', 20)
    rsi_exit_level = params.get('rsi_exit_level', 50)
    risk_per_trade = params.get('risk_per_trade', 0.5)
    
    logger.info(f"🎯 Creating Mean Reversion Scalping strategy with parameters:")
    logger.info(f" - Symbols: {symbols}")
    logger.info(f" - Timeframes: {timeframes}")
    logger.info(f" - RSI: {rsi_period} (OB: {rsi_overbought}, OS: {rsi_oversold}, Exit: {rsi_exit_level})")
    logger.info(f" - Bollinger Bands: {bb_period}, {bb_std_dev} std")
    logger.info(f" - ATR: {atr_period} (SL: {atr_stop_loss}x, TP: {atr_take_profit}x)")
    logger.info(f" - Volume SMA: {volume_sma_period}")
    logger.info(f" - Risk per Trade: {risk_per_trade}%")
    
    try:
        # Create strategy using StrategyBuilder
        strategy_builder = StrategyBuilder(symbols, timeframes)
        
        # Add indicators for each timeframe
        for timeframe in timeframes:
            # Momentum indicator
            strategy_builder.add_indicator(f'rsi_{timeframe}', rsi, period=rsi_period)
            
            # Volatility bands
            strategy_builder.add_indicator(f'bb_{timeframe}', bollinger_bands, period=bb_period, std_dev=bb_std_dev)
            
            # Volatility measurement
            strategy_builder.add_indicator(f'atr_{timeframe}', atr, period=atr_period)
            
            # Volume indicator
            strategy_builder.add_indicator(f'volume_sma_{timeframe}', volume_sma, period=volume_sma_period)
        
        # Add signal rules for entry timeframe (1m)
        entry_timeframe = '1m' if '1m' in timeframes else timeframes[0]
        
        # 1. RSI Overbought/Oversold Signal
        strategy_builder.add_signal_rule('rsi_signal', overbought_oversold,
                                       indicator=f'rsi_{entry_timeframe}',
                                       overbought=rsi_overbought,
                                       oversold=rsi_oversold)
        
        # 2. Bollinger Bands Breakout Signal
        strategy_builder.add_signal_rule('bb_signal', bollinger_bands_signals,
                                       price='close',
                                       upper_band=f'bb_{entry_timeframe}',
                                       lower_band=f'bb_{entry_timeframe}',
                                       middle_band=f'bb_{entry_timeframe}')
        
        # Set signal combination method - require both signals to agree
        strategy_builder.set_signal_combination('unanimous')
        
        # Set strategy information
        strategy_builder.set_strategy_info('Mean_Reversion_Scalping', '1.0.0')
        
        # Build and return the strategy
        strategy = strategy_builder.build()
        
        logger.info(f"✅ Mean Reversion Scalping strategy created successfully!")
        logger.info(f" - Strategy Name: {strategy.name}")
        logger.info(f" - Strategy Symbols: {strategy.symbols}")
        logger.info(f" - Strategy Timeframes: {strategy.timeframes}")
        
        return strategy
        
    except Exception as e:
        logger.error(f"❌ Error creating Mean Reversion Scalping strategy: {e}")
        import traceback
        traceback.print_exc()
        raise

class MeanReversionScalpingStrategy(StrategyBase):
    """
    Mean Reversion Scalping Strategy Class
    Implements pure mean reversion logic with quick exits
    """
    
    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        """
        Initialize the mean reversion scalping strategy
        """
        # CRITICAL: Initialize with EXACT symbols and timeframes provided
        super().__init__(
            name="Mean_Reversion_Scalping",
            symbols=symbols,
            timeframes=timeframes,
            config=config
        )
        
        # Strategy-specific parameters
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_exit_level = config.get('rsi_exit_level', 50)
        self.bb_period = config.get('bb_period', 20)
        self.bb_std_dev = config.get('bb_std_dev', 2.0)
        self.atr_period = config.get('atr_period', 14)
        self.atr_stop_loss = config.get('atr_stop_loss', 0.5)
        self.atr_take_profit = config.get('atr_take_profit', 1.0)
        self.volume_sma_period = config.get('volume_sma_period', 20)
        self.risk_per_trade = config.get('risk_per_trade', 0.5) / 100.0  # Convert to decimal
        
        # Risk management
        self.min_position_size = 0.001
        self.max_position_size = 0.05  # Small positions for scalping
        
        # Track positions
        self.positions = {}
        
        # Validate parameters
        self._validate_parameters()
        
        logger.info(f"📈 MeanReversionScalpingStrategy initialized:")
        logger.info(f" - Symbols: {self.symbols}")
        logger.info(f" - Timeframes: {self.timeframes}")
        logger.info(f" - Risk per Trade: {self.risk_per_trade * 100}%")
    
    def _validate_parameters(self):
        """Validate strategy parameters"""
        if self.rsi_oversold >= self.rsi_overbought:
            raise ValueError("RSI oversold level must be less than overbought level")
        if self.rsi_exit_level <= self.rsi_oversold or self.rsi_exit_level >= self.rsi_overbought:
            raise ValueError("RSI exit level must be between oversold and overbought levels")
        if self.atr_stop_loss <= 0 or self.atr_take_profit <= 0:
            raise ValueError("ATR multipliers must be positive")
        if self.atr_take_profit <= self.atr_stop_loss:
            raise ValueError("Take profit must be greater than stop loss")
    
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
        Generate trading signals with mean reversion logic
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
        Generate a single trading signal with mean reversion logic
        """
        try:
            if len(df) < max(self.rsi_period, self.bb_period, self.atr_period):
                return 'HOLD'
            
            # Get current values
            current_close = df['close'].iloc[-1]
            current_rsi = df[f'rsi_{timeframe}'].iloc[-1] if f'rsi_{timeframe}' in df.columns else 50
            current_atr = df[f'atr_{timeframe}'].iloc[-1] if f'atr_{timeframe}' in df.columns else 0
            
            # Get Bollinger Bands
            bb_data = df[f'bb_{timeframe}'].iloc[-1]
            if isinstance(bb_data, (tuple, list, np.ndarray)) and len(bb_data) >= 3:
                bb_upper = bb_data[0]  # Upper band
                bb_middle = bb_data[1]  # Middle band (SMA)
                bb_lower = bb_data[2]  # Lower band
            else:
                # Fallback if bb_data is not in expected format
                bb_upper = current_close * 1.02
                bb_middle = current_close
                bb_lower = current_close * 0.98
            
            # Get volume
            current_volume = df['volume'].iloc[-1]
            volume_sma = df[f'volume_sma_{timeframe}'].iloc[-1]
            
            # Volume confirmation (less strict for mean reversion)
            volume_confirmed = current_volume > volume_sma * 0.8  # Lower threshold for mean reversion
            
            # Mean reversion logic - more sensitive conditions
            
            # OVERBOUGHT: RSI > overbought AND price > upper BB = SELL
            if current_rsi > self.rsi_overbought and current_close > bb_upper and volume_confirmed:
                return 'SELL'
            
            # OVERSOLD: RSI < oversold AND price < lower BB = BUY
            elif current_rsi < self.rsi_oversold and current_close < bb_lower and volume_confirmed:
                return 'BUY'
            
            # EXIT: When price returns to mean (middle BB) or RSI crosses 50
            elif symbol in self.positions:
                position = self.positions[symbol]
                
                # For long positions
                if position['type'] == 'LONG':
                    # Exit if RSI crosses above exit level or price reaches middle BB
                    if current_rsi > self.rsi_exit_level or current_close >= bb_middle:
                        del self.positions[symbol]  # Remove position
                        return 'SELL'
                
                # For short positions
                elif position['type'] == 'SHORT':
                    # Exit if RSI crosses below exit level or price reaches middle BB
                    if current_rsi < self.rsi_exit_level or current_close <= bb_middle:
                        del self.positions[symbol]  # Remove position
                        return 'BUY'
            
            return 'HOLD'
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol} {timeframe}: {e}")
            return 'HOLD'

def create_mean_reversion_scalping_instance(symbols=None, timeframes=None, **params):
    """
    Create mean reversion scalping strategy instance
    """
    try:
        if symbols is None:
            symbols = ['BTCUSDT']
        if timeframes is None:
            timeframes = ['1m', '5m']
        
        strategy = MeanReversionScalpingStrategy(symbols, timeframes, params)
        logger.info(f"✅ Mean Reversion Scalping strategy created successfully")
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
            rsi_period=14,
            rsi_overbought=70,
            rsi_oversold=30,
            bb_period=20,
            bb_std_dev=2.0,
            atr_period=14,
            atr_stop_loss=0.5,
            atr_take_profit=1.0,
            volume_sma_period=20,
            rsi_exit_level=50,
            risk_per_trade=0.5
        )
        
        print(f"✅ Mean Reversion Scalping strategy created successfully: {strategy.name}")
        print(f" - Symbols: {strategy.symbols}")
        print(f" - Timeframes: {strategy.timeframes}")
        print(f" - Strategy Type: Mean reversion scalping with quick exits")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Mean Reversion Scalping strategy: {e}")
        return False

# For testing - MUST EXIST
if __name__ == "__main__":
    simple_test()