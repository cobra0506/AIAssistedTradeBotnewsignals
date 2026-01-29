"""
Scalping Multi-Indicator Strategy - High-Frequency Trading Strategy
====================================================================
Designed for scalping with quick entries/exits and high gain potential.
Combines multiple indicators for high-probability trading signals.

Strategy Features:
- Fast timeframes (1m, 3m, 5m) for scalping
- Multiple indicator confirmations (RSI, EMA, MACD)
- Weighted signal combination for accuracy
- Tight risk management for scalping
- Volatility-based position sizing
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
from simple_strategy.strategies.indicators_library import rsi, ema, macd, atr
from simple_strategy.strategies.signals_library import overbought_oversold, ma_crossover, macd_signals
from simple_strategy.shared.strategy_base import StrategyBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CRITICAL: STRATEGY_PARAMETERS for GUI Configuration
# This dictionary defines what parameters the GUI will show and allow users to configure
STRATEGY_PARAMETERS = {
    # RSI Parameters
    'rsi_period': {
        'type': 'int',
        'default': 9,
        'min': 5,
        'max': 21,
        'description': 'RSI calculation period (shorter for scalping)',
        'gui_hint': 'Scalping: Use 5-9 for faster signals'
    },
    'rsi_overbought': {
        'type': 'int',
        'default': 75,
        'min': 65,
        'max': 85,
        'description': 'RSI overbought level (sell signal)',
        'gui_hint': 'Higher = more conservative sells (75-80 recommended)'
    },
    'rsi_oversold': {
        'type': 'int',
        'default': 25,
        'min': 15,
        'max': 35,
        'description': 'RSI oversold level (buy signal)',
        'gui_hint': 'Lower = more conservative buys (20-25 recommended)'
    },
    
    # EMA Parameters
    'ema_fast': {
        'type': 'int',
        'default': 5,
        'min': 3,
        'max': 10,
        'description': 'Fast EMA period for trend direction',
        'gui_hint': 'Scalping: Use 3-5 for quick trend changes'
    },
    'ema_slow': {
        'type': 'int',
        'default': 13,
        'min': 8,
        'max': 21,
        'description': 'Slow EMA period for trend confirmation',
        'gui_hint': 'Scalping: Use 9-13 for trend confirmation'
    },
    
    # MACD Parameters
    'macd_fast': {
        'type': 'int',
        'default': 5,
        'min': 3,
        'max': 8,
        'description': 'MACD fast EMA period',
        'gui_hint': 'Scalping: Use 3-5 for faster MACD signals'
    },
    'macd_slow': {
        'type': 'int',
        'default': 13,
        'min': 9,
        'max': 21,
        'description': 'MACD slow EMA period',
        'gui_hint': 'Scalping: Use 9-13 for MACD confirmation'
    },
    'macd_signal': {
        'type': 'int',
        'default': 6,
        'min': 3,
        'max': 9,
        'description': 'MACD signal line period',
        'gui_hint': 'Scalping: Use 3-6 for quick MACD signals'
    },
    
    # Risk Management Parameters
    'stop_loss_atr': {
        'type': 'float',
        'default': 1.5,
        'min': 0.5,
        'max': 3.0,
        'description': 'Stop loss as ATR multiplier',
        'gui_hint': 'Scalping: Use 0.5-1.5 for tight stops'
    },
    'take_profit_atr': {
        'type': 'float',
        'default': 2.5,
        'min': 1.0,
        'max': 5.0,
        'description': 'Take profit as ATR multiplier',
        'gui_hint': 'Scalping: Use 1.5-3.0 for quick profits'
    },
    'atr_period': {
        'type': 'int',
        'default': 14,
        'min': 7,
        'max': 21,
        'description': 'ATR calculation period',
        'gui_hint': 'Use 7-14 for current volatility'
    },
    
    # Signal Weights
    'rsi_weight': {
        'type': 'float',
        'default': 0.33,
        'min': 0.1,
        'max': 0.6,
        'description': 'RSI signal weight (0.0-1.0)',
        'gui_hint': 'Higher = more importance to RSI signals'
    },
    'ema_weight': {
        'type': 'float',
        'default': 0.33,
        'min': 0.1,
        'max': 0.6,
        'description': 'EMA crossover signal weight (0.0-1.0)',
        'gui_hint': 'Higher = more importance to trend signals'
    },
    'macd_weight': {
        'type': 'float',
        'default': 0.34,
        'min': 0.1,
        'max': 0.6,
        'description': 'MACD signal weight (0.0-1.0)',
        'gui_hint': 'Higher = more importance to momentum signals'
    }
}

def create_strategy(symbols=None, timeframes=None, **params):
    """
    CREATE STRATEGY FUNCTION - Required by GUI
    Creates a multi-indicator scalping strategy with weighted signal combination.
    
    Strategy Logic:
    - Uses RSI for overbought/oversold conditions
    - Uses EMA crossovers for trend direction
    - Uses MACD for momentum confirmation
    - Combines all signals with weighted voting
    - Implements tight risk management for scalping
    
    Args:
        symbols: List of trading symbols (default: ['BTCUSDT'])
        timeframes: List of timeframes (default: ['1m', '3m', '5m'])
        **params: Strategy parameters from GUI/user input
        
    Returns:
        Built strategy instance ready for backtesting/trading
    """
    # DEBUG: Log what we receive
    logger.info(f"🔧 create_strategy called with:")
    logger.info(f" - symbols: {symbols}")
    logger.info(f" - timeframes: {timeframes}")
    logger.info(f" - params: {params}")
    
    # Handle None/empty values with defaults
    if symbols is None or len(symbols) == 0:
        logger.warning("⚠️ No symbols provided, using default: ['BTCUSDT']")
        symbols = ['BTCUSDT']
    
    if timeframes is None or len(timeframes) == 0:
        logger.warning("⚠️ No timeframes provided, using default: ['1m', '3m', '5m']")
        timeframes = ['1m', '3m', '5m']
    
    # Get parameters with defaults from STRATEGY_PARAMETERS
    rsi_period = params.get('rsi_period', 9)
    rsi_overbought = params.get('rsi_overbought', 75)
    rsi_oversold = params.get('rsi_oversold', 25)
    
    ema_fast = params.get('ema_fast', 5)
    ema_slow = params.get('ema_slow', 13)
    
    macd_fast = params.get('macd_fast', 5)
    macd_slow = params.get('macd_slow', 13)
    macd_signal = params.get('macd_signal', 6)
    
    stop_loss_atr = params.get('stop_loss_atr', 1.5)
    take_profit_atr = params.get('take_profit_atr', 2.5)
    atr_period = params.get('atr_period', 14)
    
    rsi_weight = params.get('rsi_weight', 0.33)
    ema_weight = params.get('ema_weight', 0.33)
    macd_weight = params.get('macd_weight', 0.34)
    
    logger.info(f"🎯 Creating Scalping Multi-Indicator strategy with parameters:")
    logger.info(f" - Symbols: {symbols}")
    logger.info(f" - Timeframes: {timeframes}")
    logger.info(f" - RSI: Period={rsi_period}, OB={rsi_overbought}, OS={rsi_oversold}")
    logger.info(f" - EMA: Fast={ema_fast}, Slow={ema_slow}")
    logger.info(f" - MACD: Fast={macd_fast}, Slow={macd_slow}, Signal={macd_signal}")
    logger.info(f" - Risk: SL={stop_loss_atr}xATR, TP={take_profit_atr}xATR, ATR={atr_period}")
    logger.info(f" - Weights: RSI={rsi_weight}, EMA={ema_weight}, MACD={macd_weight}")
    
    try:
        # Create strategy using StrategyBuilder
        strategy_builder = StrategyBuilder(symbols, timeframes)
        
        # Add indicators
        strategy_builder.add_indicator('rsi', rsi, period=rsi_period)
        strategy_builder.add_indicator('ema_fast', ema, period=ema_fast)
        strategy_builder.add_indicator('ema_slow', ema, period=ema_slow)
        strategy_builder.add_indicator('macd', macd, fast_period=macd_fast, slow_period=macd_slow, signal_period=macd_signal)
        strategy_builder.add_indicator('atr', atr, period=atr_period)
        
        # Add signal rules - CORRECTED VERSION
        strategy_builder.add_signal_rule('rsi_signal', overbought_oversold,
                                      indicator='rsi',
                                      overbought=rsi_overbought,
                                      oversold=rsi_oversold)
        
        strategy_builder.add_signal_rule('ema_signal', ma_crossover,
                                      fast_ma='ema_fast',
                                      slow_ma='ema_slow')
        
        # CORRECTED MACD signal rule - use indicator name, not component names
        strategy_builder.add_signal_rule('macd_signal', macd_signals,
                                      macd_line='macd',
                                      signal_line='macd')
        
        # Set weighted signal combination - FIXED
        strategy_builder.set_signal_combination('weighted',
                                              weights={
                                                  'rsi_signal': rsi_weight,
                                                  'ema_signal': ema_weight,
                                                  'macd_signal': macd_weight
                                              })
        
        # Add risk management rules
        strategy_builder.add_risk_rule('stop_loss', atr_multiplier=stop_loss_atr)
        strategy_builder.add_risk_rule('take_profit', atr_multiplier=take_profit_atr)
        
        # Set strategy information
        strategy_builder.set_strategy_info('Scalping_MultiIndicator', '1.0.0')
        
        # Build and return the strategy
        strategy = strategy_builder.build()
        
        logger.info(f"✅ Scalping Multi-Indicator strategy created successfully!")
        logger.info(f" - Strategy Name: {strategy.name}")
        logger.info(f" - Strategy Symbols: {strategy.symbols}")
        logger.info(f" - Strategy Timeframes: {strategy.timeframes}")
        
        return strategy
        
    except Exception as e:
        logger.error(f"❌ Error creating Scalping Multi-Indicator strategy: {e}")
        import traceback
        traceback.print_exc()
        raise

class ScalpingMultiIndicatorStrategy(StrategyBase):
    """
    Scalping Multi-Indicator Strategy Class
    Advanced scalping strategy with multiple indicator confirmations
    """
    
    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        """
        Initialize the scalping strategy
        """
        # CRITICAL: Initialize with EXACT symbols and timeframes provided
        super().__init__(
            name="Scalping_MultiIndicator",
            symbols=symbols,
            timeframes=timeframes,
            config=config
        )
        
        # Strategy-specific parameters
        self.rsi_period = config.get('rsi_period', 9)
        self.rsi_overbought = config.get('rsi_overbought', 75)
        self.rsi_oversold = config.get('rsi_oversold', 25)
        
        self.ema_fast = config.get('ema_fast', 5)
        self.ema_slow = config.get('ema_slow', 13)
        
        self.macd_fast = config.get('macd_fast', 5)
        self.macd_slow = config.get('macd_slow', 13)
        self.macd_signal = config.get('macd_signal', 6)
        
        # Risk management
        self.stop_loss_atr = config.get('stop_loss_atr', 1.5)
        self.take_profit_atr = config.get('take_profit_atr', 2.5)
        self.atr_period = config.get('atr_period', 14)
        
        # Signal weights
        self.rsi_weight = config.get('rsi_weight', 0.33)
        self.ema_weight = config.get('ema_weight', 0.33)
        self.macd_weight = config.get('macd_weight', 0.34)
        
        # Scalping-specific settings
        self.max_risk_per_trade = 0.01  # 1% risk per trade for scalping
        self.min_position_size = 0.001
        self.max_position_size = 1.0
        
        # Validate parameters
        self._validate_parameters()
        
        logger.info(f"📈 ScalpingMultiIndicatorStrategy initialized:")
        logger.info(f" - Symbols: {self.symbols}")
        logger.info(f" - Timeframes: {self.timeframes}")
        logger.info(f" - Max Risk per Trade: {self.max_risk_per_trade * 100}%")
        
    def _validate_parameters(self):
        """Validate strategy parameters"""
        # Validate RSI parameters
        if not (5 <= self.rsi_period <= 21):
            raise ValueError("RSI period must be between 5 and 21")
        if not (self.rsi_oversold < self.rsi_overbought):
            raise ValueError("RSI oversold level must be less than overbought level")
            
        # Validate EMA parameters
        if not (self.ema_fast < self.ema_slow):
            raise ValueError("Fast EMA period must be less than slow EMA period")
            
        # Validate MACD parameters
        if not (self.macd_fast < self.macd_slow):
            raise ValueError("MACD fast period must be less than slow period")
            
        # Validate risk parameters
        if not (0 < self.stop_loss_atr < self.take_profit_atr):
            raise ValueError("Stop loss ATR must be less than take profit ATR")
            
        # Validate weights
        total_weight = self.rsi_weight + self.ema_weight + self.macd_weight
        if not abs(total_weight - 1.0) < 0.01:
            raise ValueError("Signal weights must sum to 1.0")
    
    def calculate_position_size(self, symbol: str, current_price: float = None, signal_strength: float = 1.0) -> float:
        """
        Calculate position size - MUST EXIST
        ATR-based position sizing for scalping
        """
        try:
            # Calculate position value based on risk management
            position_value = self.balance * self.max_risk_per_trade * signal_strength
            
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
        Generate trading signals - MUST EXIST
        Multi-indicator signal generation with weighted voting
        """
        signals = {}
        
        try:
            for symbol in data:
                signals[symbol] = {}
                
                for timeframe in data[symbol]:
                    # Generate signal for each timeframe
                    signal = self._generate_single_signal(data[symbol][timeframe], symbol, timeframe)
                    signals[symbol][timeframe] = signal
                    
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            
        return signals
    
    def _generate_single_signal(self, df: pd.DataFrame, symbol: str, timeframe: str) -> str:
        """
        Generate a single trading signal - MUST EXIST
        Advanced multi-indicator signal generation
        """
        try:
            if len(df) < 30:  # Need enough data for all indicators
                return 'HOLD'
            
            # Calculate indicators
            close_prices = df['close']
            
            # RSI
            rsi_values = rsi(close_prices, self.rsi_period)
            current_rsi = rsi_values.iloc[-1]
            
            # EMAs
            ema_fast_values = ema(close_prices, self.ema_fast)
            ema_slow_values = ema(close_prices, self.ema_slow)
            
            # MACD
            macd_line, macd_signal_line, macd_histogram = macd(close_prices, self.macd_fast, self.macd_slow, self.macd_signal)
            
            # Generate individual signals
            rsi_signal = 0
            if current_rsi < self.rsi_oversold:
                rsi_signal = 1  # BUY
            elif current_rsi > self.rsi_overbought:
                rsi_signal = -1  # SELL
            
            ema_signal = 0
            if ema_fast_values.iloc[-1] > ema_slow_values.iloc[-1] and ema_fast_values.iloc[-2] <= ema_slow_values.iloc[-2]:
                ema_signal = 1  # BUY crossover
            elif ema_fast_values.iloc[-1] < ema_slow_values.iloc[-1] and ema_fast_values.iloc[-2] >= ema_slow_values.iloc[-2]:
                ema_signal = -1  # SELL crossover
            
            macd_signal = 0
            if macd_line.iloc[-1] > macd_signal_line.iloc[-1] and macd_line.iloc[-2] <= macd_signal_line.iloc[-2]:
                macd_signal = 1  # BUY crossover
            elif macd_line.iloc[-1] < macd_signal_line.iloc[-1] and macd_line.iloc[-2] >= macd_signal_line.iloc[-2]:
                macd_signal = -1  # SELL crossover
            
            # Weighted signal combination
            weighted_signal = (
                rsi_signal * self.rsi_weight +
                ema_signal * self.ema_weight +
                macd_signal * self.macd_weight
            )
            
            # Convert weighted signal to final decision
            if weighted_signal > 0.3:  # Strong buy signal
                return 'BUY'
            elif weighted_signal < -0.3:  # Strong sell signal
                return 'SELL'
            else:
                return 'HOLD'
                
        except Exception as e:
            logger.error(f"Error generating signal for {symbol} {timeframe}: {e}")
            return 'HOLD'

def create_scalping_multi_indicator_instance(symbols=None, timeframes=None, **params):
    """Create scalping multi-indicator strategy instance - OPTIONAL but recommended"""
    try:
        if symbols is None:
            symbols = ['BTCUSDT']
        if timeframes is None:
            timeframes = ['1m', '3m', '5m']
            
        strategy = ScalpingMultiIndicatorStrategy(symbols, timeframes, params)
        logger.info(f"✅ Scalping Multi-Indicator strategy created successfully")
        return strategy
        
    except Exception as e:
        logger.error(f"Error creating strategy: {e}")
        raise

def simple_test():
    """Simple test to verify the strategy works - MUST EXIST"""
    try:
        # Test strategy creation
        strategy = create_strategy(
            symbols=['BTCUSDT'],
            timeframes=['1m', '3m'],
            rsi_period=9,
            rsi_overbought=75,
            rsi_oversold=25,
            ema_fast=5,
            ema_slow=13,
            macd_fast=5,
            macd_slow=13,
            macd_signal=6,
            stop_loss_atr=1.5,
            take_profit_atr=2.5,
            atr_period=14,
            rsi_weight=0.33,
            ema_weight=0.33,
            macd_weight=0.34
        )
        
        print(f"✅ Scalping Multi-Indicator strategy created successfully: {strategy.name}")
        print(f" - Symbols: {strategy.symbols}")
        print(f" - Timeframes: {strategy.timeframes}")
        print(f" - Strategy Type: Multi-Indicator Scalping")
        print(f" - Risk Management: ATR-based stops and targets")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Scalping Multi-Indicator strategy: {e}")
        return False

# For testing - MUST EXIST
if __name__ == "__main__":
    simple_test()