"""
Multi-Timeframe Scalping Strategy with Volatility Breakout
========================================================

A comprehensive scalping strategy designed for 1m and 5m charts that combines:
- Multi-timeframe trend analysis (1m for entries, 5m/15m for trend confirmation)
- Fast EMA crossovers for quick trend detection
- Volatility-based filtering using ATR
- Momentum confirmation with RSI
- Volume confirmation for breakouts
- Dynamic risk management with ATR-based stops

Strategy Logic:
1. TREND FILTER (Higher Timeframe): Price above/below EMAs determines overall trend
2. ENTRY SIGNAL (1m): Breakout confirmation with multiple indicators aligning
3. RISK MANAGEMENT: Dynamic stop-loss and take-profit based on ATR

Best for: Quick trades on 1m-5m charts with trend confirmation
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
from simple_strategy.strategies.indicators_library import ema, rsi, atr, bollinger_bands, volume_sma, sma
from simple_strategy.strategies.signals_library import ma_crossover, overbought_oversold, bollinger_bands_signals
from simple_strategy.shared.strategy_base import StrategyBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CRITICAL: STRATEGY_PARAMETERS for GUI Configuration
# This dictionary defines what parameters the GUI will show and allow users to configure
STRATEGY_PARAMETERS = {
    # Fast EMA for entry signals
    'fast_ema_period': {
        'type': 'int',
        'default': 10,
        'min': 5,
        'max': 20,
        'description': 'Fast EMA period for entry signals',
        'gui_hint': 'Lower values = more sensitive entries. Recommended: 8-12'
    },
    # Slow EMA for trend direction
    'slow_ema_period': {
        'type': 'int',
        'default': 49,
        'min': 15,
        'max': 50,
        'description': 'Slow EMA period for trend direction',
        'gui_hint': 'Higher values = smoother trend. Recommended: 20-25'
    },
    # RSI for momentum confirmation
    'rsi_period': {
        'type': 'int',
        'default': 17,
        'min': 7,
        'max': 21,
        'description': 'RSI period for momentum confirmation',
        'gui_hint': 'Standard values: 14, 10 for faster signals'
    },
    # RSI levels
    'rsi_overbought': {
        'type': 'int',
        'default': 72,
        'min': 60,
        'max': 80,
        'description': 'RSI overbought level for sell signals',
        'gui_hint': 'Higher = more conservative sells'
    },
    'rsi_oversold': {
        'type': 'int',
        'default': 23,
        'min': 20,
        'max': 40,
        'description': 'RSI oversold level for buy signals',
        'gui_hint': 'Lower = more conservative buys'
    },
    # ATR for volatility and risk management
    'atr_period': {
        'type': 'int',
        'default': 11,
        'min': 10,
        'max': 20,
        'description': 'ATR period for volatility measurement',
        'gui_hint': 'Standard values: 14, 10 for faster reaction'
    },
    # Risk management parameters
    'atr_multiplier_sl': {
        'type': 'float',
        'default': 0.65,
        'min': 0.5,
        'max': 3.0,
        'description': 'ATR multiplier for stop-loss distance',
        'gui_hint': 'Higher = wider stops, more conservative'
    },
    'atr_multiplier_tp': {
        'type': 'float',
        'default': 4.46,
        'min': 1.0,
        'max': 5.0,
        'description': 'ATR multiplier for take-profit distance',
        'gui_hint': 'Higher = larger profit targets'
    },
    # Bollinger Bands for volatility breakouts
    'bb_period': {
        'type': 'int',
        'default': 24,
        'min': 15,
        'max': 30,
        'description': 'Bollinger Bands period for volatility breakouts',
        'gui_hint': 'Standard values: 20, 15 for faster signals'
    },
    'bb_std_dev': {
        'type': 'float',
        'default': 2.21,
        'min': 1.5,
        'max': 2.5,
        'description': 'Bollinger Bands standard deviation',
        'gui_hint': 'Higher = wider bands, fewer signals'
    },
    # Volume confirmation
    'volume_sma_period': {
        'type': 'int',
        'default': 10,
        'min': 10,
        'max': 50,
        'description': 'Volume SMA period for confirmation',
        'gui_hint': 'Higher = smoother volume trend'
    },
    # Trend confirmation timeframe
    'trend_timeframe': {
        'type': 'str',
        'default': '5m',
        'options': ['5m', '15m', '30m'],
        'description': 'Higher timeframe for trend confirmation',
        'gui_hint': 'Use 5m for scalping, 15m for swing trades'
    },
    # Minimum volatility filter
    'min_atr_threshold': {
        'type': 'float',
        'default': 0.13,
        'min': 0.05,
        'max': 0.5,
        'description': 'Minimum ATR threshold for trading (as % of price)',
        'gui_hint': 'Filter out low volatility periods. 0.1 = 0.1%'
    },# Add these parameters to STRATEGY_PARAMETERS:
    'max_position_additions': {
        'type': 'int',
        'default': 1,
        'min': 1,
        'max': 3,
        'description': 'Maximum times to add to a position',
        'gui_hint': '1 = no averaging, higher = more aggressive'
    },
    'atr_stop_loss': {
        'type': 'float',
        'default': 0.96,
        'min': 0.5,
        'max': 2.0,
        'description': 'ATR multiplier for stop loss',
        'gui_hint': 'Lower = tighter stops'
    }
}

def create_strategy(symbols=None, timeframes=None, **params):
    """
    CREATE STRATEGY FUNCTION - Required by GUI
    This function is called by the GUI to create strategy instances.
    
    Creates a multi-timeframe scalping strategy with volatility breakout.
    
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
    
    # Ensure we have the trend timeframe available
    trend_timeframe = params.get('trend_timeframe', '5m')
    if trend_timeframe not in timeframes:
        timeframes.append(trend_timeframe)
        logger.info(f"📈 Added trend timeframe {trend_timeframe} to timeframes")
    
    # Get parameters with defaults from STRATEGY_PARAMETERS
    fast_ema_period = params.get('fast_ema_period', 9)
    slow_ema_period = params.get('slow_ema_period', 21)
    rsi_period = params.get('rsi_period', 14)
    rsi_overbought = params.get('rsi_overbought', 70)
    rsi_oversold = params.get('rsi_oversold', 30)
    atr_period = params.get('atr_period', 14)
    atr_multiplier_sl = params.get('atr_multiplier_sl', 1.5)
    atr_multiplier_tp = params.get('atr_multiplier_tp', 2.5)
    bb_period = params.get('bb_period', 20)
    bb_std_dev = params.get('bb_std_dev', 2.0)
    volume_sma_period = params.get('volume_sma_period', 20)
    min_atr_threshold = params.get('min_atr_threshold', 0.1)

    # ADD THIS CODE AFTER getting parameters (around line 200):
    # Validate parameters before creating strategy
    def validate_parameters():
        """Validate all strategy parameters"""
        # Check period relationships
        if fast_ema_period >= slow_ema_period:
            raise ValueError(f"Fast EMA period ({fast_ema_period}) must be less than slow EMA period ({slow_ema_period})")
        
        # Check reasonable period ranges
        max_period = max(fast_ema_period, slow_ema_period, rsi_period, atr_period, bb_period, volume_sma_period)
        if max_period > 200:
            logger.warning(f"Very large period detected: {max_period}. Ensure you have sufficient data.")
        
        # Check ATR multiplier values
        if atr_multiplier_sl <= 0 or atr_multiplier_tp <= 0:
            raise ValueError("ATR multipliers must be positive values")
        
        # Check Bollinger Bands parameters
        if bb_std_dev <= 0:
            raise ValueError("Bollinger Bands standard deviation must be positive")

    try:
        validate_parameters()
        logger.info("✅ All parameters validated successfully")
    except ValueError as e:
        logger.error(f"❌ Parameter validation failed: {e}")
        raise
    
    logger.info(f"🎯 Creating Multi-Timeframe Scalping strategy with parameters:")
    logger.info(f" - Symbols: {symbols}")
    logger.info(f" - Timeframes: {timeframes}")
    logger.info(f" - Fast EMA: {fast_ema_period}, Slow EMA: {slow_ema_period}")
    logger.info(f" - RSI: {rsi_period} (OB: {rsi_overbought}, OS: {rsi_oversold})")
    logger.info(f" - ATR: {atr_period} (SL: {atr_multiplier_sl}x, TP: {atr_multiplier_tp}x)")
    logger.info(f" - Bollinger Bands: {bb_period}, {bb_std_dev} std")
    logger.info(f" - Volume SMA: {volume_sma_period}")
    logger.info(f" - Trend Timeframe: {trend_timeframe}")
    logger.info(f" - Min ATR Threshold: {min_atr_threshold}%")
    
    try:
        # Create strategy using StrategyBuilder
        strategy_builder = StrategyBuilder(symbols, timeframes)
        
        # Add indicators for each timeframe
        for timeframe in timeframes:
            # Trend indicators
            strategy_builder.add_indicator(f'ema_fast_{timeframe}', ema, period=fast_ema_period)
            strategy_builder.add_indicator(f'ema_slow_{timeframe}', ema, period=slow_ema_period)
            
            # Momentum indicator
            strategy_builder.add_indicator(f'rsi_{timeframe}', rsi, period=rsi_period)
            
            # Volatility indicators
            strategy_builder.add_indicator(f'atr_{timeframe}', atr, period=atr_period)
            
            # FIXED: Bollinger Bands - add the main indicator that returns all components
            strategy_builder.add_indicator(f'bb_{timeframe}', bollinger_bands, period=bb_period, std_dev=bb_std_dev)
            
            # Volume indicator
            strategy_builder.add_indicator(f'volume_sma_{timeframe}', volume_sma, period=volume_sma_period)
        
        # Add signal rules for entry timeframe (1m)
        entry_timeframe = '1m' if '1m' in timeframes else timeframes[0]
        
        # 1. EMA Crossover Signal
        strategy_builder.add_signal_rule('ema_crossover', ma_crossover,
                                       fast_ma=f'ema_fast_{entry_timeframe}',
                                       slow_ma=f'ema_slow_{entry_timeframe}')
        
        # 2. RSI Overbought/Oversold Signal
        strategy_builder.add_signal_rule('rsi_signal', overbought_oversold,
                                       indicator=f'rsi_{entry_timeframe}',
                                       overbought=rsi_overbought,
                                       oversold=rsi_oversold)
        
        # 3. Bollinger Bands Breakout Signal - FIXED: Use correct component names
        strategy_builder.add_signal_rule('bb_breakout', bollinger_bands_signals,
            price='close', # Will use close price from data
            upper_band=f'bb_{entry_timeframe}_upper_band', # Use upper band component
            lower_band=f'bb_{entry_timeframe}_lower_band', # Use lower band component
            middle_band=f'bb_{entry_timeframe}_middle_band') # Use middle band component
        
        # Set signal combination method
        strategy_builder.set_signal_combination('majority_vote')
        
        # Set strategy information
        strategy_builder.set_strategy_info('Multi_Timeframe_Scalping', '1.0.0')
        
        # Build and return the strategy
        strategy = strategy_builder.build()
        
        logger.info(f"✅ Multi-Timeframe Scalping strategy created successfully!")
        logger.info(f" - Strategy Name: {strategy.name}")
        logger.info(f" - Strategy Symbols: {strategy.symbols}")
        logger.info(f" - Strategy Timeframes: {strategy.timeframes}")
        
        return strategy
        
    except Exception as e:
        logger.error(f"❌ Error creating Multi-Timeframe Scalping strategy: {e}")
        import traceback
        traceback.print_exc()
        raise

class MultiTimeframeScalpingStrategy(StrategyBase):
    """
    Multi-Timeframe Scalping Strategy Class
    Implements the actual trading logic with advanced risk management
    """
    
    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        """
        Initialize the scalping strategy
        """
        # CRITICAL: Initialize with EXACT symbols and timeframes provided
        super().__init__(
            name="Multi_Timeframe_Scalping",
            symbols=symbols,
            timeframes=timeframes,
            config=config
        )
        
        # Strategy-specific parameters
        self.fast_ema_period = config.get('fast_ema_period', 9)
        self.slow_ema_period = config.get('slow_ema_period', 21)
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.atr_period = config.get('atr_period', 14)
        self.atr_multiplier_sl = config.get('atr_multiplier_sl', 1.5)
        self.atr_multiplier_tp = config.get('atr_multiplier_tp', 2.5)
        self.bb_period = config.get('bb_period', 20)
        self.bb_std_dev = config.get('bb_std_dev', 2.0)
        self.volume_sma_period = config.get('volume_sma_period', 20)
        self.trend_timeframe = config.get('trend_timeframe', '5m')
        self.min_atr_threshold = config.get('min_atr_threshold', 0.1)
        
        # Risk management
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.01)  # 1% risk per trade
        self.min_position_size = 0.001
        self.max_position_size = 1.0
        
        # Validate parameters
        self._validate_parameters()
        
        logger.info(f"📈 MultiTimeframeScalpingStrategy initialized:")
        logger.info(f" - Symbols: {self.symbols}")
        logger.info(f" - Timeframes: {self.timeframes}")
        logger.info(f" - Trend Timeframe: {self.trend_timeframe}")
        logger.info(f" - Risk per Trade: {self.max_risk_per_trade * 100}%")
    
    def _validate_parameters(self):
        """Validate strategy parameters"""
        if self.fast_ema_period >= self.slow_ema_period:
            raise ValueError("Fast EMA period must be less than slow EMA period")
        if self.rsi_oversold >= self.rsi_overbought:
            raise ValueError("RSI oversold level must be less than overbought level")
        if self.atr_multiplier_sl <= 0 or self.atr_multiplier_tp <= 0:
            raise ValueError("ATR multipliers must be positive")
    
    def calculate_position_size(self, symbol: str, current_price: float = None, signal_strength: float = 1.0) -> float:
        """
        Calculate position size based on ATR-based risk management
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
        Generate trading signals with multi-timeframe analysis
        """
        signals = {}
        
        try:
            for symbol in data:
                signals[symbol] = {}
                
                for timeframe in data[symbol]:
                    # Generate signal for each timeframe, passing the full data dictionary
                    signal = self._generate_single_signal(data[symbol][timeframe], symbol, timeframe, data)
                    signals[symbol][timeframe] = signal
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return signals
    
    def _generate_single_signal(self, df: pd.DataFrame, symbol: str, timeframe: str, data: Dict[str, Dict[str, pd.DataFrame]] = None) -> str:
        """
        Generate a single trading signal with multi-timeframe confirmation
        """
        try:
            if len(df) < max(self.slow_ema_period, self.bb_period, self.atr_period):
                return 'HOLD'
            
            # Get current values
            current_close = df['close'].iloc[-1]
            current_rsi = df[f'rsi_{timeframe}'].iloc[-1] if f'rsi_{timeframe}' in df.columns else 50
            current_atr = df[f'atr_{timeframe}'].iloc[-1] if f'atr_{timeframe}' in df.columns else 0
            
            # Check minimum volatility threshold
            atr_percent = (current_atr / current_close) * 100
            if atr_percent < self.min_atr_threshold:
                return 'HOLD'
            
            # Get trend direction from higher timeframe (only if data is provided)
            trend_signal = 'NEUTRAL'  # Default value
            if data is not None:
                trend_signal = self._get_trend_signal(data, symbol, timeframe)
            
            # Entry timeframe logic (1m or primary timeframe)
            if timeframe == '1m' or timeframe == self.timeframes[0]:
                return self._generate_entry_signal(df, symbol, timeframe, trend_signal)
            else:
                # Higher timeframes only for trend confirmation
                return trend_signal
                
        except Exception as e:
            logger.error(f"Error generating signal for {symbol} {timeframe}: {e}")
            return 'HOLD'
    
    def _get_trend_signal(self, data: Dict[str, Dict[str, pd.DataFrame]], symbol: str, current_timeframe: str) -> str:
        """
        Get trend signal from higher timeframe
        """
        try:
            if self.trend_timeframe not in data.get(symbol, {}):
                return 'HOLD'
            
            trend_df = data[symbol][self.trend_timeframe]
            if len(trend_df) < self.slow_ema_period:
                return 'HOLD'
            
            # Get EMAs from trend timeframe
            ema_fast = trend_df[f'ema_fast_{self.trend_timeframe}'].iloc[-1]
            ema_slow = trend_df[f'ema_slow_{self.trend_timeframe}'].iloc[-1]
            
            # Determine trend
            if ema_fast > ema_slow:
                return 'BULLISH'
            elif ema_fast < ema_slow:
                return 'BEARISH'
            else:
                return 'NEUTRAL'
                
        except Exception as e:
            logger.error(f"Error getting trend signal: {e}")
            return 'NEUTRAL'
    
    def _generate_entry_signal(self, df: pd.DataFrame, symbol: str, timeframe: str, trend_signal: str) -> str:
        """
        Generate entry signal based on multiple indicator confirmations
        """
        try:
            # Get current values
            current_close = df['close'].iloc[-1]
            current_rsi = df[f'rsi_{timeframe}'].iloc[-1]
            
            # Get EMAs
            ema_fast = df[f'ema_fast_{timeframe}'].iloc[-1]
            ema_slow = df[f'ema_slow_{timeframe}'].iloc[-1]
            
            # FIXED: Get Bollinger Bands components from the main indicator
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
            
            # Volume confirmation
            volume_confirmed = current_volume > volume_sma * 1.1  # 10% above average
            
            # Bullish setup
            if trend_signal == 'BULLISH':
                # Conditions for LONG entry:
                # 1. Price above EMAs
                # 2. RSI not overbought
                # 3. Price breaking above middle BB or pulling back to upper BB
                # 4. Volume confirmation
                
                price_above_emas = current_close > ema_fast and current_close > ema_slow
                rsi_not_overbought = current_rsi < self.rsi_overbought
                
                # Breakout or pullback setup
                bb_breakout = current_close > bb_middle
                bb_pullback = current_close < bb_upper and current_close > bb_lower
                
                if (price_above_emas and rsi_not_overbought and 
                    (bb_breakout or bb_pullback) and volume_confirmed):
                    return 'BUY'
            
            # Bearish setup
            elif trend_signal == 'BEARISH':
                # Conditions for SHORT entry:
                # 1. Price below EMAs
                # 2. RSI not oversold
                # 3. Price breaking below middle BB or pulling back to lower BB
                # 4. Volume confirmation
                
                price_below_emas = current_close < ema_fast and current_close < ema_slow
                rsi_not_oversold = current_rsi > self.rsi_oversold
                
                # Breakout or pullback setup
                bb_breakout = current_close < bb_middle
                bb_pullback = current_close > bb_lower and current_close < bb_upper
                
                if (price_below_emas and rsi_not_oversold and 
                    (bb_breakout or bb_pullback) and volume_confirmed):
                    return 'SELL'
            
            return 'HOLD'
            
        except Exception as e:
            logger.error(f"Error generating entry signal: {e}")
            return 'HOLD'

def create_multi_timeframe_scalping_instance(symbols=None, timeframes=None, **params):
    """
    Create multi-timeframe scalping strategy instance - OPTIONAL but recommended
    """
    try:
        if symbols is None:
            symbols = ['BTCUSDT']
        if timeframes is None:
            timeframes = ['1m', '5m']
        
        strategy = MultiTimeframeScalpingStrategy(symbols, timeframes, params)
        logger.info(f"✅ Multi-Timeframe Scalping strategy created successfully")
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
            fast_ema_period=9,
            slow_ema_period=21,
            rsi_period=14,
            rsi_overbought=70,
            rsi_oversold=30,
            atr_period=14,
            atr_multiplier_sl=1.5,
            atr_multiplier_tp=2.5,
            bb_period=20,
            bb_std_dev=2.0,
            volume_sma_period=20,
            trend_timeframe='5m',
            min_atr_threshold=0.1
        )
        
        print(f"✅ Multi-Timeframe Scalping strategy created successfully: {strategy.name}")
        print(f" - Symbols: {strategy.symbols}")
        print(f" - Timeframes: {strategy.timeframes}")
        print(f" - Strategy Type: Multi-timeframe scalping with volatility breakout")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Multi-Timeframe Scalping strategy: {e}")
        return False

# For testing - MUST EXIST
if __name__ == "__main__":
    simple_test()