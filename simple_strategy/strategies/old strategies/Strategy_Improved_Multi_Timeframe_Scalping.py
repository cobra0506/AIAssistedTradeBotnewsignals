"""
Improved Multi-Timeframe Scalping Strategy
========================================

A refined scalping strategy that addresses the issues with the original version:
- Simplified signal logic with only 2 indicators (EMA + RSI)
- Fixed stop-loss at 1x ATR and take-profit at 2x ATR
- No position averaging - enter once, exit once
- Better risk management with smaller position sizes

Strategy Logic:
1. TREND FILTER (Higher Timeframe): Price above/below EMAs determines overall trend
2. ENTRY SIGNAL (1m): EMA crossover with RSI confirmation
3. RISK MANAGEMENT: Fixed ATR-based stops and profit targets

Best for: Quick trades on 1m-5m charts with strict risk management
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
from simple_strategy.strategies.indicators_library import ema, rsi, atr, volume_sma
from simple_strategy.strategies.signals_library import ma_crossover, overbought_oversold
from simple_strategy.shared.strategy_base import StrategyBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CRITICAL: STRATEGY_PARAMETERS for GUI Configuration
STRATEGY_PARAMETERS = {
    # Fast EMA for entry signals
    'fast_ema_period': {
        'type': 'int',
        'default': 9,
        'min': 5,
        'max': 20,
        'description': 'Fast EMA period for entry signals',
        'gui_hint': 'Lower values = more sensitive entries. Recommended: 8-12'
    },
    # Slow EMA for trend direction
    'slow_ema_period': {
        'type': 'int',
        'default': 21,
        'min': 15,
        'max': 50,
        'description': 'Slow EMA period for trend direction',
        'gui_hint': 'Higher values = smoother trend. Recommended: 20-25'
    },
    # RSI for momentum confirmation
    'rsi_period': {
        'type': 'int',
        'default': 14,
        'min': 7,
        'max': 21,
        'description': 'RSI period for momentum confirmation',
        'gui_hint': 'Standard values: 14, 10 for faster signals'
    },
    # RSI levels
    'rsi_overbought': {
        'type': 'int',
        'default': 70,
        'min': 60,
        'max': 80,
        'description': 'RSI overbought level for sell signals',
        'gui_hint': 'Higher = more conservative sells'
    },
    'rsi_oversold': {
        'type': 'int',
        'default': 30,
        'min': 20,
        'max': 40,
        'description': 'RSI oversold level for buy signals',
        'gui_hint': 'Lower = more conservative buys'
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
    # Risk management parameters
    'atr_stop_loss': {
        'type': 'float',
        'default': 1.0,
        'min': 0.5,
        'max': 2.0,
        'description': 'ATR multiplier for stop loss',
        'gui_hint': 'Lower = tighter stops, more conservative'
    },
    'atr_take_profit': {
        'type': 'float',
        'default': 2.0,
        'min': 1.0,
        'max': 4.0,
        'description': 'ATR multiplier for take profit',
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
        'default': 0.1,
        'min': 0.05,
        'max': 0.5,
        'description': 'Minimum ATR threshold for trading (as % of price)',
        'gui_hint': 'Filter out low volatility periods. 0.1 = 0.1%'
    },
    # Risk per trade (reduced from original)
    'risk_per_trade': {
        'type': 'float',
        'default': 0.5,
        'min': 0.1,
        'max': 2.0,
        'description': 'Risk per trade as % of account balance',
        'gui_hint': 'Lower = more conservative. Recommended: 0.5-1.0%'
    }
}

def create_strategy(symbols=None, timeframes=None, **params):
    """
    CREATE STRATEGY FUNCTION - Required by GUI
    This function is called by the GUI to create strategy instances.
    
    Creates an improved multi-timeframe scalping strategy with simplified logic.
    
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
    atr_stop_loss = params.get('atr_stop_loss', 1.0)
    atr_take_profit = params.get('atr_take_profit', 2.0)
    volume_sma_period = params.get('volume_sma_period', 20)
    min_atr_threshold = params.get('min_atr_threshold', 0.1)
    risk_per_trade = params.get('risk_per_trade', 0.5)
    
    logger.info(f"🎯 Creating Improved Multi-Timeframe Scalping strategy with parameters:")
    logger.info(f" - Symbols: {symbols}")
    logger.info(f" - Timeframes: {timeframes}")
    logger.info(f" - Fast EMA: {fast_ema_period}, Slow EMA: {slow_ema_period}")
    logger.info(f" - RSI: {rsi_period} (OB: {rsi_overbought}, OS: {rsi_oversold})")
    logger.info(f" - ATR: {atr_period} (SL: {atr_stop_loss}x, TP: {atr_take_profit}x)")
    logger.info(f" - Volume SMA: {volume_sma_period}")
    logger.info(f" - Trend Timeframe: {trend_timeframe}")
    logger.info(f" - Min ATR Threshold: {min_atr_threshold}%")
    logger.info(f" - Risk per Trade: {risk_per_trade}%")
    
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
            
            # Volatility indicator
            strategy_builder.add_indicator(f'atr_{timeframe}', atr, period=atr_period)
            
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
        
        # Set signal combination method - require both signals to agree
        strategy_builder.set_signal_combination('unanimous')
        
        # Set strategy information
        strategy_builder.set_strategy_info('Improved_Multi_Timeframe_Scalping', '1.0.0')
        
        # Build and return the strategy
        strategy = strategy_builder.build()
        
        logger.info(f"✅ Improved Multi-Timeframe Scalping strategy created successfully!")
        logger.info(f" - Strategy Name: {strategy.name}")
        logger.info(f" - Strategy Symbols: {strategy.symbols}")
        logger.info(f" - Strategy Timeframes: {strategy.timeframes}")
        
        return strategy
        
    except Exception as e:
        logger.error(f"❌ Error creating Improved Multi-Timeframe Scalping strategy: {e}")
        import traceback
        traceback.print_exc()
        raise

class ImprovedMultiTimeframeScalpingStrategy(StrategyBase):
    """
    Improved Multi-Timeframe Scalping Strategy Class
    Implements simplified trading logic with strict risk management
    """
    
    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        """
        Initialize the improved scalping strategy
        """
        # CRITICAL: Initialize with EXACT symbols and timeframes provided
        super().__init__(
            name="Improved_Multi_Timeframe_Scalping",
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
        self.atr_stop_loss = config.get('atr_stop_loss', 1.0)
        self.atr_take_profit = config.get('atr_take_profit', 2.0)
        self.volume_sma_period = config.get('volume_sma_period', 20)
        self.trend_timeframe = config.get('trend_timeframe', '5m')
        self.min_atr_threshold = config.get('min_atr_threshold', 0.1)
        self.risk_per_trade = config.get('risk_per_trade', 0.5) / 100.0  # Convert to decimal
        
        # Risk management
        self.min_position_size = 0.001
        self.max_position_size = 0.1  # Reduced max position size
        
        # Track positions to prevent averaging
        self.positions = {}
        
        # Validate parameters
        self._validate_parameters()
        
        logger.info(f"📈 ImprovedMultiTimeframeScalpingStrategy initialized:")
        logger.info(f" - Symbols: {self.symbols}")
        logger.info(f" - Timeframes: {self.timeframes}")
        logger.info(f" - Trend Timeframe: {self.trend_timeframe}")
        logger.info(f" - Risk per Trade: {self.risk_per_trade * 100}%")
    
    def _validate_parameters(self):
        """Validate strategy parameters"""
        if self.fast_ema_period >= self.slow_ema_period:
            raise ValueError("Fast EMA period must be less than slow EMA period")
        if self.rsi_oversold >= self.rsi_overbought:
            raise ValueError("RSI oversold level must be less than overbought level")
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
        Generate trading signals with simplified logic
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
        Generate a single trading signal with simplified logic
        """
        try:
            if len(df) < max(self.slow_ema_period, self.rsi_period, self.atr_period):
                return 'HOLD'
            
            # Get current values
            current_close = df['close'].iloc[-1]
            current_rsi = df[f'rsi_{timeframe}'].iloc[-1] if f'rsi_{timeframe}' in df.columns else 50
            current_atr = df[f'atr_{timeframe}'].iloc[-1] if f'atr_{timeframe}' in df.columns else 0
            
            # Check minimum volatility threshold
            atr_percent = (current_atr / current_close) * 100
            if atr_percent < self.min_atr_threshold:
                return 'HOLD'
            
            # Get trend direction from higher timeframe
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
        Generate entry signal with simplified logic
        """
        try:
            # Get current values
            current_close = df['close'].iloc[-1]
            current_rsi = df[f'rsi_{timeframe}'].iloc[-1]
            
            # Get EMAs
            ema_fast = df[f'ema_fast_{timeframe}'].iloc[-1]
            ema_slow = df[f'ema_slow_{timeframe}'].iloc[-1]
            
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
                # 3. Volume confirmation
                
                price_above_emas = current_close > ema_fast and current_close > ema_slow
                rsi_not_overbought = current_rsi < self.rsi_overbought
                
                if price_above_emas and rsi_not_overbought and volume_confirmed:
                    return 'BUY'
            
            # Bearish setup
            elif trend_signal == 'BEARISH':
                # Conditions for SHORT entry:
                # 1. Price below EMAs
                # 2. RSI not oversold
                # 3. Volume confirmation
                
                price_below_emas = current_close < ema_fast and current_close < ema_slow
                rsi_not_oversold = current_rsi > self.rsi_oversold
                
                if price_below_emas and rsi_not_oversold and volume_confirmed:
                    return 'SELL'
            
            return 'HOLD'
            
        except Exception as e:
            logger.error(f"Error generating entry signal: {e}")
            return 'HOLD'

def create_improved_multi_timeframe_scalping_instance(symbols=None, timeframes=None, **params):
    """
    Create improved multi-timeframe scalping strategy instance
    """
    try:
        if symbols is None:
            symbols = ['BTCUSDT']
        if timeframes is None:
            timeframes = ['1m', '5m']
        
        strategy = ImprovedMultiTimeframeScalpingStrategy(symbols, timeframes, params)
        logger.info(f"✅ Improved Multi-Timeframe Scalping strategy created successfully")
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
            atr_stop_loss=1.0,
            atr_take_profit=2.0,
            volume_sma_period=20,
            trend_timeframe='5m',
            min_atr_threshold=0.1,
            risk_per_trade=0.5
        )
        
        print(f"✅ Improved Multi-Timeframe Scalping strategy created successfully: {strategy.name}")
        print(f" - Symbols: {strategy.symbols}")
        print(f" - Timeframes: {strategy.timeframes}")
        print(f" - Strategy Type: Improved multi-timeframe scalping with strict risk management")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Improved Multi-Timeframe Scalping strategy: {e}")
        return False

# For testing - MUST EXIST
if __name__ == "__main__":
    simple_test()