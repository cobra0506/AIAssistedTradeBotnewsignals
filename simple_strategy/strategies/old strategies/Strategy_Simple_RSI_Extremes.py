"""
Improved Simple RSI Extremes Strategy
====================================

An improved mean reversion strategy with better filters:
- Uses more extreme RSI levels for higher quality signals
- Adds trend filter (200 SMA) to trade with the trend
- Adds volume filter to confirm signal strength
- No multi-timeframe complexity

Strategy Logic:
1. OVERSOLD: RSI < 25 AND price > 200 SMA AND volume > average = BUY
2. OVERBOUGHT: RSI > 75 AND price < 200 SMA AND volume > average = SELL
3. EXIT: Opposite RSI extreme

Best for: Trending markets with mean reversion opportunities
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
from simple_strategy.strategies.indicators_library import rsi, sma, volume_sma
from simple_strategy.strategies.signals_library import overbought_oversold
from simple_strategy.shared.strategy_base import StrategyBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CRITICAL: STRATEGY_PARAMETERS for GUI Configuration
STRATEGY_PARAMETERS = {
    # RSI for mean reversion signals
    'rsi_period': {
        'type': 'int',
        'default': 8,
        'min': 7,
        'max': 21,
        'description': 'RSI period for mean reversion signals',
        'gui_hint': 'Standard values: 14, 10 for faster signals'
    },
    # RSI levels - more extreme for higher quality signals
    'rsi_overbought': {
        'type': 'int',
        'default': 77,
        'min': 70,
        'max': 80,
        'description': 'RSI overbought level for sell signals',
        'gui_hint': 'Higher = more extreme overbought, fewer but higher quality signals'
    },
    'rsi_oversold': {
        'type': 'int',
        'default': 26,
        'min': 20,
        'max': 30,
        'description': 'RSI oversold level for buy signals',
        'gui_hint': 'Lower = more extreme oversold, fewer but higher quality signals'
    },
    # Trend filter
    'trend_sma_period': {
        'type': 'int',
        'default': 233,
        'min': 50,
        'max': 300,
        'description': 'SMA period for trend filter',
        'gui_hint': 'Higher = longer term trend. 200 is standard for daily charts'
    },
    # Volume filter
    'volume_sma_period': {
        'type': 'int',
        'default': 34,
        'min': 10,
        'max': 50,
        'description': 'Volume SMA period for confirmation',
        'gui_hint': 'Higher = smoother volume trend'
    },
    # Volume multiplier
    'volume_multiplier': {
        'type': 'float',
        'default': 1.79,
        'min': 1.0,
        'max': 2.0,
        'description': 'Volume multiplier for signal confirmation',
        'gui_hint': 'Higher = stronger volume confirmation required'
    }
}

def create_strategy(symbols=None, timeframes=None, **params):
    """CREATE STRATEGY FUNCTION - Required by GUI"""
    logger.info(f"🔧 create_strategy called with:")
    logger.info(f" - symbols: {symbols}")
    logger.info(f" - timeframes: {timeframes}")
    logger.info(f" - params: {params}")
    
    if symbols is None or len(symbols) == 0:
        logger.warning("⚠️ No symbols provided, using default: ['BTCUSDT']")
        symbols = ['BTCUSDT']
    
    if timeframes is None or len(timeframes) == 0:
        logger.warning("⚠️ No timeframes provided, using default: ['5m']")
        timeframes = ['5m']
    
    # Get parameters
    rsi_period = params.get('rsi_period', 14)
    rsi_overbought = params.get('rsi_overbought', 75)
    rsi_oversold = params.get('rsi_oversold', 25)
    trend_sma_period = params.get('trend_sma_period', 200)
    volume_sma_period = params.get('volume_sma_period', 20)
    volume_multiplier = params.get('volume_multiplier', 1.2)
    
    logger.info(f"🎯 Creating Improved Simple RSI Extremes strategy with parameters:")
    logger.info(f" - Symbols: {symbols}")
    logger.info(f" - Timeframes: {timeframes}")
    logger.info(f" - RSI: {rsi_period} (OB: {rsi_overbought}, OS: {rsi_oversold})")
    logger.info(f" - Trend SMA: {trend_sma_period}")
    logger.info(f" - Volume SMA: {volume_sma_period} (Multiplier: {volume_multiplier}x)")
    
    try:
        strategy_builder = StrategyBuilder(symbols, timeframes)
        
        # Add indicators
        for timeframe in timeframes:
            strategy_builder.add_indicator(f'rsi_{timeframe}', rsi, period=rsi_period)
            strategy_builder.add_indicator(f'trend_sma_{timeframe}', sma, period=trend_sma_period)
            strategy_builder.add_indicator(f'volume_sma_{timeframe}', volume_sma, period=volume_sma_period)
        
        # Add signal rule
        entry_timeframe = timeframes[0]
        strategy_builder.add_signal_rule('rsi_signal', overbought_oversold,
                                       indicator=f'rsi_{entry_timeframe}',
                                       overbought=rsi_overbought,
                                       oversold=rsi_oversold)
        
        strategy_builder.set_signal_combination('majority_vote')
        strategy_builder.set_strategy_info('Improved_Simple_RSI_Extremes', '1.0.0')
        
        strategy = strategy_builder.build()
        
        logger.info(f"✅ Improved Simple RSI Extremes strategy created successfully!")
        return strategy
        
    except Exception as e:
        logger.error(f"❌ Error creating Improved Simple RSI Extremes strategy: {e}")
        raise

class ImprovedSimpleRSIExtremesStrategy(StrategyBase):
    """Improved Simple RSI Extremes Strategy Class"""
    
    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        super().__init__(
            name="Improved_Simple_RSI_Extremes",
            symbols=symbols,
            timeframes=timeframes,
            config=config
        )
        
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_overbought = config.get('rsi_overbought', 75)
        self.rsi_oversold = config.get('rsi_oversold', 25)
        self.trend_sma_period = config.get('trend_sma_period', 200)
        self.volume_sma_period = config.get('volume_sma_period', 20)
        self.volume_multiplier = config.get('volume_multiplier', 1.2)
        
        self._validate_parameters()
        logger.info(f"📈 ImprovedSimpleRSIExtremesStrategy initialized")
    
    def _validate_parameters(self):
        if self.rsi_oversold >= self.rsi_overbought:
            raise ValueError("RSI oversold level must be less than overbought level")
        if self.volume_multiplier < 1.0:
            raise ValueError("Volume multiplier must be at least 1.0")
    
    def calculate_position_size(self, symbol: str, current_price: float = None, signal_strength: float = 1.0) -> float:
        return 0.001  # Fixed small position
    
    def generate_signals(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, str]]:
        signals = {}
        
        try:
            for symbol in data:
                signals[symbol] = {}
                
                for timeframe in data[symbol]:
                    signal = self._generate_single_signal(data[symbol][timeframe], symbol, timeframe)
                    signals[symbol][timeframe] = signal
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return signals
    
    def _generate_single_signal(self, df: pd.DataFrame, symbol: str, timeframe: str) -> str:
        try:
            # Need enough data for all indicators
            min_periods = max(self.rsi_period, self.trend_sma_period, self.volume_sma_period)
            if len(df) < min_periods:
                return 'HOLD'
            
            # Get current values
            current_close = df['close'].iloc[-1]
            current_rsi = df[f'rsi_{timeframe}'].iloc[-1]
            current_volume = df['volume'].iloc[-1]
            
            # Get trend SMA and volume SMA
            trend_sma = df[f'trend_sma_{timeframe}'].iloc[-1]
            volume_sma = df[f'volume_sma_{timeframe}'].iloc[-1]
            
            # Check volume confirmation
            volume_confirmed = current_volume > volume_sma * self.volume_multiplier
            
            # Improved RSI extremes logic with trend and volume filters
            
            # Bullish setup: RSI oversold AND price above trend SMA AND volume confirmed
            if current_rsi <= self.rsi_oversold and current_close > trend_sma and volume_confirmed:
                return 'BUY'
            
            # Bearish setup: RSI overbought AND price below trend SMA AND volume confirmed
            elif current_rsi >= self.rsi_overbought and current_close < trend_sma and volume_confirmed:
                return 'SELL'
            
            return 'HOLD'
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol} {timeframe}: {e}")
            return 'HOLD'

def create_improved_simple_rsi_extremes_instance(symbols=None, timeframes=None, **params):
    try:
        if symbols is None:
            symbols = ['BTCUSDT']
        if timeframes is None:
            timeframes = ['5m']
        
        strategy = ImprovedSimpleRSIExtremesStrategy(symbols, timeframes, params)
        logger.info(f"✅ Improved Simple RSI Extremes strategy created successfully")
        return strategy
    except Exception as e:
        logger.error(f"Error creating strategy: {e}")
        raise

def simple_test():
    try:
        strategy = create_strategy(
            symbols=['BTCUSDT'],
            timeframes=['5m'],
            rsi_period=14,
            rsi_overbought=75,
            rsi_oversold=25,
            trend_sma_period=200,
            volume_sma_period=20,
            volume_multiplier=1.2
        )
        
        print(f"✅ Improved Simple RSI Extremes strategy created successfully: {strategy.name}")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Improved Simple RSI Extremes strategy: {e}")
        return False

if __name__ == "__main__":
    simple_test()