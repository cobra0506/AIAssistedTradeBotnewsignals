"""
Improved RSI Trend Extremes Strategy
===================================

Logic:
- Only trade longs in uptrend (EMA20 > EMA50)
    - OPEN_LONG when RSI crosses **oversold**
    - CLOSE_LONG when RSI crosses **overbought**
- Only trade shorts in downtrend (EMA20 < EMA50)
    - OPEN_SHORT when RSI crosses **overbought**
    - CLOSE_SHORT when RSI crosses **oversold**
- HOLD otherwise

Author: Thys
Date: 2026
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any

# Add parent directories for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from simple_strategy.shared.strategy_base import StrategyBase
from simple_strategy.strategies.indicators_library import rsi, ema
from simple_strategy.strategies.signals_library import crossover, crossunder, oversold_cross, overbought_cross

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GUI configurable parameters
STRATEGY_PARAMETERS = {
    'rsi_period': {'type': 'int', 'default': 14, 'min': 7, 'max': 21, 'description': 'RSI period'},
    'rsi_overbought': {'type': 'int', 'default': 75, 'min': 70, 'max': 80, 'description': 'RSI overbought level'},
    'rsi_oversold': {'type': 'int', 'default': 25, 'min': 20, 'max': 30, 'description': 'RSI oversold level'},
    'ema_fast_period': {'type': 'int', 'default': 20, 'min': 10, 'max': 50, 'description': 'Fast EMA period for trend'},
    'ema_slow_period': {'type': 'int', 'default': 50, 'min': 20, 'max': 100, 'description': 'Slow EMA period for trend'}
}

class ImprovedRSITrendExtremesStrategy(StrategyBase):
    """Improved RSI Trend Extremes Strategy Class"""
    
    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        super().__init__("Improved_RSI_Trend_Extremes", symbols, timeframes, config)
        
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_overbought = config.get('rsi_overbought', 75)
        self.rsi_oversold = config.get('rsi_oversold', 25)
        self.ema_fast_period = config.get('ema_fast_period', 20)
        self.ema_slow_period = config.get('ema_slow_period', 50)
        
        logger.info("📈 ImprovedRSITrendExtremesStrategy initialized")
    
    #def calculate_position_size(self, symbol: str, current_price: float = None, signal_strength: float = 1.0) -> float:
       # return 0.001  # fixed small position
    
    def generate_signals(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, str]]:
        """Generate signals for all symbols and timeframes"""
        signals = {}
        
        try:
            for symbol in data:
                signals[symbol] = {}
                
                for timeframe, df in data[symbol].items():
                    # Copy dataframe to avoid overwriting
                    df = df.copy()
                    
                    # Calculate indicators
                    df['rsi'] = rsi(df['close'], period=self.rsi_period)
                    df['ema_fast'] = ema(df['close'], period=self.ema_fast_period)
                    df['ema_slow'] = ema(df['close'], period=self.ema_slow_period)
                    
                    # Generate signal for this timeframe
                    signal = self._generate_single_signal(df)
                    signals[symbol][timeframe] = signal
            
            return signals
        
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return signals
    
    def _generate_single_signal(self, df: pd.DataFrame) -> str:
        """Generate signal based on trend + RSI crossovers"""
        try:
            if len(df) < max(self.rsi_period, self.ema_fast_period, self.ema_slow_period):
                return 'HOLD'
            
            # Current and previous values
            current_rsi = df['rsi'].iloc[-1]
            previous_rsi = df['rsi'].iloc[-2] if len(df) > 1 else current_rsi
            ema_fast = df['ema_fast'].iloc[-1]
            ema_slow = df['ema_slow'].iloc[-1]
            
            # Determine trend
            uptrend = ema_fast > ema_slow
            downtrend = ema_fast < ema_slow
            
            # Signals
            if uptrend:
                if oversold_cross(df['rsi'], self.rsi_oversold).iloc[-1]:
                    return 'OPEN_LONG'
                elif overbought_cross(df['rsi'], self.rsi_overbought).iloc[-1]:
                    return 'CLOSE_LONG'
            elif downtrend:
                if overbought_cross(df['rsi'], self.rsi_overbought).iloc[-1]:
                    return 'OPEN_SHORT'
                elif oversold_cross(df['rsi'], self.rsi_oversold).iloc[-1]:
                    return 'CLOSE_SHORT'
            
            return 'HOLD'
        
        except Exception as e:
            logger.error(f"Error generating single signal: {e}")
            return 'HOLD'


# Factory function for GUI
def create_strategy(symbols=None, timeframes=None, **params):
    if symbols is None:
        symbols = ['BTCUSDT']
    if timeframes is None:
        timeframes = ['5m']
    return ImprovedRSITrendExtremesStrategy(symbols, timeframes, params)


# Simple test
def simple_test():
    try:
        strategy = create_strategy(
            symbols=['BTCUSDT'],
            timeframes=['5m'],
            rsi_period=14,
            rsi_overbought=75,
            rsi_oversold=25,
            ema_fast_period=20,
            ema_slow_period=50
        )
        print(f"✅ Strategy created: {strategy.name}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    simple_test()

