"""
Improved Simple MA Crossover Strategy
======================================

An improved trend-following strategy with better responsiveness:
- Uses faster MAs for more responsive signals
- Adds RSI confirmation for better timing
- Uses multiple timeframes (1m for entries, 5m for trend)
- No complex filters

Strategy Logic:
1. TREND (5m): Price above 50 EMA = Bullish trend
2. ENTRY (1m): Fast EMA (5) crosses above Slow EMA (15) AND RSI confirms = BUY
3. EXIT (1m): Fast EMA crosses below Slow EMA AND RSI confirms = SELL

Best for: Trending markets with multiple timeframe confirmation
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
from simple_strategy.strategies.indicators_library import ema, rsi
from simple_strategy.strategies.signals_library import ma_crossover, overbought_oversold
from simple_strategy.shared.strategy_base import StrategyBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CRITICAL: STRATEGY_PARAMETERS for GUI Configuration
STRATEGY_PARAMETERS = {
    # Fast MA for entries (1m)
    'fast_ma_period': {
        'type': 'int',
        'default': 5,
        'min': 3,
        'max': 10,
        'description': 'Fast EMA period for entry signals',
        'gui_hint': 'For 1m timeframe. Lower = more responsive'
    },
    # Slow MA for entries (1m)
    'slow_ma_period': {
        'type': 'int',
        'default': 15,
        'min': 10,
        'max': 30,
        'description': 'Slow EMA period for entry signals',
        'gui_hint': 'For 1m timeframe. Higher = smoother'
    },
    # Trend MA for trend confirmation (5m)
    'trend_ma_period': {
        'type': 'int',
        'default': 50,
        'min': 20,
        'max': 100,
        'description': 'Trend EMA period for higher timeframe',
        'gui_hint': 'For 5m timeframe. Higher = longer term trend'
    },
    # RSI for confirmation
    'rsi_period': {
        'type': 'int',
        'default': 14,
        'min': 7,
        'max': 21,
        'description': 'RSI period for signal confirmation',
        'gui_hint': 'Standard values: 14, 10 for faster signals'
    },
    # RSI levels for confirmation
    'rsi_bullish_threshold': {
        'type': 'int',
        'default': 55,
        'min': 50,
        'max': 60,
        'description': 'RSI threshold for bullish confirmation',
        'gui_hint': 'Above this level confirms bullish signals'
    },
    'rsi_bearish_threshold': {
        'type': 'int',
        'default': 45,
        'min': 40,
        'max': 50,
        'description': 'RSI threshold for bearish confirmation',
        'gui_hint': 'Below this level confirms bearish signals'
    }
}

def create_strategy(symbols=None, timeframes=None, **params):
    """CREATE STRATEGY FUNCTION - Required by GUI"""
    logger.info(f"🔧 create_strategy called with:")
    logger.info(f" - symbols: {symbols}")
    logger.info(f" - timeframes: {timeframes}")
    logger.info(f" - params: {params}")
    
    if symbols is None or len(symbols) == 0:
        symbols = ['BTCUSDT']
    
    # CRITICAL: Ensure we have both timeframes
    if timeframes is None or len(timeframes) == 0:
        timeframes = ['1m', '5m']
    else:
        # Make sure we have 1m and 5m
        if '1m' not in timeframes:
            timeframes.append('1m')
        if '5m' not in timeframes:
            timeframes.append('5m')
    
    # Get parameters
    fast_ma_period = params.get('fast_ma_period', 5)
    slow_ma_period = params.get('slow_ma_period', 15)
    trend_ma_period = params.get('trend_ma_period', 50)
    rsi_period = params.get('rsi_period', 14)
    rsi_bullish_threshold = params.get('rsi_bullish_threshold', 55)
    rsi_bearish_threshold = params.get('rsi_bearish_threshold', 45)
    
    logger.info(f"🎯 Creating Improved Simple MA Crossover strategy with parameters:")
    logger.info(f" - Symbols: {symbols}")
    logger.info(f" - Timeframes: {timeframes}")
    logger.info(f" - Fast MA: {fast_ma_period}, Slow MA: {slow_ma_period}")
    logger.info(f" - Trend MA: {trend_ma_period}")
    logger.info(f" - RSI: {rsi_period} (Bullish: {rsi_bullish_threshold}, Bearish: {rsi_bearish_threshold})")
    
    try:
        strategy_builder = StrategyBuilder(symbols, timeframes)
        
        # Add indicators for each timeframe
        for timeframe in timeframes:
            if timeframe == '1m':
                # Entry indicators for 1m
                strategy_builder.add_indicator(f'fast_ma_1m', ema, period=fast_ma_period)
                strategy_builder.add_indicator(f'slow_ma_1m', ema, period=slow_ma_period)
                strategy_builder.add_indicator(f'rsi_1m', rsi, period=rsi_period)
            elif timeframe == '5m':
                # Trend indicator for 5m
                strategy_builder.add_indicator(f'trend_ma_5m', ema, period=trend_ma_period)
        
        # Add signal rules for 1m timeframe
        strategy_builder.add_signal_rule('ma_crossover', ma_crossover,
                                       fast_ma='fast_ma_1m',
                                       slow_ma='slow_ma_1m')
        
        strategy_builder.add_signal_rule('rsi_confirmation', overbought_oversold,
                                       indicator='rsi_1m',
                                       overbought=rsi_bullish_threshold,
                                       oversold=rsi_bearish_threshold)
        
        strategy_builder.set_signal_combination('majority_vote')
        strategy_builder.set_strategy_info('Improved_Simple_MA_Crossover', '1.0.0')
        
        strategy = strategy_builder.build()
        
        logger.info(f"✅ Improved Simple MA Crossover strategy created successfully!")
        return strategy
        
    except Exception as e:
        logger.error(f"❌ Error creating Improved Simple MA Crossover strategy: {e}")
        raise

class ImprovedSimpleMACrossoverStrategy(StrategyBase):
    """Improved Simple MA Crossover Strategy Class"""
    
    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        super().__init__(
            name="Improved_Simple_MA_Crossover",
            symbols=symbols,
            timeframes=timeframes,
            config=config
        )
        
        self.fast_ma_period = config.get('fast_ma_period', 5)
        self.slow_ma_period = config.get('slow_ma_period', 15)
        self.trend_ma_period = config.get('trend_ma_period', 50)
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_bullish_threshold = config.get('rsi_bullish_threshold', 55)
        self.rsi_bearish_threshold = config.get('rsi_bearish_threshold', 45)
        
        self._validate_parameters()
        logger.info(f"📈 ImprovedSimpleMACrossoverStrategy initialized")
    
    def _validate_parameters(self):
        if self.fast_ma_period >= self.slow_ma_period:
            raise ValueError("Fast MA period must be less than slow MA period")
        if self.rsi_bearish_threshold >= self.rsi_bullish_threshold:
            raise ValueError("RSI bearish threshold must be less than bullish threshold")
    
    def calculate_position_size(self, symbol: str, current_price: float = None, signal_strength: float = 1.0) -> float:
        return 0.001  # Fixed small position
    
    def generate_signals(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, str]]:
        signals = {}
        
        try:
            for symbol in data:
                signals[symbol] = {}
                
                for timeframe in data[symbol]:
                    signal = self._generate_single_signal(data[symbol][timeframe], symbol, timeframe, data)
                    signals[symbol][timeframe] = signal
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return signals
    
    def _generate_single_signal(self, df: pd.DataFrame, symbol: str, timeframe: str, all_data: Dict) -> str:
        try:
            # Only generate signals for 1m timeframe
            if timeframe != '1m':
                return 'HOLD'
            
            # Need enough data for all indicators
            min_periods = max(self.slow_ma_period, self.rsi_period)
            if len(df) < min_periods:
                return 'HOLD'
            
            # Get 1m indicators
            fast_ma = df['fast_ma_1m'].iloc[-1]
            slow_ma = df['slow_ma_1m'].iloc[-1]
            current_rsi = df['rsi_1m'].iloc[-1]
            current_close = df['close'].iloc[-1]
            
            # Get 5m trend
            bullish_trend = True  # Default to true if no 5m data
            if '5m' in all_data.get(symbol, {}) and len(all_data[symbol]['5m']) >= self.trend_ma_period:
                trend_ma = all_data[symbol]['5m']['trend_ma_5m'].iloc[-1]
                price_5m = all_data[symbol]['5m']['close'].iloc[-1]
                bullish_trend = price_5m > trend_ma
            
            # Improved MA crossover logic with RSI confirmation and trend filter
            
            # Bullish: Fast MA above Slow MA AND RSI above threshold AND bullish trend
            if fast_ma > slow_ma and current_rsi > self.rsi_bullish_threshold and bullish_trend:
                return 'BUY'
            
            # Bearish: Fast MA below Slow MA AND RSI below threshold AND not bullish trend
            elif fast_ma < slow_ma and current_rsi < self.rsi_bearish_threshold and not bullish_trend:
                return 'SELL'
            
            return 'HOLD'
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol} {timeframe}: {e}")
            return 'HOLD'

def create_improved_simple_ma_crossover_instance(symbols=None, timeframes=None, **params):
    try:
        if symbols is None:
            symbols = ['BTCUSDT']
        if timeframes is None:
            timeframes = ['1m', '5m']
        
        strategy = ImprovedSimpleMACrossoverStrategy(symbols, timeframes, params)
        logger.info(f"✅ Improved Simple MA Crossover strategy created successfully")
        return strategy
    except Exception as e:
        logger.error(f"Error creating strategy: {e}")
        raise

def simple_test():
    try:
        strategy = create_strategy(
            symbols=['BTCUSDT'],
            timeframes=['1m', '5m'],
            fast_ma_period=5,
            slow_ma_period=15,
            trend_ma_period=50,
            rsi_period=14,
            rsi_bullish_threshold=55,
            rsi_bearish_threshold=45
        )
        
        print(f"✅ Improved Simple MA Crossover strategy created successfully: {strategy.name}")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Improved Simple MA Crossover strategy: {e}")
        return False

if __name__ == "__main__":
    simple_test()