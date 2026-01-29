"""
ULTIMATE STRATEGY TEMPLATE - Standardized Structure
This is the CORRECT pattern that all strategies should follow.
Combines the best of both approaches: Simple to use but powerful.

🚫 IMPORTANT: Read the README_STRATEGY_CREATION_GUIDE.md before using this template!
"""
import sys
import os
import pandas as pd
import logging
from typing import Dict, List, Any, Optional

# Add parent directories to path for proper imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import required components
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import rsi, sma, ema, macd
from simple_strategy.strategies.signals_library import overbought_oversold, ma_crossover
from simple_strategy.shared.strategy_base import StrategyBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CRITICAL: STRATEGY_PARAMETERS for GUI Configuration
# This dictionary defines what parameters the GUI will show and allow users to configure
STRATEGY_PARAMETERS = {
    'indicator_period': {
        'type': 'int',
        'default': 14,
        'min': 5,
        'max': 50,
        'description': 'Primary indicator period',
        'gui_hint': 'Standard values: 14 for RSI, 20/50 for SMAs'
    },
    'overbought_level': {
        'type': 'int',
        'default': 70,
        'min': 60,
        'max': 90,
        'description': 'Overbought threshold for sell signals',
        'gui_hint': 'Higher values = more conservative sells'
    },
    'oversold_level': {
        'type': 'int',
        'default': 30,
        'min': 10,
        'max': 40,
        'description': 'Oversold threshold for buy signals',
        'gui_hint': 'Lower values = more conservative buys'
    }
}

def create_strategy(symbols=None, timeframes=None, **params):
    """
    CREATE STRATEGY FUNCTION - Required by GUI
    This function is called by the GUI to create strategy instances.
    
    ⚠️ CRITICAL - Read this before modifying:
    1. Use StrategyBuilder, don't create custom Strategy classes unless needed
    2. Handle None/empty values for symbols and timeframes
    3. Use correct signal combination methods: 'majority_vote', 'weighted', 'unanimous'
    4. For MACD signals, use 'macd' for both macd_line and signal_line parameters
    
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
    
    # CRITICAL: Handle None/empty values
    if symbols is None or len(symbols) == 0:
        logger.warning(f"🔧 DEBUG: symbols is None or empty, using ['BTCUSDT']")
        symbols = ['BTCUSDT']
    if timeframes is None or len(timeframes) == 0:
        logger.warning(f"🔧 DEBUG: timeframes is None or empty, using ['5m']")
        timeframes = ['5m']
    
    # Get parameters with defaults from STRATEGY_PARAMETERS
    indicator_period = params.get('indicator_period', 14)
    overbought_level = params.get('overbought_level', 70)
    oversold_level = params.get('oversold_level', 30)
    
    logger.info(f"🎯 Creating strategy with parameters:")
    logger.info(f" - Symbols: {symbols}")
    logger.info(f" - Timeframes: {timeframes}")
    logger.info(f" - Indicator Period: {indicator_period}")
    logger.info(f" - Overbought Level: {overbought_level}")
    logger.info(f" - Oversold Level: {oversold_level}")
    
    try:
        # Create strategy using StrategyBuilder
        strategy_builder = StrategyBuilder(symbols, timeframes)
        
        # Add indicators (use existing ones from indicators_library)
        strategy_builder.add_indicator('rsi', rsi, period=indicator_period)
        strategy_builder.add_indicator('sma_fast', sma, period=12)
        strategy_builder.add_indicator('sma_slow', sma, period=26)
        
        # Add signal rules (use existing ones from signals_library)
        strategy_builder.add_signal_rule('rsi_signal', overbought_oversold,
                                      indicator='rsi',
                                      overbought=overbought_level,
                                      oversold=oversold_level)
        
        strategy_builder.add_signal_rule('ma_crossover', ma_crossover,
                                      fast_ma='sma_fast',
                                      slow_ma='sma_slow')
        
        # ⚠️ CRITICAL: Set signal combination method
        # Valid methods: 'majority_vote', 'weighted', 'unanimous'
        strategy_builder.set_signal_combination('majority_vote')
        
        # Set strategy information
        strategy_builder.set_strategy_info('Template_Strategy', '1.0.0')
        
        # Build and return the strategy
        strategy = strategy_builder.build()
        
        logger.info(f"✅ Template strategy created successfully!")
        logger.info(f" - Strategy Name: {strategy.name}")
        logger.info(f" - Strategy Symbols: {strategy.symbols}")
        logger.info(f" - Strategy Timeframes: {strategy.timeframes}")
        
        return strategy
        
    except Exception as e:
        logger.error(f"❌ Error creating template strategy: {e}")
        import traceback
        traceback.print_exc()
        raise

class TemplateStrategy(StrategyBase):
    """
    Template Strategy Class - For complex custom logic
    Only create this class if you need custom logic beyond StrategyBuilder
    """
    
    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        """
        Initialize the template strategy
        """
        # CRITICAL: Initialize with EXACT symbols and timeframes provided
        super().__init__(
            name="Template_Strategy",
            symbols=symbols,
            timeframes=timeframes,
            config=config
        )
        
        # Strategy-specific parameters
        self.indicator_period = config.get('indicator_period', 14)
        self.overbought_level = config.get('overbought_level', 70)
        self.oversold_level = config.get('oversold_level', 30)
        
        # Risk management
        self.max_risk_per_trade = 0.02
        self.min_position_size = 0.001
        self.max_position_size = 1.0
        
        # Validate parameters
        self._validate_parameters()
        
        logger.info(f"📈 TemplateStrategy initialized:")
        logger.info(f" - Symbols: {self.symbols}")
        logger.info(f" - Timeframes: {self.timeframes}")
    
    def _validate_parameters(self):
        """Validate strategy parameters"""
        if not (5 <= self.indicator_period <= 50):
            raise ValueError("Indicator period must be between 5 and 50")
        if not (self.oversold_level < self.overbought_level):
            raise ValueError("Oversold level must be less than overbought level")
    
    def calculate_position_size(self, symbol: str, current_price: float = None, signal_strength: float = 1.0) -> float:
        """
        Calculate position size - MUST EXIST
        """
        try:
            position_value = self.balance * self.max_risk_per_trade * signal_strength
            
            if current_price and current_price > 0:
                position_size = position_value / current_price
            else:
                position_size = self.min_position_size
            
            position_size = max(self.min_position_size, min(position_size, self.max_position_size))
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return self.min_position_size
    
    def generate_signals(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, str]]:
        """
        Generate trading signals - MUST EXIST
        """
        signals = {}
        
        try:
            for symbol in data:
                signals[symbol] = {}
                
                for timeframe in data[symbol]:
                    signal = self._generate_single_signal(data[symbol][timeframe], symbol, timeframe)
                    signals[symbol][timeframe] = signal
                    
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
        
        return signals
    
    def _generate_single_signal(self, df: pd.DataFrame, symbol: str, timeframe: str) -> str:
        """
        Generate a single trading signal - MUST EXIST
        """
        try:
            if len(df) < 20:
                return 'HOLD'
            
            # Template strategy logic
            close_prices = df['close']
            
            # Calculate RSI
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.indicator_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.indicator_period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            current_rsi = rsi.iloc[-1]
            
            # Generate signals
            if current_rsi < self.oversold_level:
                return 'BUY'
            elif current_rsi > self.overbought_level:
                return 'SELL'
            else:
                return 'HOLD'
                
        except Exception as e:
            logger.error(f"Error generating signal for {symbol} {timeframe}: {e}")
            return 'HOLD'

def create_template_instance(symbols=None, timeframes=None, **params):
    """Create template strategy instance - OPTIONAL but recommended"""
    try:
        if symbols is None:
            symbols = ['BTCUSDT']
        if timeframes is None:
            timeframes = ['5m']
            
        strategy = TemplateStrategy(symbols, timeframes, params)
        logger.info(f"✅ Template strategy created successfully")
        return strategy
        
    except Exception as e:
        logger.error(f"Error creating strategy: {e}")
        raise

def simple_test():
    """Simple test to verify the strategy works - MUST EXIST"""
    try:
        strategy = create_strategy(
            symbols=['BTCUSDT'],
            timeframes=['5m'],
            indicator_period=14,
            overbought_level=70,
            oversold_level=30
        )
        
        print(f"✅ Template strategy created successfully: {strategy.name}")
        print(f" - Symbols: {strategy.symbols}")
        print(f" - Timeframes: {strategy.timeframes}")
        return True
        
    except Exception as e:
        print(f"❌ Error testing template strategy: {e}")
        return False

# For testing - MUST EXIST
if __name__ == "__main__":
    simple_test()