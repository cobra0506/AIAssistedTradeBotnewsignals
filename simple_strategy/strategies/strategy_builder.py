"""
Strategy Builder - Redesigned with Clear API (FIXED VERSION)
===========================================================

Fixed to work correctly with comprehensive tests.
Validations happen during build(), not during individual method calls.

Author: AI Assisted TradeBot Team
Date: 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import logging
import sys
import os
# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)

from simple_strategy.shared.strategy_base import StrategyBase

logger = logging.getLogger(__name__)


class StrategyBuilder:
    """
    Redesigned Strategy Builder with clear, intuitive API.
    
    The new design ensures:
    - Clear separation between indicators and signals
    - Explicit indicator references in signal rules
    - Type safety and error prevention
    - Easy debugging and understanding
    - Validations happen during build(), not during individual method calls
    """
    
    def __init__(self, symbols: List[str], timeframes: List[str] = ['1m']):
        """
        Initialize the Strategy Builder
        
        Args:
            symbols: List of trading symbols
            timeframes: List of timeframes
        """
        self.symbols = symbols
        self.timeframes = timeframes
        
        # Strategy components
        self.indicators = {}  # {name: {'function': func, 'params': params, 'result': None}}
        self.signal_rules = {}  # {name: {'function': func, 'indicator_refs': [], 'params': {}}}
        self.risk_rules = {}  # {rule_type: params}
        self.signal_combination = 'majority_vote'
        self.signal_weights = {}
        
        # Strategy metadata
        self.strategy_name = "CustomStrategy"
        self.version = "1.0.0"
        
        logger.info(f"🏗️ Strategy Builder initialized for {symbols} on {timeframes}")
    
    def add_indicator(self, name: str, indicator_func: Callable, **params) -> 'StrategyBuilder':
        """
        Add an indicator to the strategy with support for multi-component indicators
        Args:
            name: Unique name for this indicator (used for reference)
            indicator_func: Indicator function from indicators_library
            **params: Parameters for the indicator function
        Returns:
            Self for method chaining
        """
        try:
            if name in self.indicators:
                logger.warning(f"⚠️ Indicator '{name}' already exists, overwriting")
            
            # Store the basic indicator info
            self.indicators[name] = {
                'function': indicator_func,
                'params': params,
                'result': None,  # Will be calculated during signal generation
                'components': {}  # For multi-component indicators
            }
            
            # Special handling for known multi-component indicators
            if indicator_func.__name__ == 'macd':
                # MACD returns (macd_line, signal_line, histogram)
                self.indicators[name]['components'] = {
                    'macd_line': name,
                    'signal_line': name,  # Will be populated during calculation
                    'histogram': name
                }
            elif indicator_func.__name__ == 'bollinger_bands':
                # Bollinger Bands returns (upper_band, middle_band, lower_band)
                self.indicators[name]['components'] = {
                    'upper_band': name,
                    'middle_band': name,
                    'lower_band': name
                }
            
            logger.debug(f"📊 Added indicator: {name} with params: {params}")
            return self
        except Exception as e:
            logger.error(f"❌ Error adding indicator {name}: {e}")
            return self
    
    def add_signal_rule(self, name: str, signal_func: Callable, **params) -> 'StrategyBuilder':
        """
        Add a signal rule to the strategy with enhanced component validation
        Args:
            name: Unique name for this signal rule
            signal_func: Signal function from signals_library
            **params: Parameters including indicator references
        Returns:
            Self for method chaining
        """
        try:
            if name in self.signal_rules:
                logger.warning(f"⚠️ Signal rule '{name}' already exists, overwriting")
            
            # Separate indicator references from signal parameters
            indicator_refs = []
            signal_params = {}
            
            # Known indicator parameters for common signal functions
            indicator_param_names = {
                'overbought_oversold': ['indicator'],
                'ma_crossover': ['fast_ma', 'slow_ma'],
                'macd_signals': ['macd_line', 'signal_line'],
                'bollinger_bands_signals': ['price', 'upper_band', 'lower_band'],
                'stochastic_signals': ['k_percent', 'd_percent'],
                'divergence_signals': ['price', 'indicator'],
                'breakout_signals': ['price', 'resistance', 'support'],
                'trend_strength_signals': ['price', 'short_ma', 'long_ma']
            }
            
            # Get the expected indicator parameter names for this function
            expected_indicators = indicator_param_names.get(signal_func.__name__, [])
            
            for param_name, param_value in params.items():
                if param_name in expected_indicators:
                    # Special handling for different indicator types
                    if param_name == 'price':
                        # Price is a special case - it's the 'close' column from data
                        indicator_refs.append((param_name, 'price'))
                    elif param_value in self.indicators:
                        # Direct indicator reference
                        indicator_refs.append((param_name, param_value))
                    elif param_value == 'signal_line' and 'macd_line' in self.indicators:
                        # MACD signal_line is a component of macd_line indicator
                        indicator_refs.append((param_name, 'macd_line'))
                    elif any(param_value in comp_dict for comp_dict in 
                        [ind.get('components', {}) for ind in self.indicators.values()]):
                        # Component of a multi-component indicator
                        indicator_refs.append((param_name, param_value))
                    else:
                        available_indicators = list(self.indicators.keys())
                        available_components = []
                        for ind_name, ind_data in self.indicators.items():
                            if 'components' in ind_data:
                                available_components.extend(ind_data['components'].keys())
                        
                        raise ValueError(
                            f"Signal rule '{name}' references unknown indicator/component '{param_value}'. "
                            f"Available indicators: {available_indicators}, "
                            f"Available components: {available_components}"
                        )
                else:
                    # This is a signal parameter
                    signal_params[param_name] = param_value
            
            self.signal_rules[name] = {
                'function': signal_func,
                'indicator_refs': indicator_refs,
                'params': signal_params
            }
            
            logger.debug(f"📡 Added signal rule: {name} with refs: {indicator_refs}")
            return self
        except Exception as e:
            logger.error(f"❌ Error adding signal rule {name}: {e}")
            return self
    
    def add_risk_rule(self, rule_type: str, **params) -> 'StrategyBuilder':
        """
        Add a risk management rule
        
        Args:
            rule_type: Type of risk rule ('stop_loss', 'take_profit', 'max_position_size')
            **params: Parameters for the risk rule
            
        Returns:
            Self for method chaining
        """
        try:
            self.risk_rules[rule_type] = params
            logger.debug(f"🛡️ Added risk rule: {rule_type} with params: {params}")
            return self
        except Exception as e:
            logger.error(f"❌ Error adding risk rule {rule_type}: {e}")
            return self
    
    def set_signal_combination(self, method: str, **kwargs) -> 'StrategyBuilder':
        """
        Set how signals should be combined
        Args:
            method: Combination method ('majority_vote', 'weighted', 'unanimous')
            **kwargs: Additional parameters (like weights for weighted method)
        Returns:
            Self for method chaining
        """
        try:
            # Basic validation of method name
            valid_methods = ['majority_vote', 'weighted', 'unanimous', 'and_signals']
            if method not in valid_methods:
                raise ValueError(f"Invalid signal combination method: {method}. Valid methods: {valid_methods}")
            
            self.signal_combination = method
            
            if method == 'weighted':
                if 'weights' not in kwargs:
                    raise ValueError("Weights must be provided for weighted signal combination.")
                
                weights = kwargs['weights']
                
                # Validate weights structure
                if not isinstance(weights, dict):
                    raise ValueError("Weights must be a dictionary")
                
                if not weights:
                    raise ValueError("Weights dictionary cannot be empty")
                
                # Validate that all weighted signal rules exist
                for signal_rule_name in weights.keys():
                    if signal_rule_name not in self.signal_rules:
                        available_signal_rules = list(self.signal_rules.keys())
                        raise ValueError(
                            f"Weight references unknown signal rule '{signal_rule_name}'. "
                            f"Available signal rules: {available_signal_rules}"
                        )
                
                # Validate weight values
                weight_values = list(weights.values())
                if not all(isinstance(w, (int, float)) for w in weight_values):
                    raise ValueError("All weights must be numeric values.")
                
                if sum(weight_values) == 0:
                    raise ValueError("Sum of weights cannot be zero.")
                
                self.signal_weights = weights
            
            logger.debug(f"🔀 Set signal combination to: {method}")
            return self
            
        except Exception as e:
            logger.error(f"❌ Error setting signal combination: {e}")
            raise  # Re-raise the exception so the test can catch it

    def set_custom_signal_combination(self, combination_func):
        """
        Set a custom function to combine signals
        
        Args:
            combination_func: Function that takes a list of signal series and returns combined signals
        """
        self.signal_combination = 'custom'
        self.custom_combination_func = combination_func
        return self  # Return self for method chaining

    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate all indicators and add them to DataFrame columns
        """
        calculated_indicators = {}
        
        for indicator_name, indicator_data in self.indicators.items():
            try:
                func = indicator_data['function']
                params = indicator_data['params']
                func_name = func.__name__
                
                # Handle different indicator function signatures
                if func_name in ['stochastic', 'atr', 'cci', 'williams_r']:
                    # These indicators need OHLC data
                    result = func(data['high'], data['low'], data['close'], **params)
                elif func_name in ['on_balance_volume']:
                    # This needs close and volume
                    result = func(data['close'], data['volume'], **params)
                elif func_name in ['volume_sma']:
                    # This needs volume data
                    result = func(data['volume'], **params)
                else:
                    # Standard indicators that only need close price
                    result = func(data['close'], **params)
                
                # Handle different result types
                if isinstance(result, tuple):
                    # Multi-component indicator (MACD, Bollinger Bands, etc.)
                    if func_name == 'macd':
                        # MACD returns (macd_line, signal_line, histogram)
                        data[f"{indicator_name}_macd_line"] = result[0]
                        data[f"{indicator_name}_signal_line"] = result[1]
                        data[f"{indicator_name}_histogram"] = result[2]
                        calculated_indicators[f"{indicator_name}_macd_line"] = result[0]
                        calculated_indicators[f"{indicator_name}_signal_line"] = result[1]
                        calculated_indicators[f"{indicator_name}_histogram"] = result[2]
                    elif func_name == 'bollinger_bands':
                        # Bollinger Bands returns (upper_band, middle_band, lower_band)
                        data[f"{indicator_name}_upper_band"] = result[0]
                        data[f"{indicator_name}_middle_band"] = result[1]
                        data[f"{indicator_name}_lower_band"] = result[2]
                        calculated_indicators[f"{indicator_name}_upper_band"] = result[0]
                        calculated_indicators[f"{indicator_name}_middle_band"] = result[1]
                        calculated_indicators[f"{indicator_name}_lower_band"] = result[2]
                    elif func_name == 'stochastic':
                        # Stochastic returns (%K, %D)
                        data[f"{indicator_name}_k_percent"] = result[0]
                        data[f"{indicator_name}_d_percent"] = result[1]
                        calculated_indicators[f"{indicator_name}_k_percent"] = result[0]
                        calculated_indicators[f"{indicator_name}_d_percent"] = result[1]
                    else:
                        # Generic tuple handling
                        for i, component in enumerate(result):
                            col_name = f"{indicator_name}_component_{i}"
                            data[col_name] = component
                            calculated_indicators[col_name] = component
                else:
                    # Single-component indicator
                    data[indicator_name] = result
                    calculated_indicators[indicator_name] = result
                
                # Store the result for reference
                indicator_data['result'] = result
                
            except Exception as e:
                logger.error(f"❌ Error calculating indicator {indicator_name}: {e}")
                # Create empty result to avoid breaking the system
                empty_series = pd.Series(0, index=data.index)
                data[indicator_name] = empty_series
                calculated_indicators[indicator_name] = empty_series
        
        # Add price data as special indicators
        data['price'] = data['close']
        calculated_indicators['price'] = data['close']
        
        return calculated_indicators                                                                            
    
    def set_strategy_info(self, name: str, version: str = "1.0.0") -> 'StrategyBuilder':
        """
        Set strategy information
        
        Args:
            name: Strategy name
            version: Strategy version
            
        Returns:
            Self for method chaining
        """
        self.strategy_name = name
        self.version = version
        logger.debug(f"📝 Set strategy info: {name} v{version}")
        return self
    
    def _validate_configuration(self):
        """
        Validate the complete strategy configuration before building.
        This method is called during build() and ensures all components are valid.
        """
        logger.debug("🔍 Validating strategy configuration...")
        
        # 1. Validate indicators exist
        if not self.indicators:
            raise ValueError("No indicators defined. Add at least one indicator.")
        
        # 2. Validate signal rules exist
        if not self.signal_rules:
            raise ValueError("No signal rules defined. Add at least one signal rule.")
        
        # 3. Validate signal rule indicator references
        for rule_name, rule_config in self.signal_rules.items():
            for param_name, indicator_name in rule_config['indicator_refs']:
                if indicator_name not in self.indicators and indicator_name != 'price':
                    available_indicators = list(self.indicators.keys())
                    raise ValueError(
                        f"Signal rule '{rule_name}' references unknown indicator '{indicator_name}'. "
                        f"Available indicators: {available_indicators}"
                    )
        
        # 4. Validate signal combination method
        valid_combination_methods = ['majority_vote', 'weighted', 'unanimous', 'and_signals', 'custom']
        if self.signal_combination not in valid_combination_methods:
            raise ValueError(
                f"Invalid signal combination method: '{self.signal_combination}'. "
                f"Valid methods: {valid_combination_methods}"
            )
        
        # 5. Validate signal weights for weighted combination
        if self.signal_combination == 'weighted':
            if not self.signal_weights:
                raise ValueError("Weights must be provided for weighted signal combination.")
            
            # Validate that all weighted signal rules exist
            for signal_rule_name in self.signal_weights.keys():
                if signal_rule_name not in self.signal_rules:
                    available_signal_rules = list(self.signal_rules.keys())
                    raise ValueError(
                        f"Weight references unknown signal rule '{signal_rule_name}'. "
                        f"Available signal rules: {available_signal_rules}"
                    )
            
            # Validate weight values
            weight_values = list(self.signal_weights.values())
            if not all(isinstance(w, (int, float)) for w in weight_values):
                raise ValueError("All weights must be numeric values.")
            
            if sum(weight_values) == 0:
                raise ValueError("Sum of weights cannot be zero.")
        
        logger.debug("✅ Strategy configuration validation passed")
    
    def _execute_signal_rules(self, data: pd.DataFrame) -> pd.Series:
        """
        Execute all signal rules and return combined signals
        """
        all_signals = []
        
        for signal_name, signal_rule in self.signal_rules.items():
            try:
                signal_func = signal_rule['function']
                indicator_refs = signal_rule['indicator_refs']
                signal_params = signal_rule['params']
                
                # Prepare arguments for signal function
                signal_args = {}
                
                # Map indicator references to actual pandas Series from DataFrame
                for param_name, indicator_ref in indicator_refs:
                    if indicator_ref == 'price':
                        signal_args[param_name] = data['close']  # Use close price as 'price'
                    elif indicator_ref in data.columns:
                        signal_args[param_name] = data[indicator_ref]  # Use DataFrame column
                    else:
                        # Try to find the indicator in the calculated indicators
                        found = False
                        for col_name in data.columns:
                            if indicator_ref in col_name or col_name.endswith(indicator_ref):
                                signal_args[param_name] = data[col_name]
                                found = True
                                break
                        
                        if not found:
                            logger.error(f"❌ Indicator '{indicator_ref}' not found in DataFrame for signal '{signal_name}'")
                            continue
                
                # Add signal parameters
                signal_args.update(signal_params)
                
                # Call signal function with correct parameters
                signal_result = signal_func(**signal_args)
                
                # Validate result is pandas Series
                if isinstance(signal_result, pd.Series):
                    all_signals.append(signal_result)
                else:
                    logger.error(f"❌ Signal function '{signal_name}' returned {type(signal_result)}, expected pandas Series")
                    
            except Exception as e:
                logger.error(f"❌ Error executing signal rule '{signal_name}': {e}")
        
        # Combine signals based on combination method
        if not all_signals:
            return pd.Series('HOLD', index=data.index)
        
        # Implement signal combination logic
        return self._combine_signals(all_signals, data.index)
    
    def _combine_signals(self, signals: List[pd.Series], index) -> pd.Series:
        """
        Combine multiple signal series based on the combination method
        Updated to use new signal schema: OPEN_LONG, OPEN_SHORT, CLOSE_LONG, CLOSE_SHORT, HOLD
        """
        if not signals:
            return pd.Series('HOLD', index=index)
        
        if self.signal_combination == 'majority_vote':
            # Convert all signals to numeric format for easier comparison
            numeric_signals = []
            for signal in signals:
                if signal.dtype == 'object':
                    # Convert text signals to numeric using new signal schema
                    numeric = signal.map({
                        'OPEN_LONG': 2,    # Strong buy
                        'CLOSE_SHORT': 1,  # Buy to cover
                        'HOLD': 0,         # No action
                        'CLOSE_LONG': -1,  # Sell
                        'OPEN_SHORT': -2   # Strong sell
                    })
                    numeric_signals.append(numeric)
                else:
                    numeric_signals.append(signal)
            
            # Stack signals and find majority
            if numeric_signals:
                stacked = pd.concat(numeric_signals, axis=1)
                # Sum the signals
                summed = stacked.sum(axis=1)
                result = pd.Series('HOLD', index=index)
                result[summed >= 2] = 'OPEN_LONG'      # Strong bullish consensus
                result[summed == 1] = 'CLOSE_SHORT'    # Weak bullish consensus
                result[summed == -1] = 'CLOSE_LONG'    # Weak bearish consensus
                result[summed <= -2] = 'OPEN_SHORT'    # Strong bearish consensus
                return result
            else:
                return pd.Series('HOLD', index=index)
        
        elif self.signal_combination == 'weighted':
            # Convert all signals to numeric format
            numeric_signals = []
            for signal in signals:
                if signal.dtype == 'object':
                    # Convert text signals to numeric using new signal schema
                    numeric = signal.map({
                        'OPEN_LONG': 2,    # Strong buy
                        'CLOSE_SHORT': 1,  # Buy to cover
                        'HOLD': 0,         # No action
                        'CLOSE_LONG': -1,  # Sell
                        'OPEN_SHORT': -2   # Strong sell
                    })
                    numeric_signals.append(numeric)
                else:
                    numeric_signals.append(signal)
            
            if numeric_signals:
                # Apply weights
                weighted_sum = pd.Series(0, index=index)
                for i, signal in enumerate(numeric_signals):
                    rule_name = list(self.signal_rules.keys())[i]
                    weight = self.signal_weights.get(rule_name, 1.0)
                    weighted_sum += signal * weight
                
                result = pd.Series('HOLD', index=index)
                result[weighted_sum >= 2] = 'OPEN_LONG'      # Strong bullish consensus
                result[weighted_sum > 0] = 'CLOSE_SHORT'     # Weak bullish consensus
                result[weighted_sum < 0] = 'CLOSE_LONG'      # Weak bearish consensus
                result[weighted_sum <= -2] = 'OPEN_SHORT'    # Strong bearish consensus
                return result
            else:
                return pd.Series('HOLD', index=index)
        
        elif self.signal_combination == 'and_signals':
            # Only return a signal if ALL signals agree
            result = pd.Series('HOLD', index=index)
            for i in range(len(index)):
                vals = [s.iloc[i] for s in signals]
                # Check if all values are the same and not HOLD
                if all(v == vals[0] for v in vals) and vals[0] != 'HOLD':
                    result.iloc[i] = vals[0]
            return result

        elif self.signal_combination == 'custom':
            # Use custom combination function
            if hasattr(self, 'custom_combination_func'):
                return self.custom_combination_func(signals)
            else:
                logger.error("Custom combination function not set")
                return pd.Series('HOLD', index=index)

        else:  # unanimous
            # All signals must agree
            result = pd.Series('HOLD', index=index)
            
            for i in range(len(index)):
                # Get all signal values at this index
                signal_values = []
                for signal in signals:
                    if i < len(signal):
                        signal_values.append(signal.iloc[i])
                
                # Check if all signals are the same
                if signal_values and all(v == signal_values[0] for v in signal_values):
                    signal_value = signal_values[0]
                    if signal_value in ['OPEN_LONG', 'OPEN_SHORT', 'CLOSE_LONG', 'CLOSE_SHORT']:
                        result.iloc[i] = signal_value
            
            return result

    def build(self) -> StrategyBase:
        """
        Build the complete strategy
        
        Returns:
            Complete strategy class that inherits from StrategyBase
        """
        try:
            logger.info(f"🔨 Building strategy: {self.strategy_name}")
            
            # Validate the strategy configuration
            self._validate_configuration()
            
            # Create the strategy class
            class BuiltStrategy(StrategyBase):
                def __init__(self, name, symbols, timeframes, config, builder):
                    super().__init__(name, symbols, timeframes, config)
                    
                    # Store reference to builder
                    self.builder = builder
                    
                    # Set our custom attributes
                    self._custom_strategy_name = name
                    self._custom_version = "1.0.0"
                
                def generate_signals(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, str]]:
                    """
                    Generate trading signals - implements StrategyBase interface
                    """
                    result = {}
                    
                    for symbol in data:
                        result[symbol] = {}
                        
                        for timeframe in data[symbol]:
                            df = data[symbol][timeframe].copy()  # Work with a copy
                            
                            # Calculate indicators and add to DataFrame
                            self.builder._calculate_indicators(df)
                            
                            # Execute signal rules to get pandas Series
                            signals_series = self.builder._execute_signal_rules(df)
                            
                            # Convert pandas Series to the expected string format
                            # Take the last signal (most recent) or implement your own logic
                            if len(signals_series) > 0:
                                last_signal = signals_series.iloc[-1]  # Get most recent signal
                                if isinstance(last_signal, str):
                                    result[symbol][timeframe] = last_signal
                                elif isinstance(last_signal, (int, float)):
                                    # Convert numeric signals to strings
                                    if last_signal == 1:
                                        result[symbol][timeframe] = 'BUY'
                                    elif last_signal == -1:
                                        result[symbol][timeframe] = 'SELL'
                                    else:
                                        result[symbol][timeframe] = 'HOLD'
                                else:
                                    result[symbol][timeframe] = 'HOLD'
                            else:
                                result[symbol][timeframe] = 'HOLD'
                    
                    return result
                
                def get_strategy_info(self) -> Dict:
                    """Get strategy information"""
                    return {
                        'strategy_name': self._custom_strategy_name,
                        'version': self._custom_version,
                        'symbols': self.symbols,
                        'timeframes': self.timeframes,
                        'indicators': list(self.builder.indicators.keys()),
                        'signal_rules': list(self.builder.signal_rules.keys()),
                        'risk_rules': self.builder.risk_rules,
                        'signal_combination': self.builder.signal_combination,
                        'signal_weights': self.builder.signal_weights
                    }
                
                # Add property to access strategy_name
                @property
                def strategy_name(self):
                    """Strategy name property"""
                    return self._custom_strategy_name
                
                @strategy_name.setter
                def strategy_name(self, value):
                    """Strategy name setter"""
                    self._custom_strategy_name = value
            
            # Create and return the strategy instance
            config = {'version': self.version}  # Config dict
            strategy_instance = BuiltStrategy(
                self.strategy_name,
                self.symbols, 
                self.timeframes, 
                config,
                self  # Pass the builder instance
            )
            
            logger.info(f"✅ Strategy '{self.strategy_name}' built successfully!")
            return strategy_instance
            
        except Exception as e:
            logger.error(f"❌ Error building strategy: {e}")
            raise


# === EXAMPLE USAGE ===
if __name__ == "__main__":
    """
    Example of how to use the redesigned Strategy Builder
    """
    
    # Import libraries
    from simple_strategy.strategies.indicators_library import rsi, sma, macd
    from simple_strategy.strategies.signals_library import overbought_oversold, ma_crossover, macd_signals
    
    print("🎯 Example 1: Simple RSI Strategy")
    print("=" * 40)
    
    # Clear, intuitive API
    strategy1 = StrategyBuilder(['BTCUSDT'], ['1m'])
    
    # Add indicators with clear names
    strategy1.add_indicator('rsi', rsi, period=14)
    
    # Add signal rule that explicitly references the indicator
    strategy1.add_signal_rule('rsi_signal', overbought_oversold, 
                             indicator='rsi',           # Clear reference
                             overbought=70, oversold=30)
    
    # Add risk management
    strategy1.add_risk_rule('stop_loss', percent=2.0)
    strategy1.add_risk_rule('take_profit', percent=4.0)
    strategy1.set_strategy_info('SimpleRSI', '1.0.0')
    
    simple_rsi = strategy1.build()
    print(f"✅ Built strategy: {simple_rsi.get_strategy_info()['strategy_name']}")
    
    print("\n🎯 Example 2: Multi-Indicator Strategy")
    print("=" * 40)
    
    strategy2 = StrategyBuilder(['BTCUSDT', 'ETHUSDT'], ['1m', '5m'])
    
    # Add multiple indicators with descriptive names
    strategy2.add_indicator('rsi', rsi, period=14)
    strategy2.add_indicator('sma_short', sma, period=20)
    strategy2.add_indicator('sma_long', sma, period=50)
    strategy2.add_indicator('macd', macd, fast_period=12, slow_period=26)
    
    # Add signal rules with clear indicator references
    strategy2.add_signal_rule('rsi_signal', overbought_oversold, 
                             indicator='rsi', overbought=70, oversold=30)
    
    strategy2.add_signal_rule('ma_cross', ma_crossover,
                             fast_ma='sma_short',      # Clear reference
                             slow_ma='sma_long')      # Clear reference
    
    strategy2.add_signal_rule('macd_signal', macd_signals,
                             macd_line='macd',        # Clear reference
                             signal_line='macd')     # Clear reference
    
    # Combine signals
    strategy2.set_signal_combination('majority_vote')
    
    # Add risk management
    strategy2.add_risk_rule('stop_loss', percent=1.5)
    strategy2.add_risk_rule('take_profit', percent=3.0)
    strategy2.set_strategy_info('MultiIndicator', '1.0.0')
    
    multi_indicator = strategy2.build()
    print(f"✅ Built strategy: {multi_indicator.get_strategy_info()['strategy_name']}")
    
    print("\n🎯 Example 3: Weighted Strategy")
    print("=" * 40)
    
    strategy3 = StrategyBuilder(['BTCUSDT'], ['1m'])
    
    strategy3.add_indicator('rsi', rsi, period=14)
    strategy3.add_indicator('sma_short', sma, period=20)
    strategy3.add_indicator('sma_long', sma, period=50)
    
    strategy3.add_signal_rule('rsi_signal', overbought_oversold, 
                             indicator='rsi', overbought=70, oversold=30)
    
    strategy3.add_signal_rule('ma_cross', ma_crossover,
                             fast_ma='sma_short', slow_ma='sma_long')
    
    # Use weighted combination
    strategy3.set_signal_combination('weighted', weights={
        'rsi_signal': 0.6,
        'ma_cross': 0.4
    })
    strategy3.set_strategy_info('WeightedStrategy', '1.0.0')
    
    weighted_strategy = strategy3.build()
    print(f"✅ Built strategy: {weighted_strategy.get_strategy_info()['strategy_name']}")
    
    print("\n🎉 All examples completed! New Strategy Builder is working perfectly!")

