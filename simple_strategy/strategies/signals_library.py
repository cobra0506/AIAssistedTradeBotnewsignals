"""
Building Block Signals Library (CORRECTED VERSION)
===============================================

This library contains ALL signal processing functions that can be used in strategy building.
Each function processes indicators and returns trading signals using the correct signal types
expected by the paper trading engine.

FIXED: All functions now return 'OPEN_LONG', 'OPEN_SHORT', 'CLOSE_LONG', 'CLOSE_SHORT', or 'HOLD'

Author: AI Assisted TradeBot Team
Date: 2025
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


# === BASIC SIGNAL FUNCTIONS ===

def crossover(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """
    Detects when series1 crosses **above** series2.
    
    Returns a boolean Series: True where crossover occurs.
    """
    return (series1 > series2) & (series1.shift(1) <= series2.shift(1))


def crossunder(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """
    Detects when series1 crosses **below** series2.
    
    Returns a boolean Series: True where crossunder occurs.
    """
    return (series1 < series2) & (series1.shift(1) >= series2.shift(1))


def oversold_cross(series: pd.Series, level: float) -> pd.Series:
    """
    Detects when series crosses **below** the oversold level.
    
    Useful for long entry signals.
    """
    return (series < level) & (series.shift(1) >= level)


def overbought_cross(series: pd.Series, level: float) -> pd.Series:
    """
    Detects when series crosses **above** the overbought level.
    
    Useful for short entry signals.
    """
    return (series > level) & (series.shift(1) <= level)

def overbought_oversold(indicator, overbought=70, oversold=30):
    """
    Generate overbought/oversold signals
    
    FIXED: Now returns correct signal types
    
    Args:
        indicator: Indicator series (RSI, Stochastic, etc.)
        overbought: Overbought threshold
        oversold: Oversold threshold
        
    Returns:
        Series with 'OPEN_LONG', 'OPEN_SHORT', 'CLOSE_LONG', 'CLOSE_SHORT', or 'HOLD' signals
    """
    # Create a series with default HOLD values
    signals = pd.Series('HOLD', index=indicator.index)
    
    # For simplicity, we'll use OPEN/CLOSE signals based on indicator direction
    # In a real strategy, you'd want to track current positions
    signals[indicator < oversold] = 'OPEN_LONG'   # Oversold - buy signal
    signals[indicator > overbought] = 'OPEN_SHORT'  # Overbought - sell signal
    
    return signals

def overbought_oversold_crossover(indicator, overbought=70, oversold=30):
    """
    Generate overbought/oversold crossover signals
    
    FIXED: Now returns correct signal types
    
    Args:
        indicator: Indicator series (RSI, Stochastic, etc.)
        overbought: Overbought threshold
        oversold: Oversold threshold
        
    Returns:
        Series with 'OPEN_LONG', 'OPEN_SHORT', 'CLOSE_LONG', 'CLOSE_SHORT', or 'HOLD' signals
    """
    signals = pd.Series('HOLD', index=indicator.index)
    
    # Check for crossover from below to above overbought - potential short entry
    signals[(indicator > overbought) & (indicator.shift(1) <= overbought)] = 'OPEN_SHORT'
    
    # Check for crossover from above to below oversold - potential long entry
    signals[(indicator < oversold) & (indicator.shift(1) >= oversold)] = 'OPEN_LONG'
    
    # Check for return from overbought - close short
    signals[(indicator < overbought) & (indicator.shift(1) >= overbought)] = 'CLOSE_SHORT'
    
    # Check for return from oversold - close long
    signals[(indicator > oversold) & (indicator.shift(1) <= oversold)] = 'CLOSE_LONG'
    
    return signals

def overbought_oversold_with_trend(indicator_rsi, indicator_sma, overbought=70, oversold=30):
    """
    Generate overbought/oversold signals with trend filter
    
    FIXED: Now returns correct signal types
    
    Args:
        indicator_rsi: RSI indicator series
        indicator_sma: SMA indicator series for trend
        overbought: Overbought threshold
        oversold: Oversold threshold
        
    Returns:
        Series with 'OPEN_LONG', 'OPEN_SHORT', 'CLOSE_LONG', 'CLOSE_SHORT', or 'HOLD' signals
    """
    # Create a series with default HOLD values
    signals = pd.Series('HOLD', index=indicator_rsi.index)
    
    # Get current values
    current_rsi = indicator_rsi
    current_sma = indicator_sma
    previous_sma = current_sma.shift(1)  # Previous SMA value
    
    # Determine trend direction: price going up if current SMA > previous SMA
    uptrend = current_sma > previous_sma
    
    # Generate signals based on trend
    # In uptrend, we only go long
    buy_condition = (current_rsi < oversold) & uptrend
    sell_condition = (current_rsi > overbought) & uptrend
    
    signals[buy_condition] = 'OPEN_LONG'
    signals[sell_condition] = 'CLOSE_LONG'
    
    # In downtrend, we only go short
    short_condition = (current_rsi > overbought) & (~uptrend)
    cover_condition = (current_rsi < oversold) & (~uptrend)
    
    signals[short_condition] = 'OPEN_SHORT'
    signals[cover_condition] = 'CLOSE_SHORT'
    
    return signals

def trend_signal(ema_fast: pd.Series, ema_slow: pd.Series) -> pd.Series:
    """
    Generate trend signals based on EMA crossover.
    
    FIXED: Now returns correct signal types
    
    Args:
        ema_fast: Fast EMA series
        ema_slow: Slow EMA series
        
    Returns:
        Series with 'OPEN_LONG', 'OPEN_SHORT', 'CLOSE_LONG', 'CLOSE_SHORT', or 'HOLD' signals
    """
    try:
        # Create a series with default HOLD values
        signals = pd.Series('HOLD', index=ema_fast.index)
        
        # Generate OPEN_LONG when fast EMA crosses above slow EMA
        buy_signals = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
        signals[buy_signals] = 'OPEN_LONG'
        
        # Generate OPEN_SHORT when fast EMA crosses below slow EMA
        sell_signals = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))
        signals[sell_signals] = 'OPEN_SHORT'
        
        # Note: This simple trend signal doesn't generate close signals
        # Close signals would typically come from other indicators or conditions
        
        return signals
    except Exception as e:
        logger.error(f"Error in trend_signal: {e}")
        return pd.Series('HOLD', index=ema_fast.index)

def rsi_mean_reversion_with_trend(rsi, ema_fast, ema_slow, overbought=70, oversold=30):
    """
    RSI Mean Reversion with EMA Trend Filter - Complete implementation
    
    This is the complete implementation of your requested strategy:
    
    - Uptrend (EMA fast > EMA slow):
      - RSI crosses down below 70 → CLOSE_LONG
      - RSI crosses up above 30 → OPEN_LONG
    
    - Downtrend (EMA fast < EMA slow):
      - RSI crosses up above 70 → OPEN_SHORT
      - RSI crosses down below 30 → CLOSE_SHORT
    
    Args:
        rsi: RSI indicator series
        ema_fast: Fast EMA series
        ema_slow: Slow EMA series
        overbought: Overbought threshold (default: 70)
        oversold: Oversold threshold (default: 30)
        
    Returns:
        Series with 'OPEN_LONG', 'OPEN_SHORT', 'CLOSE_LONG', 'CLOSE_SHORT', or 'HOLD' signals
    """
    signals = pd.Series('HOLD', index=rsi.index)
    
    # Determine trend for each point
    trend = np.where(ema_fast > ema_slow, 'UPTREND', 'DOWNTREND')
    
    # Calculate RSI crossovers
    rsi_cross_above_oversold = (rsi > oversold) & (rsi.shift(1) <= oversold)
    rsi_cross_below_overbought = (rsi < overbought) & (rsi.shift(1) >= overbought)
    rsi_cross_above_overbought = (rsi > overbought) & (rsi.shift(1) <= overbought)
    rsi_cross_below_oversold = (rsi < oversold) & (rsi.shift(1) >= oversold)
    
    # UPTREND logic
    uptrend_mask = pd.Series(trend == 'UPTREND', index=rsi.index)
    signals[uptrend_mask & rsi_cross_above_oversold] = 'OPEN_LONG'
    signals[uptrend_mask & rsi_cross_below_overbought] = 'CLOSE_LONG'
    
    # DOWNTREND logic
    downtrend_mask = pd.Series(trend == 'DOWNTREND', index=rsi.index)
    signals[downtrend_mask & rsi_cross_above_overbought] = 'OPEN_SHORT'
    signals[downtrend_mask & rsi_cross_below_oversold] = 'CLOSE_SHORT'
    
    return signals

def ma_crossover(fast_ma, slow_ma):
    """
    Generate MA crossover signals
    
    FIXED: Now returns correct signal types
    
    Args:
        fast_ma: Fast moving average series
        slow_ma: Slow moving average series
        
    Returns:
        Series with 'OPEN_LONG', 'OPEN_SHORT', 'CLOSE_LONG', 'CLOSE_SHORT', or 'HOLD' signals
    """
    # Create a series with default HOLD values
    signals = pd.Series('HOLD', index=fast_ma.index)
    
    # Generate OPEN_LONG when fast MA crosses above slow MA
    buy_signals = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    signals[buy_signals] = 'OPEN_LONG'
    
    # Generate OPEN_SHORT when fast MA crosses below slow MA
    sell_signals = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
    signals[sell_signals] = 'OPEN_SHORT'
    
    # Generate close signals on reverse crossover
    # Close long when fast crosses below slow
    close_long = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
    signals[close_long] = 'CLOSE_LONG'
    
    # Close short when fast crosses above slow
    close_short = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    signals[close_short] = 'CLOSE_SHORT'
    
    return signals

def macd_signals(macd_line: pd.Series, signal_line: pd.Series, 
                 histogram: pd.Series = None) -> pd.Series:
    """
    Generate MACD-based signals
    
    FIXED: Now returns correct signal types
    
    Args:
        macd_line: MACD line
        signal_line: Signal line
        histogram: MACD histogram (optional)
        
    Returns:
        Series with 'OPEN_LONG', 'OPEN_SHORT', 'CLOSE_LONG', 'CLOSE_SHORT', or 'HOLD' signals
    """
    try:
        signals = pd.Series('HOLD', index=macd_line.index)
        
        # OPEN_LONG when MACD crosses above signal line
        buy_signals = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        signals[buy_signals] = 'OPEN_LONG'
        
        # OPEN_SHORT when MACD crosses below signal line
        sell_signals = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        signals[sell_signals] = 'OPEN_SHORT'
        
        # CLOSE_LONG when MACD crosses below signal line
        close_long = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        signals[close_long] = 'CLOSE_LONG'
        
        # CLOSE_SHORT when MACD crosses above signal line
        close_short = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        signals[close_short] = 'CLOSE_SHORT'
        
        return signals
    except Exception as e:
        logger.error(f"Error in macd_signals: {e}")
        return pd.Series('HOLD', index=macd_line.index)

def bollinger_bands_signals(price: pd.Series, upper_band: pd.Series, 
                           lower_band: pd.Series, middle_band: pd.Series = None) -> pd.Series:
    """
    Generate Bollinger Bands signals
    
    FIXED: Now returns correct signal types
    
    Args:
        price: Price series
        upper_band: Upper Bollinger Band
        lower_band: Lower Bollinger Band
        middle_band: Middle Bollinger Band (optional)
        
    Returns:
        Series with 'OPEN_LONG', 'OPEN_SHORT', 'CLOSE_LONG', 'CLOSE_SHORT', or 'HOLD' signals
    """
    try:
        signals = pd.Series('HOLD', index=price.index)
        
        # OPEN_LONG when price crosses below lower band
        buy_signals = (price < lower_band) & (price.shift(1) >= lower_band.shift(1))
        signals[buy_signals] = 'OPEN_LONG'
        
        # OPEN_SHORT when price crosses above upper band
        sell_signals = (price > upper_band) & (price.shift(1) <= upper_band.shift(1))
        signals[sell_signals] = 'OPEN_SHORT'
        
        # CLOSE_LONG when price returns to middle band
        if middle_band is not None:
            close_long = (price > middle_band) & (price.shift(1) <= middle_band.shift(1))
            signals[close_long] = 'CLOSE_LONG'
            
            # CLOSE_SHORT when price returns to middle band
            close_short = (price < middle_band) & (price.shift(1) >= middle_band.shift(1))
            signals[close_short] = 'CLOSE_SHORT'
        
        return signals
    except Exception as e:
        logger.error(f"Error in bollinger_bands_signals: {e}")
        return pd.Series('HOLD', index=price.index)

def stochastic_signals(k_percent: pd.Series, d_percent: pd.Series,
                     overbought: float=80, oversold: float=20) -> pd.Series:
    """
    Generate Stochastic signals
    
    FIXED: Now returns correct signal types
    
    Args:
        k_percent: %K line
        d_percent: %D line
        overbought: Overbought threshold
        oversold: Oversold threshold
        
    Returns:
        Series with 'OPEN_LONG', 'OPEN_SHORT', 'CLOSE_LONG', 'CLOSE_SHORT', or 'HOLD' signals
    """
    try:
        # Create a series with default HOLD values
        signals = pd.Series('HOLD', index=k_percent.index)
        
        # Generate OPEN_LONG when both %K and %D cross above oversold
        buy_signals = ((k_percent > oversold) & (k_percent.shift(1) <= oversold) &
                      (d_percent > oversold) & (d_percent.shift(1) <= oversold))
        signals[buy_signals] = 'OPEN_LONG'
        
        # Generate OPEN_SHORT when both %K and %D cross below overbought
        sell_signals = ((k_percent < overbought) & (k_percent.shift(1) >= overbought) &
                       (d_percent < overbought) & (d_percent.shift(1) >= overbought))
        signals[sell_signals] = 'OPEN_SHORT'
        
        # Generate CLOSE_LONG when %K and %D cross below overbought
        close_long = ((k_percent < overbought) & (k_percent.shift(1) >= overbought) &
                     (d_percent < overbought) & (d_percent.shift(1) >= overbought))
        signals[close_long] = 'CLOSE_LONG'
        
        # Generate CLOSE_SHORT when %K and %D cross above oversold
        close_short = ((k_percent > oversold) & (k_percent.shift(1) <= oversold) &
                      (d_percent > oversold) & (d_percent.shift(1) <= oversold))
        signals[close_short] = 'CLOSE_SHORT'
        
        return signals
    except Exception as e:
        logger.error(f"Error in stochastic_signals: {e}")
        return pd.Series('HOLD', index=k_percent.index)


# === ADVANCED SIGNAL FUNCTIONS ===

def divergence_signals(price: pd.Series, indicator: pd.Series, 
                      lookback_period: int = 20) -> pd.Series:
    """
    Generate divergence signals
    
    FIXED: Now returns correct signal types
    
    Args:
        price: Price series
        indicator: Indicator series
        lookback_period: Period to check for divergence
        
    Returns:
        Series with 'OPEN_LONG', 'OPEN_SHORT', 'CLOSE_LONG', 'CLOSE_SHORT', or 'HOLD' signals
    """
    try:
        signals = pd.Series('HOLD', index=price.index)
        
        for i in range(lookback_period, len(price)):
            price_window = price.iloc[i-lookback_period:i+1]
            indicator_window = indicator.iloc[i-lookback_period:i+1]
            
            # Bullish divergence: price makes lower low, indicator makes higher low
            if (price_window.iloc[-1] < price_window.iloc[0] and 
                indicator_window.iloc[-1] > indicator_window.iloc[0]):
                signals.iloc[i] = 'OPEN_LONG'
            
            # Bearish divergence: price makes higher high, indicator makes lower high
            elif (price_window.iloc[-1] > price_window.iloc[0] and 
                  indicator_window.iloc[-1] < indicator_window.iloc[0]):
                signals.iloc[i] = 'OPEN_SHORT'
        
        return signals
    except Exception as e:
        logger.error(f"Error in divergence_signals: {e}")
        return pd.Series('HOLD', index=price.index)

def breakout_signals(price: pd.Series, resistance: pd.Series, 
                   support: pd.Series, penetration_pct: float = 0.01) -> pd.Series:
    """
    Generate breakout signals
    
    FIXED: Now returns correct signal types
    
    Args:
        price: Price series
        resistance: Resistance level series
        support: Support level series
        penetration_pct: Penetration percentage required
        
    Returns:
        Series with 'OPEN_LONG', 'OPEN_SHORT', 'CLOSE_LONG', 'CLOSE_SHORT', or 'HOLD' signals
    """
    try:
        signals = pd.Series('HOLD', index=price.index)
        
        # OPEN_LONG breakout: price breaks above resistance
        buy_breakout = price > resistance * (1 + penetration_pct)
        signals[buy_breakout] = 'OPEN_LONG'
        
        # OPEN_SHORT breakout: price breaks below support
        sell_breakout = price < support * (1 - penetration_pct)
        signals[sell_breakout] = 'OPEN_SHORT'
        
        # CLOSE_LONG when price falls back below resistance
        close_long = price < resistance
        signals[close_long] = 'CLOSE_LONG'
        
        # CLOSE_SHORT when price rises back above support
        close_short = price > support
        signals[close_short] = 'CLOSE_SHORT'
        
        return signals
    except Exception as e:
        logger.error(f"Error in breakout_signals: {e}")
        return pd.Series('HOLD', index=price.index)


# === COMBINATION SIGNAL FUNCTIONS ===

def combine_rsi_trend_signals(rsi_signal, trend_signal, rsi_values=None, ema_fast_values=None, ema_slow_values=None, **kwargs):
    """
    Combine RSI and trend signals to generate final trading signals.
    
    FIXED: Now returns correct signal types
    
    Returns a dictionary with both the signal and the indicator values for display.
    """
    # Get current values
    current_rsi = rsi_values.iloc[-1] if rsi_values is not None and not rsi_values.empty else None
    current_ema_fast = ema_fast_values.iloc[-1] if ema_fast_values is not None and not ema_fast_values.empty else None
    current_ema_slow = ema_slow_values.iloc[-1] if ema_slow_values is not None and not ema_slow_values.empty else None
    
    # Determine trend direction
    if current_ema_fast is not None and current_ema_slow is not None:
        if current_ema_fast > current_ema_slow:
            trend_direction = "UPTREND"
        else:
            trend_direction = "DOWNTREND"
    else:
        trend_direction = "UNKNOWN"
    
    # Determine RSI condition
    if current_rsi is not None:
        if current_rsi > kwargs.get('rsi_overbought', 70):
            rsi_condition = "OVERBOUGHT"
        elif current_rsi < kwargs.get('rsi_oversold', 30):
            rsi_condition = "OVERSOLD"
        else:
            rsi_condition = "NEUTRAL"
    else:
        rsi_condition = "UNKNOWN"
    
    # Generate signal based on RSI and trend
    signal = 'HOLD'
    
    if trend_direction == "UPTREND":
        if rsi_condition == "OVERSOLD":
            signal = 'OPEN_LONG'
        elif rsi_condition == "OVERBOUGHT":
            signal = 'CLOSE_LONG'
    elif trend_direction == "DOWNTREND":
        if rsi_condition == "OVERBOUGHT":
            signal = 'OPEN_SHORT'
        elif rsi_condition == "OVERSOLD":
            signal = 'CLOSE_SHORT'
    
    # Return both signal and indicator values
    return {
        'signal': signal,
        'indicators': {
            'rsi': current_rsi,
            'rsi_condition': rsi_condition,
            'ema_fast': current_ema_fast,
            'ema_slow': current_ema_slow,
            'trend_direction': trend_direction,
            'rsi_signal': rsi_signal,
            'trend_signal': trend_signal
        }
    }

def majority_vote_signals(signal_list: List[pd.Series]) -> pd.Series:
    """
    Generate majority vote combination of signals
    
    FIXED: Now returns correct signal types
    
    Args:
        signal_list: List of signal series to combine
        
    Returns:
        Combined signal series with correct signal types
    """
    try:
        if not signal_list:
            return pd.Series()
        
        # Find common index across all signal series
        common_index = signal_list[0].index
        for series in signal_list[1:]:
            common_index = common_index.intersection(series.index)
        
        if len(common_index) == 0:
            return pd.Series()
        
        # Count votes for each signal type
        open_long_votes = pd.Series(0, index=common_index)
        open_short_votes = pd.Series(0, index=common_index)
        close_long_votes = pd.Series(0, index=common_index)
        close_short_votes = pd.Series(0, index=common_index)
        hold_votes = pd.Series(0, index=common_index)
        
        for series in signal_list:
            aligned_series = series.reindex(common_index)
            open_long_votes += (aligned_series == 'OPEN_LONG')
            open_short_votes += (aligned_series == 'OPEN_SHORT')
            close_long_votes += (aligned_series == 'CLOSE_LONG')
            close_short_votes += (aligned_series == 'CLOSE_SHORT')
            hold_votes += (aligned_series == 'HOLD')
        
        # Determine majority vote
        final_signals = pd.Series('HOLD', index=common_index)
        total_signals = len(signal_list)
        majority_threshold = total_signals / 2
        
        # Check each signal type for majority
        final_signals[open_long_votes > majority_threshold] = 'OPEN_LONG'
        final_signals[open_short_votes > majority_threshold] = 'OPEN_SHORT'
        final_signals[close_long_votes > majority_threshold] = 'CLOSE_LONG'
        final_signals[close_short_votes > majority_threshold] = 'CLOSE_SHORT'
        # Hold is default
        
        return final_signals
    except Exception as e:
        logger.error(f"Error in majority_vote_signals: {e}")
        return pd.Series('HOLD', index=common_index if 'common_index' in locals() else None)

def and_signals(signal_list: List[pd.Series]) -> pd.Series:
    """
    Combine signals such that a signal is returned only if all signals agree.
    
    FIXED: Now returns correct signal types
    
    Args:
        signal_list: List of signal series to combine
        
    Returns:
        Combined signal series with correct signal types
    """
    try:
        if not signal_list:
            return pd.Series()
        
        # Find common index
        common_index = signal_list[0].index
        for series in signal_list[1:]:
            common_index = common_index.intersection(series.index)
        
        if len(common_index) == 0:
            return pd.Series()
        
        final_signals = pd.Series('HOLD', index=common_index)
        
        # Check if all signals are the same at each timestamp
        for signal_type in ['OPEN_LONG', 'OPEN_SHORT', 'CLOSE_LONG', 'CLOSE_SHORT']:
            all_agree = pd.Series(True, index=common_index)
            for series in signal_list:
                aligned = series.reindex(common_index)
                all_agree &= (aligned == signal_type)
            final_signals[all_agree] = signal_type
        
        return final_signals
    except Exception as e:
        logger.error(f"Error in and_signals: {e}")
        return pd.Series('HOLD', index=common_index if 'common_index' in locals() else None)


# === SIGNAL REGISTRY ===

SIGNAL_REGISTRY = {
    'overbought_oversold': overbought_oversold,
    'overbought_oversold_crossover': overbought_oversold_crossover,
    'overbought_oversold_with_trend': overbought_oversold_with_trend,
    'trend_signal': trend_signal,
    'rsi_mean_reversion_with_trend': rsi_mean_reversion_with_trend,
    'ma_crossover': ma_crossover,
    'macd_signals': macd_signals,
    'bollinger_bands_signals': bollinger_bands_signals,
    'stochastic_signals': stochastic_signals,
    'divergence_signals': divergence_signals,
    'breakout_signals': breakout_signals,
    'combine_rsi_trend_signals': combine_rsi_trend_signals,
    'majority_vote_signals': majority_vote_signals,
    'and_signals': and_signals,
}


def get_signal_function(name: str):
    """
    Get signal function by name
    
    Args:
        name: Signal function name
        
    Returns:
        Signal function
    """
    if name in SIGNAL_REGISTRY:
        return SIGNAL_REGISTRY[name]
    else:
        raise ValueError(f"Signal function '{name}' not found. Available functions: {list(SIGNAL_REGISTRY.keys())}")


def list_signal_functions() -> list:
    """
    List all available signal functions
    
    Returns:
        List of signal function names
    """
    return list(SIGNAL_REGISTRY.keys())


if __name__ == "__main__":
    # Example usage
    print("📊 Available Signal Functions (CORRECTED):")
    for signal_func in list_signal_functions():
        print(f"  - {signal_func}")
    
    print("\n✅ All signal functions now use correct signal types:")
    print("  - OPEN_LONG")
    print("  - OPEN_SHORT")
    print("  - CLOSE_LONG")
    print("  - CLOSE_SHORT")
    print("  - HOLD")