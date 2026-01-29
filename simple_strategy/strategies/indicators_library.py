"""
Building Block Indicators Library
=================================

This library contains ALL technical indicators that can be used in strategy building.
Each indicator is a standalone function that can be called independently.

Author: AI Assisted TradeBot Team
Date: 2025
"""

import pandas as pd
import numpy as np
from typing import Union, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# === TREND INDICATORS ===

def trend_signal(ema_fast, ema_slow):
    """
    Simple trend filter: BUY only if fast EMA > slow EMA,
    SELL only if fast EMA < slow EMA, else HOLD.
    """
    signals = pd.Series('HOLD', index=ema_fast.index)
    signals[ema_fast > ema_slow] = 'BUY'
    signals[ema_fast < ema_slow] = 'SELL'
    return signals

def sma(data: pd.Series, period: int = 20) -> pd.Series:
    """
    Simple Moving Average
    
    Args:
        data: Price series
        period: Lookback period
        
    Returns:
        SMA series
    """
    try:
        return data.rolling(window=period).mean()
    except Exception as e:
        logger.error(f"Error calculating SMA: {e}")
        return pd.Series(index=data.index, dtype=float)


def ema(data: pd.Series, period: int = 20) -> pd.Series:
    """
    Exponential Moving Average
    Args:
        data: Price series
        period: Lookback period
    Returns:
        EMA series
    """
    try:
        # Handle edge case: period larger than data length
        if period > len(data):
            return pd.Series([np.nan] * len(data), index=data.index, dtype=float)
        
        # Handle edge case: period <= 0
        if period <= 0:
            return pd.Series([np.nan] * len(data), index=data.index, dtype=float)
        
        # Initialize result series with NaN values
        ema_series = pd.Series([np.nan] * len(data), index=data.index, dtype=float)
        
        # Find the first 'period' non-NaN values
        non_nan_indices = []
        non_nan_values = []
        for i, val in enumerate(data):
            if not pd.isna(val):
                non_nan_indices.append(i)
                non_nan_values.append(val)
                if len(non_nan_values) >= period:
                    break
        
        # If we don't have enough non-NaN values, return all NaN
        if len(non_nan_values) < period:
            return ema_series
        
        # First EMA value is the SMA of the first 'period' non-NaN values
        first_ema = sum(non_nan_values) / period
        ema_series.iloc[non_nan_indices[-1]] = first_ema
        
        # Calculate smoothing factor
        smoothing = 2 / (period + 1)
        
        # Calculate subsequent EMA values
        for i in range(non_nan_indices[-1] + 1, len(data)):
            if not pd.isna(data.iloc[i]):
                ema_series.iloc[i] = smoothing * data.iloc[i] + (1 - smoothing) * ema_series.iloc[i - 1]
        
        return ema_series
    except Exception as e:
        logger.error(f"Error calculating EMA: {e}")
        return pd.Series(index=data.index, dtype=float)


def wma(data: pd.Series, period: int = 20) -> pd.Series:
    """
    Weighted Moving Average
    
    Args:
        data: Price series
        period: Lookback period
        
    Returns:
        WMA series
    """
    try:
        weights = np.arange(1, period + 1)
        weights = weights / weights.sum()
        return data.rolling(window=period).apply(lambda x: np.dot(x, weights), raw=True)
    except Exception as e:
        logger.error(f"Error calculating WMA: {e}")
        return pd.Series(index=data.index, dtype=float)


def dema(data: pd.Series, period: int = 20) -> pd.Series:
    """
    Double Exponential Moving Average
    
    Args:
        data: Price series
        period: Lookback period
        
    Returns:
        DEMA series
    """
    try:
        ema1 = ema(data, period)
        ema2 = ema(ema1, period)
        return 2 * ema1 - ema2
    except Exception as e:
        logger.error(f"Error calculating DEMA: {e}")
        return pd.Series(index=data.index, dtype=float)


def tema(data: pd.Series, period: int = 20) -> pd.Series:
    """
    Triple Exponential Moving Average
    
    Args:
        data: Price series
        period: Lookback period
        
    Returns:
        TEMA series
    """
    try:
        ema1 = ema(data, period)
        ema2 = ema(ema1, period)
        ema3 = ema(ema2, period)
        return 3 * ema1 - 3 * ema2 + ema3
    except Exception as e:
        logger.error(f"Error calculating TEMA: {e}")
        return pd.Series(index=data.index, dtype=float)


# === MOMENTUM INDICATORS ===

def rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index
    
    Args:
        data: Price series
        period: Lookback period
        
    Returns:
        RSI series
    """
    try:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return pd.Series(index=data.index, dtype=float)


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
              k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        k_period: %K period
        d_period: %D period
        
    Returns:
        Tuple of (%K series, %D series)
    """
    try:
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    except Exception as e:
        logger.error(f"Error calculating Stochastic: {e}")
        return pd.Series(index=close.index, dtype=float), pd.Series(index=close.index, dtype=float)


def srsi(data: pd.Series, period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic RSI
    
    Args:
        data: Price series
        period: RSI period
        d_period: %D period
        
    Returns:
        Tuple of (SRSI-K series, SRSI-D series)
    """
    try:
        rsi_values = rsi(data, period)
        lowest_low = rsi_values.rolling(window=period).min()
        highest_high = rsi_values.rolling(window=period).max()
        srsi_k = 100 * ((rsi_values - lowest_low) / (highest_high - lowest_low))
        srsi_d = srsi_k.rolling(window=d_period).mean()
        return srsi_k, srsi_d
    except Exception as e:
        logger.error(f"Error calculating SRSI: {e}")
        return pd.Series(index=data.index, dtype=float), pd.Series(index=data.index, dtype=float)


def macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, 
         signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Moving Average Convergence Divergence
    
    Args:
        data: Price series
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
        
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    try:
        ema_fast = ema(data, fast_period)
        ema_slow = ema(data, slow_period)
        macd_line = ema_fast - ema_slow
        signal_line = ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    except Exception as e:
        logger.error(f"Error calculating MACD: {e}")
        return (pd.Series(index=data.index, dtype=float), 
                pd.Series(index=data.index, dtype=float), 
                pd.Series(index=data.index, dtype=float))


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """
    Commodity Channel Index
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Lookback period
        
    Returns:
        CCI series
    """
    try:
        tp = (high + low + close) / 3
        sma_tp = sma(tp, period)
        mad = tp.rolling(window=period).apply(lambda x: np.fabs(x - x.mean()).mean())
        cci = (tp - sma_tp) / (0.015 * mad)
        return cci
    except Exception as e:
        logger.error(f"Error calculating CCI: {e}")
        return pd.Series(index=close.index, dtype=float)

def williams_r(high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Williams %R Indicator
    
    Williams %R is a momentum indicator that is the inverse of the Stochastic Oscillator.
    It ranges from 0 to -100, with readings above -20 considered overbought and 
    readings below -80 considered oversold.
    
    Formula: %R = -100 * (Highest High - Close) / (Highest High - Lowest Low)
    
    Args:
        high_prices: Series of high prices
        low_prices: Series of low prices
        close_prices: Series of close prices
        period: Lookback period (default: 14)
        
    Returns:
        Series of Williams %R values
    """
    try:
        # Handle edge case: period larger than data length
        if period > len(high_prices):
            return pd.Series([np.nan] * len(high_prices), index=high_prices.index, dtype=float)
        
        # Handle edge case: period <= 0
        if period <= 0:
            return pd.Series([np.nan] * len(high_prices), index=high_prices.index, dtype=float)
        
        # Calculate highest high and lowest low over the period
        highest_high = high_prices.rolling(window=period).max()
        lowest_low = low_prices.rolling(window=period).min()
        
        # Calculate Williams %R
        williams_r = -100 * (highest_high - close_prices) / (highest_high - lowest_low)
        
        # Handle division by zero case (when highest_high equals lowest_low)
        williams_r = williams_r.replace([np.inf, -np.inf], np.nan)
        
        return williams_r
        
    except Exception as e:
        logger.error(f"Error calculating Williams %R: {e}")
        return pd.Series(index=high_prices.index, dtype=float)

# === VOLATILITY INDICATORS ===

def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands
    
    Args:
        data: Price series
        period: Lookback period
        std_dev: Standard deviation multiplier
        
    Returns:
        Tuple of (Upper band, Middle band, Lower band)
    """
    try:
        # Handle edge cases
        if period <= 0 or period > len(data):
            logger.warning(f"Invalid Bollinger Bands period {period} for data length {len(data)}")
            nan_series = pd.Series([np.nan] * len(data), index=data.index, dtype=float)
            return nan_series, nan_series, nan_series
        
        # Handle NaN values in input data
        if data.isnull().all():
            logger.warning("All NaN values in Bollinger Bands input data")
            nan_series = pd.Series([np.nan] * len(data), index=data.index, dtype=float)
            return nan_series, nan_series, nan_series
        
        middle_band = sma(data, period)
        std = data.rolling(window=period).std()
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return upper_band, middle_band, lower_band
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {e}")
        nan_series = pd.Series([np.nan] * len(data), index=data.index, dtype=float)
        return nan_series, nan_series, nan_series


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average True Range
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Lookback period
        
    Returns:
        ATR series
    """
    try:
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    except Exception as e:
        logger.error(f"Error calculating ATR: {e}")
        return pd.Series(index=close.index, dtype=float)


# === VOLUME INDICATORS ===

def volume_sma(volume_data: pd.Series, period: int = 20) -> pd.Series:
    """
    Volume Simple Moving Average
    
    Calculates the simple moving average of volume data.
    This is useful for identifying trends in trading volume and
    comparing current volume to its historical average.
    
    Args:
        volume_data: Series of volume data
        period: Lookback period for SMA calculation (default: 20)
        
    Returns:
        Series of volume SMA values
    """
    try:
        # Handle edge case: period larger than data length
        if period > len(volume_data):
            return pd.Series([np.nan] * len(volume_data), index=volume_data.index, dtype=float)
        
        # Handle edge case: period <= 0
        if period <= 0:
            return pd.Series([np.nan] * len(volume_data), index=volume_data.index, dtype=float)
        
        # Calculate Volume SMA using simple moving average
        volume_sma_values = volume_data.rolling(window=period, min_periods=period).mean()
        
        return volume_sma_values
        
    except Exception as e:
        logger.error(f"Error calculating Volume SMA: {e}")
        return pd.Series(index=volume_data.index, dtype=float)


def on_balance_volume(close_prices: pd.Series, volume_data: pd.Series) -> pd.Series:
    """
    On Balance Volume (OBV)
    
    On Balance Volume is a momentum indicator that uses volume flow to predict 
    changes in stock price. The theory is that volume precedes price movement,
    so if volume is increasing while price is flat, price will soon increase.
    
    Calculation:
    - If today's close > yesterday's close: OBV = yesterday's OBV + today's volume
    - If today's close < yesterday's close: OBV = yesterday's OBV - today's volume
    - If today's close = yesterday's close: OBV = yesterday's OBV
    
    Args:
        close_prices: Series of close prices
        volume_data: Series of volume data
        
    Returns:
        Series of OBV values
    """
    try:
        # Handle edge case: empty data
        if len(close_prices) == 0 or len(volume_data) == 0:
            return pd.Series(dtype=float)
        
        # Handle edge case: different lengths
        if len(close_prices) != len(volume_data):
            # Use the minimum length
            min_length = min(len(close_prices), len(volume_data))
            close_prices = close_prices.iloc[:min_length]
            volume_data = volume_data.iloc[:min_length]
        
        # Initialize OBV series with zeros
        obv_values = pd.Series([0.0] * len(close_prices), index=close_prices.index, dtype=float)
        
        # Calculate OBV starting from the second day (index 1)
        for i in range(1, len(close_prices)):
            if close_prices.iloc[i] > close_prices.iloc[i-1]:
                # Price increased: add volume
                obv_values.iloc[i] = obv_values.iloc[i-1] + volume_data.iloc[i]
            elif close_prices.iloc[i] < close_prices.iloc[i-1]:
                # Price decreased: subtract volume
                obv_values.iloc[i] = obv_values.iloc[i-1] - volume_data.iloc[i]
            else:
                # Price unchanged: OBV remains the same
                obv_values.iloc[i] = obv_values.iloc[i-1]
        
        return obv_values
        
    except Exception as e:
        logger.error(f"Error calculating On Balance Volume: {e}")
        return pd.Series(index=close_prices.index, dtype=float)


# === UTILITY FUNCTIONS ===

def crossover(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """
    Crossover Signal Generator
    
    Generates signals when series1 crosses above series2.
    Returns 1 when series1 crosses above series2, 0 otherwise.
    
    This is useful for detecting when one indicator crosses above another,
    such as a fast moving average crossing above a slow moving average.
    
    Args:
        series1: First data series (e.g., fast MA)
        series2: Second data series (e.g., slow MA)
        
    Returns:
        Series of binary signals (1 for crossover, 0 for no crossover)
    """
    try:
        # Handle edge case: empty data
        if len(series1) == 0 or len(series2) == 0:
            return pd.Series(dtype=float)
        
        # Handle edge case: different lengths
        if len(series1) != len(series2):
            # Use the minimum length
            min_length = min(len(series1), len(series2))
            series1 = series1.iloc[:min_length]
            series2 = series2.iloc[:min_length]
        
        # Initialize result series with zeros
        crossover_signals = pd.Series([0] * len(series1), index=series1.index, dtype=float)
        
        # Check for crossovers starting from index 1
        for i in range(1, len(series1)):
            # Current and previous values
            s1_curr = series1.iloc[i]
            s1_prev = series1.iloc[i-1]
            s2_curr = series2.iloc[i]
            s2_prev = series2.iloc[i-1]
            
            # Check for crossover: series1 was below series2, now above
            if (s1_prev <= s2_prev) and (s1_curr > s2_curr):
                crossover_signals.iloc[i] = 1
        
        return crossover_signals
        
    except Exception as e:
        logger.error(f"Error calculating crossover signals: {e}")
        return pd.Series(dtype=float)


def crossunder(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """
    Crossunder Signal Generator
    
    Generates signals when series1 crosses below series2.
    Returns 1 when series1 crosses below series2, 0 otherwise.
    
    This is useful for detecting when one indicator crosses below another,
    such as a fast moving average crossing below a slow moving average.
    
    Args:
        series1: First data series (e.g., fast MA)
        series2: Second data series (e.g., slow MA)
        
    Returns:
        Series of binary signals (1 for crossunder, 0 for no crossunder)
    """
    try:
        # Handle edge case: empty data
        if len(series1) == 0 or len(series2) == 0:
            return pd.Series(dtype=float)
        
        # Handle edge case: different lengths
        if len(series1) != len(series2):
            # Use the minimum length
            min_length = min(len(series1), len(series2))
            series1 = series1.iloc[:min_length]
            series2 = series2.iloc[:min_length]
        
        # Initialize result series with zeros
        crossunder_signals = pd.Series([0] * len(series1), index=series1.index, dtype=float)
        
        # Check for crossunders starting from index 1
        for i in range(1, len(series1)):
            # Current and previous values
            s1_curr = series1.iloc[i]
            s1_prev = series1.iloc[i-1]
            s2_curr = series2.iloc[i]
            s2_prev = series2.iloc[i-1]
            
            # Check for crossunder: series1 was above series2, now below
            if (s1_prev >= s2_prev) and (s1_curr < s2_curr):
                crossunder_signals.iloc[i] = 1
        
        return crossunder_signals
        
    except Exception as e:
        logger.error(f"Error calculating crossunder signals: {e}")
        return pd.Series(dtype=float)


def highest(data: pd.Series, period: int = 20) -> pd.Series:
    """
    Highest Value Over Period
    
    Returns the highest value in the data series over the specified period.
    This is useful for identifying resistance levels, highest high in a period,
    and for various technical analysis calculations.
    
    Args:
        data: Series of data values
        period: Lookback period for finding highest value (default: 20)
        
    Returns:
        Series of highest values over the specified period
    """
    try:
        # Handle edge case: period larger than data length
        if period > len(data):
            return pd.Series([np.nan] * len(data), index=data.index, dtype=float)
        
        # Handle edge case: period <= 0
        if period <= 0:
            return pd.Series([np.nan] * len(data), index=data.index, dtype=float)
        
        # Calculate highest value using rolling window
        highest_values = data.rolling(window=period, min_periods=1).max()
        
        return highest_values
        
    except Exception as e:
        logger.error(f"Error calculating highest values: {e}")
        return pd.Series(index=data.index, dtype=float)


def lowest(data: pd.Series, period: int = 20) -> pd.Series:
    """
    Lowest Value Over Period
    
    Returns the lowest value in the data series over the specified period.
    This is useful for identifying support levels, lowest low in a period,
    and for various technical analysis calculations.
    
    Args:
        data: Series of data values
        period: Lookback period for finding lowest value (default: 20)
        
    Returns:
        Series of lowest values over the specified period
    """
    try:
        # Handle edge case: period larger than data length
        if period > len(data):
            return pd.Series([np.nan] * len(data), index=data.index, dtype=float)
        
        # Handle edge case: period <= 0
        if period <= 0:
            return pd.Series([np.nan] * len(data), index=data.index, dtype=float)
        
        # Calculate lowest value using rolling window
        lowest_values = data.rolling(window=period, min_periods=1).min()
        
        return lowest_values
        
    except Exception as e:
        logger.error(f"Error calculating lowest values: {e}")
        return pd.Series(index=data.index, dtype=float)


# === INDICATOR REGISTRY ===
# This makes it easy to get all available indicators

INDICATOR_REGISTRY = {
    'sma': sma,
    'ema': ema,
    'wma': wma,
    'dema': dema,
    'tema': tema,
    'rsi': rsi,
    'stochastic': stochastic,
    'srsi': srsi,
    'macd': macd,
    'cci': cci,
    'williams_r': williams_r,
    'bollinger_bands': bollinger_bands,
    'atr': atr,
    'volume_sma': volume_sma,
    'on_balance_volume': on_balance_volume,
    'crossover': crossover,
    'crossunder': crossunder,
    'highest': highest,
    'lowest': lowest,
}


def get_indicator(name: str):
    """
    Get indicator function by name
    
    Args:
        name: Indicator name
        
    Returns:
        Indicator function
    """
    if name in INDICATOR_REGISTRY:
        return INDICATOR_REGISTRY[name]
    else:
        raise ValueError(f"Indicator '{name}' not found. Available indicators: {list(INDICATOR_REGISTRY.keys())}")


def list_indicators() -> list:
    """
    List all available indicators
    
    Returns:
        List of indicator names
    """
    return list(INDICATOR_REGISTRY.keys())


if __name__ == "__main__":
    # Example usage
    print("📊 Available Indicators:")
    for indicator in list_indicators():
        print(f"  - {indicator}")

'''Signals Library Functions Summary 
Trend Indicators 

     trend_signal - Generates BUY/SELL/HOLD signals based on EMA crossover
     sma - Calculates Simple Moving Average
     ema - Calculates Exponential Moving Average
     wma - Calculates Weighted Moving Average
     dema - Calculates Double Exponential Moving Average
     tema - Calculates Triple Exponential Moving Average
     

Momentum Indicators 

     rsi - Calculates Relative Strength Index
     stochastic - Calculates Stochastic Oscillator (%K and %D)
     srsi - Calculates Stochastic RSI
     macd - Calculates Moving Average Convergence Divergence
     cci - Calculates Commodity Channel Index
     williams_r - Calculates Williams %R momentum indicator
     

Volatility Indicators 

     bollinger_bands - Calculates Bollinger Bands (upper, middle, lower)
     atr - Calculates Average True Range
     

Volume Indicators 

     volume_sma - Calculates Simple Moving Average of volume
     on_balance_volume - Calculates On Balance Volume (OBV)
     

Utility Functions 

     crossover - Detects when series1 crosses above series2
     crossunder - Detects when series1 crosses below series2
     highest - Finds highest value over a period
     lowest - Finds lowest value over a period
     

Registry Functions 

     get_indicator - Gets indicator function by name
     list_indicators - Lists all available indicators
     '''