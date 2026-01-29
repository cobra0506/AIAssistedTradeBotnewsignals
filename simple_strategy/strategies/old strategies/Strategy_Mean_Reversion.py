import pandas as pd
import numpy as np

def create_strategy(lookback_period=20, entry_threshold=2.0, exit_threshold=0.5):
    """
    Simple Mean Reversion Strategy
    - Buy when price is below lower Bollinger Band
    - Sell when price is above upper Bollinger Band
    - Exit when price returns to mean
    """
    
    class MeanReversionStrategy:
        def __init__(self, lookback_period=20, entry_threshold=2.0, exit_threshold=0.5):
            self.lookback_period = lookback_period
            self.entry_threshold = entry_threshold
            self.exit_threshold = exit_threshold
            self.name = f"Mean_Reversion_LB{lookback_period}_ET{entry_threshold}"
        
        def generate_signals(self, data):
            """Generate trading signals based on mean reversion"""
            try:
                # Calculate Bollinger Bands
                data['sma'] = data['close'].rolling(window=self.lookback_period).mean()
                data['std'] = data['close'].rolling(window=self.lookback_period).std()
                data['upper_band'] = data['sma'] + (data['std'] * self.entry_threshold)
                data['lower_band'] = data['sma'] - (data['std'] * self.entry_threshold)
                data['exit_upper'] = data['sma'] + (data['std'] * self.exit_threshold)
                data['exit_lower'] = data['sma'] - (data['std'] * self.exit_threshold)
                
                # Generate signals
                signals = {}
                
                # Get the latest data
                latest = data.iloc[-1]
                prev = data.iloc[-2]
                
                # Buy signal: price below lower band
                if latest['close'] < latest['lower_band']:
                    signals[data.index[-1]] = 'BUY'
                
                # Sell signal: price above upper band
                elif latest['close'] > latest['upper_band']:
                    signals[data.index[-1]] = 'SELL'
                
                # Exit long: price above exit lower band
                elif prev['close'] < prev['exit_lower'] and latest['close'] > latest['exit_lower']:
                    signals[data.index[-1]] = 'EXIT_LONG'
                
                # Exit short: price below exit upper band
                elif prev['close'] > prev['exit_upper'] and latest['close'] < latest['exit_upper']:
                    signals[data.index[-1]] = 'EXIT_SHORT'
                
                return signals
                
            except Exception as e:
                print(f"Error generating signals: {e}")
                return {}
    
    return MeanReversionStrategy(lookback_period, entry_threshold, exit_threshold)