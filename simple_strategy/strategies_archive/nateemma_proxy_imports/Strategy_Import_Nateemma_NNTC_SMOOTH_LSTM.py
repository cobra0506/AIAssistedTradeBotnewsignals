"""Port of nateemma/strategies NNTC/NNTC_smooth_LSTM.py."""
from simple_strategy.strategies.imported_nateemma_batch1_helper import create_imported_strategy
def create_strategy(symbols=None, timeframes=None, **params):
    return create_imported_strategy("Anomaly_smooth", symbols=symbols, timeframes=timeframes, **params)
