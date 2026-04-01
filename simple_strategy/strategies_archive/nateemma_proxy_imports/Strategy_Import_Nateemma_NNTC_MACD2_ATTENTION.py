"""Port of nateemma/strategies NNTC/NNTC_macd2_Attention.py."""
from simple_strategy.strategies.imported_nateemma_batch1_helper import create_imported_strategy
def create_strategy(symbols=None, timeframes=None, **params):
    return create_imported_strategy("Anomaly_macd2", symbols=symbols, timeframes=timeframes, **params)
