"""Port of nateemma/strategies NNTC/NNTC_dwt_LSTM.py."""

from simple_strategy.strategies.imported_nateemma_batch1_helper import create_imported_strategy


def create_strategy(symbols=None, timeframes=None, **params):
    return create_imported_strategy("Anomaly_dwt", symbols=symbols, timeframes=timeframes, **params)
