from simple_strategy.strategies.imported_nateemma_direct_batch2_helper import create_direct_batch2_strategy


def create_strategy(symbols=None, timeframes=None, **params):
    return create_direct_batch2_strategy("EMA50", symbols=symbols, timeframes=timeframes, **params)
