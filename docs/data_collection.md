# Data Collection

## Purpose
Collect, validate, store, and serve historical + real-time market data for backtesting and trading.

## Key Components
- `shared_modules/data_collection/`:
  - `hybrid_system.py`: Orchestrates historical + real-time data flow.
  - `optimized_data_fetcher.py`: Concurrent historical fetching.
  - `websocket_handler.py` + `shared_websocket_manager.py`: Live data streaming.
  - `csv_manager.py`: CSV persistence.
  - `data_integrity.py`: Gap/duplicate detection and repair.
- `simple_strategy/shared/data_feeder.py`: Loads data for strategies/backtests.

## How It Works (High Level)
1. Fetch historical OHLCV with rate limits and chunking.
2. Stream live data via shared WebSocket connection.
3. Validate and store data in CSVs per symbol/timeframe.
4. Serve data to backtests and paper trading via `DataFeeder` or shared access.

## Configuration
Key settings live in `shared_modules/data_collection/config.py` (symbols, timeframes, fetch windows, batching).

## Integration Points
- Backtesting (historical CSVs)
- Paper Trading (shared WebSocket data)
- Strategy Builder (data access for testing)

