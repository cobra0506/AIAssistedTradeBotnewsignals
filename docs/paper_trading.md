# Paper Trading

## Purpose
Trade against live market data using Bybit **demo** APIs with a simulated balance layer.

## Key Components
- `simple_strategy/trading/paper_trading_engine.py`: Core trading loop, execution, and P&L tracking.
- `simple_strategy/trading/paper_trading_launcher.py`: Tkinter launcher UI.
- `simple_strategy/trading/api_manager.py`: API credential management.
- `simple_strategy/trading/parameter_manager.py`: Loads optimized params for strategies.

## How It Works
1. Load strategy via `StrategyRegistry` and optimized parameters from `ParameterManager`.
2. Pull live market data (shared WebSocket if available, CSV fallback).
3. Generate signals and execute **demo** orders via Bybit endpoints.
4. Track positions, balances, and performance in real time (with balance offset).

## Notes
- Uses shared WebSocket data where available for consistency and efficiency.
- Trades are executed on Bybit **demo** accounts (real API calls, no real capital).
- Console logging is Unicode-safe on Windows. If a terminal cannot render a symbol, logging falls back to ASCII-safe output.
- Fast finder strategies become selectable in paper trader after they are saved under `simple_strategy/strategies`.
- Unique engine run-folder strategies are not auto-discovered by paper trader until published/copied into `simple_strategy/strategies`.

## Optional Redis Integration (Planned)
An event-driven Redis pub/sub path can reduce latency by pushing confirmed candles directly to the paper trader instead of polling CSVs. This is currently a proposed enhancement.
