# GUI Dashboard

## Purpose
Provide a unified UI for strategy creation, backtesting, optimization, and paper trading.

## Key Components
- `simple_strategy/gui_monitor.py`: Main dashboard UI (strategy config, backtest, optimization).
- `simple_strategy/trading/paper_trading_launcher.py`: Paper trading UI launcher.

## Capabilities
- Strategy selection and parameter editing.
- Backtest execution with progress reporting.
- Optimization workflow integration.
- Paper trading controls and logs.

## Strategy Discovery
- Strategy files are discovered by filename: `Strategy_*.py`.
- GUI reads `STRATEGY_PARAMETERS` to render controls.
- `create_strategy()` is called with GUI-provided params.

## Notes
- Uses Tkinter.
- Intended as the primary user-facing control center.
