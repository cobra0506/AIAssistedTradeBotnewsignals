# Trading Interface

## Purpose
Bridge strategies to execution on Bybit **demo** accounts, manage API accounts, and load optimized parameters.

## Key Components
- `simple_strategy/trading/api_manager.py`: API account management.
- `simple_strategy/trading/api_gui.py`: GUI for account management.
- `simple_strategy/trading/parameter_manager.py`: Optimized parameter storage/loading.
- `simple_strategy/trading/parameter_gui.py`: GUI for parameters.
- `simple_strategy/trading/paper_trading_engine.py`: Paper trading execution.

## How It Works
1. Manage API accounts (demo/live) in JSON store (execution currently targets demo).
2. Load strategy parameters (optimized or default).
3. Execute paper trading with GUI control.

## Integration Points
- Data Collection (live prices)
- Optimization (parameter outputs)
- GUI Dashboard (tabs and controls)
