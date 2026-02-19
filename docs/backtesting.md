# Backtesting

## Purpose
Backtest strategies against historical market data with realistic execution, risk controls, and performance metrics.

## Key Components
- `simple_strategy/backtester/backtester_engine.py`: Orchestrates the backtest loop and trade execution.
- `simple_strategy/backtester/risk_manager.py`: Risk-based position sizing utilities.
- `simple_strategy/backtester/performance_tracker.py`: Trade history and performance metrics.
- `simple_strategy/shared/data_feeder.py`: Loads historical CSV data for backtests.
Note: `simple_strategy/backtester/position_manager.py` exists but the current engine uses its own internal `self.positions` tracking.

## How It Works (High Level)
1. Load historical OHLCV data for symbols/timeframes via `DataFeeder`.
2. Pre-calculate baseline indicators (EMA/RSI in `_precalculate_indicators`) for fast loop execution.
3. Generate signals per-bar:
- Builder strategies: backtester calculates builder-specific indicators first, then executes builder signal rules.
- Vectorized strategies: uses `generate_signals_vectorized` when present.
- Other strategies: falls back to `generate_signals` on rolling history windows.
4. Execute trades with slippage/spread/fee simulation and risk-based sizing.
5. Close remaining positions and compute metrics.

## Usage
```python
from simple_strategy.backtester.backtester_engine import BacktesterEngine
from simple_strategy.shared.data_feeder import DataFeeder

data_feeder = DataFeeder(data_dir='data')
backtest = BacktesterEngine(data_feeder=data_feeder, strategy=my_strategy)
results = backtest.run_backtest(
    symbols=['BTCUSDT'],
    timeframes=['1m'],
    start_date='2023-01-01',
    end_date='2023-12-31'
)
```

## Key Outputs
`run_backtest` returns a dict with:
- `win_rate`, `sharpe_ratio`, `max_drawdown`, `total_return`, `total_trades`
- `start_time`, `end_time`, `duration`

## Stored Results
Backtest results are saved to `simple_strategy/optimization_results/backtest_results.json` keyed by strategy name (overwrites on re-run).
Stored fields include initial balance and net profit/loss.

## Notes
- `progress_callback` can be provided to track progress.
- Generated Fast/Unique strategies built with `StrategyBuilder` now run directly in backtester without manual edits.

## Integration Points
- Strategy Builder (strategy objects)
- Data Collection (CSV data)
- Optimization (uses the same backtester loop)
