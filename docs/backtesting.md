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

2. Pre-calculate indicators (currently EMA/RSI in `_precalculate_indicators`).

3. Generate signals on each bar close (builder rules or `generate_signals_vectorized`).

4. Queue orders and fill on the next bar open by default (`execution_delay_bars`).

5. Optional intrabar stop/target checks use bar high/low (`use_intrabar_stops`).

6. Apply spread/slippage/fees and risk-based sizing.

7. Close remaining positions and compute metrics.

## Realism Settings

- `execution_delay_bars`: Bars to wait before filling a signal (default 1).

- `use_intrabar_stops`: Use bar high/low to trigger stops/targets (default True).

- `stop_loss_pct` / `take_profit_pct`: Optional stop/target percent from entry.

- `enable_liquidation`: Force close when liquidation price is hit (default True).
- `leverage`: Leverage used for liquidation math (default 5.0).
- `maintenance_margin_pct`: Maintenance margin percent (default 0.005 = 0.5%).

- `spread_pct`, `slippage_pct`, `fee_pct`: Execution/fee simulation.

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

## Integration Points

- Strategy Builder (strategy objects)

- Data Collection (CSV data)

- Optimization (uses the same backtester loop)
