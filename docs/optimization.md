# Optimization

## Purpose
Automate parameter tuning for strategies using Bayesian optimization (Optuna).

## Key Components
- `simple_strategy/optimization/bayesian_optimizer.py`: Runs Optuna study and backtests.
- `simple_strategy/optimization/parameter_space.py`: Defines search space.
- `simple_strategy/trading/parameter_manager.py`: Saves optimized params to JSON.

## How It Works
1. Build a `ParameterSpace` from strategy parameters.
2. For each trial, create a strategy instance and run a backtest.
3. Score the trial by metric (e.g., sharpe_ratio).
4. Persist best parameters (with `last_optimized` date) via `ParameterManager`.

## Usage
```python
optimizer = BayesianOptimizer(data_feeder=data_feeder, n_trials=50)
best_params, best_score, duration_sec = optimizer.optimize(
    strategy_name='Strategy_1_Trend_Following',
    parameter_space=param_space,
    symbols=['BTCUSDT'],
    timeframes=['1m'],
    start_date='2023-01-01',
    end_date='2023-12-31',
    metric='sharpe_ratio'
)
```

## Outputs
- `optimization_results/trials.csv`
- `optimization_results/winning_summary.txt` (includes duration)
- Saved parameters via `ParameterManager` (includes `last_optimized`)
