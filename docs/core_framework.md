# Core Framework

## Purpose
Provide foundational interfaces and utilities shared across strategies, backtesting, and trading.

## Key Components
- `simple_strategy/shared/strategy_base.py`: Base class + common strategy utilities.
- `simple_strategy/shared/data_feeder.py`: Historical data loading and caching.

## Responsibilities
- Standard strategy interface (`generate_signals`).
- Position sizing and basic risk validation helpers.
- Data access abstraction for backtesting and strategy testing.

## Integration Points
- Strategy files extend `StrategyBase`.
- Backtester consumes strategies built on this base.
- Data collection writes CSVs consumed via `DataFeeder`.

