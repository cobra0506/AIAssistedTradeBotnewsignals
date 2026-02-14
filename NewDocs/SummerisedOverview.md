Nutshell Feature Summary (Code-Verified)

- Config-driven multi-symbol & multi-interval data collection: `DataCollectionConfig` defines symbols, timeframes, historical depth, and WebSocket enablement.

- Hybrid data collection: historical fetch + optional live WebSocket streaming for the same symbol/timeframe matrix.

- WebSocket subscriptions: built from configured timeframes and symbols (kline.{interval}.{symbol}).

- Backtester loads CSVs via `DataFeeder`, precalculates indicators, generates signals on bar close, queues orders for next-bar open by default, and can trigger intrabar stops/targets or liquidation using high/low.

- Paper trader loads a strategy from the registry and generates signals per symbol; uses shared WebSocket data when available and CSV fallback otherwise.

- Optimizer supports walk-forward runs and saves results to `simple_strategy/optimization_results/walk_forward_results.json`.

Data Pipeline (Current)

1) Config defines symbols/timeframes + collection mode.

2) Hybrid collector fetches historical data and optionally starts WebSocket streaming.

3) WebSocket handler subscribes to the symbol/timeframe matrix and writes candles to CSV.

4) `DataFeeder` loads CSVs for backtesting.

5) Paper trader consumes live data (shared WebSocket) or CSV fallback.

Strategy Builder + Signals/Indicators

- Strategy Builder: Available but has known limitations around indicator injection and signal parameter passing.

- Indicators Library: Core indicator functions used by strategies (SMA, EMA, RSI, etc.).

- Signals Library: Contains multiple signal helpers with mixed return types (string vs numeric). Use with care or normalize outputs.

Shared Strategy Interface (Backtester + Paper Trader)

- `StrategyBase` defines the common interface (`generate_signals`) for strategies.

- Backtester uses `generate_signals_vectorized` when available for speed.

- Paper trader uses `generate_signals_vectorized` when available, otherwise falls back to `generate_signals`.
