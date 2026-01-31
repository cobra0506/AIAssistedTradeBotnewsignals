Nutshell Feature Summary (from code)

    Config-driven multi-symbol & multi-interval data collection: The data collection config defines the symbol list and timeframes (intervals), along with collection options like historical depth, WebSocket enablement, and minimum candles for indicators.

    Historical data fetcher + live WebSocket stream: The hybrid collector uses the config’s symbols/timeframes, can fetch all Bybit symbols when enabled, fetches historical data, and (optionally) starts WebSocket live updates for the same symbols/timeframes in a non-blocking way.

    WebSocket subscriptions honor config intervals/symbols: The WebSocket handler subscribes to kline.{interval}.{symbol} pairs using config.TIMEFRAMES and the configured symbols list (or fetched list).

    Backtester consumes collected CSVs for symbols/timeframes: The backtester uses DataFeeder to load symbol/timeframe CSVs, precalculates indicators, then generates signals from the strategy (or builder rules).

    Paper trader uses shared data access for live/latest candles: Paper trading initializes shared data access and pulls latest data (from CSVs produced by data collection), then generates signals via the selected strategy implementation.

How the Data Pipeline Works (Code-Based)

    Config defines symbols/timeframes + collection mode.
    DataCollectionConfig holds SYMBOLS, TIMEFRAMES, DAYS_TO_FETCH, and ENABLE_WEBSOCKET (plus other collection controls).

    Hybrid collector orchestrates both historical + live data.
    HybridTradingSystem.fetch_data_hybrid() uses config values, optionally fetches all Bybit symbols, starts WebSocket streaming, then fetches historical data for the symbol/timeframe matrix.

    WebSocket handler subscribes to the configured symbol/timeframe matrix.
    The WebSocket handler builds kline.{interval}.{symbol} subscriptions from config.TIMEFRAMES and the symbol list, then streams and processes live candles.

Strategy Builder + Signals/Indicators Libraries

    Strategy Builder: Provides a clean API to register indicators and signal rules, with validation of indicator references and rule params (e.g., for RSI or MACD rules).

    Indicators Library: Contains the actual indicator functions (SMA, EMA, RSI, Stochastic, etc.) used by strategies and the builder.

    Signals Library: Contains signal-processing functions that return trading actions (OPEN_LONG, OPEN_SHORT, CLOSE_LONG, CLOSE_SHORT, HOLD), ensuring compatibility with the paper trader and other execution components.

Shared Strategy Interface (Backtester + Paper Trader)

    StrategyBase defines the common interface (generate_signals) and shared behavior/metadata, ensuring a consistent strategy contract for both backtesting and paper trading (and future real trading).

    Backtester executes strategies by calling generate_signals or builder rules on precalculated indicator data.

    Paper trader loads a strategy from the registry and generates signals via strategy.generate_signals(...), using shared data access to supply live/latest data.
