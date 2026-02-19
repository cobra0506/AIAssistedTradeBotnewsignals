I need a strict technical review of what was implemented in my trading bot project today. Please challenge assumptions and try to break it mentally.

Project constraints:
- Backtester and Paper Trader share core modules; changes must not break either.
- Signal schema is non-negotiable: OPEN_LONG, CLOSE_LONG, OPEN_SHORT, CLOSE_SHORT, HOLD.
- Goal: find usable profitable strategies faster, then find more unique/advanced strategies.

What is now implemented:

1) Backtester speed upgrade
- File: simple_strategy/backtester/backtester_engine.py
- Hot loop optimized (less per-row overhead, array access improvements).
- Added print_trade_logs switch so heavy runs avoid trade-by-trade print overhead.
- Indicator-validity handling hardened for edge cases.
- Benchmark observed: total runtime around 27.69s -> 6.87s on same test set (~4x faster), with matching metrics.

2) Fast Strategy Factory (lego style) added
- Files:
  - simple_strategy/fast_finder/runner.py
  - simple_strategy/fast_finder/gui.py
  - simple_strategy/fast_finder/__init__.py
- GUI launch integrated in dashboard (button opens new window).
- Generates many random candidate lego strategies, runs parallel backtests, ranks top K, writes reports.
- Outputs:
  - top_results.json
  - summary.txt
  - top_metrics.csv
  - strategy files under run folder
  - optional publish of top N into simple_strategy/strategies

3) Basic and Advanced lego search profiles
- basic: existing search space behavior
- advanced: expanded search space with richer blocks/signals (MACD, Bollinger, stochastic, breakout, divergence)
- Strategy builder compatibility fixes for component mapping/validation were added:
  - simple_strategy/strategies/strategy_builder.py

4) A/B quality presets added
- Preset A (5% target profile) and Preset B (10% target profile), plus custom mode.
- GUI has buttons:
  - APPLY PRESET A (5%)
  - APPLY PRESET B (10%)

5) Advanced context filters added
- Multi-timeframe context filter (higher timeframe trend gating lower timeframe entries).
- Multi-symbol regime filter (market regime symbol gating entries).
- Configurable fast/slow EMA context periods and regime symbol.

6) Tests
- Smoke tests pass:
  - tests/smoke/test_fast_finder_smoke.py
  - tests/smoke/test_backtester_smoke.py
  - tests/smoke/test_paper_trader_import_smoke.py

What I need from you:
1) Find design flaws, hidden bugs, and reliability risks.
2) Check if any part can create false profitability or backtest/live mismatch.
3) Check if advanced filters are implemented in a robust way or only superficially.
4) Check whether defaults are likely too strict/too loose (for example advanced + preset can sometimes produce zero trades).
5) Suggest concrete fixes with file/function-level precision.

Required response format:
1) Top critical issues first (severity: Critical/High/Medium/Low).
2) For each issue: why it matters, how it can fail, and exact fix.
3) A prioritized action plan:
- Must fix now
- Should fix next
- Nice to have
4) Additional tests I should add immediately (exact test ideas/cases).
5) If you think architecture should change, propose the smallest safe migration path.
