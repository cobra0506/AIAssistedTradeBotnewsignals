# Strategy Source Catalog

Last updated: 2026-03-27

## Goal
- Build a master intake list of outside strategy sources
- Keep one file per source place
- Work through the places in order
- Port promising strategy logic into this project later

## How To Use This Catalog
1. Pick one place file.
2. Work through its listed strategies in batches.
3. Port the first 10 into our backtester.
4. Backtest, optimize, and paper trade only the ones that survive.
5. Move to the next 10, then the next place.

## Place Status Meanings
- `enumerated`: strategy/example names were pulled and listed
- `partial`: the place was catalogued, but not every candidate is extracted yet
- `pending_manual_enumeration`: the place is real and in scope, but the source is dynamic/community-driven and still needs manual intake passes

## GitHub Places
- [freqtrade/freqtrade templates](./places/github_freqtrade_freqtrade_templates.md)
- [nateemma/strategies](./places/github_nateemma_strategies.md)
- [mementum/backtrader samples](./places/github_mementum_backtrader_samples.md)
- [kernc/backtesting.py examples](./places/github_kernc_backtesting_py_examples.md)
- [polakowo/vectorbt examples](./places/github_polakowo_vectorbt_examples.md)
- [tradingstrategy-ai/getting-started notebooks](./places/github_tradingstrategy_ai_getting_started.md)

## TradingView Places
- [TradingView public strategy scripts](./places/tradingview_public_strategies.md)

## Forums And Shared-Idea Places
- [Forex Factory forum](./places/forum_forex_factory.md)
- [futures.io forum](./places/forum_futures_io.md)
- [Elite Trader forums](./places/forum_elite_trader.md)
- [Trade2Win community](./places/forum_trade2win.md)
- [Reddit r/algotrading](./places/forum_reddit_algotrading.md)
- [Reddit r/Daytrading](./places/forum_reddit_daytrading.md)

## Current High-Level Findings
- `rsi_osob`-style ideas have been the only ones surviving the current overnight auto-evolve search
- `ema_cross` and `sma_cross` did not survive the current overnight precheck
- Because of that, the outside-strategy intake path is now the preferred next path

## Recommended Order
1. GitHub concrete repos with ready-made strategy code
2. TradingView public strategy scripts
3. Framework example libraries
4. Forums and shared ideas

## Notes
- "All" means all places in this catalog version.
- Some community sources are too dynamic to auto-enumerate fully in one pass.
- Those places are still included now, with their own place files, and marked clearly.
