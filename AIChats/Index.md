# AI Chats Index (Updated 2026-02-18 07:39)

## Status
- Reviewed files: `AIChats/MyPrompt.md`, `AIChats/DeepSeek.md`, `AIChats/ZAI.md`, `AIChats/Grok.md`.
- New replies are present (all updated at `07:39`).
- This file is the merged hub view.

## Goal
- Phase 1: get good usable strategies fast.
- Phase 2: get more unique and advanced strategies.
- Hard rule: keep backtester and paper trader aligned with same strategy files and same signal schema.

## User Constraints (Locked)
1. Runtime:
   - Overnight: 8-12h max
   - Multi-day: 2-4d max
2. Mode: balanced (speed + realism)
3. Gates: relaxed first pass, stricter promotion pass
4. Hardware:
   - i5-11600K (6C/12T), 32 GB RAM, RTX 5060 Ti 16 GB
5. Data universe:
   - 18 symbols
   - 1m and 5m

## What Changed In New AI Replies
1. All 3 now give stronger phased plans.
2. All 3 strongly prioritize profiling + speed work first.
3. All 3 propose random high-volume candidate generation before advanced search.

## My Assessment Of Each Reply
1. `AIChats/Grok.md`
   - Strongest on practical sequencing and impact/effort tradeoffs.
   - Best for immediate execution order.
   - Some file names are generic guesses (example: `auto_evolve/evolve_runner.py`).
2. `AIChats/DeepSeek.md`
   - Strongest on validation and overfit controls.
   - Useful gate/walk-forward structure.
   - Also assumes some modules not in repo yet.
3. `AIChats/ZAI.md`
   - Strongest on architecture patterns and code skeletons.
   - Useful for designing new generator modules.
   - Several examples are conceptual and need repo-specific mapping.

## Repo-Fit Notes
1. Existing `auto_evolve` entry is `simple_strategy/auto_evolve/run_evolution.py` (not `evolve_runner.py`).
2. Existing modules include:
   - `simple_strategy/auto_evolve/candidate_builder.py`
   - `simple_strategy/auto_evolve/evaluator.py`
   - `simple_strategy/auto_evolve/gates.py`
3. New modules suggested by AIs are still ideas and would need to be added deliberately.

## Adopt Now / Later
1. Adopt now:
   - Profile baseline first.
   - Parallel candidate evaluation.
   - Indicator/result caching.
   - Two-stage gates (quick filter then strict filter).
2. Adopt next:
   - Random strategy factory for 1000+ candidates.
   - Walk-forward and stronger out-of-sample checks.
3. Adopt later:
   - Full grammar/GP structure evolution.
   - Regime-specific scoring and multi-symbol portfolio logic.

## Practical 5-Symbol Starter Basket
1. `BNBUSDT`
2. `XRPUSDT`
3. `ADAUSDT`
4. `AAVEUSDT`
5. `ARBUSDT`

## Recommended Gates (Two-Level)
1. Level A (discovery, relaxed):
   - Return >= 5%
   - Max drawdown <= 15%
   - Win rate >= 45%
   - Trades >= 1/day average
2. Level B (promotion, strict):
   - Return >= 10%
   - Max drawdown <= 10%
   - Win rate >= 50%
   - Trades >= 1/day average

## Invariants
1. Signal schema stays:
   - `OPEN_LONG`, `CLOSE_LONG`, `OPEN_SHORT`, `CLOSE_SHORT`, `HOLD`
2. Backtester and paper trader must remain compatible with same strategy files.
