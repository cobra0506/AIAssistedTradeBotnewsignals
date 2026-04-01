# Auto Evolve Campaign Plan

## Goal
- Run auto evolve in an organized way
- Find 1 or more candidate strategies that:
  - pass the final gate
  - have non-negative final return
  - are worth later backtest and paper trade

## Why We Are Doing This
- The full strategy space is too large to brute-force
- Mixed runs can let one family dominate too early
- We need a repeatable way to work through the strategy space in sections

## Current Situation
- `rsi_osob` is the only family that has been surviving mixed overnight runs
- `ema_cross` and `sma_cross` are being generated, but so far they are not surviving to finalist level
- Even the stronger `rsi_osob` finalists are still failing the final out-of-sample return check

## Campaign Structure
- We use focused family campaigns first
- Each campaign tests one strategy family at a time
- Initial overnight campaign configs:
  - `simple_strategy/auto_evolve/configs/overnight_rsi_osob.json`
  - `simple_strategy/auto_evolve/configs/overnight_ema_cross.json`
  - `simple_strategy/auto_evolve/configs/overnight_sma_cross.json`

## What Counts As Success
- Minimum success for a run:
  - at least 1 finalist
  - finalist passes final gate
  - final return is `>= 0.0%`
- Stronger success:
  - more than 1 final-passed candidate
  - enough trades to be meaningful
  - candidate looks stable enough for later backtest and paper trade

## How To Run
- Example commands:
```powershell
python -m simple_strategy.auto_evolve.run_evolution --config simple_strategy/auto_evolve/configs/overnight_rsi_osob.json --population 8 --generations 6 --workers 1 --max-runtime-hours 9.5
python -m simple_strategy.auto_evolve.run_evolution --config simple_strategy/auto_evolve/configs/overnight_ema_cross.json --population 8 --generations 6 --workers 1 --max-runtime-hours 9.5
python -m simple_strategy.auto_evolve.run_evolution --config simple_strategy/auto_evolve/configs/overnight_sma_cross.json --population 8 --generations 6 --workers 1 --max-runtime-hours 9.5
```

## How To Judge Each Run
- Check:
  - `summary.txt`
  - `top10_metrics.csv`
  - `progress.json`
  - `top_results.json`
- Questions to answer:
  - did any candidate pass the final gate?
  - which family survived?
  - did the run mostly fail on precheck, timeout, or final return?

## Immediate Next Steps
1. Run the 3 focused overnight family campaigns.
2. Compare which family:
   - survives precheck
   - avoids timeouts
   - produces final-passed candidates
3. Keep the winning family profile and tune it further.
4. Keep the losing family profiles for later fixes instead of guessing.

## Longer-Term Plan
1. Add more campaign configs for other family groups.
2. Use the campaign registry to track what has already been tested.
3. Penalize or skip already-tested finalist fingerprints.
4. Run selected campaigns in parallel, but keep each run light enough to avoid timeouts.
5. Build a library of final-passed strategies for later backtest and paper trade.

## Important Reality Check
- We will not test every mathematical combination in the full universe of possibilities.
- We can still cover the strategy space systematically by:
  - splitting it into families
  - running multiple seeds
  - keeping records
  - promoting only strategies that pass out-of-sample checks

## Decision Rule
- If a family campaign repeatedly produces:
  - `0` survivors, or
  - only negative final returns,
- then do not keep burning overnight runs on it without changing that family profile.
