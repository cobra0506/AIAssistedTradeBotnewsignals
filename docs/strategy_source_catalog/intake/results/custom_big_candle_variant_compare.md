# Big Candle Variant Compare

- Window used: latest 31 days

## Direct compare
- `2.0%` trigger, `0.5x` target:
  - return `%`: `1.6412`
  - trades: `45`
  - sharpe: `5.0398`
- `3.0%` trigger, `1.0x` full-candle target:
  - return `%`: `2.3710`
  - trades: `14`
  - sharpe: `36.9647`
- `3.0%` trigger, `1.0x` full target, `max_hold_bars=4`, `min_body_ratio=0.5`, `cooldown_bars=2`:
  - return `%`: `-0.1132`
  - trades: `16`
  - sharpe: `-3.8766`

## Read
- The plain `3% + full target` version beat the current `2% + half target` on the latest unseen month.
- But that same `3% + full target` version was weak on the prior train month.
- The extra filters did not help. They made the strategy worse on unseen data.
