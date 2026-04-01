# Exact Port Standard

This is the new standard for imported strategies.

## Goal
- Read the source strategy first
- Extract the real trade logic
- Rebuild that same logic in this backtester
- Keep this project's regime rule:
  - uptrend -> only `OPEN_LONG`
  - downtrend -> only `OPEN_SHORT`
  - sideways -> `HOLD`

## Rules
- Do not create proxy strategies and call them exact ports
- Do not simplify ML, anomaly, or framework logic into hand-made rules unless the file is explicitly marked as an approximation
- If the source strategy is only a thin wrapper and the real logic lives in a shared base class, analyze the shared base class too
- If the source strategy depends on model training, anomaly detection, or unavailable framework internals, mark it as:
  - `not_directly_portable`
  - with the exact reason

## Workflow
1. Read the source file
2. Read any shared base classes it depends on
3. Identify whether the strategy is:
   - direct rule-based
   - shared-base rule-based
   - ML/anomaly/model-based
4. If direct rule-based:
   - rebuild it in this backtester
   - test it
5. If ML/anomaly/model-based:
   - do not fake an exact port
   - record why it cannot be ported exactly without the shared framework

## Output Required Per Batch
- source files reviewed
- actual trade logic summary
- portability status
- if portable:
  - new strategy file(s)
  - backtest summary
- if not portable:
  - exact blocker
  - recommendation for next source
