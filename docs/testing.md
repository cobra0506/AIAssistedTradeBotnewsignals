# Testing

## Purpose
Validate indicator calculations, signal correctness, and component integration to reduce regressions.

## What’s Covered
- Signal library behavior
- Calculation accuracy (P&L, metrics, risk sizing)
- Integration tests across strategies + backtester

## Where Tests Live
`tests/` contains unit/integration suites (e.g., signal tests, backtester tests).

## Running
```bash
# If pytest is installed
python -m pytest
```
Some tests also run under `unittest` style in `tests/`.
