# Testing

## Purpose
Validate indicator calculations, signal correctness, and component integration to reduce regressions.

## What's Covered
- Signal library behavior
- Calculation accuracy (PnL, metrics, risk sizing)
- Integration tests across strategies, backtester, and paper trader paths

## Where Tests Live
`tests/` contains smoke suites and RL suites.

## Running
```bash
# Syntax/import check
python -m compileall .

# Full project test suite (recommended)
python -m unittest discover -s tests -p "test_*.py" -v

# Optional split runs
python -m unittest discover -s tests/smoke -p "test_*.py" -v
python -m unittest discover -s tests/rl_ai_tests -p "test_*.py" -v
```

`pytest` can still be used if installed, but `unittest` is the baseline in this repo.

## Important Regression Coverage
- Builder-generated strategies are validated in backtester smoke tests.
- Paper trader logging is validated to survive Windows Unicode console encode errors.
