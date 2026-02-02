# Signals Library

## Purpose
Signal processing helpers for translating indicator series into trading signals.

## Key File
- `simple_strategy/strategies/signals_library.py`

## Examples
- Overbought/oversold signals
- MA crossovers
- MACD, Bollinger, stochastic signals

## Known Issues (Current)
- Return types are inconsistent across functions (strings vs numeric).
- Some helper functions expect specific parameter shapes and can mis-handle inputs.

## Recommendation
Use explicit strategy classes or normalize signal outputs before combining.

