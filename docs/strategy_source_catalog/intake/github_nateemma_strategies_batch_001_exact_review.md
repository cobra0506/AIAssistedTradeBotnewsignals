# Batch 001 Exact Review: nateemma/strategies

- Source place: [github_nateemma_strategies.md](../places/github_nateemma_strategies.md)
- Batch: `github_nateemma_strategies_batch_001`
- Review standard: [exact_port_standard.md](../exact_port_standard.md)

## Verdict
- These first 10 files are **not standalone rule strategies**
- They are thin wrappers over a shared ML/anomaly framework
- So they are **not directly portable as exact standalone strategies**

## What The 10 Files Actually Do
Each file only sets:
- a subclass of `Anomaly`
- one shared `signal_type`

Examples:
- `Anomaly_adx.py` -> `SignalType.ADX`
- `Anomaly_all.py` -> `SignalType.ALL`
- `Anomaly_aroon.py` -> `SignalType.Aroon`
- `Anomaly_bbw.py` -> `SignalType.Bollinger_Width`

## Where The Real Logic Lives
- Shared strategy base:
  - `Anomaly/Anomaly.py`
- Shared training-signal definitions:
  - `utils/TrainingSignals.py`
- Shared indicator construction:
  - `utils/DataframePopulator.py`

## Why They Are Not Direct Exact Ports
- The live trade logic depends on anomaly-model predictions:
  - `predict_buy > 0.5`
  - `predict_sell > 0.5`
- Those predictions come from trained classifiers
- The training labels come from future-looking signal functions
- Future-looking label logic cannot be used directly as live trade logic in this backtester

## Portability Status Per File
1. `Anomaly/Anomaly_adx.py`
   - Status: `not_directly_portable`
   - Reason: wrapper over shared Anomaly classifier framework
2. `Anomaly/Anomaly_all.py`
   - Status: `not_directly_portable`
   - Reason: wrapper over shared Anomaly classifier framework
3. `Anomaly/Anomaly_aroon.py`
   - Status: `not_directly_portable`
   - Reason: wrapper over shared Anomaly classifier framework
4. `Anomaly/Anomaly_bbw.py`
   - Status: `not_directly_portable`
   - Reason: wrapper over shared Anomaly classifier framework
5. `Anomaly/Anomaly_dwt.py`
   - Status: `not_directly_portable`
   - Reason: wrapper over shared Anomaly classifier framework
6. `Anomaly/Anomaly_fbb.py`
   - Status: `not_directly_portable`
   - Reason: wrapper over shared Anomaly classifier framework
7. `Anomaly/Anomaly_fwr.py`
   - Status: `not_directly_portable`
   - Reason: wrapper over shared Anomaly classifier framework
8. `Anomaly/Anomaly_highlow.py`
   - Status: `not_directly_portable`
   - Reason: wrapper over shared Anomaly classifier framework
9. `Anomaly/Anomaly_jump.py`
   - Status: `not_directly_portable`
   - Reason: wrapper over shared Anomaly classifier framework
10. `Anomaly/Anomaly_macd.py`
   - Status: `not_directly_portable`
   - Reason: wrapper over shared Anomaly classifier framework

## Recommendation
- Do not restart exact-port work from these 10 as if they are ordinary rule files
- Either:
  - port the entire shared Anomaly framework as a separate project
  - or move to the next GitHub source files that contain direct rule logic
