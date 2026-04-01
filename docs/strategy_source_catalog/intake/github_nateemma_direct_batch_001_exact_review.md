# Exact Review: nateemma direct batch 001

## Scope
- Source batch: `github_nateemma_direct_batch_001.md`
- Port standard: `exact_port_standard.md`
- New helper: `imported_nateemma_direct_batch1_helper.py`

## What was ported exactly
- The real rule conditions from these source files:
  - `archived/ADXDM.py`
  - `archived/BBBHold.py`
  - `archived/BBKCBounce.py`
  - `archived/BTCBigDrop.py`
  - `archived/BTCEMABounce.py`
  - `archived/BTCJump.py`
  - `archived/BTCMACDCross.py`
  - `archived/BTCNDrop.py`
  - `archived/BTCNSeq.py`
  - `archived/BigDrop.py`

## Explicit project adaptation
- The source strategies are mainly long-only.
- For this project they were adapted in one explicit, repeatable way:
  - uptrend -> allow only new longs
  - downtrend -> allow only new shorts
  - sideways -> no new entries
- Close signals stay allowed.
- For long-only source files, the short side is the mirrored opposite of the same source rule family.

## Benchmark symbol handling
- BTC-led source strategies now use `BTCUSDT` as the informative benchmark symbol.
- If `BTCUSDT` data is missing, those strategies return `HOLD` instead of guessing.

## Status
- `ADXDM` -> exact port ready
- `BBBHold` -> exact port ready
- `BBKCBounce` -> exact port ready
- `BTCBigDrop` -> exact port ready
- `BTCEMABounce` -> exact port ready
- `BTCJump` -> exact port ready
- `BTCMACDCross` -> exact port ready
- `BTCNDrop` -> exact port ready
- `BTCNSeq` -> exact port ready
- `BigDrop` -> exact port ready
