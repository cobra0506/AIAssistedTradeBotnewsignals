# Exact Review: nateemma direct batch 002

## Scope
- Source batch: `github_nateemma_direct_batch_002.md`
- Port standard: `exact_port_standard.md`
- Helper: `imported_nateemma_direct_batch2_helper.py`

## What was ported
- `archived/BollingerBounce.py`
- `archived/BuyDips.py`
- `archived/DCBBBounce.py`
- `archived/DonchianBounce.py`
- `archived/DonchianChannel.py`
- `archived/EMA003.py`
- `archived/EMA50.py`
- `archived/EMABounce.py`
- `archived/EMABreakout.py`
- `archived/EMACross.py`

## Explicit project adaptation
- uptrend -> allow only new longs
- downtrend -> allow only new shorts
- sideways -> no new entries
- For long-only source files, the short side is the mirrored opposite of the same source rule family.
