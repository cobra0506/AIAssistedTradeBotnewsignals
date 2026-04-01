# GitHub Intake Batch: mementum/backtrader batch 001

- Source place: `github_mementum_backtrader_samples`
- Status: `queued`
- Goal:
  - review the first exact rule-based or clearly strategy-like backtrader samples
  - port only the ones with real entry and exit logic
  - skip pure framework demos if they have no reusable trade rules

## First 10 candidate files
1. `samples/btfd/btfd.py`
2. `samples/cheat-on-open/cheat-on-open.py`
3. `samples/kselrsi/ksignal.py`
4. `samples/lrsi/lrsi-test.py`
5. `samples/macd-settings/macd-settings.py`
6. `samples/multidata-strategy/multidata-strategy.py`
7. `samples/multidata-strategy/multidata-strategy-unaligned.py`
8. `samples/pinkfish-challenge/pinkfish-challenge.py`
9. `samples/psar/psar.py`
10. `samples/sigsmacross/sigsmacross.py`

## Intake notes
- Some backtrader samples are execution demos, not true strategies.
- This batch is the first pass to separate:
  - real strategy logic
  - indicator demos
  - broker and order examples
