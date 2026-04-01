"""Strategy: Test Bidirectional Strategy (Updated for OPEN/CLOSE schema)

Trade logic (simple):
- In test mode, alternate signals to validate both long and short paths.
- If no position -> OPEN_LONG.
- Next signal -> CLOSE_LONG and OPEN_SHORT.
- Next signal -> CLOSE_SHORT and OPEN_LONG.
- This ensures both sides (long/short) are exercised in backtester/paper trader."""

from simple_strategy.strategies.builder_presets import build_strategy_preset

STRATEGY_PARAMETERS = {
    'test_mode': {
        'type': 'bool',
        'default': True,
        'description': 'Enable test mode (cycles long/short signals)',
        'gui_hint': 'When enabled, cycles OPEN/CLOSE to test execution paths'
    }
}


def create_strategy(symbols=None, timeframes=None, **params):
    return build_strategy_preset("Strategy_test_bidirectional_strategy", symbols=symbols, timeframes=timeframes, **params)

if __name__ == "__main__":
    strategy = create_strategy()
    print(strategy.get_strategy_info())
