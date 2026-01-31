"""
Strategy: Test Bidirectional Strategy (Updated for OPEN/CLOSE schema)

Trade logic (simple):
- In test mode, alternate signals to validate both long and short paths.
- If no position -> OPEN_LONG.
- Next signal -> CLOSE_LONG and OPEN_SHORT.
- Next signal -> CLOSE_SHORT and OPEN_LONG.
- This ensures both sides (long/short) are exercised in backtester/paper trader.
"""

import os
import sys
import logging
from typing import Dict, List, Any

import pandas as pd

from simple_strategy.shared.strategy_base import StrategyBase

# Add parent directories to path for proper imports when run directly
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

logger = logging.getLogger(__name__)

STRATEGY_PARAMETERS = {
    'test_mode': {
        'type': 'bool',
        'default': True,
        'description': 'Enable test mode (cycles long/short signals)',
        'gui_hint': 'When enabled, cycles OPEN/CLOSE to test execution paths'
    }
}


class TestBidirectionalStrategy(StrategyBase):
    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        super().__init__(
            name="Test_Bidirectional_Strategy",
            symbols=symbols,
            timeframes=timeframes,
            config=config
        )
        self.test_mode = config.get('test_mode', True)
        self._position_state: Dict[tuple, Dict[str, Any]] = {}
        self._toggle_state: Dict[tuple, bool] = {}

    def _apply_position_rules(self, position_key: tuple, raw_signal: str) -> str:
        position = self._position_state.get(position_key)

        if raw_signal == 'OPEN_LONG':
            if position is not None:
                return 'HOLD'
            self._position_state[position_key] = {'is_short': False}
            return raw_signal

        if raw_signal == 'OPEN_SHORT':
            if position is not None:
                return 'HOLD'
            self._position_state[position_key] = {'is_short': True}
            return raw_signal

        if raw_signal == 'CLOSE_LONG':
            if position is None or position.get('is_short', False):
                return 'HOLD'
            self._position_state.pop(position_key, None)
            return raw_signal

        if raw_signal == 'CLOSE_SHORT':
            if position is None or not position.get('is_short', False):
                return 'HOLD'
            self._position_state.pop(position_key, None)
            return raw_signal

        return 'HOLD'

    def generate_signals(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, str]]:
        signals: Dict[str, Dict[str, str]] = {}

        for symbol in data:
            signals[symbol] = {}
            for timeframe, df in data[symbol].items():
                if df is None or len(df) < 2:
                    signals[symbol][timeframe] = 'HOLD'
                    continue

                position_key = (symbol, timeframe)
                position = self._position_state.get(position_key)
                toggle = self._toggle_state.get(position_key, False)

                if not self.test_mode:
                    signals[symbol][timeframe] = 'HOLD'
                    continue

                if position is None:
                    raw_signal = 'OPEN_LONG'
                elif not position.get('is_short', False):
                    raw_signal = 'CLOSE_LONG' if not toggle else 'OPEN_SHORT'
                else:
                    raw_signal = 'CLOSE_SHORT' if not toggle else 'OPEN_LONG'

                # Flip toggle each call to cycle behaviors
                self._toggle_state[position_key] = not toggle

                signals[symbol][timeframe] = self._apply_position_rules(position_key, raw_signal)

        return signals


def create_strategy(symbols=None, timeframes=None, **params):
    if symbols is None or len(symbols) == 0:
        symbols = ['BTCUSDT']
    if timeframes is None or len(timeframes) == 0:
        timeframes = ['1m']

    config = {
        'test_mode': params.get('test_mode', True)
    }
    return TestBidirectionalStrategy(symbols, timeframes, config)
