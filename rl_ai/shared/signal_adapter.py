from dataclasses import dataclass
from typing import Dict


VALID_SIGNALS = {
    "OPEN_LONG",
    "CLOSE_LONG",
    "OPEN_SHORT",
    "CLOSE_SHORT",
    "HOLD",
}


ACTION_TO_SIGNAL: Dict[int, str] = {
    0: "HOLD",
    1: "OPEN_LONG",
    2: "CLOSE_LONG",
    3: "OPEN_SHORT",
    4: "CLOSE_SHORT",
}


@dataclass
class PositionState:
    has_position: bool = False
    is_short: bool = False


class SignalAdapter:
    """
    Maps RL actions to the strict project signal schema.

    Action mapping:
    - 0 => HOLD
    - 1 => OPEN_LONG
    - 2 => CLOSE_LONG
    - 3 => OPEN_SHORT
    - 4 => CLOSE_SHORT
    """

    @staticmethod
    def normalize_action(action) -> int:
        try:
            return int(action)
        except (TypeError, ValueError):
            return 0

    @classmethod
    def action_to_signal(
        cls,
        action,
        has_position: bool = False,
        is_short: bool = False,
    ) -> str:
        action_id = cls.normalize_action(action)
        signal = ACTION_TO_SIGNAL.get(action_id, "HOLD")
        return cls._apply_position_rules(signal, has_position=has_position, is_short=is_short)

    @staticmethod
    def signal_to_action(signal: str) -> int:
        for action_id, signal_name in ACTION_TO_SIGNAL.items():
            if signal_name == signal:
                return action_id
        return 0

    @staticmethod
    def _apply_position_rules(signal: str, has_position: bool, is_short: bool) -> str:
        if signal not in VALID_SIGNALS:
            return "HOLD"

        if signal == "HOLD":
            return "HOLD"

        if signal == "OPEN_LONG":
            return "HOLD" if has_position else "OPEN_LONG"

        if signal == "OPEN_SHORT":
            return "HOLD" if has_position else "OPEN_SHORT"

        if signal == "CLOSE_LONG":
            if not has_position or is_short:
                return "HOLD"
            return "CLOSE_LONG"

        if signal == "CLOSE_SHORT":
            if not has_position or not is_short:
                return "HOLD"
            return "CLOSE_SHORT"

        return "HOLD"
