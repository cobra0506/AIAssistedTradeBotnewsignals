"""Strategy: RL Hybrid

Modes:
- rl_only: use RL signal directly.
- rl_confirm: RL signal is allowed only when a second strategy confirms it.

This is additive integration for Phase 3:
- Keeps strict signal schema.
- Does not change backtester/paper-trader core paths."""

from simple_strategy.strategies.builder_presets import build_strategy_preset

STRATEGY_PARAMETERS = {
    "rl_mode": {
        "type": "str",
        "default": "rl_only",
        "options": ["rl_only", "rl_confirm"],
        "description": "RL signal mode",
        "gui_hint": "rl_only = RL standalone, rl_confirm = RL must match confirm strategy",
    },
    "model_path": {
        "type": "str",
        "default": "rl_ai/models/phase2_policy.npz",
        "description": "Path to trained RL model (.npz or .zip)",
        "gui_hint": "Use .npz for linear model, .zip for deep SB3 model",
    },
    "deterministic": {
        "type": "bool",
        "default": True,
        "description": "Use greedy RL action (no sampling)",
        "gui_hint": "Recommended True for stable behavior",
    },
    "entry_timeframe": {
        "type": "str",
        "default": "1m",
        "options": ["1m", "3m", "5m", "15m"],
        "description": "Timeframe where RL emits signals",
        "gui_hint": "Use 1m first for paper testing",
    },
    "confirm_strategy_name": {
        "type": "str",
        "default": "Strategy_1_Trend_Following",
        "description": "Strategy file name used for confirmation mode",
        "gui_hint": "Only used when rl_mode=rl_confirm",
    },
    "indicator_columns": {
        "type": "str",
        "default": "",
        "description": "Comma-separated indicator column names for RL features",
        "gui_hint": "Example: rsi_1,ema_fast_1,ema_slow_1",
    },
    "min_bars": {
        "type": "int",
        "default": 100,
        "min": 10,
        "max": 5000,
        "description": "Minimum candles before strategy can emit non-HOLD",
        "gui_hint": "Use larger values if indicators need warmup",
    },
    "initial_virtual_balance": {
        "type": "float",
        "default": 10000.0,
        "description": "Virtual balance used by internal RL risk guard",
        "gui_hint": "Does not change exchange order sizing",
    },
    "position_size_pct": {
        "type": "float",
        "default": 0.05,
        "min": 0.0,
        "max": 1.0,
        "description": "Virtual position size fraction used by risk guard",
        "gui_hint": "0.05 = 5% per open signal",
    },
    "max_position_size_pct": {
        "type": "float",
        "default": 0.10,
        "min": 0.0,
        "max": 1.0,
        "description": "Hard cap for virtual position size",
        "gui_hint": "Open signals are blocked if effective size is 0",
    },
    "max_drawdown_pct": {
        "type": "float",
        "default": 0.15,
        "min": 0.0,
        "max": 1.0,
        "description": "Hard drawdown stop (fraction)",
        "gui_hint": "0.15 = stop after 15% drawdown",
    },
    "fee_pct": {
        "type": "float",
        "default": 0.0006,
        "min": 0.0,
        "max": 0.01,
        "description": "Virtual per-side fee for risk tracking",
        "gui_hint": "Used only for internal guard accounting",
    },
}


def create_strategy(symbols=None, timeframes=None, **params):
    return build_strategy_preset("Strategy_RL_Hybrid", symbols=symbols, timeframes=timeframes, **params)

if __name__ == "__main__":
    strategy = create_strategy()
    print(strategy.get_strategy_info())
