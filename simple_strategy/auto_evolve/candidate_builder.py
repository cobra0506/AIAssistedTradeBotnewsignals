import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd

from simple_strategy.strategies import indicators_library, signals_library
from simple_strategy.strategies.strategy_builder import StrategyBuilder


@dataclass
class GeneSpec:
    key: Tuple[str, ...]
    kind: str
    low: float = 0.0
    high: float = 1.0
    step: float = 1.0
    choices: List[Any] = None


class CandidateBuilder:
    """Encode/decode candidate genomes and build executable strategies."""

    def __init__(self, search_space: Dict[str, Any], max_active_signals: int = 3):
        self.search_space = deepcopy(search_space)
        self.max_active_signals = max_active_signals
        self._component_to_indicators = self._build_component_map()
        self.gene_specs: List[GeneSpec] = self._build_gene_specs()

    def _build_component_map(self) -> Dict[str, List[str]]:
        component_map: Dict[str, List[str]] = {}
        for indicator_id, indicator_def in self.search_space.get("indicators", {}).items():
            func_name = indicator_def.get("function")
            components: List[str] = []
            if func_name == "macd":
                components = ["macd_line", "signal_line", "histogram"]
            elif func_name == "bollinger_bands":
                components = ["upper_band", "middle_band", "lower_band"]
            elif func_name == "stochastic":
                components = ["k_percent", "d_percent"]

            for component in components:
                component_map.setdefault(component, []).append(indicator_id)
        return component_map

    def _build_gene_specs(self) -> List[GeneSpec]:
        specs: List[GeneSpec] = []

        for signal_id in self.signal_ids:
            specs.append(
                GeneSpec(
                    key=("signal_enabled", signal_id),
                    kind="binary",
                    low=0,
                    high=1,
                    step=1,
                )
            )

        combo_choices = self.search_space.get("signal_combination_methods", ["majority_vote"])
        specs.append(
            GeneSpec(
                key=("signal_combination",),
                kind="categorical_index",
                low=0,
                high=max(0, len(combo_choices) - 1),
                step=1,
                choices=combo_choices,
            )
        )

        for indicator_id, indicator_def in self.search_space["indicators"].items():
            for param_name, param_def in indicator_def.get("params", {}).items():
                specs.append(self._param_gene(("indicator_params", indicator_id, param_name), param_def))

        for signal_id, signal_def in self.search_space["signals"].items():
            for param_name, param_def in signal_def.get("params", {}).items():
                specs.append(self._param_gene(("signal_params", signal_id, param_name), param_def))

        return specs

    @staticmethod
    def _param_gene(key: Tuple[str, ...], param_def: Dict[str, Any]) -> GeneSpec:
        ptype = param_def["type"]
        if ptype == "int":
            return GeneSpec(key=key, kind="int", low=param_def["low"], high=param_def["high"], step=param_def.get("step", 1))
        if ptype == "float":
            return GeneSpec(key=key, kind="float", low=param_def["low"], high=param_def["high"], step=param_def.get("step", 0.1))
        if ptype == "choice":
            choices = list(param_def.get("choices", []))
            return GeneSpec(key=key, kind="categorical_index", low=0, high=max(0, len(choices) - 1), step=1, choices=choices)
        raise ValueError(f"Unsupported parameter type: {ptype}")

    @property
    def signal_ids(self) -> List[str]:
        return list(self.search_space.get("signals", {}).keys())

    def random_genome(self, rng: random.Random) -> List[float]:
        genome: List[float] = []
        for spec in self.gene_specs:
            if spec.kind in {"binary", "int", "categorical_index"}:
                genome.append(rng.randint(int(spec.low), int(spec.high)))
            elif spec.kind == "float":
                steps = int(round((spec.high - spec.low) / spec.step))
                pick = rng.randint(0, max(0, steps))
                genome.append(spec.low + pick * spec.step)
            else:
                raise ValueError(f"Unsupported gene kind: {spec.kind}")
        return genome

    def mutate_gene(self, value: float, spec: GeneSpec, rng: random.Random) -> float:
        if spec.kind == "binary":
            return 1 - int(round(value))
        if spec.kind in {"int", "categorical_index"}:
            step = int(spec.step)
            delta = rng.choice([-step, step])
            return int(min(spec.high, max(spec.low, int(round(value)) + delta)))
        if spec.kind == "float":
            delta = rng.choice([-1.0, 1.0]) * spec.step
            raw = float(value) + delta
            clamped = min(spec.high, max(spec.low, raw))
            return round(clamped / spec.step) * spec.step
        raise ValueError(f"Unsupported gene kind: {spec.kind}")

    def decode(self, genome: List[float]) -> Dict[str, Any]:
        if len(genome) != len(self.gene_specs):
            raise ValueError(f"Genome length mismatch: got {len(genome)}, expected {len(self.gene_specs)}")

        candidate: Dict[str, Any] = {
            "active_signals": [],
            "signal_combination": "majority_vote",
            "indicator_params": {k: {} for k in self.search_space["indicators"].keys()},
            "signal_params": {k: {} for k in self.search_space["signals"].keys()},
        }

        for value, spec in zip(genome, self.gene_specs):
            self._assign_gene(candidate, value, spec)

        self._normalize_candidate(candidate)
        return candidate

    def _assign_gene(self, candidate: Dict[str, Any], value: float, spec: GeneSpec) -> None:
        key = spec.key

        if key[0] == "signal_enabled":
            if int(round(value)) == 1 and key[1] not in candidate["active_signals"]:
                candidate["active_signals"].append(key[1])
            return

        if key[0] == "signal_combination":
            idx = int(min(spec.high, max(spec.low, round(value))))
            candidate["signal_combination"] = spec.choices[idx]
            return

        if key[0] == "indicator_params":
            bucket = candidate["indicator_params"][key[1]]
            bucket[key[2]] = self._coerce_value(value, spec)
            return

        if key[0] == "signal_params":
            bucket = candidate["signal_params"][key[1]]
            bucket[key[2]] = self._coerce_value(value, spec)
            return

        raise ValueError(f"Unknown gene target: {key}")

    @staticmethod
    def _coerce_value(value: float, spec: GeneSpec) -> Any:
        if spec.kind == "int":
            return int(round(value))
        if spec.kind == "float":
            return float(value)
        if spec.kind == "categorical_index":
            idx = int(round(value))
            idx = max(0, min(idx, len(spec.choices) - 1))
            return spec.choices[idx]
        if spec.kind == "binary":
            return int(round(value))
        return value

    def _normalize_candidate(self, candidate: Dict[str, Any]) -> None:
        if not candidate["active_signals"]:
            candidate["active_signals"] = [self.signal_ids[0]]

        if len(candidate["active_signals"]) > self.max_active_signals:
            candidate["active_signals"] = candidate["active_signals"][: self.max_active_signals]

        ema_fast = candidate["indicator_params"]["ema_fast"].get("period", 10)
        ema_slow = candidate["indicator_params"]["ema_slow"].get("period", 30)
        if ema_fast >= ema_slow:
            candidate["indicator_params"]["ema_slow"]["period"] = ema_fast + 1

        sma_fast = candidate["indicator_params"]["sma_fast"].get("period", 10)
        sma_slow = candidate["indicator_params"]["sma_slow"].get("period", 30)
        if sma_fast >= sma_slow:
            candidate["indicator_params"]["sma_slow"]["period"] = sma_fast + 1

        if "macd_main" in candidate["indicator_params"]:
            macd_fast = candidate["indicator_params"]["macd_main"].get("fast_period", 12)
            macd_slow = candidate["indicator_params"]["macd_main"].get("slow_period", 26)
            if macd_fast >= macd_slow:
                candidate["indicator_params"]["macd_main"]["slow_period"] = macd_fast + 1

        for signal_id in ["rsi_osob", "rsi_trend_combo"]:
            if signal_id in candidate["signal_params"]:
                overbought = candidate["signal_params"][signal_id].get("overbought", 70)
                oversold = candidate["signal_params"][signal_id].get("oversold", 30)
                if overbought <= oversold:
                    candidate["signal_params"][signal_id]["overbought"] = oversold + 5

    def required_indicators(self, candidate: Dict[str, Any]) -> List[str]:
        required = set()
        signals = self.search_space["signals"]

        for signal_id in candidate["active_signals"]:
            signal_def = signals[signal_id]
            for input_name, input_ref in signal_def.get("inputs", {}).items():
                if input_ref == "price":
                    continue
                indicator_id = self._resolve_indicator_id(input_ref)
                required.add(indicator_id)

        return sorted(required)

    def _resolve_indicator_id(self, input_ref: str) -> str:
        if input_ref in self.search_space["indicators"]:
            return input_ref

        for indicator_id in self.search_space["indicators"].keys():
            if input_ref.startswith(indicator_id + "_"):
                return indicator_id

        if input_ref in self._component_to_indicators:
            owners = self._component_to_indicators[input_ref]
            if owners:
                return owners[0]

        raise ValueError(f"Cannot resolve indicator dependency for '{input_ref}'")

    def build_strategy(self, candidate: Dict[str, Any], symbols: List[str], timeframes: List[str], strategy_name: str):
        builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)

        required_indicator_ids = self.required_indicators(candidate)

        for indicator_id in required_indicator_ids:
            indicator_def = self.search_space["indicators"][indicator_id]
            indicator_func = getattr(indicators_library, indicator_def["function"])
            indicator_params = candidate["indicator_params"].get(indicator_id, {})
            builder.add_indicator(indicator_id, indicator_func, **indicator_params)

        for signal_id in candidate["active_signals"]:
            signal_def = self.search_space["signals"][signal_id]
            signal_func = getattr(signals_library, signal_def["function"])

            inputs = deepcopy(signal_def.get("inputs", {}))
            params = deepcopy(candidate["signal_params"].get(signal_id, {}))
            call_kwargs = {}
            call_kwargs.update(inputs)
            call_kwargs.update(params)
            builder.add_signal_rule(signal_id, signal_func, **call_kwargs)

        builder.set_signal_combination(candidate.get("signal_combination", "majority_vote"))
        builder.set_strategy_info(strategy_name, "1.0.0")
        strategy = builder.build()

        builder_ref = strategy.builder

        def _vectorized(data_map):
            result = {}
            for symbol, by_tf in data_map.items():
                result[symbol] = {}
                for timeframe, frame in by_tf.items():
                    local_df = frame.copy()
                    builder_ref._calculate_indicators(local_df)
                    signal_series = builder_ref._execute_signal_rules(local_df)
                    if not isinstance(signal_series, pd.Series):
                        signal_series = pd.Series(["HOLD"] * len(local_df), index=local_df.index)
                    result[symbol][timeframe] = signal_series
            return result

        strategy.generate_signals_vectorized = _vectorized

        # Force backtester to use vectorized signal path to avoid hardcoded builder pre-calc route.
        if hasattr(strategy, "builder"):
            delattr(strategy, "builder")

        return strategy
