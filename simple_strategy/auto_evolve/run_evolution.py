import argparse
import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Tuple

from .candidate_builder import CandidateBuilder
from .checkpoint import load_json, load_latest_checkpoint, save_generation_checkpoint, save_json
from .evaluator import CandidateEvaluator, build_synthetic_data
from .gates import gate_train_metrics
from .reports import candidate_fingerprint, export_strategy_files, top_unique_results, write_reports
from .search_space import load_run_config, load_search_space
from simple_strategy.shared.data_feeder import DataFeeder

logger = logging.getLogger(__name__)


def _configure_run_logging() -> Path:
    log_path = Path("simple_strategy") / "auto_evolve" / "latest_run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
    return log_path


def _import_deap_or_raise():
    try:
        from deap import base, creator, tools  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "DEAP is not installed. Install it with: pip install deap"
        ) from exc
    return base, creator, tools


class EvolutionRunner:
    def __init__(
        self,
        run_config: Dict[str, Any],
        search_space: Dict[str, Any],
        data_dir: str,
        run_dir: Path,
        smoke_mode: bool = False,
        max_runtime_hours: float = 0.0,
    ):
        self.run_config = deepcopy(run_config)
        self.search_space = deepcopy(search_space)
        self.data_dir = data_dir
        self.run_dir = run_dir
        self.smoke_mode = smoke_mode
        self.max_runtime_seconds = float(max_runtime_hours) * 3600.0 if float(max_runtime_hours) > 0 else None
        self.base_symbols = list(self.run_config.get("symbols", []))

        if smoke_mode:
            self.run_config["train_start"] = "2024-01-01"
            self.run_config["train_end"] = "2024-01-02"
            self.run_config["validation_start"] = "2024-01-02"
            self.run_config["validation_end"] = "2024-01-03"
            self.run_config["final_start"] = "2024-01-03"
            self.run_config["final_end"] = "2024-01-04"
            self.run_config["min_trades"] = 1
            self.run_config["validation_min_trades"] = 1
            self.run_config["min_train_return_pct"] = -100.0
            self.run_config["max_train_drawdown_pct"] = 1000.0
            self.run_config["min_validation_return_pct"] = -100.0
            self.run_config["max_validation_drawdown_pct"] = 1000.0
            self.run_config["max_validation_gap_pct"] = 1000.0
            self.run_config["max_stress_drop_pct"] = 1000.0
            self.run_config["min_stress_return_pct"] = -100.0
            self.run_config["max_stress_drawdown_pct"] = 1000.0
            self.run_config["enable_inner_optimization"] = False
            self.run_config["shortlist_enable_inner_optimization"] = False
            self.run_config["fast_search_symbol_count"] = 1
            self.run_config["shortlist_symbol_count"] = 1

        self._apply_search_space_filters()
        self.builder = CandidateBuilder(
            search_space=self.search_space,
            max_active_signals=int(self.run_config.get("max_active_signals", 3)),
            max_active_filters=int(self.run_config.get("max_active_filters", 2)),
        )
        self.search_run_config = self._build_stage_run_config(
            symbol_count=int(self.run_config.get("fast_search_symbol_count", 2)),
            enable_inner_optimization=bool(self.run_config.get("enable_inner_optimization", False)),
            inner_opt_trials=int(self.run_config.get("inner_opt_trials", 0)),
            stage_prefix="search",
        )
        self.final_run_config = self._build_stage_run_config(
            symbol_count=int(self.run_config.get("final_symbol_count", 0)),
            enable_inner_optimization=False,
            inner_opt_trials=0,
            stage_prefix="final",
        )
        self.search_evaluator = CandidateEvaluator(self.builder, self.search_run_config, data_dir=self.data_dir)
        self.search_stage_data = None
        self.final_stage_data = None

        self.synthetic_data = None
        if smoke_mode:
            self.synthetic_data = build_synthetic_data(
                symbol=self.search_run_config["symbols"][0],
                timeframe=self.search_run_config["timeframes"][0],
                candles=1200,
                start="2024-01-01",
            )
        else:
            self.search_stage_data = self._load_stage_data(self.search_run_config)
            self.final_stage_data = self._load_stage_data(self.final_run_config)

        self.random = random.Random(int(self.run_config.get("seed", 42)))
        self.progress_lock = Lock()
        self.total_expected_evals = 0
        self.total_completed_evals = 0
        self.current_generation = -1
        self.current_generation_completed = 0
        self.current_generation_total = 0
        self.current_best_score = None
        self.current_candidate = ""
        self.last_progress_message = "Starting run"
        self.run_started_ts = time.time()

        self.base, self.creator, self.tools = _import_deap_or_raise()
        self.toolbox = self._build_toolbox()

    def _campaign_registry_path(self) -> Path:
        return self.run_dir.parent / "campaign_registry.json"

    def _update_campaign_registry(self, top_results: List[Dict[str, Any]]) -> None:
        registry_path = self._campaign_registry_path()
        registry_payload = load_json(registry_path)
        registry = registry_payload.get("runs", []) if isinstance(registry_payload, dict) else []
        if not isinstance(registry, list):
            registry = []

        finalists = [row for row in top_results if bool(row.get("final_passed", False))]
        registry_entry = {
            "run_name": self.run_dir.name,
            "run_dir": str(self.run_dir),
            "recorded_at": datetime.utcnow().isoformat() + "Z",
            "profile_name": str(self.run_config.get("profile_name", "") or ""),
            "config_path": str(self.run_config.get("config_path", "") or ""),
            "seed": int(self.run_config.get("seed", 0) or 0),
            "population_size": int(self.run_config.get("population_size", 0) or 0),
            "generations": int(self.run_config.get("generations", 0) or 0),
            "top_result_count": len(top_results),
            "finalist_count": len(finalists),
            "top_candidate_fingerprints": [
                candidate_fingerprint(row["candidate"], search_space=self.search_space)
                for row in top_results
            ],
            "finalist_fingerprints": [
                candidate_fingerprint(row["candidate"], search_space=self.search_space)
                for row in finalists
            ],
        }

        registry = [row for row in registry if str(row.get("run_name", "")) != self.run_dir.name]
        registry.append(registry_entry)
        save_json(registry_path, {"runs": registry})

    def _write_progress(self, status: str, message: str = "") -> None:
        payload = {
            "status": status,
            "message": message or self.last_progress_message,
            "run_started_at": self.run_started_ts,
            "updated_at": time.time(),
            "generations": int(self.run_config.get("generations", 0)),
            "population_size": int(self.run_config.get("population_size", 0)),
            "generation": self.current_generation,
            "generation_completed": self.current_generation_completed,
            "generation_total": self.current_generation_total,
            "evaluated": self.total_completed_evals,
            "expected_evaluations": self.total_expected_evals,
            "best_score": self.current_best_score,
            "current_candidate": self.current_candidate,
        }
        save_json(self.run_dir / "progress.json", payload)

    def _load_stage_data(self, stage_run_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        symbols = list(stage_run_config.get("symbols", []))
        timeframes = list(stage_run_config.get("timeframes", []))
        if not symbols or not timeframes:
            return {}

        start_candidates = [
            stage_run_config.get("train_start"),
            stage_run_config.get("validation_start"),
            stage_run_config.get("final_start"),
        ]
        end_candidates = [
            stage_run_config.get("train_end"),
            stage_run_config.get("validation_end"),
            stage_run_config.get("final_end"),
        ]
        start_values = [value for value in start_candidates if value]
        end_values = [value for value in end_candidates if value]
        if not start_values or not end_values:
            return {}

        feeder = DataFeeder(data_dir=self.data_dir)
        logger.info(
            "Preloading Auto Evolve stage data | symbols=%s | timeframes=%s | range=%s -> %s",
            len(symbols),
            ",".join(timeframes),
            min(start_values),
            max(end_values),
        )
        return feeder.get_data_for_symbols(
            symbols=symbols,
            timeframes=timeframes,
            start_date=min(start_values),
            end_date=max(end_values),
        )

    def _select_symbols(self, symbol_count: int) -> List[str]:
        symbols = list(self.base_symbols)
        if not symbols:
            return []
        if symbol_count <= 0:
            return symbols
        return symbols[: min(len(symbols), symbol_count)]

    def _build_stage_run_config(
        self,
        *,
        symbol_count: int,
        enable_inner_optimization: bool,
        inner_opt_trials: int,
        stage_prefix: str = "",
    ) -> Dict[str, Any]:
        stage_config = deepcopy(self.run_config)
        stage_config["symbols"] = self._select_symbols(symbol_count)
        stage_config["enable_inner_optimization"] = bool(enable_inner_optimization)
        stage_config["inner_opt_trials"] = int(inner_opt_trials)
        self._apply_stage_overrides(stage_config, stage_prefix)
        return stage_config

    def _apply_search_space_filters(self) -> None:
        allowed_signals = self.run_config.get("allowed_signals", [])
        if isinstance(allowed_signals, list) and allowed_signals:
            allowed_signal_ids = {str(signal_id) for signal_id in allowed_signals}
            self.search_space["signals"] = {
                signal_id: deepcopy(signal_def)
                for signal_id, signal_def in self.search_space.get("signals", {}).items()
                if signal_id in allowed_signal_ids
            }

        allowed_filters = self.run_config.get("allowed_filters", [])
        if isinstance(allowed_filters, list) and allowed_filters:
            allowed_filter_ids = {str(filter_id) for filter_id in allowed_filters}
            self.search_space["filters"] = {
                filter_id: deepcopy(filter_def)
                for filter_id, filter_def in self.search_space.get("filters", {}).items()
                if filter_id in allowed_filter_ids
            }

    def _apply_stage_overrides(self, stage_config: Dict[str, Any], stage_prefix: str) -> None:
        prefix = str(stage_prefix or "").strip()
        if not prefix:
            return

        override_fields = [
            "train_start",
            "train_end",
            "validation_start",
            "validation_end",
            "final_start",
            "final_end",
            "min_trades",
            "validation_min_trades",
            "min_train_return_pct",
            "max_train_drawdown_pct",
            "min_validation_return_pct",
            "max_validation_drawdown_pct",
            "max_validation_gap_pct",
            "enable_stress_gate",
            "max_stress_drop_pct",
            "min_stress_return_pct",
            "max_stress_drawdown_pct",
            "candidate_timeout_seconds",
            "precheck_enabled",
            "precheck_days",
            "precheck_symbol_count",
            "precheck_min_signal_events",
            "precheck_signal_score_weight",
        ]

        for field in override_fields:
            override_key = f"{prefix}_{field}"
            if override_key in self.run_config:
                stage_config[field] = deepcopy(self.run_config[override_key])

    def _build_toolbox(self):
        if not hasattr(self.creator, "FitnessMax"):
            self.creator.create("FitnessMax", self.base.Fitness, weights=(1.0,))
        if not hasattr(self.creator, "Individual"):
            self.creator.create("Individual", list, fitness=self.creator.FitnessMax)

        toolbox = self.base.Toolbox()
        toolbox.register("individual", self._new_individual)
        toolbox.register("population", self.tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", self.tools.cxTwoPoint)
        toolbox.register("select", self.tools.selTournament, tournsize=int(self.run_config.get("tournament_size", 3)))
        toolbox.register("clone", self._clone_individual)
        return toolbox

    def _new_individual(self):
        genome = self.builder.random_genome(self.random)
        return self.creator.Individual(genome)

    def _clone_individual(self, individual):
        return self.creator.Individual(list(individual))

    def _individual_fingerprint(self, individual) -> str:
        candidate = self.builder.decode(list(individual))
        return candidate_fingerprint(candidate, search_space=self.search_space)

    def _candidate_family_key(self, candidate: Dict[str, Any]) -> str:
        active_signals = sorted(str(signal_id) for signal_id in candidate.get("active_signals", []))
        if not active_signals:
            return "__none__"
        return "|".join(active_signals)

    def _individual_family_key(self, individual) -> str:
        candidate = self.builder.decode(list(individual))
        return self._candidate_family_key(candidate)

    def _mutate_individual(self, individual):
        mutation_rate = float(self.run_config.get("gene_mutation_rate", 0.15))
        for idx, spec in enumerate(self.builder.gene_specs):
            if self.random.random() < mutation_rate:
                individual[idx] = self.builder.mutate_gene(individual[idx], spec, self.random)
        return individual

    def _select_elites(self, population, elite_count: int):
        ranked = sorted(
            population,
            key=lambda ind: float(ind.fitness.values[0]) if hasattr(ind.fitness, "values") and ind.fitness.values else -1e9,
            reverse=True,
        )
        elites = []
        seen_fingerprints = set()
        seen_families = set()
        family_diversity_mode = str(self.run_config.get("elite_diversity_mode", "") or "").strip()

        if family_diversity_mode == "active_signals":
            for individual in ranked:
                fingerprint = self._individual_fingerprint(individual)
                family_key = self._individual_family_key(individual)
                if fingerprint in seen_fingerprints or family_key in seen_families:
                    continue
                seen_fingerprints.add(fingerprint)
                seen_families.add(family_key)
                elites.append(self.toolbox.clone(individual))
                if len(elites) >= elite_count:
                    break

        for individual in ranked:
            fingerprint = self._individual_fingerprint(individual)
            if fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)
            elites.append(self.toolbox.clone(individual))
            if len(elites) >= elite_count:
                break

        if len(elites) < elite_count:
            for individual in ranked:
                clone = self.toolbox.clone(individual)
                elites.append(clone)
                if len(elites) >= elite_count:
                    break
        return elites[:elite_count]

    def _select_diverse_results(
        self,
        results: List[Dict[str, Any]],
        limit: int,
        diversity_mode: str = "",
    ) -> List[Dict[str, Any]]:
        ranked = sorted(results, key=lambda r: float(r.get("score", -1e9)), reverse=True)
        if not ranked or limit <= 0:
            return []

        unique_rows: List[Dict[str, Any]] = []
        seen_fingerprints = set()
        for row in ranked:
            fingerprint = candidate_fingerprint(row["candidate"], search_space=self.search_space)
            if fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)
            unique_rows.append(row)

        diversity_mode = str(diversity_mode or "").strip()
        if diversity_mode != "active_signals":
            return unique_rows[:limit]

        selected: List[Dict[str, Any]] = []
        seen_families = set()
        for row in unique_rows:
            family_key = self._candidate_family_key(row["candidate"])
            if family_key in seen_families:
                continue
            seen_families.add(family_key)
            selected.append(row)
            if len(selected) >= limit:
                return selected

        for row in unique_rows:
            fingerprint = candidate_fingerprint(row["candidate"], search_space=self.search_space)
            if any(
                candidate_fingerprint(existing["candidate"], search_space=self.search_space) == fingerprint
                for existing in selected
            ):
                continue
            selected.append(row)
            if len(selected) >= limit:
                break
        return selected

    def _inject_diversity(self, population, protected_count: int) -> None:
        if not population:
            return

        seen = set()
        for idx, individual in enumerate(population):
            fingerprint = self._individual_fingerprint(individual)
            if fingerprint in seen and idx >= protected_count:
                replacement = self.toolbox.individual()
                population[idx] = replacement
                fingerprint = self._individual_fingerprint(replacement)
            seen.add(fingerprint)

        immigrant_count = max(0, int(self.run_config.get("diversity_injection_count", 0)))
        if immigrant_count <= 0:
            return

        start_idx = max(protected_count, len(population) - immigrant_count)
        for idx in range(start_idx, len(population)):
            population[idx] = self.toolbox.individual()
            if hasattr(population[idx].fitness, "values"):
                del population[idx].fitness.values

    def _evaluate_one(self, generation: int, idx: int, individual) -> Dict[str, Any]:
        candidate = self.builder.decode(list(individual))
        evaluator = self.search_evaluator
        if int(self.search_run_config.get("workers", 1)) > 1:
            evaluator = CandidateEvaluator(self.builder, self.search_run_config, data_dir=self.data_dir)

        logger.info("Generation %s candidate %s started", generation + 1, idx + 1)
        result = evaluator.evaluate_candidate(
            candidate=candidate,
            generation=generation,
            candidate_idx=idx,
            data_override=self.synthetic_data or self.search_stage_data,
            use_inner_optimization=bool(self.search_run_config.get("enable_inner_optimization", False)),
        )
        result["generation"] = generation
        result["index"] = idx
        logger.info(
            "Generation %s candidate %s finished | passed=%s | score=%.4f | failed_gate=%s",
            generation + 1,
            idx + 1,
            bool(result.get("passed", False)),
            float(result.get("score", -1e9)),
            result.get("failed_gate", ""),
        )
        return result

    def _record_candidate_completion(self, generation: int, idx: int, result: Dict[str, Any]) -> None:
        with self.progress_lock:
            self.total_completed_evals += 1
            if generation == self.current_generation:
                self.current_generation_completed += 1
            score = float(result.get("score", -1e9))
            if self.current_best_score is None or score > float(self.current_best_score):
                self.current_best_score = score
            self.current_candidate = f"G{generation + 1} C{idx + 1}"
            self.last_progress_message = (
                f"Generation {generation + 1}: {self.current_generation_completed}/{self.current_generation_total} "
                f"candidates done"
            )
            self._write_progress("running", self.last_progress_message)

    def _refine_top_results(self, all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        passed_results = [row for row in all_results if bool(row.get("passed", False))]
        if not passed_results:
            return []

        shortlist_top_k = int(self.run_config.get("shortlist_top_k", max(10, int(self.run_config.get("top_k", 10)))))
        shortlist = top_unique_results(passed_results, top_k=shortlist_top_k, search_space=self.search_space)
        if not shortlist:
            return []

        refine_run_config = self._build_stage_run_config(
            symbol_count=int(self.run_config.get("shortlist_symbol_count", 6)),
            enable_inner_optimization=False,
            inner_opt_trials=0,
            stage_prefix="refine",
        )
        refine_evaluator = CandidateEvaluator(self.builder, refine_run_config, data_dir=self.data_dir)
        refine_stage_data = self.synthetic_data or self._load_stage_data(refine_run_config)

        heavy_opt_enabled = bool(self.run_config.get("shortlist_enable_inner_optimization", True))
        heavy_opt_top_k = max(0, int(self.run_config.get("shortlist_inner_opt_top_k", 4)))
        heavy_opt_trials = int(self.run_config.get("shortlist_inner_opt_trials", self.run_config.get("inner_opt_trials", 0)))

        refined_results: List[Dict[str, Any]] = []
        for idx, row in enumerate(shortlist):
            candidate = deepcopy(row["candidate"])
            use_heavy_opt = heavy_opt_enabled and idx < heavy_opt_top_k and heavy_opt_trials > 0
            if use_heavy_opt:
                opt_run_config = deepcopy(refine_run_config)
                opt_run_config["enable_inner_optimization"] = True
                opt_run_config["inner_opt_trials"] = heavy_opt_trials
                evaluator = CandidateEvaluator(self.builder, opt_run_config, data_dir=self.data_dir)
            else:
                evaluator = refine_evaluator

            refined = evaluator.evaluate_candidate(
                candidate=candidate,
                generation=int(row.get("generation", -1)),
                candidate_idx=idx,
                data_override=refine_stage_data,
                use_inner_optimization=use_heavy_opt,
            )
            refined["generation"] = row.get("generation", -1)
            refined["index"] = row.get("index", idx)
            refined_results.append(refined)

        return sorted(refined_results, key=lambda r: float(r.get("score", -1e9)), reverse=True)

    def _evaluate_population(self, generation: int, population) -> List[Dict[str, Any]]:
        workers = int(self.run_config.get("workers", 1))
        results: List[Dict[str, Any]] = [None] * len(population)

        if workers <= 1:
            for idx, individual in enumerate(population):
                results[idx] = self._evaluate_one(generation, idx, individual)
                self._record_candidate_completion(generation, idx, results[idx])
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self._evaluate_one, generation, idx, ind): idx
                    for idx, ind in enumerate(population)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    results[idx] = future.result()
                    self._record_candidate_completion(generation, idx, results[idx])

        return results

    def _assign_fitness(self, population, eval_results: List[Dict[str, Any]]) -> None:
        min_parent_score = float(self.run_config.get("parent_min_score", -1e9) or -1e9)
        for individual, row in zip(population, eval_results):
            score = float(row.get("score", -1e9))
            if not bool(row.get("passed", False)) or score < min_parent_score:
                score = -1e9
            individual.fitness.values = (score,)

    def run(self, resume_payload: Dict[str, Any] = None) -> Dict[str, Any]:
        population_size = int(self.run_config.get("population_size", 64))
        generations = int(self.run_config.get("generations", 40))
        elite_count = max(1, int(self.run_config.get("elite_count", 5)))
        crossover_p = float(self.run_config.get("crossover_probability", 0.7))
        mutation_p = float(self.run_config.get("mutation_probability", 0.3))

        all_results: List[Dict[str, Any]] = []
        start_generation = 0

        if resume_payload:
            start_generation = int(resume_payload.get("generation", -1)) + 1
            raw_population = resume_payload.get("population", [])
            population = [self.creator.Individual(list(genome)) for genome in raw_population]
            all_results = list(resume_payload.get("all_results", []))
        else:
            population = self.toolbox.population(n=population_size)

        self.total_expected_evals = generations * population_size if generations > 0 and population_size > 0 else 0
        self.total_completed_evals = len(all_results)
        self.run_started_ts = time.time()
        self._write_progress("starting", "Preparing first generation")
        for generation in range(start_generation, generations):
            if self.max_runtime_seconds is not None:
                elapsed_seconds = max(0.0, time.time() - self.run_started_ts)
                if elapsed_seconds >= self.max_runtime_seconds:
                    logger.warning(
                        "Max runtime reached before generation %s/%s", generation + 1, generations
                    )
                    print(
                        "Max runtime reached ({:.2f}h >= {:.2f}h). Stopping before generation {}/{}.".format(
                            elapsed_seconds / 3600.0,
                            self.max_runtime_seconds / 3600.0,
                            generation + 1,
                            generations,
                        )
                    )
                    break

            self.current_generation = generation
            self.current_generation_completed = 0
            self.current_generation_total = len(population)
            self.current_candidate = ""
            self.last_progress_message = f"Generation {generation + 1}/{generations} started"
            self._write_progress("running", self.last_progress_message)
            logger.info(
                "Generation %s/%s started | population=%s | workers=%s",
                generation + 1,
                generations,
                len(population),
                int(self.run_config.get("workers", 1)),
            )
            eval_results = self._evaluate_population(generation, population)
            self._assign_fitness(population, eval_results)
            all_results.extend(eval_results)

            best_score = max(row.get("score", -1e9) for row in eval_results)
            self.current_best_score = float(best_score)
            self.last_progress_message = f"Generation {generation + 1}/{generations} complete"
            self._write_progress("running", self.last_progress_message)
            logger.info(
                "Generation %s/%s complete | best_score=%.4f | evaluated=%s/%s",
                generation + 1,
                generations,
                float(best_score),
                self.total_completed_evals,
                self.total_expected_evals,
            )
            print(f"Generation {generation + 1}/{generations} | best_score={best_score:.4f}")

            checkpoint_interval = int(self.run_config.get("checkpoint_every_generations", 2))
            if (generation + 1) % checkpoint_interval == 0 or generation == generations - 1:
                payload = {
                    "generation": generation,
                    "population": [list(ind) for ind in population],
                    "all_results": all_results,
                    "config": self.run_config,
                    "search_space": self.search_space,
                }
                save_generation_checkpoint(self.run_dir, generation, payload)

            if generation == generations - 1:
                break

            elites = self._select_elites(population, elite_count)
            selected = self.toolbox.select(population, len(population) - elite_count)
            offspring = [self.toolbox.clone(ind) for ind in selected]

            for i in range(0, len(offspring) - 1, 2):
                if self.random.random() < crossover_p:
                    self.toolbox.mate(offspring[i], offspring[i + 1])
                    if hasattr(offspring[i].fitness, "values"):
                        del offspring[i].fitness.values
                    if hasattr(offspring[i + 1].fitness, "values"):
                        del offspring[i + 1].fitness.values

            for i in range(len(offspring)):
                if self.random.random() < mutation_p:
                    offspring[i] = self._mutate_individual(offspring[i])
                    if hasattr(offspring[i].fitness, "values"):
                        del offspring[i].fitness.values

            population = elites + offspring
            self._inject_diversity(population, protected_count=len(elites))

        top_k = int(self.run_config.get("top_k", 10))
        refined_results = []
        if bool(self.run_config.get("enable_refine_stage", True)):
            refined_results = self._refine_top_results(all_results)
        finalist_min_search_score = float(
            self.run_config.get("finalist_min_search_score", self.run_config.get("parent_min_score", -1e9)) or -1e9
        )
        candidate_pool = refined_results or [
            row
            for row in all_results
            if bool(row.get("passed", False)) and float(row.get("score", -1e9)) >= finalist_min_search_score
        ]
        finalist_diversity_mode = str(self.run_config.get("finalist_diversity_mode", "") or "").strip()
        if finalist_diversity_mode:
            top_results = self._select_diverse_results(candidate_pool, limit=top_k, diversity_mode=finalist_diversity_mode)
        else:
            top_results = top_unique_results(
                candidate_pool,
                top_k=top_k,
                search_space=self.search_space,
            )

        final_evaluator = CandidateEvaluator(self.builder, self.final_run_config, data_dir=self.data_dir)
        final_timeout_seconds = float(self.final_run_config.get("candidate_timeout_seconds", 0) or 0.0)
        final_min_trades = int(self.run_config.get("final_min_trades", max(1, int(self.run_config.get("validation_min_trades", 1)))))
        final_min_return_pct = float(self.run_config.get("final_min_return_pct", self.run_config.get("min_validation_return_pct", -5.0)))
        final_max_drawdown_pct = float(self.run_config.get("final_max_drawdown_pct", self.run_config.get("max_validation_drawdown_pct", 45.0)))

        for row in top_results:
            try:
                final_deadline_ts = time.time() + final_timeout_seconds if final_timeout_seconds > 0 else None
                row["final_metrics"] = final_evaluator.run_final_segment(
                    row["candidate"],
                    data_override=self.synthetic_data or self.final_stage_data,
                    deadline_ts=final_deadline_ts,
                )
                row["final_score"] = float(row["score"]) + 0.1 * float(row["final_metrics"].get("total_return", 0.0))
                final_gate = gate_train_metrics(
                    row["final_metrics"],
                    min_trades=final_min_trades,
                    min_return=final_min_return_pct,
                    max_drawdown=final_max_drawdown_pct,
                )
                row["final_passed"] = bool(final_gate.passed)
                row["final_failed_gate"] = "" if final_gate.passed else "final_gate"
                row["final_failure_reason"] = "" if final_gate.passed else final_gate.reason
            except Exception as exc:
                row["final_metrics"] = {"total_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0, "total_trades": 0}
                row["final_score"] = float(row["score"])
                row["final_passed"] = False
                row["final_failed_gate"] = "final_error"
                row["final_failure_reason"] = str(exc)
                row["final_error"] = str(exc)

        top_results = sorted(
            top_results,
            key=lambda r: (
                bool(r.get("final_passed", False)),
                float(r.get("final_score", r.get("score", -1e9))),
            ),
            reverse=True,
        )

        write_reports(self.run_dir, top_results, self.run_config)
        if bool(self.run_config.get("save_strategy_files", True)):
            export_strategy_files(
                run_dir=self.run_dir,
                candidate_builder=self.builder,
                top_results=[row for row in top_results if bool(row.get("final_passed", row.get("passed", False)))],
                symbols=self.run_config["symbols"],
                timeframes=self.run_config["timeframes"],
                publish_top_n=int(self.run_config.get("publish_top_n", 0)),
            )

        self._update_campaign_registry(top_results)

        self.current_generation = generations - 1 if generations > 0 else -1
        self.current_generation_completed = self.current_generation_total
        self.last_progress_message = "Run complete"
        self._write_progress("complete", self.last_progress_message)
        logger.info("Run complete | evaluated=%s | output=%s", len(all_results), self.run_dir)

        return {
            "run_dir": str(self.run_dir),
            "top_results": top_results,
            "total_evaluated": len(all_results),
        }


def _resolve_run_dir(base_output_dir: str, resume_dir: str = "") -> Path:
    if resume_dir:
        return Path(resume_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(base_output_dir) / f"run_{ts}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DEAP outer-loop strategy evolution.")
    parser.add_argument("--config", default="simple_strategy/auto_evolve/configs/default.json", help="Run config JSON path")
    parser.add_argument("--search-space", default="", help="Optional search space override JSON path")
    parser.add_argument("--data-dir", default="data", help="Data directory used by DataFeeder")
    parser.add_argument("--resume-run-dir", default="", help="Existing run dir to resume from")
    parser.add_argument("--smoke", action="store_true", help="Run a synthetic-data smoke evolution")
    parser.add_argument("--population", type=int, default=0, help="Override population size")
    parser.add_argument("--generations", type=int, default=0, help="Override generation count")
    parser.add_argument("--workers", type=int, default=0, help="Override worker count")
    parser.add_argument("--seed", type=int, default=0, help="Override random seed")
    parser.add_argument("--max-runtime-hours", type=float, default=0.0, help="Stop run after this many hours (best effort)")
    return parser.parse_args()


def main() -> int:
    logging.getLogger("simple_strategy.backtester.performance_tracker").setLevel(logging.WARNING)
    logging.getLogger("simple_strategy.shared.strategy_base").setLevel(logging.WARNING)

    args = parse_args()

    run_config = load_run_config(args.config)
    search_space = load_search_space(args.search_space)
    run_config["config_path"] = args.config
    run_config["profile_name"] = Path(args.config).stem
    run_dir = _resolve_run_dir(run_config.get("output_dir", "simple_strategy/auto_evolve/output"), args.resume_run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = _configure_run_logging()
    logger.info("Auto Evolve run starting")
    logger.info("Run log: %s", log_path)

    resume_payload = None
    if args.resume_run_dir:
        resume_payload = load_latest_checkpoint(run_dir)
        if resume_payload:
            resume_config = resume_payload.get("config", {})
            if isinstance(resume_config, dict) and resume_config:
                run_config = deepcopy(resume_config)
            resume_space = resume_payload.get("search_space", {})
            if isinstance(resume_space, dict) and resume_space:
                search_space = deepcopy(resume_space)
        else:
            print(f"No checkpoint found in {run_dir}. Starting a fresh run.")

    if args.population > 0:
        run_config["population_size"] = args.population
    if args.generations > 0:
        run_config["generations"] = args.generations
    if args.workers > 0:
        run_config["workers"] = args.workers
    if args.seed > 0:
        run_config["seed"] = args.seed

    effective_max_runtime_hours = float(args.max_runtime_hours or run_config.get("max_runtime_hours", 0.0) or 0.0)
    if effective_max_runtime_hours > 0:
        run_config["max_runtime_hours"] = effective_max_runtime_hours

    save_json(run_dir / "run_config.resolved.json", run_config)
    save_json(run_dir / "search_space.resolved.json", search_space)

    runner = EvolutionRunner(
        run_config=run_config,
        search_space=search_space,
        data_dir=args.data_dir,
        run_dir=run_dir,
        smoke_mode=args.smoke,
        max_runtime_hours=effective_max_runtime_hours,
    )
    result = runner.run(resume_payload=resume_payload)

    logger.info("Auto Evolve finished successfully")
    print(f"Run complete. Evaluated: {result['total_evaluated']}")
    print(f"Output: {result['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
