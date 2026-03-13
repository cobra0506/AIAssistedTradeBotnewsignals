import argparse
import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .candidate_builder import CandidateBuilder
from .checkpoint import load_latest_checkpoint, save_generation_checkpoint, save_json
from .evaluator import CandidateEvaluator, build_synthetic_data
from .reports import export_strategy_files, top_unique_results, write_reports
from .search_space import load_run_config, load_search_space


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

        self.builder = CandidateBuilder(
            search_space=self.search_space,
            max_active_signals=int(self.run_config.get("max_active_signals", 3)),
            max_active_filters=int(self.run_config.get("max_active_filters", 2)),
        )
        self.search_run_config = self._build_stage_run_config(
            symbol_count=int(self.run_config.get("fast_search_symbol_count", 2)),
            enable_inner_optimization=bool(self.run_config.get("enable_inner_optimization", False)),
            inner_opt_trials=int(self.run_config.get("inner_opt_trials", 0)),
        )
        self.search_evaluator = CandidateEvaluator(self.builder, self.search_run_config, data_dir=self.data_dir)

        self.synthetic_data = None
        if smoke_mode:
            self.synthetic_data = build_synthetic_data(
                symbol=self.search_run_config["symbols"][0],
                timeframe=self.search_run_config["timeframes"][0],
                candles=1200,
                start="2024-01-01",
            )

        self.random = random.Random(int(self.run_config.get("seed", 42)))

        self.base, self.creator, self.tools = _import_deap_or_raise()
        self.toolbox = self._build_toolbox()

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
    ) -> Dict[str, Any]:
        stage_config = deepcopy(self.run_config)
        stage_config["symbols"] = self._select_symbols(symbol_count)
        stage_config["enable_inner_optimization"] = bool(enable_inner_optimization)
        stage_config["inner_opt_trials"] = int(inner_opt_trials)
        return stage_config

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

    def _mutate_individual(self, individual):
        mutation_rate = float(self.run_config.get("gene_mutation_rate", 0.15))
        for idx, spec in enumerate(self.builder.gene_specs):
            if self.random.random() < mutation_rate:
                individual[idx] = self.builder.mutate_gene(individual[idx], spec, self.random)
        return individual

    def _evaluate_one(self, generation: int, idx: int, individual) -> Dict[str, Any]:
        candidate = self.builder.decode(list(individual))
        evaluator = self.search_evaluator
        if int(self.search_run_config.get("workers", 1)) > 1:
            evaluator = CandidateEvaluator(self.builder, self.search_run_config, data_dir=self.data_dir)

        result = evaluator.evaluate_candidate(
            candidate=candidate,
            generation=generation,
            candidate_idx=idx,
            data_override=self.synthetic_data,
            use_inner_optimization=bool(self.search_run_config.get("enable_inner_optimization", False)),
        )
        result["generation"] = generation
        result["index"] = idx
        return result

    def _refine_top_results(self, all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        shortlist_top_k = int(self.run_config.get("shortlist_top_k", max(10, int(self.run_config.get("top_k", 10)))))
        shortlist = top_unique_results(all_results, top_k=shortlist_top_k, search_space=self.search_space)
        if not shortlist:
            return []

        refine_run_config = self._build_stage_run_config(
            symbol_count=int(self.run_config.get("shortlist_symbol_count", 6)),
            enable_inner_optimization=False,
            inner_opt_trials=0,
        )
        refine_evaluator = CandidateEvaluator(self.builder, refine_run_config, data_dir=self.data_dir)

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
                data_override=self.synthetic_data,
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
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self._evaluate_one, generation, idx, ind): idx
                    for idx, ind in enumerate(population)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    results[idx] = future.result()

        return results

    def _assign_fitness(self, population, eval_results: List[Dict[str, Any]]) -> None:
        for individual, row in zip(population, eval_results):
            individual.fitness.values = (float(row.get("score", -1e9)),)

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

        run_started_ts = time.time()
        for generation in range(start_generation, generations):
            if self.max_runtime_seconds is not None:
                elapsed_seconds = max(0.0, time.time() - run_started_ts)
                if elapsed_seconds >= self.max_runtime_seconds:
                    print(
                        "Max runtime reached ({:.2f}h >= {:.2f}h). Stopping before generation {}/{}.".format(
                            elapsed_seconds / 3600.0,
                            self.max_runtime_seconds / 3600.0,
                            generation + 1,
                            generations,
                        )
                    )
                    break

            eval_results = self._evaluate_population(generation, population)
            self._assign_fitness(population, eval_results)
            all_results.extend(eval_results)

            best_score = max(row.get("score", -1e9) for row in eval_results)
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

            elites = self.tools.selBest(population, elite_count)
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

        top_k = int(self.run_config.get("top_k", 10))
        refined_results = self._refine_top_results(all_results)
        top_results = top_unique_results(
            refined_results or all_results,
            top_k=top_k,
            search_space=self.search_space,
        )

        final_evaluator = CandidateEvaluator(self.builder, self.run_config, data_dir=self.data_dir)

        for row in top_results:
            try:
                row["final_metrics"] = final_evaluator.run_final_segment(
                    row["candidate"],
                    data_override=self.synthetic_data,
                )
                row["final_score"] = float(row["score"]) + 0.1 * float(row["final_metrics"].get("total_return", 0.0))
            except Exception as exc:
                row["final_metrics"] = {"total_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0, "total_trades": 0}
                row["final_score"] = float(row["score"])
                row["final_error"] = str(exc)

        top_results = sorted(top_results, key=lambda r: float(r.get("final_score", r.get("score", -1e9))), reverse=True)

        write_reports(self.run_dir, top_results, self.run_config)
        if bool(self.run_config.get("save_strategy_files", True)):
            export_strategy_files(
                run_dir=self.run_dir,
                candidate_builder=self.builder,
                top_results=top_results,
                symbols=self.run_config["symbols"],
                timeframes=self.run_config["timeframes"],
                publish_top_n=int(self.run_config.get("publish_top_n", 0)),
            )

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
    run_dir = _resolve_run_dir(run_config.get("output_dir", "simple_strategy/auto_evolve/output"), args.resume_run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"Run complete. Evaluated: {result['total_evaluated']}")
    print(f"Output: {result['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
