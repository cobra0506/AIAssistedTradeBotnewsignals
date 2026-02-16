# Auto Evolution (DEAP + Bayesian)

This module adds an unattended strategy search loop without changing existing project flows.

## What it does
- Outer loop: DEAP evolves strategy structure (indicator/signal combinations)
- Inner loop: Optuna tunes parameters for each promising structure
- Evaluation gates: train, validation, stress, and final segment
- Outputs: checkpoints, top results, and optional generated strategy files

## Run
```powershell
python -m simple_strategy.auto_evolve.run_evolution --config simple_strategy/auto_evolve/configs/default.json
```

## Resume
```powershell
python -m simple_strategy.auto_evolve.run_evolution --resume-run-dir simple_strategy/auto_evolve/output/run_YYYYMMDD_HHMMSS
```

## Smoke test (synthetic data)
```powershell
python -m simple_strategy.auto_evolve.run_evolution --smoke --population 8 --generations 2 --workers 1
```
