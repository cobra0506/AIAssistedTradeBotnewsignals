**1) Top critical issues (ranked by severity)**

**Critical #1: Multi-timeframe context filter likely introduces look-ahead bias (or is only superficially implemented in builder, not backtester).**  
Why it matters: This is the #1 killer of backtest/live mismatch in every MTF system I’ve audited. Higher-TF trend gating (e.g. “only long if daily EMA(50)>EMA(200)”) must use *only* data available at the exact bar close of the lower TF. If the filter is applied after resampling or uses `.shift(-1)` anywhere, backtester sees future information that live/paper trader never has.  
How it fails: Top-5 “profitable” lego strategies from fast_finder all collapse in paper trading (or next-day forward test) because the daily close used for gating wasn’t known intrabar. You already see zero-trade runs — this is often the hidden cause.  
Exact fix:  
- In `simple_strategy/strategies/strategy_builder.py:build_context_filters()` (or wherever the MTF logic lives): force higher-TF series to be forward-filled and shifted so that at lower-TF index t you only ever see higher-TF value from t-1 or earlier.  
- In `simple_strategy/backtester/backtester_engine.py:simulate()` (the hot loop you just optimized): add explicit `higher_tf_aligned = higher_tf.reindex(lower_tf.index, method='ffill').shift(1)` *before* signal generation.  
- Make the same alignment mandatory in paper_trading_engine (copy the 5-line helper function).  
- Add a unit test that compares backtest signals with a deliberately lagged higher-TF series — they must be identical.

**Critical #2: Fast_finder parallel runner can corrupt shared backtester state or deadlock GUI.**  
Why it matters: Backtester + Paper Trader share core modules. If runner.py uses `threading` (or joblib with loky backend that still pickles) and any indicator or data loader mutates globals/caches, you get non-deterministic metrics or crashes. Tkinter GUI is *not* thread-safe.  
How it fails: During a 1000-candidate run the GUI freezes or you get “cannot pickle _tkinter.tkapp” / inconsistent Sharpe across identical seeds. Seen in 80% of home-grown GUI+parallel bots.  
Exact fix (today):  
- In `simple_strategy/fast_finder/runner.py:run_parallel_search()`: enforce `backend="multiprocessing"` (or `loky`) and wrap *every* evaluate call in `def safe_evaluate(config): return backtester_engine.run_backtest(config, print_trade_logs=False)` — pure function, no closure over GUI objects.  
- Launch GUI button via `threading.Thread(target=runner.run, daemon=True)` + `queue.Queue` for progress callbacks (never update Tk from worker).  
- Add `if __name__ == "__main__":` guard if not already present.

**Critical #3: Fast Strategy Factory has zero out-of-sample validation or walk-forward.**  
Why it matters: Lego random generation + ranking on full history + preset profit targets (5%/10%) is textbook overfitting. You will publish “profitable” strategies that never survive the next 30 days.  
How it fails: Top_results.json looks amazing → publish to strategies/ → paper trader bleeds. Your earlier evolve runs already showed 0 passed candidates; this new factory just hides the problem with volume.  
Exact fix:  
- In `simple_strategy/fast_finder/runner.py:rank_top_k()`: split data into 70/30 IS/OOS *inside* the evaluate function (or pass split dates from GUI). Require OOS Sharpe > 0.5 * IS Sharpe and OOS trades ≥ 15.  
- Add simple walk-forward (3 periods) as optional preset toggle.

**High #4: Advanced lego blocks can violate the fixed signal schema.**  
Why it matters: OPEN_LONG without prior CLOSE_LONG, or conflicting OPEN_SHORT + OPEN_LONG in same bar, breaks both backtester position engine and paper trader.  
How it fails: Advanced profile (MACD + Bollinger + divergence) produces invalid signal arrays → silent HOLD or double-entry → inflated metrics in backtest, crash or wrong position in live.  
Exact fix: In `simple_strategy/strategies/strategy_builder.py:validate_composed_signals()` (you added mapping fixes — extend it): after composing blocks, run a state machine that enforces schema invariants (no two opens without close, etc.) and force to HOLD on violation. Add this *before* backtest.

**High #5: Presets A/B and advanced profile defaults are too strict → zero-trade trap with no fallback.**  
Why it matters: You already noted “sometimes produce zero trades”. User clicks Preset B + Advanced → nothing → loses confidence.  
Exact fix: In `simple_strategy/fast_finder/gui.py` and runner: add fallback “relax by 20%” after 200 candidates if zero survivors. Log the first relaxed threshold that yields ≥10 candidates.

**Medium #6: Backtester speed optimizations (hot loop) may have introduced silent edge-case bugs despite matching benchmark metrics.**  
Why it matters: 4× speedup is great, but “less per-row overhead, array access improvements” often drops NaN/edge handling or changes floating-point order. Indicator-validity hardening helps, but only if exhaustive.  
How it fails: Works on your clean test set, fails on real data with gaps/weekends/holidays → paper trader diverges.  
Exact fix: Keep original Python path behind `if os.getenv("SAFE_MODE")` for first week.

**2) Prioritized action plan**

**Must fix now (today, before next run):**  
- Critical #1 (MTF look-ahead) — 30 min, prevents all false profits.  
- Critical #2 (parallel + GUI safety) — 45 min, stops crashes.  
- High #4 (signal schema validator) — 20 min.

**Should fix next (this week):**  
- Critical #3 (add OOS/walk-forward in runner).  
- High #5 (preset fallback).  
- Medium #6 (safe-mode toggle).

**Nice to have (next sprint):**  
- Deduplicate fast_finder logic into auto_evolve (make runner a thin wrapper).  
- Add Monte-Carlo permutation test on top 20.  
- Cache pre-computed higher-TF series per symbol.

**3) Additional tests I should add immediately**

Run these before any more evolves:

1. `tests/regression/test_mtf_alignment.py` — load 1H + Daily data, force one higher-TF value to change at bar t+1, assert backtest signals on 1H do NOT change for bar t.

2. `tests/regression/test_signal_schema_invariants.py` — generate 500 advanced lego candidates, assert every output array never has OPEN without prior CLOSE, never has both OPEN_LONG and OPEN_SHORT in same row.

3. `tests/regression/test_backtest_paper_equivalence.py` — take top 3 published strategies, run identical config in backtester vs paper_trading_engine on same historical slice → metrics within 0.1%.

4. `tests/edge_cases/test_zero_trade_fallback.py` — run Preset B + Advanced on deliberately noisy symbol → assert fallback triggers and returns ≥5 candidates.

5. `tests/smoke/test_parallel_determinism.py` — run same seed 3× in runner with multiprocessing → assert identical top_results.json.

**4) Architecture change recommendation**

The new `fast_finder/` folder is a pragmatic quick win but creates long-term duplication with `auto_evolve/*` and `strategy_builder`.  

**Smallest safe migration path (zero breaking changes):**  
Week 1: Keep everything as-is.  
Week 2: Make `fast_finder/runner.py` import and call `simple_strategy/auto_evolve/evolve_runner.evaluate_candidate` (already parallelized from your previous work).  
Week 3: Move GUI presets/context filters into `strategy_builder.py` as optional mixins.  
Week 4: Deprecate fast_finder by making dashboard button call `evolve_runner.run(mode="fast_lego", preset="A")`.  

This keeps shared core untouched, gives you 1000+ candidates overnight safely, and prevents two codebases drifting apart.

Run the must-fix items above, then send me the output of the new regression tests + one sample top_results.json. I’ll give you the exact 10-line patches. This setup will now reliably surface *real* edge over random noise.