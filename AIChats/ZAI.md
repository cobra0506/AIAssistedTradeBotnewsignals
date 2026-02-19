This is a strict technical review based on the implementation details provided. My assessment focuses on the intersection of **Speed vs. Correctness** and **Generation vs. Robustness**.

Here is the breakdown.

### 1. Critical Issues

#### Issue A: Multi-Timeframe (MTF) Look-Ahead Bias
**Severity: Critical**
**Location:** `simple_strategy/backtester/backtester_engine.py` (or wherever MTF data is merged) & `simple_strategy/strategies/strategy_builder.py`.

*   **Why it matters:** When you use a "Higher Timeframe Context Filter" (e.g., Daily Trend to filter Hourly entries), the most common error is leaking future information. If at 10:00 AM you check the Daily Close, the daily candle hasn't closed yet. If your code uses the *close* of the daily candle at 10:00 AM, you are peeking into the future.
*   **How it fails:** The Fast Finder will generate strategies that look incredibly profitable because they "know" the daily trend direction before the day is over. When you run this in Paper Trading, the performance will collapse because the daily signal is uncertain until 23:59.
*   **Exact Fix:**
    1.  **Data Alignment:** Do not merge MTF data by timestamp index directly.
    2.  **Forward Fill (`ffill`):** When resampling Higher Timeframe (HTF) to Lower Timeframe (LTF), take the HTF signal (e.g., SMA value) and assign it to *all* LTF candles belonging to that HTF period.
    3.  **Shift:** Ensure the HTF signal for the *current* LTF candle is actually the signal from the *previous completed* HTF candle.
    4.  **Code Change (Pandas):**
        ```python
        # Wrong (Peeking)
        df_merged = df.join(htf_df, rsuffix='_htf')
        
        # Right (Safe)
        htf_signal = htf_df['sma_200'].shift(1) # Use previous close value
        df_merged = df.join(htf_signal, rsuffix='_htf').ffill()
        ```

#### Issue B: Shared Code Logic (Vectorized vs. Live Tick)
**Severity: Critical**
**Location:** `simple_strategy/strategies/strategy_builder.py`

*   **Why it matters:** You optimized the Backtester for speed (likely using vectorization or array access), but the Paper Trader processes data tick-by-tick (streaming).
*   **How it fails:** If you modified `strategy_builder` to generate signals using pandas operations (e.g., `np.where(df['rsi'] < 30)`), the Paper Trader cannot call this method on a single tick without passing a DataFrame, or it might calculate indicators incorrectly on a rolling window. If the logic diverges, the backtest says "Profit", the paper trader says "Loss" (or doesn't trade).
*   **Exact Fix:**
    1.  Ensure the `Strategy` class has a dedicated method `calculate_indicators` that works on a DataFrame.
    2.  Ensure the Paper Trader calls this method on a "sliding window" DataFrame (e.g., `df.iloc[-lookback_period:]`) or maintains a rolling state buffer.
    3.  **Verification:** Write a test that feeds the *exact same data sequence* to both the Backtester (in one chunk) and the Paper Trader (tick-by-tick) and asserts the generated signals are identical at the close of every candle.

#### Issue C: Index Alignment from "Indicator-Validity Handling"
**Severity: High**
**Location:** `simple_strategy/backtester/backtester_engine.py`

*   **Why it matters:** You mentioned "Indicator-validity handling hardened". If this involves dropping `NaN` values (e.g., `df.dropna()`), and you don't re-align the signals index with the price index, your entry/exit signals will be offset by the number of dropped rows.
*   **How it fails:** Signal `t` gets aligned with Price `t+X`. The backtester buys at the wrong price (usually lagging), drastically reducing realized profit and increasing drawdown.
*   **Exact Fix:**
    1.  Never drop rows from the price data.
    2.  Only fill/forward-fill indicator columns.
    3.  Explicitly assert `len(signals) == len(prices)` and `signals.index.equals(prices.index)` before the execution loop.

---

### 2. High Priority Issues

#### Issue D: Lack of Out-of-Sample (OOS) Testing in Fast Finder
**Severity: High**
**Location:** `simple_strategy/fast_finder/runner.py`

*   **Why it matters:** "Generates many random candidates... ranks top K". If you rank them based on the *same* data used to generate them, you are curve-fitting. You will find the strategy that best fits the noise of the last 6 months.
*   **How it fails:** Strategies pass the "Fast Finder" but fail immediately in Paper Trading.
*   **Exact Fix:**
    1.  In `runner.py`, split input data into `Train` (e.g., first 70%) and `Test` (last 30%).
    2.  Run the optimization/fitness check on `Train`.
    3.  **Crucial:** Re-run the backtest on `Test` data for the top K candidates.
    4.  Only "publish" strategies that are profitable on *both* sets (or at least don't lose money on Test).

#### Issue E: "Target Profile" Risk (A/B Presets)
**Severity: High**
**Location:** `simple_strategy/fast_finder/gui.py` / Logic

*   **Why it matters:** You have "5% target" and "10% target".
*   **How it fails:** If "target" refers to **Return**, the bot will select the highest leverage/volatility strategies to hit that number, ignoring risk. If it refers to **Win Rate**, it will select strategies with huge Risk/Reward ratios (1 winner pays for 10 losses).
*   **Exact Fix:**
    1.  Change the presets to be **Risk-Adjusted** targets. e.g., "Sharpe Ratio > 1.5" or "Max Drawdown < 15%".
    2.  Ensure the filter is `Return > X AND Drawdown < Y`. Never filter on Return alone.

---

### 3. Prioritized Action Plan

#### Must Fix Now (Before running overnight)
1.  **Verify MTF Logic (Issue A):** Manually inspect the code where HTF data is joined to LTF data. Ensure `ffill()` and `shift(1)` are used. Run a backtest of a known losing strategy with MTF enabled. If it suddenly becomes profitable, you have a bias bug.
2.  **Add Index Assertion (Issue C):** Add `assert signals.index.equals(price_data.index)` in `backtester_engine.py` before the loop runs.
3.  **Parity Test (Issue B):** Run the smoke test for paper trader against the backtester for a 100-candle window. Log every signal. They must be identical.

#### Should Fix Next (This Week)
4.  **Implement OOS Split (Issue D):** Update `runner.py` to slice data. This is the single biggest improvement to strategy quality.
5.  **Review Preset Logic (Issue E):** Check the filtering logic in `runner.py`. Add a minimum trade count filter (e.g., `trades >= 30`) to avoid selecting strategies that took 1 lucky trade.

#### Nice to Have (Next Iteration)
6.  **Transaction Cost Injection:** Ensure the "Fast Finder" is injecting costs (slippage + fees) *inside* the hot loop, not subtracted at the end. (Mentioned in non-negotiable constraints, but often skipped in "fast" prototypes).

---

### 4. Additional Tests to Add Immediately

1.  **The "Determinism" Test:**
    *   *Idea:* Run the exact same Fast Finder run twice.
    *   *Assert:* The `top_results.json` must be byte-identical.
    *   *Why:* If random seeds aren't fixed, you can't debug failures.

2.  **The "Flatline" Test:**
    *   *Idea:* Feed a data file that is a straight line (price = 100 for all rows).
    *   *Assert:* No strategy should generate `OPEN_LONG` or `OPEN_SHORT` (unless it's a breakout logic on flat data, which is bad, but more importantly, equity should remain flat minus costs).
    *   *Why:* Checks for logic errors that buy/sell due to `NaN` comparisons or indicator initialization glitches.

3.  **The "Short Circuit" Test:**
    *   *Idea:* Create a strategy that *always* returns `HOLD`.
    *   *Assert:* Backtest completes in milliseconds. Number of trades = 0. Final Equity = Initial Equity.
    *   *Why:* Tests the loop overhead and ensures "0 trades" doesn't crash the reporting engine.

---

### 5. Architecture Suggestion (Migration Path)

**Current State:** `Backtester` and `Strategy Builder` are tightly coupled.
**Problem:** Adding "Advanced Filters" (Regime/MTF) likely clutters the `strategy_builder.py` with spaghetti `if/else` logic about data shapes.

**Proposed Safe Migration:**

**Step 1: Introduce `DataContext` Class**
Create a class that holds the *aligned* price data, indicators, and context filters.
```python
class DataContext:
    def __init__(self, price_df, mtf_df, regime_df):
        self.primary = price_df
        # Handle alignment here, once, centrally
        self.context = self._merge_context(mtf_df, regime_df) 
        
    def _merge_context(self, mtf, regime):
        # Implementation of the ffill/shift logic from Issue A
        pass
```

**Step 2: Refactor `StrategyBuilder`**
Change `calculate_signals` signature from `calculate_signals(data)` to `calculate_signals(context: DataContext)`.

**Step 3: Isolate "Generator" Logic**
Keep the `RandomStrategyFactory` strictly for defining *parameters* (JSON/Dict).
Keep the `StrategyBuilder` strictly for *execution* (Pandas/Numpy).

**Why this helps:** It decouples the *data preparation* (where the bugs live) from the *signal generation*. It makes it much easier to unit test the MTF alignment logic independently of the trading logic.