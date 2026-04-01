# Backtester Diagnostics 2026-03-29

## Findings
- Core long/short execution failed the zero-friction sanity tests.
- Trade execution matches a hand-calculated scripted example.
- The default max_positions setting is materially blocking entries in multi-symbol tests.
- Fees/spread/slippage materially reduce imported-strategy results, but do not fully explain the losses.

## Test Results

### buy_hold_long_zero_friction
- Status: `FAIL`
- expected_market_return_pct: `3.4799`
- backtester_return_pct: `5.1227`
- final_balance: `10512.275`
- trade_count: `1`

### always_short_zero_friction
- Status: `FAIL`
- expected_return_pct: `-3.4799`
- backtester_return_pct: `-5.1227`
- final_balance: `9487.725`
- trade_count: `1`

### scripted_trade_hand_calc_parity
- Status: `PASS`
- expected_final_balance: `9996.6406`
- actual_final_balance: `9996.6406`
- reported_total_return_pct: `-0.0226`
- actual_balance_return_pct: `-0.0336`
- trade_count: `2`
- note: `If reported return differs from actual balance return, fee reporting is wrong.`

### ema_cross_no_optimization
- Status: `PASS`
- total_return_pct: `1.0025`
- final_balance: `10098.5957`
- trade_count: `3`
- max_drawdown_pct: `0.0`
- blocked_max_positions: `4594`

### import_mfi_current_settings
- Status: `INFO`
- total_return_pct: `-1.0213`
- final_balance: `9867.7657`
- trade_count: `110`
- win_rate_pct: `49.0909`
- max_drawdown_pct: `1.3223`
- blocked_max_positions: `17`
- blocked_symbol_already_open: `0`

### import_mfi_max_positions_10
- Status: `INFO`
- total_return_pct: `-0.8723`
- final_balance: `9877.9939`
- trade_count: `127`
- win_rate_pct: `51.9685`
- max_drawdown_pct: `1.2201`
- blocked_max_positions: `0`
- blocked_symbol_already_open: `0`

### import_mfi_max_positions_10_zero_friction
- Status: `INFO`
- total_return_pct: `0.3603`
- final_balance: `10036.0257`
- trade_count: `127`
- win_rate_pct: `61.4173`
- max_drawdown_pct: `0.0`
- blocked_max_positions: `0`
- blocked_symbol_already_open: `0`

### import_mfi_no_trend_gate_zero_friction
- Status: `INFO`
- total_return_pct: `-1.3065`
- final_balance: `9869.3472`
- trade_count: `683`
- win_rate_pct: `60.6149`
- max_drawdown_pct: `1.3065`
- blocked_max_positions: `0`
- blocked_symbol_already_open: `0`
