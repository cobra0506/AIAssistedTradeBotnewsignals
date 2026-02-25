#paper_trading_launcher.py

import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os
import json
import random
import time
import traceback
import importlib
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def write_launcher_crash_log(error):
    logs_dir = os.path.join(project_root, "Logs")
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    crash_log_path = os.path.join(logs_dir, f"paper_trader_launcher_crash_{timestamp}.log")
    with open(crash_log_path, "w", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} Paper trader launcher crashed\n")
        f.write(f"Error: {error}\n\n")
        f.write(traceback.format_exc())
    return crash_log_path

class PaperTradingLauncher:
    def __init__(self, api_account=None, strategy_name=None, simulated_balance=None):
        # If parameters not provided, get from command line arguments
        if api_account is None:
            import sys
            if len(sys.argv) >= 4:
                api_account = sys.argv[1]
                strategy_name = sys.argv[2]
                simulated_balance = sys.argv[3]
            else:
                # Default values for testing
                api_account = "Demo Account 1"
                strategy_name = "Strategy_Simple_RSI"
                simulated_balance = "1000"

        self.api_account = api_account
        self.strategy_name = strategy_name
        self.simulated_balance = float(simulated_balance)  # Convert to float
        
        # Create GUI window
        self.root = tk.Tk()
        self.root.title(f"Paper Trading - {strategy_name}")
        self.root.geometry("1180x720")
        self.root.minsize(1024, 680)
        
        # Initialize trading engine
        self.trading_engine = None
        self.graceful_stop_window = None
        self.graceful_stop_status_var = None
        self.graceful_stop_count_var = None
        self.graceful_close_now_btn = None
        self.graceful_safely_close_btn = None
        self._graceful_stop_after_id = None
        self.strategy_parameter_definitions = {}
        self.strategy_parameter_widgets = {}
        self.optimized_strategy_params = {}

        self.create_widgets()
        
    def create_widgets(self):
        # Header with performance info
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill="x", padx=10, pady=5)
        header_frame.columnconfigure(0, weight=1)
        header_frame.columnconfigure(1, weight=0)

        # Left side - Strategy and Account info
        left_frame = ttk.Frame(header_frame)
        left_frame.grid(row=0, column=0, sticky="ew", padx=(0, 12))
        
        ttk.Label(left_frame, text=f"Paper Trading: {self.strategy_name}", 
                font=("Arial", 14, "bold")).pack(anchor="w")
        
        account_frame = ttk.Frame(left_frame)
        account_frame.pack(fill="x", pady=2)
        ttk.Label(account_frame, text=f"Account: {self.api_account}").pack(side="left", padx=5)
        ttk.Label(account_frame, text=f"Initial Balance: ${self.simulated_balance}").pack(side="left", padx=5)
        
        # Parameter status
        param_frame = ttk.Frame(left_frame)
        param_frame.pack(fill="x", pady=2)
        
        # Check for optimized parameters
        from simple_strategy.trading.parameter_manager import ParameterManager
        pm = ParameterManager()
        optimized_params = pm.get_parameters(self.strategy_name)
        self.optimized_strategy_params = dict(optimized_params or {})

        if optimized_params:
            param_status = f"Using optimized parameters (Last: {optimized_params.get('last_optimized', 'Unknown')})"
            param_color = "green"
        else:
            param_status = "Using default parameters (Not optimized)"
            param_color = "orange"
        
        ttk.Label(param_frame, text=param_status, foreground=param_color).pack(side="left", padx=5)
        
        # Right side - Performance info
        right_frame = ttk.Frame(header_frame)
        right_frame.grid(row=0, column=1, sticky="ne")

        # Performance display in header (split into two columns)
        perf_header_frame = ttk.LabelFrame(right_frame, text="Performance", padding=5)
        perf_header_frame.pack(fill="both", expand=True)

        left_perf_frame = ttk.Frame(perf_header_frame)
        left_perf_frame.pack(side="left", padx=5)

        right_perf_frame = ttk.Frame(perf_header_frame)
        right_perf_frame.pack(side="left", padx=5)

        self.perf_header_labels = {}

        left_items = [
            ("Account Value:", "account_value", "$1000.00"),
            ("If Close Now:", "liquidation_value", "$1000.00"),
            ("Available Bal:", "available_balance", "$1000.00"),
            ("Open:", "open_positions", "0")
        ]

        right_items = [
            ("Realized P&L:", "realized_pnl", "$0.00"),
            ("Unrealized P&L:", "unrealized_pnl", "$0.00"),
            ("Total Trades:", "total_trades", "0"),
            ("Win Rate:", "win_rate", "0.0%")
        ]

        for i, (label_text, key, default) in enumerate(left_items):
            label = ttk.Label(left_perf_frame, text=label_text)
            label.grid(row=i, column=0, sticky="e", padx=2)
            value_label = ttk.Label(left_perf_frame, text=default, font=("Arial", 10, "bold"))
            value_label.grid(row=i, column=1, sticky="w", padx=2)
            self.perf_header_labels[key] = value_label

        for i, (label_text, key, default) in enumerate(right_items):
            label = ttk.Label(right_perf_frame, text=label_text)
            label.grid(row=i, column=0, sticky="e", padx=2)
            value_label = ttk.Label(right_perf_frame, text=default, font=("Arial", 10, "bold"))
            value_label.grid(row=i, column=1, sticky="w", padx=2)
            self.perf_header_labels[key] = value_label

        
        # Control buttons and status
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill="x", padx=10, pady=5)
        
        self.start_btn = ttk.Button(control_frame, text="START TRADING", 
                                command=self.start_trading)
        self.start_btn.pack(side="left", padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="STOP TRADING", 
                                command=self.stop_trading, state="disabled")
        self.stop_btn.pack(side="left", padx=5)

        self.data_feed_status_var = tk.StringVar(value="Data feed: checking...")
        self.data_feed_status_label = ttk.Label(
            control_frame,
            textvariable=self.data_feed_status_var,
            font=("Arial", 9, "bold"),
            foreground="gray",
        )
        self.data_feed_status_label.pack(side="left", padx=10)
        
        # Status with color
        self.status_var = tk.StringVar(value="STOPPED")
        self.status_label = ttk.Label(control_frame, textvariable=self.status_var, 
                                    font=("Arial", 10, "bold"))
        self.status_label.pack(side="left", padx=20)
        
        # NEW: Trading controls
        trading_controls_frame = ttk.LabelFrame(self.root, text="Trading Controls", padding=10)
        trading_controls_frame.pack(fill="x", padx=10, pady=5)
        
        # Max positions
        max_positions_frame = ttk.Frame(trading_controls_frame)
        max_positions_frame.pack(fill="x", pady=2)
        ttk.Label(max_positions_frame, text="Max Positions:").pack(side="left", padx=5)
        self.max_positions_var = tk.IntVar(value=20)
        max_positions_spinbox = ttk.Spinbox(max_positions_frame, from_=1, to=1000, textvariable=self.max_positions_var, width=10)
        max_positions_spinbox.pack(side="left", padx=5)

        timeframe_frame = ttk.Frame(trading_controls_frame)
        timeframe_frame.pack(fill="x", pady=2)
        ttk.Label(timeframe_frame, text="Strategy Entry Timeframe:").pack(side="left", padx=5)
        self.entry_timeframe_var = tk.StringVar(value="1m")
        ttk.Entry(timeframe_frame, textvariable=self.entry_timeframe_var, width=10).pack(side="left", padx=5)
        ttk.Label(timeframe_frame, text="Examples: 1m, 5m, 1h").pack(side="left", padx=2)
        self._build_strategy_parameter_controls(trading_controls_frame)

        # Exchange stop-loss controls
        exchange_sl_frame = ttk.Frame(trading_controls_frame)
        exchange_sl_frame.pack(fill="x", pady=2)
        self.exchange_sl_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            exchange_sl_frame,
            text="Enable Exchange Stop Loss",
            variable=self.exchange_sl_enabled_var,
            command=self.toggle_exchange_sl_entry,
        ).pack(side="left", padx=5)
        self.exchange_trailing_enabled_var = tk.BooleanVar(value=False)
        self.exchange_trailing_check = ttk.Checkbutton(
            exchange_sl_frame,
            text="Use Trailing Stop",
            variable=self.exchange_trailing_enabled_var,
            command=self._update_exchange_stop_mode_label,
            state="disabled",
        )
        self.exchange_trailing_check.pack(side="left", padx=5)
        self.exchange_sl_pct_var = tk.DoubleVar(value=2.0)
        self.exchange_sl_pct_spinbox = ttk.Spinbox(
            exchange_sl_frame,
            from_=0.1,
            to=20.0,
            increment=0.1,
            textvariable=self.exchange_sl_pct_var,
            width=6,
            command=self._update_exchange_stop_mode_label,
            state="disabled",
        )
        self.exchange_sl_pct_spinbox.pack(side="left", padx=2)
        self.exchange_sl_pct_spinbox.bind("<KeyRelease>", lambda _event: self._update_exchange_stop_mode_label())
        self.exchange_sl_pct_spinbox.bind("<FocusOut>", lambda _event: self._update_exchange_stop_mode_label())
        ttk.Label(exchange_sl_frame, text="%").pack(side="left", padx=2)
        self.exchange_stop_mode_var = tk.StringVar(value="No stop loss")
        self.exchange_stop_mode_label = ttk.Label(
            trading_controls_frame,
            textvariable=self.exchange_stop_mode_var,
            font=("Arial", 9, "bold"),
            foreground="gray",
        )
        self.exchange_stop_mode_label.pack(anchor="w", padx=8, pady=(0, 2))

        atr_frame = ttk.Frame(trading_controls_frame)
        atr_frame.pack(fill="x", pady=2)
        self.atr_mode_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            atr_frame,
            text="Use ATR Stop + TP",
            variable=self.atr_mode_enabled_var,
            command=self.toggle_atr_controls,
        ).pack(side="left", padx=5)
        ttk.Label(atr_frame, text="ATR Period:").pack(side="left", padx=(8, 2))
        self.atr_period_var = tk.IntVar(value=14)
        self.atr_period_spinbox = ttk.Spinbox(
            atr_frame,
            from_=2,
            to=200,
            textvariable=self.atr_period_var,
            width=5,
            state="disabled",
            command=self._update_exchange_stop_mode_label,
        )
        self.atr_period_spinbox.pack(side="left", padx=2)
        ttk.Label(atr_frame, text="SL x").pack(side="left", padx=(8, 2))
        self.atr_sl_mult_var = tk.DoubleVar(value=1.3)
        self.atr_sl_mult_spinbox = ttk.Spinbox(
            atr_frame,
            from_=0.1,
            to=10.0,
            increment=0.1,
            textvariable=self.atr_sl_mult_var,
            width=5,
            state="disabled",
            command=self._update_exchange_stop_mode_label,
        )
        self.atr_sl_mult_spinbox.pack(side="left", padx=2)
        ttk.Label(atr_frame, text="TP x").pack(side="left", padx=(8, 2))
        self.atr_tp_mult_var = tk.DoubleVar(value=1.7)
        self.atr_tp_mult_spinbox = ttk.Spinbox(
            atr_frame,
            from_=0.1,
            to=10.0,
            increment=0.1,
            textvariable=self.atr_tp_mult_var,
            width=5,
            state="disabled",
            command=self._update_exchange_stop_mode_label,
        )
        self.atr_tp_mult_spinbox.pack(side="left", padx=2)
        self.atr_period_spinbox.bind("<KeyRelease>", lambda _event: self._update_exchange_stop_mode_label())
        self.atr_sl_mult_spinbox.bind("<KeyRelease>", lambda _event: self._update_exchange_stop_mode_label())
        self.atr_tp_mult_spinbox.bind("<KeyRelease>", lambda _event: self._update_exchange_stop_mode_label())
        
        # Stop trading at percentage
        stop_percentage_frame = ttk.Frame(trading_controls_frame)
        stop_percentage_frame.pack(fill="x", pady=2)
        self.stop_at_percentage_var = tk.BooleanVar(value=False)
        stop_percentage_check = ttk.Checkbutton(stop_percentage_frame, text="Stop trading when balance falls below", 
                                            variable=self.stop_at_percentage_var, command=self.toggle_percentage_entry)
        stop_percentage_check.pack(side="left", padx=5)
        self.stop_percentage_value = tk.IntVar(value=10)
        self.stop_percentage_spinbox = ttk.Spinbox(stop_percentage_frame, from_=1, to=100, textvariable=self.stop_percentage_value, width=5, state="disabled")
        self.stop_percentage_spinbox.pack(side="left", padx=2)
        ttk.Label(stop_percentage_frame, text="% of initial balance").pack(side="left", padx=2)
        
        # Trading log
        log_frame = ttk.LabelFrame(self.root, text="Trading Log", padding=10)
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create text widget with scrollbar
        self.log_text = tk.Text(log_frame, height=15, width=80)
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.log_message("Paper trading window initialized")
        self.update_performance()
        self.refresh_data_feed_status()
        self._update_exchange_stop_mode_label()

    def toggle_exchange_sl_entry(self):
        """Enable/disable fixed or trailing stop-loss controls."""
        enabled = bool(self.exchange_sl_enabled_var.get())
        if enabled:
            self.exchange_sl_pct_spinbox.config(state="normal")
            self.exchange_trailing_check.config(state="normal")
        else:
            self.exchange_sl_pct_spinbox.config(state="disabled")
            self.exchange_trailing_check.config(state="disabled")
            self.exchange_trailing_enabled_var.set(False)
        self._update_exchange_stop_mode_label()

    def toggle_atr_controls(self):
        """Enable/disable ATR risk controls."""
        atr_enabled = bool(self.atr_mode_enabled_var.get())
        state = "normal" if atr_enabled else "disabled"
        self.atr_period_spinbox.config(state=state)
        self.atr_sl_mult_spinbox.config(state=state)
        self.atr_tp_mult_spinbox.config(state=state)
        self._update_exchange_stop_mode_label()

    def toggle_percentage_entry(self):
        """Enable/disable stop-on-balance-percent control."""
        state = "normal" if bool(self.stop_at_percentage_var.get()) else "disabled"
        self.stop_percentage_spinbox.config(state=state)

    def _to_bool(self, value, default=False):
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _load_strategy_parameter_definitions(self):
        try:
            module_name = f"simple_strategy.strategies.{self.strategy_name}"
            strategy_module = importlib.import_module(module_name)
            params = getattr(strategy_module, "STRATEGY_PARAMETERS", {})
            if isinstance(params, dict):
                return params
        except Exception:
            return {}
        return {}

    def _build_strategy_parameter_controls(self, parent_frame):
        self.strategy_parameter_widgets = {}
        self.strategy_parameter_definitions = self._load_strategy_parameter_definitions()

        strategy_params_frame = ttk.LabelFrame(parent_frame, text="Strategy Parameters", padding=8)
        strategy_params_frame.pack(fill="x", pady=2)

        if not self.strategy_parameter_definitions:
            ttk.Label(
                strategy_params_frame,
                text="No strategy-specific parameters detected.",
                foreground="gray",
            ).pack(anchor="w", padx=5, pady=2)
            return

        for param_name, param_info in self.strategy_parameter_definitions.items():
            if param_name == "entry_timeframe":
                continue

            param_type = str(param_info.get("type", "str")).strip().lower()
            default_value = param_info.get("default", "")
            initial_value = self.optimized_strategy_params.get(param_name, default_value)

            row = ttk.Frame(strategy_params_frame)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=f"{param_name}:").pack(side="left", padx=5)

            if param_type == "bool":
                var = tk.BooleanVar(
                    value=self._to_bool(initial_value, default=self._to_bool(default_value))
                )
                ttk.Checkbutton(row, variable=var).pack(side="left", padx=4)
            else:
                text_value = "" if initial_value is None else str(initial_value)
                var = tk.StringVar(value=text_value)
                width = 10 if param_type in {"int", "float"} else 14
                ttk.Entry(row, textvariable=var, width=width).pack(side="left", padx=4)

            hint = str(param_info.get("gui_hint", "")).strip()
            if hint:
                ttk.Label(row, text=hint, foreground="gray").pack(side="left", padx=6)

            self.strategy_parameter_widgets[param_name] = {
                "var": var,
                "type": param_type,
                "info": dict(param_info or {}),
            }

    def _coerce_strategy_param_value(self, param_name, param_type, raw_value):
        if param_type == "bool":
            return self._to_bool(raw_value, default=False)

        if param_type == "int":
            try:
                return int(float(raw_value))
            except (TypeError, ValueError):
                raise ValueError(f"{param_name} must be an integer.")

        if param_type == "float":
            try:
                return float(raw_value)
            except (TypeError, ValueError):
                raise ValueError(f"{param_name} must be a number.")

        return "" if raw_value is None else str(raw_value)

    def _collect_strategy_param_overrides(self):
        if not isinstance(self.strategy_parameter_widgets, dict) or not self.strategy_parameter_widgets:
            return {}

        overrides = {}
        errors = []
        for param_name, meta in self.strategy_parameter_widgets.items():
            var = meta.get("var")
            param_type = str(meta.get("type", "str")).strip().lower()
            try:
                raw_value = var.get()
                overrides[param_name] = self._coerce_strategy_param_value(
                    param_name, param_type, raw_value
                )
            except Exception as exc:
                errors.append(str(exc))

        if errors:
            messagebox.showerror(
                "Invalid Strategy Parameters",
                "\n".join(errors[:5]),
            )
            return None

        return overrides

    def _update_exchange_stop_mode_label(self):
        """Show active stop mode in plain language."""
        try:
            atr_enabled = bool(self.atr_mode_enabled_var.get())
            sl_enabled = bool(self.exchange_sl_enabled_var.get())
            trailing_enabled = bool(self.exchange_trailing_enabled_var.get())
            sl_pct = float(self.exchange_sl_pct_var.get())
            atr_period = int(float(self.atr_period_var.get()))
            atr_sl_mult = float(self.atr_sl_mult_var.get())
            atr_tp_mult = float(self.atr_tp_mult_var.get())
        except Exception:
            self.exchange_stop_mode_var.set("No stop loss")
            self.exchange_stop_mode_label.config(foreground="gray")
            return

        if atr_enabled:
            self.exchange_stop_mode_var.set(
                f"ATR stop/TP: ATR {atr_period}, SL x{atr_sl_mult:.2f}, TP x{atr_tp_mult:.2f}"
            )
            self.exchange_stop_mode_label.config(foreground="green")
            return

        if not sl_enabled:
            self.exchange_stop_mode_var.set("No stop loss")
            self.exchange_stop_mode_label.config(foreground="gray")
            return

        if trailing_enabled:
            self.exchange_stop_mode_var.set(f"Trailing stop loss {sl_pct:.2f}%")
            self.exchange_stop_mode_label.config(foreground="green")
            return

        self.exchange_stop_mode_var.set(f"Stop loss {sl_pct:.2f}%")
        self.exchange_stop_mode_label.config(foreground="green")

    def update_performance_header(self, performance_data):
        """Update header metrics without touching the log widget."""
        if not isinstance(performance_data, dict):
            return
        if not hasattr(self, "perf_header_labels"):
            return

        account_value = float(
            performance_data.get(
                "account_value",
                performance_data.get("current_balance", self.simulated_balance),
            )
        )
        liquidation_value = float(
            performance_data.get(
                "liquidation_value",
                performance_data.get("current_balance", self.simulated_balance),
            )
        )
        available_balance = float(
            performance_data.get(
                "available_balance",
                performance_data.get("current_balance", self.simulated_balance),
            )
        )
        open_positions = int(performance_data.get("open_positions", 0))
        realized_pnl = float(performance_data.get("realized_pnl", 0.0))
        unrealized_pnl = float(performance_data.get("unrealized_pnl", 0.0))
        total_trades = int(performance_data.get("total_trades", 0))
        win_rate = float(performance_data.get("win_rate", 0.0))

        updates = {
            "account_value": f"${account_value:,.2f}",
            "liquidation_value": f"${liquidation_value:,.2f}",
            "available_balance": f"${available_balance:,.2f}",
            "open_positions": f"{open_positions}",
            "realized_pnl": f"${realized_pnl:,.2f}",
            "unrealized_pnl": f"${unrealized_pnl:,.2f}",
            "total_trades": f"{total_trades}",
            "win_rate": f"{win_rate:.1f}%",
        }

        for key, value in updates.items():
            label = self.perf_header_labels.get(key)
            if label is None:
                continue
            label.config(text=value)

        for key, pnl_value in (
            ("realized_pnl", realized_pnl),
            ("unrealized_pnl", unrealized_pnl),
        ):
            label = self.perf_header_labels.get(key)
            if label is None:
                continue
            if pnl_value > 0:
                label.config(foreground="green")
            elif pnl_value < 0:
                label.config(foreground="red")
            else:
                label.config(foreground="black")

    def update_status(self, status):
        """Update status display with appropriate colors"""
        self.status_var.set(status)
        
        # Update color based on status
        if "RUNNING" in status:
            self.status_label.config(foreground="green")
        else:
            self.status_label.config(foreground="black")

    def _set_stopped_ui(self):
        self.status_var.set("STOPPED")
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def _close_graceful_stop_dialog(self):
        if self._graceful_stop_after_id is not None:
            try:
                self.root.after_cancel(self._graceful_stop_after_id)
            except Exception:
                pass
            self._graceful_stop_after_id = None

        if self.graceful_stop_window is not None:
            try:
                if self.graceful_stop_window.winfo_exists():
                    self.graceful_stop_window.destroy()
            except Exception:
                pass
        self.graceful_stop_window = None
        self.graceful_stop_status_var = None
        self.graceful_stop_count_var = None
        self.graceful_close_now_btn = None
        self.graceful_safely_close_btn = None

    def _safe_close_after_graceful_stop(self):
        if self.trading_engine:
            try:
                self.trading_engine.stop_trading(immediate=True)
            except Exception:
                pass
        self.log_message("Paper trading safely stopped.")
        self._close_graceful_stop_dialog()
        self._set_stopped_ui()

    def _force_close_now_from_dialog(self):
        if not self.trading_engine:
            return
        if hasattr(self.trading_engine, "request_force_close_all"):
            self.trading_engine.request_force_close_all()
            self.status_var.set("STOPPING (FORCE CLOSE)")
            self.log_message("Force close requested: closing open positions now...")
        else:
            self.log_message("Force close API not available, stopping immediately.")
            self.trading_engine.stop_trading(immediate=True)
            self._safe_close_after_graceful_stop()

    def _refresh_graceful_stop_dialog(self):
        if self.graceful_stop_window is None:
            return
        try:
            if not self.graceful_stop_window.winfo_exists():
                self._close_graceful_stop_dialog()
                return
        except Exception:
            self._close_graceful_stop_dialog()
            return

        open_positions = 0
        is_running = False
        graceful_requested = False
        if self.trading_engine:
            try:
                if hasattr(self.trading_engine, "get_shutdown_state"):
                    state = self.trading_engine.get_shutdown_state() or {}
                    open_positions = int(state.get("open_positions", 0))
                    is_running = bool(state.get("is_running", False))
                    graceful_requested = bool(state.get("graceful_stop_requested", False))
                else:
                    open_positions = len(getattr(self.trading_engine, "current_positions", {}))
                    is_running = bool(getattr(self.trading_engine, "is_running", False))
            except Exception:
                open_positions = len(getattr(self.trading_engine, "current_positions", {}))
                is_running = bool(getattr(self.trading_engine, "is_running", False))

        if self.graceful_stop_count_var is not None:
            self.graceful_stop_count_var.set(f"Open positions: {open_positions}")

        if self.graceful_stop_status_var is not None:
            if open_positions > 0:
                self.graceful_stop_status_var.set(
                    "New trades are blocked. Waiting for open positions to close..."
                )
            elif graceful_requested or not is_running:
                self.graceful_stop_status_var.set(
                    "All positions are closed. You can safely close now."
                )
            else:
                self.graceful_stop_status_var.set("Waiting for shutdown state...")

        if self.graceful_safely_close_btn is not None:
            self.graceful_safely_close_btn.config(
                state="normal" if open_positions == 0 else "disabled"
            )
        if self.graceful_close_now_btn is not None and open_positions == 0:
            self.graceful_close_now_btn.config(state="disabled")

        self._graceful_stop_after_id = self.root.after(1000, self._refresh_graceful_stop_dialog)

    def _open_graceful_stop_dialog(self):
        if self.graceful_stop_window is not None:
            try:
                if self.graceful_stop_window.winfo_exists():
                    self.graceful_stop_window.lift()
                    self._refresh_graceful_stop_dialog()
                    return
            except Exception:
                pass

        win = tk.Toplevel(self.root)
        win.title("Graceful Stop")
        win.geometry("440x180")
        win.transient(self.root)
        win.grab_set()
        self.graceful_stop_window = win

        self.graceful_stop_status_var = tk.StringVar(
            value="Graceful stop requested. New entries are disabled."
        )
        self.graceful_stop_count_var = tk.StringVar(value="Open positions: 0")

        ttk.Label(
            win,
            text="Paper trader is stopping gracefully.",
            font=("Arial", 11, "bold"),
        ).pack(anchor="w", padx=12, pady=(12, 6))
        ttk.Label(
            win,
            textvariable=self.graceful_stop_status_var,
            wraplength=410,
        ).pack(anchor="w", padx=12, pady=2)
        ttk.Label(
            win,
            textvariable=self.graceful_stop_count_var,
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", padx=12, pady=(4, 8))

        button_row = ttk.Frame(win)
        button_row.pack(fill="x", padx=12, pady=(4, 12))
        self.graceful_close_now_btn = ttk.Button(
            button_row,
            text="Close Now",
            command=self._force_close_now_from_dialog,
        )
        self.graceful_close_now_btn.pack(side="left", padx=(0, 8))
        self.graceful_safely_close_btn = ttk.Button(
            button_row,
            text="Safely Close",
            state="disabled",
            command=self._safe_close_after_graceful_stop,
        )
        self.graceful_safely_close_btn.pack(side="left")

        def _on_close_attempt():
            if self.graceful_safely_close_btn and str(self.graceful_safely_close_btn.cget("state")) == "normal":
                self._safe_close_after_graceful_stop()
            else:
                messagebox.showinfo(
                    "Graceful Stop",
                    "Use 'Close Now' or wait until open positions reach 0.",
                )

        win.protocol("WM_DELETE_WINDOW", _on_close_attempt)
        self._refresh_graceful_stop_dialog()

    def stop_trading(self, graceful=True):
        """Stop paper trading; graceful mode blocks entries and waits for open positions to close."""
        if not self.trading_engine:
            self.log_message("Paper trading is not running.")
            self._set_stopped_ui()
            return

        open_positions = len(getattr(self.trading_engine, "current_positions", {}))
        if not graceful or open_positions == 0:
            self.trading_engine.stop_trading(immediate=True)
            self.log_message("Paper trading stopped.")
            self._close_graceful_stop_dialog()
            self._set_stopped_ui()
            return

        if hasattr(self.trading_engine, "request_graceful_stop"):
            self.trading_engine.request_graceful_stop()
        else:
            self.log_message("Graceful stop API unavailable. Stopping immediately.")
            self.trading_engine.stop_trading(immediate=True)
            self._set_stopped_ui()
            return

        self.log_message(
            f"Graceful stop requested with {open_positions} open position(s). "
            "New entries are blocked until positions are closed."
        )
        self.status_var.set("STOPPING (GRACEFUL)")
        self.stop_btn.config(state="disabled")
        self._open_graceful_stop_dialog()

    def start_real_trading(self):
        """Start REAL trading using the trading engine"""
        import threading
        
        def trading_loop():
            try:
                # Start the real trading engine
                success = self.trading_engine.start_trading()
                if success:
                    self.log_message("Real trading started successfully")
                else:
                    self.log_message("Failed to start real trading")
                    self.stop_trading(graceful=False)
            except Exception as e:
                self.log_message(f"Error in real trading: {e}")
                self.stop_trading(graceful=False)
        
        # Start real trading in separate thread
        thread = threading.Thread(target=trading_loop)
        thread.daemon = True
        thread.start()
        
        # After a short delay, override the should_continue_trading method with the correct max_positions value
        self.root.after(1000, lambda: setattr(self.trading_engine, 'should_continue_trading', 
            lambda: self.check_should_continue_trading(self.max_positions_var.get(), 
                self.stop_percentage_value.get() if self.stop_at_percentage_var.get() else None)))

    def check_should_continue_trading(self, max_positions, stop_trading_at_percentage):
        """Check if trading should continue based on balance and optional settings"""
        # Check if we've reached the maximum number of open positions
        if len(self.trading_engine.current_positions) >= max_positions:
            self.log_message(f"Max positions reached: {len(self.trading_engine.current_positions)} >= {max_positions}")
            return False
        
        # Optional: Check if balance is below a percentage threshold (if enabled)
        if stop_trading_at_percentage:
            min_balance = self.trading_engine.initial_balance * (stop_trading_at_percentage / 100)
            if self.trading_engine.simulated_balance < min_balance:
                self.log_message(f"Balance below threshold: ${self.trading_engine.simulated_balance:.2f} < ${min_balance:.2f} ({stop_trading_at_percentage}%)")
                return False
        
        return True

    def log_message(self, message):
        """Add message to trading log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")

    def _is_pid_alive(self, pid):
        if not pid:
            return False
        try:
            os.kill(int(pid), 0)
            return True
        except PermissionError:
            return True
        except Exception:
            return False

    def _format_age(self, age_seconds):
        if age_seconds is None:
            return "unknown"
        if age_seconds < 60:
            return f"{int(age_seconds)}s ago"
        return f"{int(age_seconds // 60)}m ago"

    def _normalize_timeframe_input(self, value):
        text = str(value).strip().lower()
        if not text:
            return "1m"

        try:
            if text.endswith("m"):
                minutes = int(text[:-1])
            elif text.endswith("h"):
                minutes = int(text[:-1]) * 60
            elif text.endswith("d"):
                minutes = int(text[:-1]) * 1440
            else:
                minutes = int(text)
        except (TypeError, ValueError):
            return None

        if minutes <= 0:
            return None

        return f"{minutes}m"

    def _get_data_collector_state(self):
        state = {
            "collector_running": False,
            "data_age_sec": None,
            "data_files": 0,
            "symbol_count": 0,
            "symbols_with_1m": 0,
            "fresh_1m_symbols": 0,
        }
        data_dir = os.path.join(project_root, "data")
        status_path = os.path.join(data_dir, "collection_status.json")

        if os.path.exists(status_path):
            try:
                with open(status_path, "r", encoding="utf-8") as f:
                    status = json.load(f)
                running_flag = bool(status.get("running", False))
                pid = status.get("pid")
                state["collector_running"] = running_flag and self._is_pid_alive(pid)
            except Exception:
                state["collector_running"] = False

        if os.path.isdir(data_dir):
            latest_mtime = None
            symbols = set()
            for filename in os.listdir(data_dir):
                if not filename.endswith(".csv"):
                    continue
                full_path = os.path.join(data_dir, filename)
                try:
                    mtime = os.path.getmtime(full_path)
                except Exception:
                    continue
                latest_mtime = mtime if latest_mtime is None else max(latest_mtime, mtime)
                state["data_files"] += 1
                stem = filename[:-4]
                if "_" in stem:
                    symbols.add(stem.split("_", 1)[0])
                if stem.endswith("_1"):
                    state["symbols_with_1m"] += 1
                    try:
                        with open(full_path, "r", encoding="utf-8", errors="ignore") as candle_file:
                            rows = candle_file.readlines()
                        if rows:
                            last_line = rows[-1].strip()
                            if last_line and "," in last_line:
                                ts_str = last_line.split(",", 1)[0].strip()
                                ts_ms = int(float(ts_str))
                                age_sec = max(0.0, time.time() - (ts_ms / 1000.0))
                                if age_sec <= 180:
                                    state["fresh_1m_symbols"] += 1
                    except Exception:
                        pass
            state["symbol_count"] = len(symbols)
            if latest_mtime is not None:
                state["data_age_sec"] = max(0.0, time.time() - latest_mtime)

        return state

    def refresh_data_feed_status(self):
        state = self._get_data_collector_state()
        age_text = self._format_age(state["data_age_sec"])
        is_fresh = state["data_age_sec"] is not None and state["data_age_sec"] <= 180
        fresh_1m = int(state.get("fresh_1m_symbols", 0))
        total_1m = int(state.get("symbols_with_1m", 0))
        coverage = (fresh_1m / total_1m) if total_1m > 0 else 0.0
        coverage_text = f"{fresh_1m}/{total_1m} 1m fresh" if total_1m > 0 else "no 1m files"

        if state["collector_running"] and is_fresh and coverage >= 0.95:
            text = (
                f"Data feed LIVE | symbols {state['symbol_count']} | {coverage_text} | latest {age_text}"
            )
            color = "green"
        elif state["collector_running"] and coverage > 0:
            text = (
                f"Data feed PARTIAL | symbols {state['symbol_count']} | {coverage_text} | latest {age_text}"
            )
            color = "orange"
        elif state["collector_running"]:
            text = (
                f"Collector running, waiting for fresh candles | symbols {state['symbol_count']} | "
                f"{coverage_text} | latest {age_text}"
            )
            color = "orange"
        elif state["data_files"] == 0:
            text = "Collector OFF | no CSV data found"
            color = "red"
        else:
            text = (
                f"Collector OFF | stale data | symbols {state['symbol_count']} | latest {age_text}"
            )
            color = "red"

        self.data_feed_status_var.set(text)
        self.data_feed_status_label.config(foreground=color)
        self.root.after(5000, self.refresh_data_feed_status)

    def _confirm_start_with_stale_data(self):
        state = self._get_data_collector_state()
        if state["collector_running"]:
            return True

        age_text = self._format_age(state["data_age_sec"])
        message = (
            "Data collector is not running.\n\n"
            f"Symbols in CSV: {state['symbol_count']}\n"
            f"Latest candle: {age_text}\n\n"
            "Paper trader may use stale data.\n"
            "Continue anyway?"
        )
        return messagebox.askyesno("Data Collector Not Running", message)

    def update_status(self, status):
        """Update status display"""
        self.status_var.set(status)
    
    def start_trading(self):
        """Start paper trading"""
        try:
            self._close_graceful_stop_dialog()
            if not self._confirm_start_with_stale_data():
                self.log_message("Trading cancelled: data collector is not running.")
                return

            entry_timeframe = self._normalize_timeframe_input(self.entry_timeframe_var.get())
            if entry_timeframe is None:
                messagebox.showerror(
                    "Invalid Timeframe",
                    "Use timeframe like 1m, 5m, 15m, 1h, or 1d.",
                )
                return
            self.entry_timeframe_var.set(entry_timeframe)
            self.log_message(f"Strategy Entry Timeframe: {entry_timeframe}")

            # Check for optimized parameters first
            from simple_strategy.trading.parameter_manager import ParameterManager
            pm = ParameterManager()
            optimized_params = pm.get_parameters(self.strategy_name)
            
            if not optimized_params:
                # Ask user what to do
                result = messagebox.askyesno(
                    "No Optimized Parameters",
                    f"No optimized parameters found for '{self.strategy_name}'.\n\n"
                    f"Do you want to continue with default parameters?\n\n"
                    f"Yes = Use default parameters\n"
                    f"No = Cancel and optimize first"
                )
                if not result:
                    self.log_message("Trading cancelled - no optimized parameters")
                    return

            strategy_param_overrides = self._collect_strategy_param_overrides()
            if strategy_param_overrides is None:
                return
            if strategy_param_overrides:
                self.log_message("[INFO] Strategy parameter overrides:")
                for key in sorted(strategy_param_overrides.keys()):
                    self.log_message(f"  {key}: {strategy_param_overrides[key]}")

            exchange_sl_enabled = bool(self.exchange_sl_enabled_var.get())
            exchange_trailing_enabled = bool(self.exchange_trailing_enabled_var.get())
            exchange_sl_pct_input = float(self.exchange_sl_pct_var.get())
            atr_mode_enabled = bool(self.atr_mode_enabled_var.get())
            atr_period = 14
            atr_sl_mult = 1.3
            atr_tp_mult = 1.7
            if atr_mode_enabled:
                atr_period = int(float(self.atr_period_var.get()))
                atr_sl_mult = float(self.atr_sl_mult_var.get())
                atr_tp_mult = float(self.atr_tp_mult_var.get())
                if atr_period < 2:
                    messagebox.showerror("Invalid ATR", "ATR period must be at least 2.")
                    return
                if atr_sl_mult <= 0 or atr_tp_mult <= 0:
                    messagebox.showerror("Invalid ATR", "ATR multipliers must be greater than 0.")
                    return
            if not atr_mode_enabled and exchange_sl_enabled and exchange_sl_pct_input <= 0:
                messagebox.showerror("Invalid Stop Loss", "Stop-loss % must be greater than 0.")
                return
            if not atr_mode_enabled and exchange_trailing_enabled and not exchange_sl_enabled:
                messagebox.showerror("Invalid Stop Mode", "Enable Exchange Stop Loss before trailing mode.")
                return
            exchange_sl_pct = max(0.0001, exchange_sl_pct_input / 100.0)
            risk_exit_mode = (
                'atr' if atr_mode_enabled
                else 'trailing' if (exchange_sl_enabled and exchange_trailing_enabled)
                else 'fixed' if exchange_sl_enabled
                else 'none'
            )
            effective_exchange_sl = risk_exit_mode in {'fixed', 'trailing', 'atr'}
            effective_exchange_trailing = risk_exit_mode == 'trailing'
            runtime_strategy_params = {"entry_timeframe": entry_timeframe}
            runtime_strategy_params.update(strategy_param_overrides or {})
            self.log_message(
                f"Risk Exit Mode: {risk_exit_mode.upper()} "
                f"(SL {exchange_sl_pct_input:.2f}%, ATR {atr_period}, SLx{atr_sl_mult:.2f}, TPx{atr_tp_mult:.2f})"
            )

            # Import and create trading engine
            from simple_strategy.trading.paper_trading_engine import PaperTradingEngine
            self.trading_engine = PaperTradingEngine(
                self.api_account,
                self.strategy_name,
                self.simulated_balance,
                log_callback=self.log_message,
                status_callback=self.update_status,
                performance_callback=self.update_performance,
                enable_exchange_stop_loss=effective_exchange_sl,
                enable_exchange_trailing_stop=effective_exchange_trailing,
                exchange_stop_loss_pct=exchange_sl_pct,
                risk_exit_mode=risk_exit_mode,
                risk_sl_pct=exchange_sl_pct,
                risk_atr_period=atr_period,
                risk_atr_sl_multiplier=atr_sl_mult,
                risk_atr_tp_multiplier=atr_tp_mult,
                strategy_params_override=runtime_strategy_params,
            )
            
            # Initialize shared data access after engine creation
            self.trading_engine.initialize_shared_data_access()

            # Start performance update timer
            self.update_performance_timer()
            
            # Start trading in a separate thread (simplified for now)
            self.log_message("Starting paper trading...")
            self.status_var.set("RUNNING")
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            
            # Start REAL trading
            self.start_real_trading()
            
        except KeyboardInterrupt:
            self.log_message("Trading start interrupted while loading engine.")
            messagebox.showerror(
                "Startup Interrupted",
                "Trading startup was interrupted while loading dependencies.\nPlease click START TRADING again.",
            )
        except Exception as e:
            self.log_message(f"Error starting trading: {e}")
            messagebox.showerror("Error", f"Failed to start trading: {e}")
    
    def update_performance(self, performance_data=None):
        """Update performance display"""
        try:
            # If no data provided, try to get it from the engine
            if not performance_data and hasattr(self, 'trading_engine') and self.trading_engine:
                performance_data = self.trading_engine.get_performance_summary()
            
            # Update header performance display
            if performance_data:
                self.update_performance_header(performance_data)
                
                # Log performance to the trading log instead of a separate text widget
                initial_balance = performance_data.get('initial_balance', self.simulated_balance)
                current_balance = performance_data.get('current_balance', self.simulated_balance)
                open_positions = performance_data.get('open_positions', 0)
                closed_trades = performance_data.get('closed_trades', 0)
                total_trades = performance_data.get('total_trades', 0)
                win_rate = performance_data.get('win_rate', 0.0)
                pnl = performance_data.get('pnl', 0.0)
                realized_pnl = performance_data.get('realized_pnl', 0.0)
                unrealized_pnl = performance_data.get('unrealized_pnl', 0.0)
                
                # Show both realized and unrealized P&L if available
                pnl_text = f"Realized: ${realized_pnl:.2f}, Unrealized: ${unrealized_pnl:.2f}"

                    
                # Log to the trading log instead of a separate widget
                self.log_message(f"Performance Update:")
                self.log_message(f"  Initial Balance: ${initial_balance:.2f}")
                self.log_message(f"  Current Balance: ${current_balance:.2f}")
                self.log_message(f"  Open Positions: {open_positions}")
                self.log_message(f"  Closed Trades: {closed_trades}")
                self.log_message(f"  Total Trades: {total_trades}")
                self.log_message(f"  Win Rate: {win_rate:.1f}%")
                self.log_message(f"  Profit/Loss: {pnl_text}")
            else:
                # Fallback to initial values
                self.update_performance_header({
                    'balance': self.simulated_balance,
                    'open_positions': 0,
                    'pnl': 0.0,
                    'win_rate': 0.0
                })
                self.log_message(f"Performance: Initial Balance: ${self.simulated_balance:.2f}")
            
        except Exception as e:
            print(f"Error updating performance: {e}")
            self.log_message(f"Error updating performance: {str(e)}")
    
    def run(self):
        """Run the paper trading window"""
        self.root.mainloop()

    def get_real_time_performance(self):
        """Get real-time performance data from trading engine"""
        try:
            if hasattr(self, 'trading_engine') and self.trading_engine:
                # Get simulated balance directly from engine
                simulated_balance = self.trading_engine.simulated_balance
                
                # Calculate P&L
                initial_capital = self.simulated_balance
                total_pnl = simulated_balance - initial_capital
                
                # Get open positions count
                open_positions = len(self.trading_engine.current_positions) if hasattr(self.trading_engine, 'current_positions') else 0
                
                # Completed trades are CLOSE_* events in the current signal schema.
                completed_trades = (
                    sum(
                        1
                        for trade in self.trading_engine.trades
                        if trade.get('type') in ('CLOSE_LONG', 'CLOSE_SHORT')
                    )
                    if hasattr(self.trading_engine, 'trades')
                    else 0
                )

                # Calculate win rate from completed close trades.
                winning_trades = (
                    sum(
                        1
                        for trade in self.trading_engine.trades
                        if trade.get('type') in ('CLOSE_LONG', 'CLOSE_SHORT')
                        and trade.get('pnl', 0) > 0
                    )
                    if hasattr(self.trading_engine, 'trades')
                    else 0
                )
                win_rate = (winning_trades / completed_trades * 100) if completed_trades > 0 else 0
                
                return {
                    'initial_balance': initial_capital,
                    'current_balance': simulated_balance,
                    'open_positions': open_positions,
                    'closed_trades': completed_trades,
                    'total_trades': len(self.trading_engine.trades) if hasattr(self.trading_engine, 'trades') else 0,
                    'win_rate': win_rate,
                    'total_pnl': total_pnl
                }
            else:
                return None
        except Exception as e:
            print(f"Error getting real-time performance: {e}")
            return None

    def update_performance_timer(self):
        """Update performance display every 5 seconds"""
        try:
            # Try to get real-time performance data
            real_time_data = self.get_real_time_performance()
            
            if real_time_data:
                # Use real-time data
                initial_balance = real_time_data['initial_balance']
                current_balance = real_time_data['current_balance']
                open_positions = real_time_data['open_positions']
                closed_trades = real_time_data['closed_trades']
                total_trades = real_time_data['total_trades']
                win_rate = real_time_data['win_rate']
                pnl = real_time_data['total_pnl']
                
                perf_text = f"""Initial Balance: ${initial_balance:.2f}
    Current Balance: ${current_balance:.2f}
    Open Positions: {open_positions}
    Closed Trades: {closed_trades}
    Total Trades: {total_trades}
    Win Rate: {win_rate:.1f}%
    Profit/Loss: ${pnl:.2f}"""
            else:
                # Fallback to dummy data if engine not available
                perf_text = f"""Initial Balance: ${self.simulated_balance:.2f}
    Current Balance: ${self.simulated_balance:.2f}
    Open Positions: 0
    Closed Trades: 0
    Total Trades: 0
    Win Rate: 0.0%
    Profit/Loss: $0.00"""
            
            if hasattr(self, "perf_text"):
                self.perf_text.delete(1.0, "end")
                self.perf_text.insert(1.0, perf_text)
            else:
                self.log_message("Performance timer updated (no perf_text widget).")

            
            # Schedule next update
            self.root.after(5000, self.update_performance_timer)
        except Exception as e:
            print(f"Error in performance timer: {e}")
            # Schedule next update even if there's an error
            self.root.after(5000, self.update_performance_timer)

            
if __name__ == "__main__":
    try:
        # Get parameters from command line or use defaults
        import sys
        if len(sys.argv) >= 4:
            launcher = PaperTradingLauncher(sys.argv[1], sys.argv[2], sys.argv[3])
        else:
            launcher = PaperTradingLauncher()
        launcher.run()
    except BaseException as e:
        crash_log_path = write_launcher_crash_log(e)
        print(f"Paper trader launcher failed: {e}")
        print(f"Crash log written to: {crash_log_path}")
        try:
            messagebox.showerror(
                "Paper Trader Crash",
                f"Paper trader failed to start.\n\nError: {e}\n\nCrash log:\n{crash_log_path}",
            )
        except Exception:
            pass
