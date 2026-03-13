# root/main.py - Dashboard GUI
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
import os
import json
import time
import signal
from datetime import datetime
from simple_strategy.trading.parameter_gui import ParameterGUI

class ToolTip:
    """Small hover tooltip for Tk widgets."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event=None):
        if self.tip_window or not self.text:
            return
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#fff8dc",
            relief="solid",
            borderwidth=1,
            font=("Arial", 8),
            wraplength=260,
        )
        label.pack(ipadx=6, ipady=3)

    def _hide(self, _event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None

class TradingBotDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Trading Bot Control Center")
        self.root.geometry("600x500")
        self.auto_evolve_process = None
        self.ae_status_refresh_ms = 5000
        self.ae_status_job = None
        self.data_collection_process = None
        self.data_collection_startup_log_path = None
        self.paper_data_status_job = None
        self.create_widgets()
    
    def create_widgets(self):
        # Make the main dashboard vertically scrollable so all sections/buttons remain reachable.
        self.main_canvas = tk.Canvas(self.root, highlightthickness=0)
        self.main_scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.main_canvas.yview)
        self.main_canvas.configure(yscrollcommand=self.main_scrollbar.set)

        self.main_scrollbar.pack(side="right", fill="y")
        self.main_canvas.pack(side="left", fill="both", expand=True)

        self.main_content_frame = ttk.Frame(self.main_canvas)
        self.main_canvas_window = self.main_canvas.create_window((0, 0), window=self.main_content_frame, anchor="nw")

        self.main_content_frame.bind("<Configure>", self._on_main_content_configure)
        self.main_canvas.bind("<Configure>", self._on_main_canvas_configure)
        self._bind_mousewheel(self.main_canvas)
        self._bind_mousewheel(self.main_content_frame)

        # Data Collection Section
        self.create_data_collection_section()
        # Simple Strategy Section (NEW - FUNCTIONAL)
        self.create_simple_strategy_section()
        # Auto evolution section (DEAP + Bayesian)
        self.create_auto_evolve_section()
        # Placeholder section for SL AI and active section for RL AI
        self.create_placeholder_section("🤖 SL AI MODULE", "sl_ai")
        self.create_rl_ai_section()
        # Bottom buttons
        self.create_bottom_buttons()

    def _add_tooltip(self, widget, text):
        ToolTip(widget, text)

    def _on_main_content_configure(self, _event):
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))

    def _on_main_canvas_configure(self, event):
        self.main_canvas.itemconfigure(self.main_canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        if event.num == 4:
            self.main_canvas.yview_scroll(-1, "units")
            return
        if event.num == 5:
            self.main_canvas.yview_scroll(1, "units")
            return

        delta = int(-1 * (event.delta / 120)) if event.delta else 0
        if delta != 0:
            self.main_canvas.yview_scroll(delta, "units")

    def _bind_mousewheel(self, widget):
        widget.bind("<MouseWheel>", self._on_mousewheel)
        widget.bind("<Button-4>", self._on_mousewheel)
        widget.bind("<Button-5>", self._on_mousewheel)
    
    def create_data_collection_section(self):
        # Data Collection Frame
        dc_frame = ttk.LabelFrame(self.main_content_frame, text="📊 DATA COLLECTION MODULE", padding=10)
        dc_frame.pack(fill="x", padx=10, pady=5)
        
        # Status
        self.dc_status = tk.StringVar(value="🔴 STOPPED")
        status_label = ttk.Label(dc_frame, textvariable=self.dc_status, font=("Arial", 10, "bold"))
        status_label.pack(anchor="w")
        
        # Buttons
        button_frame = ttk.Frame(dc_frame)
        button_frame.pack(fill="x", pady=5)
        
        self.dc_start_btn = ttk.Button(button_frame, text="START DATA COLLECTION",
                                    command=self.start_data_collection)
        self.dc_start_btn.pack(side="left", padx=5)
        
        self.dc_stop_btn = ttk.Button(button_frame, text="STOP DATA COLLECTION",
                                    command=self.stop_data_collection, state="disabled")
        self.dc_stop_btn.pack(side="left", padx=5)
        
        ttk.Button(button_frame, text="SETTINGS",
                  command=self.open_data_collection_settings).pack(side="left", padx=5)
    
    def create_simple_strategy_section(self):
        # Simple Strategy Frame with Tabs (NEW - PHASE 3)
        ss_frame = ttk.LabelFrame(self.main_content_frame, text="📈 SIMPLE STRATEGY MODULE", padding=10)
        ss_frame.pack(fill="x", padx=10, pady=5)
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(ss_frame)
        self.notebook.pack(fill="x", expand=False)
        
        # Create three tabs
        self.create_backtesting_tab()
        self.create_paper_trading_tab()
        self.create_live_trading_tab()
        
        # Status label (outside tabs)
        self.ss_status = tk.StringVar(value="🔴 STOPPED")
        status_label = ttk.Label(ss_frame, textvariable=self.ss_status, font=("Arial", 10, "bold"))
        status_label.pack(anchor="w", pady=(5, 0))

    def create_auto_evolve_section(self):
        # Auto Evolution Frame
        ae_frame = ttk.LabelFrame(self.main_content_frame, text="🧬 AUTO EVOLVE MODULE", padding=10)
        ae_frame.pack(fill="x", padx=10, pady=5)

        self.ae_status = tk.StringVar(value="⚫ IDLE")
        ttk.Label(ae_frame, textvariable=self.ae_status, font=("Arial", 10, "bold")).pack(anchor="w")
        self.ae_run_var = tk.StringVar(value="Run: -")
        self.ae_progress_var = tk.StringVar(value="Progress: 0%")
        self.ae_eval_var = tk.StringVar(value="Evaluated: 0/0")
        self.ae_eta_var = tk.StringVar(value="ETA: -")
        ttk.Label(ae_frame, textvariable=self.ae_run_var, font=("Arial", 9)).pack(anchor="w")
        ttk.Label(ae_frame, textvariable=self.ae_progress_var, font=("Arial", 9)).pack(anchor="w")
        ttk.Label(ae_frame, textvariable=self.ae_eval_var, font=("Arial", 9)).pack(anchor="w")
        ttk.Label(ae_frame, textvariable=self.ae_eta_var, font=("Arial", 9)).pack(anchor="w")
        self.ae_progressbar = ttk.Progressbar(ae_frame, orient="horizontal", mode="determinate", maximum=100)
        self.ae_progressbar.pack(fill="x", pady=(3, 6))

        top_button_row = ttk.Frame(ae_frame)
        top_button_row.pack(fill="x", pady=5)
        full_btn = ttk.Button(top_button_row, text="RUN FULL SEARCH",
                              command=self.start_auto_evolve_full)
        full_btn.pack(side="left", padx=5)
        overnight_btn = ttk.Button(top_button_row, text="RUN OVERNIGHT",
                                   command=self.start_auto_evolve_overnight)
        overnight_btn.pack(side="left", padx=5)
        multiday_btn = ttk.Button(top_button_row, text="RUN MULTI-DAY",
                                  command=self.start_auto_evolve_multiday)
        multiday_btn.pack(side="left", padx=5)

        profile_row = ttk.Frame(ae_frame)
        profile_row.pack(fill="x", pady=2)
        smoke_btn = ttk.Button(profile_row, text="RUN SMOKE TEST",
                               command=self.start_auto_evolve_smoke)
        smoke_btn.pack(side="left", padx=5)
        refresh_btn = ttk.Button(profile_row, text="REFRESH STATUS",
                                 command=self.refresh_auto_evolve_status)
        refresh_btn.pack(side="left", padx=5)
        seed_label = ttk.Label(profile_row, text="SEED (blank = config):")
        seed_label.pack(side="left", padx=(12, 5))
        self.ae_seed_var = tk.StringVar(value="")
        seed_entry = ttk.Entry(profile_row, textvariable=self.ae_seed_var, width=10)
        seed_entry.pack(side="left")

        help_text = (
            "All normal runs now search fast on 2 symbols first, then re-check only the better "
            "candidates on more symbols with heavier tuning."
        )
        ttk.Label(
            ae_frame,
            text=help_text,
            foreground="#666666",
            wraplength=560,
            justify="left",
        ).pack(anchor="w", pady=(2, 4))

        bottom_button_row = ttk.Frame(ae_frame)
        bottom_button_row.pack(fill="x", pady=2)
        ttk.Button(bottom_button_row, text="RESUME LAST RUN",
                   command=self.resume_auto_evolve_last).pack(side="left", padx=5)
        ttk.Button(bottom_button_row, text="OPEN OUTPUT FOLDER",
                   command=self.open_auto_evolve_output).pack(side="left", padx=5)
        ttk.Button(bottom_button_row, text="OPEN FAST LEGO FINDER",
                   command=self.open_fast_strategy_finder).pack(side="left", padx=5)
        ttk.Button(bottom_button_row, text="OPEN UNIQUE ADVANCED ENGINE",
                   command=self.open_unique_advanced_engine).pack(side="left", padx=5)
        ttk.Button(bottom_button_row, text="DELETE ALL RUNS",
                   command=self.clear_auto_evolve_runs).pack(side="left", padx=5)
        self._add_tooltip(smoke_btn, "Quick wiring check only. Does not do a deep search.")
        self._add_tooltip(overnight_btn, "Balanced run. Fast 2-symbol search first, then the better results are checked on 6 symbols.")
        self._add_tooltip(full_btn, "Stronger run. Fast 2-symbol search first, then shortlisted results are checked harder on 6 symbols.")
        self._add_tooltip(multiday_btn, "Deepest run. Fast 2-symbol search first, then shortlisted results are checked on 8 symbols with heavier tuning.")
        self._add_tooltip(refresh_btn, "Reload the latest run status without starting a new run.")
        self._add_tooltip(seed_label, "Use a number to repeat the same search path. Leave blank to use the config value.")
        self._add_tooltip(seed_entry, "Same seed = same random starting point.")
        self.refresh_auto_evolve_status()
        self._schedule_auto_evolve_status_refresh()

    def create_backtesting_tab(self):
        # Backtesting Tab
        backtesting_frame = ttk.Frame(self.notebook)
        self.notebook.add(backtesting_frame, text="🧪 BACKTESTING")
        
        # Buttons for backtesting
        button_frame = ttk.Frame(backtesting_frame)
        button_frame.pack(fill="x", pady=10)
        
        ttk.Button(button_frame, text="OPEN BACKTESTER", 
                command=self.start_simple_strategy).pack(side="left", padx=5)
        ttk.Button(button_frame, text="SETTINGS", 
                command=self.open_simple_strategy_settings).pack(side="left", padx=5)
        ttk.Button(button_frame, text="PARAMETER MANAGER", 
                command=self.open_parameter_manager).pack(side="left", padx=5)
        ttk.Button(button_frame, text="API MANAGER", 
               command=self.open_api_manager).pack(side="left", padx=5)
        
        # Info label
        info_label = ttk.Label(backtesting_frame, 
                            text="Open multiple backtest windows to test different strategies simultaneously",
                            font=("Arial", 9), foreground="gray")
        info_label.pack(anchor="w", pady=(5, 0))

    def create_paper_trading_tab(self):
        # Paper Trading Tab
        paper_frame = ttk.Frame(self.notebook)
        self.notebook.add(paper_frame, text="📄 PAPER TRADING")
        
        # Account selection
        account_frame = ttk.Frame(paper_frame)
        account_frame.pack(fill="x", pady=10)
        
        ttk.Label(account_frame, text="Select Demo Account:").pack(side="left", padx=5)
        self.paper_account_var = tk.StringVar()
        self.paper_account_combo = ttk.Combobox(account_frame, textvariable=self.paper_account_var, width=20)
        self.paper_account_combo.pack(side="left", padx=5)
        
        # Strategy selection
        strategy_frame = ttk.Frame(paper_frame)
        strategy_frame.pack(fill="x", pady=5)
        
        ttk.Label(strategy_frame, text="Select Strategy:").pack(side="left", padx=5)
        self.paper_strategy_var = tk.StringVar()
        self.paper_strategy_combo = ttk.Combobox(strategy_frame, textvariable=self.paper_strategy_var, width=20)
        self.paper_strategy_combo.pack(side="left", padx=5)
        
        # Balance simulation
        balance_frame = ttk.Frame(paper_frame)
        balance_frame.pack(fill="x", pady=5)
        
        ttk.Label(balance_frame, text="Simulated Balance:").pack(side="left", padx=5)
        self.paper_balance_var = tk.StringVar(value="1000")
        ttk.Entry(balance_frame, textvariable=self.paper_balance_var, width=10).pack(side="left", padx=5)
        ttk.Label(balance_frame, text="$").pack(side="left")
        
        # NEW: Bybit balance section - BUTTON FIRST, THEN LABEL
        bybit_balance_frame = ttk.Frame(paper_frame)
        bybit_balance_frame.pack(fill="x", pady=5)
        
        ttk.Label(bybit_balance_frame, text="Current Bybit Balance:").pack(side="left", padx=5)
        
        # Button first
        self.get_balance_btn = ttk.Button(bybit_balance_frame, text="Get Balance", 
                                        command=self.get_current_bybit_balance)
        self.get_balance_btn.pack(side="left", padx=5)
        
        # Then the label
        self.bybit_balance_var = tk.StringVar(value="Click to fetch")
        self.bybit_balance_label = ttk.Label(bybit_balance_frame, textvariable=self.bybit_balance_var, 
                                            font=("Arial", 10, "bold"), foreground="blue")
        self.bybit_balance_label.pack(side="left", padx=5)

        # Data feed status (collector + CSV freshness)
        data_status_frame = ttk.Frame(paper_frame)
        data_status_frame.pack(fill="x", pady=5)
        ttk.Label(data_status_frame, text="Data Feed Status:").pack(side="left", padx=5)
        self.paper_data_status_var = tk.StringVar(value="Checking...")
        self.paper_data_status_label = ttk.Label(
            data_status_frame,
            textvariable=self.paper_data_status_var,
            font=("Arial", 9, "bold"),
            foreground="gray",
        )
        self.paper_data_status_label.pack(side="left", padx=5)

        global_rules_frame = ttk.LabelFrame(paper_frame, text="Global Rules", padding=6)
        global_rules_frame.pack(fill="x", pady=5)
        ttk.Label(global_rules_frame, text="Enable:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.paper_enable_global_rules_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(global_rules_frame, variable=self.paper_enable_global_rules_var).grid(
            row=0, column=1, sticky="w", padx=5, pady=2
        )
        ttk.Label(global_rules_frame, text="Profile:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.paper_global_rules_profile_var = tk.StringVar(value="balanced")
        ttk.Combobox(
            global_rules_frame,
            textvariable=self.paper_global_rules_profile_var,
            values=["safe", "balanced", "aggressive"],
            state="readonly",
            width=12,
        ).grid(row=1, column=1, sticky="w", padx=5, pady=2)
        ttk.Label(global_rules_frame, text="Min 24h Notional (USDT):").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.paper_min_24h_notional_var = tk.StringVar(value="50000000")
        ttk.Entry(global_rules_frame, textvariable=self.paper_min_24h_notional_var, width=14).grid(
            row=2, column=1, sticky="w", padx=5, pady=2
        )
        exchange_sl_label = ttk.Label(global_rules_frame, text="Use Fixed / Trailing Stop:")
        exchange_sl_label.grid(row=3, column=0, sticky="w", padx=5, pady=2)
        self.paper_enable_exchange_sl_var = tk.BooleanVar(value=False)
        paper_exchange_sl_check = ttk.Checkbutton(
            global_rules_frame,
            variable=self.paper_enable_exchange_sl_var,
            command=self._toggle_paper_exchange_sl_controls,
        )
        paper_exchange_sl_check.grid(
            row=3, column=1, sticky="w", padx=5, pady=2
        )
        trailing_label = ttk.Label(global_rules_frame, text="Trail the Stop:")
        trailing_label.grid(row=4, column=0, sticky="w", padx=5, pady=2)
        self.paper_enable_exchange_trailing_var = tk.BooleanVar(value=False)
        self.paper_exchange_trailing_check = ttk.Checkbutton(
            global_rules_frame,
            variable=self.paper_enable_exchange_trailing_var,
            command=self._update_paper_exchange_sl_mode_label,
        )
        self.paper_exchange_trailing_check.grid(row=4, column=1, sticky="w", padx=5, pady=2)
        paper_sl_pct_label = ttk.Label(global_rules_frame, text="Stop Distance (%):")
        paper_sl_pct_label.grid(row=5, column=0, sticky="w", padx=5, pady=2)
        self.paper_exchange_sl_pct_var = tk.StringVar(value="2.0")
        self.paper_exchange_sl_pct_entry = ttk.Entry(
            global_rules_frame, textvariable=self.paper_exchange_sl_pct_var, width=14
        )
        self.paper_exchange_sl_pct_entry.grid(row=5, column=1, sticky="w", padx=5, pady=2)
        self.paper_exchange_sl_mode_var = tk.StringVar(value="No stop loss")
        self.paper_exchange_sl_mode_label = ttk.Label(
            global_rules_frame,
            textvariable=self.paper_exchange_sl_mode_var,
            font=("Arial", 9, "bold"),
            foreground="gray",
        )
        self.paper_exchange_sl_mode_label.grid(row=6, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        atr_mode_label = ttk.Label(global_rules_frame, text="Use ATR Stop + Take Profit:")
        atr_mode_label.grid(row=7, column=0, sticky="w", padx=5, pady=2)
        self.paper_enable_atr_exit_var = tk.BooleanVar(value=False)
        self.paper_enable_atr_exit_check = ttk.Checkbutton(
            global_rules_frame,
            variable=self.paper_enable_atr_exit_var,
            command=self._toggle_paper_atr_controls,
        )
        self.paper_enable_atr_exit_check.grid(row=7, column=1, sticky="w", padx=5, pady=2)
        paper_atr_period_label = ttk.Label(global_rules_frame, text="ATR Period:")
        paper_atr_period_label.grid(row=8, column=0, sticky="w", padx=5, pady=2)
        self.paper_atr_period_var = tk.StringVar(value="14")
        self.paper_atr_period_entry = ttk.Entry(global_rules_frame, textvariable=self.paper_atr_period_var, width=14)
        self.paper_atr_period_entry.grid(row=8, column=1, sticky="w", padx=5, pady=2)
        paper_atr_sl_label = ttk.Label(global_rules_frame, text="ATR Stop Multiplier:")
        paper_atr_sl_label.grid(row=9, column=0, sticky="w", padx=5, pady=2)
        self.paper_atr_sl_mult_var = tk.StringVar(value="1.3")
        self.paper_atr_sl_mult_entry = ttk.Entry(global_rules_frame, textvariable=self.paper_atr_sl_mult_var, width=14)
        self.paper_atr_sl_mult_entry.grid(row=9, column=1, sticky="w", padx=5, pady=2)
        paper_atr_tp_label = ttk.Label(global_rules_frame, text="ATR Take Profit Multiplier:")
        paper_atr_tp_label.grid(row=10, column=0, sticky="w", padx=5, pady=2)
        self.paper_atr_tp_mult_var = tk.StringVar(value="1.7")
        self.paper_atr_tp_mult_entry = ttk.Entry(global_rules_frame, textvariable=self.paper_atr_tp_mult_var, width=14)
        self.paper_atr_tp_mult_entry.grid(row=10, column=1, sticky="w", padx=5, pady=2)
        self.paper_exit_help_label = ttk.Label(
            global_rules_frame,
            text="Choose one style: fixed/trailing stop, or ATR stop + ATR take profit.",
            font=("Arial", 8),
            foreground="gray",
        )
        self.paper_exit_help_label.grid(row=11, column=0, columnspan=2, sticky="w", padx=5, pady=(2, 4))
        self.paper_exchange_sl_pct_var.trace_add("write", lambda *_: self._update_paper_exchange_sl_mode_label())
        self.paper_atr_period_var.trace_add("write", lambda *_: self._update_paper_exchange_sl_mode_label())
        self.paper_atr_sl_mult_var.trace_add("write", lambda *_: self._update_paper_exchange_sl_mode_label())
        self.paper_atr_tp_mult_var.trace_add("write", lambda *_: self._update_paper_exchange_sl_mode_label())
        self._toggle_paper_atr_controls()
        self._toggle_paper_exchange_sl_controls()
        self._add_tooltip(exchange_sl_label, "Turns on a normal percent-based stop. Use this for fixed or trailing stops.")
        self._add_tooltip(paper_exchange_sl_check, "Enable a fixed stop or trailing stop using the percent below.")
        self._add_tooltip(trailing_label, "If enabled, the stop follows price instead of staying fixed.")
        self._add_tooltip(self.paper_exchange_trailing_check, "Trailing stop uses the same percent distance as the stop field below.")
        self._add_tooltip(paper_sl_pct_label, "Percent distance for fixed stop or trailing stop. Example: 2 means 2%.")
        self._add_tooltip(self.paper_exchange_sl_pct_entry, "Used only for fixed stop or trailing stop.")
        self._add_tooltip(atr_mode_label, "Uses ATR to place a volatility-based stop and take-profit.")
        self._add_tooltip(self.paper_enable_atr_exit_check, "ATR mode overrides the fixed/trailing stop settings above.")
        self._add_tooltip(paper_atr_period_label, "How many candles ATR uses. 14 is a common default.")
        self._add_tooltip(self.paper_atr_period_entry, "Higher values are smoother. Lower values react faster.")
        self._add_tooltip(paper_atr_sl_label, "Stop distance in ATR units. Example: 1.3 means 1.3 x ATR.")
        self._add_tooltip(self.paper_atr_sl_mult_entry, "Lower = tighter stop. Higher = wider stop.")
        self._add_tooltip(paper_atr_tp_label, "Profit target in ATR units. Example: 1.7 means 1.7 x ATR.")
        self._add_tooltip(self.paper_atr_tp_mult_entry, "Higher = farther take-profit target.")
        
        # Start button (changed to OPEN like backtesting)
        button_frame = ttk.Frame(paper_frame)
        button_frame.pack(fill="x", pady=10)
        ttk.Button(button_frame, text="OPEN PAPER TRADER",
                command=self.open_paper_trader).pack(side="left", padx=5)
        ttk.Button(button_frame, text="START TRADING",
                command=self.start_paper_trading).pack(side="left", padx=5)
        ttk.Button(button_frame, text="STOP TRADING",
                command=self.stop_paper_trading).pack(side="left", padx=5)
        
        # Load accounts and strategies when tab is created
        self.load_paper_trading_options()

        # Initialize trading engine
        self.paper_trading_engine = None
        self.trading_running = False
        self._refresh_paper_data_status()

    def get_current_bybit_balance(self):
        """Fetch and display the current Bybit balance using the existing PaperTradingEngine"""
        try:
            # Disable button during fetch to prevent multiple clicks
            self.get_balance_btn.config(state="disabled")
            self.bybit_balance_var.set("Fetching...")
            
            account = self.paper_account_var.get()
            if not account:
                messagebox.showerror("Error", "Please select an account first!")
                self.get_balance_btn.config(state="normal")
                return
            
            # If paper trading is already running, use the existing engine
            if self.trading_running and self.paper_trading_engine:
                balance_info = self.paper_trading_engine.get_real_balance()
            else:
                # Create a temporary engine just to get the balance
                from simple_strategy.trading.paper_trading_engine import PaperTradingEngine
                temp_engine = PaperTradingEngine(account, "dummy", 0)
                balance_info = temp_engine.get_real_balance()
            
            # Extract balance values
            available_balance = balance_info['available_balance']
            margin_balance = balance_info['margin_balance']
            
            # Update the label with balance information
            self.bybit_balance_var.set(f"Available: ${available_balance:.2f} | Margin: ${margin_balance:.2f}")
            
        except Exception as e:
            self.bybit_balance_var.set("Error fetching balance")
            messagebox.showerror("Error", f"Failed to fetch balance: {str(e)}")
        finally:
            # Re-enable button
            self.get_balance_btn.config(state="normal")

    def _is_pid_alive(self, pid):
        """Return True if a process id is alive."""
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

    def _get_data_collector_state(self):
        """Read collector status and CSV freshness."""
        state = {
            "collector_running": False,
            "data_age_sec": None,
            "data_files": 0,
            "symbol_count": 0,
            "symbols_with_1m": 0,
            "fresh_1m_symbols": 0,
        }
        project_root = os.path.dirname(__file__)
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

    def _refresh_paper_data_status(self):
        """Refresh paper-trader data readiness visual."""
        if not hasattr(self, "paper_data_status_var"):
            return

        state = self._get_data_collector_state()
        age_text = self._format_age(state["data_age_sec"])
        is_fresh = state["data_age_sec"] is not None and state["data_age_sec"] <= 180
        fresh_1m = int(state.get("fresh_1m_symbols", 0))
        total_1m = int(state.get("symbols_with_1m", 0))
        coverage = (fresh_1m / total_1m) if total_1m > 0 else 0.0
        coverage_text = f"{fresh_1m}/{total_1m} 1m fresh" if total_1m > 0 else "no 1m files"

        if state["collector_running"] and is_fresh and coverage >= 0.95:
            text = (
                f"LIVE: collector running | symbols {state['symbol_count']} | "
                f"latest {age_text} | {coverage_text}"
            )
            color = "green"
        elif state["collector_running"] and coverage > 0:
            text = (
                f"PARTIAL LIVE: collector running | symbols {state['symbol_count']} | "
                f"{coverage_text} | latest {age_text}"
            )
            color = "orange"
        elif state["collector_running"]:
            text = (
                f"Collector running, waiting for fresh candles | symbols {state['symbol_count']} | "
                f"latest {age_text} | {coverage_text}"
            )
            color = "orange"
        elif state["data_files"] == 0:
            text = "Collector OFF: no CSV data found"
            color = "red"
        else:
            text = (
                f"Collector OFF: stale data | symbols {state['symbol_count']} | "
                f"latest {age_text}"
            )
            color = "red"

        self.paper_data_status_var.set(text)
        self.paper_data_status_label.config(foreground=color)
        if self.paper_data_status_job is not None:
            try:
                self.root.after_cancel(self.paper_data_status_job)
            except Exception:
                pass
        self.paper_data_status_job = self.root.after(5000, self._refresh_paper_data_status)

    def _confirm_paper_trading_with_stale_data(self):
        """Warn before starting paper trading if collector is not running."""
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

    def _toggle_paper_exchange_sl_controls(self):
        """Enable/disable trailing toggle and % input from the Exchange SL toggle."""
        enabled = bool(self.paper_enable_exchange_sl_var.get())
        trailing_state = "normal" if enabled else "disabled"
        pct_state = "normal" if enabled else "disabled"
        self.paper_exchange_trailing_check.config(state=trailing_state)
        self.paper_exchange_sl_pct_entry.config(state=pct_state)
        if not enabled:
            self.paper_enable_exchange_trailing_var.set(False)
        self._update_paper_exchange_sl_mode_label()

    def _toggle_paper_atr_controls(self):
        enabled = bool(self.paper_enable_atr_exit_var.get())
        state = "normal" if enabled else "disabled"
        self.paper_atr_period_entry.config(state=state)
        self.paper_atr_sl_mult_entry.config(state=state)
        self.paper_atr_tp_mult_entry.config(state=state)
        self._update_paper_exchange_sl_mode_label()

    def _update_paper_exchange_sl_mode_label(self):
        """Show a clear stop/exit summary."""
        use_atr = bool(self.paper_enable_atr_exit_var.get()) if hasattr(self, "paper_enable_atr_exit_var") else False
        enabled = bool(self.paper_enable_exchange_sl_var.get())
        trailing = bool(self.paper_enable_exchange_trailing_var.get()) and enabled
        try:
            pct_value = float(self.paper_exchange_sl_pct_var.get())
            pct_text = f"{pct_value:.2f}%"
        except Exception:
            pct_text = "invalid %"
        try:
            atr_period = int(float(self.paper_atr_period_var.get()))
        except Exception:
            atr_period = 0
        try:
            atr_sl = float(self.paper_atr_sl_mult_var.get())
        except Exception:
            atr_sl = 0.0
        try:
            atr_tp = float(self.paper_atr_tp_mult_var.get())
        except Exception:
            atr_tp = 0.0

        if use_atr:
            text = f"ATR exit active: stop {atr_sl:.2f} x ATR, target {atr_tp:.2f} x ATR, ATR period {atr_period}"
            color = "purple"
        elif not enabled:
            text = "No protective exit configured"
            color = "gray"
        elif trailing:
            text = f"Trailing stop active: {pct_text} distance"
            color = "green"
        else:
            text = f"Fixed stop active: {pct_text} distance"
            color = "blue"

        self.paper_exchange_sl_mode_var.set(text)
        self.paper_exchange_sl_mode_label.config(foreground=color)

    def create_live_trading_tab(self):
        # Live Trading Tab
        live_frame = ttk.Frame(self.notebook)
        self.notebook.add(live_frame, text="💰 LIVE TRADING")
        
        # Account selection
        account_frame = ttk.Frame(live_frame)
        account_frame.pack(fill="x", pady=10)
        
        ttk.Label(account_frame, text="Select Live Account:").pack(side="left", padx=5)
        self.live_account_var = tk.StringVar()
        self.live_account_combo = ttk.Combobox(account_frame, textvariable=self.live_account_var, width=20)
        self.live_account_combo.pack(side="left", padx=5)
        
        # Strategy selection
        strategy_frame = ttk.Frame(live_frame)
        strategy_frame.pack(fill="x", pady=5)
        
        ttk.Label(strategy_frame, text="Select Strategy:").pack(side="left", padx=5)
        self.live_strategy_var = tk.StringVar()
        self.live_strategy_combo = ttk.Combobox(strategy_frame, textvariable=self.live_strategy_var, width=20)
        self.live_strategy_combo.pack(side="left", padx=5)
        
        # Warning label
        warning_label = ttk.Label(live_frame, 
                                text="⚠️ WARNING: This will trade with real money!",
                                font=("Arial", 10, "bold"), foreground="red")
        warning_label.pack(anchor="w", pady=10)
        
        # Start button (changed to OPEN like backtesting)
        button_frame = ttk.Frame(live_frame)
        button_frame.pack(fill="x", pady=10)
        
        ttk.Button(button_frame, text="OPEN LIVE TRADER", 
                command=self.open_live_trader).pack(side="left", padx=5)
        
        # Load accounts and strategies when tab is created
        self.load_live_trading_options()

    def create_rl_ai_section(self):
        frame = ttk.LabelFrame(self.main_content_frame, text="🧠 RL AI MODULE", padding=10)
        frame.pack(fill="x", padx=10, pady=5)

        self.rl_status = tk.StringVar(value="⚫ IDLE (No RL model trained yet)")
        ttk.Label(frame, textvariable=self.rl_status, font=("Arial", 10, "bold")).pack(anchor="w")
        ttk.Label(
            frame,
            text="Run RL training/evaluation from a dedicated control window.",
            font=("Arial", 9),
        ).pack(anchor="w", pady=(2, 6))

        button_frame = ttk.Frame(frame)
        button_frame.pack(fill="x", pady=5)
        ttk.Button(
            button_frame,
            text="OPEN RL CONTROL PANEL",
            command=self.open_rl_ai_window,
        ).pack(side="left", padx=5)
    
    def create_placeholder_section(self, title, module_name):
        frame = ttk.LabelFrame(self.main_content_frame, text=f"{title} (Coming Soon)", padding=10)
        frame.pack(fill="x", padx=10, pady=5)
        
        status_var = tk.StringVar(value="⚫ NOT IMPLEMENTED")
        ttk.Label(frame, textvariable=status_var, font=("Arial", 10, "bold")).pack(anchor="w")
        
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill="x", pady=5)
        
        ttk.Button(button_frame, text="START", state="disabled").pack(side="left", padx=5)
        ttk.Button(button_frame, text="SETTINGS", state="disabled").pack(side="left", padx=5)
    
    def create_bottom_buttons(self):
        bottom_frame = ttk.Frame(self.main_content_frame)
        bottom_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Button(bottom_frame, text="📋 SYSTEM LOGS").pack(side="left", padx=5)
        ttk.Button(bottom_frame, text="🔧 GLOBAL SETTINGS").pack(side="left", padx=5)
        ttk.Button(bottom_frame, text="❌ EXIT", command=self.root.quit).pack(side="right", padx=5)

    def open_rl_ai_window(self):
        try:
            from rl_ai.gui.rl_ai_control_gui import RLAIControlGUI

            rl_window = tk.Toplevel(self.root)
            rl_window.title("RL AI Control Panel")
            rl_window.geometry("960x720")
            RLAIControlGUI(rl_window)

            if hasattr(self, "rl_status"):
                self.rl_status.set("🟢 READY")
        except Exception as e:
            if hasattr(self, "rl_status"):
                self.rl_status.set("🔴 FAILED")
            messagebox.showerror("RL AI", f"Failed to open RL control panel: {e}")

    def open_fast_strategy_finder(self):
        try:
            from simple_strategy.fast_finder.gui import FastStrategyFinderGUI

            window = tk.Toplevel(self.root)
            FastStrategyFinderGUI(window)
        except Exception as e:
            messagebox.showerror("Fast Finder", f"Failed to open fast strategy finder: {e}")

    def open_unique_advanced_engine(self):
        try:
            from simple_strategy.unique_engine.gui import UniqueStrategyEngineGUI

            window = tk.Toplevel(self.root)
            UniqueStrategyEngineGUI(window)
        except Exception as e:
            messagebox.showerror("Unique Engine", f"Failed to open unique advanced engine: {e}")

    def _auto_evolve_config_path(self):
        return os.path.join("simple_strategy", "auto_evolve", "configs", "default.json")

    def _auto_evolve_overnight_config_path(self):
        return os.path.join("simple_strategy", "auto_evolve", "configs", "overnight.json")

    def _auto_evolve_multiday_config_path(self):
        return os.path.join("simple_strategy", "auto_evolve", "configs", "multiday.json")

    def _auto_evolve_output_dir(self):
        project_root = os.path.dirname(__file__)
        default_output = os.path.join("simple_strategy", "auto_evolve", "runs")
        config_rel = self._auto_evolve_config_path()
        config_abs = os.path.join(project_root, config_rel)

        output_rel = default_output
        try:
            with open(config_abs, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
                output_rel = payload.get("output_dir", default_output)
        except Exception:
            output_rel = default_output

        return os.path.join(project_root, output_rel)

    def _auto_evolve_latest_run_dir(self):
        output_dir = self._auto_evolve_output_dir()
        if not os.path.exists(output_dir):
            return ""
        run_dirs = [
            entry.path
            for entry in os.scandir(output_dir)
            if entry.is_dir() and entry.name.startswith("run_")
        ]
        if not run_dirs:
            return ""
        return max(run_dirs, key=os.path.getmtime)

    @staticmethod
    def _auto_evolve_run_is_complete(run_dir):
        return (
            os.path.exists(os.path.join(run_dir, "summary.txt"))
            and os.path.exists(os.path.join(run_dir, "top_results.json"))
        )

    def _has_recent_unfinished_run(self, max_age_seconds=900):
        latest_run_dir = self._auto_evolve_latest_run_dir()
        if not latest_run_dir:
            return False
        if self._auto_evolve_run_is_complete(latest_run_dir):
            return False

        checkpoint_path = os.path.join(latest_run_dir, "checkpoints", "latest.json")
        run_config_path = os.path.join(latest_run_dir, "run_config.resolved.json")
        reference_path = checkpoint_path if os.path.exists(checkpoint_path) else run_config_path
        if not os.path.exists(reference_path):
            return False

        last_update_age = max(0.0, time.time() - os.path.getmtime(reference_path))
        return last_update_age <= float(max_age_seconds)

    @staticmethod
    def _read_json_file(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    @staticmethod
    def _format_duration(seconds):
        if seconds is None:
            return "-"
        total = max(0, int(seconds))
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        if minutes > 0:
            return f"{minutes}m {secs}s"
        return f"{secs}s"

    @staticmethod
    def _run_name_to_timestamp(run_name):
        if not isinstance(run_name, str) or not run_name.startswith("run_"):
            return None
        try:
            parsed = datetime.strptime(run_name[4:], "%Y%m%d_%H%M%S")
            return parsed.timestamp()
        except Exception:
            return None

    def _schedule_auto_evolve_status_refresh(self):
        try:
            self.ae_status_job = self.root.after(self.ae_status_refresh_ms, self._auto_evolve_status_tick)
        except Exception:
            self.ae_status_job = None

    def _auto_evolve_status_tick(self):
        self.refresh_auto_evolve_status()
        self._schedule_auto_evolve_status_refresh()

    def refresh_auto_evolve_status(self):
        try:
            latest_run_dir = self._auto_evolve_latest_run_dir()
            if not latest_run_dir:
                self.ae_status.set("⚫ IDLE")
                self.ae_run_var.set("Run: -")
                self.ae_progress_var.set("Progress: 0%")
                self.ae_eval_var.set("Evaluated: 0/0")
                self.ae_eta_var.set("ETA: -")
                if hasattr(self, "ae_progressbar"):
                    self.ae_progressbar["value"] = 0
                return

            run_name = os.path.basename(latest_run_dir)
            run_config = self._read_json_file(os.path.join(latest_run_dir, "run_config.resolved.json"))
            checkpoint = self._read_json_file(os.path.join(latest_run_dir, "checkpoints", "latest.json"))
            has_summary = os.path.exists(os.path.join(latest_run_dir, "summary.txt"))
            has_top_results = os.path.exists(os.path.join(latest_run_dir, "top_results.json"))

            checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
            checkpoint_config = checkpoint_config if isinstance(checkpoint_config, dict) else {}
            generations = int(checkpoint_config.get("generations", run_config.get("generations", 0)) or 0)
            population_size = int(checkpoint_config.get("population_size", run_config.get("population_size", 0)) or 0)
            checkpoint_population = checkpoint.get("population", [])
            if isinstance(checkpoint_population, list) and checkpoint_population:
                population_size = len(checkpoint_population)
            expected_evals = generations * population_size if generations > 0 and population_size > 0 else 0

            generation_idx = int(checkpoint.get("generation", -1) or -1)
            all_results = checkpoint.get("all_results", [])
            evaluated = len(all_results) if isinstance(all_results, list) else 0

            progress_pct = 0.0
            if expected_evals > 0 and evaluated > 0:
                progress_pct = min(100.0, (100.0 * evaluated) / expected_evals)
            elif generations > 0 and generation_idx >= 0:
                progress_pct = min(100.0, (100.0 * (generation_idx + 1)) / generations)

            if has_summary and has_top_results:
                progress_pct = 100.0

            run_start_path = os.path.join(latest_run_dir, "run_config.resolved.json")
            run_start_ts = self._run_name_to_timestamp(run_name)
            if run_start_ts is None:
                if os.path.exists(run_start_path):
                    run_start_ts = os.path.getmtime(run_start_path)
                else:
                    run_start_ts = os.path.getctime(latest_run_dir)
            elapsed_sec = max(0.0, time.time() - run_start_ts)

            eta_sec = None
            if 0.0 < progress_pct < 100.0:
                eta_sec = elapsed_sec * ((100.0 - progress_pct) / progress_pct)

            checkpoint_path = os.path.join(latest_run_dir, "checkpoints", "latest.json")
            last_update_ts = os.path.getmtime(checkpoint_path) if os.path.exists(checkpoint_path) else run_start_ts
            seconds_since_update = max(0.0, time.time() - last_update_ts)
            process_running = self.auto_evolve_process is not None and self.auto_evolve_process.poll() is None

            if has_summary and has_top_results:
                status_text = "✅ COMPLETE"
            elif process_running:
                status_text = "🟢 RUNNING"
            elif seconds_since_update < 3600 and (evaluated > 0 or generation_idx >= 0):
                status_text = "🟢 RUNNING (DETECTED)"
            elif elapsed_sec < 1800 and not has_summary:
                status_text = "🟡 STARTING"
            elif evaluated > 0 or generation_idx >= 0:
                status_text = "🟡 INCOMPLETE / STOPPED"
            else:
                status_text = "🟡 WAITING FOR FIRST CHECKPOINT"

            completed_generations = max(0, generation_idx + 1)
            generation_text = f"{completed_generations}/{generations}" if generations > 0 else "-"
            eval_text = f"{evaluated}/{expected_evals}" if expected_evals > 0 else f"{evaluated}/?"

            self.ae_status.set(status_text)
            self.ae_run_var.set(f"Run: {run_name}")
            self.ae_progress_var.set(f"Progress: {progress_pct:.1f}% | Generations: {generation_text}")
            self.ae_eval_var.set(f"Evaluated: {eval_text}")
            self.ae_eta_var.set(
                f"Elapsed: {self._format_duration(elapsed_sec)} | ETA: {self._format_duration(eta_sec)} | Last update: {self._format_duration(seconds_since_update)} ago"
            )
            if hasattr(self, "ae_progressbar"):
                self.ae_progressbar["value"] = progress_pct
        except Exception:
            self.ae_status.set("🟡 STATUS UNKNOWN")

    def _launch_auto_evolve(self, extra_args):
        try:
            if self.auto_evolve_process is not None and self.auto_evolve_process.poll() is not None:
                self.auto_evolve_process = None

            if self.auto_evolve_process is not None and self.auto_evolve_process.poll() is None:
                messagebox.showinfo("Auto Evolve", "A run is already active. Wait for it to finish, or resume it.")
                self.refresh_auto_evolve_status()
                return

            if self._has_recent_unfinished_run():
                messagebox.showinfo(
                    "Auto Evolve",
                    "An unfinished recent run was detected. Use RESUME LAST RUN or wait before starting another run.",
                )
                self.refresh_auto_evolve_status()
                return

            project_root = os.path.dirname(__file__)
            seed_args = self._get_auto_evolve_seed_args()
            if seed_args is None:
                return
            command = [sys.executable, "-m", "simple_strategy.auto_evolve.run_evolution"] + extra_args + seed_args
            self.auto_evolve_process = subprocess.Popen(command, cwd=project_root)
            self.ae_status.set("🟢 RUNNING")
            self.refresh_auto_evolve_status()
        except Exception as e:
            self.ae_status.set("🔴 FAILED")
            messagebox.showerror("Auto Evolve Error", f"Failed to start auto evolution: {e}")

    def _get_auto_evolve_seed_args(self):
        seed_text = self.ae_seed_var.get().strip() if hasattr(self, "ae_seed_var") else ""
        if not seed_text:
            return []
        try:
            seed = int(seed_text)
        except ValueError:
            messagebox.showerror("Auto Evolve Error", "Seed must be a positive integer.")
            return None
        if seed <= 0:
            messagebox.showerror("Auto Evolve Error", "Seed must be greater than 0.")
            return None
        return ["--seed", str(seed)]

    def start_auto_evolve_full(self):
        config_path = self._auto_evolve_config_path()
        self._launch_auto_evolve(["--config", config_path])

    def start_auto_evolve_overnight(self):
        config_path = self._auto_evolve_overnight_config_path()
        self._launch_auto_evolve([
            "--config", config_path,
            "--population", "16",
            "--generations", "12",
            "--workers", "4",
            "--max-runtime-hours", "9.5",
        ])

    def start_auto_evolve_multiday(self):
        config_path = self._auto_evolve_multiday_config_path()
        self._launch_auto_evolve([
            "--config", config_path,
            "--population", "24",
            "--generations", "120",
            "--workers", "4",
            "--max-runtime-hours", "92",
        ])

    def start_auto_evolve_smoke(self):
        self._launch_auto_evolve(["--smoke", "--population", "8", "--generations", "2", "--workers", "1"])

    def resume_auto_evolve_last(self):
        output_dir = self._auto_evolve_output_dir()
        if not os.path.exists(output_dir):
            messagebox.showinfo("Auto Evolve", "No output folder found yet.")
            return

        run_dirs = [entry.path for entry in os.scandir(output_dir) if entry.is_dir()]
        if not run_dirs:
            messagebox.showinfo("Auto Evolve", "No previous run found to resume.")
            return

        latest_run_dir = max(run_dirs, key=os.path.getmtime)
        self._launch_auto_evolve(["--resume-run-dir", latest_run_dir])

    def open_auto_evolve_output(self):
        output_dir = self._auto_evolve_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        try:
            os.startfile(output_dir)
        except Exception:
            messagebox.showinfo("Auto Evolve Output", f"Output folder: {output_dir}")

    def clear_auto_evolve_runs(self):
        output_dir = self._auto_evolve_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        run_dirs = [entry.path for entry in os.scandir(output_dir) if entry.is_dir()]
        if not run_dirs:
            messagebox.showinfo("Auto Evolve", "No run folders to delete.")
            return

        should_delete = messagebox.askyesno(
            "Delete Runs",
            f"Delete all {len(run_dirs)} run folders in:\n{output_dir}?",
        )
        if not should_delete:
            return

        import shutil

        deleted = 0
        for run_dir in run_dirs:
            try:
                shutil.rmtree(run_dir)
                deleted += 1
            except Exception:
                pass

        messagebox.showinfo("Auto Evolve", f"Deleted {deleted} run folder(s).")
        self.refresh_auto_evolve_status()
    
    def start_data_collection(self):
        try:
            # Start data collection using the launcher script in data_collection folder
            project_root = os.path.dirname(__file__)
            launcher_path = os.path.join(
                os.path.dirname(__file__),
                "shared_modules",
                "data_collection",
                "launch_data_collection.py",
            )
            startup_log_path = self._create_startup_log_path("data_collector_startup")
            creation_flags = 0
            if os.name == "nt":
                creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            with open(startup_log_path, "a", encoding="utf-8") as startup_log_file:
                self.data_collection_process = subprocess.Popen(
                    [sys.executable, launcher_path],
                    stdout=startup_log_file,
                    stderr=startup_log_file,
                    cwd=project_root,
                    creationflags=creation_flags,
                )
            self.data_collection_startup_log_path = startup_log_path
            
            # Update UI
            self.dc_status.set("🟢 RUNNING")
            self.dc_start_btn.config(state="disabled")
            self.dc_stop_btn.config(state="normal")
            self._refresh_paper_data_status()
            self.root.after(1500, self._verify_data_collection_process_alive)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start data collection: {e}")
    
    def stop_data_collection(self):
        if self.data_collection_process:
            try:
                self.data_collection_process.terminate()
                self.data_collection_process = None
                self.data_collection_startup_log_path = None
                
                # Update UI
                self.dc_status.set("🔴 STOPPED")
                self.dc_start_btn.config(state="normal")
                self.dc_stop_btn.config(state="disabled")
                self._refresh_paper_data_status()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to stop data collection: {e}")

    def _verify_data_collection_process_alive(self):
        """Show a clear error when collector exits right after launch."""
        process = self.data_collection_process
        if process is None:
            return
        exit_code = process.poll()
        if exit_code is None:
            return

        self.data_collection_process = None
        self.dc_status.set("🔴 STOPPED")
        self.dc_start_btn.config(state="normal")
        self.dc_stop_btn.config(state="disabled")
        self._refresh_paper_data_status()
        startup_hint = ""
        if self.data_collection_startup_log_path:
            startup_hint = f"\nStartup log: {self.data_collection_startup_log_path}"
        self.data_collection_startup_log_path = None
        messagebox.showerror(
            "Data Collection Failed",
            f"Data collector exited immediately (code {exit_code}).\n"
            f"Open it again and check latest file in Logs/ for details.{startup_hint}",
        )

    def _create_startup_log_path(self, prefix):
        logs_dir = os.path.join(os.path.dirname(__file__), "Logs")
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return os.path.join(logs_dir, f"{prefix}_{timestamp}.log")

    def _verify_paper_trader_process_alive(self, process, startup_log_path):
        if process is None:
            return
        exit_code = process.poll()
        if exit_code is None:
            return
        startup_hint = ""
        if startup_log_path:
            startup_hint = f"\nStartup log: {startup_log_path}"
        messagebox.showerror(
            "Paper Trader Failed",
            f"Paper trader closed right after launch (code {exit_code}).{startup_hint}",
        )

    def load_paper_trading_options(self):
        """Load demo accounts and strategies for paper trading"""
        try:
            # Load demo accounts from api_accounts.json
            api_accounts_file = os.path.join(os.path.dirname(__file__), 
                                        "simple_strategy", "trading", "api_accounts.json")
            if os.path.exists(api_accounts_file):
                import json
                with open(api_accounts_file, 'r') as f:
                    accounts = json.load(f)
                    demo_accounts = list(accounts.get('demo_accounts', {}).keys())
                    self.paper_account_combo['values'] = demo_accounts
                    if demo_accounts:
                        self.paper_account_combo.current(0)
            
            # Load strategies (only files starting with "Strategy" and ending with ".py")
            strategies_dir = os.path.join(os.path.dirname(__file__), "simple_strategy", "strategies")
            if os.path.exists(strategies_dir):
                strategies = [f.replace('.py', '') for f in os.listdir(strategies_dir) 
                            if f.startswith('Strategy') and f.endswith('.py')]
                self.paper_strategy_combo['values'] = strategies
                if strategies:
                    self.paper_strategy_combo.current(0)
                    
        except Exception as e:
            print(f"Error loading paper trading options: {e}")

    def load_live_trading_options(self):
        """Load live accounts and strategies for live trading"""
        try:
            # Load live accounts from api_accounts.json
            api_accounts_file = os.path.join(os.path.dirname(__file__), 
                                        "simple_strategy", "trading", "api_accounts.json")
            if os.path.exists(api_accounts_file):
                import json
                with open(api_accounts_file, 'r') as f:
                    accounts = json.load(f)
                    live_accounts = list(accounts.get('live_accounts', {}).keys())
                    self.live_account_combo['values'] = live_accounts
                    if live_accounts:
                        self.live_account_combo.current(0)
            
            # Load strategies (only files starting with "Strategy" and ending with ".py")
            strategies_dir = os.path.join(os.path.dirname(__file__), "simple_strategy", "strategies")
            if os.path.exists(strategies_dir):
                strategies = [f.replace('.py', '') for f in os.listdir(strategies_dir) 
                            if f.startswith('Strategy') and f.endswith('.py')]
                self.paper_strategy_combo['values'] = strategies
                if strategies:
                    self.paper_strategy_combo.current(0)
                    
        except Exception as e:
            print(f"Error loading live trading options: {e}")

    def open_paper_trader(self):
        """Open paper trading window (allows multiple instances like backtesting)"""
        try:
            # For now, show message with selected options
            account = self.paper_account_var.get()
            strategy = self.paper_strategy_var.get()
            balance = self.paper_balance_var.get()
            
            # Open paper trading window like backtesting
            try:
                project_root = os.path.dirname(__file__)
                launcher_path = os.path.join(os.path.dirname(__file__), 
                                        "simple_strategy", "trading", "paper_trading_launcher.py")
                startup_log_path = self._create_startup_log_path("paper_trader_startup")
                creation_flags = 0
                if os.name == "nt":
                    creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                
                # Pass parameters as command line arguments
                with open(startup_log_path, "a", encoding="utf-8") as startup_log_file:
                    paper_process = subprocess.Popen(
                        [sys.executable, launcher_path, account, strategy, balance],
                        stdout=startup_log_file,
                        stderr=startup_log_file,
                        cwd=project_root,
                        creationflags=creation_flags,
                    )
                self.root.after(
                    2500,
                    lambda p=paper_process, lp=startup_log_path: self._verify_paper_trader_process_alive(p, lp),
                )
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open paper trader: {e}")
                
            # In the future, this will open a paper trading window like backtesting
            # launcher_path = "paper_trading_launcher.py"
            # subprocess.Popen([sys.executable, launcher_path])
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open paper trader: {e}")

    def open_live_trader(self):
        """Open live trading window (allows multiple instances like backtesting)"""
        try:
            # For now, show warning with selected options
            account = self.live_account_var.get()
            strategy = self.live_strategy_var.get()
            
            message = f"Live Trading Options:\nAccount: {account}\nStrategy: {strategy}"
            messagebox.showwarning("Live Trading", message + "\n\n⚠️ WARNING: This will trade with real money!\n\nLive trading feature will be implemented in the next phase!")
            
            # In the future, this will open a live trading window like backtesting
            # launcher_path = "live_trading_launcher.py"
            # subprocess.Popen([sys.executable, launcher_path])
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open live trader: {e}")
    def open_simple_strategy_settings(self):
        # Open simple strategy settings
        messagebox.showinfo("Settings", "Simple strategy settings would open here")
    
    def open_data_collection_settings(self):
        # Open data collection settings (could open config file or settings GUI)
        messagebox.showinfo("Settings", "Data collection settings would open here")
    
    def start_simple_strategy(self):
        """Start the simple strategy backtester with optimized parameters"""
        try:
            # Import the correct GUI class
            from simple_strategy.gui_monitor import SimpleStrategyGUI
            
            # Create a new window for the backtester
            backtester_window = tk.Toplevel(self.root)
            backtester_window.title("Strategy Backtester")
            backtester_window.geometry("1000x700")
            
            # Create the backtester GUI
            backtester_gui = SimpleStrategyGUI(backtester_window)
            
            # Let the GUI handle its own initialization
            # Don't try to set parameters here, let the GUI do it
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start backtester: {e}")
    
    def open_simple_strategy_settings(self):
        # Open simple strategy settings (could open config file or settings GUI)
        messagebox.showinfo("Settings", "Simple strategy settings would open here")

    def open_api_manager(self):
        # Open API Manager (Phase 2 implementation)
        try:
            launcher_path = os.path.join(os.path.dirname(__file__), 
                                    "simple_strategy", "trading", "api_gui.py")
            subprocess.Popen([sys.executable, launcher_path])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open API Manager: {e}")

    def open_parameter_manager(self):  
        # Open the parameter manager GUI
        param_window = tk.Toplevel(self.root)
        ParameterGUI(param_window)

    def open_api_manager(self):
        # Open API manager GUI
        api_window = tk.Toplevel(self.root)
        api_window.title("API Account Manager")
        from simple_strategy.trading.api_gui import APIGUI
        APIGUI(api_window)

    def start_paper_trading(self):
        """Start paper trading with the new engine"""
        try:
            if self.trading_running:
                messagebox.showinfo("Already Running", "Paper trading is already running!")
                return
            
            # Get selected options
            account = self.paper_account_var.get()
            strategy = self.paper_strategy_var.get()
            balance = self.paper_balance_var.get()
            min_24h_notional_usdt = float(self.paper_min_24h_notional_var.get())
            enable_exchange_sl = bool(self.paper_enable_exchange_sl_var.get())
            enable_exchange_trailing = bool(self.paper_enable_exchange_trailing_var.get())
            exchange_sl_pct_input = float(self.paper_exchange_sl_pct_var.get())
            enable_atr_exit = bool(self.paper_enable_atr_exit_var.get())
            atr_period = 14
            atr_sl_mult = 1.3
            atr_tp_mult = 1.7
            if enable_atr_exit:
                atr_period = int(float(self.paper_atr_period_var.get()))
                atr_sl_mult = float(self.paper_atr_sl_mult_var.get())
                atr_tp_mult = float(self.paper_atr_tp_mult_var.get())

            if enable_atr_exit:
                if atr_period < 2:
                    messagebox.showerror("Invalid ATR", "ATR period must be at least 2.")
                    return
                if atr_sl_mult <= 0 or atr_tp_mult <= 0:
                    messagebox.showerror("Invalid ATR", "ATR multipliers must be greater than 0.")
                    return

            if not enable_atr_exit and enable_exchange_sl and exchange_sl_pct_input <= 0:
                messagebox.showerror("Invalid Stop Loss", "Exchange SL % must be greater than 0.")
                return
            if not enable_atr_exit and enable_exchange_trailing and not enable_exchange_sl:
                messagebox.showerror("Invalid Stop Mode", "Enable Exchange SL before enabling trailing stop.")
                return
            exchange_sl_pct = max(0.0001, exchange_sl_pct_input / 100.0)
            risk_exit_mode = (
                'atr' if enable_atr_exit
                else 'trailing' if (enable_exchange_sl and enable_exchange_trailing)
                else 'fixed' if enable_exchange_sl
                else 'none'
            )
            enable_exchange_stop_loss = risk_exit_mode in {'fixed', 'trailing', 'atr'}
            enable_exchange_trailing_stop = risk_exit_mode == 'trailing'
            global_rules_config = {
                'enable_global_rules': bool(self.paper_enable_global_rules_var.get()),
                'global_rules_profile': str(self.paper_global_rules_profile_var.get()).strip().lower() or "balanced",
                'global_rules': {
                    'min_24h_notional_usdt': min_24h_notional_usdt,
                },
            }
            
            if not account or not strategy:
                messagebox.showerror("Error", "Please select both account and strategy!")
                return

            if not self._confirm_paper_trading_with_stale_data():
                return
             
            # Initialize and start trading engine
            from simple_strategy.trading.paper_trading_engine import PaperTradingEngine
            self.paper_trading_engine = PaperTradingEngine(
                account,
                strategy,
                balance,
                global_rules_config=global_rules_config,
                enable_exchange_stop_loss=enable_exchange_stop_loss,
                enable_exchange_trailing_stop=enable_exchange_trailing_stop,
                exchange_stop_loss_pct=exchange_sl_pct,
                risk_exit_mode=risk_exit_mode,
                risk_sl_pct=exchange_sl_pct,
                risk_atr_period=atr_period,
                risk_atr_sl_multiplier=atr_sl_mult,
                risk_atr_tp_multiplier=atr_tp_mult,
            )
            
            # Start trading in a separate thread to avoid freezing GUI
            import threading
            self.trading_thread = threading.Thread(target=self.paper_trading_engine.start_trading)
            self.trading_thread.daemon = True
            self.trading_thread.start()
            
            self.trading_running = True
            messagebox.showinfo("Success", f"Paper trading started for {strategy}!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start paper trading: {str(e)}")
    
    def stop_paper_trading(self): 
        """Stop paper trading"""
        try:
            if not self.trading_running or not self.paper_trading_engine:
                messagebox.showinfo("Not Running", "Paper trading is not running!")
                return
            
            # Stop the trading engine
            self.paper_trading_engine.is_running = False
            self.trading_running = False
            
            messagebox.showinfo("Success", "Paper trading stopped!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop paper trading: {str(e)}")

def _handle_gui_sigint(_signum, _frame):
    # Prevent accidental Ctrl+C in terminal from killing the dashboard.
    print("Ctrl+C ignored while GUI is running. Close the window to exit.")

if __name__ == "__main__":
    try:
        signal.signal(signal.SIGINT, _handle_gui_sigint)
    except Exception:
        pass

    root = tk.Tk()
    app = TradingBotDashboard(root)

    while True:
        try:
            root.mainloop()
            break
        except KeyboardInterrupt:
            try:
                if root.winfo_exists():
                    messagebox.showwarning(
                        "Interrupt Ignored",
                        "Ctrl+C was pressed in the terminal.\nThe dashboard will stay open.",
                    )
                else:
                    break
            except Exception:
                break
