import os
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any, Dict

from simple_strategy.unique_engine.runner import DEFAULT_UNIQUE_CONFIG, QUALITY_PRESETS, run_unique_engine


class UniqueStrategyEngineGUI:
    def __init__(self, root: tk.Toplevel):
        self.root = root
        self.root.title("Unique Advanced Strategy Engine")
        self.root.geometry("1020x780")

        self.worker_thread: threading.Thread | None = None
        self.stop_event: threading.Event | None = None
        self.run_started_ts: float | None = None
        self.last_run_dir = ""

        self._build_vars()
        self._build_layout()
        self._sync_quality_preset_fields(log_change=False)

    def _build_vars(self) -> None:
        self.symbols_var = tk.StringVar(value=",".join(DEFAULT_UNIQUE_CONFIG["symbols"]))
        self.timeframes_var = tk.StringVar(value=",".join(DEFAULT_UNIQUE_CONFIG["timeframes"]))
        self.start_date_var = tk.StringVar(value=str(DEFAULT_UNIQUE_CONFIG["start_date"]))
        self.end_date_var = tk.StringVar(value=str(DEFAULT_UNIQUE_CONFIG["end_date"]))
        self.blueprints_var = tk.StringVar(value=str(DEFAULT_UNIQUE_CONFIG["blueprints"]))

        self.candidate_per_blueprint_var = tk.StringVar(
            value=str(DEFAULT_UNIQUE_CONFIG["candidate_count_per_blueprint"])
        )
        self.workers_var = tk.StringVar(value=str(DEFAULT_UNIQUE_CONFIG["workers"]))
        self.top_k_var = tk.StringVar(value=str(DEFAULT_UNIQUE_CONFIG["top_k"]))
        self.publish_top_n_var = tk.StringVar(value=str(DEFAULT_UNIQUE_CONFIG["publish_top_n"]))
        self.quality_preset_var = tk.StringVar(value=str(DEFAULT_UNIQUE_CONFIG["quality_preset"]))

        self.min_return_var = tk.StringVar(value=str(DEFAULT_UNIQUE_CONFIG["min_return_pct"]))
        self.max_drawdown_var = tk.StringVar(value=str(DEFAULT_UNIQUE_CONFIG["max_drawdown_pct"]))
        self.min_win_rate_var = tk.StringVar(value=str(DEFAULT_UNIQUE_CONFIG["min_win_rate_pct"]))
        self.min_trades_var = tk.StringVar(value=str(DEFAULT_UNIQUE_CONFIG["min_trades"]))
        self.min_test_return_var = tk.StringVar(value=str(DEFAULT_UNIQUE_CONFIG["min_test_return_pct"]))
        self.max_test_drawdown_var = tk.StringVar(value=str(DEFAULT_UNIQUE_CONFIG["max_test_drawdown_pct"]))
        self.min_test_win_rate_var = tk.StringVar(value=str(DEFAULT_UNIQUE_CONFIG["min_test_win_rate_pct"]))
        self.min_test_trades_var = tk.StringVar(value=str(DEFAULT_UNIQUE_CONFIG["min_test_trades"]))

        self.train_ratio_var = tk.StringVar(value=str(DEFAULT_UNIQUE_CONFIG["train_ratio"]))
        self.data_dir_var = tk.StringVar(value=str(DEFAULT_UNIQUE_CONFIG["data_dir"]))
        self.output_dir_var = tk.StringVar(value=str(DEFAULT_UNIQUE_CONFIG["output_dir"]))
        self.search_space_var = tk.StringVar(value=str(DEFAULT_UNIQUE_CONFIG["search_space_path"]))
        self.seed_var = tk.StringVar(value=str(DEFAULT_UNIQUE_CONFIG["seed"]))
        self.enable_mtf_var = tk.BooleanVar(value=bool(DEFAULT_UNIQUE_CONFIG["enable_mtf_context_filter"]))
        self.enable_regime_var = tk.BooleanVar(value=bool(DEFAULT_UNIQUE_CONFIG["enable_regime_filter"]))
        self.enable_split_var = tk.BooleanVar(value=bool(DEFAULT_UNIQUE_CONFIG["enable_train_test_split"]))
        self.enable_fallback_var = tk.BooleanVar(value=bool(DEFAULT_UNIQUE_CONFIG["enable_gate_fallback"]))
        self.context_fast_var = tk.StringVar(value=str(DEFAULT_UNIQUE_CONFIG["context_fast_period"]))
        self.context_slow_var = tk.StringVar(value=str(DEFAULT_UNIQUE_CONFIG["context_slow_period"]))
        self.regime_symbol_var = tk.StringVar(value=str(DEFAULT_UNIQUE_CONFIG["regime_symbol"]))

        self.status_var = tk.StringVar(value="IDLE")
        self.progress_var = tk.StringVar(value="Progress: 0/0")
        self.best_var = tk.StringVar(value="Best Score: -")
        self.elapsed_var = tk.StringVar(value="Elapsed: 0s")
        self.run_dir_var = tk.StringVar(value="Run Dir: -")

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root)
        container.pack(fill="both", expand=True)

        self.main_canvas = tk.Canvas(container, highlightthickness=0)
        self.main_scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.main_canvas.yview)
        self.main_canvas.configure(yscrollcommand=self.main_scrollbar.set)
        self.main_scrollbar.pack(side="right", fill="y")
        self.main_canvas.pack(side="left", fill="both", expand=True)

        top_frame = ttk.Frame(self.main_canvas, padding=10)
        self.main_canvas_window = self.main_canvas.create_window((0, 0), window=top_frame, anchor="nw")
        top_frame.bind("<Configure>", self._on_main_content_configure)
        self.main_canvas.bind("<Configure>", self._on_main_canvas_configure)
        self._bind_mousewheel(self.main_canvas)
        self._bind_mousewheel(top_frame)

        config_frame = ttk.LabelFrame(top_frame, text="Core Settings", padding=10)
        config_frame.pack(fill="x", pady=4)
        self._grid_row(config_frame, 0, "Symbols (comma):", self.symbols_var)
        self._grid_row(config_frame, 1, "Timeframes (comma):", self.timeframes_var)
        self._grid_row(config_frame, 2, "Start Date:", self.start_date_var)
        self._grid_row(config_frame, 3, "End Date:", self.end_date_var)
        self._grid_row(
            config_frame,
            4,
            "Blueprints (all or comma list):",
            self.blueprints_var,
        )

        speed_frame = ttk.LabelFrame(top_frame, text="Search Settings", padding=10)
        speed_frame.pack(fill="x", pady=4)
        ttk.Label(speed_frame, text="Quality Preset:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        preset_combo = ttk.Combobox(
            speed_frame,
            textvariable=self.quality_preset_var,
            values=["custom", "A", "B"],
            state="readonly",
            width=20,
        )
        preset_combo.grid(row=0, column=1, sticky="w", padx=5, pady=3)
        preset_combo.bind("<<ComboboxSelected>>", self._on_quality_preset_selected)
        ttk.Button(speed_frame, text="APPLY PRESET A (5%)", command=self.apply_preset_a).grid(
            row=0, column=2, sticky="w", padx=5, pady=3
        )
        ttk.Button(speed_frame, text="APPLY PRESET B (10%)", command=self.apply_preset_b).grid(
            row=0, column=3, sticky="w", padx=5, pady=3
        )
        self._grid_row(speed_frame, 1, "Candidates per Blueprint:", self.candidate_per_blueprint_var)
        self._grid_row(speed_frame, 2, "Workers:", self.workers_var)
        self._grid_row(speed_frame, 3, "Top K:", self.top_k_var)
        self._grid_row(speed_frame, 4, "Publish Top N:", self.publish_top_n_var)

        gate_frame = ttk.LabelFrame(top_frame, text="Quality Gates", padding=10)
        gate_frame.pack(fill="x", pady=4)
        self._grid_row(gate_frame, 0, "Min Return %:", self.min_return_var)
        self._grid_row(gate_frame, 1, "Max Drawdown %:", self.max_drawdown_var)
        self._grid_row(gate_frame, 2, "Min Win Rate %:", self.min_win_rate_var)
        self._grid_row(gate_frame, 3, "Min Trades:", self.min_trades_var)
        self._grid_row(gate_frame, 4, "Min Test Return %:", self.min_test_return_var)
        self._grid_row(gate_frame, 5, "Max Test Drawdown %:", self.max_test_drawdown_var)
        self._grid_row(gate_frame, 6, "Min Test Win Rate %:", self.min_test_win_rate_var)
        self._grid_row(gate_frame, 7, "Min Test Trades:", self.min_test_trades_var)

        filter_frame = ttk.LabelFrame(top_frame, text="Context and Reliability", padding=10)
        filter_frame.pack(fill="x", pady=4)
        ttk.Checkbutton(
            filter_frame,
            text="Enable Multi-Timeframe Context Filter",
            variable=self.enable_mtf_var,
        ).grid(row=0, column=0, sticky="w", padx=5, pady=3)
        ttk.Checkbutton(
            filter_frame,
            text="Enable Multi-Symbol Regime Filter",
            variable=self.enable_regime_var,
        ).grid(row=0, column=1, sticky="w", padx=5, pady=3)
        ttk.Checkbutton(
            filter_frame,
            text="Enable Train/Test Split",
            variable=self.enable_split_var,
        ).grid(row=1, column=0, sticky="w", padx=5, pady=3)
        ttk.Checkbutton(
            filter_frame,
            text="Enable Gate Fallback",
            variable=self.enable_fallback_var,
        ).grid(row=1, column=1, sticky="w", padx=5, pady=3)
        self._grid_row(filter_frame, 2, "Train Ratio (0.5-0.95):", self.train_ratio_var)
        self._grid_row(filter_frame, 3, "Context Fast EMA:", self.context_fast_var)
        self._grid_row(filter_frame, 4, "Context Slow EMA:", self.context_slow_var)
        self._grid_row(filter_frame, 5, "Regime Symbol (blank=first):", self.regime_symbol_var)

        io_frame = ttk.LabelFrame(top_frame, text="Data and Output", padding=10)
        io_frame.pack(fill="x", pady=4)
        self._grid_row(io_frame, 0, "Data Dir:", self.data_dir_var)
        self._grid_row(io_frame, 1, "Output Dir:", self.output_dir_var)
        self._grid_row(io_frame, 2, "Search Space JSON (optional):", self.search_space_var)
        self._grid_row(io_frame, 3, "Seed:", self.seed_var)

        action_frame = ttk.Frame(top_frame)
        action_frame.pack(fill="x", pady=8)
        self.start_btn = ttk.Button(action_frame, text="START UNIQUE ENGINE", command=self.start_run)
        self.start_btn.pack(side="left", padx=4)
        self.stop_btn = ttk.Button(action_frame, text="STOP", command=self.stop_run, state="disabled")
        self.stop_btn.pack(side="left", padx=4)
        ttk.Button(action_frame, text="OPEN OUTPUT DIR", command=self.open_output_dir).pack(side="left", padx=4)
        ttk.Button(action_frame, text="OPEN LAST RUN", command=self.open_last_run).pack(side="left", padx=4)

        status_frame = ttk.LabelFrame(top_frame, text="Status", padding=10)
        status_frame.pack(fill="x", pady=4)
        ttk.Label(status_frame, textvariable=self.status_var, font=("Arial", 10, "bold")).pack(anchor="w")
        ttk.Label(status_frame, textvariable=self.progress_var).pack(anchor="w")
        ttk.Label(status_frame, textvariable=self.best_var).pack(anchor="w")
        ttk.Label(status_frame, textvariable=self.elapsed_var).pack(anchor="w")
        ttk.Label(status_frame, textvariable=self.run_dir_var).pack(anchor="w")
        self.progress_bar = ttk.Progressbar(status_frame, orient="horizontal", mode="determinate", maximum=100)
        self.progress_bar.pack(fill="x", pady=(4, 0))

        log_frame = ttk.LabelFrame(top_frame, text="Run Log", padding=10)
        log_frame.pack(fill="both", expand=True, pady=4)
        self.log_text = ScrolledText(log_frame, height=16, wrap="word")
        self.log_text.pack(fill="both", expand=True)
        self._bind_mousewheel(self.log_text)

    def _on_main_content_configure(self, _event) -> None:
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))

    def _on_main_canvas_configure(self, event) -> None:
        self.main_canvas.itemconfigure(self.main_canvas_window, width=event.width)

    def _on_mousewheel(self, event) -> None:
        if event.num == 4:
            self.main_canvas.yview_scroll(-1, "units")
            return
        if event.num == 5:
            self.main_canvas.yview_scroll(1, "units")
            return

        delta = int(-1 * (event.delta / 120)) if event.delta else 0
        if delta != 0:
            self.main_canvas.yview_scroll(delta, "units")

    def _bind_mousewheel(self, widget: Any) -> None:
        widget.bind("<MouseWheel>", self._on_mousewheel)
        widget.bind("<Button-4>", self._on_mousewheel)
        widget.bind("<Button-5>", self._on_mousewheel)

    @staticmethod
    def _grid_row(parent: ttk.LabelFrame, row: int, label: str, var: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=3)
        ttk.Entry(parent, textvariable=var, width=92).grid(row=row, column=1, sticky="we", padx=5, pady=3)
        parent.grid_columnconfigure(1, weight=1)

    def _append_log(self, text: str) -> None:
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")

    def _on_quality_preset_selected(self, _event=None) -> None:
        self._sync_quality_preset_fields(log_change=True)

    def _sync_quality_preset_fields(self, log_change: bool = True) -> None:
        preset_key = self.quality_preset_var.get().strip().upper()
        preset = QUALITY_PRESETS.get(preset_key)
        if not preset:
            return

        self.min_return_var.set(str(preset.get("min_return_pct", self.min_return_var.get())))
        self.max_drawdown_var.set(str(preset.get("max_drawdown_pct", self.max_drawdown_var.get())))
        self.min_win_rate_var.set(str(preset.get("min_win_rate_pct", self.min_win_rate_var.get())))
        self.min_trades_var.set(str(preset.get("min_trades", self.min_trades_var.get())))
        self.min_test_return_var.set(str(preset.get("min_test_return_pct", self.min_test_return_var.get())))
        self.max_test_drawdown_var.set(str(preset.get("max_test_drawdown_pct", self.max_test_drawdown_var.get())))
        self.min_test_win_rate_var.set(str(preset.get("min_test_win_rate_pct", self.min_test_win_rate_var.get())))
        self.min_test_trades_var.set(str(preset.get("min_test_trades", self.min_test_trades_var.get())))

        if log_change:
            self._append_log(
                "Applied preset {preset}: train >= {ret:.1f}% return, <= {dd:.1f}% DD, >= {win:.1f}% win, >= {trades} trades.".format(
                    preset=preset_key,
                    ret=float(preset.get("min_return_pct", 0.0)),
                    dd=float(preset.get("max_drawdown_pct", 0.0)),
                    win=float(preset.get("min_win_rate_pct", 0.0)),
                    trades=int(preset.get("min_trades", 0)),
                )
            )

    def apply_preset_a(self) -> None:
        self.quality_preset_var.set("A")
        self._sync_quality_preset_fields(log_change=True)

    def apply_preset_b(self) -> None:
        self.quality_preset_var.set("B")
        self._sync_quality_preset_fields(log_change=True)

    def _build_run_config(self) -> Dict[str, Any]:
        return {
            "quality_preset": self.quality_preset_var.get().strip(),
            "blueprints": self.blueprints_var.get().strip(),
            "symbols": self.symbols_var.get().strip(),
            "timeframes": self.timeframes_var.get().strip(),
            "start_date": self.start_date_var.get().strip(),
            "end_date": self.end_date_var.get().strip(),
            "candidate_count_per_blueprint": int(self.candidate_per_blueprint_var.get().strip()),
            "workers": int(self.workers_var.get().strip()),
            "top_k": int(self.top_k_var.get().strip()),
            "publish_top_n": int(self.publish_top_n_var.get().strip()),
            "min_return_pct": float(self.min_return_var.get().strip()),
            "max_drawdown_pct": float(self.max_drawdown_var.get().strip()),
            "min_win_rate_pct": float(self.min_win_rate_var.get().strip()),
            "min_trades": int(self.min_trades_var.get().strip()),
            "min_test_return_pct": float(self.min_test_return_var.get().strip()),
            "max_test_drawdown_pct": float(self.max_test_drawdown_var.get().strip()),
            "min_test_win_rate_pct": float(self.min_test_win_rate_var.get().strip()),
            "min_test_trades": int(self.min_test_trades_var.get().strip()),
            "enable_train_test_split": bool(self.enable_split_var.get()),
            "train_ratio": float(self.train_ratio_var.get().strip()),
            "enable_gate_fallback": bool(self.enable_fallback_var.get()),
            "data_dir": self.data_dir_var.get().strip(),
            "output_dir": self.output_dir_var.get().strip(),
            "search_space_path": self.search_space_var.get().strip(),
            "seed": int(self.seed_var.get().strip()),
            "enable_mtf_context_filter": bool(self.enable_mtf_var.get()),
            "enable_regime_filter": bool(self.enable_regime_var.get()),
            "context_fast_period": int(self.context_fast_var.get().strip()),
            "context_slow_period": int(self.context_slow_var.get().strip()),
            "regime_symbol": self.regime_symbol_var.get().strip(),
        }

    def start_run(self) -> None:
        if self.worker_thread is not None and self.worker_thread.is_alive():
            messagebox.showinfo("Unique Engine", "Run is already in progress.")
            return

        try:
            config = self._build_run_config()
        except Exception as exc:
            messagebox.showerror("Unique Engine", f"Invalid settings: {exc}")
            return

        self.stop_event = threading.Event()
        self.run_started_ts = time.time()
        self.progress_bar["value"] = 0
        self.status_var.set("RUNNING")
        self.progress_var.set("Progress: 0/0")
        self.best_var.set("Best Score: -")
        self.elapsed_var.set("Elapsed: 0s")
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self._append_log(
            "Starting unique advanced engine (preset={preset}, blueprints={blueprints})...".format(
                preset=config.get("quality_preset", "custom"),
                blueprints=config.get("blueprints", "all"),
            )
        )

        def _progress(done: int, total: int, best_score: float, row: Dict[str, Any]) -> None:
            self.root.after(0, lambda: self._on_progress(done, total, best_score, row))

        def _work() -> None:
            result = None
            error_text = ""
            try:
                result = run_unique_engine(config_override=config, progress_callback=_progress, stop_event=self.stop_event)
            except Exception as exc:
                error_text = str(exc)
            self.root.after(0, lambda: self._on_finished(result, error_text))

        self.worker_thread = threading.Thread(target=_work, daemon=True)
        self.worker_thread.start()

    def _on_progress(self, done: int, total: int, best_score: float, row: Dict[str, Any]) -> None:
        percent = 0.0 if total <= 0 else min(100.0, (done / total) * 100.0)
        self.progress_bar["value"] = percent
        self.progress_var.set(f"Progress: {done}/{total}")
        self.best_var.set(f"Best Score: {best_score:.4f}")
        elapsed = 0 if self.run_started_ts is None else int(time.time() - self.run_started_ts)
        self.elapsed_var.set(f"Elapsed: {elapsed}s")

        metrics = row.get("metrics", {})
        self._append_log(
            "blueprint={bp} candidate #{idx} score={score:.4f} pass={passed} return={ret:.2f}% dd={dd:.2f}% trades={trades}".format(
                bp=str(row.get("blueprint_id", "")),
                idx=int(row.get("index", -1)),
                score=float(row.get("score", 0.0)),
                passed=bool(row.get("passed", False)),
                ret=float(metrics.get("total_return", 0.0)),
                dd=float(metrics.get("max_drawdown", 0.0)),
                trades=int(metrics.get("total_trades", 0)),
            )
        )

    def _on_finished(self, result: Dict[str, Any] | None, error_text: str) -> None:
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

        if error_text:
            self.status_var.set("FAILED")
            self._append_log(f"Run failed: {error_text}")
            messagebox.showerror("Unique Engine", f"Run failed: {error_text}")
            return

        if result is None:
            self.status_var.set("FAILED")
            self._append_log("Run failed: unknown error")
            messagebox.showerror("Unique Engine", "Run failed: unknown error")
            return

        self.last_run_dir = str(result.get("run_dir", ""))
        self.run_dir_var.set(f"Run Dir: {self.last_run_dir or '-'}")

        stopped = bool(result.get("stopped", False))
        self.status_var.set("STOPPED" if stopped else "COMPLETE")
        self._append_log(
            "Run complete. evaluated={evaluated}/{requested} elapsed={elapsed:.2f}s".format(
                evaluated=int(result.get("evaluated", 0)),
                requested=int(result.get("requested_candidates", 0)),
                elapsed=float(result.get("elapsed_seconds", 0.0)),
            )
        )

        fallback = result.get("fallback", {}) or {}
        self._append_log(
            "Fallback: triggered={triggered} reason={reason}".format(
                triggered=bool(fallback.get("triggered", False)),
                reason=str(fallback.get("reason", "")),
            )
        )

        published = result.get("published_strategy_files", []) or []
        if published:
            self._append_log(f"Published strategy files: {len(published)}")
            for path in published:
                self._append_log(f"  {path}")

        top_results = result.get("top_results", []) or []
        if top_results:
            best = top_results[0]
            metrics = best.get("metrics", {})
            self._append_log(
                "Best result: blueprint={bp} score={score:.4f} return={ret:.2f}% dd={dd:.2f}% win={win:.2f}% trades={trades}".format(
                    bp=str(best.get("blueprint_id", "")),
                    score=float(best.get("score", 0.0)),
                    ret=float(metrics.get("total_return", 0.0)),
                    dd=float(metrics.get("max_drawdown", 0.0)),
                    win=float(metrics.get("win_rate", 0.0)),
                    trades=int(metrics.get("total_trades", 0)),
                )
            )

    def stop_run(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
            self.status_var.set("STOPPING...")
            self._append_log("Stop requested.")

    def open_output_dir(self) -> None:
        path = self.output_dir_var.get().strip()
        if not path:
            return
        os.makedirs(path, exist_ok=True)
        try:
            os.startfile(path)
        except Exception:
            messagebox.showinfo("Unique Engine", f"Output dir: {path}")

    def open_last_run(self) -> None:
        if not self.last_run_dir:
            messagebox.showinfo("Unique Engine", "No run folder yet.")
            return
        try:
            os.startfile(self.last_run_dir)
        except Exception:
            messagebox.showinfo("Unique Engine", f"Run dir: {self.last_run_dir}")
