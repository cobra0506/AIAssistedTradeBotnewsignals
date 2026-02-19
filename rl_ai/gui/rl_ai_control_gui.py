import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


class RLAIControlGUI:
    def __init__(self, root):
        self.root = root
        self.process = None
        self.log_queue = queue.Queue()
        self.log_poll_ms = 200

        self._build_ui()
        self._set_defaults()
        self._refresh_mode_fields()
        self._poll_log_queue()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _project_root(self):
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    def _build_ui(self):
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill="both", expand=True)

        title = ttk.Label(container, text="RL AI Control Panel", font=("Arial", 12, "bold"))
        title.pack(anchor="w")

        self.status_var = tk.StringVar(value="IDLE")
        ttk.Label(container, textvariable=self.status_var, font=("Arial", 10, "bold")).pack(anchor="w", pady=(2, 8))

        form = ttk.LabelFrame(container, text="Run Settings", padding=10)
        form.pack(fill="x")

        mode_row = ttk.Frame(form)
        mode_row.pack(fill="x", pady=3)
        ttk.Label(mode_row, text="Run mode:", width=18).pack(side="left")
        self.mode_var = tk.StringVar(value="governed_train")
        self.mode_combo = ttk.Combobox(
            mode_row,
            textvariable=self.mode_var,
            values=("governed_train", "basic_train", "deep_train", "evaluate"),
            state="readonly",
            width=22,
        )
        self.mode_combo.pack(side="left", fill="x", expand=True)
        self.mode_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_mode_fields())

        self.data_csv_var = tk.StringVar()
        self._build_path_row(form, "Data CSV:", self.data_csv_var, self._browse_data_csv)

        episodes_row = ttk.Frame(form)
        episodes_row.pack(fill="x", pady=3)
        ttk.Label(episodes_row, text="Episodes:", width=18).pack(side="left")
        self.episodes_var = tk.StringVar(value="20")
        self.episodes_entry = ttk.Entry(episodes_row, textvariable=self.episodes_var, width=12)
        self.episodes_entry.pack(side="left")

        deep_algo_row = ttk.Frame(form)
        deep_algo_row.pack(fill="x", pady=3)
        ttk.Label(deep_algo_row, text="Deep algorithm:", width=18).pack(side="left")
        self.deep_algorithm_var = tk.StringVar(value="ppo")
        self.deep_algorithm_combo = ttk.Combobox(
            deep_algo_row,
            textvariable=self.deep_algorithm_var,
            values=("ppo", "a2c", "dqn"),
            state="readonly",
            width=12,
        )
        self.deep_algorithm_combo.pack(side="left")

        timesteps_row = ttk.Frame(form)
        timesteps_row.pack(fill="x", pady=3)
        ttk.Label(timesteps_row, text="Deep timesteps:", width=18).pack(side="left")
        self.deep_timesteps_var = tk.StringVar(value="200000")
        self.deep_timesteps_entry = ttk.Entry(timesteps_row, textvariable=self.deep_timesteps_var, width=12)
        self.deep_timesteps_entry.pack(side="left")

        self.model_out_var = tk.StringVar(value="rl_ai/models/governed_policy.npz")
        self.model_out_entry = self._build_path_row(form, "Model output:", self.model_out_var, self._browse_model_out)

        self.model_path_var = tk.StringVar(value="rl_ai/models/governed_policy.npz")
        self.model_path_entry = self._build_path_row(form, "Model path:", self.model_path_var, self._browse_model_path)

        self.report_dir_var = tk.StringVar(value="rl_ai/reports/governed")
        self.report_dir_entry = self._build_path_row(form, "Report folder:", self.report_dir_var, self._browse_report_dir)

        self.registry_path_var = tk.StringVar(value="rl_ai/models/model_registry.json")
        self.registry_path_entry = self._build_path_row(
            form,
            "Registry path:",
            self.registry_path_var,
            self._browse_registry_path,
        )

        options_row = ttk.Frame(form)
        options_row.pack(fill="x", pady=(4, 0))
        self.skip_walk_forward_var = tk.BooleanVar(value=False)
        self.skip_promotion_var = tk.BooleanVar(value=False)
        self.skip_walk_forward_check = ttk.Checkbutton(
            options_row,
            text="Skip walk-forward",
            variable=self.skip_walk_forward_var,
        )
        self.skip_walk_forward_check.pack(side="left", padx=(0, 10))
        self.skip_promotion_check = ttk.Checkbutton(
            options_row,
            text="Skip promotion",
            variable=self.skip_promotion_var,
        )
        self.skip_promotion_check.pack(side="left")

        buttons = ttk.Frame(container)
        buttons.pack(fill="x", pady=10)
        self.start_btn = ttk.Button(buttons, text="START RUN", command=self.start_run)
        self.start_btn.pack(side="left", padx=(0, 6))
        self.stop_btn = ttk.Button(buttons, text="STOP RUN", command=self.stop_run, state="disabled")
        self.stop_btn.pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="OPEN REPORTS", command=self.open_reports_dir).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="OPEN MODELS", command=self.open_models_dir).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="CLEAR LOG", command=self.clear_log).pack(side="right")

        log_frame = ttk.LabelFrame(container, text="Run Log", padding=8)
        log_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_frame, height=18, wrap="word")
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scroll.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=log_scroll.set)

        note = ttk.Label(
            container,
            text="Tip: use deep_train + PPO for stronger models (requires torch/stable-baselines3/gymnasium).",
            font=("Arial", 9),
        )
        note.pack(anchor="w", pady=(8, 0))

    def _build_path_row(self, parent, label, var, browse_cmd):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, width=18).pack(side="left")
        entry = ttk.Entry(row, textvariable=var)
        entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(row, text="Browse", command=browse_cmd, width=8).pack(side="left")
        return entry

    def _set_defaults(self):
        if self.data_csv_var.get().strip():
            return
        guessed = self._guess_data_csv()
        if guessed:
            self.data_csv_var.set(guessed)

    def _guess_data_csv(self):
        data_dir = os.path.join(self._project_root(), "data")
        if not os.path.isdir(data_dir):
            return ""
        csv_files = sorted(
            file_name
            for file_name in os.listdir(data_dir)
            if file_name.lower().endswith(".csv")
        )
        if not csv_files:
            return ""
        return os.path.join("data", csv_files[0])

    def _browse_data_csv(self):
        selected = filedialog.askopenfilename(
            title="Select Data CSV",
            initialdir=self._project_root(),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if selected:
            self.data_csv_var.set(self._to_project_relative(selected))

    def _browse_model_out(self):
        selected = filedialog.asksaveasfilename(
            title="Model output file",
            initialdir=os.path.join(self._project_root(), "rl_ai", "models"),
            defaultextension=".zip",
            filetypes=[
                ("Deep model", "*.zip"),
                ("NumPy model", "*.npz"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            self.model_out_var.set(self._to_project_relative(selected))

    def _browse_model_path(self):
        selected = filedialog.askopenfilename(
            title="Select trained model",
            initialdir=os.path.join(self._project_root(), "rl_ai", "models"),
            filetypes=[
                ("Any RL model", "*.zip;*.npz"),
                ("Deep model", "*.zip"),
                ("NumPy model", "*.npz"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            self.model_path_var.set(self._to_project_relative(selected))

    def _browse_report_dir(self):
        selected = filedialog.askdirectory(
            title="Select report folder",
            initialdir=os.path.join(self._project_root(), "rl_ai", "reports"),
        )
        if selected:
            self.report_dir_var.set(self._to_project_relative(selected))

    def _browse_registry_path(self):
        selected = filedialog.asksaveasfilename(
            title="Registry file",
            initialdir=os.path.join(self._project_root(), "rl_ai", "models"),
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if selected:
            self.registry_path_var.set(self._to_project_relative(selected))

    def _to_project_relative(self, path):
        project_root = self._project_root()
        abs_path = os.path.abspath(path)
        try:
            rel_path = os.path.relpath(abs_path, project_root)
            if not rel_path.startswith(".."):
                return rel_path
        except ValueError:
            pass
        return abs_path

    def _refresh_mode_fields(self):
        mode = self.mode_var.get().strip()
        is_evaluate = mode == "evaluate"
        is_governed = mode == "governed_train"
        is_deep = mode == "deep_train"

        self._set_state(self.model_out_entry, "disabled" if is_evaluate else "normal")
        self._set_state(self.model_path_entry, "normal" if is_evaluate else "disabled")
        self._set_state(self.registry_path_entry, "normal" if is_governed else "disabled")
        self._set_state(self.skip_walk_forward_check, "normal" if is_governed else "disabled")
        self._set_state(self.skip_promotion_check, "normal" if is_governed else "disabled")
        self._set_state(self.episodes_entry, "disabled" if is_deep else "normal")
        self._set_state(self.deep_algorithm_combo, "readonly" if is_deep else "disabled")
        self._set_state(self.deep_timesteps_entry, "normal" if is_deep else "disabled")

    @staticmethod
    def _set_state(widget, state):
        try:
            widget.configure(state=state)
        except Exception:
            pass

    def _validate_int(self, value, name):
        try:
            parsed = int(str(value).strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer.") from exc
        if parsed < 1:
            raise ValueError(f"{name} must be >= 1.")
        return parsed

    def _build_command(self):
        mode = self.mode_var.get().strip()
        data_csv = self.data_csv_var.get().strip()
        report_dir = self.report_dir_var.get().strip()

        if not data_csv:
            raise ValueError("Select a data CSV file.")

        if mode == "governed_train":
            episodes = self._validate_int(self.episodes_var.get(), "Episodes")
            command = [
                sys.executable,
                "-m",
                "rl_ai.01_library_based.governed_training",
                "--data-csv",
                data_csv,
                "--model-out",
                self.model_out_var.get().strip() or "rl_ai/models/governed_policy.npz",
                "--episodes",
                str(episodes),
                "--report-dir",
                report_dir or "rl_ai/reports/governed",
                "--registry-path",
                self.registry_path_var.get().strip() or "rl_ai/models/model_registry.json",
            ]
            if self.skip_walk_forward_var.get():
                command.append("--skip-walk-forward")
            if self.skip_promotion_var.get():
                command.append("--skip-promotion")
            return command

        if mode == "basic_train":
            episodes = self._validate_int(self.episodes_var.get(), "Episodes")
            return [
                sys.executable,
                "-m",
                "rl_ai.01_library_based.train_pipeline",
                "--data-csv",
                data_csv,
                "--model-out",
                self.model_out_var.get().strip() or "rl_ai/models/phase2_policy.npz",
                "--episodes",
                str(episodes),
                "--report-dir",
                report_dir or "rl_ai/reports/latest",
            ]

        if mode == "deep_train":
            timesteps = self._validate_int(self.deep_timesteps_var.get(), "Deep timesteps")
            algorithm = self.deep_algorithm_var.get().strip().lower()
            if algorithm not in {"ppo", "a2c", "dqn"}:
                raise ValueError("Deep algorithm must be ppo, a2c, or dqn.")
            return [
                sys.executable,
                "-m",
                "rl_ai.01_library_based.deep_train_pipeline",
                "--data-csv",
                data_csv,
                "--algorithm",
                algorithm,
                "--timesteps",
                str(timesteps),
                "--model-out",
                self.model_out_var.get().strip() or "rl_ai/models/deep_policy.zip",
                "--report-dir",
                report_dir or "rl_ai/reports/deep_latest",
            ]

        if mode == "evaluate":
            episodes = self._validate_int(self.episodes_var.get(), "Episodes")
            model_path = self.model_path_var.get().strip()
            if not model_path:
                raise ValueError("Select a trained model file for evaluate mode.")
            command = [
                sys.executable,
                "-m",
                "rl_ai.01_library_based.evaluate_pipeline",
                "--data-csv",
                data_csv,
                "--model-path",
                model_path,
                "--episodes",
                str(episodes),
            ]
            if report_dir:
                command.extend(["--report-dir", report_dir])
            return command

        raise ValueError("Unknown RL run mode selected.")

    def start_run(self):
        if self.process and self.process.poll() is None:
            messagebox.showinfo("RL AI", "A run is already active.")
            return

        try:
            command = self._build_command()
        except ValueError as e:
            messagebox.showerror("Invalid Settings", str(e))
            return

        project_root = self._project_root()
        self._append_log(f"$ {' '.join(command)}")
        try:
            self.process = subprocess.Popen(
                command,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            self.process = None
            messagebox.showerror("RL AI", f"Failed to start RL run: {e}")
            return

        self.status_var.set("RUNNING")
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        threading.Thread(target=self._read_process_output, daemon=True).start()
        threading.Thread(target=self._wait_for_process, daemon=True).start()

    def stop_run(self):
        if not self.process or self.process.poll() is not None:
            messagebox.showinfo("RL AI", "No active run to stop.")
            return

        self.status_var.set("STOPPING")
        try:
            self.process.terminate()
            self._append_log("Requested stop.")
        except Exception as e:
            self._append_log(f"Stop failed: {e}")

    def _read_process_output(self):
        if not self.process or not self.process.stdout:
            return
        try:
            for line in self.process.stdout:
                self.log_queue.put(("line", line.rstrip()))
        except Exception as e:
            self.log_queue.put(("line", f"[output read error] {e}"))

    def _wait_for_process(self):
        if not self.process:
            return
        exit_code = self.process.wait()
        self.log_queue.put(("exit", exit_code))

    def _poll_log_queue(self):
        while True:
            try:
                item_type, payload = self.log_queue.get_nowait()
            except queue.Empty:
                break

            if item_type == "line":
                self._append_log(str(payload))
            elif item_type == "exit":
                exit_code = int(payload)
                if exit_code == 0:
                    self.status_var.set("COMPLETED")
                    self._append_log("Run completed successfully.")
                else:
                    self.status_var.set("FAILED")
                    self._append_log(f"Run failed with exit code {exit_code}.")
                self.start_btn.configure(state="normal")
                self.stop_btn.configure(state="disabled")
                self.process = None

        self.root.after(self.log_poll_ms, self._poll_log_queue)

    def _append_log(self, text):
        self.log_text.insert("end", f"{text}\n")
        self.log_text.see("end")

    def clear_log(self):
        self.log_text.delete("1.0", "end")

    def open_reports_dir(self):
        report_dir = self.report_dir_var.get().strip() or os.path.join("rl_ai", "reports")
        self._open_path(report_dir)

    def open_models_dir(self):
        self._open_path(os.path.join("rl_ai", "models"))

    def _open_path(self, path):
        project_root = self._project_root()
        target = path if os.path.isabs(path) else os.path.join(project_root, path)
        os.makedirs(target, exist_ok=True)
        try:
            os.startfile(target)
        except Exception:
            messagebox.showinfo("Path", target)

    def _on_close(self):
        if self.process and self.process.poll() is None:
            should_close = messagebox.askyesno(
                "RL AI",
                "A run is active. Stop it and close this window?",
            )
            if not should_close:
                return
            try:
                self.process.terminate()
            except Exception:
                pass
        self.root.destroy()
