# simple_strategy/gui_monitor.py - Dynamic GUI for Simple Strategy Backtester
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import sys
import threading  
from datetime import datetime, timedelta
from pathlib import Path


# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import strategy registry
from strategies.strategy_registry import StrategyRegistry
from simple_strategy.trading.parameter_manager import ParameterManager

class SimpleStrategyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Dynamic Strategy Backtester")
        self.root.geometry("900x700")
        
        # Initialize strategy registry
        try:
            self.strategy_registry = StrategyRegistry()
            self.strategies = self.strategy_registry.get_all_strategies()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load strategy registry: {e}")
            self.strategies = {}
        
        # Initialize ParameterManager - ADD THIS
        self.param_manager = ParameterManager()

        # Initialize variables
        self.current_strategy = None
        self.param_widgets = {}
        
        self.create_widgets()

    def load_optimized_parameters(self, strategy_name):
        """Load optimized parameters if they exist, otherwise use defaults"""
        # Get optimized parameters
        optimized_params = self.param_manager.get_parameters(strategy_name)
        
        # Return the parameters as-is (including last_optimized)
        return optimized_params if optimized_params else None
    
    def create_widgets(self):
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_strategy_tab()
        self.create_backtest_tab()
        self.create_results_tab()
        
        # Status bar
        self.create_status_bar()
    
    def create_strategy_tab(self):
        # Strategy Configuration Tab
        strategy_frame = ttk.Frame(self.notebook)
        self.notebook.add(strategy_frame, text="Strategy Configuration")
        
        # Strategy Selection
        select_frame = ttk.LabelFrame(strategy_frame, text="Strategy Selection", padding=10)
        select_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(select_frame, text="Select Strategy:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        
        # Strategy dropdown
        strategy_names = list(self.strategies.keys()) if self.strategies else ["No strategies found"]
        self.strategy_var = tk.StringVar(value=strategy_names[0] if strategy_names else "")
        self.strategy_combo = ttk.Combobox(select_frame, textvariable=self.strategy_var,
                                        values=strategy_names, state="readonly", width=40)
        self.strategy_combo.grid(row=0, column=1, padx=5, pady=5)
        self.strategy_combo.bind('<<ComboboxSelected>>', self.on_strategy_selected)
        
        # Strategy description
        self.description_text = tk.Text(select_frame, height=3, width=60)
        self.description_text.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        
        # Strategy Parameters - with scrolling
        self.param_frame = ttk.LabelFrame(strategy_frame, text="Strategy Parameters", padding=10)
        self.param_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create canvas and scrollbar for parameters
        self.param_canvas = tk.Canvas(self.param_frame)
        self.param_scrollbar = ttk.Scrollbar(self.param_frame, orient="vertical", command=self.param_canvas.yview)
        self.param_canvas.configure(yscrollcommand=self.param_scrollbar.set)
        self.param_canvas.pack(side="left", fill="both", expand=True)
        self.param_scrollbar.pack(side="right", fill="y")
        
        # Create frame inside canvas for parameters
        self.param_inner_frame = ttk.Frame(self.param_canvas)
        self.param_canvas_window = self.param_canvas.create_window((0, 0), window=self.param_inner_frame, anchor="nw")
        
        # Configure canvas scrollregion when inner frame changes
        self.param_inner_frame.bind("<Configure>", self._on_param_frame_configure)
        
        # Create a container frame OUTSIDE the canvas for the button and info text
        self.bottom_container = ttk.Frame(strategy_frame)
        self.bottom_container.pack(fill="x", padx=10, pady=5)
        
        # Create Strategy Button - now outside the canvas
        self.create_btn = ttk.Button(self.bottom_container, text="🔧 Create Strategy", command=self.create_strategy)
        self.create_btn.pack(pady=5)

        # Add optimization button that calls the same function as backtest tab
        self.optimize_strategy_btn = ttk.Button(self.bottom_container, text="🚀 Run Optimization", command=self.optimize_from_backtest_tab)
        self.optimize_strategy_btn.pack(pady=5)
        
        # Strategy Info - now outside the canvas
        self.strategy_info_text = tk.Text(self.bottom_container, height=5, width=70)
        self.strategy_info_text.pack(pady=5, fill="x", expand=True)
        
        # Bind mouse wheel scrolling
        self._bind_mouse_wheel()
        
        # Initialize with first strategy
        if strategy_names:
            self.on_strategy_selected()
    
    def optimize_from_backtest_tab(self):
        """Optimize parameters using current strategy parameters as starting point"""
        strategy_name = self.strategy_var.get()
        if not strategy_name:
            messagebox.showwarning("Warning", "Please select a strategy first")
            self.notebook.select(0)  # Switch to strategy tab
            return
        
        # Check if strategy is created first
        if not hasattr(self, 'current_strategy') or self.current_strategy is None:
            messagebox.showwarning("Warning", "Please create the strategy first before optimizing")
            self.notebook.select(0)  # Switch to strategy tab
            return
        
        # Create optimization window with scrolling
        opt_window = tk.Toplevel(self.root)
        opt_window.title(f"Optimize {strategy_name}")
        opt_window.geometry("500x600")  # Increased height to accommodate scrolling
        
        # Create a main frame with scrollbar
        main_frame = ttk.Frame(opt_window)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create canvas and scrollbar
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Create frame inside canvas for content
        content_frame = ttk.Frame(canvas)
        canvas_window = canvas.create_window((0, 0), window=content_frame, anchor="nw")
        
        # Optimization settings
        ttk.Label(content_frame, text="Optimization Settings", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Number of trials
        ttk.Label(content_frame, text="Number of Trials:").pack()
        trials_var = tk.StringVar(value="20")
        ttk.Entry(content_frame, textvariable=trials_var).pack()
        
        # Use current backtest settings
        ttk.Label(content_frame, text="Symbol:").pack()
        symbol_var = tk.StringVar(value=self.symbols_var.get())
        ttk.Entry(content_frame, textvariable=symbol_var).pack()
        
        ttk.Label(content_frame, text="Timeframe:").pack()
        timeframe_var = tk.StringVar(value=self.timeframes_var.get().rstrip('m'))
        ttk.Entry(content_frame, textvariable=timeframe_var).pack()
        
        ttk.Label(content_frame, text="Start Date:").pack()
        start_date_var = tk.StringVar(value=self.start_date_var.get())
        ttk.Entry(content_frame, textvariable=start_date_var).pack()
        
        ttk.Label(content_frame, text="End Date:").pack()
        end_date_var = tk.StringVar(value=self.end_date_var.get())
        ttk.Entry(content_frame, textvariable=end_date_var).pack()
        
        # Add info about current parameters
        info_frame = ttk.LabelFrame(content_frame, text="Current Strategy Parameters", padding=10)
        info_frame.pack(fill="x", padx=10, pady=10)
        
        # Display current parameters in a scrollable text widget
        params_text = tk.Text(info_frame, height=8, width=50, wrap="word")
        params_text.pack(side="left", fill="both", expand=True)
        
        params_scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=params_text.yview)
        params_scrollbar.pack(side="right", fill="y")
        params_text.config(yscrollcommand=params_scrollbar.set)
        
        # Insert current parameters
        current_params_text = "Current parameters:\n"
        for param_name, var in self.param_widgets.items():
            if hasattr(var, 'get'):
                current_params_text += f"  {param_name}: {var.get()}\n"
        
        params_text.insert("1.0", current_params_text)
        params_text.config(state="disabled")  # Make it read-only
        
        def run_optimization():
            try:
                # Debug: Show current parameters before optimization
                print("=== BEFORE OPTIMIZATION ===")
                self.debug_current_parameters()
                
                # Import optimizer
                from simple_strategy.optimization import BayesianOptimizer, ParameterSpace
                from simple_strategy.shared.data_feeder import DataFeeder
                
                # Create parameter space DYNAMICALLY based on strategy parameters
                param_space = ParameterSpace()
                
                # DYNAMIC parameter space creation - works with ANY strategy
                if strategy_name in self.strategies:
                    strategy_info = self.strategies[strategy_name]
                    strategy_params = strategy_info.get('parameters', {})
                    
                    print(f"🔧 Creating parameter space for {strategy_name} with parameters: {strategy_params}")
                    
                    # Add each parameter from the strategy's STRATEGY_PARAMETERS
                    for param_name, param_def in strategy_params.items():
                        param_type = param_def.get('type', 'int')
                        
                        if param_type == 'int':
                            min_val = param_def.get('min', 1)
                            max_val = param_def.get('max', 100)
                            step = 1
                            param_space.add_int(param_name, min_val, max_val, step=step)
                            print(f"   Added int parameter: {param_name} [{min_val}, {max_val}]")
                            
                        elif param_type == 'float':
                            min_val = param_def.get('min', 0.1)
                            max_val = param_def.get('max', 10.0)
                            param_space.add_float(param_name, min_val, max_val)
                            print(f"   Added float parameter: {param_name} [{min_val}, {max_val}]")
                            
                        elif param_type == 'str' and 'options' in param_def:
                            options = param_def['options']
                            param_space.add_categorical(param_name, options)
                            print(f"   Added categorical parameter: {param_name} {options}")
                            
                        else:
                            # Default to int for unknown types
                            min_val = param_def.get('min', 1)
                            max_val = param_def.get('max', 100)
                            param_space.add_int(param_name, min_val, max_val)
                            print(f"   Added default int parameter: {param_name} [{min_val}, {max_val}]")
                else:
                    # Fallback if strategy not found
                    param_space.add_int('period', 5, 50, step=1)
                    param_space.add_float('threshold', 0.1, 5.0)
                    print(f"   Using fallback parameters for unknown strategy")
                
                # Show progress
                progress_label = ttk.Label(content_frame, text="🚀 Starting optimization...")
                progress_label.pack(pady=5)
                opt_window.update()
                
                # Get symbols (support multiple symbols)
                symbols_input = symbol_var.get()
                if ',' in symbols_input:
                    symbols_list = [s.strip() for s in symbols_input.split(',') if s.strip()]
                else:
                    symbols_list = [symbols_input]
                
                # Run optimization
                data_feeder = DataFeeder(data_dir=self.data_dir_var.get())
                optimizer = BayesianOptimizer(
                    data_feeder=data_feeder,
                    study_name=f'{strategy_name}_optimization',
                    direction='maximize',
                    n_trials=int(trials_var.get()),
                    timeout=3600
                )
                
                best_params, best_score = optimizer.optimize(
                    strategy_name=strategy_name,
                    parameter_space=param_space,
                    symbols=symbols_list,
                    timeframes=[timeframe_var.get()],
                    start_date=start_date_var.get(),
                    end_date=end_date_var.get(),
                    metric='sharpe_ratio'
                )
                
                # FIXED: Better string formatting with error handling
                try:
                    if best_score == float('-inf'):
                        score_text = "No valid strategies found"
                    else:
                        score_text = f"{best_score:.4f}"
                except Exception:
                    score_text = str(best_score)
                
                # FIXED: Safe string building without f-strings
                result_msg = "🎉 Optimization Complete!\n\n"
                result_msg += "📊 Strategy: " + str(strategy_name) + "\n"
                result_msg += "📈 Best Sharpe Ratio: " + score_text + "\n\n"
                result_msg += "⚙️ Best Parameters:\n"
                for param, value in best_params.items():
                    result_msg += "   " + str(param) + ": " + str(value) + "\n"
                
                result_msg += "\n📊 Optimized over " + str(len(symbols_list)) + " symbols: " + ", ".join(symbols_list)
                result_msg += "\n✅ Parameters updated in Strategy Configuration tab!"
                result_msg += "\n💡 Click 'Create Strategy' then 'Run Backtest' to test the optimized strategy"
                
                messagebox.showinfo("Optimization Results", result_msg)
                
                # FIXED: Refresh the entire parameter display to show optimized values
                self.on_strategy_selected()
                
                # Debug: Show parameters after update
                print("=== AFTER PARAMETER UPDATE ===")
                self.debug_current_parameters()
                
                # Save winning summary
                try:
                    optimizer.save_optimization_summary()
                except Exception as save_error:
                    print(f"Warning: Could not save optimization summary: {save_error}")
                
                # Switch to strategy tab to show updated parameters
                self.notebook.select(0)  # Switch to strategy configuration tab
                
                # Close the optimization popup
                opt_window.destroy()
                
                # FIXED: Bring the backtest window to front and set focus
                self.root.deiconify()  # Ensure window is not minimized
                self.root.lift()        # Bring window to front
                self.root.focus_force() # Set focus to the window
                self.root.grab_set()   # Grab focus to ensure it's the active window
                
            except Exception as e:
                # FIXED: Better error handling
                error_msg = "Optimization failed: " + str(e)
                messagebox.showerror("Error", error_msg)
                import traceback
                traceback.print_exc()
        
        # Add run optimization button at the bottom
        run_button = ttk.Button(content_frame, text="🚀 Run Optimization", command=run_optimization)
        run_button.pack(pady=20)
        
        # Update canvas scrollregion when content is added
        def update_scrollregion(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        content_frame.bind("<Configure>", update_scrollregion)
        update_scrollregion()  # Initial update
    

    def _on_param_frame_configure(self, event=None):
        """Update the scrollregion to encompass the inner frame"""
        self.param_canvas.configure(scrollregion=self.param_canvas.bbox("all"))
    
    def _bind_mouse_wheel(self):
        """Bind mouse wheel scrolling to the parameter canvas"""
        def _on_mousewheel(event):
            # Check if mouse is over the parameter canvas
            if self.param_canvas.winfo_containing(event.x, event.y):
                # Scroll the canvas
                self.param_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        # Bind to all mouse wheel events
        self.param_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        # For Linux
        self.param_canvas.bind_all("<Button-4>", lambda e: self.param_canvas.yview_scroll(-1, "units"))
        self.param_canvas.bind_all("<Button-5>", lambda e: self.param_canvas.yview_scroll(1, "units"))
    
    def on_strategy_selected(self, event=None):
        """Handle strategy selection change"""
        strategy_name = self.strategy_var.get()
        if not strategy_name:
            return
        
        # Get strategy info
        strategy_info = self.strategies.get(strategy_name)
        if not strategy_info:
            return
        
        # Clear previous parameter widgets
        for widget in self.param_inner_frame.winfo_children():
            widget.destroy()
        self.param_widgets.clear()
        
        # Load optimized parameters if available
        optimized_params = self.load_optimized_parameters(strategy_name)
        
        # Get default parameters from strategy info
        parameters_def = strategy_info.get('parameters', {})
        default_params = {}
        for param_name, param_info in parameters_def.items():
            default_params[param_name] = param_info.get('default', 0)
        
        # If no default parameters found, try alternative approach
        if not default_params:
            default_params = strategy_info.get('default_params', {})
        
        # Use optimized parameters if available, otherwise use defaults
        current_params = optimized_params if optimized_params else default_params
        
        # Store current parameters for later use
        self.current_params = current_params.copy()
        
        # Update description
        self.description_text.delete(1.0, tk.END)
        description = strategy_info.get('description', 'No description available')
        self.description_text.insert(tk.END, description)
        
        # Create parameter widgets with SLIDERS
        row = 0
        for param_name, param_value in current_params.items():
            # Parameter name label
            ttk.Label(self.param_inner_frame, text=f"{param_name}:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
            
            # Get parameter range from strategy definition
            param_def = parameters_def.get(param_name, {})
            min_val = param_def.get('min', 0)
            max_val = param_def.get('max', 100)
            
            if isinstance(param_value, bool):
                # Boolean parameter - use checkbox
                var = tk.BooleanVar(value=param_value)
                widget = ttk.Checkbutton(self.param_inner_frame, variable=var)
                widget.var = var
            elif isinstance(param_value, (int, float)):
                # Numeric parameter - use SLIDER with clickable value display
                slider_frame = ttk.Frame(self.param_inner_frame)
                slider_frame.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
                
                # Create variable for this parameter
                if isinstance(param_value, int):
                    var = tk.IntVar(value=int(param_value))
                    widget = ttk.Scale(slider_frame, from_=min_val, to=max_val, 
                                    variable=var, orient="horizontal", length=200)
                else:
                    var = tk.DoubleVar(value=float(param_value))
                    widget = ttk.Scale(slider_frame, from_=min_val, to=max_val, 
                                    variable=var, orient="horizontal", length=200)
                
                widget.pack(side="left", fill="x", expand=True)
                
                # Add clickable value label that opens input dialog
                value_label = ttk.Label(slider_frame, text=str(param_value), width=10, 
                                    cursor="hand2", foreground="blue")
                value_label.pack(side="right", padx=5)
                
                # Create a custom update function for this specific parameter
                def make_update_func(label, variable):
                    def update_value_label(x):
                        label.config(text=f"{variable.get():.2f}" if isinstance(variable.get(), float) else str(variable.get()))
                    return update_value_label
                
                # Create a custom click function for this specific parameter
                def make_click_func(param_name, variable, label, min_val, max_val):
                    def on_label_click(event):
                        # Create a dialog to input custom value
                        dialog = tk.Toplevel(self.root)
                        dialog.title(f"Set {param_name}")
                        dialog.geometry("300x150")
                        dialog.transient(self.root)
                        dialog.grab_set()
                        
                        ttk.Label(dialog, text=f"Enter value for {param_name} ({min_val} - {max_val}):").pack(pady=10)
                        
                        entry_var = tk.StringVar(value=str(variable.get()))
                        entry = ttk.Entry(dialog, textvariable=entry_var)
                        entry.pack(pady=5, padx=20, fill="x")
                        entry.select_range(0, tk.END)
                        entry.focus()
                        
                        def apply_value():
                            try:
                                new_value = float(entry_var.get())
                                if min_val <= new_value <= max_val:
                                    variable.set(new_value)
                                    label.config(text=f"{new_value:.2f}" if isinstance(new_value, float) else str(new_value))
                                    dialog.destroy()
                                else:
                                    messagebox.showerror("Error", f"Value must be between {min_val} and {max_val}")
                            except ValueError:
                                messagebox.showerror("Error", "Please enter a valid number")
                        
                        button_frame = ttk.Frame(dialog)
                        button_frame.pack(pady=10)
                        
                        ttk.Button(button_frame, text="Apply", command=apply_value).pack(side="left", padx=5)
                        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side="left", padx=5)
                        
                        # Bind Enter key to apply
                        entry.bind('<Return>', lambda e: apply_value())
                    
                    return on_label_click
                
                # Set up the slider update and label click
                widget.config(command=make_update_func(value_label, var))
                value_label.bind('<Button-1>', make_click_func(param_name, var, value_label, min_val, max_val))
                
            else:
                # String parameter - use entry
                widget = ttk.Entry(self.param_inner_frame, width=15)
                widget.insert(0, str(param_value))
                var = widget  # For string parameters, the widget itself is the var
            
            self.param_widgets[param_name] = var
            row += 1
        
        # Update strategy info text
        self.strategy_info_text.delete(1.0, tk.END)
        info_text = f"Strategy: {strategy_name}\n\n"
        
        if optimized_params:
            info_text += "✅ Using Optimized Parameters\n"
            info_text += f"Last Optimized: {self.param_manager.get_parameters(strategy_name).get('last_optimized', 'Unknown')}\n\n"
        else:
            info_text += "⚠️ Using Default Parameters (Not Optimized Yet)\n\n"
        
        for param, value in current_params.items():
            info_text += f"{param}: {value}\n"
        
        self.strategy_info_text.insert(tk.END, info_text)
        
        # Update canvas scrollregion
        self.param_inner_frame.update_idletasks()
        self.param_canvas.configure(scrollregion=self.param_canvas.bbox("all"))
    
    def update_parameters(self, parameters):
        """Update parameter widgets based on strategy parameters"""
        # Clear existing parameter widgets (except buttons and info text)
        for widget in self.param_inner_frame.winfo_children():
            if widget.grid_info() and widget.grid_info()['row'] >= 0 and widget.grid_info()['row'] < 100:
                widget.destroy()
        
        self.param_widgets.clear()
        
        row = 0
        for param_name, param_info in parameters.items():
            # Parameter label
            label_text = f"{param_name.replace('_', ' ').title()}"
            if 'description' in param_info:
                label_text += f"\n({param_info['description']})"
            
            ttk.Label(self.param_inner_frame, text=label_text).grid(row=row, column=0, sticky="w", padx=5, pady=5)
            
            # Parameter input based on type
            default_value = param_info.get('default', 0)
            
            if param_info.get('type') == 'int':
                var = tk.IntVar(value=default_value)
                min_val = param_info.get('min', 1)
                max_val = param_info.get('max', 100)
                widget = ttk.Spinbox(self.param_inner_frame, from_=min_val, to=max_val, 
                                   textvariable=var, width=15)
            elif param_info.get('type') == 'float':
                var = tk.DoubleVar(value=default_value)
                min_val = param_info.get('min', 0.1)
                max_val = param_info.get('max', 10.0)
                widget = ttk.Spinbox(self.param_inner_frame, from_=min_val, to=max_val, 
                                   textvariable=var, width=15, increment=0.1)
            elif param_info.get('type') == 'str' and 'options' in param_info:
                var = tk.StringVar(value=default_value)
                widget = ttk.Combobox(self.param_inner_frame, textvariable=var, 
                                    values=param_info['options'], state="readonly", width=20)
            else:  # string or other
                var = tk.StringVar(value=str(default_value))
                widget = ttk.Entry(self.param_inner_frame, textvariable=var, width=20)
            
            widget.grid(row=row, column=1, padx=5, pady=5)
            self.param_widgets[param_name] = var
            row += 1
        
        # Update canvas scrollregion after adding all parameters
        self.param_inner_frame.update_idletasks()
        self._on_param_frame_configure()
    
    def create_backtest_tab(self):
        # Backtest Configuration Tab
        backtest_frame = ttk.Frame(self.notebook)
        self.notebook.add(backtest_frame, text="Backtest Configuration")
        
        # Data Directory
        dir_frame = ttk.LabelFrame(backtest_frame, text="Data Directory", padding=10)
        dir_frame.pack(fill="x", padx=10, pady=5)
        
        self.data_dir_var = tk.StringVar(value="data")
        ttk.Entry(dir_frame, textvariable=self.data_dir_var, width=50).pack(side="left", padx=5)
        ttk.Button(dir_frame, text="Browse", command=self.browse_data_dir).pack(side="left", padx=5)
        
        # Symbols and Timeframes
        config_frame = ttk.LabelFrame(backtest_frame, text="Backtest Configuration", padding=10)
        config_frame.pack(fill="both", expand=True, padx=10, pady=5)

        ttk.Button(config_frame, text="🔍 Check Data Files", command=self.check_data_files).grid(row=5, column=0, columnspan=2, pady=5)
        
        # Symbols
        ttk.Label(config_frame, text="Symbols (comma-separated):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.symbols_var = tk.StringVar(value="BNBUSDT, XRPUSDT, ADAUSDT, DOTUSDT, ATOMUSDT, ALGOUSDT, VETUSDT, ICPUSDT, FILUSDT, AAVEUSDT, COMPUSDT, CRVUSDT, SNXUSDT")
        ttk.Entry(config_frame, textvariable=self.symbols_var, width=40).grid(row=0, column=1, padx=5, pady=5)
        
        # Timeframes
        ttk.Label(config_frame, text="Timeframes (comma-separated):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.timeframes_var = tk.StringVar(value="5m")
        ttk.Entry(config_frame, textvariable=self.timeframes_var, width=40).grid(row=1, column=1, padx=5, pady=5)
        
        # Date Range
        today = datetime.today().date()
        start_date_default = today - timedelta(days=7)
        end_date_default = today

        ttk.Label(config_frame, text="Start Date:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.start_date_var = tk.StringVar(value=start_date_default.strftime("%Y-%m-%d"))
        ttk.Entry(config_frame, textvariable=self.start_date_var, width=40).grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(config_frame, text="End Date:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.end_date_var = tk.StringVar(value=end_date_default.strftime("%Y-%m-%d"))
        ttk.Entry(config_frame, textvariable=self.end_date_var, width=40).grid(row=3, column=1, padx=5, pady=5)

        # Trading Settings
        trading_frame = ttk.LabelFrame(backtest_frame, text="Trading Settings", padding=10)
        trading_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(trading_frame, text="Initial Balance ($):").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.initial_balance_var = tk.StringVar(value="10000")
        ttk.Entry(trading_frame, textvariable=self.initial_balance_var, width=15).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(trading_frame, text="Max Positions:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.max_positions_var = tk.StringVar(value="3")
        ttk.Entry(trading_frame, textvariable=self.max_positions_var, width=15).grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(trading_frame, text="Risk Per Trade (%):").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.risk_per_trade_var = tk.StringVar(value="2.0")
        ttk.Entry(trading_frame, textvariable=self.risk_per_trade_var, width=15).grid(row=2, column=1, padx=5, pady=2)

        # Run Backtest Button
        self.run_btn = ttk.Button(config_frame, text="🚀 Run Backtest", command=self.run_backtest)
        self.run_btn.grid(row=4, column=0, columnspan=2, pady=10)
        
        # === ADDED: Progress Bar and Percentage ===
        self.progress_frame = ttk.LabelFrame(backtest_frame, text="Backtest Progress", padding=10)
        self.progress_frame.pack(fill="x", padx=10, pady=10)

        # Variable for progress bar (0 to 100)
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", padx=5, pady=5)

        # Variable for percentage text
        self.progress_percent_var = tk.StringVar(value="0%")
        self.progress_label = ttk.Label(self.progress_frame, textvariable=self.progress_percent_var, font=("Arial", 10, "bold"))
        self.progress_label.pack(pady=2)
        # ==========================================
        
        # ADD THIS: Optimize Parameters Button
        #self.optimize_btn = ttk.Button(config_frame, text="🎯 Optimize Parameters", 
        #                            command=self.optimize_from_backtest_tab)
        #self.optimize_btn.grid(row=6, column=0, columnspan=2, pady=5)
    
    def create_results_tab(self):
        # Results Tab
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="Results")
        
        # Results Text
        self.results_text = tk.Text(results_frame, height=25, width=90)
        self.results_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.results_text.config(yscrollcommand=scrollbar.set)
    
    def create_status_bar(self):
        # Status Bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def browse_data_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.data_dir_var.set(directory)
    
    def create_strategy(self):
        """Create and configure the selected strategy with current parameters"""
        try:
            strategy_name = self.strategy_var.get()
            if not strategy_name:
                self.strategy_info_text.delete(1.0, tk.END)
                self.strategy_info_text.insert(1.0, "⚠️ Please select a strategy first!")
                self.status_var.set("⚠️ Please select a strategy first")
                return

            # Get current parameter values from the GUI (handles sliders, entries, and checkboxes)
            current_params = {}
            for param_name, var in self.param_widgets.items():
                if isinstance(var, tk.IntVar):
                    current_params[param_name] = var.get()
                elif isinstance(var, tk.DoubleVar):
                    current_params[param_name] = var.get()
                elif isinstance(var, tk.StringVar):
                    current_params[param_name] = var.get()
                elif isinstance(var, tk.BooleanVar):
                    current_params[param_name] = var.get()
                elif isinstance(var, ttk.Entry):
                    # Handle entry widgets (for string parameters)
                    current_params[param_name] = var.get()
                else:
                    # Fallback for any other widget type
                    current_params[param_name] = var.get()

            # Create strategy instance
            strategy_info = self.strategies.get(strategy_name)
            if strategy_info and 'create_func' in strategy_info:
                # Get symbols and timeframes from GUI
                symbols = [s.strip() for s in self.symbols_var.get().split(',') if s.strip()]
                timeframes = [t.strip() for t in self.timeframes_var.get().split(',') if t.strip()]
                
                # Create strategy with symbols and timeframes
                self.current_strategy = strategy_info['create_func'](
                    symbols=symbols,
                    timeframes=timeframes,
                    **current_params
                )
                
                print(f"🔧 GUI DEBUG: Created strategy with symbols: {symbols}")
                print(f"🔧 GUI DEBUG: Created strategy with timeframes: {timeframes}")

                # Ensure parameters are set as attributes on the strategy object
                if strategy_info and 'create_func' in strategy_info:
                    # Set parameters as attributes for easy access
                    for param_name, param_value in current_params.items():
                        setattr(self.current_strategy, param_name, param_value)
                    
                    # Also set as a params dictionary for compatibility
                    self.current_strategy.params = current_params
                    
                    print(f"🔧 GUI DEBUG: Set strategy parameters: {current_params}")

            else:
                # Fallback strategy
                class SimpleStrategy:
                    def __init__(self, name, parameters):
                        self.name = name
                        self.parameters = parameters
                        self.symbols = []
                        self.timeframes = ['1m']
                        
                    def generate_signals(self, data):
                        signals = {}
                        for symbol in data:
                            for timeframe in data[symbol]:
                                df = data[symbol][timeframe].copy()
                                if len(df) > 20:
                                    df['sma20'] = df['close'].rolling(window=20).mean()
                                    df['signal'] = 0
                                    df.loc[df['close'] < df['sma20'], 'signal'] = 1
                                    df.loc[df['close'] > df['sma20'], 'signal'] = -1
                                    signals[symbol] = {timeframe: df[['signal']].copy()}
                            return signals
                
                self.current_strategy = SimpleStrategy(strategy_name, current_params)

            # Update GUI
            self.strategy_info_text.delete(1.0, tk.END)
            info_text = f"✅ Strategy '{strategy_name}' created successfully!\n\n"
            info_text += f"📊 Parameters:\n"
            for param, value in current_params.items():
                info_text += f" • {param}: {value}\n"
            info_text += f"\n💡 Now you can run optimization to improve these parameters!"
            self.strategy_info_text.insert(1.0, info_text)
            
            # Switch to backtest tab
            #self.notebook.select(1)
            self.status_var.set(f"✅ Strategy '{strategy_name}' created and configured")
            
        except Exception as e:
            self.strategy_info_text.delete(1.0, tk.END)
            self.strategy_info_text.insert(1.0, f"❌ Error creating strategy: {str(e)}")
            self.status_var.set(f"❌ Error: {str(e)}")
            import traceback
            print(f"Error in create_strategy: {traceback.format_exc()}")
    
    def update_parameters_with_best(self, best_params):
        """Update GUI parameter fields with optimized values"""
        try:
            # Debug: Print what we received
            print(f"DEBUG: Received best_params: {best_params}")
            
            # Get current strategy parameters
            strategy_name = self.strategy_var.get()
            if strategy_name in self.strategies:
                strategy_info = self.strategies[strategy_name]
                strategy_params = strategy_info.get('parameters', {})
                
                # Debug: Print available strategy parameters
                print(f"DEBUG: Strategy parameters: {strategy_params}")
                
                # Map optimization parameters to strategy parameters
                param_mapping = self.get_parameter_mapping(strategy_name)
                
                # Debug: Print parameter mapping
                print(f"DEBUG: Parameter mapping: {param_mapping}")
                
                # Debug: Print available GUI widgets
                print(f"DEBUG: Available GUI widgets: {list(self.param_widgets.keys())}")
                
                # Update each parameter
                updated_count = 0
                for opt_param, value in best_params.items():
                    if opt_param in param_mapping:
                        strategy_param = param_mapping[opt_param]
                        if strategy_param in self.param_widgets:
                            widget = self.param_widgets[strategy_param]
                            
                            # Get the old value for debugging
                            old_value = widget.get()
                            
                            # Update based on widget type
                            if isinstance(widget, tk.IntVar):
                                widget.set(int(value))
                                print(f"DEBUG: Updated IntVar {strategy_param}: {old_value} -> {value}")
                            elif isinstance(widget, tk.DoubleVar):
                                widget.set(float(value))
                                print(f"DEBUG: Updated DoubleVar {strategy_param}: {old_value} -> {value}")
                            elif isinstance(widget, tk.StringVar):
                                widget.set(str(value))
                                print(f"DEBUG: Updated StringVar {strategy_param}: {old_value} -> {value}")
                            elif isinstance(widget, (ttk.Entry, ttk.Combobox)):
                                # Handle Entry/Combobox widgets
                                widget.delete(0, tk.END)
                                widget.insert(0, str(value))
                                print(f"DEBUG: Updated Entry/Combobox {strategy_param}: {old_value} -> {value}")
                            else:
                                print(f"DEBUG: Unknown widget type for {strategy_param}: {type(widget)}")
                            
                            updated_count += 1
                        else:
                            print(f"DEBUG: Widget not found for parameter: {strategy_param}")
                    else:
                        print(f"DEBUG: No mapping for optimization parameter: {opt_param}")
                
                print(f"DEBUG: Updated {updated_count} parameters")
                
                if updated_count == 0:
                    print("DEBUG: No parameters were updated - checking for direct matches")
                    # Try direct parameter name matches
                    for param_name, value in best_params.items():
                        if param_name in self.param_widgets:
                            widget = self.param_widgets[param_name]
                            old_value = widget.get()
                            
                            # Update based on widget type
                            if isinstance(widget, tk.IntVar):
                                widget.set(int(value))
                            elif isinstance(widget, tk.DoubleVar):
                                widget.set(float(value))
                            elif isinstance(widget, tk.StringVar):
                                widget.set(str(value))
                            elif isinstance(widget, (ttk.Entry, ttk.Combobox)):
                                widget.delete(0, tk.END)
                                widget.insert(0, str(value))
                            
                            print(f"DEBUG: Direct update {param_name}: {old_value} -> {value}")
                                
            else:
                print(f"DEBUG: Strategy {strategy_name} not found in strategies")
                                
        except Exception as e:
            print(f"DEBUG: Error in update_parameters_with_best: {e}")
            import traceback
            traceback.print_exc()

    def get_parameter_mapping(self, strategy_name):
        """Get mapping from optimization parameters to strategy parameters"""
        # Define how optimization parameters map to strategy parameters
        if 'Trend_Following' in strategy_name:
            return {
                'fast_period': 'fast_period',
                'slow_period': 'slow_period', 
                'ma_type': 'ma_type'
            }
        elif 'RSI' in strategy_name and 'Extremes' in strategy_name:
            return {
                'rsi_period': 'rsi_period',
                'rsi_oversold': 'oversold',
                'rsi_overbought': 'overbought'
            }
        elif 'MA_Crossover' in strategy_name or 'Moving Average' in strategy_name:
            return {
                'fast_period': 'fast_period',
                'slow_period': 'slow_period',
                'ma_type': 'ma_type'
            }
        else:
            # Default mapping - parameter names should match
            return {k: k for k in ['period', 'threshold', 'fast_period', 'slow_period', 'ma_type']}
    
    def _extract_metrics(self, results):
        """Extract metrics from results dictionary, handling different key formats"""
        metrics = {}
        
        # Handle both '_pct' and non-'_pct' versions of keys
        key_mappings = {
            'total_return': ['total_return_pct', 'total_return'],
            'win_rate': ['win_rate_pct', 'win_rate'],
            'sharpe_ratio': ['sharpe_ratio'],
            'max_drawdown': ['max_drawdown_pct', 'max_drawdown'],
            'total_trades': ['total_trades']
        }
        
        for metric_name, possible_keys in key_mappings.items():
            value = None
            for key in possible_keys:
                if key in results:
                    value = results[key]
                    break
            metrics[metric_name] = value if value is not None else 0
        
        return metrics
    
    def run_backtest(self):
        """Run backtest with current strategy and configuration using Threading"""
        
        # Validation: Check if strategy is created
        if not hasattr(self, 'current_strategy') or self.current_strategy is None:
            messagebox.showerror("Error", "Please create a strategy first")
            return
        
        # Validation: Check if strategy has required methods
        if not hasattr(self.current_strategy, 'generate_signals'):
            messagebox.showerror("Error", "Strategy does not have required generate_signals method")
            return
        
        # Get trading settings
        try:
            initial_balance = float(self.initial_balance_var.get())
            max_positions = int(self.max_positions_var.get())
            risk_per_trade = float(self.risk_per_trade_var.get()) / 100.0
        except ValueError:
            messagebox.showerror("Error", "Invalid trading settings!")
            return
        
        self.status_var.set("Running backtest...")
        self.results_text.delete(1.0, tk.END)
        
        # Reset progress bar
        self.progress_var.set(0)
        self.progress_percent_var.set("0%")
        self.root.update_idletasks()
        
        # Disable button while running
        self.run_btn.config(state="disabled")
        
        # Get backtest parameters
        symbols = [s.strip() for s in self.symbols_var.get().split(',') if s.strip()]
        timeframes = [t.strip() for t in self.timeframes_var.get().split(',') if t.strip()]
        start_date = self.start_date_var.get()
        end_date = self.end_date_var.get()
        
        # === Define the Progress Callback ===
        def update_progress(percent):
            # Update GUI variables (Thread-safe in Python for simple variables)
            self.progress_var.set(percent)
            self.progress_percent_var.set(f"{percent:.1f}%")
            self.root.update_idletasks() # Force immediate redraw
            
        # === Define the Backtest Task (to run in thread) ===
        def backtest_task():
            try:
                # Import inside thread or at top (ensure available)
                from simple_strategy.backtester.backtester_engine import BacktesterEngine
                from shared.data_feeder import DataFeeder
                from simple_strategy.backtester.position_manager import PositionManager
                
                # Create data feeder
                data_feeder = DataFeeder(data_dir=self.data_dir_var.get())
                
                # Display backtest info
                self.results_text.insert(tk.END, f"🚀 BACKTEST CONFIGURATION\n")
                self.results_text.insert(tk.END, f"=" * 50 + "\n")
                self.results_text.insert(tk.END, f"Strategy: {self.current_strategy.name}\n")
                self.results_text.insert(tk.END, f"Symbols: {symbols}\n")
                self.results_text.insert(tk.END, f"Timeframes: {timeframes}\n")
                self.results_text.insert(tk.END, f"Date Range: {start_date} to {end_date}\n")
                self.results_text.insert(tk.END, f"Initial Balance: ${initial_balance}\n")
                self.results_text.insert(tk.END, f"Max Positions: {max_positions}\n")
                self.results_text.insert(tk.END, f"Risk Per Trade: {risk_per_trade*100:.1f}%\n")
                self.results_text.insert(tk.END, f"=" * 50 + "\n\n")
                
                # Check data files
                self.results_text.insert(tk.END, "🔍 Checking data files...\n")
                self.root.update()
                
                for symbol in symbols:
                    for timeframe in timeframes:
                        clean_timeframe = timeframe.rstrip('m')
                        file_path = os.path.join(self.data_dir_var.get(), f"{symbol}_{clean_timeframe}.csv")
                        if os.path.exists(file_path):
                            self.results_text.insert(tk.END, f"✅ Found: {file_path}\n")
                        else:
                            self.results_text.insert(tk.END, f"❌ Missing: {file_path}\n")
                
                self.results_text.insert(tk.END, "\n⏳ Running backtest...\n")
                self.root.update()
                
                # Create backtester
                backtester = BacktesterEngine(
                    data_feeder=data_feeder,
                    strategy=self.current_strategy,
                    risk_manager=None,
                    config={
                        'processing_mode': 'sequential',
                        'batch_size': 1000,
                        'memory_limit_percent': 70,
                        'enable_parallel_processing': False
                    }
                )

                # Run Backtest (Passing the callback here!)
                results = backtester.run_backtest(
                    strategy=self.current_strategy,
                    data=None, # Let engine load data based on config
                    start_date=start_date,
                    end_date=end_date,
                    initial_balance=initial_balance,
                    symbols=symbols,
                    timeframes=timeframes,
                    progress_callback=update_progress # <--- LINKING THE PROGRESS BAR
                )
                
                # Display results
                self.results_text.insert(tk.END, "\n" + "="*50 + "\n")
                self.results_text.insert(tk.END, "📊 BACKTEST RESULTS\n")
                self.results_text.insert(tk.END, "="*50 + "\n")
                
                start_time = results.get('start_time', 'N/A')
                end_time = results.get('end_time', 'N/A')
                duration = results.get('duration', 'N/A')
                
                self.results_text.insert(tk.END, f"🕒 Start Time: {start_time}\n")
                self.results_text.insert(tk.END, f"🕒 End Time:   {end_time}\n")
                self.results_text.insert(tk.END, f"⏱️ Duration:  {duration}\n")
                self.results_text.insert(tk.END, "-" * 30 + "\n")

                if results and isinstance(results, dict):
                    metrics = self._extract_metrics(results)
                    total_return_pct = metrics['total_return']
                    win_rate_pct = metrics['win_rate']
                    sharpe_ratio = metrics['sharpe_ratio']
                    max_drawdown_pct = metrics['max_drawdown']
                    total_trades = metrics['total_trades']
                    
                    start_amount = initial_balance
                    total_pnl = (total_return_pct / 100) * start_amount
                    end_amount = start_amount + total_pnl
                    
                    self.results_text.insert(tk.END, f"💰 FINANCIAL SUMMARY\n")
                    self.results_text.insert(tk.END, f"-" * 30 + "\n")
                    self.results_text.insert(tk.END, f"💵 Start Amount: ${start_amount:,.2f}\n")
                    self.results_text.insert(tk.END, f"💵 End Amount: ${end_amount:,.2f}\n")
                    
                    if total_pnl >= 0:
                        self.results_text.insert(tk.END, f"📈 Gain/Loss: +${total_pnl:,.2f}\n")
                    else:
                        self.results_text.insert(tk.END, f"📉 Gain/Loss: ${total_pnl:,.2f}\n")
                    
                    self.results_text.insert(tk.END, f"-" * 30 + "\n\n")
                    self.results_text.insert(tk.END, f"📊 PERFORMANCE METRICS\n")
                    self.results_text.insert(tk.END, f"-" * 30 + "\n")
                    self.results_text.insert(tk.END, f"💰 Total Return: {total_return_pct:.2f}%\n")
                    self.results_text.insert(tk.END, f"🎯 Win Rate: {win_rate_pct:.2f}%\n")
                    self.results_text.insert(tk.END, f"📈 Sharpe Ratio: {sharpe_ratio:.2f}\n")
                    self.results_text.insert(tk.END, f"📉 Max Drawdown: {max_drawdown_pct:.2f}%\n")
                    self.results_text.insert(tk.END, f"🔄 Total Trades: {total_trades}\n")
                else:
                    self.results_text.insert(tk.END, "❌ No results returned\n")

                # Finish: Update status and re-enable button
                self.status_var.set("✅ Backtest completed successfully")
                self.run_btn.config(state="normal")
                
            except Exception as e:
                # Error handling
                self.results_text.insert(tk.END, f"\n❌ Backtest execution error: {str(e)}\n")
                import traceback
                self.results_text.insert(tk.END, f"Traceback: {traceback.format_exc()}\n")
                self.status_var.set("❌ Backtest execution failed")
                self.run_btn.config(state="normal")

        # === Start the Thread ===
        # We start the backtest_task in a separate thread so the GUI doesn't freeze
        thread = threading.Thread(target=backtest_task)
        thread.start()

    def check_data_files(self):
        """Check if required data files exist"""
        symbols = self.symbols_var.get().split(',')
        symbols = [s.strip() for s in symbols if s.strip()]
        
        timeframes = self.timeframes_var.get().split(',')
        timeframes = [t.strip() for t in timeframes if t.strip()]
        
        # Clear results tab
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"🔍 Checking data directory: {self.data_dir_var.get()}\n\n")
        
        # Check files (remove 'm' from timeframe)
        missing_files = []
        found_files = []
        
        for symbol in symbols:
            for timeframe in timeframes:
                # Remove 'm' from timeframe if present
                clean_timeframe = timeframe.rstrip('m')
                filename = f"{self.data_dir_var.get()}\\{symbol}_{clean_timeframe}.csv"
                if not os.path.exists(filename):
                    missing_files.append(filename)
                else:
                    found_files.append(filename)
        
        # Display results
        self.results_text.insert(tk.END, f"📁 Found {len(found_files)} CSV files:\n")
        for file in found_files:
            self.results_text.insert(tk.END, f"  - {os.path.basename(file)}\n")
        
        if missing_files:
            self.results_text.insert(tk.END, f"\n❌ Missing files:\n")
            for file in missing_files:
                self.results_text.insert(tk.END, f"  - {os.path.basename(file)}\n")
        else:
            self.results_text.insert(tk.END, f"\n✅ All required files found!\n")

    def get_parameter_mapping(self, strategy_name):
        """Get mapping from optimization parameters to strategy parameters"""
        # Define how optimization parameters map to strategy parameters
        if 'Trend_Following' in strategy_name:
            return {
                'fast_period': 'fast_period',
                'slow_period': 'slow_period', 
                'ma_type': 'ma_type'
            }
        elif 'RSI' in strategy_name and 'Extremes' in strategy_name:
            return {
                'rsi_period': 'rsi_period',
                'rsi_oversold': 'oversold',
                'rsi_overbought': 'overbought'
            }
        elif 'MA_Crossover' in strategy_name or 'Moving Average' in strategy_name:
            return {
                'fast_period': 'fast_period',
                'slow_period': 'slow_period',
                'ma_type': 'ma_type'
            }
        else:
            # Default mapping - parameter names should match
            return {k: k for k in ['period', 'threshold', 'fast_period', 'slow_period', 'ma_type']}

    def debug_current_parameters(self):
        """Debug method to show current parameter values"""
        print("=== DEBUG: Current GUI Parameters ===")
        for param_name, widget in self.param_widgets.items():
            value = widget.get()
            print(f"{param_name}: {value} (widget type: {type(widget).__name__})")
        print("=== END DEBUG ===")

if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleStrategyGUI(root)
    root.mainloop()