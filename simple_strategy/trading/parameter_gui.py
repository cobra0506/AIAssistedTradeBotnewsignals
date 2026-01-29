import tkinter as tk
from tkinter import ttk
from simple_strategy.trading.parameter_manager import ParameterManager

class ParameterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Parameter Manager")
        self.root.geometry("600x400")
        
        self.pm = ParameterManager()
        self.create_widgets()
        self.refresh_strategy_list()
    
    def create_widgets(self):
        # Strategy selection
        ttk.Label(self.root, text="Select Strategy:").pack(pady=5)
        
        self.strategy_var = tk.StringVar()
        self.strategy_combo = ttk.Combobox(self.root, textvariable=self.strategy_var)
        self.strategy_combo.pack(pady=5)
        self.strategy_combo.bind('<<ComboboxSelected>>', self.on_strategy_selected)
        
        # Parameters display
        self.params_frame = ttk.LabelFrame(self.root, text="Parameters")
        self.params_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status_var).pack(pady=5)
        
        # Buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Refresh", command=self.refresh_strategy_list).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Close", command=self.root.quit).pack(side="left", padx=5)
    
    def refresh_strategy_list(self):
        strategies = self.pm.get_all_strategies()
        self.strategy_combo['values'] = strategies
        if strategies:
            self.strategy_combo.current(0)
            self.on_strategy_selected()
        self.status_var.set(f"Found {len(strategies)} optimized strategies")
    
    def on_strategy_selected(self, event=None):
        strategy_name = self.strategy_var.get()
        if not strategy_name:
            return
        
        # Clear previous parameters
        for widget in self.params_frame.winfo_children():
            widget.destroy()
        
        # Display parameters
        params = self.pm.get_parameters(strategy_name)
        for param, value in params.items():
            param_text = f"{param}: {value}"
            if param == 'last_optimized':
                param_text += " ✅"
            
            ttk.Label(self.params_frame, text=param_text).pack(anchor="w", padx=10, pady=2)