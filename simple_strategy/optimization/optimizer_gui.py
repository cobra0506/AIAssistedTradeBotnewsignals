import tkinter as tk
from tkinter import ttk, messagebox
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from simple_strategy.optimization import BayesianOptimizer, ParameterSpace
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.shared.data_feeder import DataFeeder

class OptimizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Strategy Optimizer")
        self.root.geometry("600x500")
        
        self.create_widgets()
    
    def create_widgets(self):
        # Symbol selection
        ttk.Label(self.root, text="Symbol:").grid(row=0, column=0, padx=5, pady=5)
        self.symbol_var = tk.StringVar(value="BTCUSDT")
        ttk.Entry(self.root, textvariable=self.symbol_var).grid(row=0, column=1, padx=5, pady=5)
        
        # Timeframe selection
        ttk.Label(self.root, text="Timeframe:").grid(row=1, column=0, padx=5, pady=5)
        self.timeframe_var = tk.StringVar(value="60")
        ttk.Combobox(self.root, textvariable=self.timeframe_var, 
                    values=["1", "5", "15", "60", "240", "1440"]).grid(row=1, column=1, padx=5, pady=5)
        
        # Date range
        ttk.Label(self.root, text="Start Date:").grid(row=2, column=0, padx=5, pady=5)
        self.start_date_var = tk.StringVar(value="2025-09-23")
        ttk.Entry(self.root, textvariable=self.start_date_var).grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(self.root, text="End Date:").grid(row=3, column=0, padx=5, pady=5)
        self.end_date_var = tk.StringVar(value="2025-10-21")
        ttk.Entry(self.root, textvariable=self.end_date_var).grid(row=3, column=1, padx=5, pady=5)
        
        # Number of trials
        ttk.Label(self.root, text="Trials:").grid(row=4, column=0, padx=5, pady=5)
        self.trials_var = tk.StringVar(value="20")
        ttk.Entry(self.root, textvariable=self.trials_var).grid(row=4, column=1, padx=5, pady=5)
        
        # Optimize button
        ttk.Button(self.root, text="Optimize Strategy", 
                  command=self.optimize_strategy).grid(row=5, column=0, columnspan=2, pady=10)
        
        # Results display
        self.results_text = tk.Text(self.root, height=15, width=70)
        self.results_text.grid(row=6, column=0, columnspan=2, padx=5, pady=5)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.results_text.yview)
        scrollbar.grid(row=6, column=2, sticky='ns')
        self.results_text.configure(yscrollcommand=scrollbar.set)
    
    def optimize_strategy(self):
        try:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Starting optimization...\n\n")
            
            # Get parameters
            symbol = self.symbol_var.get()
            timeframe = self.timeframe_var.get()
            start_date = self.start_date_var.get()
            end_date = self.end_date_var.get()
            trials = int(self.trials_var.get())
            
            # Create parameter space
            param_space = ParameterSpace()
            param_space.add_int('rsi_period', 5, 30, step=1)
            param_space.add_float('rsi_oversold', 20, 40)
            param_space.add_float('rsi_overbought', 60, 80)
            param_space.add_int('sma_short_period', 5, 20, step=1)
            param_space.add_int('sma_long_period', 20, 50, step=5)
            
            # Create optimizer
            data_feeder = DataFeeder(data_dir='data')
            optimizer = BayesianOptimizer(
                data_feeder=data_feeder,
                study_name=f'gui_optimization_{symbol}',
                direction='maximize',
                n_trials=trials,
                timeout=3600
            )
            
            # Run optimization - FIX THIS LINE
            best_params, best_score = optimizer.optimize(
                strategy_name='Strategy_1_Trend_Following',  # <-- CHANGED FROM strategy_builder_class
                parameter_space=param_space,
                symbols=[symbol],
                timeframes=[timeframe],
                start_date=start_date,
                end_date=end_date,
                metric='sharpe_ratio'
            )
            
            # Display results
            results = f"""
    🎉 OPTIMIZATION COMPLETE! 🎉

    📈 Best Sharpe Ratio: {best_score:.4f}

    ⚙️ Best Parameters:
    """
            for param, value in best_params.items():
                results += f"   {param}: {value}\n"
            
            results += f"""
    📊 Symbol: {symbol}
    ⏰ Timeframe: {timeframe} minutes
    📅 Date Range: {start_date} to {end_date}
    🔢 Trials Run: {trials}

    💡 You can now use these parameters in your strategies!
    """
            
            self.results_text.insert(tk.END, results)
            
        except Exception as e:
            messagebox.showerror("Error", f"Optimization failed: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = OptimizerGUI(root)
    root.mainloop()