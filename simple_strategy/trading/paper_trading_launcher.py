#paper_trading_launcher.py

import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os
import json
import random
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

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
        self.root.geometry("800x600")
        
        # Initialize trading engine
        self.trading_engine = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # Header with performance info
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill="x", padx=10, pady=5)
        
        # Left side - Strategy and Account info
        left_frame = ttk.Frame(header_frame)
        left_frame.pack(side="left", fill="x", expand=True)
        
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
        
        if optimized_params:
            param_status = f"✅ Using optimized parameters (Last: {optimized_params.get('last_optimized', 'Unknown')})"
            param_color = "green"
        else:
            param_status = "⚠️ Using default parameters (Not optimized)"
            param_color = "orange"
        
        ttk.Label(param_frame, text=param_status, foreground=param_color).pack(side="left", padx=5)
        
        # Right side - Performance info
        right_frame = ttk.Frame(header_frame)
        right_frame.pack(side="right", fill="y")
        
        # Performance display in header
        perf_header_frame = ttk.LabelFrame(right_frame, text="Performance", padding=5)
        perf_header_frame.pack(side="right", padx=10)
        
        self.perf_header_labels = {}
        # UPDATED: Changed label text and dictionary keys
        perf_items = [
            ("Account Value:", "account_value", "$1000.00"), # Renamed from P&L
            ("Available Bal:", "available_balance", "$1000.00"), # Renamed from Balance
            ("Open:", "open_positions", "0"),
            ("Total Trades:", "total_trades", "0"), # Renamed from Closed
            ("Win Rate:", "win_rate", "0.0%")
        ]
        
        for i, (label_text, key, default) in enumerate(perf_items):
            label = ttk.Label(perf_header_frame, text=label_text)
            label.grid(row=i, column=0, sticky="e", padx=2)
            value_label = ttk.Label(perf_header_frame, text=default, font=("Arial", 10, "bold"))
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
        
        # Status with color
        self.status_var = tk.StringVar(value="🔴 STOPPED")
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

    def update_status(self, status):
        """Update status display with appropriate colors"""
        self.status_var.set(status)
        
        # Update color based on status
        if "RUNNING" in status:
            self.status_label.config(foreground="green")
        else:
            self.status_label.config(foreground="black")

    def stop_trading(self):
        """Stop paper trading"""
        if self.trading_engine:
            self.trading_engine.stop_trading()
        
        self.log_message("Paper trading stopped")
        self.update_status("🔴 STOPPED")  # Use the update_status method instead
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def update_performance_header(self, performance_data):
        """Update performance display in header"""
        try:
            if performance_data:
                # UPDATED: Use new keys to update header performance labels
                if 'account_value' in performance_data:
                    account_val = performance_data['account_value']
                    account_text = f"${account_val:.2f}"
                    # Color code P&L based on account value vs initial balance
                    if account_val > self.simulated_balance:
                        self.perf_header_labels['account_value'].config(text=account_text, foreground="green")
                    elif account_val < self.simulated_balance:
                        self.perf_header_labels['account_value'].config(text=account_text, foreground="red")
                    else:
                        self.perf_header_labels['account_value'].config(text=account_text, foreground="black")

                if 'available_balance' in performance_data:
                    self.perf_header_labels['available_balance'].config(text=f"${performance_data['available_balance']:.2f}")
                
                if 'open_positions' in performance_data:
                    self.perf_header_labels['open_positions'].config(text=str(performance_data['open_positions']))
                
                if 'total_trades' in performance_data:
                    self.perf_header_labels['total_trades'].config(text=str(performance_data['total_trades']))
                
                if 'win_rate' in performance_data:
                    self.perf_header_labels['win_rate'].config(text=f"{performance_data['win_rate']:.1f}%")
                    
        except Exception as e:
            print(f"Error updating performance header: {e}")

    def toggle_percentage_entry(self):
        """Enable/disable the percentage entry based on checkbox state"""
        if self.stop_at_percentage_var.get():
            self.stop_percentage_spinbox.config(state="normal")
        else:
            self.stop_percentage_spinbox.config(state="disabled")

    def start_trading(self):
        """Start paper trading"""
        try:
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
            
            # Get trading control values
            max_positions = self.max_positions_var.get()
            stop_trading_at_percentage = self.stop_percentage_value.get() if self.stop_at_percentage_var.get() else None
            
            # Log the values to verify they're being read correctly
            self.log_message(f"Starting with max_positions={max_positions}, stop_percentage={stop_trading_at_percentage}")
            
            # Import and create trading engine with new parameters
            from simple_strategy.trading.paper_trading_engine import PaperTradingEngine
            self.trading_engine = PaperTradingEngine(
                self.api_account,
                self.strategy_name,
                self.simulated_balance,
                log_callback=self.log_message,
                status_callback=self.update_status,
                performance_callback=self.update_performance,
                max_positions=max_positions,
                stop_trading_at_percentage=stop_trading_at_percentage
            )
            
            # Update max_positions in the engine's should_continue_trading method directly
            self.trading_engine.should_continue_trading = lambda: self.check_should_continue_trading(max_positions, stop_trading_at_percentage)
            
            # Initialize shared data access after engine creation
            self.trading_engine.initialize_shared_data_access()

            # Start performance update timer
            self.update_performance_timer()
            
            # Start trading in a separate thread
            self.log_message("Starting paper trading...")
            self.update_status("🟢 RUNNING")
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            
            # Start REAL trading
            self.start_real_trading()
            
        except Exception as e:
            self.log_message(f"Error starting trading: {e}")
            messagebox.showerror("Error", f"Failed to start trading: {e}")

    def check_should_continue_trading(self, max_positions, stop_trading_at_percentage):
        """Check if trading should continue based on balance and optional settings"""
        # Check if we've reached the maximum number of open positions
        if len(self.trading_engine.current_positions) >= max_positions:
            self.log_message(f"🛑 Max positions reached: {len(self.trading_engine.current_positions)} >= {max_positions}")
            return False
        
        # Optional: Check if balance is below a percentage threshold (if enabled)
        if stop_trading_at_percentage:
            min_balance = self.trading_engine.initial_balance * (stop_trading_at_percentage / 100)
            if self.trading_engine.simulated_balance < min_balance:
                self.log_message(f"🛑 Balance below threshold: ${self.trading_engine.simulated_balance:.2f} < ${min_balance:.2f} ({stop_trading_at_percentage}%)")
                return False
        
        return True

    def start_real_trading(self):
        """Start REAL trading using the trading engine"""
        import threading
        
        def trading_loop():
            try:
                # Start the real trading engine
                success = self.trading_engine.start_trading()
                if success:
                    self.log_message("✅ Real trading started successfully")
                else:
                    self.log_message("❌ Failed to start real trading")
                    self.stop_trading()
            except Exception as e:
                self.log_message(f"❌ Error in real trading: {e}")
                self.stop_trading()
        
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
            self.log_message(f"🛑 Max positions reached: {len(self.trading_engine.current_positions)} >= {max_positions}")
            return False
        
        # Optional: Check if balance is below a percentage threshold (if enabled)
        if stop_trading_at_percentage:
            min_balance = self.trading_engine.initial_balance * (stop_trading_at_percentage / 100)
            if self.trading_engine.simulated_balance < min_balance:
                self.log_message(f"🛑 Balance below threshold: ${self.trading_engine.simulated_balance:.2f} < ${min_balance:.2f} ({stop_trading_at_percentage}%)")
                return False
        
        return True

    def check_should_continue_trading(self, max_positions, stop_trading_at_percentage):
        """Check if trading should continue based on balance and optional settings"""
        # Check if we've reached the maximum number of open positions
        if len(self.trading_engine.current_positions) >= max_positions:
            self.log_message(f"🛑 Max positions reached: {len(self.trading_engine.current_positions)} >= {max_positions}")
            return False
        
        # Optional: Check if balance is below a percentage threshold (if enabled)
        if stop_trading_at_percentage:
            min_balance = self.trading_engine.initial_balance * (stop_trading_at_percentage / 100)
            if self.trading_engine.simulated_balance < min_balance:
                self.log_message(f"🛑 Balance below threshold: ${self.trading_engine.simulated_balance:.2f} < ${min_balance:.2f} ({stop_trading_at_percentage}%)")
                return False
        
        return True
            
    def log_message(self, message):
        """Add message to trading log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")

    def update_status(self, status):
        """Update status display"""
        self.status_var.set(status)
    
    def start_trading(self):
        """Start paper trading"""
        try:
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
            
            # Import and create trading engine
            from simple_strategy.trading.paper_trading_engine import PaperTradingEngine
            self.trading_engine = PaperTradingEngine(
                self.api_account,
                self.strategy_name,
                self.simulated_balance,
                log_callback=self.log_message,
                status_callback=self.update_status,
                performance_callback=self.update_performance
            )
            
            # Initialize shared data access after engine creation
            self.trading_engine.initialize_shared_data_access()

            # Start performance update timer
            self.update_performance_timer()
            
            # Start trading in a separate thread (simplified for now)
            self.log_message("Starting paper trading...")
            self.status_var.set("🟢 RUNNING")
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            
            # Start REAL trading
            self.start_real_trading()
            
        except Exception as e:
            self.log_message(f"Error starting trading: {e}")
            messagebox.showerror("Error", f"Failed to start trading: {e}")
    
    def stop_trading(self):
        """Stop paper trading"""
        if self.trading_engine:
            self.trading_engine.stop_trading()
        
        self.log_message("Paper trading stopped")
        self.status_var.set("🔴 STOPPED")
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
    
    def start_real_trading(self):
        """Start REAL trading using the trading engine"""
        import threading
        
        def trading_loop():
            try:
                # Start the real trading engine
                success = self.trading_engine.start_trading()
                if success:
                    self.log_message("✅ Real trading started successfully")
                else:
                    self.log_message("❌ Failed to start real trading")
                    self.stop_trading()
            except Exception as e:
                self.log_message(f"❌ Error in real trading: {e}")
                self.stop_trading()
        
        # Start real trading in separate thread
        thread = threading.Thread(target=trading_loop)
        thread.daemon = True
        thread.start()
    
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
                
                # Show both realized and unrealized P&L if available
                if 'realized_pnl' in performance_data and 'unrealized_pnl' in performance_data:
                    realized_pnl = performance_data['realized_pnl']
                    unrealized_pnl = performance_data['unrealized_pnl']
                    pnl_text = f"Realized: ${realized_pnl:.2f}, Unrealized: ${unrealized_pnl:.2f}"
                else:
                    pnl_text = f"${pnl:.2f}"
                    
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
                
                # Calculate completed trades (sells)
                completed_trades = sum(1 for trade in self.trading_engine.trades if trade['type'] == 'SELL') if hasattr(self.trading_engine, 'trades') else 0
                
                # Calculate win rate
                winning_trades = sum(1 for trade in self.trading_engine.trades if trade['type'] == 'SELL' and trade.get('pnl', 0) > 0) if hasattr(self.trading_engine, 'trades') else 0
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
            
            self.perf_text.delete(1.0, "end")
            self.perf_text.insert(1.0, perf_text)
            
            # Schedule next update
            self.root.after(5000, self.update_performance_timer)
        except Exception as e:
            print(f"Error in performance timer: {e}")
            # Schedule next update even if there's an error
            self.root.after(5000, self.update_performance_timer)

            
if __name__ == "__main__":
    # Get parameters from command line or use defaults
    import sys
    if len(sys.argv) >= 4:
        launcher = PaperTradingLauncher(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        launcher = PaperTradingLauncher()
    launcher.run()
