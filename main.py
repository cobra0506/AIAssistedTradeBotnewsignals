# root/main.py - Dashboard GUI
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
import os
from simple_strategy.trading.parameter_gui import ParameterGUI

class TradingBotDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Trading Bot Control Center")
        self.root.geometry("600x500")
        self.create_widgets()
    
    def create_widgets(self):
        # Data Collection Section
        self.create_data_collection_section()
        # Simple Strategy Section (NEW - FUNCTIONAL)
        self.create_simple_strategy_section()
        # Placeholder sections for future modules
        self.create_placeholder_section("🤖 SL AI MODULE", "sl_ai")
        self.create_placeholder_section("🧠 RL AI MODULE", "rl_ai")
        # Bottom buttons
        self.create_bottom_buttons()
    
    def create_data_collection_section(self):
        # Data Collection Frame
        dc_frame = ttk.LabelFrame(self.root, text="📊 DATA COLLECTION MODULE", padding=10)
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
        ss_frame = ttk.LabelFrame(self.root, text="📈 SIMPLE STRATEGY MODULE", padding=10)
        ss_frame.pack(fill="x", padx=10, pady=5)
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(ss_frame)
        self.notebook.pack(fill="both", expand=True)
        
        # Create three tabs
        self.create_backtesting_tab()
        self.create_paper_trading_tab()
        self.create_live_trading_tab()
        
        # Status label (outside tabs)
        self.ss_status = tk.StringVar(value="🔴 STOPPED")
        status_label = ttk.Label(ss_frame, textvariable=self.ss_status, font=("Arial", 10, "bold"))
        status_label.pack(anchor="w", pady=(5, 0))

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
    
    def create_placeholder_section(self, title, module_name):
        frame = ttk.LabelFrame(self.root, text=f"{title} (Coming Soon)", padding=10)
        frame.pack(fill="x", padx=10, pady=5)
        
        status_var = tk.StringVar(value="⚫ NOT IMPLEMENTED")
        ttk.Label(frame, textvariable=status_var, font=("Arial", 10, "bold")).pack(anchor="w")
        
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill="x", pady=5)
        
        ttk.Button(button_frame, text="START", state="disabled").pack(side="left", padx=5)
        ttk.Button(button_frame, text="SETTINGS", state="disabled").pack(side="left", padx=5)
    
    def create_bottom_buttons(self):
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Button(bottom_frame, text="📋 SYSTEM LOGS").pack(side="left", padx=5)
        ttk.Button(bottom_frame, text="🔧 GLOBAL SETTINGS").pack(side="left", padx=5)
        ttk.Button(bottom_frame, text="❌ EXIT", command=self.root.quit).pack(side="right", padx=5)
    
    def start_data_collection(self):
        try:
            # Start data collection using the launcher script in data_collection folder
            launcher_path = os.path.join(os.path.dirname(__file__), 
                                      "shared_modules", "data_collection", "launch_data_collection.py")
            self.data_collection_process = subprocess.Popen([sys.executable, launcher_path])
            
            # Update UI
            self.dc_status.set("🟢 RUNNING")
            self.dc_start_btn.config(state="disabled")
            self.dc_stop_btn.config(state="normal")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start data collection: {e}")
    
    def stop_data_collection(self):
        if self.data_collection_process:
            try:
                self.data_collection_process.terminate()
                self.data_collection_process = None
                
                # Update UI
                self.dc_status.set("🔴 STOPPED")
                self.dc_start_btn.config(state="normal")
                self.dc_stop_btn.config(state="disabled")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to stop data collection: {e}")

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
                launcher_path = os.path.join(os.path.dirname(__file__), 
                                        "simple_strategy", "trading", "paper_trading_launcher.py")
                
                # Pass parameters as command line arguments
                subprocess.Popen([sys.executable, launcher_path, 
                                account, strategy, balance])
                
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
            
            if not account or not strategy:
                messagebox.showerror("Error", "Please select both account and strategy!")
                return
            
            # Initialize and start trading engine
            from simple_strategy.trading.paper_trading_engine import PaperTradingEngine
            self.paper_trading_engine = PaperTradingEngine(account, strategy, balance)
            
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

if __name__ == "__main__":
    root = tk.Tk()
    app = TradingBotDashboard(root)
    root.mainloop()