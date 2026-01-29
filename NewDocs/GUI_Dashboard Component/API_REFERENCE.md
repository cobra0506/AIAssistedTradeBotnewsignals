# GUI/Dashboard Component API Reference

## Overview

This document provides a comprehensive API reference for all GUI/Dashboard components. Each component exposes specific methods, properties, and events for integration and extension.

## 1. Main Dashboard API (`TradingBotDashboard`)

### Class: TradingBotDashboard

#### Constructor
```python
def __init__(self, root)

Parameters: 

     root (tk.Tk): Root tkinter window object
     

Description:
Initializes the main dashboard GUI with all components and sets up the main control center. 
Public Methods 
Widget Creation Methods 

def create_widgets(self)

Description:
Creates and layouts all main dashboard widgets including data collection section, strategy section, and placeholder sections. 

Returns: None 

def create_data_collection_section(self)

Description:
Creates the data collection control section with start/stop buttons and status display. 

Returns: None 

def create_simple_strategy_section(self)

Description:
Creates the simple strategy module with tabbed interface for backtesting, paper trading, and live trading. 

Returns: None 

def create_backtesting_tab(self)

Description:
Creates the backtesting tab with controls for opening backtester, settings, parameter manager, and API manager. 

Returns: None 

def create_paper_trading_tab(self)

Description:
Creates the paper trading tab with account selection, strategy selection, and balance simulation controls. 

Returns: None 

def create_live_trading_tab(self)

Description:
Creates the live trading tab with account selection, strategy selection, and appropriate safety warnings. 

Returns: None 

def create_placeholder_section(self, title, module_name)

Parameters: 

     title (str): Display title for the placeholder section
     module_name (str): Internal module identifier
     

Description:
Creates placeholder sections for future modules (SL AI, RL AI) with disabled controls. 

Returns: None 

def create_bottom_buttons(self)

Description:
Creates bottom control buttons for system logs, global settings, and application exit. 

Returns: None 
Control Methods 

def start_data_collection(self)

Description:
Starts the data collection subsystem as a subprocess. Updates UI to reflect running state. 

Returns: None 

Exceptions: 

     Shows error messagebox if subprocess creation fails
     

def stop_data_collection(self)

Description:
Stops the running data collection subprocess. Updates UI to reflect stopped state. 

Returns: None 

Exceptions: 

     Shows error messagebox if subprocess termination fails
     

Strategy Management Methods 

def start_simple_strategy(self)

Description:
Opens the backtesting interface for strategy development and testing. 

Returns: None 

def open_simple_strategy_settings(self)

Description:
Opens settings dialog for simple strategy configuration. 

Returns: None 

def open_parameter_manager(self)

Description:
Opens the parameter management GUI in a new window. 

Returns: None 

def open_api_manager(self)

Description:
Opens the API account management GUI in a new window. 

Returns: None 
Trading Interface Methods 

def open_paper_trader(self)

Description:
Opens paper trading interface with selected account and strategy. 

Returns: None 

def open_live_trader(self)

Description:
Opens live trading interface with selected account and strategy. 

Returns: None 
Configuration Loading Methods 

def load_paper_trading_options(self)

Description:
Loads demo accounts and available strategies for paper trading interface. 

Returns: None 

Exceptions: 

     Logs errors if loading fails but doesn't interrupt execution
     

def load_live_trading_options(self)

Description:
Loads live accounts and available strategies for live trading interface. 

Returns: None 

Exceptions: 

     Logs errors if loading fails but doesn't interrupt execution
     

Properties 

     root (tk.Tk): Main window object
     dc_status (tk.StringVar): Data collection status variable
     ss_status (tk.StringVar): Simple strategy status variable
     data_collection_process (subprocess.Popen): Data collection subprocess handle
     Various GUI widget references for internal use
     

2. Data Collection Monitor API (DataCollectionGUI) 
Class: DataCollectionGUI 
Constructor 

def __init__(self)

Description:
Initializes the data collection monitoring GUI with real-time status display and controls. 
Public Methods 
Setup Methods 

def setup_gui(self)

Description:
Sets up all GUI components including status panel, configuration panel, controls, and log display. 

Returns: None 

def update_config(self)

Description:
Updates the GUI configuration object when checkbox values change. 

Returns: None 

def get_config_summary(self)

Returns: str
Description:
Returns a human-readable summary of current configuration settings. 
Status Management Methods 

def update_status(self, connection=None, websocket=None, symbols=None)

Parameters: 

     connection (str, optional): API connection status
     websocket (str, optional): WebSocket connection status  
     symbols (int, optional): Number of symbols being tracked
     

Description:
Updates status indicators with color-coded display. 

Returns: None 
Control Methods 

def start_collection(self)

Description:
Starts data collection in a separate thread. Updates UI to reflect running state. 

Returns: None 

Exceptions: 

     Shows error messagebox if thread creation fails
     

def stop_collection(self)

Description:
Stops the running data collection. Updates UI to reflect stopped state. 

Returns: None 

def test_connection(self)

Description:
Tests API connection and displays results. 

Returns: None 
Monitoring Methods 

def start_gui_updater(self)

Description:
Starts the GUI update loop for processing log messages and updating display. 

Returns: None 

def start_system_stats_updater(self)

Description:
Starts system resource monitoring (memory, CPU usage) with periodic updates. 

Returns: None 
Logging Methods 

def log_message(self, message)

Parameters: 

     message (str): Message to log
     

Description:
Adds message to the thread-safe log queue for display. 

Returns: None 
Properties 

     root (tk.Tk): Main window object
     gui_config (DataCollectionConfig): Configuration object
     hybrid_system (HybridTradingSystem): Backend system reference
     running (bool): System running state
     log_queue (queue.Queue): Thread-safe message queue
     Status variables: connection_status, websocket_status, symbols_count, errors_count
     System stats: memory_usage, cpu_usage
     

3. Parameter Manager GUI API (ParameterGUI) 
Class: ParameterGUI 
Constructor 

def __init__(self, root)

Parameters: 

     root (tk.Tk): Root tkinter window object
     

Description:
Initializes the parameter management GUI with strategy selection and parameter display. 
Public Methods 
Setup Methods 

def create_widgets(self)

Description:
Creates GUI components including strategy selection, parameter display, and control buttons. 

Returns: None 
Strategy Management Methods 

def refresh_strategy_list(self)

Description:
Refreshes the list of available strategies from the ParameterManager. 

Returns: None 

def on_strategy_selected(self, event=None)

Parameters: 

     event (tk.Event, optional): Tkinter event object
     

Description:
Handles strategy selection and displays associated parameters. 

Returns: None 
Properties 

     root (tk.Tk): Main window object
     pm (ParameterManager): Parameter manager backend reference
     strategy_var (tk.StringVar): Selected strategy variable
     strategy_combo (ttk.Combobox): Strategy selection widget
     params_frame (ttk.Frame): Parameter display frame
     status_var (tk.StringVar): Status display variable
     

4. API Account Manager GUI API (APIGUI) 
Class: APIGUI 
Constructor 

def __init__(self, root)

Parameters: 

     root (tk.Tk): Root tkinter window object
     

Description:
Initializes the API account management GUI with tabbed interface for demo and live accounts. 
Public Methods 
Setup Methods 

def create_widgets(self)

Description:
Creates main GUI components including title, notebook tabs, and control buttons. 

Returns: None 

def create_account_tab(self, parent, account_type)

Parameters: 

     parent (tk.Widget): Parent widget for the tab
     account_type (str): Type of account ("demo" or "live")
     

Description:
Creates a tab for managing accounts of the specified type. 

Returns: None 
Account Management Methods 

def refresh_account_lists(self)

Description:
Refreshes the display of both demo and live account lists. 

Returns: None 

def add_account(self, account_type)

Parameters: 

     account_type (str): Type of account to add ("demo" or "live")
     

Description:
Opens dialog for adding a new API account. 

Returns: None 

def edit_account(self, account_type)

Parameters: 

     account_type (str): Type of account to edit ("demo" or "live")
     

Description:
Opens dialog for editing an existing API account. 

Returns: None 

def delete_account(self, account_type)

Parameters: 

     account_type (str): Type of account to delete ("demo" or "live")
     

Description:
Deletes selected API account after confirmation. 

Returns: None 
Properties 

     root (tk.Tk): Main window object
     manager (APIManager): API manager backend reference
     demo_tree (ttk.Treeview): Demo accounts treeview
     live_tree (ttk.Treeview): Live accounts treeview
     

Event Handlers and Callbacks 
Common Event Patterns 
Button Command Handlers 

# Standard button command pattern
def button_handler(self):
    try:
        # Perform action
        result = self.perform_action()
        
        # Update UI
        self.update_status("Success")
        messagebox.showinfo("Success", "Action completed successfully")
        
    except Exception as e:
        # Handle errors
        self.update_status("Error")
        messagebox.showerror("Error", f"Failed: {str(e)}")
        self.log_message(f"ERROR: {str(e)}")

Selection Change Handlers

# Standard selection handler pattern
def on_selection_changed(self, event=None):
    selected_item = self.widget.get()
    if selected_item:
        self.load_item_details(selected_item)
        self.update_controls_for_selection(selected_item)

Configuration Change Handlers

# Configuration change handler pattern
def on_config_changed(self):
    # Update configuration object
    self.config.setting = self.variable.get()
    
    # Apply changes
    self.apply_configuration_changes()
    
    # Log the change
    self.log_message(f"Configuration updated: {self.config.setting}")

Integration Points 
External Component Integration 
Subprocess Management 

# Launch external components
def launch_component(self, component_path):
    try:
        process = subprocess.Popen([sys.executable, component_path])
        return process
    except Exception as e:
        messagebox.showerror("Launch Error", f"Failed to launch component: {str(e)}")
        return None

Backend Module Integration

# Integrate with backend modules
def integrate_with_backend(self):
    # Import backend modules
    from backend_module import BackendClass
    
    # Create backend instance
    self.backend = BackendClass()
    
    # Set up callbacks
    self.backend.set_status_callback(self.update_status)
    self.backend.set_error_callback(self.handle_error)

Configuration Integration 
Load Configuration 

def load_configuration(self):
    config_file = "config.json"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        # Apply configuration to GUI
        self.apply_configuration_to_gui(config_data)

Save Configuration

def save_configuration(self):
    config_data = self.extract_configuration_from_gui()
    
    with open("config.json", 'w') as f:
        json.dump(config_data, f, indent=2)
    
    messagebox.showinfo("Success", "Configuration saved successfully")

Error Handling Patterns 
Standard Error Handling 

def safe_operation(self):
    try:
        # Attempt operation
        result = self.risky_operation()
        
        # Handle success
        self.handle_success(result)
        return result
        
    except ValueError as ve:
        # Handle validation errors
        self.handle_validation_error(ve)
        
    except ConnectionError as ce:
        # Handle connection errors
        self.handle_connection_error(ce)
        
    except Exception as e:
        # Handle unexpected errors
        self.handle_unexpected_error(e)
        
    finally:
        # Cleanup
        self.cleanup_operation()

User Feedback

def show_user_feedback(self, title, message, message_type="info"):
    if message_type == "info":
        messagebox.showinfo(title, message)
    elif message_type == "warning":
        messagebox.showwarning(title, message)
    elif message_type == "error":
        messagebox.showerror(title, message)
    elif message_type == "question":
        return messagebox.askyesno(title, message)

