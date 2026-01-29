# GUI Integration Guide

## Overview

This guide provides detailed information about integrating with the GUI/Dashboard component, including strategy integration, event handling, and custom component development.

## Strategy Integration

### Automatic Strategy Discovery

The GUI automatically discovers and loads strategies using a standardized pattern:

#### File Naming Convention

✅ Valid: Strategy_MyStrategy.py
✅ Valid: Strategy_RSI_EMA.py
❌ Invalid: my_strategy.py
❌ Invalid: strategy_builder.py


#### Required Strategy Structure
```python
# Strategy_MyStrategy.py

def create_strategy(symbols=None, timeframes=None, **params):
    """
    Required function for strategy creation.
    
    Args:
        symbols: List of trading symbols
        timeframes: List of timeframes
        **params: Strategy parameters
    
    Returns:
        Strategy object compatible with backtesting engine
    """
    # Strategy implementation
    return strategy_object

# Optional but recommended
STRATEGY_PARAMETERS = {
    'rsi_period': {
        'type': 'int',
        'default': 14,
        'min': 2,
        'max': 50,
        'description': 'RSI calculation period'
    },
    'ema_short': {
        'type': 'int', 
        'default': 12,
        'min': 1,
        'max': 100,
        'description': 'Short EMA period'
    }
}

Strategy Loading Process 

    Discovery Phase 

# GUI scans strategies directory
strategies_dir = "simple_strategy/strategies/"
strategy_files = [f for f in os.listdir(strategies_dir) 
                 if f.startswith("Strategy_") and f.endswith(".py")]

Import Phase

# Dynamic import of strategy modules
import importlib.util
spec = importlib.util.spec_from_file_location(strategy_name, strategy_path)
strategy_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strategy_module)

Validation Phase

# Check for required components
if hasattr(strategy_module, 'create_strategy'):
    valid_strategies.append(strategy_name)

Parameter Extraction

# Extract parameters for GUI controls
if hasattr(strategy_module, 'STRATEGY_PARAMETERS'):
    parameters = strategy_module.STRATEGY_PARAMETERS

Parameter Definition Format 
Supported Types 

STRATEGY_PARAMETERS = {
    # Integer parameters
    'period': {
        'type': 'int',
        'default': 14,
        'min': 1,
        'max': 100,
        'description': 'Calculation period'
    },
    
    # Float parameters
    'threshold': {
        'type': 'float',
        'default': 0.5,
        'min': 0.0,
        'max': 1.0,
        'description': 'Signal threshold'
    },
    
    # String parameters with options
    'signal_type': {
        'type': 'str',
        'default': 'crossover',
        'options': ['crossover', 'crossunder', 'level'],
        'description': 'Type of signal to generate'
    },
    
    # Boolean parameters
    'use_filter': {
        'type': 'bool',
        'default': True,
        'description': 'Enable additional filtering'
    }
}

GUI Control Mapping 

The GUI automatically creates appropriate controls based on parameter types: 

     int: Scale widget with min/max bounds
     float: Entry widget with validation
     str with options: Combobox with predefined choices
     str without options: Entry widget
     bool: Checkbutton widget
     

Event Handling Integration 
Custom Event Handlers 
Adding Custom Event Handlers 

class CustomGUIComponent:
    def __init__(self, parent):
        self.parent = parent
        self.setup_custom_events()
    
    def setup_custom_events(self):
        # Bind custom events
        self.parent.bind('<<CustomEvent>>', self.handle_custom_event)
    
    def handle_custom_event(self, event):
        # Handle custom event
        print(f"Custom event received: {event}")

Triggering Custom Events

def trigger_custom_event(self, data=None):
    event = tk.Event()
    event.data = data
    self.event_generate('<<CustomEvent>>', when='tail')

Status Update Integration 
Standard Status Updates 

def update_component_status(self, status, message=None):
    """
    Standard status update pattern.
    
    Args:
        status (str): Status type ("running", "stopped", "error")
        message (str, optional): Additional status message
    """
    status_map = {
        'running': ('🟢 RUNNING', 'green'),
        'stopped': ('🔴 STOPPED', 'red'),
        'error': ('⚠️ ERROR', 'orange'),
        'warning': ('⚠️ WARNING', 'orange')
    }
    
    if status in status_map:
        text, color = status_map[status]
        self.status_var.set(text)
        self.status_label.config(foreground=color)
    
    if message:
        self.log_message(message)

Progress Updates

def update_progress(self, current, total, message=None):
    """
    Update progress indicators.
    
    Args:
        current (int): Current progress value
        total (int): Total progress value
        message (str, optional): Progress message
    """
    if total > 0:
        percentage = (current / total) * 100
        self.progress_var.set(f"{percentage:.1f}%")
        
        if message:
            self.progress_message_var.set(message)

Custom Component Development 
Creating Custom GUI Components 
Base Component Template 

import tkinter as tk
from tkinter import ttk

class CustomGUIComponent:
    def __init__(self, parent, title="Custom Component"):
        self.parent = parent
        self.title = title
        
        # Create main frame
        self.frame = ttk.LabelFrame(parent, text=title, padding="10")
        
        # Initialize component
        self.setup_ui()
        self.setup_event_handlers()
        self.setup_data_bindings()
    
    def setup_ui(self):
        """Create UI elements"""
        # Override in subclasses
        pass
    
    def setup_event_handlers(self):
        """Set up event handlers"""
        # Override in subclasses
        pass
    
    def setup_data_bindings(self):
        """Set up data bindings"""
        # Override in subclasses
        pass
    
    def get_frame(self):
        """Return the main frame for embedding"""
        return self.frame
    
    def destroy(self):
        """Clean up resources"""
        self.frame.destroy()

Example: Custom Monitor Component

class CustomMonitorComponent(CustomGUIComponent):
    def __init__(self, parent, data_source):
        self.data_source = data_source
        super().__init__(parent, "Custom Monitor")
        
        # Start monitoring
        self.start_monitoring()
    
    def setup_ui(self):
        # Status display
        self.status_var = tk.StringVar(value="Initializing...")
        ttk.Label(self.frame, textvariable=self.status_var).pack(pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.frame, mode='indeterminate')
        self.progress.pack(fill='x', pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(self.frame)
        button_frame.pack(pady=5)
        
        ttk.Button(button_frame, text="Start", 
                  command=self.start_monitoring).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Stop", 
                  command=self.stop_monitoring).pack(side='left', padx=5)
    
    def setup_event_handlers(self):
        # Set up periodic updates
        self.update_interval = 1000  # 1 second
        self.update_job = None
    
    def start_monitoring(self):
        """Start monitoring process"""
        self.status_var.set("Monitoring...")
        self.progress.start()
        self.schedule_update()
    
    def stop_monitoring(self):
        """Stop monitoring process"""
        self.status_var.set("Stopped")
        self.progress.stop()
        if self.update_job:
            self.parent.after_cancel(self.update_job)
    
    def schedule_update(self):
        """Schedule next update"""
        try:
            # Get data from source
            data = self.data_source.get_data()
            
            # Process data
            processed_data = self.process_data(data)
            
            # Update display
            self.update_display(processed_data)
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
        
        # Schedule next update
        self.update_job = self.parent.after(self.update_interval, self.schedule_update)
    
    def process_data(self, data):
        """Process monitoring data"""
        # Override with custom processing logic
        return data
    
    def update_display(self, data):
        """Update display with processed data"""
        # Override with custom display logic
        pass

Integration with Main Dashboard 
Adding Custom Components to Dashboard 

class EnhancedTradingBotDashboard(TradingBotDashboard):
    def __init__(self, root):
        super().__init__(root)
        
        # Add custom components
        self.add_custom_components()
    
    def add_custom_components(self):
        """Add custom GUI components"""
        # Create custom monitor
        from custom_components import CustomMonitorComponent
        self.custom_monitor = CustomMonitorComponent(self.root, "Custom Monitor")
        self.custom_monitor.get_frame().pack(fill='x', padx=10, pady=5)
        
        # Create custom analyzer
        from custom_components import CustomAnalyzerComponent  
        self.custom_analyzer = CustomAnalyzerComponent(self.root)
        self.custom_analyzer.get_frame().pack(fill='x', padx=10, pady=5)

Custom Menu Integration

def add_custom_menu(self):
    """Add custom menu items"""
    menubar = self.root.cget('menu')
    if not menubar:
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
    
    # Add custom menu
    custom_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Custom", menu=custom_menu)
    
    # Add menu items
    custom_menu.add_command(label="Custom Tool 1", command=self.custom_tool_1)
    custom_menu.add_command(label="Custom Tool 2", command=self.custom_tool_2)
    custom_menu.add_separator()
    custom_menu.add_command(label="About", command=self.show_about)

Configuration Integration 
Custom Configuration Parameters 
Adding Custom Configuration 

class CustomConfig:
    def __init__(self):
        # Standard configuration
        self.LIMIT_TO_50_ENTRIES = True
        self.ENABLE_WEBSOCKET = True
        
        # Custom configuration
        self.CUSTOM_SETTING_1 = "default_value"
        self.CUSTOM_SETTING_2 = 42
        self.ENABLE_CUSTOM_FEATURE = False

GUI Integration

def add_custom_config_controls(self, config_frame):
    """Add custom configuration controls"""
    # Custom setting 1
    ttk.Label(config_frame, text="Custom Setting 1:").grid(row=5, column=0, sticky='w')
    self.custom_var1 = tk.StringVar(value=self.gui_config.CUSTOM_SETTING_1)
    ttk.Entry(config_frame, textvariable=self.custom_var1).grid(row=5, column=1, sticky='ew')
    
    # Custom setting 2
    ttk.Label(config_frame, text="Custom Setting 2:").grid(row=6, column=0, sticky='w')
    self.custom_var2 = tk.IntVar(value=self.gui_config.CUSTOM_SETTING_2)
    ttk.Spinbox(config_frame, from_=0, to=100, textvariable=self.custom_var2).grid(row=6, column=1, sticky='ew')
    
    # Custom feature toggle
    self.custom_feature_var = tk.BooleanVar(value=self.gui_config.ENABLE_CUSTOM_FEATURE)
    ttk.Checkbutton(config_frame, text="Enable Custom Feature", 
                   variable=self.custom_feature_var).grid(row=7, column=0, columnspan=2, sticky='w')

Data Visualization Integration 
Chart Integration 
Matplotlib Integration 

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ChartComponent:
    def __init__(self, parent):
        self.parent = parent
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Initial plot
        self.update_chart()
    
    def update_chart(self, data=None):
        """Update chart with new data"""
        self.ax.clear()
        
        if data:
            self.ax.plot(data['x'], data['y'], 'b-')
            self.ax.set_title('Custom Chart')
            self.ax.set_xlabel('X Axis')
            self.ax.set_ylabel('Y Axis')
            self.ax.grid(True)
        
        self.canvas.draw()

Real-time Data Updates

class RealTimeChartComponent(ChartComponent):
    def __init__(self, parent, data_source):
        super().__init__(parent)
        self.data_source = data_source
        
        # Start real-time updates
        self.update_interval = 1000  # 1 second
        self.start_real_time_updates()
    
    def start_real_time_updates(self):
        """Start real-time chart updates"""
        self.update_chart_real_time()
    
    def update_chart_real_time(self):
        """Update chart with real-time data"""
        try:
            # Get latest data
            data = self.data_source.get_latest_data()
            
            # Update chart
            self.update_chart(data)
            
        except Exception as e:
            print(f"Chart update error: {e}")
        
        # Schedule next update
        self.parent.after(self.update_interval, self.update_chart_real_time)

Testing and Debugging 
GUI Testing Framework 
Unit Testing GUI Components 

import unittest
import tkinter as tk

class TestGUIComponent(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.root = tk.Tk()
        self.root.withdraw()  # Hide window during tests
        
        # Create component to test
        self.component = YourGUIComponent(self.root)
    
    def tearDown(self):
        """Clean up test environment"""
        self.component.destroy()
        self.root.destroy()
    
    def test_component_creation(self):
        """Test component creation"""
        self.assertIsNotNone(self.component)
        self.assertTrue(hasattr(self.component, 'frame'))
    
    def test_widget_creation(self):
        """Test widget creation"""
        # Test that expected widgets exist
        self.assertTrue(hasattr(self.component, 'expected_widget'))
    
    def test_functionality(self):
        """Test component functionality"""
        # Test specific functionality
        result = self.component.test_method()
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()

Integration Testing

class TestGUIIntegration(unittest.TestCase):
    def test_dashboard_integration(self):
        """Test integration with main dashboard"""
        root = tk.Tk()
        root.withdraw()
        
        try:
            # Create dashboard
            dashboard = TradingBotDashboard(root)
            
            # Test component access
            self.assertTrue(hasattr(dashboard, 'dc_status'))
            self.assertTrue(hasattr(dashboard, 'ss_status'))
            
            # Test functionality
            dashboard.start_data_collection()
            self.assertEqual(dashboard.dc_status.get(), "🟢 RUNNING")
            
        finally:
            root.destroy()

Debugging Techniques 
GUI Debugging Helpers 

class GUIDebugHelper:
    @staticmethod
    def print_widget_hierarchy(widget, level=0):
        """Print widget hierarchy for debugging"""
        indent = "  " * level
        print(f"{indent}{widget.__class__.__name__}: {widget.winfo_class()}")
        
        for child in widget.winfo_children():
            GUIDebugHelper.print_widget_hierarchy(child, level + 1)
    
    @staticmethod
    def trace_variable_changes(var_name, var):
        """Trace variable changes for debugging"""
        def callback(*args):
            print(f"{var_name} changed to: {var.get()}")
        
        var.trace_add('write', callback)
        return callback
    
    @staticmethod
    def log_event_bindings(widget):
        """Log all event bindings for a widget"""
        for binding in widget.bindtags():
            print(f"Bindings for {binding}:")
            try:
                print(widget.bind(binding))
            except:
                print("  No bindings")

