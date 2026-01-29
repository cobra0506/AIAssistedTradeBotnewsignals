# shared_modules/data_collection/main.py
import asyncio
from .console_main import console_main

def run_data_collection():
    """Entry point for data collection functionality"""
    try:
        # Import and start GUI
        print("Starting AI Assisted TradeBot GUI...")
        from . import gui_monitor
        gui_monitor.main()
    except ImportError:
        print("GUI not available, running in console mode...")
        asyncio.run(console_main())
    except Exception as e:
        print(f"Failed to start GUI: {e}")
        print("Falling back to console mode...")
        asyncio.run(console_main())