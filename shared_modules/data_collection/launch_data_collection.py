# shared_modules/data_collection/launch_data_collection.py
import sys
import json
import os
import signal
import traceback
from datetime import datetime

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def update_collection_status(running=True):
    """Update the data collection status file"""
    # Use absolute import instead of relative import
    from shared_modules.data_collection.config import DataCollectionConfig
    
    config = DataCollectionConfig()
    
    # Create data directory if it doesn't exist
    if not os.path.exists(config.DATA_DIR):
        os.makedirs(config.DATA_DIR)
    
    status_file = os.path.join(config.DATA_DIR, "collection_status.json")
    
    status = {
        'running': running,
        'last_updated': datetime.now().isoformat(),
        'pid': os.getpid()
    }
    
    with open(status_file, 'w') as f:
        json.dump(status, f)

def write_launcher_crash_log(error):
    logs_dir = os.path.join(project_root, "Logs")
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    crash_log_path = os.path.join(logs_dir, f"data_collector_launcher_crash_{timestamp}.log")
    with open(crash_log_path, "w", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} Data collector launcher crashed\n")
        f.write(f"Error: {error}\n\n")
        f.write(traceback.format_exc())
    return crash_log_path

if __name__ == "__main__":
    crash_log_path = None
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass
    try:
        # Update status to show data collection is starting
        update_collection_status(running=True)

        # Import and run the data collection GUI
        from shared_modules.data_collection.gui_monitor import main

        # Run the main function
        main()
    except Exception as e:
        print(f"Error in data collection: {e}")
        crash_log_path = write_launcher_crash_log(e)
        print(f"Crash log written to: {crash_log_path}")
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "Data Collection Crash",
                f"Data collector failed to start.\n\nError: {e}\n\nCrash log:\n{crash_log_path}",
            )
            root.destroy()
        except Exception:
            pass
    finally:
        # Update status to show data collection is stopping
        try:
            update_collection_status(running=False)
        except Exception as status_error:
            print(f"Warning: Failed to update collection status: {status_error}")
