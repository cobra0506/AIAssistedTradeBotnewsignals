# shared_modules/data_collection/launch_data_collection.py
import sys
import json
import os
from datetime import datetime

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import and run the data collection GUI
from shared_modules.data_collection.gui_monitor import main

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

if __name__ == "__main__":
    # Update status to show data collection is starting
    update_collection_status(running=True)
    
    try:
        # Run the main function
        main()
    except Exception as e:
        print(f"Error in data collection: {e}")
    finally:
        # Update status to show data collection is stopping
        update_collection_status(running=False)