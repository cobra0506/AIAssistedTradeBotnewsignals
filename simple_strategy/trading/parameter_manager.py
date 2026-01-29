import json
import os
from datetime import datetime

class ParameterManager:
    def __init__(self):
        # Set the path to our parameters file
        self.params_file = os.path.join(os.path.dirname(__file__), '..', 'optimization_results', 'optimized_parameters.json')
        self.parameters = {}
        self.load_parameters()
    
    def load_parameters(self):
        """Load parameters from the JSON file"""
        try:
            if os.path.exists(self.params_file):
                with open(self.params_file, 'r') as f:
                    self.parameters = json.load(f)
            else:
                # Create empty parameters if file doesn't exist
                self.parameters = {}
                self.save_parameters()
        except Exception as e:
            print(f"Error loading parameters: {e}")
            self.parameters = {}
    
    def save_parameters(self):
        """Save parameters to the JSON file"""
        try:
            # Make sure the directory exists
            os.makedirs(os.path.dirname(self.params_file), exist_ok=True)
            
            with open(self.params_file, 'w') as f:
                json.dump(self.parameters, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving parameters: {e}")
            return False
    
    def update_parameters(self, strategy_name, params):
        """Update parameters for a specific strategy"""
        # Add the optimization date
        params['last_optimized'] = datetime.now().strftime('%Y-%m-%d')
        self.parameters[strategy_name] = params
        return self.save_parameters()
    
    def get_parameters(self, strategy_name):
        """Get parameters for a specific strategy"""
        return self.parameters.get(strategy_name, {})
    
    def get_all_strategies(self):
        """Get all strategy names that have optimized parameters"""
        return list(self.parameters.keys())