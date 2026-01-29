import os
import importlib.util
from typing import Dict, List, Type
import sys
import glob

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from .strategy_builder import StrategyBuilder

class StrategyRegistry:
    def __init__(self):
        self.strategies = {}
        self._discover_strategies()
    
    def _discover_strategies(self):
        # Scan for strategy files in the strategies folder
        strategy_dir = os.path.dirname(__file__)
        
        print(f"🔍 Scanning for strategies in: {strategy_dir}")
        
        # Look for files that start with "Strategy_" and end with ".py"
        pattern = os.path.join(strategy_dir, "Strategy_*.py")
        strategy_files = glob.glob(pattern)
        
        print(f"📁 Found strategy files: {strategy_files}")
        
        # Filter out non-strategy files
        strategy_files = [f for f in strategy_files if not f.endswith('strategy_builder.py') 
                         and not f.endswith('strategy_registry.py')]
        
        print(f"📁 Filtered strategy files: {strategy_files}")
        
        for file_path in strategy_files:
            file_name = os.path.basename(file_path)
            strategy_name = file_name.replace('.py', '')
            
            try:
                print(f"⚡ Loading strategy: {strategy_name}")
                
                # Create a module spec with proper package information
                spec = importlib.util.spec_from_file_location(
                    f"strategies.{strategy_name}", file_path,
                    submodule_search_locations=[strategy_dir]
                )
                
                if spec is None:
                    print(f"❌ Could not create spec for {strategy_name}")
                    continue
                    
                module = importlib.util.module_from_spec(spec)
                if module is None:
                    print(f"❌ Could not create module for {strategy_name}")
                    continue
                
                # Add the module to sys.modules before loading to fix relative imports
                sys.modules[f"strategies.{strategy_name}"] = module
                
                # Set __package__ attribute to enable relative imports
                module.__package__ = "strategies"
                
                spec.loader.exec_module(module)
                
                # Check if it has a create_strategy function
                if hasattr(module, 'create_strategy'):
                    # Get strategy parameters if available
                    params = getattr(module, 'STRATEGY_PARAMETERS', {})
                    
                    self.strategies[strategy_name] = {
                        'module': module,
                        'create_func': getattr(module, 'create_strategy'),
                        'description': getattr(module, '__doc__', 'No description'),
                        'parameters': params
                    }
                    
                    print(f"✅ Successfully loaded strategy: {strategy_name}")
                    print(f"   Parameters: {params}")
                else:
                    print(f"⚠️ Strategy {strategy_name} missing create_strategy function")
                    
            except Exception as e:
                print(f"❌ Error loading strategy {strategy_name}: {e}")
                import traceback
                traceback.print_exc()
    
    def get_all_strategies(self) -> Dict[str, dict]:
        print(f"📋 Returning {len(self.strategies)} strategies: {list(self.strategies.keys())}")
        return self.strategies
    
    def get_strategy(self, name: str):
        return self.strategies.get(name)
    
    def create_strategy_instance(self, name: str, **kwargs):
        if name in self.strategies:
            return self.strategies[name]['create_func'](**kwargs)
        return None