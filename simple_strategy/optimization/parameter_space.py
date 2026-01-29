from typing import Dict, Any, List, Union
import numpy as np

class ParameterSpace:
    """Define parameter spaces for optimization"""
    
    def __init__(self):
        self.parameters = {}
    
    def add_categorical(self, name: str, choices: List[Any]):
        """Add categorical parameter (e.g., ['sma', 'ema'])"""
        self.parameters[name] = {
            'type': 'categorical',
            'choices': choices
        }
    
    def add_int(self, name: str, low: int, high: int, step: int = 1):
        """Add integer parameter (e.g., RSI period: 5-30)"""
        self.parameters[name] = {
            'type': 'int',
            'low': low,
            'high': high,
            'step': step
        }
    
    def add_float(self, name: str, low: float, high: float, log: bool = False):
        """Add float parameter (e.g., stop-loss: 0.01-0.05)"""
        self.parameters[name] = {
            'type': 'float',
            'low': low,
            'high': high,
            'log': log
        }
    
    def get_parameters(self) -> Dict[str, Any]:
        """Return parameter definitions"""
        return self.parameters
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate parameter values against defined space"""
        for name, value in params.items():
            if name not in self.parameters:
                return False
            
            param_def = self.parameters[name]
            
            if param_def['type'] == 'categorical':
                if value not in param_def['choices']:
                    return False
            elif param_def['type'] == 'int':
                if not (param_def['low'] <= value <= param_def['high']):
                    return False
            elif param_def['type'] == 'float':
                if not (param_def['low'] <= value <= param_def['high']):
                    return False
        
        return True