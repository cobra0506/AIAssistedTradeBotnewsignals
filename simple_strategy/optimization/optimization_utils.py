import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def calculate_risk_adjusted_metrics(returns: pd.Series, risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Calculate risk-adjusted performance metrics
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default: 2%)
    
    Returns:
        Dictionary with risk-adjusted metrics
    """
    try:
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + returns.mean()) ** 252 - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        if volatility > 0:
            sharpe_ratio = (annual_return - risk_free_rate) / volatility
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        if max_drawdown != 0:
            calmar_ratio = annual_return / abs(max_drawdown)
        else:
            calmar_ratio = 0.0
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate
        }
    
    except Exception as e:
        logger.error(f"Error calculating risk-adjusted metrics: {e}")
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'win_rate': 0.0
        }

def validate_optimization_params(params: Dict[str, Any], param_space: Dict[str, Any]) -> bool:
    """
    Validate optimization parameters against parameter space
    
    Args:
        params: Parameters to validate
        param_space: Parameter space definition
    
    Returns:
        True if parameters are valid, False otherwise
    """
    try:
        for param_name, param_value in params.items():
            if param_name not in param_space:
                logger.warning(f"Unknown parameter: {param_name}")
                return False
            
            param_def = param_space[param_name]
            
            if param_def['type'] == 'categorical':
                if param_value not in param_def['choices']:
                    logger.warning(f"Invalid value for {param_name}: {param_value}")
                    return False
            
            elif param_def['type'] == 'int':
                if not (param_def['low'] <= param_value <= param_def['high']):
                    logger.warning(f"Value {param_value} for {param_name} out of range [{param_def['low']}, {param_def['high']}]")
                    return False
            
            elif param_def['type'] == 'float':
                if not (param_def['low'] <= param_value <= param_def['high']):
                    logger.warning(f"Value {param_value} for {param_name} out of range [{param_def['low']}, {param_def['high']}]")
                    return False
        
        return True
    
    except Exception as e:
        logger.error(f"Error validating parameters: {e}")
        return False

def format_optimization_results(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Format optimization results for display
    
    Args:
        results: Optimization results dictionary
    
    Returns:
        Formatted DataFrame
    """
    try:
        data = []
        
        for trial_id, trial_data in results.items():
            row = {
                'trial_id': trial_id,
                'score': trial_data.get('score', 0.0),
                'status': trial_data.get('state', 'unknown')
            }
            
            # Add parameters
            params = trial_data.get('params', {})
            for param_name, param_value in params.items():
                row[param_name] = param_value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    except Exception as e:
        logger.error(f"Error formatting optimization results: {e}")
        return pd.DataFrame()

def get_top_trials(study, n_top: int = 10) -> List[Dict[str, Any]]:
    """
    Get top N trials from optimization study
    
    Args:
        study: Optuna study object
        n_top: Number of top trials to return
    
    Returns:
        List of top trial dictionaries
    """
    try:
        if study is None or len(study.trials) == 0:
            return []
        
        # Sort trials by value (descending for maximization, ascending for minimization)
        direction = study.direction
        sorted_trials = sorted(study.trials, 
                             key=lambda x: x.value if x.value is not None else float('-inf'),
                             reverse=(direction == 'maximize'))
        
        # Get top N trials
        top_trials = sorted_trials[:n_top]
        
        results = []
        for trial in top_trials:
            trial_data = {
                'trial_id': trial.number,
                'score': trial.value,
                'params': trial.params,
                'state': trial.state,
                'datetime_start': trial.datetime_start,
                'datetime_complete': trial.datetime_complete
            }
            results.append(trial_data)
        
        return results
    
    except Exception as e:
        logger.error(f"Error getting top trials: {e}")
        return []