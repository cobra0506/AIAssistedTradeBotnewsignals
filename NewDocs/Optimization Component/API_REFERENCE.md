# Optimization Component API Reference

## Core Classes

### BayesianOptimizer

Main optimization engine using Optuna framework for Bayesian optimization.

**Current Implementation Status**: ✅ BAYESIAN OPTIMATION FULLY IMPLEMENTED
- Grid search and random search methods are planned for future implementation

#### Class Definition

```python
class BayesianOptimizer:
    def __init__(self, 
                 data_feeder: DataFeeder,
                 study_name: str = None,
                 direction: str = 'maximize',
                 n_trials: int = 100,
                 timeout: int = None)

**Note**: Current implementation uses Bayesian optimization via Optuna's TPE sampler. Grid search and random search methods are planned for future implementation.

Parameters 

     data_feeder (DataFeeder): DataFeeder instance for backtesting
     study_name (str, optional): Name for the Optuna study. Defaults to timestamp-based name
     direction (str): Optimization direction, 'maximize' or 'minimize'. Defaults to 'maximize'
     n_trials (int): Number of optimization trials. Defaults to 100
     timeout (int, optional): Timeout in seconds. Defaults to None (no timeout)
     

Methods 
optimize() 

**Current Implementation**: Uses Bayesian optimization. Grid search and random search methods are planned for future implementation.

def optimize(self,
            strategy_name: str,
            parameter_space: ParameterSpace,
            symbols: List[str],
            timeframes: List[str],
            start_date: str,
            end_date: str,
            metric: str = 'sharpe_ratio') -> Tuple[Dict[str, Any], float]

Run optimization for specified strategy and parameters. 

Parameters: 

     strategy_name (str): Name of the strategy to optimize (from StrategyRegistry)
     parameter_space (ParameterSpace): ParameterSpace instance defining optimization parameters
     symbols (List[str]): List of symbols to test
     timeframes (List[str]): List of timeframes to test
     start_date (str): Start date for backtest (YYYY-MM-DD format)
     end_date (str): End date for backtest (YYYY-MM-DD format)
     metric (str): Performance metric to optimize. Options: 'sharpe_ratio', 'total_return', 'win_rate'. Defaults to 'sharpe_ratio'
     

Returns: 

     Tuple[Dict[str, Any], float]: (best_parameters, best_score)
     

Raises: 

     ValueError: If strategy_name not found in registry
     RuntimeError: If optimization fails
     

Example: 

optimizer = BayesianOptimizer(data_feeder=data_feeder)
best_params, best_score = optimizer.optimize(
    strategy_name='Strategy_1_Trend_Following',
    parameter_space=param_space,
    symbols=['BTCUSDT'],
    timeframes=['1h'],
    start_date='2023-01-01',
    end_date='2023-12-31',
    metric='sharpe_ratio'
)

create_objective_function()

def create_objective_function(self,
                           symbols: List[str],
                           timeframes: List[str],
                           start_date: str,
                           end_date: str,
                           metric: str = 'sharpe_ratio') -> Callable

Create objective function for optimization. 

Parameters: 

     symbols (List[str]): List of symbols to test
     timeframes (List[str]): List of timeframes to test
     start_date (str): Start date for backtest
     end_date (str): End date for backtest
     metric (str): Performance metric to optimize
     

Returns: 

     Callable: Objective function compatible with Optuna
     

ParameterSpace 

Define and manage parameter spaces for optimization. 
Class Definition 

class ParameterSpace:
    def __init__(self)

Methods 
add_categorical() 

def add_categorical(self, name: str, choices: List[Any]) -> None

Add categorical parameter to parameter space. 

Parameters: 

     name (str): Parameter name
     choices (List[Any]): List of possible values
     

Example: 

param_space = ParameterSpace()
param_space.add_categorical('ma_type', ['sma', 'ema'])

add_int()

def add_int(self, name: str, low: int, high: int, step: int = 1) -> None

Add integer parameter to parameter space. 

Parameters: 

     name (str): Parameter name
     low (int): Minimum value
     high (int): Maximum value
     step (int, optional): Step size. Defaults to 1
     

Example: 

param_space.add_int('rsi_period', 5, 30, step=1)

add_float()

def add_float(self, name: str, low: float, high: float, log: bool = False) -> None

Add float parameter to parameter space. 

Parameters: 

     name (str): Parameter name
     low (float): Minimum value
     high (float): Maximum value
     log (bool, optional): Use logarithmic scale. Defaults to False
     

Example: 

param_space.add_float('rsi_oversold', 20.0, 40.0)
param_space.add_float('stop_loss', 0.01, 0.05, log=True)

get_parameters()

def get_parameters(self) -> Dict[str, Any]

Return parameter definitions. 

Returns: 

     Dict[str, Any]: Dictionary of parameter definitions
     

validate_params() 

def validate_params(self, params: Dict[str, Any]) -> bool

Validate parameter values against defined space. 

Parameters: 

     params (Dict[str, Any]): Parameters to validate
     

Returns: 

     bool: True if parameters are valid, False otherwise
     

ResultsAnalyzer 

Analyze and visualize optimization results. 
Class Definition 

class ResultsAnalyzer:
    def __init__(self, study=None)

Parameters 

     study (optuna.Study, optional): Optuna study object. Defaults to None
     

Methods 
set_study() 

def set_study(self, study) -> None

Set the Optuna study for analysis. 

Parameters: 

     study (optuna.Study): Optuna study object
     

get_best_params() 

def get_best_params(self) -> Optional[Dict[str, Any]]

Get the best parameters from the study. 

Returns: 

     Optional[Dict[str, Any]]: Best parameters or None if no study
     

get_best_score() 

def get_best_score(self) -> Optional[float]

Get the best score from the study. 

Returns: 

     Optional[float]: Best score or None if no study
     

get_parameter_importance() 

def get_parameter_importance(self) -> Optional[Dict[str, float]]

Get parameter importance scores. 

Returns: 

     Optional[Dict[str, float]]: Parameter importance scores or None if no study
     

get_trials_dataframe() 

def get_trials_dataframe(self) -> pd.DataFrame

Get optimization trials as DataFrame. 

Returns: 

     pd.DataFrame: DataFrame containing trial information
     

plot_optimization_history() 

def plot_optimization_history(self, save_path: str = None) -> None

Plot optimization history. 

Parameters: 

     save_path (str, optional): Path to save plot. Defaults to None (display only)
     

plot_parameter_importance() 

def plot_parameter_importance(self, save_path: str = None) -> None

Plot parameter importance. 

Parameters: 

     save_path (str, optional): Path to save plot. Defaults to None (display only)
     

plot_parallel_coordinate() 

def plot_parallel_coordinate(self, save_path: str = None) -> None

Plot parallel coordinate plot. 

Parameters: 

     save_path (str, optional): Path to save plot. Defaults to None (display only)
     

plot_contour() 

def plot_contour(self, params: List[str] = None, save_path: str = None) -> None

Plot contour plot. 

Parameters: 

     params (List[str], optional): List of parameters to plot. Defaults to None (all parameters)
     save_path (str, optional): Path to save plot. Defaults to None (display only)
     

generate_summary_report() 

def generate_summary_report(self, save_path: str = None) -> str

Generate a summary report of the optimization. 

Parameters: 

     save_path (str, optional): Path to save report. Defaults to None
     

Returns: 

     str: Summary report text
     

OptimizerGUI 

User-friendly GUI interface for optimization. 
Class Definition 

class OptimizerGUI:
    def __init__(self, root)

Parameters 

     root (tk.Tk): Tkinter root window
     

Methods 
create_widgets() 

def create_widgets(self) -> None

Create GUI widgets for parameter input and control. 
optimize_strategy() 

def optimize_strategy(self) -> None

Run optimization with GUI parameters. 
Utility Functions 
calculate_risk_adjusted_metrics() 

def calculate_risk_adjusted_metrics(returns: pd.Series, 
                                 risk_free_rate: float = 0.02) -> Dict[str, float]

Calculate risk-adjusted performance metrics. 

Parameters: 

     returns (pd.Series): Series of returns
     risk_free_rate (float, optional): Annual risk-free rate. Defaults to 0.02 (2%)
     

Returns: 

     Dict[str, float]: Dictionary containing:
         'total_return': Total return
         'annual_return': Annualized return
         'volatility': Annualized volatility
         'sharpe_ratio': Sharpe ratio
         'max_drawdown': Maximum drawdown
         'calmar_ratio': Calmar ratio
         'win_rate': Win rate
         
     

validate_optimization_params() 

def validate_optimization_params(params: Dict[str, Any], 
                              param_space: Dict[str, Any]) -> bool

Validate optimization parameters against parameter space. 

Parameters: 

     params (Dict[str, Any]): Parameters to validate
     param_space (Dict[str, Any]): Parameter space definition
     

Returns: 

     bool: True if parameters are valid, False otherwise
     

format_optimization_results() 

def format_optimization_results(results: Dict[str, Any]) -> pd.DataFrame

Format optimization results for display. 

Parameters: 

     results (Dict[str, Any]): Optimization results dictionary
     

Returns: 

     pd.DataFrame: Formatted results DataFrame
     

get_top_trials() 

def get_top_trials(study, n_top: int = 10) -> List[Dict[str, Any]]

Get top N trials from optimization study. 

Parameters: 

     study (optuna.Study): Optuna study object
     n_top (int, optional): Number of top trials to return. Defaults to 10
     

Returns: 

     List[Dict[str, Any]]: List of top trial dictionaries containing:
         'trial_id': Trial number
         'score': Trial score
         'params': Trial parameters
         'state': Trial state
         'datetime_start': Start datetime
         'datetime_complete': Completion datetime
         
     

Usage Examples 
Basic Optimization 

from simple_strategy.optimization import BayesianOptimizer, ParameterSpace
from simple_strategy.shared.data_feeder import DataFeeder

# Initialize components
data_feeder = DataFeeder(data_dir='data')
optimizer = BayesianOptimizer(data_feeder=data_feeder)

# Define parameter space
param_space = ParameterSpace()
param_space.add_int('rsi_period', 5, 30, step=1)
param_space.add_float('rsi_oversold', 20, 40)
param_space.add_float('rsi_overbought', 60, 80)

# Run optimization
best_params, best_score = optimizer.optimize(
    strategy_name='Strategy_1_Trend_Following',
    parameter_space=param_space,
    symbols=['BTCUSDT'],
    timeframes=['1h'],
    start_date='2023-01-01',
    end_date='2023-12-31',
    metric='sharpe_ratio'
)

print(f"Best parameters: {best_params}")
print(f"Best Sharpe ratio: {best_score:.4f}")

Results Analysis

from simple_strategy.optimization.results_analyzer import ResultsAnalyzer

# Create analyzer
analyzer = ResultsAnalyzer(study=optimizer.study)

# Get best results
best_params = analyzer.get_best_params()
best_score = analyzer.get_best_score()

# Analyze parameter importance
importance = analyzer.get_parameter_importance()
if importance:
    print("Parameter Importance:")
    for param, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {param}: {score:.4f}")

# Generate visualizations
analyzer.plot_optimization_history()
analyzer.plot_parameter_importance()

# Generate summary report
report = analyzer.generate_summary_report()
print(report)

GUI Usage

import tkinter as tk
from simple_strategy.optimization.optimizer_gui import OptimizerGUI

# Create GUI
root = tk.Tk()
app = OptimizerGUI(root)
root.mainloop()

Advanced Parameter Space

# Complex parameter space with multiple parameter types
param_space = ParameterSpace()

# Technical indicators
param_space.add_int('rsi_period', 5, 30, step=1)
param_space.add_float('rsi_oversold', 20, 40)
param_space.add_float('rsi_overbought', 60, 80)

# Moving averages
param_space.add_int('sma_short_period', 5, 20, step=1)
param_space.add_int('sma_long_period', 20, 50, step=5)
param_space.add_categorical('ma_type', ['sma', 'ema'])

# Strategy parameters
param_space.add_float('stop_loss', 0.01, 0.05, log=True)
param_space.add_float('take_profit', 0.02, 0.10)
param_space.add_int('position_size', 1, 10)

# Optimization with complex space
best_params, best_score = optimizer.optimize(
    strategy_name='AdvancedStrategy',
    parameter_space=param_space,
    symbols=['BTCUSDT', 'ETHUSDT'],
    timeframes=['1h', '4h'],
    start_date='2023-01-01',
    end_date='2023-12-31',
    metric='sharpe_ratio'
)

Error Handling 
Common Exceptions 
ValueError 

     Raised when invalid parameters are provided
     Raised when strategy name is not found in registry
     

RuntimeError 

     Raised when optimization fails
     Raised when backtest execution fails
     

Error Handling Examples 

try:
    best_params, best_score = optimizer.optimize(
        strategy_name='InvalidStrategyName',
        parameter_space=param_space,
        symbols=['BTCUSDT'],
        timeframes=['1h'],
        start_date='2023-01-01',
        end_date='2023-12-31'
    )
except ValueError as e:
    print(f"Strategy not found: {e}")
except RuntimeError as e:
    print(f"Optimization failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

Performance Considerations 
Optimization Speed 

     Bayesian Optimization: More efficient than grid search for complex parameter spaces
     Parallel Processing: Set n_jobs parameter for parallel execution
     Early Stopping: Use timeout parameter to limit optimization time
     

Memory Usage 

     Large Studies: Consider saving results to disk periodically
     Parameter Space Size: Larger parameter spaces require more memory
     Data Caching: DataFeeder handles data caching automatically
     

Best Practices 

    Start Simple: Begin with small parameter spaces and few trials 
    Validate Parameters: Always validate parameter spaces before optimization 
    Monitor Progress: Use logging to track optimization progress 
    Save Results: Save optimization results for later analysis 
    Use Appropriate Metrics: Choose metrics that align with trading objectives 

Integration Notes 
Strategy Registry Integration 

The optimization component integrates with the Strategy Registry: 

from simple_strategy.strategies.strategy_registry import StrategyRegistry

# Get available strategies
registry = StrategyRegistry()
strategies = registry.list_strategies()
print(f"Available strategies: {strategies}")

# Optimize registered strategy
best_params, best_score = optimizer.optimize(
    strategy_name='Strategy_1_Trend_Following',  # Must be in registry
    parameter_space=param_space,
    # ... other parameters
)

Backtest Engine Integration 

Optimization uses the Backtest Engine for performance evaluation: 

from simple_strategy.backtester.backtester_engine import BacktesterEngine

# Optimization automatically creates and uses BacktesterEngine
# Manual backtest with optimized parameters
backtester = BacktesterEngine(data_feeder=data_feeder, strategy=strategy)
results = backtester.run_backtest(
    symbols=['BTCUSDT'],
    timeframes=['1h'],
    start_date='2023-01-01',
    end_date='2023-12-31'
)

DataFeeder Integration 

Optimization requires DataFeeder for historical data access: 

from simple_strategy.shared.data_feeder import DataFeeder

# Initialize DataFeeder
data_feeder = DataFeeder(data_dir='data')

# Use in optimization
optimizer = BayesianOptimizer(data_feeder=data_feeder)

## Future Implementation Plans

### Additional Optimization Methods (📋 PLANNED)
The following optimization methods are planned for future implementation:

#### Grid Search Optimization
- Exhaustive search through all parameter combinations
- Guaranteed to find global optimum within parameter space
- Computationally expensive for large parameter spaces

#### Random Search Optimization
- Random sampling of parameter combinations
- More efficient than grid search for high-dimensional spaces
- Simple implementation with good performance in practice

#### Parallel Processing
- Support for parallel execution of optimization trials
- Significant performance improvement for computationally intensive strategies
- Integration with multiprocessing and distributed computing frameworks

#### Early Stopping
- Intelligent stopping criteria based on convergence detection
- Prevention of over-optimization and wasted computational resources
- Configurable sensitivity and patience parameters