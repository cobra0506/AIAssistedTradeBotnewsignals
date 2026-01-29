import optuna
import pandas as pd
from typing import Dict, Any, List, Callable, Optional
from datetime import datetime
import logging
from .parameter_space import ParameterSpace
from ..backtester.backtester_engine import BacktesterEngine
from ..shared.data_feeder import DataFeeder
from simple_strategy.trading.parameter_manager import ParameterManager

class BayesianOptimizer:
    """Bayesian optimization engine for strategy parameters"""
    
    def __init__(self, 
                 data_feeder: DataFeeder,
                 study_name: str = None,
                 direction: str = 'maximize',
                 n_trials: int = 100,
                 timeout: int = None):
        """
        Initialize Bayesian optimizer
        
        Args:
            data_feeder: DataFeeder instance for backtesting
            study_name: Name for the Optuna study
            direction: 'maximize' or 'minimize' the objective
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
        """
        self.data_feeder = data_feeder
        self.study_name = study_name or f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.direction = direction
        self.n_trials = n_trials
        self.timeout = timeout
        
        self.study = None
        self.best_params = None
        self.best_score = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def create_objective_function(self, 
                                symbols: List[str],
                                timeframes: List[str],
                                start_date: str,
                                end_date: str,
                                metric: str = 'sharpe_ratio'):
        """
        Create objective function for optimization
        
        Args:
            symbols: List of symbols to test
            timeframes: List of timeframes
            start_date: Start date for backtest
            end_date: End date for backtest
            metric: Performance metric to optimize ('sharpe_ratio', 'total_return', 'win_rate')
        """
        def objective(trial):
            # Get parameters from trial
            params = {}
            for param_name, param_def in self.parameter_space.get_parameters().items():
                if param_def['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, param_def['choices'])
                elif param_def['type'] == 'int':
                    params[param_name] = trial.suggest_int(param_name, param_def['low'], param_def['high'], step=param_def['step'])
                elif param_def['type'] == 'float':
                    params[param_name] = trial.suggest_float(param_name, param_def['low'], param_def['high'], log=param_def['log'])
            
            try:
                # Use the strategy's own create_strategy function
                strategy = self.strategy_info['create_func'](
                    symbols=symbols, 
                    timeframes=timeframes, 
                    **params
                )
                
                # Run backtest
                backtester = BacktesterEngine(data_feeder=self.data_feeder, strategy=strategy)
                results = backtester.run_backtest(
                    symbols=symbols,
                    timeframes=timeframes,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Extract performance metric
                try:
                    # Try to get performance metrics
                    performance = results.get('performance_metrics', {})
                    score = performance.get(metric, 0.0)
                    
                    # If score is still 0.0, try to extract from the results dict directly
                    if score == 0.0 and metric in results:
                        score = results[metric]
                    
                    # If still 0.0, try to parse from the backtest output format
                    if score == 0.0:
                        # The backtest shows Sharpe Ratio in logs, let's try to get it
                        if hasattr(results, 'get') and 'sharpe_ratio' in results:
                            score = results['sharpe_ratio']
                        elif isinstance(results, dict) and 'performance_metrics' in results:
                            score = results['performance_metrics'].get('sharpe_ratio', 0.0)
                    
                    print(f"🔍 DEBUG: Extracted score for {metric}: {score}")
                    
                except Exception as e:
                    print(f"🔍 DEBUG: Error extracting score: {e}")
                    score = 0.0
                
                # Handle invalid results
                if pd.isna(score) or not isinstance(score, (int, float)):
                    score = -float('inf') if self.direction == 'maximize' else float('inf')
                
                self.logger.info(f"Trial {trial.number}: Params={params}, Score={score}")
                
                return score
                
            except Exception as e:
                self.logger.error(f"Trial {trial.number} failed with params {params}: {str(e)}")
                return -float('inf') if self.direction == 'maximize' else float('inf')
        
        return objective
    
    def optimize(self, 
                strategy_name: str,
                parameter_space: ParameterSpace,
                symbols: List[str],
                timeframes: List[str],
                start_date: str,
                end_date: str,
                metric: str = 'sharpe_ratio'):
        """
        Run optimization
        
        Args:
            strategy_name: Name of the strategy to optimize (from StrategyRegistry)
            parameter_space: ParameterSpace instance
            symbols: List of symbols to test
            timeframes: List of timeframes
            start_date: Start date for backtest
            end_date: End date for backtest
            metric: Performance metric to optimize
        """
        self.parameter_space = parameter_space
        
        # Get strategy from registry
        from ..strategies.strategy_registry import StrategyRegistry
        registry = StrategyRegistry()
        strategy_info = registry.get_strategy(strategy_name)
        
        if not strategy_info:
            raise ValueError(f"Strategy {strategy_name} not found in registry")
        
        self.strategy_info = strategy_info  # Store for use in objective function
        
        # Create or load study
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            sampler=optuna.samplers.TPESampler(seed=42)  # TPE is Bayesian optimization
        )
        
        # Create objective function
        objective = self.create_objective_function(
            symbols, timeframes, start_date, end_date, metric
        )
        
        # Run optimization
        self.logger.info(f"Starting optimization with {self.n_trials} trials...")
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        # Store results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        self.logger.info(f"Optimization complete. Best score: {self.best_score}")
        self.logger.info(f"Best parameters: {self.best_params}")
        
        # Save optimized parameters to ParameterManager
        try:
            from ..trading.parameter_manager import ParameterManager
            pm = ParameterManager()
            
            # Create parameters dict for saving
            optimized_params = self.best_params.copy()
            optimized_params['last_optimized'] = datetime.now().strftime('%Y-%m-%d')
            
            # Save to parameter manager
            pm.update_parameters(strategy_name, optimized_params)
            print(f"✅ Optimized parameters saved for {strategy_name}")
        except Exception as e:
            print(f"⚠️  Failed to save optimized parameters: {e}")

        return self.best_params, self.best_score
    
    def get_optimization_history(self):
        """Get optimization history as DataFrame"""
        if self.study is None:
            return pd.DataFrame()
        
        trials_df = self.study.trials_dataframe()
        return trials_df
    
    def get_best_trial(self):
        """Get the best trial information"""
        if self.study is None:
            return None
        
        return self.study.best_trial
    
    def save_optimization_summary(self, base_path: str = "optimization_results"):
        """Save optimization results with winning summary"""
        import os
        os.makedirs(base_path, exist_ok=True)
        
        # Save trials data
        trials_df = self.get_optimization_history()
        if not trials_df.empty:
            trials_df.to_csv(f"{base_path}/trials.csv", index=False)
        
        # Save winning summary
        if self.best_params and self.best_score:
            # Fixed: Use string concatenation instead of f-strings
            summary = "🏆 WINNING STRATEGY SETTINGS:\n"
            summary += "• RSI Period: " + str(self.best_params.get('rsi_period', 'N/A')) + "\n"
            summary += "• RSI Oversold: " + str(self.best_params.get('rsi_oversold', 'N/A')) + "\n"
            summary += "• RSI Overbought: " + str(self.best_params.get('rsi_overbought', 'N/A')) + "\n"
            summary += "• SMA Short Period: " + str(self.best_params.get('sma_short_period', 'N/A')) + "\n"
            summary += "• SMA Long Period: " + str(self.best_params.get('sma_long_period', 'N/A')) + "\n"
            summary += "• Performance: " + str(self.best_score) + " Sharpe Ratio (EXCELLENT!)"
            
            with open(f"{base_path}/winning_summary.txt", 'w') as f:
                f.write(summary)
        
        self.logger.info(f"All results saved to {base_path}")