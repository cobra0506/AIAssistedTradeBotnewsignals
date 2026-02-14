import optuna
import pandas as pd
from typing import Dict, Any, List, Callable, Optional
from datetime import datetime, timedelta
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
                                metric: str = 'sharpe_ratio',
                                backtest_config: Optional[Dict[str, Any]] = None):
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
                effective_config = dict(backtest_config) if isinstance(backtest_config, dict) else {}
                effective_config['use_saved_params'] = False
                results = backtester.run_backtest(
                    symbols=symbols,
                    timeframes=timeframes,
                    start_date=start_date,
                    end_date=end_date,
                    config=effective_config
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
                
                # Penalize zero-trade runs
                total_trades = 0
                if isinstance(results, dict):
                    total_trades = results.get('total_trades', 0) or results.get('performance_metrics', {}).get('total_trades', 0)
                if total_trades == 0:
                    score = -1e12 if self.direction == 'maximize' else 1e12
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
            metric: str = 'sharpe_ratio',
            backtest_config: Optional[Dict[str, Any]] = None,
            save_params: bool = True,
            progress_callback: Optional[Callable[[int, int], None]] = None):
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
        start_time = datetime.now()
        self.last_optimization_duration_sec = None
        
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
            symbols, timeframes, start_date, end_date, metric, backtest_config
        )
        
        # Run optimization
        self.logger.info(f"Starting optimization with {self.n_trials} trials...")
        callbacks = None
        if progress_callback:
            def _optuna_progress_callback(study, trial):
                progress_callback(len(study.trials), self.n_trials or 0)
            callbacks = [_optuna_progress_callback]
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            callbacks=callbacks
        )
        
        # Store results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        end_time = datetime.now()
        self.last_optimization_duration_sec = (end_time - start_time).total_seconds()
        
        self.logger.info(f"Optimization complete. Best score: {self.best_score}")
        self.logger.info(f"Best parameters: {self.best_params}")
        self.logger.info(f"Optimization duration (sec): {self.last_optimization_duration_sec}")
        
        # Save optimized parameters to ParameterManager
        if save_params:
            try:
                from ..trading.parameter_manager import ParameterManager
                pm = ParameterManager()
                
                # Create parameters dict for saving
                optimized_params = self.best_params.copy()
                optimized_params['last_optimized'] = datetime.now().strftime('%Y-%m-%d')
                
                # Save to parameter manager
                pm.update_parameters(strategy_name, optimized_params)
                print(f"Optimized parameters saved for {strategy_name}")
            except Exception as e:
                print(f"Failed to save optimized parameters: {e}")
        return self.best_params, self.best_score, self.last_optimization_duration_sec
    
    def optimize_walk_forward(self,
            strategy_name: str,
            parameter_space: ParameterSpace,
            symbols: List[str],
            timeframes: List[str],
            start_date: str,
            end_date: str,
            metric: str = 'sharpe_ratio',
            train_days: int = 90,
            test_days: int = 30,
            step_days: Optional[int] = None,
            backtest_config: Optional[Dict[str, Any]] = None,
            progress_callback: Optional[Callable[[int, int], None]] = None):
        """Run walk-forward optimization (train then test on rolling windows)."""
        import os
        import json
        if step_days is None:
            step_days = test_days
        if step_days <= 0:
            step_days = 1
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        results = []
        base_name = self.study_name
        window_start = start_dt
        window_index = 1
        while window_start + timedelta(days=train_days + test_days) <= end_dt:
            train_start = window_start
            train_end = window_start + timedelta(days=train_days)
            test_start = train_end
            test_end = train_end + timedelta(days=test_days)
            self.study_name = f"{base_name}_wf_{window_index}"
            best_params, best_score, duration_sec = self.optimize(
                strategy_name=strategy_name,
                parameter_space=parameter_space,
                symbols=symbols,
                timeframes=timeframes,
                start_date=train_start.strftime('%Y-%m-%d'),
                end_date=train_end.strftime('%Y-%m-%d'),
                metric=metric,
                backtest_config=backtest_config,
                save_params=False,
                progress_callback=progress_callback
            )
            strategy = self.strategy_info['create_func'](
                symbols=symbols,
                timeframes=timeframes,
                **best_params
            )
            backtester = BacktesterEngine(data_feeder=self.data_feeder, strategy=strategy)
            effective_config = dict(backtest_config) if isinstance(backtest_config, dict) else {}
            effective_config['use_saved_params'] = False
            test_results = backtester.run_backtest(
                symbols=symbols,
                timeframes=timeframes,
                start_date=test_start.strftime('%Y-%m-%d'),
                end_date=test_end.strftime('%Y-%m-%d'),
                config=effective_config
            )
            test_score = test_results.get(metric, 0.0) if isinstance(test_results, dict) else 0.0
            test_total_trades = 0
            if isinstance(test_results, dict):
                test_total_trades = test_results.get('total_trades', 0) or test_results.get('performance_metrics', {}).get('total_trades', 0)
            if test_total_trades == 0:
                test_score = -1e12 if self.direction == 'maximize' else 1e12
            test_metrics = {}
            if isinstance(test_results, dict):
                test_metrics = {
                    'win_rate': test_results.get('win_rate', 0.0),
                    'sharpe_ratio': test_results.get('sharpe_ratio', 0.0),
                    'max_drawdown': test_results.get('max_drawdown', 0.0),
                    'total_return': test_results.get('total_return', 0.0),
                    'total_trades': test_results.get('total_trades', 0),
                    'start_time': test_results.get('start_time', ''),
                    'end_time': test_results.get('end_time', ''),
                    'duration': test_results.get('duration', '')
                }
            results.append({
                'window': window_index,
                'train_start': train_start.strftime('%Y-%m-%d'),
                'train_end': train_end.strftime('%Y-%m-%d'),
                'test_start': test_start.strftime('%Y-%m-%d'),
                'test_end': test_end.strftime('%Y-%m-%d'),
                'best_params': best_params,
                'train_score': best_score,
                'test_score': test_score,
                'test_metrics': test_metrics,
                'optimization_duration_sec': duration_sec
            })
            window_start = window_start + timedelta(days=step_days)
            window_index += 1
        self.study_name = base_name
        results_path = os.path.join(os.path.dirname(__file__), '..', 'optimization_results', 'walk_forward_results.json')
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        payload = {
            'strategy_name': strategy_name,
            'metric': metric,
            'symbols': symbols,
            'timeframes': timeframes,
            'train_days': train_days,
            'test_days': test_days,
            'step_days': step_days,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'windows': results
        }
        payload['saved_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        history = []
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    existing = json.load(f)
                if isinstance(existing, dict):
                    if isinstance(existing.get('history'), list):
                        history = existing.get('history', [])
                    elif 'strategy_name' in existing and 'windows' in existing:
                        history = [existing]
            except Exception:
                history = []
        
        history.append(payload)
        output = payload.copy()
        output['history'] = history
        
        with open(results_path, 'w') as f:
            json.dump(output, f, indent=4)
        return results
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
            
            if self.last_optimization_duration_sec is not None:
                summary += "\nâ€¢ Optimization Duration (sec): " + str(self.last_optimization_duration_sec)
            
            with open(f"{base_path}/winning_summary.txt", 'w') as f:
                f.write(summary)
        
        self.logger.info(f"All results saved to {base_path}")
