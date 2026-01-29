# example_optimization.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from simple_strategy.optimization import BayesianOptimizer, ParameterSpace
from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.shared.data_feeder import DataFeeder

def main():
    # Initialize data feeder
    data_feeder = DataFeeder(data_dir='data')
    
    # Create parameter space - customized for your indicators and signals
    param_space = ParameterSpace()
    
    # RSI parameters
    param_space.add_int('rsi_period', 5, 30, step=1)
    param_space.add_float('rsi_oversold', 20, 40)
    param_space.add_float('rsi_overbought', 60, 80)
    
    # Moving average parameters
    param_space.add_int('sma_short_period', 5, 20, step=1)
    param_space.add_int('sma_long_period', 20, 50, step=5)
    
    # EMA parameters
    param_space.add_int('ema_short_period', 5, 20, step=1)
    param_space.add_int('ema_long_period', 20, 50, step=5)
    
    # Stochastic parameters
    param_space.add_int('stoch_k_period', 5, 20, step=1)
    param_space.add_int('stoch_d_period', 3, 10, step=1)
    
    # MACD parameters
    param_space.add_int('macd_fast', 8, 20, step=1)
    param_space.add_int('macd_slow', 20, 40, step=1)
    param_space.add_int('macd_signal', 5, 15, step=1)
    
    # Create optimizer
    optimizer = BayesianOptimizer(
        data_feeder=data_feeder,
        study_name='multi_strategy_optimization',
        direction='maximize',
        n_trials=100,  # Number of optimization trials
        timeout=7200   # 2 hour timeout
    )
    
    # Run optimization
    best_params, best_score = optimizer.optimize(
        strategy_name='Strategy_1_Trend_Following',  # <-- CHANGED TO THIS
        parameter_space=param_space,
        symbols=['BTCUSDT'],
        timeframes=['1h'],
        start_date='2023-01-01',
        end_date='2023-12-31',
        metric='sharpe_ratio'  # Could also be 'total_return', 'win_rate', etc.
    )
    
    print(f"Best parameters: {best_params}")
    print(f"Best Sharpe ratio: {best_score}")
    
    # Get optimization history
    history = optimizer.get_optimization_history()
    print(f"Optimization completed: {len(history)} trials")
    
    # Save results
    history.to_csv('optimization_results.csv', index=False)
    print("Results saved to optimization_results.csv")

if __name__ == "__main__":
    main()