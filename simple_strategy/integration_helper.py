"""
Integration Helper Module - Strategy Builder + Backtest Engine
Provides utilities and adapters to ensure seamless integration between Strategy Builder and Backtest Engine
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.backtester.backtester_engine import BacktesterEngine
from simple_strategy.backtester.risk_manager import RiskManager
from simple_strategy.shared.data_feeder import DataFeeder

logger = logging.getLogger(__name__)

class StrategyBacktestIntegration:
    """
    Integration class that simplifies the process of using Strategy Builder with Backtest Engine
    """
    
    def __init__(self, data_feeder: DataFeeder, risk_manager: Optional[RiskManager] = None):
        """
        Initialize the integration helper
        
        Args:
            data_feeder: DataFeeder instance for data access
            risk_manager: RiskManager instance (optional)
        """
        self.data_feeder = data_feeder
        self.risk_manager = risk_manager or RiskManager()
        self.backtest_config = {
            "processing_mode": "sequential",
            "batch_size": 1000,
            "memory_limit_percent": 70
        }
    
    def create_and_backtest_strategy(self, 
                                   symbols: List[str], 
                                   timeframes: List[str],
                                   strategy_config: Dict[str, Any],
                                   start_date: datetime,
                                   end_date: datetime,
                                   backtest_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a strategy using Strategy Builder and run backtest
        
        Args:
            symbols: List of trading symbols
            timeframes: List of timeframes
            strategy_config: Strategy configuration dictionary
            start_date: Backtest start date
            end_date: Backtest end date
            backtest_config: Optional backtest configuration overrides
            
        Returns:
            Backtest results dictionary
        """
        try:
            logger.info(f"🏗️ Creating strategy for symbols: {symbols}, timeframes: {timeframes}")
            
            # 1. Create Strategy Builder
            strategy_builder = StrategyBuilder(symbols, timeframes)
            
            # 2. Configure indicators
            if 'indicators' in strategy_config:
                for indicator_config in strategy_config['indicators']:
                    self._add_indicator_to_builder(strategy_builder, indicator_config)
            
            # 3. Configure signal rules
            if 'signal_rules' in strategy_config:
                for signal_config in strategy_config['signal_rules']:
                    self._add_signal_rule_to_builder(strategy_builder, signal_config)
            
            # 4. Configure risk rules
            if 'risk_rules' in strategy_config:
                for rule_type, rule_params in strategy_config['risk_rules'].items():
                    strategy_builder.add_risk_rule(rule_type, **rule_params)
            
            # 5. Configure signal combination
            if 'signal_combination' in strategy_config:
                combo_config = strategy_config['signal_combination']
                method = combo_config.get('method', 'majority_vote')
                weights = combo_config.get('weights', {})
                strategy_builder.set_signal_combination(method, weights=weights)
            
            # 6. Set strategy info
            strategy_name = strategy_config.get('name', 'IntegratedStrategy')
            strategy_version = strategy_config.get('version', '1.0.0')
            strategy_builder.set_strategy_info(strategy_name, strategy_version)
            
            # 7. Build the strategy
            logger.info(f"🔨 Building strategy: {strategy_name}")
            strategy = strategy_builder.build()
            
            # 8. Configure and run backtest
            logger.info("🚀 Running backtest...")
            config = {**self.backtest_config, **(backtest_config or {})}
            
            backtester = BacktesterEngine(
                data_feeder=self.data_feeder,
                strategy=strategy,
                risk_manager=self.risk_manager,
                config=config
            )
            
            results = backtester.run_backtest(symbols, timeframes, start_date, end_date)
            
            logger.info(f"✅ Backtest completed for strategy: {strategy_name}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Strategy creation and backtest failed: {e}")
            raise
    
    def _add_indicator_to_builder(self, builder: StrategyBuilder, indicator_config: Dict[str, Any]):
        """Add indicator to Strategy Builder based on configuration"""
        name = indicator_config['name']
        indicator_func = indicator_config['function']
        params = indicator_config.get('params', {})
        
        builder.add_indicator(name, indicator_func, **params)
        logger.debug(f"Added indicator: {name}")
    
    def _add_signal_rule_to_builder(self, builder: StrategyBuilder, signal_config: Dict[str, Any]):
        """Add signal rule to Strategy Builder based on configuration"""
        name = signal_config['name']
        signal_func = signal_config['function']
        params = signal_config.get('params', {})
        
        builder.add_signal_rule(name, signal_func, **params)
        logger.debug(f"Added signal rule: {name}")
    
    def validate_strategy_config(self, strategy_config: Dict[str, Any]) -> bool:
        """
        Validate strategy configuration before creating strategy
        
        Args:
            strategy_config: Strategy configuration dictionary
            
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check required fields
            if 'indicators' not in strategy_config or not strategy_config['indicators']:
                logger.error("Strategy must have at least one indicator")
                return False
            
            if 'signal_rules' not in strategy_config or not strategy_config['signal_rules']:
                logger.error("Strategy must have at least one signal rule")
                return False
            
            # Validate indicator configurations
            for indicator_config in strategy_config['indicators']:
                required_fields = ['name', 'function']
                for field in required_fields:
                    if field not in indicator_config:
                        logger.error(f"Indicator config missing required field: {field}")
                        return False
            
            # Validate signal rule configurations
            for signal_config in strategy_config['signal_rules']:
                required_fields = ['name', 'function']
                for field in required_fields:
                    if field not in signal_config:
                        logger.error(f"Signal rule config missing required field: {field}")
                        return False
            
            logger.info("✅ Strategy configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Strategy configuration validation failed: {e}")
            return False


# Example strategy configurations for common use cases
EXAMPLE_STRATEGIES = {
    'rsi_strategy': {
        'name': 'RSI Strategy',
        'version': '1.0.0',
        'indicators': [
            {
                'name': 'rsi',
                'function': 'rsi',  # This would be the actual function reference
                'params': {'period': 14}
            }
        ],
        'signal_rules': [
            {
                'name': 'rsi_signal',
                'function': 'overbought_oversold',  # This would be the actual function reference
                'params': {
                    'indicator': 'rsi',
                    'overbought': 70,
                    'oversold': 30
                }
            }
        ],
        'risk_rules': {
            'stop_loss': {'percent': 2.0},
            'take_profit': {'percent': 4.0}
        },
        'signal_combination': {
            'method': 'majority_vote'
        }
    },
    
    'ma_crossover_strategy': {
        'name': 'MA Crossover Strategy',
        'version': '1.0.0',
        'indicators': [
            {
                'name': 'sma_short',
                'function': 'sma',
                'params': {'period': 20}
            },
            {
                'name': 'sma_long',
                'function': 'sma',
                'params': {'period': 50}
            }
        ],
        'signal_rules': [
            {
                'name': 'ma_cross',
                'function': 'ma_crossover',
                'params': {
                    'fast_ma': 'sma_short',
                    'slow_ma': 'sma_long'
                }
            }
        ],
        'risk_rules': {
            'stop_loss': {'percent': 1.5},
            'take_profit': {'percent': 3.0}
        },
        'signal_combination': {
            'method': 'majority_vote'
        }
    },
    
    'combined_strategy': {
        'name': 'Combined RSI + MA Strategy',
        'version': '1.0.0',
        'indicators': [
            {
                'name': 'rsi',
                'function': 'rsi',
                'params': {'period': 14}
            },
            {
                'name': 'sma_short',
                'function': 'sma',
                'params': {'period': 20}
            },
            {
                'name': 'sma_long',
                'function': 'sma',
                'params': {'period': 50}
            }
        ],
        'signal_rules': [
            {
                'name': 'rsi_signal',
                'function': 'overbought_oversold',
                'params': {
                    'indicator': 'rsi',
                    'overbought': 70,
                    'oversold': 30
                }
            },
            {
                'name': 'ma_cross',
                'function': 'ma_crossover',
                'params': {
                    'fast_ma': 'sma_short',
                    'slow_ma': 'sma_long'
                }
            }
        ],
        'risk_rules': {
            'stop_loss': {'percent': 1.0},
            'take_profit': {'percent': 2.5}
        },
        'signal_combination': {
            'method': 'weighted',
            'weights': {
                'rsi_signal': 0.6,
                'ma_cross': 0.4
            }
        }
    }
}