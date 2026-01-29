"""
Example: Complete Strategy Builder + Backtest Engine Integration
Demonstrates how to create strategies using Strategy Builder and run backtests
"""
import os
import sys
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent  # Go up to the project root
sys.path.insert(0, str(project_root))

from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import rsi, sma, macd
from simple_strategy.strategies.signals_library import overbought_oversold, ma_crossover
from simple_strategy.backtester.backtester_engine import BacktesterEngine
from simple_strategy.backtester.risk_manager import RiskManager
from simple_strategy.shared.data_feeder import DataFeeder
from simple_strategy.integration_helper import StrategyBacktestIntegration, EXAMPLE_STRATEGIES

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def example_basic_integration():
    """Example of basic Strategy Builder + Backtest Engine integration"""
    print("🚀 Example 1: Basic Integration")
    print("=" * 50)
    
    try:
        # Create a temporary directory for test data
        import tempfile
        temp_dir = tempfile.mkdtemp()
        print(f"📁 Using temporary directory: {temp_dir}")
        
        # Create sample data for testing
        from tests.test_strategy_builder_backtest_integration import TestStrategyBuilderBacktestIntegration
        test_instance = TestStrategyBuilderBacktestIntegration()
        test_instance.temp_dir = temp_dir
        test_instance._create_test_data()
        
        # Initialize components
        data_feeder = DataFeeder(data_dir=temp_dir)
        risk_manager = RiskManager(max_risk_per_trade=0.02, max_portfolio_risk=0.10)
        
        # Create strategy using Strategy Builder
        strategy_builder = StrategyBuilder(['BTCUSDT'], ['1m'])
        strategy_builder.add_indicator('rsi', rsi, period=14)
        strategy_builder.add_signal_rule('rsi_signal', overbought_oversold, 
                                      indicator='rsi', overbought=70, oversold=30)
        strategy_builder.add_risk_rule('stop_loss', percent=2.0)
        strategy_builder.add_risk_rule('take_profit', percent=4.0)
        strategy_builder.set_strategy_info('BasicRSI', '1.0.0')
        
        strategy = strategy_builder.build()
        
        # Run backtest
        backtester = BacktesterEngine(
            data_feeder=data_feeder,
            strategy=strategy,
            risk_manager=risk_manager,
            config={"processing_mode": "sequential"}
        )
        
        results = backtester.run_backtest(
            symbols=['BTCUSDT'],
            timeframes=['1m'],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 3)
        )
        
        print(f"✅ Basic integration completed!")
        print(f"   - Results: {results}")
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
        
    except Exception as e:
        logger.error(f"❌ Basic integration example failed: {e}")
        import traceback
        print(f"❌ Full traceback: {traceback.format_exc()}")
        raise

def main():
    """Run all integration examples"""
    print("🎯 Strategy Builder + Backtest Engine Integration Examples")
    print("=" * 60)
    
    try:
        # Run examples
        example_basic_integration()
        
        print("\n🎉 Integration example completed successfully!")
        print("\n📋 Summary:")
        print("   ✅ Strategy Builder creates strategies compatible with Backtest Engine")
        print("   ✅ All components integrate seamlessly")
        
    except Exception as e:
        logger.error(f"❌ Integration examples failed: {e}")
        raise

if __name__ == '__main__':
    main()