Testing Framework Component - API Reference 
Overview 

This document provides a comprehensive API reference for the Testing Framework Component. The framework exposes various classes, functions, and utilities for creating, executing, and managing tests across the AIAssistedTradeBot system. 
Core Testing Classes 
1. TestSignalFunctions 

Location: tests/test_all_signals.py 

Description: Comprehensive test suite for all trading signal functions with 100% coverage. 
Class Definition 

class TestSignalFunctions(unittest.TestCase):
    """Fixed test suite for signal functions"""

Methods 
setUp() 

def setUp(self):
    """Set up test data that will actually trigger signals"""

Purpose: Initialize controlled test data with predictable patterns for signal testing.
Returns: None
Side Effects: Creates instance variables for test data (prices, RSI, moving averages, etc.) 
test_overbought_oversold_signals() 

def test_overbought_oversold_signals(self):
    """Test overbought/oversold signal generation"""

Purpose: Validate the overbought_oversold() signal function.
Parameters: None (uses instance test data)
Returns: None
Validations: 

     Returns pandas Series
     Same length as input data
     Generates appropriate signals for overbought/oversold conditions
     

test_ma_crossover_signals() 

def test_ma_crossover_signals(self):
    """Test moving average crossover signals"""

Purpose: Validate the ma_crossover() signal function.
Parameters: None (uses instance test data)
Returns: None
Validations: 

     Returns pandas Series
     Detects moving average crossovers correctly
     Handles edge cases properly
     

test_macd_signals() 

def test_macd_signals(self):
    """Test MACD-based signals"""

Purpose: Validate the macd_signals() signal function.
Parameters: None (uses instance test data)
Returns: None
Validations: 

     Returns pandas Series
     Detects MACD line/signal line crossovers
     Handles various MACD scenarios
     

test_bollinger_bands_signals() 

def test_bollinger_bands_signals(self):
    """Test Bollinger Bands signals"""

Purpose: Validate the bollinger_bands_signals() signal function.
Parameters: None (uses instance test data)
Returns: None
Validations: 

     Returns pandas Series
     Detects price touches of upper/lower bands
     Handles band breakout scenarios
     

test_stochastic_signals() 

def test_stochastic_signals(self):
    """Test Stochastic signals"""

Purpose: Validate the stochastic_signals() signal function.
Parameters: None (uses instance test data)
Returns: None
Validations: 

     Returns pandas Series
     Detects Stochastic oscillator signals
     Handles %K and %D crossovers
     

test_divergence_signals() 

def test_divergence_signals(self):
    """Test divergence signals"""

Purpose: Validate the divergence_signals() signal function.
Parameters: None (uses instance test data)
Returns: None
Validations: 

     Returns pandas Series
     Detects price/indicator divergences
     Handles bullish/bearish divergence scenarios
     

test_breakout_signals() 

def test_breakout_signals(self):
    """Test breakout signals"""

Purpose: Validate the breakout_signals() signal function.
Parameters: None (uses instance test data)
Returns: None
Validations: 

     Returns pandas Series
     Detects support/resistance breakouts
     Handles various breakout scenarios
     

test_trend_strength_signals() 

def test_trend_strength_signals(self):
    """Test trend strength signals"""

Purpose: Validate the trend_strength_signals() signal function.
Parameters: None (uses instance test data)
Returns: None
Validations: 

     Returns pandas Series
     Assesses trend strength accurately
     Handles multiple timeframe analysis
     

2. TestCalculationAccuracy 

Location: tests/test_calculation_accuracy.py 

Description: Test suite for validating mathematical accuracy of trading calculations and performance metrics. 
Class Definition 

class TestCalculationAccuracy(unittest.TestCase):
    """Tests for mathematical accuracy of backtest calculations"""

Methods 
setUp() 

def setUp(self):
    """Set up test data with known outcomes"""

Purpose: Initialize controlled test data with known patterns for calculation validation.
Returns: None
Side Effects: Creates test data directory, generates OHLCV data, initializes strategy and backtester. 
tearDown() 

def tearDown(self):
    """Clean up test files"""

Purpose: Clean up temporary test files and directories.
Returns: None
Side Effects: Removes test data directory and all contained files. 
test_trade_execution_calculation() 

def test_trade_execution_calculation(self):
    """Test accuracy of trade execution calculations"""

Purpose: Validate mathematical accuracy of trade execution prices and timing.
Parameters: None (uses instance test data)
Returns: None
Validations: 

     Trade entry/exit prices match expected values
     Trade timestamps are accurate
     Signal types are correctly identified
     

test_position_sizing_calculation() 

def test_position_sizing_calculation(self):
    """Test accuracy of position sizing calculations"""

Purpose: Validate position sizing calculations including quantity and risk management.
Parameters: None (uses instance test data)
Returns: None
Validations: 

     Position quantities are calculated correctly
     Risk limits are respected
     Position sizes are reasonable for account size
     

test_performance_metrics_calculation() 

def test_performance_metrics_calculation(self):
    """Test accuracy of performance metrics calculations"""

Purpose: Validate calculation of performance metrics like total return, win rate, etc.
Parameters: None (uses instance test data)
Returns: None
Validations: 

     Total return calculation matches manual calculation
     Win rate is calculated correctly
     Performance metrics are within expected ranges
     

test_drawdown_calculation() 

def test_drawdown_calculation(self):
    """Test accuracy of drawdown calculations"""

Purpose: Validate maximum drawdown and drawdown period calculations.
Parameters: None (uses instance test data)
Returns: None
Validations: 

     Maximum drawdown is calculated correctly
     Drawdown periods are identified accurately
     Drawdown values are non-negative
     

test_sharpe_ratio_calculation() 

def test_sharpe_ratio_calculation(self):
    """Test accuracy of Sharpe ratio calculation"""

Purpose: Validate Sharpe ratio calculation for risk-adjusted performance assessment.
Parameters: None (uses instance test data)
Returns: None
Validations: 

     Sharpe ratio is calculated using correct formula
     Handles edge cases (zero volatility, negative returns)
     Results are within reasonable ranges
     

test_risk_management_calculations() 

def test_risk_management_calculations(self):
    """Test accuracy of risk management calculations"""

Purpose: Validate risk management calculations including stop-loss and position limits.
Parameters: None (uses instance test data)
Returns: None
Validations: 

     Risk metrics are calculated correctly
     Position limits are enforced
     Stop-loss levels are accurate
     

Test Runner Functions 
1. run_comprehensive_tests() 

Location: tests/run_comprehensive_tests.py 

Description: Main function to run all comprehensive test suites and generate detailed reports. 
Function Definition 

def run_comprehensive_tests():
    """Run all comprehensive tests and generate report"""

Parameters: None
Returns: Dictionary containing test results and confidence assessment

{
    'overall_success_rate': float,      # Overall success percentage
    'confidence_level': str,            # HIGH, MEDIUM-HIGH, MEDIUM, LOW
    'total_tests': int,                 # Total number of tests run
    'total_failures': int,              # Total number of test failures
    'total_errors': int,                # Total number of test errors
    'detailed_results': dict           # Detailed results by test suite
}

Test Suites Executed: 

     Signal Functions (test_all_signals.py)
     Integration Tests (test_integration.py)
     Calculation Accuracy (test_calculation_accuracy.py)
     

Output: Detailed console report with: 

     Test execution progress
     Individual suite results
     Overall confidence assessment
     Recommendations for improvements
     

Debugging and Utility Functions 
1. debug_signal_function() 

Location: tests/debug_signals.py 

Description: Utility function for debugging signal function behavior with detailed output. 
Function Definition 

def debug_signal_function(signal_func, test_data):
    """Debug signal function behavior with detailed output"""

Parameters: 

     signal_func (callable): The signal function to debug
     test_data (dict): Test data containing prices, indicators, etc.
     

Returns: Dictionary containing debug information 

{
    'input_data': dict,        # Input data used for testing
    'output_signals': pd.Series, # Generated signals
    'signal_count': int,       # Number of signals generated
    'signal_types': list,      # Types of signals generated
    'execution_time': float,   # Function execution time
    'memory_usage': float      # Memory usage during execution
}

2. generate_test_data() 

Location: tests/generate_test_data.py 

Description: Generate controlled test data with known patterns for consistent testing. 
Function Definition 

def generate_test_data(pattern_type='trending', periods=50, seed=42):
    """Generate test data with predictable patterns"""

Parameters: 

     pattern_type (str): Type of pattern to generate ('trending', 'ranging', 'volatile')
     periods (int): Number of data periods to generate
     seed (int): Random seed for reproducible results
     

Returns: Dictionary containing generated test data 

{
    'dates': pd.DatetimeIndex,  # Timestamps for data
    'prices': pd.Series,        # Price data
    'volume': pd.Series,        # Volume data
    'indicators': dict,          # Calculated indicators
    'metadata': dict            # Generation metadata
}

Pattern Types: 

     'trending': Consistent upward or downward price movement
     'ranging': Price movement within defined boundaries
     'volatile': High volatility with large price swings
     

Configuration and Constants 
1. TEST_DATA_CONFIG 

Location: tests/test_config.py 

Description: Configuration constants for test data generation and management. 
Definition 

TEST_DATA_CONFIG = {
    'default_periods': 50,           # Default number of data periods
    'seed': 42,                      # Random seed for reproducibility
    'patterns': [                     # Available test patterns
        'trending', 
        'ranging', 
        'volatile'
    ],
    'indicators': [                   # Available indicators
        'rsi', 'sma', 'ema', 'macd', 
        'bollinger', 'stochastic'
    ],
    'min_periods': 20,               # Minimum periods for indicator calculation
    'max_periods': 200,              # Maximum periods for memory management
    'price_range': (50, 200),        # Price range for test data
    'volume_range': (1000, 10000)    # Volume range for test data
}

2. TEST_ENVIRONMENT 

Location: tests/test_config.py 

Description: Test environment configuration and requirements. 
Definition 

TEST_ENVIRONMENT = {
    'python_version': '3.8+',         # Required Python version
    'dependencies': 'requirements.txt', # Dependencies file
    'data_files': 'Generated at runtime', # Data file handling
    'external_data': False,           # External data dependencies
    'memory_limit': '1GB',            # Memory limit for test execution
    'timeout': 300,                  # Test execution timeout (seconds)
    'parallel_execution': True,       # Enable parallel test execution
    'verbose_output': False           # Enable detailed test output
}

Test Result Structures 
1. Test Suite Result 

Description: Structure for individual test suite results. 
Definition 

TestSuiteResult = {
    'tests_run': int,                 # Number of tests executed
    'failures': int,                  # Number of test failures
    'errors': int,                    # Number of test errors
    'success_rate': float,            # Success rate percentage
    'execution_time': float,          # Execution time in seconds
    'memory_usage': float,            # Memory usage in MB
    'test_details': list              # Detailed test results
}

2. Comprehensive Test Report 

Description: Structure for comprehensive test report with confidence assessment. 
Definition 

ComprehensiveTestReport = {
    'overall_success_rate': float,      # Overall success percentage
    'confidence_level': str,            # Confidence level assessment
    'total_tests': int,                 # Total tests executed
    'total_failures': int,              # Total failures across all suites
    'total_errors': int,                # Total errors across all suites
    'execution_time': float,            # Total execution time
    'detailed_results': dict,          # Results by test suite
    'recommendations': list,            # Improvement recommendations
    'production_ready': bool,           # Production readiness flag
    'last_updated': str                 # Timestamp of last test run
}

Error Handling and Exceptions 
1. TestFrameworkError 

Description: Base exception class for testing framework errors. 
Definition 

class TestFrameworkError(Exception):
    """Base exception for testing framework errors"""
    pass

2. TestDataGenerationError 

Description: Exception raised when test data generation fails. 
Definition 

class TestDataGenerationError(TestFrameworkError):
    """Exception for test data generation failures"""
    pass

3. TestExecutionError 

Description: Exception raised when test execution fails. 
Definition 

class TestExecutionError(TestFrameworkError):
    """Exception for test execution failures"""
    pass

4. TestValidationError 

Description: Exception raised when test validation fails. 
Definition 

class TestValidationError(TestFrameworkError):
    """Exception for test validation failures"""
    pass

Usage Examples 
1. Running Individual Test Suites 

# Run signal function tests
python tests/test_all_signals.py

# Run calculation accuracy tests
python tests/test_calculation_accuracy.py

# Run integration tests
python tests/test_integration.py

2. Running Comprehensive Test Suite

# Run all tests with detailed reporting
python tests/run_comprehensive_tests.py

# Example output:
# 🚀 COMPREHENSIVE TEST SUITE FOR 95% CONFIDENCE
# ==================================================
# Test Run Started: 2025-06-23 14:30:15
# ==================================================
# 
# 📊 Running Signal Functions...
# --------------------------------------------------
# 🧪 Testing overbought_oversold signals...
#  Signal types generated: {'BUY', 'SELL'}
#  Total signals: 8
#  ✅ overbought_oversold signals generated successfully
# ... (more test output)
# 
# 📈 Signal Functions Results:
#  Tests Run: 13
#  Failures: 0
#  Errors: 0
#  Success Rate: 100.0%
# 
# 🏆 COMPREHENSIVE TEST REPORT
# ==================================================
# Total Test Time: 15.23 seconds
# Total Tests Run: 53
# Total Failures: 0
# Total Errors: 0
# Overall Success Rate: 100.0%
# 
# 🎯 CONFIDENCE ASSESSMENT:
# ✅ 95%+ Confidence: System is ready for production

3. Debugging Signal Functions

from tests.debug_signals import debug_signal_function
from simple_strategy.strategies.signals_library import overbought_oversold

# Create test data
test_data = {
    'rsi_series': pd.Series([25, 30, 35, 70, 75, 80, 30, 25]),
    'prices': pd.Series([100, 102, 104, 106, 108, 110, 108, 106])
}

# Debug signal function
debug_info = debug_signal_function(overbought_oversold, test_data)
print(f"Signals generated: {debug_info['signal_count']}")
print(f"Signal types: {debug_info['signal_types']}")

4. Generating Test Data

from tests.generate_test_data import generate_test_data

# Generate trending test data
trending_data = generate_test_data('trending', periods=100)
print(f"Generated {len(trending_data['prices'])} data points")

# Generate volatile test data
volatile_data = generate_test_data('volatile', periods=50)
print(f"Price range: {volatile_data['prices'].min():.2f} - {volatile_data['prices'].max():.2f}")

5. Programmatic Test Execution

from tests.run_comprehensive_tests import run_comprehensive_tests

# Run tests programmatically
results = run_comprehensive_tests()

# Check results
if results['overall_success_rate'] >= 95:
    print("✅ System ready for production")
    print(f"Confidence Level: {results['confidence_level']}")
else:
    print("⚠️ System needs improvements")
    print(f"Success Rate: {results['overall_success_rate']:.1f}%")

Best Practices 
1. Test Development 

     Use descriptive test method names
     Include comprehensive docstrings
     Test both success and error scenarios
     Validate data types and structures
     Use controlled, reproducible test data
     

2. Test Execution 

     Run tests before code changes
     Execute comprehensive test suite regularly
     Monitor test execution time and memory usage
     Review test results and confidence levels
     Address failures promptly
     

3. Test Maintenance 

     Update tests when adding new features
     Maintain 100% coverage for critical components
     Document any known limitations or issues
     Review and optimize test performance
     Keep test data and configurations updated
     

Last Updated: 2025-06-23
Version: 1.0
Status: PRODUCTION READY 
