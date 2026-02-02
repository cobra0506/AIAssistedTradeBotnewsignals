AI Assisted TradeBot - Getting Started

Overview
This repository contains a crypto trading bot with data collection, backtesting, optimization, and paper trading support.

Documentation
All current docs live in:
- docs/
- docs/strategies/

Quick Start
1) Install dependencies
pip install -r requirements.txt

2) Run the GUI dashboard
python main.py

Common Entry Points
- Data collection: shared_modules/data_collection/launch_data_collection.py
- Backtesting: simple_strategy/backtester/backtester_engine.py
- Optimization: simple_strategy/optimization/bayesian_optimizer.py
- Paper trading: simple_strategy/trading/paper_trading_launcher.py

Notes
- Strategies live in simple_strategy/strategies/Strategy_*.py
- The Strategy Builder has known limitations; see docs/strategies/strategy_builder.md
