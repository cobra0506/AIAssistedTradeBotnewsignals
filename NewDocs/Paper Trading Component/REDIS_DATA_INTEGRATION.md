Redis Data Integration for Low-Latency Trading
1. Purpose of this Document

This document describes an architectural enhancement to the existing Paper Trading system. The goal is to integrate a Redis data bus to achieve lower latency and higher efficiency when processing live market data.

This change does NOT alter the core trading logic. The system will continue to place real orders on the Bybit demo account. This enhancement only improves how data gets from the exchange to the strategy logic.
2. The Problem: Stale Data

Currently, the paper trading engine loads historical data from CSV files. As new data arrives via the data_collector's WebSocket, the CSV files are updated. However, the paper trading engine's in-memory data becomes stale until it is manually refreshed. This delay can cause the strategy to operate on outdated information, leading to missed or delayed trading signals.
3. The Solution: A Real-Time Data Bus with Redis

We will introduce Redis as a high-speed, in-memory message broker between the data_collector and the PaperTradingEngine.
How It Works:

    Data Collector (Publisher): When the data_collector receives a new, confirmed candle from the exchange's WebSocket, it will perform two actions:
        Existing Action: Save the candle to the CSV file for persistent storage.
        New Action: Instantly publish the candle data to a specific Redis channel (e.g., data:BTCUSDT:1m).

    Paper Trading Engine (Subscriber): The PaperTradingEngine will establish a subscription to the relevant Redis channels. Instead of relying on a slow polling loop, it will listen for incoming messages.

    Event-Driven Strategy Execution: When a new candle message arrives on a Redis channel, the engine will:
        Instantly update its in-memory data for that symbol.
        Immediately run the strategy logic for that specific symbol.
        If a BUY or SELL signal is generated, it will execute the real trade on the demo account without delay.

Benefits of this Approach:

    Low Latency: Trades are triggered the instant a new candle is confirmed, not when a polling loop gets around to checking.
    High Efficiency: The system is event-driven and only uses CPU resources when new data is available, making it scalable to hundreds of symbols.
    Data Integrity: The strategy always operates on the most up-to-date, confirmed market data.

4. Implementation Steps

This is a focused change to the data flow.
Step 1: Modify the Data Collector

In the WebSocket handler that processes new candles, add the logic to publish to Redis.

File: shared_modules/data_collection/ (The file handling WebSocket messages).

# Add this to your data collectorimport redisimport json# On startup, connect to Redisredis_client = redis.Redis(host='localhost', port=6379, db=0)def on_new_candle(symbol, timeframe, candle_data):    """    This function is called when a new confirmed candle is received.    """    # ... your existing logic to save to CSV ...    # NEW: Publish to a specific Redis channel    channel_name = f"data:{symbol}:{timeframe}"    redis_client.publish(channel_name, json.dumps(candle_data))

 
Step 2: Modify the Paper Trading Engine 

Refactor the main trading loop to listen for Redis events instead of polling. 

File: simple_strategy/trading/PaperTradingEngine.py 

# Add this to your PaperTradingEngine
import redis
import threading

def __init__(self, ...):
    # ... existing initialization code ...
    self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
    self.pubsub = self.redis_client.pubsub()
    self.redis_listener_thread = None

def start_redis_listener(self, symbols_to_monitor):
    """Starts a background thread to listen for Redis messages."""
    for symbol in symbols_to_monitor:
        self.pubsub.subscribe(f"data:{symbol}:1m") # Assuming 1m timeframe

    self.redis_listener_thread = threading.Thread(target=self.listen_for_redis_data, daemon=True)
    self.redis_listener_thread.start()

def listen_for_redis_data(self):
    """This function runs in a separate thread and blocks while listening."""
    for message in self.pubsub.listen():
        if message['type'] == 'message':
            channel = message['channel'].decode('utf-8')
            data = json.loads(message['data'])
            symbol = channel.split(':')[1]
            
            # Update in-memory data and run strategy for this symbol
            self.process_new_candle_and_trade(symbol, data)

The main start_trading loop will be simplified to just start this listener and keep the main thread alive. 

Status: Proposed Enhancement
Phase: Performance Optimization
Last Updated: Dec 2025