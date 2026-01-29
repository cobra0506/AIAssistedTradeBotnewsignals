# websocket_handler.py - Fixed with smaller batches and better connection management
import websockets
import json
import asyncio
import ssl
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Callable
from .config import DataCollectionConfig
from .csv_manager import CSVManager
from .logging_utils import setup_logging

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

class WebSocketHandler:
    def __init__(self, config: DataCollectionConfig, symbols: List[str] = None):
        self.config = config
        self.ws_url = "wss://stream.bybit.com/v5/public/linear"
        self.running = False
        self.real_time_data = {} # Store real-time data: {symbol_timeframe: [candles]}
        self.callbacks = [] # Callback functions for processing candles
        self.debug_callbacks = [] # Debug callbacks for all messages
        self.lock = asyncio.Lock() # For thread-safe operations
        self.connection = None # Store the connection object
        self.subscription_count = 0 # Track successful subscriptions
        self.failed_subscriptions = [] # Track failed subscriptions
        self.csv_manager = CSVManager(config) # CSV manager for data operations
        
         # FIX: Use config timeframes instead of hardcoding
        self.intervals = config.TIMEFRAMES  # This will use ['1', '5', '15'] from your config
                      
        # Use provided symbols or fall back to config symbols
        if symbols:
            self.symbols = symbols
        else:
            self.symbols = config.SYMBOLS
        
        # Only initialize if WebSocket is enabled
        if not self.config.ENABLE_WEBSOCKET:
            logger.info("WebSocket disabled by configuration")
            return

    async def connect(self):
        """Connect to WebSocket and start listening in a separate task"""
        if not self.config.ENABLE_WEBSOCKET:
            logger.info("WebSocket connection skipped (ENABLE_WEBSOCKET=False)")
            return
        
        self.running = True
        logger.info(f"[CONNECT] Connecting to WebSocket: {self.ws_url}")
        
        try:
            # Establish connection
            connection = await self._connect_with_ssl()
            if connection:
                self.connection = connection
                logger.info("[OK] WebSocket connection established!")
                
                # CRITICAL FIX: Start listener in a separate task
                self.listener_task = asyncio.create_task(self._listen_for_messages(connection))
                logger.info("[OK] WebSocket listener started in background task")
                
                # Return immediately instead of blocking
                return
                
        except Exception as e:
            logger.error(f"[FAIL] Connection attempt failed: {e}")
            await asyncio.sleep(1)
            # If connection failed
            logger.error("[FAIL] All connection attempts failed.")
            await self._reconnect()

    async def _connect_with_ssl(self):
        try:
            self.ssl_context = ssl.create_default_context()
            # Optionally configure SSL context if needed (e.g., for testnet or custom certs)
            connection = await websockets.connect(
                self.ws_url,  # WebSocket endpoint
                ping_interval=None,
                ping_timeout=None,
                ssl=self.ssl_context
            )
            logger.info("WebSocket connected with SSL")
            return connection  # <-- THIS IS THE CRITICAL FIX: Return the connection object
        except Exception as e:
            logger.error(f"SSL connection error: {e}")
            raise

    async def _listen_for_messages(self, connection):
        """Listen for messages on an established connection"""
        try:
            # Subscribe to all symbols and timeframes efficiently
            await self._subscribe_to_symbols_in_batches(connection)
            
            logger.info("[OK] All subscriptions completed successfully!")
            
            # Start heartbeat task
            heartbeat_task = asyncio.create_task(self._heartbeat(connection))
            
            # Listen for messages
            try:
                async for message in connection:
                    if not self.running:
                        break
                        
                    # Call debug callbacks for all messages
                    for callback in self.debug_callbacks:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(message)
                        else:
                            callback(message)
                    
                    # Process the message
                    try:
                        await self._process_message(message)
                    except Exception as e:
                        logger.error(f"[FAIL] Error processing message: {e}")
                        
            except websockets.exceptions.ConnectionClosed as e:
                logger.error(f"[FAIL] WebSocket connection closed: {e}")
            except Exception as e:
                logger.error(f"[FAIL] Error while listening: {e}")
                
        except Exception as e:
            logger.error(f"[FAIL] Error in message listening: {e}")
        finally:
            # Cancel heartbeat task when done
            if 'heartbeat_task' in locals():
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"[FAIL] Error in heartbeat task: {e}")

    async def _subscribe_to_symbols_in_batches(self, connection):
        """Subscribe to all symbols and timeframes efficiently in large batches"""
        # Create all subscription arguments at once
        all_args = []
        for symbol in self.symbols:
            for interval in self.intervals:
                all_args.append(f"kline.{interval}.{symbol}")
        
        logger.info(f"Subscribing to {len(all_args)} total symbol-interval pairs")
        
        # Send subscriptions in large batches (like your previous code did)
        batch_size = 300  # Much larger batch size like your previous code
        for i in range(0, len(all_args), batch_size):
            batch_args = all_args[i:i + batch_size]
            subscription_msg = json.dumps({"op": "subscribe", "args": batch_args})
            
            # Simple retry logic
            for attempt in range(3):
                try:
                    await connection.send(subscription_msg)
                    logger.info(f"Subscribed to batch {i//batch_size + 1}/{(len(all_args)-1)//batch_size + 1} "
                            f"({len(batch_args)} subscriptions)")
                    break
                except Exception as e:
                    logger.error(f"Subscription error for batch {i//batch_size + 1} (attempt {attempt + 1}): {e}")
                    if attempt < 2:
                        await asyncio.sleep(1)  # Short delay before retry
                    else:
                        logger.error(f"Max retries reached for batch {i//batch_size + 1}")
                        # Continue with next batch instead of failing completely
                        break
            
            # Very short delay between batches (unlike the 10 seconds you had)
            if i + batch_size < len(all_args):  # No delay after the last batch
                await asyncio.sleep(0.5)  # Just half a second between batches
    
    logger.info("All subscription batches sent")

    async def _heartbeat(self, connection):
        while True:
            try:
                await connection.send(json.dumps({"op": "ping"}))  # <-- Use the connection parameter
                logger.debug("Sent application-level ping")
            except Exception as e:
                logger.error(f"Heartbeat ping failed: {e}")
                break  # Existing behavior: break on error to trigger reconnect in caller
            await asyncio.sleep(20)

    async def _check_connection_health(self, websocket):
        """Check if WebSocket connection is healthy with improved error handling"""
        try:
            if not websocket:
                return False
                
            # Check if connection is closed
            if hasattr(websocket, 'closed') and websocket.closed:
                return False
                
            # Check if connection state is valid
            if hasattr(websocket, 'state') and websocket.state != 1:  # 1 = STATE_OPEN
                return False
                
            # Send a ping with longer timeout to avoid false positives
            try:
                pong_waiter = await websocket.ping()
                await asyncio.wait_for(pong_waiter, timeout=10.0)
                return True
            except asyncio.TimeoutError:
                logger.warning("[HEALTH] Connection health check timeout")
                return False
            except Exception as e:
                logger.warning(f"[HEALTH] Connection health check error: {e}")
                return False
                
        except Exception as e:
            logger.error(f"[FAIL] Connection health check failed: {e}")
            return False

    async def _handle_connection_recovery(self, websocket):
        """Handle connection recovery when health check fails"""
        try:
            logger.info("[RECOVERY] Attempting connection recovery...")
            
            # Close existing connection if it exists
            if websocket and not hasattr(websocket, 'closed'):
                await websocket.close()
            
            # Short delay before reconnection attempt
            await asyncio.sleep(2)
            
            # Attempt to reconnect
            connection_attempts = [
                self._connect_with_ssl,
                self._connect_without_ssl
            ]
            
            for attempt in connection_attempts:
                try:
                    new_connection = await attempt()
                    if new_connection:
                        self.connection = new_connection
                        logger.info("[RECOVERY] Connection re-established successfully")
                        return True
                except Exception as e:
                    logger.error(f"[RECOVERY] Reconnection attempt failed: {e}")
                    await asyncio.sleep(1)
            
            logger.error("[RECOVERY] All reconnection attempts failed")
            return False
            
        except Exception as e:
            logger.error(f"[RECOVERY] Error during recovery: {e}")
            return False

    async def _process_message(self, message):
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)
            
            # DEBUG: Log the raw message structure for kline topics
            if "topic" in data and "kline" in data.get("topic", ""):
                logger.debug(f"[RAW] Received kline message: {json.dumps(data, indent=2)}")
            
            # Handle subscription confirmation
            if data.get("op") == "subscribe":
                if data.get("success") is True:
                    self.subscription_count += 1
                    logger.info(f"[OK] Subscription successful: {data.get('req_id', 'unknown')}")
                else:
                    self.failed_subscriptions.append(data.get('req_id', 'unknown'))
                    logger.error(f"[FAIL] Subscription failed: {data.get('ret_msg', 'unknown')}")
                return
            
            # Handle data messages
            if "topic" in data and "data" in data:
                topic = data["topic"]
                
                # Parse topic to get symbol and timeframe
                parts = topic.split(".")
                if len(parts) >= 3 and parts[0] == "kline":
                    timeframe = parts[1]
                    symbol = parts[2]
                    
                    # Convert Bybit timeframe back to our format
                    interval_map = {'1': '1', '5': '5', '15': '15', '60': '60', '240': '240', 'D': '1440'}
                    our_timeframe = interval_map.get(timeframe, timeframe)
                    
                    candle_data = data["data"]
                    
                    # Handle both single candle and list of candles
                    if isinstance(candle_data, list):
                        for candle in candle_data:
                            await self._process_candle(symbol, our_timeframe, candle)
                    else:
                        await self._process_candle(symbol, our_timeframe, candle_data)
                        
        except Exception as e:
            logger.error(f"[FAIL] Error processing message: {e}")

    async def _process_candle(self, symbol: str, timeframe: str, candle_data: Dict):
        """Process a single candle and save to CSV"""
        try:
            # Create candle object
            candle = {
                'timestamp': self._parse_timestamp(candle_data.get('start', 0)),
                'open': candle_data.get('open', '0'),
                'high': candle_data.get('high', '0'),
                'low': candle_data.get('low', '0'),
                'close': candle_data.get('close', '0'),
                'volume': candle_data.get('volume', '0'),
                'turnover': candle_data.get('turnover', '0'),
                'confirm': candle_data.get('confirm', False)
            }
            
            # DEBUG: Log the candle details to see what we're getting
            candle_time = datetime.fromtimestamp(candle['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
            #logger.info(f"[DEBUG] {symbol}_{timeframe} candle at {candle_time}: confirm={candle['confirm']}")
            
            # Store candle in memory
            key = f"{symbol}_{timeframe}"
            async with self.lock:
                if key not in self.real_time_data:
                    self.real_time_data[key] = []
                
                # Check if candle already exists (avoid duplicates)
                existing_timestamps = {c['timestamp'] for c in self.real_time_data[key]}
                if candle['timestamp'] not in existing_timestamps:
                    self.real_time_data[key].append(candle)
                    
                    # Only save CONFIRMED candles to CSV
                    if candle.get('confirm', False):
                        logger.info(f"[CONFIRMED] Saving confirmed candle for {symbol}_{timeframe} at {candle_time}")
                        await self._update_csv_with_candle(symbol, timeframe, candle)
                    else:
                        logger.debug(f"[UNCONFIRMED] Skipping unconfirmed candle for {symbol}_{timeframe} at {candle_time}")
                else:
                    # If candle exists but this one is confirmed, update it
                    existing_candle = next((c for c in self.real_time_data[key] if c['timestamp'] == candle['timestamp']), None)
                    if existing_candle and candle.get('confirm', False) and not existing_candle.get('confirm', False):
                        existing_candle.update(candle)
                        logger.info(f"[CONFIRMED] Updating to confirmed candle for {symbol}_{timeframe} at {candle_time}")
                        await self._update_csv_with_candle(symbol, timeframe, existing_candle)
                    else:
                        logger.debug(f"[DUPLICATE] Skipping duplicate candle for {symbol}_{timeframe} at {candle_time}")
                    
        except Exception as e:
            logger.error(f"[FAIL] Error processing candle for {symbol}_{timeframe}: {e}")

    async def _update_csv_with_candle(self, symbol: str, timeframe: str, candle: Dict):
        """Update CSV file with a new candle"""
        try:
            # Skip if WebSocket is disabled
            if not self.config.ENABLE_WEBSOCKET:
                return
                
            # Convert timestamp to datetime for CSV
            timestamp = candle['timestamp']
            dt = datetime.fromtimestamp(timestamp / 1000)  # Convert milliseconds to seconds
            
            # Create CSV row
            csv_row = {
                'timestamp': timestamp,
                'datetime': dt.strftime('%Y-%m-%d %H:%M:%S'),
                'open': float(candle['open']),
                'high': float(candle['high']),
                'low': float(candle['low']),
                'close': float(candle['close']),
                'volume': float(candle['volume'])
            }
            
            # Use the new update method instead of append_new_data
            success = self.csv_manager.update_candle(symbol, timeframe, csv_row)
            
            if success:
                logger.info(f"[OK] CSV updated for {symbol}_{timeframe} with confirmed candle at {dt}")
            else:
                logger.error(f"[FAIL] Failed to update CSV for {symbol}_{timeframe}")
                
        except Exception as e:
            logger.error(f"[FAIL] Error updating CSV for {symbol}_{timeframe}: {e}")

    def _parse_timestamp(self, timestamp):
        """Parse timestamp from Bybit format"""
        if isinstance(timestamp, str):
            return int(timestamp)
        return int(timestamp)

    async def _reconnect(self):
        """Handle reconnection logic"""
        if not self.config.ENABLE_WEBSOCKET:
            return
            
        while self.running:
            try:
                logger.info("[PROCESS] Attempting to reconnect WebSocket...")
                await asyncio.sleep(5) # Wait before retry
                await self.connect()
                break
            except Exception as e:
                logger.error(f"[FAIL] Reconnection failed: {e}")
                await asyncio.sleep(10) # Wait longer before retry

    def add_callback(self, callback: Callable):
        """Add a callback function for processing candles"""
        self.callbacks.append(callback)

    def add_debug_callback(self, callback: Callable):
        """Add a debug callback for all messages"""
        self.debug_callbacks.append(callback)

    def get_real_time_data(self, symbol=None, timeframe=None):
        """Get current real-time data
        Args:
            symbol: Optional symbol to filter by
            timeframe: Optional timeframe to filter by
        Returns:
            If symbol and timeframe provided: list of candles for that symbol/timeframe
            If neither provided: dict of all real-time data
        """
        if symbol and timeframe:
            # Get data for specific symbol/timeframe
            key = f"{symbol}_{timeframe}"
            return self.real_time_data.get(key, [])
        else:
            # Get all real-time data
            return self.real_time_data

    async def disconnect(self):
        """Properly shutdown the WebSocket connection"""
        self.running = False
        
        # Cancel the listener task if it exists
        if hasattr(self, 'listener_task') and self.listener_task:
            self.listener_task.cancel()
            try:
                await self.listener_task
            except asyncio.CancelledError:
                pass
        
        # Close the connection if it exists
        if self.connection:
            await self.connection.close()
            self.connection = None
        
        logger.info("[OK] WebSocket disconnected properly")