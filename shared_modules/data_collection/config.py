# config.py - Ensure it has these optimized settings
import os

class DataCollectionConfig:
    # API settings
    BYBIT_API_KEY = os.getenv('BYBIT_API_KEY', '')
    BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET', '')
    API_BASE_URL = 'https://api.bybit.com'
    
    # Data settings
    SYMBOLS = ['BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT', 'ATOMUSDT', 'ALGOUSDT', 'VETUSDT', 'ICPUSDT', 'FILUSDT', 'AAVEUSDT', 'COMPUSDT', 'CRVUSDT', 'SNXUSDT', 'SUSHIUSDT', 'ARBUSDT', 'OPUSDT', 'NEARUSDT', 'GRTUSDT']
    #['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'LINKUSDT', 'DOGEUSDT', 'AVAXUSDT', 'UNIUSDT']  # Add more symbols as needed
    TIMEFRAMES = ['1', '5', '15', '60','120', '240']  # Add more timeframes as needed
    DATA_DIR = 'data'
    
    # Data collection mode
    # True = Keep only last 50 entries (for simple strategy testing)
    # False = Get full historical data (for AI training)
    LIMIT_TO_50_ENTRIES = True
    
    # Fetch all symbols from Bybit
    # True = Get all available symbols from Bybit
    # False = Use only symbols in SYMBOLS list
    FETCH_ALL_SYMBOLS = True
    
    # WebSocket settings
    # True = Start WebSocket and continue collecting live data
    # False = Only fetch historical data and exit (for AI training)
    ENABLE_WEBSOCKET = True
    
    # Automatic integrity check after data collection
    RUN_INTEGRITY_CHECK = False  # Disabled for speed
    
    # Automatic gap filling after data collection
    RUN_GAP_FILLING = False  # Disabled for speed
    
    # Fetch settings
    DAYS_TO_FETCH = 90 #365
    BULK_BATCH_SIZE = 20         # Number of concurrent requests to make at once 50, 20, 10
    BULK_REQUEST_DELAY_MS = 10  # Delay between requests in milliseconds (100 = 0.1 seconds) 50, 10, 200
    BULK_MAX_RETRIES = 5

    # Minimum candles required for indicators (EMA, RSI, etc)
    MIN_CANDLES = 200

