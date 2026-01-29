# logging_utils.py - Centralized logging configuration with Windows-compatible file handling
import os
import sys
import logging
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
import codecs

def setup_logging():
    """Setup logging configuration with Windows-compatible file handlers"""
    # Create Logs directory if it doesn't exist
    logs_dir = "Logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Generate log filename with current date and time (to avoid conflicts)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    full_log_file = os.path.join(logs_dir, f"trade_bot_{current_time}.log")
    error_log_file = os.path.join(logs_dir, f"trade_bot_errors_{current_time}.log")
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Clear existing handlers - THIS IS THE FIXED PART
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)  # Fixed: Call removeHandler on logger, not handler
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Full log file handler (DEBUG and above) - Using TimedRotatingFileHandler for Windows compatibility
    full_file_handler = TimedRotatingFileHandler(
        full_log_file,
        when='midnight',  # Rotate at midnight
        interval=1,       # Every 1 day
        backupCount=7,    # Keep 7 days of logs
        encoding='utf-8'
    )
    full_file_handler.setLevel(logging.DEBUG)
    full_file_handler.setFormatter(formatter)
    logger.addHandler(full_file_handler)
    
    # Error log file handler (WARNING and above) - Using TimedRotatingFileHandler for Windows compatibility
    error_file_handler = TimedRotatingFileHandler(
        error_log_file,
        when='midnight',  # Rotate at midnight
        interval=1,       # Every 1 day
        backupCount=7,    # Keep 7 days of logs
        encoding='utf-8'
    )
    error_file_handler.setLevel(logging.WARNING)
    error_file_handler.setFormatter(formatter)
    logger.addHandler(error_file_handler)
    
    # Console handler with Unicode support
    if sys.platform == 'win32':
        # On Windows, try to use UTF-8 encoding
        try:
            # For Windows 10 and later, try to set console to UTF-8
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except:
            # Fallback to replacing problematic characters
            class UnicodeSafeStream:
                def __init__(self, stream):
                    self.stream = stream
                
                def write(self, msg):
                    # Replace emojis and other problematic Unicode characters
                    msg = msg.replace('✅', '[OK]')
                    msg = msg.replace('❌', '[FAIL]')
                    msg = msg.replace('📊', '[DATA]')
                    msg = msg.replace('🔄', '[PROCESS]')
                    msg = msg.replace('⏳', '[WAIT]')
                    msg = msg.replace('📡', '[WS]')
                    msg = msg.replace('💾', '[SAVE]')
                    msg = msg.replace('🔌', '[CONNECT]')
                    msg = msg.replace('📋', '[INFO]')
                    msg = msg.replace('🔍', '[DEBUG]')
                    msg = msg.replace('💓', '[HEARTBEAT]')
                    self.stream.write(msg)
                
                def flush(self):
                    self.stream.flush()
                
                def close(self):
                    self.stream.close()
            
            console_handler = logging.StreamHandler(UnicodeSafeStream(sys.stdout))
        else:
            console_handler = logging.StreamHandler(sys.stdout)
    else:
        # On other platforms, use standard console handler
        console_handler = logging.StreamHandler(sys.stdout)
    
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Reduce WebSocket logging to avoid excessive output
    websocket_logger = logging.getLogger('websockets')
    websocket_logger.setLevel(logging.INFO)
    
    logger.info("Logging system initialized")
    return logger