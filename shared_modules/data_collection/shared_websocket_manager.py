# shared_websocket_manager.py
import asyncio
import logging
from typing import Dict, List, Any, Optional
from .config import DataCollectionConfig
from .websocket_handler import WebSocketHandler
from .logging_utils import setup_logging

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

class SharedWebSocketManager:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SharedWebSocketManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not SharedWebSocketManager._initialized:
            self.websocket_handler: Optional[WebSocketHandler] = None
            self.config: Optional[DataCollectionConfig] = None
            self.subscribers = []
            SharedWebSocketManager._initialized = True
    
    async def initialize(self, config: DataCollectionConfig):
        """Initialize the shared WebSocket connection"""
        if self.websocket_handler is None:
            self.config = config
            self.websocket_handler = WebSocketHandler(config)
            if config.ENABLE_WEBSOCKET:
                await self.websocket_handler.connect()
                logger.info("[SHARED] Shared WebSocket connection established")
            else:
                logger.info("[SHARED] WebSocket disabled by configuration")
    
    def get_websocket_handler(self) -> WebSocketHandler:
        """Get the shared WebSocket handler"""
        return self.websocket_handler
    
    def add_subscriber(self, callback):
        """Add a subscriber to receive WebSocket data"""
        if callback not in self.subscribers:
            self.subscribers.append(callback)
    
    def remove_subscriber(self, callback):
        """Remove a subscriber"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    async def shutdown(self):
        """Shutdown the shared WebSocket connection"""
        if self.websocket_handler:
            await self.websocket_handler.disconnect()
        logger.info("[SHARED] Shared WebSocket connection shutdown")