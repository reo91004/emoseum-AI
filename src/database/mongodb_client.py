# src/database/mongodb_client.py

import logging
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import MongoClient
from pymongo.database import Database

logger = logging.getLogger(__name__)


class MongoDBClient:
    """MongoDB client supporting both sync and async operations"""
    
    def __init__(self, mongodb_url: str, database_name: str):
        self.mongodb_url = mongodb_url
        self.database_name = database_name
        
        # Async client for API
        self._async_client: Optional[AsyncIOMotorClient] = None
        self._async_db: Optional[AsyncIOMotorDatabase] = None
        
        # Sync client for CLI
        self._sync_client: Optional[MongoClient] = None
        self._sync_db: Optional[Database] = None
    
    @property
    def async_client(self) -> AsyncIOMotorClient:
        """Get async MongoDB client (for API)"""
        if self._async_client is None:
            self._async_client = AsyncIOMotorClient(self.mongodb_url)
        return self._async_client
    
    @property
    def async_db(self) -> AsyncIOMotorDatabase:
        """Get async MongoDB database (for API)"""
        if self._async_db is None:
            self._async_db = self.async_client[self.database_name]
        return self._async_db
    
    @property
    def sync_client(self) -> MongoClient:
        """Get sync MongoDB client (for CLI)"""
        if self._sync_client is None:
            self._sync_client = MongoClient(self.mongodb_url)
        return self._sync_client
    
    @property
    def sync_db(self) -> Database:
        """Get sync MongoDB database (for CLI)"""
        if self._sync_db is None:
            self._sync_db = self.sync_client[self.database_name]
        return self._sync_db
    
    def test_connection(self) -> bool:
        """Test MongoDB connection"""
        try:
            # Test sync connection
            self.sync_client.admin.command('ping')
            logger.info(f"Successfully connected to MongoDB: {self.database_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False
    
    def close(self):
        """Close MongoDB connections"""
        if self._async_client:
            self._async_client.close()
        if self._sync_client:
            self._sync_client.close()
    
    def create_indexes(self):
        """Create necessary indexes for performance"""
        try:
            # Users collection indexes
            self.sync_db.users.create_index("user_id", unique=True)
            
            # Gallery items collection indexes
            self.sync_db.gallery_items.create_index("user_id")
            self.sync_db.gallery_items.create_index("stage")
            self.sync_db.gallery_items.create_index("created_date")
            
            # Cost tracking collection indexes
            self.sync_db.cost_tracking.create_index("user_id")
            self.sync_db.cost_tracking.create_index("timestamp")
            self.sync_db.cost_tracking.create_index([("user_id", 1), ("timestamp", -1)])
            
            # Therapy sessions collection indexes (API용)
            self.sync_db.therapy_sessions.create_index("user_id")
            self.sync_db.therapy_sessions.create_index("stage")
            self.sync_db.therapy_sessions.create_index("start_time")
            
            logger.info("MongoDB indexes created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create MongoDB indexes: {e}")
            raise