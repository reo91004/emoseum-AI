# api/database/connection.py

import os
import logging
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class MongoDB:
    """MongoDB connection manager with async support"""
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        
        # MongoDB configuration
        self.mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
        self.database_name = os.getenv("MONGODB_DATABASE", "emoseum")
        self.max_pool_size = int(os.getenv("MONGODB_MAX_POOL_SIZE", "10"))
        self.min_pool_size = int(os.getenv("MONGODB_MIN_POOL_SIZE", "1"))
        
    async def connect(self) -> None:
        """Connect to MongoDB"""
        try:
            logger.info(f"Connecting to MongoDB at {self.mongodb_url}...")
            
            self.client = AsyncIOMotorClient(
                self.mongodb_url,
                maxPoolSize=self.max_pool_size,
                minPoolSize=self.min_pool_size,
                serverSelectionTimeoutMS=5000
            )
            
            # Test connection
            await self.client.admin.command('ping')
            
            self.database = self.client[self.database_name]
            logger.info(f"Successfully connected to MongoDB database: {self.database_name}")
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise RuntimeError(f"MongoDB connection failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during MongoDB connection: {e}")
            raise
            
    async def disconnect(self) -> None:
        """Disconnect from MongoDB"""
        if self.client is not None:
            logger.info("Disconnecting from MongoDB...")
            self.client.close()
            self.client = None
            self.database = None
            logger.info("MongoDB connection closed")
            
    async def get_database(self) -> AsyncIOMotorDatabase:
        """Get database instance"""
        if self.database is None:
            await self.connect()
        return self.database
    
    async def health_check(self) -> bool:
        """Check MongoDB connection health"""
        try:
            if self.client is not None:
                await self.client.admin.command('ping')
                return True
            return False
        except Exception:
            return False


# Global MongoDB instance
mongodb = MongoDB()


async def get_database() -> AsyncIOMotorDatabase:
    """Dependency for getting database in FastAPI"""
    return await mongodb.get_database()