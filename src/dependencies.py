# src/dependencies.py

import logging
from functools import lru_cache
from .database.mongodb_client import MongoDBClient
from .config import settings

logger = logging.getLogger(__name__)


@lru_cache()
def get_mongodb_client() -> MongoDBClient:
    """Get MongoDB client instance (cached)"""
    client = MongoDBClient(
        mongodb_url=settings.mongodb_url,
        database_name=settings.mongodb_database
    )
    
    # Test connection and create indexes
    if client.test_connection():
        client.create_indexes()
        logger.info("MongoDB client initialized successfully")
    else:
        raise RuntimeError("Failed to connect to MongoDB")
    
    return client


def get_sync_db():
    """Get synchronous MongoDB database for CLI usage"""
    client = get_mongodb_client()
    return client.sync_db


def get_async_db():
    """Get asynchronous MongoDB database for API usage"""
    client = get_mongodb_client()
    return client.async_db