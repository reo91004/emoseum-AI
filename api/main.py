# api/main.py

import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pathlib import Path

from .config import settings
from .middleware import setup_middleware
from .database.connection import mongodb
from .database.collections import Collections
from .dependencies import initialize_act_therapy_system

# Import routers
from .routers import auth, users, therapy, gallery, training, system

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for the application"""
    # Startup
    logger.info("Starting Emoseum API...")
    
    try:
        # Connect to MongoDB
        await mongodb.connect()
        
        # Create database indexes
        database = await mongodb.get_database()
        await Collections.create_indexes(database)
        
        # Initialize ACT Therapy System
        await initialize_act_therapy_system()
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Emoseum API...")
    await mongodb.disconnect()
    logger.info("API shutdown completed")


# Create FastAPI application
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    lifespan=lifespan,
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None
)

# Setup middleware
setup_middleware(app)

# Include routers
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(therapy.router)
app.include_router(gallery.router)
app.include_router(training.router)
app.include_router(system.router)


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirect to docs"""
    if settings.environment != "production":
        return RedirectResponse(url="/docs")
    return {"message": "Emoseum ACT Therapy API", "version": settings.api_version}


@app.get("/health", tags=["Health"])
async def health_check():
    """Simple health check"""
    return {
        "status": "healthy",
        "api_version": settings.api_version,
        "environment": settings.environment
    }


# Exception handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "detail": "The requested endpoint does not exist"}


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return {"error": "Internal server error", "detail": "An unexpected error occurred"}


if __name__ == "__main__":
    import uvicorn
    
    # Ensure data directory exists
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(exist_ok=True)
    
    # Run the application
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower()
    )