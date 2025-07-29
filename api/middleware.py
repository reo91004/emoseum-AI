# api/middleware.py

import logging
import time
import uuid
from typing import Callable
from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests and responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Log request
        logger.info(
            f"Request {request_id}: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        # Process request
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log response
        logger.info(
            f"Response {request_id}: {response.status_code} "
            f"completed in {process_time:.3f}s"
        )
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as e:
            logger.error(f"Unhandled error: {e}", exc_info=True)
            return Response(
                content="Internal server error",
                status_code=500,
                media_type="text/plain"
            )


def setup_middleware(app):
    """Configure all middleware for the application"""
    
    # Error handling (should be first)
    app.add_middleware(ErrorHandlingMiddleware)
    
    # Logging
    app.add_middleware(LoggingMiddleware)
    
    # CORS
    from .config import settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,  # credentials를 false로 설정
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Trusted hosts (optional, for production)
    if settings.environment == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*.emoseum.com", "emoseum.com"]
        )
    
    logger.info("Middleware configured successfully")