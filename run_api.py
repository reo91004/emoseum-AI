# run_api.py

"""
FastAPI 서버 실행 스크립트
"""

import uvicorn
import os
import sys
from pathlib import Path

# Add api directory to path
sys.path.append(str(Path(__file__).parent / "api"))

from api.config import settings


def main():
    """Run the FastAPI server"""

    # Ensure data directory exists
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(exist_ok=True)

    print(f"🚀 Starting Emoseum API server...")
    print(f"📍 Host: {settings.api_host}:{settings.api_port}")
    print(f"🌍 Environment: {settings.environment}")
    print(f"📊 MongoDB: {settings.mongodb_url}/{settings.mongodb_database}")
    print(f"🖼️  Image Service: {settings.image_generation_service}")

    # Run server
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    main()
