# api/config.py

import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """API configuration settings"""
    
    # API Settings
    api_title: str = "Emoseum ACT Therapy API"
    api_version: str = "1.0.0"
    api_description: str = "API for ACT-based digital therapy system with personalized image generation"
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    
    # Security
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:8000", "http://localhost:8000"],
        env="CORS_ORIGINS"
    )
    
    # Database
    mongodb_url: str = Field(default="mongodb://localhost:27017", env="MONGODB_URL")
    mongodb_database: str = Field(default="emoseum", env="MONGODB_DATABASE")
    
    # Image Generation
    image_generation_service: str = Field(default="local_gpu", env="IMAGE_GENERATION_SERVICE")
    external_gpu_endpoint: Optional[str] = Field(default=None, env="EXTERNAL_GPU_ENDPOINT")
    colab_notebook_url: Optional[str] = Field(default=None, env="COLAB_NOTEBOOK_URL")
    
    # OpenAI
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    
    # Paths
    data_dir: str = Field(default="data", env="EMOSEUM_DATA_DIR")
    logs_dir: str = Field(default="logs", env="EMOSEUM_LOGS_DIR")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    enable_file_logging: bool = Field(default=True, env="ENABLE_FILE_LOGGING")
    
    # Safety
    enable_safety_checks: bool = Field(default=True, env="ENABLE_SAFETY_CHECKS")
    max_prompt_length: int = Field(default=1000, env="MAX_PROMPT_LENGTH")
    
    # Rate Limiting
    rate_limit_calls: int = Field(default=10, env="RATE_LIMIT_CALLS")
    rate_limit_period: int = Field(default=60, env="RATE_LIMIT_PERIOD")
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"  # 정의되지 않은 필드 무시
    }


# Create settings instance
settings = Settings()