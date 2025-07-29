# src/config.py

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Shared configuration for CLI and API"""
    
    # MongoDB settings
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_database: str = "emoseum"
    
    # OpenAI API settings
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    
    # Stable Diffusion settings
    sd_model_path: Optional[str] = None
    sd_use_local: bool = False
    sd_use_colab: bool = False
    
    # Image service settings
    image_service_type: str = "external"  # local, external, colab
    external_image_api_url: Optional[str] = None
    
    # Data directories
    data_dir: str = "data"
    gallery_images_dir: str = "data/gallery_images"
    user_loras_dir: str = "data/user_loras"
    preferences_dir: str = "data/preferences"
    
    # Logging
    log_level: str = "INFO"
    
    # API specific (when used in API context)
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_title: str = "Emoseum ACT Therapy API"
    api_version: str = "1.0.0"
    api_description: str = "REST API for ACT-based digital therapy using GPT and Stable Diffusion"
    
    # JWT settings (when used in API context)
    secret_key: str = "your-secret-key-change-this-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Environment
    environment: str = "development"
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"
    }


# Global settings instance
settings = Settings()