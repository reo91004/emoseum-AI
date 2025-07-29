# api/config.py

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from pathlib import Path
import logging

# 루트 디렉토리의 .env 파일 로드
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """API 설정 클래스"""

    # === API 서버 설정 ===
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    environment: str = os.getenv("ENVIRONMENT", "development")

    # === 보안 설정 ===
    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "emoseum_default_secret_key")
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30

    # === OpenAI 설정 (기존 유지) ===
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # === Supabase 설정 ===
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_anon_key: str = os.getenv("SUPABASE_ANON_KEY", "")
    supabase_service_role_key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

    # === 이미지 생성 설정 ===
    image_backend: str = os.getenv("IMAGE_BACKEND", "local")
    local_model_path: str = os.getenv(
        "LOCAL_MODEL_PATH", "runwayml/stable-diffusion-v1-5"
    )
    remote_gpu_url: Optional[str] = os.getenv("REMOTE_GPU_URL")
    remote_gpu_token: Optional[str] = os.getenv("REMOTE_GPU_TOKEN")
    colab_notebook_url: Optional[str] = os.getenv("COLAB_NOTEBOOK_URL")
    colab_access_token: Optional[str] = os.getenv("COLAB_ACCESS_TOKEN")

    # === CORS 설정 ===
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "https://yourdomain.com",
    ]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]

    # === 보안 및 제한 설정 ===
    max_upload_size: int = int(os.getenv("MAX_UPLOAD_SIZE", "10485760"))  # 10MB
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    max_diary_length: int = int(os.getenv("MAX_DIARY_LENGTH", "5000"))

    # === 로깅 설정 ===
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = os.getenv("LOG_FILE", "api.log")

    # === 기타 설정 ===
    api_title: str = "Emoseum API"
    api_description: str = "ACT-based Digital Therapy System"
    api_version: str = "2.0.0"

    def validate_required_settings(self) -> None:
        """필수 설정값 검증"""
        missing = []

        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")

        if not self.supabase_url:
            missing.append("SUPABASE_URL")

        if not self.supabase_anon_key:
            missing.append("SUPABASE_ANON_KEY")

        if missing:
            error_msg = f"Missing required environment variables: {', '.join(missing)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("✅ 모든 필수 환경변수가 설정되었습니다.")

    def get_image_backend_config(self) -> dict:
        """이미지 백엔드 설정 반환"""
        config = {
            "backend": self.image_backend,
            "local_model_path": self.local_model_path,
        }

        if self.image_backend == "remote":
            config.update(
                {
                    "remote_url": self.remote_gpu_url,
                    "remote_token": self.remote_gpu_token,
                }
            )
        elif self.image_backend == "colab":
            config.update(
                {
                    "colab_url": self.colab_notebook_url,
                    "colab_token": self.colab_access_token,
                }
            )

        return config

    def get_database_config(self) -> dict:
        """데이터베이스 설정 반환"""
        return {
            "url": self.supabase_url,
            "anon_key": self.supabase_anon_key,
            "service_role_key": self.supabase_service_role_key,
        }

    def is_production(self) -> bool:
        """프로덕션 환경 여부 확인"""
        return self.environment.lower() == "production"

    class Config:
        env_file = str(env_path)
        case_sensitive = False


# 전역 설정 인스턴스
settings = Settings()

# 시작 시 설정 검증
try:
    settings.validate_required_settings()
    logger.info(f"🚀 환경 설정 로드 완료 (.env from {env_path})")
    logger.info(f"📊 환경: {settings.environment}")
    logger.info(f"🖼️  이미지 백엔드: {settings.image_backend}")
    logger.info(f"🗄️  데이터베이스: Supabase")
except ValueError as e:
    logger.error(f"❌ 환경 설정 오류: {e}")
    raise
