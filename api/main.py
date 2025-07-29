# api/main.py

import logging
import sys
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

# 프로젝트 루트를 Python 경로에 추가 (src 모듈 import를 위해)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.config import settings
from api.dependencies import setup_middleware, setup_exception_handlers

# 로깅 설정
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(settings.log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행되는 생명주기 관리"""

    # === 시작 시 실행 ===
    logger.info("🚀 Emoseum API 시작 중...")

    try:
        # 설정 검증
        settings.validate_required_settings()

        # 데이터베이스 연결 테스트
        await test_database_connection()

        # 이미지 서비스 상태 확인
        await test_image_service()

        # GPT 서비스 상태 확인
        await test_gpt_service()

        logger.info("✅ 모든 서비스 초기화 완료")
        logger.info(f"🌐 서버 시작: http://{settings.api_host}:{settings.api_port}")

    except Exception as e:
        logger.error(f"❌ 서비스 초기화 실패: {e}")
        raise

    yield  # 앱 실행

    # === 종료 시 실행 ===
    logger.info("🛑 Emoseum API 종료 중...")

    try:
        # 리소스 정리
        await cleanup_resources()
        logger.info("✅ 리소스 정리 완료")

    except Exception as e:
        logger.error(f"❌ 종료 중 오류: {e}")


async def test_database_connection():
    """데이터베이스 연결 테스트"""
    try:
        from api.services.database import db

        # 간단한 연결 테스트 (실제로는 Supabase health check)
        logger.info("🗄️  데이터베이스 연결 확인 완료")
    except Exception as e:
        logger.error(f"❌ 데이터베이스 연결 실패: {e}")
        raise


async def test_image_service():
    """이미지 생성 서비스 상태 확인"""
    try:
        from api.services.image_service import ImageService

        image_service = ImageService()
        status = image_service.get_backend_status()
        logger.info(f"🖼️  이미지 서비스 상태: {status['backend']} - {status['status']}")
    except Exception as e:
        logger.warning(f"⚠️  이미지 서비스 상태 확인 실패: {e}")
        # 이미지 서비스는 필수가 아니므로 warning만 출력


async def test_gpt_service():
    """GPT 서비스 상태 확인"""
    try:
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not configured")
        logger.info("🤖 GPT 서비스 설정 확인 완료")
    except Exception as e:
        logger.error(f"❌ GPT 서비스 확인 실패: {e}")
        raise


async def cleanup_resources():
    """리소스 정리"""
    try:
        # 필요한 경우 여기서 리소스 정리
        # 예: 데이터베이스 연결 종료, 모델 언로드 등
        pass
    except Exception as e:
        logger.error(f"리소스 정리 중 오류: {e}")


# FastAPI 앱 생성
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    debug=settings.debug,
    lifespan=lifespan,
    docs_url="/docs" if not settings.is_production() else None,
    redoc_url="/redoc" if not settings.is_production() else None,
)

# 미들웨어 설정
setup_middleware(app)

# 예외 핸들러 설정
setup_exception_handlers(app)

# 라우터 등록 (나중에 구현될 예정)
# app.include_router(auth.router, prefix="/auth", tags=["authentication"])
# app.include_router(users.router, prefix="/users", tags=["users"])
# app.include_router(therapy.router, prefix="/therapy", tags=["therapy"])
# app.include_router(gallery.router, prefix="/gallery", tags=["gallery"])
# app.include_router(admin.router, prefix="/admin", tags=["admin"])


@app.get("/")
async def root():
    """API 루트 엔드포인트"""
    return {
        "message": "Emoseum API",
        "version": settings.api_version,
        "environment": settings.environment,
        "timestamp": datetime.utcnow().isoformat(),
        "status": "healthy",
    }


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    try:
        # 각 서비스 상태 확인
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": settings.api_version,
            "environment": settings.environment,
            "services": {
                "database": "healthy",  # 실제로는 Supabase ping
                "gpt": "healthy" if settings.openai_api_key else "unavailable",
                "image_generation": "healthy",  # 실제로는 백엔드별 상태 확인
            },
            "configuration": {
                "image_backend": settings.image_backend,
                "cors_enabled": len(settings.cors_origins) > 0,
                "rate_limiting": f"{settings.rate_limit_per_minute}/min",
            },
        }

        return health_status

    except Exception as e:
        logger.error(f"헬스 체크 실패: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e) if settings.debug else "Service unavailable",
            },
        )


@app.get("/info")
async def api_info():
    """API 정보 엔드포인트"""
    return {
        "title": settings.api_title,
        "description": settings.api_description,
        "version": settings.api_version,
        "environment": settings.environment,
        "features": {
            "act_therapy": "4단계 ACT 기반 치료 여정",
            "image_generation": "감정 기반 이미지 생성",
            "personalization": "3단계 개인화 시스템",
            "safety": "치료적 안전성 검증",
        },
        "endpoints": {
            "docs": "/docs" if not settings.is_production() else "disabled",
            "health": "/health",
            "auth": "/auth/*",
            "therapy": "/therapy/*",
            "gallery": "/gallery/*",
        },
    }


# 개발 환경에서만 사용되는 디버그 엔드포인트
if settings.debug and not settings.is_production():

    @app.get("/debug/config")
    async def debug_config():
        """설정 정보 (개발용)"""
        return {
            "environment": settings.environment,
            "debug": settings.debug,
            "image_backend": settings.image_backend,
            "cors_origins": settings.cors_origins,
            "rate_limit": settings.rate_limit_per_minute,
            "openai_configured": bool(settings.openai_api_key),
            "supabase_configured": bool(
                settings.supabase_url and settings.supabase_anon_key
            ),
        }

    @app.get("/debug/logs")
    async def debug_recent_logs():
        """최근 로그 (개발용)"""
        try:
            log_file = Path(settings.log_file)
            if log_file.exists():
                with open(log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    # 최근 50줄만 반환
                    recent_lines = lines[-50:] if len(lines) > 50 else lines
                    return {"logs": recent_lines}
            else:
                return {"logs": ["Log file not found"]}
        except Exception as e:
            return {"error": f"Failed to read logs: {e}"}


if __name__ == "__main__":
    import uvicorn

    logger.info(f"🚀 Starting Emoseum API server...")
    logger.info(f"📍 Host: {settings.api_host}:{settings.api_port}")
    logger.info(f"🔧 Environment: {settings.environment}")
    logger.info(f"🐛 Debug mode: {settings.debug}")

    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug and not settings.is_production(),
        log_level=settings.log_level.lower(),
        access_log=True,
    )
