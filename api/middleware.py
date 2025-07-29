# api/middleware.py

import time
import logging
from typing import Dict, Set
from datetime import datetime, timedelta
from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import asyncio
from collections import defaultdict

from .config import settings

logger = logging.getLogger(__name__)


class RateLimitMiddleware:
    """Rate Limiting 미들웨어"""

    def __init__(self, calls: int = 60, period: int = 60):
        self.calls = calls
        self.period = period
        self.clients: Dict[str, list] = defaultdict(list)

    async def __call__(self, request: Request, call_next):
        client_ip = self._get_client_ip(request)
        current_time = time.time()

        # 클라이언트의 요청 기록 정리 (기간 지난 것들 제거)
        self.clients[client_ip] = [
            req_time
            for req_time in self.clients[client_ip]
            if current_time - req_time < self.period
        ]

        # 요청 수 확인
        if len(self.clients[client_ip]) >= self.calls:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded", "retry_after": self.period},
                headers={"Retry-After": str(self.period)},
            )

        # 현재 요청 기록
        self.clients[client_ip].append(current_time)

        response = await call_next(request)
        return response

    def _get_client_ip(self, request: Request) -> str:
        """클라이언트 IP 주소 추출"""
        # X-Forwarded-For 헤더에서 실제 IP 추출 (프록시 환경 대응)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"


class RequestLoggingMiddleware:
    """요청 로깅 미들웨어"""

    async def __call__(self, request: Request, call_next):
        start_time = time.time()

        # 요청 로깅
        logger.info(
            f"📥 {request.method} {request.url} - Client: {self._get_client_ip(request)}"
        )

        # 요청 처리
        response = await call_next(request)

        # 응답 시간 계산
        process_time = time.time() - start_time

        # 응답 로깅
        logger.info(
            f"📤 {request.method} {request.url} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.3f}s"
        )

        # 응답 헤더에 처리 시간 추가
        response.headers["X-Process-Time"] = str(process_time)

        return response

    def _get_client_ip(self, request: Request) -> str:
        """클라이언트 IP 주소 추출"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"


class SecurityHeadersMiddleware:
    """보안 헤더 추가 미들웨어"""

    async def __call__(self, request: Request, call_next):
        response = await call_next(request)

        # 보안 헤더 추가
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "X-API-Version": settings.api_version,
        }

        for header, value in security_headers.items():
            response.headers[header] = value

        return response


class ErrorHandlingMiddleware:
    """전역 에러 처리 미들웨어"""

    async def __call__(self, request: Request, call_next):
        try:
            return await call_next(request)
        except HTTPException:
            # HTTPException은 그대로 전파
            raise
        except Exception as e:
            logger.error(f"Unhandled exception: {str(e)}", exc_info=True)

            # 프로덕션에서는 상세 에러 정보 숨김
            if settings.is_production():
                detail = "Internal server error"
            else:
                detail = f"Internal server error: {str(e)}"

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "success": False,
                    "error_code": "INTERNAL_SERVER_ERROR",
                    "error_message": detail,
                    "timestamp": datetime.utcnow().isoformat(),
                    "path": str(request.url),
                },
            )


def setup_middleware(app: FastAPI) -> None:
    """모든 미들웨어 설정"""

    # 1. CORS 설정 (Unity 클라이언트 지원)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )
    logger.info(f"✅ CORS 설정 완료 - Origins: {settings.cors_origins}")

    # 2. 신뢰할 수 있는 호스트 설정 (프로덕션 환경)
    if settings.is_production():
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=[
                "yourdomain.com",
                "*.yourdomain.com",
                "localhost",
                "127.0.0.1",
            ],
        )
        logger.info("✅ Trusted Host 미들웨어 설정 완료")

    # 3. Rate Limiting
    rate_limit_middleware = RateLimitMiddleware(
        calls=settings.rate_limit_per_minute, period=60
    )
    app.middleware("http")(rate_limit_middleware)
    logger.info(f"✅ Rate Limiting 설정 완료 - {settings.rate_limit_per_minute}/min")

    # 4. 보안 헤더
    security_middleware = SecurityHeadersMiddleware()
    app.middleware("http")(security_middleware)
    logger.info("✅ 보안 헤더 미들웨어 설정 완료")

    # 5. 요청 로깅 (개발 환경에서만)
    if settings.debug or not settings.is_production():
        logging_middleware = RequestLoggingMiddleware()
        app.middleware("http")(logging_middleware)
        logger.info("✅ 요청 로깅 미들웨어 설정 완료")

    # 6. 전역 에러 처리
    error_middleware = ErrorHandlingMiddleware()
    app.middleware("http")(error_middleware)
    logger.info("✅ 에러 처리 미들웨어 설정 완료")


# 커스텀 예외 핸들러들
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Validation 에러 핸들러"""
    logger.warning(f"Validation error on {request.url}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error_code": "VALIDATION_ERROR",
            "error_message": "Request validation failed",
            "details": exc.errors() if hasattr(exc, "errors") else str(exc),
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url),
        },
    )


async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP 예외 핸들러"""
    logger.warning(f"HTTP exception on {request.url}: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_code": f"HTTP_{exc.status_code}",
            "error_message": exc.detail,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url),
        },
    )


async def authentication_exception_handler(request: Request, exc: HTTPException):
    """인증 예외 전용 핸들러"""
    logger.warning(f"Authentication error on {request.url}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_code": "AUTHENTICATION_FAILED",
            "error_message": exc.detail,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url),
        },
        headers={"WWW-Authenticate": "Bearer"},
    )


def setup_exception_handlers(app: FastAPI) -> None:
    """예외 핸들러 설정"""

    # Validation 에러 핸들러
    app.add_exception_handler(RequestValidationError, validation_exception_handler)

    # HTTP 예외 핸들러
    app.add_exception_handler(HTTPException, http_exception_handler)

    # 401 인증 에러 전용 핸들러
    @app.exception_handler(401)
    async def custom_auth_exception_handler(request: Request, exc: HTTPException):
        return await authentication_exception_handler(request, exc)

    logger.info("✅ 예외 핸들러 설정 완료")
