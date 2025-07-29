# api/utils/__init__.py

"""
API Utilities Package

이 패키지는 API에서 사용되는 유틸리티 함수들과 헬퍼 클래스들을 제공한다.
- auth.py: JWT 고급 기능 및 인증 유틸리티
- exceptions.py: 커스텀 예외 클래스들
"""

from .auth import (
    TokenManager,
    generate_reset_token,
    verify_reset_token,
    hash_password,
    verify_password,
    create_api_key,
)

from .exceptions import (
    EmoseumAPIException,
    AuthenticationError,
    AuthorizationError,
    ValidationError,
    ResourceNotFoundError,
    ConflictError,
    RateLimitError,
    ServiceUnavailableError,
    TherapySessionError,
    PersonalizationError,
)

__all__ = [
    # Auth utilities
    "TokenManager",
    "generate_reset_token",
    "verify_reset_token",
    "hash_password",
    "verify_password",
    "create_api_key",
    # Exception classes
    "EmoseumAPIException",
    "AuthenticationError",
    "AuthorizationError",
    "ValidationError",
    "ResourceNotFoundError",
    "ConflictError",
    "RateLimitError",
    "ServiceUnavailableError",
    "TherapySessionError",
    "PersonalizationError",
]
