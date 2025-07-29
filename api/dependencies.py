# api/dependencies.py

from typing import Optional, Annotated
from fastapi import Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
import logging

from .config import settings

logger = logging.getLogger(__name__)

# JWT 보안 스키마
security = HTTPBearer()


class AuthenticationError(HTTPException):
    """인증 실패 예외"""

    def __init__(self, detail: str = "Could not validate credentials"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


class AuthorizationError(HTTPException):
    """권한 없음 예외"""

    def __init__(self, detail: str = "Not enough permissions"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
        )


def create_access_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
    """JWT 액세스 토큰 생성"""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.jwt_access_token_expire_minutes
        )

    to_encode = {
        "user_id": user_id,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access",
    }

    try:
        encoded_jwt = jwt.encode(
            to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
        )
        logger.debug(f"JWT 토큰 생성됨 for user: {user_id}")
        return encoded_jwt
    except Exception as e:
        logger.error(f"JWT 토큰 생성 실패: {e}")
        raise AuthenticationError("Failed to create access token")


def verify_token(token: str) -> str:
    """JWT 토큰 검증 및 사용자 ID 반환"""
    try:
        payload = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )

        user_id: str = payload.get("user_id")
        token_type: str = payload.get("type")

        if user_id is None:
            logger.warning("JWT 토큰에 user_id가 없음")
            raise AuthenticationError("Invalid token: missing user_id")

        if token_type != "access":
            logger.warning(f"잘못된 토큰 타입: {token_type}")
            raise AuthenticationError("Invalid token type")

        logger.debug(f"JWT 토큰 검증 성공: {user_id}")
        return user_id

    except JWTError as e:
        logger.warning(f"JWT 토큰 검증 실패: {e}")
        raise AuthenticationError("Invalid token")
    except Exception as e:
        logger.error(f"토큰 검증 중 예상치 못한 오류: {e}")
        raise AuthenticationError("Token validation error")


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
) -> str:
    """현재 인증된 사용자 ID 반환"""
    token = credentials.credentials
    return verify_token(token)


async def get_optional_user(
    authorization: Annotated[Optional[str], Header()] = None,
) -> Optional[str]:
    """선택적 사용자 인증 (토큰이 없어도 OK)"""
    if not authorization:
        return None

    try:
        # "Bearer " 제거
        if authorization.startswith("Bearer "):
            token = authorization[7:]
            return verify_token(token)
        else:
            return None
    except AuthenticationError:
        # 토큰이 잘못되어도 None 반환 (선택적이므로)
        return None


async def get_database():
    """데이터베이스 연결 의존성"""
    try:
        from .services.database import db

        yield db
    except Exception as e:
        logger.error(f"데이터베이스 연결 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection failed",
        )


async def get_therapy_service():
    """치료 서비스 의존성"""
    try:
        from .services.therapy_service import TherapyService

        yield TherapyService()
    except Exception as e:
        logger.error(f"치료 서비스 초기화 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Therapy service initialization failed",
        )


async def get_image_service():
    """이미지 생성 서비스 의존성"""
    try:
        from .services.image_service import ImageService

        yield ImageService()
    except Exception as e:
        logger.error(f"이미지 서비스 초기화 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Image service initialization failed",
        )


def validate_content_length(
    content_length: Annotated[Optional[int], Header()] = None,
) -> None:
    """요청 크기 검증"""
    if content_length and content_length > settings.max_upload_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Request too large. Maximum size: {settings.max_upload_size} bytes",
        )


def validate_diary_text(diary_text: str) -> str:
    """일기 텍스트 유효성 검증"""
    if not diary_text or not diary_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Diary text cannot be empty"
        )

    if len(diary_text) > settings.max_diary_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Diary text too long. Maximum length: {settings.max_diary_length} characters",
        )

    return diary_text.strip()


async def check_system_health():
    """시스템 상태 확인 의존성"""
    try:
        # OpenAI API 키 확인
        if not settings.openai_api_key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OpenAI API not configured",
            )

        # Supabase 설정 확인
        if not settings.supabase_url or not settings.supabase_anon_key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database not configured",
            )

        return True

    except Exception as e:
        logger.error(f"시스템 상태 확인 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System health check failed",
        )


# 타입 별칭들
CurrentUser = Annotated[str, Depends(get_current_user)]
OptionalUser = Annotated[Optional[str], Depends(get_optional_user)]
DatabaseDep = Annotated[object, Depends(get_database)]
TherapyServiceDep = Annotated[object, Depends(get_therapy_service)]
ImageServiceDep = Annotated[object, Depends(get_image_service)]
