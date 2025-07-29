# api/utils/auth.py

import secrets
import bcrypt
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from jose import jwt, JWTError
from dataclasses import dataclass
import hashlib
import base64

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class TokenPayload:
    """JWT 토큰 페이로드 구조"""

    user_id: str
    token_type: str  # access, refresh, reset, api
    issued_at: datetime
    expires_at: datetime
    permissions: List[str] = None
    metadata: Dict[str, Any] = None


class TokenManager:
    """고급 JWT 토큰 관리 클래스"""

    def __init__(self):
        self.secret_key = settings.jwt_secret_key
        self.algorithm = settings.jwt_algorithm
        self.access_token_expire_minutes = settings.jwt_access_token_expire_minutes
        self.refresh_token_expire_days = 30
        self.reset_token_expire_minutes = 15

    def create_access_token(
        self,
        user_id: str,
        permissions: Optional[List[str]] = None,
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        """액세스 토큰 생성"""

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=self.access_token_expire_minutes
            )

        payload = {
            "user_id": user_id,
            "token_type": "access",
            "iat": datetime.utcnow(),
            "exp": expire,
            "permissions": permissions or ["user:basic"],
            "jti": secrets.token_urlsafe(16),  # JWT ID for revocation
        }

        try:
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            logger.debug(f"액세스 토큰 생성: {user_id}")
            return token
        except Exception as e:
            logger.error(f"액세스 토큰 생성 실패: {e}")
            raise

    def create_refresh_token(self, user_id: str) -> str:
        """리프레시 토큰 생성"""

        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)

        payload = {
            "user_id": user_id,
            "token_type": "refresh",
            "iat": datetime.utcnow(),
            "exp": expire,
            "jti": secrets.token_urlsafe(32),
        }

        try:
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            logger.debug(f"리프레시 토큰 생성: {user_id}")
            return token
        except Exception as e:
            logger.error(f"리프레시 토큰 생성 실패: {e}")
            raise

    def verify_token(
        self, token: str, expected_type: Optional[str] = None
    ) -> TokenPayload:
        """토큰 검증 및 페이로드 반환"""

        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # 토큰 타입 확인
            token_type = payload.get("token_type", "access")
            if expected_type and token_type != expected_type:
                raise ValueError(f"Expected {expected_type} token, got {token_type}")

            # 페이로드 구조화
            token_payload = TokenPayload(
                user_id=payload["user_id"],
                token_type=token_type,
                issued_at=datetime.fromtimestamp(payload["iat"]),
                expires_at=datetime.fromtimestamp(payload["exp"]),
                permissions=payload.get("permissions", []),
                metadata=payload.get("metadata", {}),
            )

            logger.debug(f"토큰 검증 성공: {token_payload.user_id} ({token_type})")
            return token_payload

        except jwt.ExpiredSignatureError:
            logger.warning("만료된 토큰")
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError as e:
            logger.warning(f"유효하지 않은 토큰: {e}")
            raise ValueError("Invalid token")
        except Exception as e:
            logger.error(f"토큰 검증 실패: {e}")
            raise ValueError(f"Token verification failed: {str(e)}")

    def refresh_access_token(self, refresh_token: str) -> Dict[str, str]:
        """리프레시 토큰으로 새 액세스 토큰 생성"""

        try:
            # 리프레시 토큰 검증
            payload = self.verify_token(refresh_token, expected_type="refresh")

            # 새 액세스 토큰 생성
            new_access_token = self.create_access_token(
                user_id=payload.user_id, permissions=payload.permissions
            )

            # 새 리프레시 토큰도 생성 (보안 강화)
            new_refresh_token = self.create_refresh_token(payload.user_id)

            logger.info(f"토큰 갱신 완료: {payload.user_id}")

            return {
                "access_token": new_access_token,
                "refresh_token": new_refresh_token,
                "token_type": "bearer",
            }

        except Exception as e:
            logger.error(f"토큰 갱신 실패: {e}")
            raise

    def create_reset_token(self, user_id: str, email: str) -> str:
        """비밀번호 재설정 토큰 생성"""

        expire = datetime.utcnow() + timedelta(minutes=self.reset_token_expire_minutes)

        payload = {
            "user_id": user_id,
            "email": email,
            "token_type": "reset",
            "iat": datetime.utcnow(),
            "exp": expire,
            "jti": secrets.token_urlsafe(16),
        }

        try:
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            logger.info(f"비밀번호 재설정 토큰 생성: {user_id}")
            return token
        except Exception as e:
            logger.error(f"재설정 토큰 생성 실패: {e}")
            raise

    def decode_reset_token(self, token: str) -> Dict[str, str]:
        """비밀번호 재설정 토큰 디코딩"""

        try:
            payload = self.verify_token(token, expected_type="reset")
            return {
                "user_id": payload.user_id,
                "email": payload.metadata.get("email", ""),
            }
        except Exception as e:
            logger.error(f"재설정 토큰 디코딩 실패: {e}")
            raise

    def create_api_key(
        self, user_id: str, name: str, permissions: List[str]
    ) -> Dict[str, str]:
        """API 키 생성 (장기간 유효한 토큰)"""

        # API 키는 만료되지 않음 (또는 매우 긴 기간)
        expire = datetime.utcnow() + timedelta(days=365)

        payload = {
            "user_id": user_id,
            "token_type": "api_key",
            "iat": datetime.utcnow(),
            "exp": expire,
            "permissions": permissions,
            "api_key_name": name,
            "jti": secrets.token_urlsafe(32),
        }

        try:
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

            # API 키는 특별한 접두사를 붙여서 식별하기 쉽게 함
            api_key = f"emoseum_api_{base64.urlsafe_b64encode(token.encode()).decode()}"

            logger.info(f"API 키 생성: {user_id} ({name})")

            return {
                "api_key": api_key,
                "name": name,
                "permissions": permissions,
                "created_at": datetime.utcnow().isoformat(),
                "expires_at": expire.isoformat(),
            }

        except Exception as e:
            logger.error(f"API 키 생성 실패: {e}")
            raise

    def verify_api_key(self, api_key: str) -> TokenPayload:
        """API 키 검증"""

        try:
            # API 키 접두사 제거
            if not api_key.startswith("emoseum_api_"):
                raise ValueError("Invalid API key format")

            encoded_token = api_key[12:]  # "emoseum_api_" 제거
            token = base64.urlsafe_b64decode(encoded_token.encode()).decode()

            return self.verify_token(token, expected_type="api_key")

        except Exception as e:
            logger.error(f"API 키 검증 실패: {e}")
            raise

    def revoke_token(self, token_jti: str):
        """토큰 철회 (블랙리스트에 추가)"""
        # 실제 구현에서는 Redis나 데이터베이스에 JTI를 저장해야 함
        logger.info(f"토큰 철회: {token_jti}")
        # TODO: 실제 블랙리스트 구현

    def is_token_revoked(self, token_jti: str) -> bool:
        """토큰이 철회되었는지 확인"""
        # 실제 구현에서는 블랙리스트를 확인해야 함
        return False  # TODO: 실제 블랙리스트 확인


# 편의 함수들
def hash_password(password: str) -> str:
    """비밀번호 해싱 (bcrypt 사용)"""
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(password: str, hashed_password: str) -> bool:
    """비밀번호 검증"""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception as e:
        logger.error(f"비밀번호 검증 실패: {e}")
        return False


def generate_reset_token(user_id: str, email: str) -> str:
    """비밀번호 재설정 토큰 생성 (편의 함수)"""
    token_manager = TokenManager()
    return token_manager.create_reset_token(user_id, email)


def verify_reset_token(token: str) -> Dict[str, str]:
    """비밀번호 재설정 토큰 검증 (편의 함수)"""
    token_manager = TokenManager()
    return token_manager.decode_reset_token(token)


def create_api_key(
    user_id: str, name: str, permissions: List[str] = None
) -> Dict[str, str]:
    """API 키 생성 (편의 함수)"""
    if permissions is None:
        permissions = ["user:basic"]

    token_manager = TokenManager()
    return token_manager.create_api_key(user_id, name, permissions)


def generate_secure_random_string(length: int = 32) -> str:
    """안전한 랜덤 문자열 생성"""
    return secrets.token_urlsafe(length)


def generate_user_session_id() -> str:
    """사용자 세션 ID 생성"""
    timestamp = str(int(datetime.utcnow().timestamp()))
    random_part = secrets.token_urlsafe(16)
    return f"session_{timestamp}_{random_part}"


def hash_api_call(user_id: str, endpoint: str, timestamp: str) -> str:
    """API 호출 해시 생성 (중복 요청 방지용)"""
    data = f"{user_id}:{endpoint}:{timestamp}"
    return hashlib.sha256(data.encode()).hexdigest()


class PermissionManager:
    """권한 관리 클래스"""

    # 권한 계층 구조
    PERMISSIONS = {
        "user:basic": "Basic user permissions",
        "user:premium": "Premium user permissions",
        "user:therapy": "Therapy session permissions",
        "user:gallery": "Gallery management permissions",
        "admin:read": "Admin read permissions",
        "admin:write": "Admin write permissions",
        "admin:system": "System administration permissions",
        "api:read": "API read access",
        "api:write": "API write access",
    }

    # 권한 상속 관계
    PERMISSION_HIERARCHY = {
        "user:premium": ["user:basic"],
        "user:therapy": ["user:basic", "user:gallery"],
        "admin:write": ["admin:read", "user:premium", "user:therapy"],
        "admin:system": ["admin:write", "admin:read"],
        "api:write": ["api:read"],
    }

    @classmethod
    def has_permission(
        cls, user_permissions: List[str], required_permission: str
    ) -> bool:
        """사용자가 특정 권한을 가지고 있는지 확인"""

        # 직접적인 권한 확인
        if required_permission in user_permissions:
            return True

        # 상속 권한 확인
        for user_perm in user_permissions:
            inherited_perms = cls.PERMISSION_HIERARCHY.get(user_perm, [])
            if required_permission in inherited_perms:
                return True

        return False

    @classmethod
    def get_effective_permissions(cls, user_permissions: List[str]) -> List[str]:
        """사용자의 유효한 모든 권한 반환 (상속 포함)"""

        effective_perms = set(user_permissions)

        for perm in user_permissions:
            inherited_perms = cls.PERMISSION_HIERARCHY.get(perm, [])
            effective_perms.update(inherited_perms)

        return list(effective_perms)

    @classmethod
    def validate_permissions(cls, permissions: List[str]) -> bool:
        """권한 목록이 유효한지 검증"""

        for perm in permissions:
            if perm not in cls.PERMISSIONS:
                return False

        return True


# 전역 토큰 매니저 인스턴스
token_manager = TokenManager()


# 데코레이터 함수들
def require_permission(required_permission: str):
    """권한 확인 데코레이터"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            # 실제 구현에서는 현재 사용자의 권한을 확인해야 함
            # 여기서는 간단한 구조만 제공
            logger.debug(f"권한 확인: {required_permission}")
            return func(*args, **kwargs)

        return wrapper

    return decorator


def rate_limit(calls_per_minute: int = 60):
    """Rate limiting 데코레이터"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            # 실제 구현에서는 Redis나 메모리 캐시를 사용해야 함
            logger.debug(f"Rate limit 확인: {calls_per_minute}/min")
            return func(*args, **kwargs)

        return wrapper

    return decorator
