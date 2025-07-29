# api/routers/auth.py

from typing import Annotated
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
import bcrypt
import logging

from ..models.user import (
    UserRegisterRequest,
    UserLoginRequest,
    TokenResponse,
    UserResponse,
    UserProfile,
)
from ..models.common import APIResponse, ErrorResponse
from ..services.database import db, SupabaseService
from ..dependencies import (
    create_access_token,
    security,
    AuthenticationError,
    AuthorizationError,
)
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])


def hash_password(password: str) -> str:
    """비밀번호 해싱"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """비밀번호 검증"""
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


@router.post(
    "/register",
    response_model=APIResponse[TokenResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Register new user",
    description="Create a new user account with unique user_id and return JWT token",
)
async def register_user(
    request: UserRegisterRequest,
    db_service: Annotated[SupabaseService, Depends(lambda: db)],
) -> APIResponse[TokenResponse]:
    """사용자 회원가입"""

    try:
        # 사용자 ID 중복 확인
        existing_user = await db_service.get_user(request.user_id)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"User ID '{request.user_id}' already exists",
            )

        # 이메일 중복 확인 (이메일이 제공된 경우)
        if request.email:
            # 이메일 중복 체크 로직 (Supabase에서 이메일로 조회)
            # 여기서는 간단히 패스하고 실제 구현 시 추가
            pass

        # 비밀번호 해싱
        hashed_password = hash_password(request.password)

        # 사용자 생성
        user_data = {
            "user_id": request.user_id,
            "email": request.email,
            "password_hash": hashed_password,
            "created_date": datetime.utcnow().isoformat(),
            "updated_date": datetime.utcnow().isoformat(),
        }

        created_user = await db_service.create_user(user_data)

        # JWT 토큰 생성
        access_token = create_access_token(
            user_id=request.user_id,
            expires_delta=timedelta(minutes=settings.jwt_access_token_expire_minutes),
        )

        token_response = TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.jwt_access_token_expire_minutes * 60,
            user_id=request.user_id,
        )

        logger.info(f"새 사용자 등록 완료: {request.user_id}")

        return APIResponse(
            success=True, message="User registration successful", data=token_response
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"사용자 등록 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User registration failed",
        )


@router.post(
    "/login",
    response_model=APIResponse[TokenResponse],
    summary="User login",
    description="Authenticate user and return JWT token",
)
async def login_user(
    request: UserLoginRequest,
    db_service: Annotated[SupabaseService, Depends(lambda: db)],
) -> APIResponse[TokenResponse]:
    """사용자 로그인"""

    try:
        # 사용자 조회
        user = await db_service.get_user(request.user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid user ID or password",
            )

        # 비밀번호 검증
        if not verify_password(request.password, user.get("password_hash", "")):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid user ID or password",
            )

        # 마지막 로그인 시간 업데이트
        await db_service.update_user(
            request.user_id,
            {
                "last_login_date": datetime.utcnow().isoformat(),
                "updated_date": datetime.utcnow().isoformat(),
            },
        )

        # JWT 토큰 생성
        access_token = create_access_token(
            user_id=request.user_id,
            expires_delta=timedelta(minutes=settings.jwt_access_token_expire_minutes),
        )

        token_response = TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.jwt_access_token_expire_minutes * 60,
            user_id=request.user_id,
        )

        logger.info(f"사용자 로그인 성공: {request.user_id}")

        return APIResponse(
            success=True, message="Login successful", data=token_response
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"로그인 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Login failed"
        )


@router.post(
    "/refresh",
    response_model=APIResponse[TokenResponse],
    summary="Refresh access token",
    description="Generate new access token from valid existing token",
)
async def refresh_token(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    db_service: Annotated[SupabaseService, Depends(lambda: db)],
) -> APIResponse[TokenResponse]:
    """JWT 토큰 갱신"""

    try:
        from jose import jwt, JWTError

        # 현재 토큰에서 사용자 정보 추출
        try:
            payload = jwt.decode(
                credentials.credentials,
                settings.jwt_secret_key,
                algorithms=[settings.jwt_algorithm],
            )
            user_id = payload.get("user_id")
            if not user_id:
                raise JWTError("Invalid token payload")

        except JWTError:
            raise AuthenticationError("Invalid or expired token")

        # 사용자 존재 확인
        user = await db_service.get_user(user_id)
        if not user:
            raise AuthenticationError("User not found")

        # 새 토큰 생성
        new_access_token = create_access_token(
            user_id=user_id,
            expires_delta=timedelta(minutes=settings.jwt_access_token_expire_minutes),
        )

        token_response = TokenResponse(
            access_token=new_access_token,
            token_type="bearer",
            expires_in=settings.jwt_access_token_expire_minutes * 60,
            user_id=user_id,
        )

        logger.info(f"토큰 갱신 성공: {user_id}")

        return APIResponse(
            success=True, message="Token refreshed successfully", data=token_response
        )

    except (AuthenticationError, HTTPException):
        raise
    except Exception as e:
        logger.error(f"토큰 갱신 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed",
        )


@router.post(
    "/logout",
    response_model=APIResponse[dict],
    summary="User logout",
    description="Logout user (client should discard token)",
)
async def logout_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
) -> APIResponse[dict]:
    """사용자 로그아웃"""

    try:
        from jose import jwt, JWTError

        # 토큰에서 사용자 정보 추출 (검증용)
        try:
            payload = jwt.decode(
                credentials.credentials,
                settings.jwt_secret_key,
                algorithms=[settings.jwt_algorithm],
            )
            user_id = payload.get("user_id")

        except JWTError:
            raise AuthenticationError("Invalid token")

        # 실제로는 토큰 블랙리스트에 추가해야 하지만,
        # 현재는 클라이언트에서 토큰 삭제하도록 안내
        logger.info(f"사용자 로그아웃: {user_id}")

        return APIResponse(
            success=True,
            message="Logout successful. Please discard your token.",
            data={"logged_out": True},
        )

    except (AuthenticationError, HTTPException):
        raise
    except Exception as e:
        logger.error(f"로그아웃 처리 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Logout failed"
        )


@router.get(
    "/verify",
    response_model=APIResponse[dict],
    summary="Verify token",
    description="Verify if current token is valid",
)
async def verify_token(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    db_service: Annotated[SupabaseService, Depends(lambda: db)],
) -> APIResponse[dict]:
    """토큰 유효성 검증"""

    try:
        from jose import jwt, JWTError

        # 토큰 디코딩 및 검증
        try:
            payload = jwt.decode(
                credentials.credentials,
                settings.jwt_secret_key,
                algorithms=[settings.jwt_algorithm],
            )
            user_id = payload.get("user_id")
            exp = payload.get("exp")

            if not user_id:
                raise JWTError("Invalid token payload")

        except JWTError:
            raise AuthenticationError("Invalid or expired token")

        # 사용자 존재 확인
        user = await db_service.get_user(user_id)
        if not user:
            raise AuthenticationError("User not found")

        # 토큰 만료 시간 계산
        exp_datetime = datetime.fromtimestamp(exp)
        time_until_expiry = (exp_datetime - datetime.utcnow()).total_seconds()

        return APIResponse(
            success=True,
            message="Token is valid",
            data={
                "valid": True,
                "user_id": user_id,
                "expires_in_seconds": int(time_until_expiry),
                "expires_at": exp_datetime.isoformat(),
            },
        )

    except (AuthenticationError, HTTPException):
        raise
    except Exception as e:
        logger.error(f"토큰 검증 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token verification failed",
        )
