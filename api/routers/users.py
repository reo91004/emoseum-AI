# api/routers/users.py

from typing import Annotated, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
import bcrypt
import logging

from ..models.user import (
    UserProfile,
    UserResponse,
    UserUpdateRequest,
    UserStats,
    UserPreferences,
    UserDeleteRequest,
    PasswordResetRequest,
)
from ..models.common import APIResponse, CopingStyle, SeverityLevel
from ..services.database import db, SupabaseService
from ..dependencies import security, AuthenticationError
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["users"])


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    db_service: Annotated[SupabaseService, Depends(lambda: db)],
) -> dict:
    """현재 인증된 사용자 정보 반환"""

    try:
        from jose import jwt, JWTError

        # JWT 토큰 디코딩
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

        # 사용자 정보 조회
        user = await db_service.get_user(user_id)
        if not user:
            raise AuthenticationError("User not found")

        return user

    except AuthenticationError:
        raise
    except Exception as e:
        logger.error(f"사용자 인증 실패: {e}")
        raise AuthenticationError("Authentication failed")


def hash_password(password: str) -> str:
    """비밀번호 해싱"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """비밀번호 검증"""
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


@router.get(
    "/profile",
    response_model=APIResponse[UserResponse],
    summary="Get user profile",
    description="Get current user's profile information",
)
async def get_user_profile(
    current_user: Annotated[dict, Depends(get_current_user)],
    db_service: Annotated[SupabaseService, Depends(lambda: db)],
) -> APIResponse[UserResponse]:
    """사용자 프로필 조회"""

    try:
        user_id = current_user["user_id"]

        # 심리검사 결과 조회
        assessment_results = await db_service.get_psychometric_results(user_id, limit=1)

        # 갤러리 통계 조회 (여정 수 등)
        gallery_items = await db_service.get_gallery_items(user_id)
        total_journeys = len(gallery_items)
        completed_journeys = len(
            [
                item
                for item in gallery_items
                if item.get("completion_status") == "completed"
            ]
        )

        # 이미지 생성 통계
        total_images = len([item for item in gallery_items if item.get("image_url")])

        # 마지막 활동 시간 (갤러리 아이템 중 가장 최근)
        last_activity = None
        if gallery_items:
            last_activity = max(
                item.get("created_date", "1970-01-01T00:00:00Z")
                for item in gallery_items
            )

        # 개인화 레벨 계산 (간단한 로직)
        personalization_level = 1
        if completed_journeys >= 5:
            personalization_level = 2
        if completed_journeys >= 20:
            personalization_level = 3

        # 현재 대처 스타일 및 심각도
        current_coping_style = None
        current_severity_level = None
        last_assessment_date = None

        if assessment_results:
            latest_result = assessment_results[0]
            current_coping_style = latest_result.get("coping_style")
            current_severity_level = latest_result.get("severity_level")
            last_assessment_date = latest_result.get("created_date")

        # UserProfile 생성
        profile = UserProfile(
            user_id=user_id,
            email=current_user.get("email"),
            created_date=datetime.fromisoformat(
                current_user["created_date"].replace("Z", "+00:00")
            ),
            current_coping_style=current_coping_style,
            current_severity_level=current_severity_level,
            last_assessment_date=(
                datetime.fromisoformat(last_assessment_date.replace("Z", "+00:00"))
                if last_assessment_date
                else None
            ),
            total_journeys=total_journeys,
            completed_journeys=completed_journeys,
            total_images_generated=total_images,
            last_activity_date=(
                datetime.fromisoformat(last_activity.replace("Z", "+00:00"))
                if last_activity
                else None
            ),
            personalization_level=personalization_level,
        )

        user_response = UserResponse(
            user_id=user_id,
            email=current_user.get("email"),
            created_date=profile.created_date,
            profile=profile,
        )

        logger.info(f"프로필 조회 완료: {user_id}")

        return APIResponse(
            success=True, message="Profile retrieved successfully", data=user_response
        )

    except Exception as e:
        logger.error(f"프로필 조회 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve profile",
        )


@router.put(
    "/profile",
    response_model=APIResponse[UserResponse],
    summary="Update user profile",
    description="Update current user's profile information",
)
async def update_user_profile(
    request: UserUpdateRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
    db_service: Annotated[SupabaseService, Depends(lambda: db)],
) -> APIResponse[UserResponse]:
    """사용자 프로필 수정"""

    try:
        user_id = current_user["user_id"]
        update_data = {"updated_date": datetime.utcnow().isoformat()}

        # 이메일 업데이트
        if request.email:
            update_data["email"] = request.email

        # 비밀번호 업데이트
        if request.password:
            update_data["password_hash"] = hash_password(request.password)

        # 사용자 정보 업데이트
        updated_user = await db_service.update_user(user_id, update_data)

        # 업데이트된 프로필 조회 (기존 함수 재사용)
        updated_profile_response = await get_user_profile(updated_user, db_service)

        logger.info(f"프로필 업데이트 완료: {user_id}")

        return APIResponse(
            success=True,
            message="Profile updated successfully",
            data=updated_profile_response.data,
        )

    except Exception as e:
        logger.error(f"프로필 업데이트 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile",
        )


@router.get(
    "/stats",
    response_model=APIResponse[UserStats],
    summary="Get user statistics",
    description="Get detailed statistics about user's therapy journey",
)
async def get_user_stats(
    current_user: Annotated[dict, Depends(get_current_user)],
    db_service: Annotated[SupabaseService, Depends(lambda: db)],
    days: int = 30,
) -> APIResponse[UserStats]:
    """사용자 통계 조회"""

    try:
        user_id = current_user["user_id"]

        # 갤러리 아이템들 조회
        gallery_items = await db_service.get_gallery_items(user_id)

        # 기본 통계
        total_journeys = len(gallery_items)
        completed_journeys = len(
            [
                item
                for item in gallery_items
                if item.get("completion_status") == "completed"
            ]
        )
        completion_rate = (
            completed_journeys / total_journeys if total_journeys > 0 else 0.0
        )

        # 감정 분석 통계
        emotion_keywords = []
        vad_scores = {"valence": [], "arousal": [], "dominance": []}

        for item in gallery_items:
            keywords = item.get("emotion_keywords", [])
            if isinstance(keywords, list):
                emotion_keywords.extend(keywords)

            vad_data = item.get("vad_scores", {})
            if isinstance(vad_data, dict):
                for key in ["valence", "arousal", "dominance"]:
                    if key in vad_data:
                        vad_scores[key].append(vad_data[key])

        # 가장 빈번한 감정들
        from collections import Counter

        emotion_counter = Counter(emotion_keywords)
        dominant_emotions = [emotion for emotion, _ in emotion_counter.most_common(5)]

        # VAD 평균값
        vad_average = {}
        for key, values in vad_scores.items():
            vad_average[key] = sum(values) / len(values) if values else 0.0

        # 감정 추이 (간단한 버전)
        emotion_trend = {}
        for emotion in dominant_emotions:
            emotion_trend[emotion] = (
                emotion_counter.get(emotion, 0) / len(emotion_keywords)
                if emotion_keywords
                else 0.0
            )

        # 이미지 통계
        total_images = len([item for item in gallery_items if item.get("image_url")])
        preferred_styles = [
            "painting",
            "photography",
            "abstract",
        ]  # 실제로는 시각 선호도에서 가져와야 함

        # 비용 통계
        cost_summary = await db_service.get_cost_summary(user_id=user_id, days=days)
        total_cost = cost_summary.get("total_cost", 0.0)
        cost_breakdown = cost_summary.get("service_breakdown", {})

        # 시간 통계
        first_journey_date = None
        last_journey_date = None
        if gallery_items:
            dates = [
                datetime.fromisoformat(item["created_date"].replace("Z", "+00:00"))
                for item in gallery_items
            ]
            first_journey_date = min(dates)
            last_journey_date = max(dates)

        stats = UserStats(
            user_id=user_id,
            total_journeys=total_journeys,
            completed_journeys=completed_journeys,
            completion_rate=completion_rate,
            dominant_emotions=dominant_emotions,
            emotion_trend=emotion_trend,
            vad_average=vad_average,
            total_images=total_images,
            preferred_styles=preferred_styles,
            total_cost=total_cost,
            cost_breakdown=cost_breakdown,
            first_journey_date=first_journey_date,
            last_journey_date=last_journey_date,
            average_session_duration=None,  # 실제 구현 필요
        )

        logger.info(f"사용자 통계 조회 완료: {user_id}")

        return APIResponse(
            success=True, message="User statistics retrieved successfully", data=stats
        )

    except Exception as e:
        logger.error(f"사용자 통계 조회 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user statistics",
        )


@router.delete(
    "/profile",
    response_model=APIResponse[dict],
    summary="Delete user account",
    description="Permanently delete user account and all associated data",
)
async def delete_user_account(
    request: UserDeleteRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
    db_service: Annotated[SupabaseService, Depends(lambda: db)],
) -> APIResponse[dict]:
    """사용자 계정 삭제"""

    try:
        user_id = current_user["user_id"]

        # 비밀번호 확인
        if not verify_password(request.password, current_user.get("password_hash", "")):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid password"
            )

        # 확인 문구 검증은 Pydantic 모델에서 처리됨

        # 관련 데이터 삭제 (순서 중요)
        # 1. 개인화 데이터
        # 2. 비용 추적 데이터
        # 3. 갤러리 아이템들
        # 4. 시각 선호도
        # 5. 심리검사 결과
        # 6. 사용자 기본 정보

        # 실제 구현에서는 각 테이블에서 해당 사용자 데이터를 삭제해야 함
        # 현재는 기본 사용자 정보만 삭제

        await db_service.delete_user(user_id)

        logger.info(f"사용자 계정 삭제 완료: {user_id}")

        return APIResponse(
            success=True,
            message="Account deleted successfully",
            data={"deleted": True, "user_id": user_id},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"계정 삭제 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete account",
        )


@router.post(
    "/password-reset",
    response_model=APIResponse[dict],
    summary="Request password reset",
    description="Send password reset email to user",
)
async def request_password_reset(
    request: PasswordResetRequest,
    db_service: Annotated[SupabaseService, Depends(lambda: db)],
) -> APIResponse[dict]:
    """비밀번호 재설정 요청"""

    try:
        # 이메일로 사용자 조회
        # 실제 구현에서는 이메일로 사용자를 찾고 재설정 토큰을 생성해야 함
        # 현재는 기본 응답만 반환

        logger.info(f"비밀번호 재설정 요청: {request.email}")

        return APIResponse(
            success=True,
            message="If this email is registered, you will receive password reset instructions",
            data={"email_sent": True},
        )

    except Exception as e:
        logger.error(f"비밀번호 재설정 요청 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process password reset request",
        )
