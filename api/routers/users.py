# api/routers/users.py

# ==============================================================================
# 이 파일은 사용자 관련 API 엔드포인트를 정의한다.
# 프로필 조회, 심리측정 평가, 시각적 선호도 설정 등의 기능을 제공한다.
# ==============================================================================

import logging
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from ..database.connection import get_database
from ..database.collections import Collections
from ..dependencies import get_current_user, get_act_therapy_system
from ..models.user import (
    UserProfileResponse, 
    PsychometricAssessmentRequest,
    PsychometricResultResponse,
    UpdateVisualPreferencesRequest,
    UpdateUserSettingsRequest,
    UserStatusResponse,
    PsychometricResults,
    VisualPreferences,
    UserSettings
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("/profile", response_model=UserProfileResponse)
async def get_user_profile(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """현재 사용자 프로필 조회"""
    try:
        # current_user에서 데이터 사용 (ACT 시스템에서 채워짐)
        # 응답 모델로 변환
        profile = UserProfileResponse(
            user_id=current_user["user_id"],
            created_date=current_user["created_date"],
            psychometric_results=PsychometricResults(
                phq9_score=current_user["psychometric_results"][-1]["phq9_score"],
                cesd_score=current_user["psychometric_results"][-1]["cesd_score"],
                meaq_score=current_user["psychometric_results"][-1]["meaq_score"],
                ciss_score=current_user["psychometric_results"][-1]["ciss_score"],
                coping_style=current_user["psychometric_results"][-1]["coping_style"],
                severity_level=current_user["psychometric_results"][-1]["severity_level"],
                assessment_date=current_user["psychometric_results"][-1]["test_date"]
            ) if current_user.get("psychometric_results") else None,
            visual_preferences=VisualPreferences(**current_user.get("visual_preferences", {})),
            personalization_level=1,
            settings=UserSettings(language="ko", notifications=True)
        )
        
        return profile
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user profile"
        )


@router.put("/profile", response_model=UserProfileResponse)
async def update_user_profile(
    settings: UpdateUserSettingsRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """사용자 프로필 설정 업데이트"""
    try:
        # 현재는 설정이 별도로 처리되므로 현재 프로필만 반환
        # 완전한 구현에서는 ACT 시스템을 통해 사용자 설정을 업데이트해야 함
        logger.info(f"Profile update requested for user {current_user['user_id']}: {settings}")
        
        # 현재 프로필 반환
        return await get_user_profile(current_user)
        
    except Exception as e:
        logger.error(f"Error updating user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="사용자 프로필 업데이트 실패"
        )


@router.post("/assessment", response_model=PsychometricResultResponse)
async def conduct_assessment(
    assessment: PsychometricAssessmentRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
    act_system = Depends(get_act_therapy_system)
):
    """심리측정 평가 수행"""
    try:
        # ACT 시스템을 사용하여 평가 처리
        result = act_system.conduct_psychometric_assessment(
            user_id=current_user["user_id"],
            phq9_score=assessment.phq9_score,
            cesd_score=assessment.cesd_score,
            meaq_score=assessment.meaq_score,
            ciss_score=assessment.ciss_score
        )
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="평가 처리 실패"
            )
        
        # 심각도에 따른 권장사항
        recommendations = {
            "mild": "정기적인 ACT 치료 세션과 자기 관리를 계속하세요.",
            "moderate": "정기적인 ACT 치료 세션을 권장합니다. 증상이 지속되면 상담을 고려하세요.",
            "severe": "ACT 치료와 함께 상담사의 상담을 강력히 권장합니다."
        }
        
        return PsychometricResultResponse(
            coping_style=result["coping_style"],
            severity_level=result["severity_level"],
            recommendation=recommendations.get(result["severity_level"], ""),
            assessment_date=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Assessment error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="평가 실패"
        )


@router.put("/visual-preferences", response_model=UserProfileResponse)
async def update_visual_preferences(
    preferences: UpdateVisualPreferencesRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    act_system = Depends(get_act_therapy_system)
):
    """사용자 시각적 선호도 업데이트"""
    try:
        # API 모델을 ACT 시스템 형식으로 변환
        visual_prefs = {}
        if preferences.preferred_styles is not None:
            # API 스타일을 ACT 시스템 스타일로 매핑
            if "painting" in preferences.preferred_styles:
                visual_prefs["art_style"] = "painting"
            elif "photography" in preferences.preferred_styles:
                visual_prefs["art_style"] = "photography"
            elif "abstract" in preferences.preferred_styles:
                visual_prefs["art_style"] = "abstract"
            else:
                visual_prefs["art_style"] = "painting"
        
        if preferences.color_preferences is not None:
            # 색상 선호도 매핑
            if "warm" in preferences.color_preferences:
                visual_prefs["color_tone"] = "warm"
            elif "cool" in preferences.color_preferences:
                visual_prefs["color_tone"] = "cool"
            elif "pastel" in preferences.color_preferences:
                visual_prefs["color_tone"] = "pastel"
            else:
                visual_prefs["color_tone"] = "warm"
        
        if preferences.complexity_level is not None:
            complexity_map = {"low": "simple", "medium": "balanced", "high": "complex"}
            visual_prefs["complexity"] = complexity_map.get(preferences.complexity_level, "balanced")
        
        # ACT 시스템을 통해 업데이트
        if visual_prefs:
            act_system.user_manager.set_visual_preferences(current_user["user_id"], visual_prefs)
        
        # 업데이트된 프로필 반환
        return await get_user_profile(current_user)
        
    except Exception as e:
        logger.error(f"Error updating visual preferences: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="시각적 선호도 업데이트 실패"
        )


@router.get("/status", response_model=UserStatusResponse)
async def get_user_status(
    current_user: Dict[str, Any] = Depends(get_current_user),
    act_system = Depends(get_act_therapy_system)
):
    """사용자 상태 및 활동 요약 조회"""
    try:
        # ACT 시스템을 통해 사용자 통계 조회
        user_stats = act_system.user_manager.get_user_stats(current_user["user_id"])
        
        # ACT 시스템을 통해 갤러리 데이터 조회
        gallery_data = act_system.get_user_gallery(current_user["user_id"], limit=1)
        
        # 미완료 여정 조회
        incomplete_journeys = act_system.gallery_manager.get_incomplete_journeys(current_user["user_id"])
        
        return UserStatusResponse(
            user_id=current_user["user_id"],
            is_active=True,
            last_activity=gallery_data.get("items", [{}])[0].get("created_date") if gallery_data.get("items") else None,
            completed_journeys=gallery_data.get("total_items", 0),
            current_session_id=incomplete_journeys[0].item_id if incomplete_journeys else None,
            personalization_level=1
        )
        
    except Exception as e:
        logger.error(f"Error getting user status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="사용자 상태 조회 실패"
        )