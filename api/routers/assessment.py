# api/routers/assessment.py

from typing import Annotated, List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
import logging

from ..models.assessment import (
    PsychometricTestRequest,
    PsychometricResult,
    PsychometricResponse,
    VisualPreferencesRequest,
    VisualPreferences,
    VisualPreferencesResponse,
    AssessmentSummary,
    PersonalizationMetrics,
)
from ..models.common import APIResponse, CopingStyle, SeverityLevel
from .users import get_current_user
from ..services.database import db, SupabaseService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/assessment", tags=["assessment"])


def calculate_psychometric_scores(request: PsychometricTestRequest) -> dict:
    """심리검사 점수 계산 및 해석"""

    # PHQ-9 점수 계산 (0-27)
    phq9_score = sum(request.phq9_answers)

    # CES-D 점수 계산 (0-60)
    cesd_score = sum(request.cesd_answers)

    # MEAQ 점수 계산 (25-125)
    meaq_score = sum(request.meaq_answers)

    # CISS 점수 계산 (48-240)
    ciss_score = sum(request.ciss_answers)

    # CISS 하위 척도 계산 (각 16문항씩)
    ciss_task_focused = sum(request.ciss_answers[0:16]) / 16  # 과제 중심
    ciss_emotion_focused = sum(request.ciss_answers[16:32]) / 16  # 감정 중심
    ciss_avoidance_focused = sum(request.ciss_answers[32:48]) / 16  # 회피 중심

    # MEAQ 하위 요인 계산 (각 4-5문항씩, 단순화된 버전)
    meaq_behavioral_avoidance = sum(request.meaq_answers[0:4]) / 4
    meaq_distress_aversion = sum(request.meaq_answers[4:8]) / 4
    meaq_procrastination = sum(request.meaq_answers[8:12]) / 4
    meaq_distraction = sum(request.meaq_answers[12:16]) / 4
    meaq_repression = sum(request.meaq_answers[16:20]) / 4
    meaq_denial = sum(request.meaq_answers[20:25]) / 5

    # 대처 스타일 결정 (CISS 기반)
    coping_style = CopingStyle.BALANCED
    if ciss_task_focused > max(ciss_emotion_focused, ciss_avoidance_focused):
        coping_style = CopingStyle.CONFRONTATIONAL
    elif ciss_avoidance_focused > max(ciss_task_focused, ciss_emotion_focused):
        coping_style = CopingStyle.AVOIDANT

    # 심각도 수준 결정 (PHQ-9 기반)
    if phq9_score <= 4:
        severity_level = SeverityLevel.MINIMAL
    elif phq9_score <= 9:
        severity_level = SeverityLevel.MILD
    elif phq9_score <= 14:
        severity_level = SeverityLevel.MODERATE
    else:
        severity_level = SeverityLevel.SEVERE

    # 해석 생성
    interpretation = {
        "phq9_interpretation": get_phq9_interpretation(phq9_score),
        "cesd_interpretation": get_cesd_interpretation(cesd_score),
        "meaq_interpretation": get_meaq_interpretation(meaq_score),
        "ciss_interpretation": get_ciss_interpretation(
            ciss_task_focused, ciss_emotion_focused, ciss_avoidance_focused
        ),
        "overall_assessment": f"Based on your responses, you show a {coping_style.value} coping style with {severity_level.value} level symptoms.",
    }

    # 권장사항 생성
    recommendations = generate_recommendations(coping_style, severity_level, phq9_score)

    return {
        "phq9_score": phq9_score,
        "cesd_score": cesd_score,
        "meaq_score": meaq_score,
        "ciss_score": ciss_score,
        "coping_style": coping_style,
        "severity_level": severity_level,
        "interpretation": interpretation,
        "recommendations": recommendations,
        "ciss_task_focused": ciss_task_focused,
        "ciss_emotion_focused": ciss_emotion_focused,
        "ciss_avoidance_focused": ciss_avoidance_focused,
        "meaq_behavioral_avoidance": meaq_behavioral_avoidance,
        "meaq_distress_aversion": meaq_distress_aversion,
        "meaq_procrastination": meaq_procrastination,
        "meaq_distraction": meaq_distraction,
        "meaq_repression": meaq_repression,
        "meaq_denial": meaq_denial,
    }


def get_phq9_interpretation(score: int) -> str:
    """PHQ-9 점수 해석"""
    if score <= 4:
        return "Minimal depression symptoms"
    elif score <= 9:
        return "Mild depression symptoms"
    elif score <= 14:
        return "Moderate depression symptoms"
    else:
        return "Severe depression symptoms"


def get_cesd_interpretation(score: int) -> str:
    """CES-D 점수 해석"""
    if score < 16:
        return "Below depression threshold"
    elif score < 24:
        return "Mild to moderate depression symptoms"
    else:
        return "Significant depression symptoms"


def get_meaq_interpretation(score: int) -> str:
    """MEAQ 점수 해석"""
    if score < 60:
        return "Low emotional avoidance"
    elif score < 90:
        return "Moderate emotional avoidance"
    else:
        return "High emotional avoidance"


def get_ciss_interpretation(task: float, emotion: float, avoidance: float) -> str:
    """CISS 점수 해석"""
    dominant = max(task, emotion, avoidance)
    if dominant == task:
        return "Primarily uses task-focused coping strategies"
    elif dominant == emotion:
        return "Primarily uses emotion-focused coping strategies"
    else:
        return "Primarily uses avoidance-focused coping strategies"


def generate_recommendations(
    coping_style: CopingStyle, severity: SeverityLevel, phq9_score: int
) -> List[str]:
    """개인화된 권장사항 생성"""
    recommendations = []

    # 심각도 기반 권장사항
    if severity == SeverityLevel.SEVERE or phq9_score >= 15:
        recommendations.append("Consider consulting with a mental health professional")
        recommendations.append(
            "Monitor symptoms closely and seek immediate help if suicidal thoughts occur"
        )

    # 대처 스타일 기반 권장사항
    if coping_style == CopingStyle.AVOIDANT:
        recommendations.extend(
            [
                "Practice mindfulness and acceptance-based techniques",
                "Gradually expose yourself to avoided situations in small steps",
                "Focus on building emotional tolerance through ACT exercises",
            ]
        )
    elif coping_style == CopingStyle.CONFRONTATIONAL:
        recommendations.extend(
            [
                "Balance problem-solving with emotional processing",
                "Practice self-compassion and emotional validation",
                "Consider when acceptance might be more helpful than action",
            ]
        )
    else:  # BALANCED
        recommendations.extend(
            [
                "Continue developing your flexible coping skills",
                "Practice recognizing when to use different coping strategies",
                "Build on your existing balanced approach to challenges",
            ]
        )

    # 일반적 권장사항
    recommendations.extend(
        [
            "Engage in regular Emoseum therapy journeys",
            "Maintain consistent sleep and exercise habits",
            "Connect with supportive friends and family",
        ]
    )

    return recommendations


@router.post(
    "",
    response_model=APIResponse[PsychometricResponse],
    summary="Submit psychometric assessment",
    description="Submit PHQ-9, CES-D, MEAQ, and CISS assessments for analysis",
)
async def submit_assessment(
    request: PsychometricTestRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
    db_service: Annotated[SupabaseService, Depends(lambda: db)],
) -> APIResponse[PsychometricResponse]:
    """심리검사 제출 및 분석"""

    try:
        user_id = current_user["user_id"]

        # 점수 계산 및 해석
        assessment_data = calculate_psychometric_scores(request)

        # 결과 저장
        result_data = {
            "user_id": user_id,
            **assessment_data,
            "created_date": datetime.utcnow().isoformat(),
        }

        saved_result = await db_service.save_psychometric_result(user_id, result_data)

        # PsychometricResult 모델 생성
        result = PsychometricResult(
            user_id=user_id, created_date=datetime.utcnow(), **assessment_data
        )

        # 위험도 평가
        risk_level = "low"
        if assessment_data["severity_level"] == SeverityLevel.SEVERE:
            risk_level = "high"
        elif assessment_data["severity_level"] == SeverityLevel.MODERATE:
            risk_level = "medium"

        # 전문가 상담 필요 여부
        referral_needed = (
            assessment_data["phq9_score"] >= 15
            or assessment_data["severity_level"] == SeverityLevel.SEVERE
        )

        # 다음 단계 권장사항
        next_steps = ["Complete your visual preferences setup"]
        if not referral_needed:
            next_steps.append("Begin your first therapy journey")
        else:
            next_steps.insert(
                0, "Schedule consultation with mental health professional"
            )

        response = PsychometricResponse(
            result=result,
            risk_level=risk_level,
            next_steps=next_steps,
            referral_needed=referral_needed,
        )

        logger.info(
            f"심리검사 완료: {user_id}, 대처스타일: {assessment_data['coping_style']}, 심각도: {assessment_data['severity_level']}"
        )

        return APIResponse(
            success=True, message="Assessment completed successfully", data=response
        )

    except Exception as e:
        logger.error(f"심리검사 처리 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process assessment",
        )


@router.get(
    "/results",
    response_model=APIResponse[List[PsychometricResult]],
    summary="Get assessment results",
    description="Get user's psychometric assessment history",
)
async def get_assessment_results(
    current_user: Annotated[dict, Depends(get_current_user)],
    db_service: Annotated[SupabaseService, Depends(lambda: db)],
    limit: int = 10,
) -> APIResponse[List[PsychometricResult]]:
    """심리검사 결과 조회"""

    try:
        user_id = current_user["user_id"]

        # 결과 조회
        results_data = await db_service.get_psychometric_results(user_id, limit=limit)

        # PsychometricResult 모델로 변환
        results = []
        for data in results_data:
            result = PsychometricResult(
                user_id=data["user_id"],
                phq9_score=data["phq9_score"],
                cesd_score=data["cesd_score"],
                meaq_score=data["meaq_score"],
                ciss_score=data["ciss_score"],
                coping_style=CopingStyle(data["coping_style"]),
                severity_level=SeverityLevel(data["severity_level"]),
                interpretation=data["interpretation"],
                recommendations=data["recommendations"],
                ciss_task_focused=data["ciss_task_focused"],
                ciss_emotion_focused=data["ciss_emotion_focused"],
                ciss_avoidance_focused=data["ciss_avoidance_focused"],
                meaq_behavioral_avoidance=data["meaq_behavioral_avoidance"],
                meaq_distress_aversion=data["meaq_distress_aversion"],
                meaq_procrastination=data["meaq_procrastination"],
                meaq_distraction=data["meaq_distraction"],
                meaq_repression=data["meaq_repression"],
                meaq_denial=data["meaq_denial"],
                created_date=datetime.fromisoformat(
                    data["created_date"].replace("Z", "+00:00")
                ),
            )
            results.append(result)

        logger.info(f"심리검사 결과 조회: {user_id}, {len(results)}개")

        return APIResponse(
            success=True,
            message="Assessment results retrieved successfully",
            data=results,
        )

    except Exception as e:
        logger.error(f"심리검사 결과 조회 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve assessment results",
        )


@router.post(
    "/preferences/visual",
    response_model=APIResponse[VisualPreferencesResponse],
    summary="Set visual preferences",
    description="Configure user's visual preferences for image generation",
)
async def set_visual_preferences(
    request: VisualPreferencesRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
    db_service: Annotated[SupabaseService, Depends(lambda: db)],
) -> APIResponse[VisualPreferencesResponse]:
    """시각 선호도 설정"""

    try:
        user_id = current_user["user_id"]

        # 스타일 선호도 가중치 계산
        style_preferences = {}
        total_styles = len(request.preferred_art_styles)
        for style in request.preferred_art_styles:
            style_preferences[style] = 1.0 / total_styles

        # 시각 선호도 데이터 생성
        preferences_data = {
            "user_id": user_id,
            "style_preferences": style_preferences,
            "color_preferences": request.color_preferences,
            "complexity_level": request.complexity_level,
            "brightness_preference": request.brightness_preference,
            "saturation_preference": request.saturation_preference,
            "preferred_themes": request.preferred_themes,
            "avoided_elements": request.avoided_elements,
            "learning_confidence": 0.0,  # 초기값
            "updated_date": datetime.utcnow().isoformat(),
        }

        # 데이터베이스에 저장
        saved_preferences = await db_service.save_visual_preferences(
            user_id, preferences_data
        )

        # VisualPreferences 모델 생성
        preferences = VisualPreferences(
            user_id=user_id,
            style_preferences=style_preferences,
            color_preferences=request.color_preferences,
            complexity_level=request.complexity_level,
            brightness_preference=request.brightness_preference,
            saturation_preference=request.saturation_preference,
            preferred_themes=request.preferred_themes,
            avoided_elements=request.avoided_elements,
            learned_preferences=None,
            learning_confidence=0.0,
            created_date=datetime.utcnow(),
            updated_date=datetime.utcnow(),
        )

        # 개인화 레벨 계산 (Level 1 완료)
        personalization_level = 1

        # 스타일 권장사항 생성
        recommendations = [
            f"Your preferred {', '.join(request.preferred_art_styles)} styles will be emphasized",
            f"Images will be generated with {request.complexity_level}/5 complexity level",
            "You can update these preferences anytime as you explore different styles",
        ]

        response = VisualPreferencesResponse(
            preferences=preferences,
            personalization_level=personalization_level,
            recommendations=recommendations,
        )

        logger.info(f"시각 선호도 설정 완료: {user_id}")

        return APIResponse(
            success=True, message="Visual preferences saved successfully", data=response
        )

    except Exception as e:
        logger.error(f"시각 선호도 설정 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save visual preferences",
        )


@router.get(
    "/preferences/visual",
    response_model=APIResponse[VisualPreferences],
    summary="Get visual preferences",
    description="Get user's current visual preferences",
)
async def get_visual_preferences(
    current_user: Annotated[dict, Depends(get_current_user)],
    db_service: Annotated[SupabaseService, Depends(lambda: db)],
) -> APIResponse[VisualPreferences]:
    """시각 선호도 조회"""

    try:
        user_id = current_user["user_id"]

        # 시각 선호도 조회
        preferences_data = await db_service.get_visual_preferences(user_id)

        if not preferences_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Visual preferences not found. Please set up your preferences first.",
            )

        # VisualPreferences 모델 생성
        preferences = VisualPreferences(
            user_id=preferences_data["user_id"],
            style_preferences=preferences_data["style_preferences"],
            color_preferences=preferences_data["color_preferences"],
            complexity_level=preferences_data["complexity_level"],
            brightness_preference=preferences_data["brightness_preference"],
            saturation_preference=preferences_data["saturation_preference"],
            preferred_themes=preferences_data.get("preferred_themes", []),
            avoided_elements=preferences_data.get("avoided_elements", []),
            learned_preferences=preferences_data.get("learned_preferences"),
            learning_confidence=preferences_data.get("learning_confidence", 0.0),
            created_date=datetime.fromisoformat(
                preferences_data["created_date"].replace("Z", "+00:00")
            ),
            updated_date=datetime.fromisoformat(
                preferences_data["updated_date"].replace("Z", "+00:00")
            ),
        )

        logger.info(f"시각 선호도 조회 완료: {user_id}")

        return APIResponse(
            success=True,
            message="Visual preferences retrieved successfully",
            data=preferences,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"시각 선호도 조회 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve visual preferences",
        )


@router.put(
    "/preferences/visual",
    response_model=APIResponse[VisualPreferencesResponse],
    summary="Update visual preferences",
    description="Update user's visual preferences",
)
async def update_visual_preferences(
    request: VisualPreferencesRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
    db_service: Annotated[SupabaseService, Depends(lambda: db)],
) -> APIResponse[VisualPreferencesResponse]:
    """시각 선호도 수정"""

    try:
        user_id = current_user["user_id"]

        # 기존 선호도 조회
        existing_preferences = await db_service.get_visual_preferences(user_id)
        if not existing_preferences:
            # 없으면 새로 생성
            return await set_visual_preferences(request, current_user, db_service)

        # 업데이트할 데이터 준비 (기존 set_visual_preferences와 동일한 로직)
        style_preferences = {}
        total_styles = len(request.preferred_art_styles)
        for style in request.preferred_art_styles:
            style_preferences[style] = 1.0 / total_styles

        update_data = {
            "style_preferences": style_preferences,
            "color_preferences": request.color_preferences,
            "complexity_level": request.complexity_level,
            "brightness_preference": request.brightness_preference,
            "saturation_preference": request.saturation_preference,
            "preferred_themes": request.preferred_themes,
            "avoided_elements": request.avoided_elements,
            "updated_date": datetime.utcnow().isoformat(),
        }

        # 데이터베이스 업데이트
        updated_preferences = await db_service.save_visual_preferences(
            user_id, update_data
        )

        # VisualPreferences 모델 생성 (업데이트된 데이터로)
        preferences = VisualPreferences(
            user_id=user_id,
            style_preferences=style_preferences,
            color_preferences=request.color_preferences,
            complexity_level=request.complexity_level,
            brightness_preference=request.brightness_preference,
            saturation_preference=request.saturation_preference,
            preferred_themes=request.preferred_themes,
            avoided_elements=request.avoided_elements,
            learned_preferences=existing_preferences.get("learned_preferences"),
            learning_confidence=existing_preferences.get("learning_confidence", 0.0),
            created_date=datetime.fromisoformat(
                existing_preferences["created_date"].replace("Z", "+00:00")
            ),
            updated_date=datetime.utcnow(),
        )

        # 개인화 레벨 유지
        personalization_level = 1
        if preferences.learned_preferences:
            personalization_level = 2

        recommendations = [
            "Your visual preferences have been updated",
            "New preferences will be applied to future image generations",
            "You may notice different styles in your next therapy journey",
        ]

        response = VisualPreferencesResponse(
            preferences=preferences,
            personalization_level=personalization_level,
            recommendations=recommendations,
        )

        logger.info(f"시각 선호도 업데이트 완료: {user_id}")

        return APIResponse(
            success=True,
            message="Visual preferences updated successfully",
            data=response,
        )

    except Exception as e:
        logger.error(f"시각 선호도 업데이트 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update visual preferences",
        )
