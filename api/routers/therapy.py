# api/routers/therapy.py

from typing import Annotated, List, Optional, Dict, Any
from datetime import datetime
from uuid import uuid4
from fastapi import APIRouter, Depends, HTTPException, status, Query
import logging

from ..models.therapy import (
    JourneyStartRequest,
    JourneyResponse,
    ReflectionRequest,
    ReflectionResponse,
    DefusionRequest,
    DefusionResponse,
    ClosureResponse,
    GalleryItem,
    GalleryListResponse,
    EmotionAnalysis,
    VADScores,
    ImageGenerationResult,
    CuratorMessage,
    TherapyAnalytics,
    EmergencyResponse,
)
from ..models.common import APIResponse, CompletionStatus, PaginatedResponse
from ..routers.users import get_current_user
from ..services.database import db, SupabaseService
from ..services.therapy_service import TherapyService
from ..services.image_service import ImageService
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/therapy", tags=["therapy"])

# 서비스 인스턴스들 (의존성 주입으로 나중에 개선 가능)
therapy_service = TherapyService()
image_service = ImageService()


def check_emergency_signals(
    diary_text: str, emotion_keywords: List[str]
) -> Optional[EmergencyResponse]:
    """응급 상황 신호 감지"""

    # 위험 키워드 감지
    crisis_keywords = [
        "suicide",
        "kill myself",
        "end it all",
        "die",
        "death",
        "hopeless",
        "worthless",
        "can't go on",
        "better off dead",
    ]

    text_lower = diary_text.lower()
    risk_factors = []
    alert_level = "low"

    # 직접적 위험 신호
    direct_threats = ["suicide", "kill myself", "end it all"]
    if any(keyword in text_lower for keyword in direct_threats):
        alert_level = "critical"
        risk_factors.append("Direct self-harm ideation expressed")

    # 간접적 위험 신호
    indirect_signals = ["hopeless", "worthless", "can't go on"]
    if any(keyword in text_lower for keyword in indirect_signals):
        if alert_level != "critical":
            alert_level = "high"
        risk_factors.append("Expressions of hopelessness or worthlessness")

    # 감정 키워드 분석
    high_risk_emotions = ["despair", "hopelessness", "worthless", "trapped"]
    if any(emotion in emotion_keywords for emotion in high_risk_emotions):
        if alert_level == "low":
            alert_level = "medium"
        risk_factors.append("High-risk emotional states detected")

    # 응급 상황이 감지된 경우
    if alert_level in ["high", "critical"]:
        return EmergencyResponse(
            alert_level=alert_level,
            risk_factors=risk_factors,
            immediate_actions=[
                "Please consider reaching out to a mental health professional immediately",
                "Contact a crisis helpline if you're having thoughts of self-harm",
                "Reach out to a trusted friend or family member for support",
            ],
            referral_recommended=True,
            crisis_resources=[
                {
                    "name": "National Suicide Prevention Lifeline",
                    "phone": "988",
                    "available": "24/7",
                },
                {
                    "name": "Crisis Text Line",
                    "text": "HOME to 741741",
                    "available": "24/7",
                },
            ],
            emergency_contacts=[
                {
                    "name": "Emergency Services",
                    "phone": "911",
                    "when": "Immediate danger",
                }
            ],
            journey_suspended=alert_level == "critical",
            admin_notified=True,
            followup_scheduled=datetime.utcnow() if alert_level == "critical" else None,
        )

    return None


@router.post(
    "/journey",
    response_model=APIResponse[JourneyResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Start new therapy journey",
    description="Begin ACT therapy journey with emotional diary entry (The Moment)",
)
async def start_journey(
    request: JourneyStartRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
    db_service: Annotated[SupabaseService, Depends(lambda: db)],
) -> APIResponse[JourneyResponse]:
    """치료 여정 시작 (The Moment 단계)"""

    try:
        user_id = current_user["user_id"]
        journey_id = str(uuid4())

        # 사용자 프로필 및 선호도 조회
        user_profile = await therapy_service.get_user_profile(user_id, db_service)

        # 감정 분석 수행
        emotion_analysis_result = await therapy_service.analyze_emotions(
            diary_text=request.diary_text, user_profile=user_profile
        )

        # 응급 상황 확인
        emergency = check_emergency_signals(
            request.diary_text, emotion_analysis_result.get("emotion_keywords", [])
        )

        if emergency and emergency.journey_suspended:
            logger.warning(f"응급 상황 감지로 여정 중단: {user_id}")
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED, detail=emergency.dict()
            )

        # VAD 점수 객체 생성
        vad_data = emotion_analysis_result.get("vad_scores", {})
        vad_scores = VADScores(
            valence=vad_data.get("valence", 0.0),
            arousal=vad_data.get("arousal", 0.0),
            dominance=vad_data.get("dominance", 0.0),
            confidence=vad_data.get("confidence", 0.5),
        )

        # 감정 분석 객체 생성
        emotion_analysis = EmotionAnalysis(
            emotion_keywords=emotion_analysis_result.get("emotion_keywords", []),
            vad_scores=vad_scores,
            dominant_emotion=emotion_analysis_result.get("dominant_emotion", "neutral"),
            emotion_intensity=emotion_analysis_result.get("emotion_intensity", 0.5),
            emotion_categories=emotion_analysis_result.get("emotion_categories", {}),
            sentiment_polarity=emotion_analysis_result.get("sentiment_polarity", 0.0),
            complexity_score=emotion_analysis_result.get("complexity_score", 0.5),
        )

        # 갤러리 아이템 생성 (초기 상태)
        gallery_data = {
            "id": journey_id,
            "user_id": user_id,
            "journey_id": journey_id,
            "diary_text": request.diary_text,
            "emotion_keywords": emotion_analysis_result.get("emotion_keywords", []),
            "vad_scores": vad_data,
            "completion_status": CompletionStatus.STARTED.value,
            "mood_rating": request.mood_rating,
            "trigger_events": request.trigger_events or [],
            "physical_symptoms": request.physical_symptoms or [],
            "created_date": datetime.utcnow().isoformat(),
        }

        # 데이터베이스에 저장
        saved_item = await db_service.create_gallery_item(gallery_data)

        # 응답 생성
        journey_response = JourneyResponse(
            journey_id=journey_id,
            user_id=user_id,
            status=CompletionStatus.STARTED,
            created_date=datetime.utcnow(),
            diary_text=request.diary_text,
            emotion_analysis=emotion_analysis,
            next_step="reflection",
            estimated_completion_time=15,
        )

        logger.info(f"새 치료 여정 시작: {user_id}, 여정ID: {journey_id}")

        # 응급 상황 경고가 있다면 포함
        message = "Journey started successfully"
        if emergency:
            message += f" (Alert: {emergency.alert_level} risk level detected)"

        return APIResponse(success=True, message=message, data=journey_response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"치료 여정 시작 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start therapy journey",
        )


@router.post(
    "/journey/{journey_id}/reflect",
    response_model=APIResponse[ReflectionResponse],
    summary="Generate reflection images",
    description="Generate personalized images based on emotions (Reflection phase)",
)
async def reflect_journey(
    journey_id: str,
    request: ReflectionRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
    db_service: Annotated[SupabaseService, Depends(lambda: db)],
) -> APIResponse[ReflectionResponse]:
    """성찰 단계 - 이미지 생성 (Reflection)"""

    try:
        user_id = current_user["user_id"]

        # 여정 존재 및 권한 확인
        gallery_item = await db_service.get_gallery_item(journey_id)
        if not gallery_item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Journey not found"
            )

        if gallery_item["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this journey",
            )

        if gallery_item["completion_status"] != CompletionStatus.STARTED.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Journey not ready for reflection phase",
            )

        # 사용자 프로필 및 선호도 조회
        user_profile = await therapy_service.get_user_profile(user_id, db_service)

        # GPT를 통한 이미지 프롬프트 생성
        prompt_data = await therapy_service.generate_image_prompt(
            gallery_item=gallery_item,
            user_profile=user_profile,
            custom_additions=request.custom_prompt_additions,
            style_override=request.image_style_override,
            use_personalization=request.use_personalization,
        )

        # 이미지 생성
        images = []
        total_cost = 0.0
        token_usage = {"prompt_tokens": 0, "total_tokens": 0}

        for i in range(request.image_count):
            generation_result = await image_service.generate_image(
                prompt=prompt_data["final_prompt"],
                width=512,
                height=512,
                num_inference_steps=20,
                guidance_scale=7.5,
                seed=None,
            )

            if generation_result["success"]:
                image_result = ImageGenerationResult(
                    image_url=generation_result["image_url"],
                    image_prompt=prompt_data["final_prompt"],
                    backend_used=generation_result["backend"],
                    generation_time=generation_result["generation_time"],
                    model_version=generation_result.get("metadata", {}).get(
                        "model_version"
                    ),
                    seed_used=generation_result.get("metadata", {}).get("seed"),
                    parameters=generation_result.get("metadata", {}),
                )
                images.append(image_result)

                # 비용 추적
                if "cost" in generation_result:
                    total_cost += generation_result["cost"]
            else:
                logger.error(f"이미지 생성 실패: {generation_result.get('error')}")

        if not images:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate any images",
            )

        # 대표 이미지 선택 (첫 번째)
        main_image = images[0]

        # 갤러리 아이템 업데이트
        update_data = {
            "image_url": main_image.image_url,
            "image_prompt": main_image.image_prompt,
            "completion_status": CompletionStatus.REFLECTING.value,
            "updated_date": datetime.utcnow().isoformat(),
        }

        await db_service.update_gallery_item(journey_id, update_data)

        # GPT 토큰 사용량 추적
        if "token_usage" in prompt_data:
            token_usage = prompt_data["token_usage"]

        # 비용 로깅
        if total_cost > 0:
            cost_data = {
                "user_id": user_id,
                "service_type": "image_generation",
                "tokens_used": 0,  # 이미지는 토큰 기반이 아님
                "cost_usd": total_cost,
                "api_call_metadata": {
                    "journey_id": journey_id,
                    "image_count": len(images),
                    "backend": main_image.backend_used,
                },
            }
            await db_service.log_cost(cost_data)

        # 응답 생성
        response = ReflectionResponse(
            journey_id=journey_id,
            images=images,
            prompt_reasoning=prompt_data.get(
                "reasoning", "GPT-generated prompt based on emotional analysis"
            ),
            personalization_applied=prompt_data.get("personalization_factors", []),
            generation_cost=total_cost,
            token_usage=token_usage,
        )

        logger.info(f"이미지 생성 완료: {journey_id}, {len(images)}개 생성")

        return APIResponse(
            success=True,
            message=f"Generated {len(images)} reflection images successfully",
            data=response,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"이미지 생성 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate reflection images",
        )


@router.post(
    "/journey/{journey_id}/defuse",
    response_model=APIResponse[DefusionResponse],
    summary="Create guestbook entry",
    description="Write guestbook entry for cognitive defusion (Defusion phase)",
)
async def defuse_journey(
    journey_id: str,
    request: DefusionRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
    db_service: Annotated[SupabaseService, Depends(lambda: db)],
) -> APIResponse[DefusionResponse]:
    """탈융합 단계 - 방명록 작성 (Defusion)"""

    try:
        user_id = current_user["user_id"]

        # 여정 존재 및 권한 확인
        gallery_item = await db_service.get_gallery_item(journey_id)
        if not gallery_item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Journey not found"
            )

        if gallery_item["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this journey",
            )

        if gallery_item["completion_status"] != CompletionStatus.REFLECTING.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Journey not ready for defusion phase",
            )

        # 인지적 거리두기 점수 계산 (간단한 버전)
        original_emotion_keywords = gallery_item.get("emotion_keywords", [])
        original_vad = gallery_item.get("vad_scores", {})

        # 방명록 내용의 감정 분석
        guestbook_emotion_analysis = await therapy_service.analyze_emotions(
            diary_text=request.guestbook_content,
            user_profile=await therapy_service.get_user_profile(user_id, db_service),
        )

        # 인지적 거리 계산 (VAD 차이 기반)
        original_valence = original_vad.get("valence", 0.0)
        guestbook_valence = guestbook_emotion_analysis.get("vad_scores", {}).get(
            "valence", 0.0
        )

        cognitive_distance_score = min(abs(guestbook_valence - original_valence), 1.0)

        # 재구성 품질 점수 (키워드 다양성 기반)
        original_emotions = set(original_emotion_keywords)
        guestbook_emotions = set(guestbook_emotion_analysis.get("emotion_keywords", []))

        if original_emotions:
            reframing_quality = len(guestbook_emotions - original_emotions) / len(
                original_emotions
            )
            reframing_quality = min(reframing_quality, 1.0)
        else:
            reframing_quality = 0.5

        # 갤러리 아이템 업데이트
        update_data = {
            "selected_image_url": request.selected_image_url,
            "guestbook_title": request.guestbook_title,
            "guestbook_content": request.guestbook_content,
            "guestbook_tags": request.guestbook_tags,
            "completion_status": CompletionStatus.DEFUSING.value,
            "image_satisfaction": request.image_satisfaction,
            "emotional_shift": request.emotional_shift,
            "updated_date": datetime.utcnow().isoformat(),
        }

        await db_service.update_gallery_item(journey_id, update_data)

        # 개인화 데이터 수집 (Level 2 학습용)
        personalization_data = {
            "user_id": user_id,
            "interaction_type": "guestbook_feedback",
            "feedback_data": {
                "image_satisfaction": request.image_satisfaction,
                "emotional_shift": request.emotional_shift,
                "cognitive_distance": cognitive_distance_score,
                "reframing_quality": reframing_quality,
                "tags_used": request.guestbook_tags,
            },
            "learning_weights": {
                "emotional_progress": cognitive_distance_score * 0.6
                + reframing_quality * 0.4
            },
        }

        await db_service.save_personalization_data(user_id, personalization_data)

        # 큐레이터 메시지 준비 시간 추정
        estimated_wait_time = 2  # 2분 (GPT 생성 시간)

        response = DefusionResponse(
            journey_id=journey_id,
            guestbook_saved=True,
            cognitive_distance_score=cognitive_distance_score,
            reframing_quality=reframing_quality,
            closure_ready=True,
            estimated_wait_time=estimated_wait_time,
        )

        logger.info(
            f"방명록 작성 완료: {journey_id}, 인지적 거리: {cognitive_distance_score:.2f}"
        )

        return APIResponse(
            success=True, message="Guestbook entry saved successfully", data=response
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"방명록 작성 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save guestbook entry",
        )


@router.post(
    "/journey/{journey_id}/close",
    response_model=APIResponse[ClosureResponse],
    summary="Generate curator message",
    description="Generate personalized curator message for closure (Closure phase)",
)
async def close_journey(
    journey_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
    db_service: Annotated[SupabaseService, Depends(lambda: db)],
) -> APIResponse[ClosureResponse]:
    """완료 단계 - 큐레이터 메시지 생성 (Closure)"""

    try:
        user_id = current_user["user_id"]

        # 여정 존재 및 권한 확인
        gallery_item = await db_service.get_gallery_item(journey_id)
        if not gallery_item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Journey not found"
            )

        if gallery_item["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this journey",
            )

        if gallery_item["completion_status"] != CompletionStatus.DEFUSING.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Journey not ready for closure phase",
            )

        # 사용자 프로필 조회
        user_profile = await therapy_service.get_user_profile(user_id, db_service)

        # GPT를 통한 큐레이터 메시지 생성
        curator_data = await therapy_service.generate_curator_message(
            gallery_item=gallery_item, user_profile=user_profile
        )

        # CuratorMessage 객체 생성
        curator_message = CuratorMessage(
            opening=curator_data.get("opening", ""),
            recognition=curator_data.get("recognition", ""),
            personal_note=curator_data.get("personal_note", ""),
            guidance=curator_data.get("guidance", ""),
            closing=curator_data.get("closing", ""),
            full_message=curator_data.get("full_message", ""),
            generation_method="gpt",
            personalization_level=user_profile.get("personalization_level", 1),
            therapeutic_appropriateness=curator_data.get("therapeutic_score", 0.8),
            personalization_relevance=curator_data.get("personalization_score", 0.7),
        )

        # 여정 완료 시간 계산
        start_time = datetime.fromisoformat(
            gallery_item["created_date"].replace("Z", "+00:00")
        )
        end_time = datetime.utcnow()
        journey_duration = (end_time - start_time).total_seconds() / 60  # 분 단위

        # 총 비용 계산
        cost_summary = await db_service.get_cost_summary(user_id=user_id, days=1)
        total_cost = cost_summary.get("total_cost", 0.0)

        # 달성한 마일스톤들
        milestones_achieved = [
            "completed_moment_phase",
            "completed_reflection_phase",
            "completed_defusion_phase",
        ]

        # 갤러리 아이템 최종 업데이트
        update_data = {
            "curator_message": curator_message.dict(),
            "completion_status": CompletionStatus.COMPLETED.value,
            "total_duration": journey_duration,
            "total_cost": total_cost,
            "completed_date": datetime.utcnow().isoformat(),
            "updated_date": datetime.utcnow().isoformat(),
        }

        await db_service.update_gallery_item(journey_id, update_data)

        # 비용 로깅 (큐레이터 메시지 생성)
        if "token_usage" in curator_data:
            cost_data = {
                "user_id": user_id,
                "service_type": "gpt_curator",
                "tokens_used": curator_data["token_usage"].get("total_tokens", 0),
                "cost_usd": curator_data.get("cost", 0.0),
                "api_call_metadata": {
                    "journey_id": journey_id,
                    "message_sections": 5,
                    "personalization_level": user_profile.get(
                        "personalization_level", 1
                    ),
                },
            }
            await db_service.log_cost(cost_data)

        # 다음 여정 제안
        completed_journeys = len(await db_service.get_gallery_items(user_id))
        next_suggestion = None

        if completed_journeys < 5:
            next_suggestion = (
                "Consider exploring different emotional themes in your next journey"
            )
        elif completed_journeys < 10:
            next_suggestion = (
                "You might benefit from focusing on specific triggers or patterns"
            )
        else:
            next_suggestion = (
                "You've made great progress! Continue with regular emotional check-ins"
            )

        # 학습 데이터 수집 완료 표시
        learning_data_collected = True

        response = ClosureResponse(
            journey_id=journey_id,
            curator_message=curator_message,
            journey_duration=journey_duration,
            total_cost=total_cost,
            milestones_achieved=milestones_achieved,
            next_journey_suggestion=next_suggestion,
            learning_data_collected=learning_data_collected,
        )

        logger.info(f"치료 여정 완료: {journey_id}, 소요시간: {journey_duration:.1f}분")

        return APIResponse(
            success=True, message="Journey completed successfully", data=response
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"치료 여정 완료 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to complete journey",
        )


@router.get(
    "/journey/{journey_id}",
    response_model=APIResponse[GalleryItem],
    summary="Get specific journey",
    description="Retrieve details of a specific therapy journey",
)
async def get_journey(
    journey_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
    db_service: Annotated[SupabaseService, Depends(lambda: db)],
) -> APIResponse[GalleryItem]:
    """특정 치료 여정 조회"""

    try:
        user_id = current_user["user_id"]

        # 갤러리 아이템 조회
        gallery_data = await db_service.get_gallery_item(journey_id)
        if not gallery_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Journey not found"
            )

        if gallery_data["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this journey",
            )

        # GalleryItem 모델 생성
        gallery_item = await convert_to_gallery_item(gallery_data)

        logger.info(f"치료 여정 조회: {journey_id}")

        return APIResponse(
            success=True, message="Journey retrieved successfully", data=gallery_item
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"치료 여정 조회 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve journey",
        )


@router.get(
    "/journeys",
    response_model=APIResponse[GalleryListResponse],
    summary="Get user's journeys",
    description="Retrieve list of user's therapy journeys with filtering options",
)
async def get_journeys(
    current_user: Annotated[dict, Depends(get_current_user)],
    db_service: Annotated[SupabaseService, Depends(lambda: db)],
    status_filter: Optional[CompletionStatus] = Query(
        None, description="Filter by completion status"
    ),
    limit: int = Query(20, ge=1, le=100, description="Number of items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
) -> APIResponse[GalleryListResponse]:
    """사용자의 치료 여정 목록 조회"""

    try:
        user_id = current_user["user_id"]

        # 갤러리 아이템들 조회
        gallery_items_data = await db_service.get_gallery_items(
            user_id=user_id,
            limit=limit,
            offset=offset,
            status_filter=status_filter.value if status_filter else None,
        )

        # GalleryItem 모델들로 변환
        gallery_items = []
        for data in gallery_items_data:
            item = await convert_to_gallery_item(data)
            gallery_items.append(item)

        # 전체 개수 조회 (간단한 버전)
        total_count = len(await db_service.get_gallery_items(user_id))

        # 필터링 정보
        applied_filters = {}
        if status_filter:
            applied_filters["status"] = status_filter.value

        # 사용 가능한 태그들 수집
        all_tags = set()
        for item in gallery_items:
            all_tags.update(item.guestbook_tags)
        available_tags = list(all_tags)

        # 완료 상태별 통계
        completion_stats = {}
        for status in CompletionStatus:
            count = len(
                [item for item in gallery_items if item.completion_status == status]
            )
            completion_stats[status.value] = count

        # 감정 분포 (간단한 버전)
        emotion_distribution = {}
        for item in gallery_items:
            for emotion in item.emotion_analysis.emotion_keywords[:3]:  # 상위 3개만
                emotion_distribution[emotion] = emotion_distribution.get(emotion, 0) + 1

        response = GalleryListResponse(
            items=gallery_items,
            total_count=total_count,
            applied_filters=applied_filters,
            available_tags=available_tags,
            completion_stats=completion_stats,
            emotion_distribution=emotion_distribution,
        )

        logger.info(f"치료 여정 목록 조회: {user_id}, {len(gallery_items)}개")

        return APIResponse(
            success=True,
            message=f"Retrieved {len(gallery_items)} journeys successfully",
            data=response,
        )

    except Exception as e:
        logger.error(f"치료 여정 목록 조회 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve journeys",
        )


async def convert_to_gallery_item(data: Dict[str, Any]) -> GalleryItem:
    """데이터베이스 데이터를 GalleryItem 모델로 변환"""

    # 감정 분석 데이터 변환
    vad_data = data.get("vad_scores", {})
    vad_scores = VADScores(
        valence=vad_data.get("valence", 0.0),
        arousal=vad_data.get("arousal", 0.0),
        dominance=vad_data.get("dominance", 0.0),
        confidence=vad_data.get("confidence", 0.5),
    )

    emotion_analysis = EmotionAnalysis(
        emotion_keywords=data.get("emotion_keywords", []),
        vad_scores=vad_scores,
        dominant_emotion=data.get("dominant_emotion", "neutral"),
        emotion_intensity=data.get("emotion_intensity", 0.5),
        emotion_categories=data.get("emotion_categories", {}),
        sentiment_polarity=data.get("sentiment_polarity", 0.0),
        complexity_score=data.get("complexity_score", 0.5),
    )

    # 큐레이터 메시지 데이터 변환
    curator_data = data.get("curator_message", {})
    if isinstance(curator_data, dict) and curator_data:
        curator_message = CuratorMessage(**curator_data)
    else:
        # 기본 큐레이터 메시지
        curator_message = CuratorMessage(
            opening="",
            recognition="",
            personal_note="",
            guidance="",
            closing="",
            full_message="",
            generation_method="pending",
            personalization_level=1,
            therapeutic_appropriateness=0.0,
            personalization_relevance=0.0,
        )

    return GalleryItem(
        id=data["id"],
        user_id=data["user_id"],
        journey_id=data["journey_id"],
        diary_text=data["diary_text"],
        emotion_analysis=emotion_analysis,
        image_url=data.get("image_url", ""),
        image_prompt=data.get("image_prompt", ""),
        guestbook_title=data.get("guestbook_title", ""),
        guestbook_content=data.get("guestbook_content", ""),
        guestbook_tags=data.get("guestbook_tags", []),
        curator_message=curator_message,
        completion_status=CompletionStatus(data.get("completion_status", "started")),
        total_duration=data.get("total_duration", 0.0),
        total_cost=data.get("total_cost", 0.0),
        user_rating=data.get("user_rating"),
        user_feedback=data.get("user_feedback"),
        is_favorite=data.get("is_favorite", False),
        therapeutic_effectiveness=data.get("therapeutic_effectiveness"),
        emotion_shift_analysis=data.get("emotion_shift_analysis"),
        created_date=datetime.fromisoformat(
            data["created_date"].replace("Z", "+00:00")
        ),
        updated_date=datetime.fromisoformat(
            data.get("updated_date", data["created_date"]).replace("Z", "+00:00")
        ),
    )
