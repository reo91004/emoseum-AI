# api/routers/therapy.py

# ==============================================================================
# 이 파일은 치료 세션 관련 API 엔드포인트를 정의한다.
# ACT 4단계 여정(The Moment, Reflection, Defusion, Closure)을 통한
# 감정 탐색과 치료적 경험을 제공한다.
# ==============================================================================

import logging
import uuid
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import FileResponse
from motor.motor_asyncio import AsyncIOMotorDatabase
import os

from ..database.connection import get_database
from ..database.collections import Collections
from ..dependencies import get_current_user, get_act_therapy_system, RateLimiter
from ..models.therapy import (
    StartSessionRequest,
    SessionResponse,
    DiaryEntryRequest,
    DiaryAnalysisResponse,
    ImageGenerationResponse,
    GuestbookRequest,
    GuestbookResponse,
    CuratorMessageResponse,
    TherapySessionDetailResponse,
    JourneyStage,
    EmotionAnalysis,
    GeneratedImage,
    ImageGenerationMetadata,
    GuestbookEntry,
    CuratorMessage
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/therapy", tags=["Therapy Sessions"])

# 엔드포인트별 요청 제한
diary_rate_limiter = RateLimiter(calls=10, period=60)
image_rate_limiter = RateLimiter(calls=5, period=60)


@router.post("/sessions", response_model=SessionResponse)
async def start_therapy_session(
    current_user: Dict[str, Any] = Depends(get_current_user),
    act_system = Depends(get_act_therapy_system)
):
    """새로운 치료 세션 시작"""
    try:
        # ACT 시스템을 통해 미완료 세션 확인
        incomplete_journeys = act_system.gallery_manager.get_incomplete_journeys(current_user["user_id"])
        
        if incomplete_journeys:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Please complete your current session before starting a new one. Session ID: {incomplete_journeys[0].item_id}"
            )
        
        # 새 세션 ID 생성
        session_id = str(uuid.uuid4())
        
        return SessionResponse(
            session_id=session_id,
            user_id=current_user["user_id"],
            created_date=datetime.utcnow(),
            journey_stage=JourneyStage.THE_MOMENT,
            is_completed=False
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting therapy session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start therapy session"
        )


@router.post("/sessions/{session_id}/diary", response_model=DiaryAnalysisResponse)
async def submit_diary_entry(
    session_id: str,
    diary: DiaryEntryRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    act_system = Depends(get_act_therapy_system),
    _: Any = Depends(diary_rate_limiter)
):
    """일기 제출 및 감정 분석, 성찰 이미지 생성"""
    try:
        # ACT 시스템을 통한 감정 여정 처리 (일기 분석 + 이미지 생성)
        result = act_system.process_emotion_journey(
            user_id=current_user["user_id"],
            diary_text=diary.diary_text
        )
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process diary entry"
            )
        
        # 이후 API 호출을 위한 gallery_item_id 저장
        gallery_item_id = result["gallery_item_id"]
        
        return DiaryAnalysisResponse(
            session_id=gallery_item_id,  # API 호환성을 위해 gallery_item_id를 session_id로 사용
            emotion_analysis=EmotionAnalysis(
                keywords=result["emotion_analysis"]["keywords"],
                vad_scores=result["emotion_analysis"]["vad_scores"],
                primary_emotion=result["emotion_analysis"]["keywords"][0] if result["emotion_analysis"]["keywords"] else "neutral",
                intensity=0.7
            ),
            next_stage=JourneyStage.DEFUSION  # 이미지가 이미 생성되었으므로 reflection 단계 건너뛰기
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing diary entry: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process diary entry"
        )


@router.post("/sessions/{session_id}/reflect", response_model=ImageGenerationResponse)
async def generate_reflection_image(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    act_system = Depends(get_act_therapy_system),
    _: Any = Depends(image_rate_limiter)
):
    """성찰 이미지 정보 조회 (이미지는 일기 단계에서 생성됨)"""
    try:
        # 갤러리 아이템 조회 (session_id는 실제로 gallery_item_id)
        gallery_item = act_system.gallery_manager.get_gallery_item(session_id)
        
        if not gallery_item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="세션을 찾을 수 없음 - 일기 작성이 실패했을 수 있음"
            )
        
        if gallery_item.user_id != current_user["user_id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="이 세션에 대한 접근 거부"
            )
        
        # 파일 경로를 URL로 변환
        if gallery_item.reflection_image_path:
            image_filename = os.path.basename(gallery_item.reflection_image_path)
            image_url = f"/therapy/images/{image_filename}"
        else:
            image_url = "/therapy/images/default.png"
        
        return ImageGenerationResponse(
            session_id=session_id,
            image_url=image_url,
            prompt_used=gallery_item.reflection_prompt or "Reflection image",
            generation_time=30.0
            next_stage=JourneyStage.DEFUSION
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting reflection image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get reflection image"
        )


@router.post("/sessions/{session_id}/guestbook", response_model=GuestbookResponse)
async def create_guestbook_entry(
    session_id: str,
    guestbook: GuestbookRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    act_system = Depends(get_act_therapy_system)
):
    """방명록 작성 (3단계: Defusion)"""
    try:
        # session_id is actually gallery_item_id
        gallery_item_id = session_id
        
        # ACT 시스템을 통한 방명록 완성
        result = act_system.complete_guestbook(
            user_id=current_user["user_id"],
            gallery_item_id=gallery_item_id,
            guestbook_title=guestbook.title,
            guestbook_tags=guestbook.tags
        )
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create guestbook entry"
            )
        
        return GuestbookResponse(
            session_id=session_id,
            guestbook_entry=GuestbookEntry(
                title=guestbook.title,
                tags=guestbook.tags,
                reflection=guestbook.reflection or ""
            ),
            next_stage=JourneyStage.CLOSURE
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating guestbook entry: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create guestbook entry"
        )


@router.post("/sessions/{session_id}/curator", response_model=CuratorMessageResponse)
async def generate_curator_message(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    act_system = Depends(get_act_therapy_system)
):
    """큐레이터 메시지 생성 (4단계: Closure)"""
    try:
        # session_id is actually gallery_item_id
        gallery_item_id = session_id
        
        # ACT 시스템을 통한 큐레이터 메시지 생성
        result = act_system.create_curator_message(
            user_id=current_user["user_id"],
            gallery_item_id=gallery_item_id
        )
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate curator message"
            )
        
        # ACT 시스템 결과에서 메시지 내용 추출
        curator_message_content = result.get("curator_message", {})
        if isinstance(curator_message_content, dict) and "content" in curator_message_content:
            # 메시지 내용을 하나의 문자열로 포맷
            content = curator_message_content["content"]
            message_parts = []
            if content.get("opening"):
                message_parts.append(content["opening"])
            if content.get("recognition"):
                message_parts.append(content["recognition"])
            if content.get("personal_note"):
                message_parts.append(content["personal_note"])
            if content.get("guidance"):
                message_parts.append(content["guidance"])
            if content.get("closing"):
                message_parts.append(content["closing"])
            
            message = " ".join(message_parts)
        else:
            message = "Thank you for completing your emotional journey."
        
        return CuratorMessageResponse(
            session_id=session_id,
            curator_message=CuratorMessage(
                message=message,
                message_type="encouragement",
                personalization_data={}
            ),
            journey_completed=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating curator message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate curator message"
        )


@router.get("/sessions/{session_id}", response_model=TherapySessionDetailResponse)
async def get_session_details(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    act_system = Depends(get_act_therapy_system)
):
    """치료 세션 상세 조회"""
    try:
        # session_id is actually gallery_item_id
        gallery_item = act_system.gallery_manager.get_gallery_item(session_id)
        
        if not gallery_item or gallery_item.user_id != current_user["user_id"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="세션을 찾을 수 없음"
            )
        
        # 여정 단계 및 완료 상태 결정
        completion_status = gallery_item.get_completion_status()
        if completion_status["curator_message"]:
            journey_stage = JourneyStage.CLOSURE
            is_completed = True
        elif completion_status["guestbook"]:
            journey_stage = JourneyStage.CLOSURE
            is_completed = False
        elif completion_status["reflection"]:
            journey_stage = JourneyStage.DEFUSION
            is_completed = False
        else:
            journey_stage = JourneyStage.THE_MOMENT
            is_completed = False
        
        # 응답 구성
        response = TherapySessionDetailResponse(
            session_id=session_id,
            user_id=gallery_item.user_id,
            created_date=datetime.fromisoformat(gallery_item.created_date),
            journey_stage=journey_stage,
            is_completed=is_completed,
            diary_text=gallery_item.diary_text,
            emotion_analysis=EmotionAnalysis(
                keywords=gallery_item.emotion_keywords,
                vad_scores=gallery_item.vad_scores,
                primary_emotion=gallery_item.emotion_keywords[0] if gallery_item.emotion_keywords else "neutral",
                intensity=0.7
            ) if gallery_item.emotion_keywords else None,
            generated_image=GeneratedImage(
                image_path=f"/therapy/images/{os.path.basename(gallery_item.reflection_image_path)}" if gallery_item.reflection_image_path else "",
                prompt_used=gallery_item.reflection_prompt or "",
                generation_metadata=ImageGenerationMetadata(
                    generation_time=30.0,
                    model_version="stable-diffusion-v1-5"
                )
            ) if gallery_item.reflection_image_path else None,
            guestbook_entry=GuestbookEntry(
                title=gallery_item.guestbook_title,
                tags=gallery_item.guestbook_tags,
                reflection=""
            ) if gallery_item.guestbook_title else None,
            curator_message=CuratorMessage(
                message="Curator message generated",
                message_type="encouragement",
                personalization_data={}
            ) if gallery_item.curator_message else None
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get session details"
        )


@router.get("/images/{filename}")
async def get_generated_image(
    filename: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """생성된 이미지 제공"""
    try:
        # 보안 검사 - 사용자가 이미지를 소유하고 있는지 확인
        user_prefix = f"{current_user['user_id']}_"
        if not filename.startswith(user_prefix):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="접근 거부"
            )
        
        # 파일 경로 구성
        image_path = os.path.join("data", "gallery_images", "reflection", filename)
        
        if not os.path.exists(image_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="이미지를 찾을 수 없음"
            )
        
        return FileResponse(image_path, media_type="image/png")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to serve image"
        )