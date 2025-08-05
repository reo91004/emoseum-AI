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
from ..services.emoseum_client import emoseum_client
from ..services.diary_exploration_service import get_api_diary_exploration_service
from ..models.therapy import (
    StartSessionRequest,
    SessionResponse,
    DiaryEntryRequest,
    DiaryAnalysisResponse,
    ImageGenerationResponse,
    ArtworkTitleRequest,
    ArtworkTitleResponse,
    DocentMessageResponse,
    TherapySessionDetailResponse,
    DiaryExplorationRequest,
    DiaryExplorationResponse,
    DiaryFollowUpRequest,
    ExplorationQuestion,
    JourneyStage,
    EmotionAnalysis,
    GeneratedImage,
    ImageGenerationMetadata,
    ArtworkTitle,
    DocentMessage,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/therapy", tags=["Therapy Sessions"])

# 엔드포인트별 요청 제한
diary_rate_limiter = RateLimiter(calls=10, period=60)
image_rate_limiter = RateLimiter(calls=5, period=60)


@router.post("/sessions", response_model=SessionResponse)
async def start_therapy_session(
    current_user: Dict[str, Any] = Depends(get_current_user),
    act_system=Depends(get_act_therapy_system),
):
    """새로운 치료 세션 시작"""
    logger.info(f"Starting therapy session for user: {current_user['user_id']}")
    try:
        # 미완료 세션 체크 비활성화 (테스트용)
        # incomplete_journeys = act_system.gallery_manager.get_incomplete_journeys(current_user["user_id"])
        logger.info("Skipping incomplete journey check for testing")

        # 새 세션 ID 생성
        session_id = str(uuid.uuid4())

        return SessionResponse(
            session_id=session_id,
            user_id=current_user["user_id"],
            created_date=datetime.utcnow(),
            journey_stage=JourneyStage.THE_MOMENT,
            is_completed=False,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting therapy session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start therapy session",
        )


@router.post("/sessions/{session_id}/diary", response_model=DiaryAnalysisResponse)
async def submit_diary_entry(
    session_id: str,
    diary: DiaryEntryRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    act_system=Depends(get_act_therapy_system),
    _: Any = Depends(diary_rate_limiter),
):
    """일기 제출 및 감정 분석, 성찰 이미지 생성"""
    try:
        # ACT 시스템을 통한 감정 여정 처리 (일기 분석 + 이미지 생성)
        result = act_system.process_emotion_journey(
            user_id=current_user["user_id"], diary_text=diary.diary_text
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process diary entry",
            )

        # 이후 API 호출을 위한 gallery_item_id 저장
        gallery_item_id = result["gallery_item_id"]

        # DB에 journey_stage와 is_completed 업데이트
        try:
            db = await get_database()
            await db["gallery_items"].update_one(
                {"item_id": gallery_item_id},
                {"$set": {"journey_stage": "defusion", "is_completed": False}},
            )
            logger.info(f"Updated journey stage for {gallery_item_id}")
        except Exception as e:
            logger.error(f"Failed to update journey stage: {e}")

        # Emoseum 서버로 결과 전송
        try:
            # gallery_item에서 reflection_prompt 가져오기
            gallery_item = act_system.gallery_manager.get_gallery_item(gallery_item_id)
            reflection_prompt = gallery_item.reflection_prompt if gallery_item else ""

            logger.info(f"Reflection prompt from gallery_item: '{reflection_prompt}'")

            await emoseum_client.update_diary_from_ai(
                diary_id=diary.diary_id,
                keywords=result["emotion_analysis"]["keywords"],
                image_path=result.get("reflection_image", {}).get("image_path", ""),
                reflection_prompt=reflection_prompt,
            )
        except Exception as e:
            logger.warning(f"Failed to sync with central server: {e}")

        return DiaryAnalysisResponse(
            session_id=gallery_item_id,  # API 호환성을 위해 gallery_item_id를 session_id로 사용
            emotion_analysis=EmotionAnalysis(
                keywords=result["emotion_analysis"]["keywords"],
                vad_scores=result["emotion_analysis"]["vad_scores"],
                primary_emotion=(
                    result["emotion_analysis"]["keywords"][0]
                    if result["emotion_analysis"]["keywords"]
                    else "neutral"
                ),
                intensity=0.7,
            ),
            next_stage=JourneyStage.DEFUSION,  # 이미지가 이미 생성되었으므로 reflection 단계 건너뛰기
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing diary entry: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process diary entry",
        )


@router.post("/sessions/{session_id}/reflect", response_model=ImageGenerationResponse)
async def generate_reflection_image(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    act_system=Depends(get_act_therapy_system),
    _: Any = Depends(image_rate_limiter),
):
    """성찰 이미지 정보 조회 (이미지는 일기 단계에서 생성됨)"""
    try:
        # 갤러리 아이템 조회 (session_id는 실제로 gallery_item_id)
        gallery_item = act_system.gallery_manager.get_gallery_item(session_id)

        if not gallery_item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="세션을 찾을 수 없음 - 일기 작성이 실패했을 수 있음",
            )

        if gallery_item.user_id != current_user["user_id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="이 세션에 대한 접근 거부"
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
            generation_time=30.0,
            next_stage=JourneyStage.DEFUSION,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting reflection image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get reflection image",
        )


@router.post("/sessions/{session_id}/artwork-title", response_model=ArtworkTitleResponse)
async def create_artwork_title(
    session_id: str,
    artwork_title: ArtworkTitleRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    act_system=Depends(get_act_therapy_system),
):
    """작품 제목 작성 (3단계: Defusion)"""
    try:
        # session_id is actually gallery_item_id
        gallery_item_id = session_id

        # ACT 시스템을 통한 작품 제목 완성
        result = act_system.complete_artwork_title(
            user_id=current_user["user_id"],
            gallery_item_id=gallery_item_id,
            artwork_title=artwork_title.title,
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create artwork title",
            )

        return ArtworkTitleResponse(
            session_id=session_id,
            artwork_title=ArtworkTitle(
                title=artwork_title.title,
                reflection=artwork_title.reflection or "",
            ),
            next_stage=JourneyStage.CLOSURE,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating artwork title: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create artwork title",
        )


@router.post("/sessions/{session_id}/docent", response_model=DocentMessageResponse)
async def generate_docent_message(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    act_system=Depends(get_act_therapy_system),
):
    """도슨트 메시지 생성 (4단계: Closure)"""
    try:
        # session_id is actually gallery_item_id
        gallery_item_id = session_id

        # ACT 시스템을 통한 도슨트 메시지 생성
        result = act_system.create_docent_message(
            user_id=current_user["user_id"], gallery_item_id=gallery_item_id
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate docent message",
            )

        # ACT 시스템 결과에서 메시지 내용 추출
        docent_message_content = result.get("docent_message", {})
        if (
            isinstance(docent_message_content, dict)
            and "content" in docent_message_content
        ):
            # 메시지 내용을 하나의 문자열로 포맷
            content = docent_message_content["content"]
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

        return DocentMessageResponse(
            session_id=session_id,
            docent_message=DocentMessage(
                message=message, message_type="encouragement", personalization_data={}
            ),
            journey_completed=True,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating docent message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate curator message",
        )


@router.post("/diary/explore", response_model=DiaryExplorationResponse)
async def explore_diary_emotions(
    request: DiaryExplorationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    exploration_service=Depends(get_api_diary_exploration_service),
    rate_limit=Depends(diary_rate_limiter)
):
    """일기 내용에 대한 심화 탐색 질문 생성"""
    try:
        logger.info(f"사용자 {current_user['user_id']} 일기 심화 탐색 요청")
        
        # 일기 심화 탐색 질문 생성
        result = await exploration_service.generate_exploration_questions(
            diary_text=request.diary_text,
            emotion_keywords=request.emotion_keywords
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"질문 생성 실패: {result.get('error', 'Unknown error')}"
            )
        
        # 응답 형식에 맞게 변환
        questions = []
        for q in result.get("questions", []):
            questions.append(ExplorationQuestion(
                question=q.get("question", ""),
                category=q.get("category", "general"),
                explanation=q.get("explanation", "")
            ))
        
        # 감정 분석 데이터가 있으면 변환
        emotion_analysis = None
        if "emotion_analysis" in result:
            ea = result["emotion_analysis"]
            emotion_analysis = EmotionAnalysis(
                keywords=ea.get("keywords", []),
                vad_scores=ea.get("vad_scores", [0.5, 0.5, 0.5]),
                primary_emotion=ea.get("primary_emotion", "neutral"),
                intensity=ea.get("confidence", 0.5)  # confidence를 intensity로 사용
            )
        
        return DiaryExplorationResponse(
            success=True,
            questions=questions,
            exploration_theme=result.get("exploration_theme", "감정 탐색"),
            encouragement=result.get("encouragement", "천천히 자신의 감정을 탐색해보세요."),
            emotion_analysis=emotion_analysis,
            generation_timestamp=result.get("generation_timestamp")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"일기 심화 탐색 중 오류 발생: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="일기 심화 탐색 중 오류가 발생했습니다."
        )


@router.post("/diary/explore/follow-up", response_model=DiaryExplorationResponse)
async def generate_follow_up_question(
    request: DiaryFollowUpRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    exploration_service=Depends(get_api_diary_exploration_service),
    rate_limit=Depends(diary_rate_limiter)
):
    """이전 답변을 바탕으로 후속 질문 생성"""
    try:
        logger.info(f"사용자 {current_user['user_id']} 후속 질문 생성 요청")
        
        # 후속 질문 생성
        result = await exploration_service.generate_follow_up_question(
            diary_text=request.diary_text,
            previous_question=request.previous_question,
            user_response=request.user_response,
            emotion_keywords=request.emotion_keywords
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"후속 질문 생성 실패: {result.get('error', 'Unknown error')}"
            )
        
        # 응답 형식에 맞게 변환
        questions = []
        for q in result.get("questions", []):
            questions.append(ExplorationQuestion(
                question=q.get("question", ""),
                category=q.get("category", "general"),
                explanation=q.get("explanation", "")
            ))
        
        # 감정 분석 데이터가 있으면 변환
        emotion_analysis = None
        if "emotion_analysis" in result:
            ea = result["emotion_analysis"]
            emotion_analysis = EmotionAnalysis(
                keywords=ea.get("keywords", []),
                vad_scores=ea.get("vad_scores", [0.5, 0.5, 0.5]),
                primary_emotion=ea.get("primary_emotion", "neutral"),
                intensity=ea.get("confidence", 0.5)
            )
        
        return DiaryExplorationResponse(
            success=True,
            questions=questions,
            exploration_theme=result.get("exploration_theme", "Continued Exploration"),
            encouragement=result.get("encouragement", "Thank you for sharing. Let's continue exploring."),
            emotion_analysis=emotion_analysis,
            generation_timestamp=result.get("generation_timestamp")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"후속 질문 생성 중 오류 발생: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="후속 질문 생성 중 오류가 발생했습니다."
        )


@router.get("/diary/explore/categories")
async def get_exploration_categories(
    current_user: Dict[str, Any] = Depends(get_current_user),
    exploration_service=Depends(get_api_diary_exploration_service)
):
    """질문 카테고리 정보 조회"""
    try:
        result = await exploration_service.get_question_categories()
        return result
    except Exception as e:
        logger.error(f"질문 카테고리 조회 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="질문 카테고리 조회에 실패했습니다."
        )


@router.get("/diary/explore/safety")
async def get_safety_guidelines(
    current_user: Dict[str, Any] = Depends(get_current_user),
    exploration_service=Depends(get_api_diary_exploration_service)
):
    """안전 가이드라인 조회"""
    try:
        result = await exploration_service.get_safety_guidelines()
        return result
    except Exception as e:
        logger.error(f"안전 가이드라인 조회 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="안전 가이드라인 조회에 실패했습니다."
        )


@router.get("/sessions/{session_id}", response_model=TherapySessionDetailResponse)
async def get_session_details(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    act_system=Depends(get_act_therapy_system),
):
    """치료 세션 상세 조회"""
    try:
        # session_id is actually gallery_item_id
        gallery_item = act_system.gallery_manager.get_gallery_item(session_id)

        if not gallery_item or gallery_item.user_id != current_user["user_id"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="세션을 찾을 수 없음"
            )

        # 여정 단계 및 완료 상태 결정
        completion_status = gallery_item.get_completion_status()
        logger.info(f"Completion status for {session_id}: {completion_status}")
        logger.info(
            f"Gallery item data: reflection_path={gallery_item.reflection_image_path}, artwork_title={gallery_item.artwork_title}, docent_message={gallery_item.docent_message}"
        )

        if completion_status["docent_message"]:
            journey_stage = JourneyStage.CLOSURE
            is_completed = True
        elif completion_status["artwork_title"]:
            journey_stage = JourneyStage.CLOSURE
            is_completed = False
        elif completion_status["reflection"]:
            journey_stage = JourneyStage.DEFUSION
            is_completed = False
        else:
            journey_stage = JourneyStage.THE_MOMENT
            is_completed = False

        logger.info(f"Final stage: {journey_stage}, completed: {is_completed}")

        # 응답 구성
        response = TherapySessionDetailResponse(
            session_id=session_id,
            user_id=gallery_item.user_id,
            created_date=datetime.fromisoformat(gallery_item.created_date),
            journey_stage=journey_stage,
            is_completed=is_completed,
            diary_text=gallery_item.diary_text,
            emotion_analysis=(
                EmotionAnalysis(
                    keywords=gallery_item.emotion_keywords,
                    vad_scores=gallery_item.vad_scores,
                    primary_emotion=(
                        gallery_item.emotion_keywords[0]
                        if gallery_item.emotion_keywords
                        else "neutral"
                    ),
                    intensity=0.7,
                )
                if gallery_item.emotion_keywords
                else None
            ),
            generated_image=(
                GeneratedImage(
                    image_path=(
                        f"/therapy/images/{os.path.basename(gallery_item.reflection_image_path)}"
                        if gallery_item.reflection_image_path
                        else ""
                    ),
                    prompt_used=gallery_item.reflection_prompt or "",
                    generation_metadata=ImageGenerationMetadata(
                        generation_time=30.0, model_version="stable-diffusion-v1-5"
                    ),
                )
                if gallery_item.reflection_image_path
                else None
            ),
            artwork_title=(
                ArtworkTitle(
                    title=gallery_item.artwork_title,
                    reflection="",
                )
                if gallery_item.artwork_title
                else None
            ),
            docent_message=(
                DocentMessage(
                    message="Docent message generated",
                    message_type="encouragement",
                    personalization_data={},
                )
                if gallery_item.docent_message
                else None
            ),
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get session details",
        )


@router.get("/images/{filename}")
async def get_generated_image(
    filename: str, current_user: Dict[str, Any] = Depends(get_current_user)
):
    """생성된 이미지 제공"""
    try:
        # 보안 검사 - 사용자가 이미지를 소유하고 있는지 확인
        user_prefix = f"{current_user['user_id']}_"
        if not filename.startswith(user_prefix):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="접근 거부"
            )

        # 파일 경로 구성
        image_path = os.path.join("data", "gallery_images", "reflection", filename)

        if not os.path.exists(image_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="이미지를 찾을 수 없음"
            )

        return FileResponse(image_path, media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to serve image",
        )
