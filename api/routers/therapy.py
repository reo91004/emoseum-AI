# api/routers/therapy.py

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

# Rate limiters for different endpoints
diary_rate_limiter = RateLimiter(calls=10, period=60)
image_rate_limiter = RateLimiter(calls=5, period=60)


@router.post("/sessions", response_model=SessionResponse)
async def start_therapy_session(
    current_user: Dict[str, Any] = Depends(get_current_user),
    act_system = Depends(get_act_therapy_system)
):
    """Start a new therapy session"""
    try:
        # Check for incomplete sessions through ACT system
        incomplete_journeys = act_system.gallery_manager.get_incomplete_journeys(current_user["user_id"])
        
        if incomplete_journeys:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Please complete your current session before starting a new one. Session ID: {incomplete_journeys[0].item_id}"
            )
        
        # Create new session ID (placeholder for now)
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
    """Submit diary entry for emotion analysis and reflection image generation"""
    try:
        # Process emotion journey through ACT system (combines diary analysis + image generation)
        result = act_system.process_emotion_journey(
            user_id=current_user["user_id"],
            diary_text=diary.diary_text
        )
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process diary entry"
            )
        
        # Store gallery_item_id for subsequent API calls
        # In a real implementation, you'd want to map session_id to gallery_item_id
        gallery_item_id = result["gallery_item_id"]
        
        return DiaryAnalysisResponse(
            session_id=gallery_item_id,  # Use gallery_item_id as session_id for API compatibility
            emotion_analysis=EmotionAnalysis(
                keywords=result["emotion_analysis"]["keywords"],
                vad_scores=result["emotion_analysis"]["vad_scores"],
                primary_emotion=result["emotion_analysis"]["keywords"][0] if result["emotion_analysis"]["keywords"] else "neutral",
                intensity=0.7  # Default intensity
            ),
            next_stage=JourneyStage.DEFUSION  # Skip reflection since image is already generated
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
    """Get reflection image info (image already generated in diary step)"""
    try:
        # Get gallery item (session_id is actually gallery_item_id)
        gallery_item = act_system.gallery_manager.get_gallery_item(session_id)
        
        if not gallery_item or gallery_item.user_id != current_user["user_id"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        # Convert file path to URL
        if gallery_item.reflection_image_path:
            image_filename = os.path.basename(gallery_item.reflection_image_path)
            image_url = f"/therapy/images/{image_filename}"
        else:
            image_url = "/therapy/images/placeholder.png"
        
        return ImageGenerationResponse(
            session_id=session_id,
            image_url=image_url,
            prompt_used=gallery_item.reflection_prompt or "Reflection image",
            generation_time=30.0,  # Default value
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
    """Create guestbook entry (Stage 3: Defusion)"""
    try:
        # session_id is actually gallery_item_id
        gallery_item_id = session_id
        
        # Complete guestbook through ACT system
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
    """Generate curator message (Stage 4: Closure)"""
    try:
        # session_id is actually gallery_item_id
        gallery_item_id = session_id
        
        # Generate curator message through ACT system
        result = act_system.create_curator_message(
            user_id=current_user["user_id"],
            gallery_item_id=gallery_item_id
        )
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate curator message"
            )
        
        # Extract message content from ACT system result
        curator_message_content = result.get("curator_message", {})
        if isinstance(curator_message_content, dict) and "content" in curator_message_content:
            # Format the message content into a single string
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
    """Get therapy session details"""
    try:
        # session_id is actually gallery_item_id
        gallery_item = act_system.gallery_manager.get_gallery_item(session_id)
        
        if not gallery_item or gallery_item.user_id != current_user["user_id"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        # Determine journey stage and completion status
        status = gallery_item.get_completion_status()
        if status["curator_message"]:
            journey_stage = JourneyStage.CLOSURE
            is_completed = True
        elif status["guestbook"]:
            journey_stage = JourneyStage.CLOSURE
            is_completed = False
        elif status["reflection"]:
            journey_stage = JourneyStage.DEFUSION
            is_completed = False
        else:
            journey_stage = JourneyStage.THE_MOMENT
            is_completed = False
        
        # Build response
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
                image_url=f"/therapy/images/{os.path.basename(gallery_item.reflection_image_path)}" if gallery_item.reflection_image_path else None,
                prompt=gallery_item.reflection_prompt or "",
                metadata=ImageGenerationMetadata(
                    generation_time=30.0,
                    model_used="stable-diffusion"
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
    """Serve generated images"""
    try:
        # Security check - ensure user owns the image
        user_prefix = f"{current_user['user_id']}_"
        if not filename.startswith(user_prefix):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Build file path
        image_path = os.path.join("data", "gallery_images", "reflection", filename)
        
        if not os.path.exists(image_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Image not found"
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