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
    db: AsyncIOMotorDatabase = Depends(get_database),
    act_system = Depends(get_act_therapy_system)
):
    """Start a new therapy session"""
    try:
        # Check for incomplete sessions
        incomplete_session = await db[Collections.GALLERY_ITEMS].find_one(
            {"user_id": current_user["user_id"], "is_completed": False}
        )
        
        if incomplete_session:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Please complete your current session before starting a new one. Session ID: {incomplete_session['session_id']}"
            )
        
        # Create new session
        session_id = str(uuid.uuid4())
        journey_data = act_system.start_emotion_journey(current_user["user_id"])
        
        if not journey_data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to start therapy session"
            )
        
        return SessionResponse(
            session_id=journey_data["session_id"],
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
    db: AsyncIOMotorDatabase = Depends(get_database),
    act_system = Depends(get_act_therapy_system),
    _: Any = Depends(diary_rate_limiter)
):
    """Submit diary entry for emotion analysis (Stage 1: The Moment)"""
    try:
        # Verify session ownership and stage
        session = await db[Collections.GALLERY_ITEMS].find_one({
            "session_id": session_id,
            "user_id": current_user["user_id"]
        })
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        if session["journey_stage"] != JourneyStage.THE_MOMENT:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid stage. Current stage: {session['journey_stage']}"
            )
        
        # Process diary entry
        result = act_system.process_diary_entry(
            user_id=current_user["user_id"],
            session_id=session_id,
            diary_text=diary.diary_text
        )
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process diary entry"
            )
        
        return DiaryAnalysisResponse(
            session_id=session_id,
            emotion_analysis=EmotionAnalysis(
                keywords=result["keywords"],
                vad_scores=result["vad_scores"],
                primary_emotion=result["primary_emotion"],
                intensity=result["intensity"]
            ),
            next_stage=JourneyStage.REFLECTION
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
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
    act_system = Depends(get_act_therapy_system),
    _: Any = Depends(image_rate_limiter)
):
    """Generate reflection image (Stage 2: Reflection)"""
    try:
        # Verify session ownership and stage
        session = await db[Collections.GALLERY_ITEMS].find_one({
            "session_id": session_id,
            "user_id": current_user["user_id"]
        })
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        if session["journey_stage"] != JourneyStage.REFLECTION:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid stage. Current stage: {session['journey_stage']}"
            )
        
        # Generate image
        result = act_system.generate_reflection_image(
            user_id=current_user["user_id"],
            session_id=session_id
        )
        
        if not result or not result.get("image_path"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate reflection image"
            )
        
        # Convert file path to URL
        image_filename = os.path.basename(result["image_path"])
        image_url = f"/therapy/images/{image_filename}"
        
        return ImageGenerationResponse(
            session_id=session_id,
            image_url=image_url,
            prompt_used=result["prompt"],
            generation_time=result.get("generation_time", 30.0),
            next_stage=JourneyStage.DEFUSION
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating reflection image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate reflection image"
        )


@router.post("/sessions/{session_id}/guestbook", response_model=GuestbookResponse)
async def create_guestbook_entry(
    session_id: str,
    guestbook: GuestbookRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
    act_system = Depends(get_act_therapy_system)
):
    """Create guestbook entry (Stage 3: Defusion)"""
    try:
        # Verify session ownership and stage
        session = await db[Collections.GALLERY_ITEMS].find_one({
            "session_id": session_id,
            "user_id": current_user["user_id"]
        })
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        if session["journey_stage"] != JourneyStage.DEFUSION:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid stage. Current stage: {session['journey_stage']}"
            )
        
        # Create guestbook entry
        result = act_system.create_guestbook_entry(
            user_id=current_user["user_id"],
            session_id=session_id,
            title=guestbook.title,
            tags=guestbook.tags
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
    db: AsyncIOMotorDatabase = Depends(get_database),
    act_system = Depends(get_act_therapy_system)
):
    """Generate curator message (Stage 4: Closure)"""
    try:
        # Verify session ownership and stage
        session = await db[Collections.GALLERY_ITEMS].find_one({
            "session_id": session_id,
            "user_id": current_user["user_id"]
        })
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        if session["journey_stage"] != JourneyStage.CLOSURE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid stage. Current stage: {session['journey_stage']}"
            )
        
        # Generate curator message
        result = act_system.generate_curator_message(
            user_id=current_user["user_id"],
            session_id=session_id
        )
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate curator message"
            )
        
        return CuratorMessageResponse(
            session_id=session_id,
            curator_message=CuratorMessage(
                message=result["message"],
                message_type=result.get("message_type", "encouragement"),
                personalization_data=result.get("personalization_data", {})
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
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Get therapy session details"""
    try:
        session = await db[Collections.GALLERY_ITEMS].find_one({
            "session_id": session_id,
            "user_id": current_user["user_id"]
        })
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        # Build response
        response = TherapySessionDetailResponse(
            session_id=session["session_id"],
            user_id=session["user_id"],
            created_date=session["created_date"],
            journey_stage=session["journey_stage"],
            is_completed=session["is_completed"],
            diary_text=session.get("diary_text"),
            emotion_analysis=EmotionAnalysis(**session["emotion_analysis"]) if session.get("emotion_analysis") else None,
            generated_image=GeneratedImage(**session["generated_image"]) if session.get("generated_image") else None,
            guestbook_entry=GuestbookEntry(**session["guestbook_entry"]) if session.get("guestbook_entry") else None,
            curator_message=CuratorMessage(**session["curator_message"]) if session.get("curator_message") else None
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