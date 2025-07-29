# api/routers/users.py

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
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Get current user profile"""
    try:
        user_data = await db[Collections.USERS].find_one({"user_id": current_user["user_id"]})
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Convert MongoDB document to response model
        profile = UserProfileResponse(
            user_id=user_data["user_id"],
            created_date=user_data["created_date"],
            psychometric_results=PsychometricResults(**user_data["psychometric_results"]) if user_data.get("psychometric_results") else None,
            visual_preferences=VisualPreferences(**user_data.get("visual_preferences", {})),
            personalization_level=user_data.get("personalization_level", 1),
            settings=UserSettings(**user_data.get("settings", {}))
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
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Update user profile settings"""
    try:
        update_data = {}
        if settings.language is not None:
            update_data["settings.language"] = settings.language
        if settings.notifications is not None:
            update_data["settings.notifications"] = settings.notifications
            
        if update_data:
            await db[Collections.USERS].update_one(
                {"user_id": current_user["user_id"]},
                {"$set": update_data}
            )
        
        # Return updated profile
        return await get_user_profile(current_user, db)
        
    except Exception as e:
        logger.error(f"Error updating user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user profile"
        )


@router.post("/assessment", response_model=PsychometricResultResponse)
async def conduct_assessment(
    assessment: PsychometricAssessmentRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
    act_system = Depends(get_act_therapy_system)
):
    """Conduct psychometric assessment"""
    try:
        # Use ACT system to process assessment
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
                detail="Assessment processing failed"
            )
        
        # Get recommendation based on severity
        recommendations = {
            "mild": "Continue with regular ACT therapy sessions and self-care practices.",
            "moderate": "Regular ACT therapy sessions recommended. Consider professional support if symptoms persist.",
            "severe": "Professional mental health consultation strongly recommended alongside ACT therapy."
        }
        
        return PsychometricResultResponse(
            coping_style=result.coping_style,
            severity_level=result.severity_level,
            recommendation=recommendations.get(result.severity_level, ""),
            assessment_date=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Assessment error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Assessment failed"
        )


@router.put("/visual-preferences", response_model=UserProfileResponse)
async def update_visual_preferences(
    preferences: UpdateVisualPreferencesRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Update user's visual preferences"""
    try:
        update_data = {}
        if preferences.preferred_styles is not None:
            update_data["visual_preferences.preferred_styles"] = preferences.preferred_styles
        if preferences.color_preferences is not None:
            update_data["visual_preferences.color_preferences"] = preferences.color_preferences
        if preferences.complexity_level is not None:
            update_data["visual_preferences.complexity_level"] = preferences.complexity_level
        if preferences.art_movements is not None:
            update_data["visual_preferences.art_movements"] = preferences.art_movements
            
        if update_data:
            await db[Collections.USERS].update_one(
                {"user_id": current_user["user_id"]},
                {"$set": update_data}
            )
        
        # Return updated profile
        return await get_user_profile(current_user, db)
        
    except Exception as e:
        logger.error(f"Error updating visual preferences: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update visual preferences"
        )


@router.get("/status", response_model=UserStatusResponse)
async def get_user_status(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Get user status and activity summary"""
    try:
        # Get user data
        user_data = await db[Collections.USERS].find_one({"user_id": current_user["user_id"]})
        
        # Get personalization data
        personalization = await db[Collections.PERSONALIZATION_DATA].find_one(
            {"user_id": current_user["user_id"]}
        )
        
        # Get current active session
        active_session = await db[Collections.GALLERY_ITEMS].find_one(
            {"user_id": current_user["user_id"], "is_completed": False},
            sort=[("created_date", -1)]
        )
        
        # Get last activity
        last_item = await db[Collections.GALLERY_ITEMS].find_one(
            {"user_id": current_user["user_id"]},
            sort=[("created_date", -1)]
        )
        
        return UserStatusResponse(
            user_id=current_user["user_id"],
            is_active=True,
            last_activity=last_item["created_date"] if last_item else None,
            completed_journeys=personalization.get("training_eligibility", {}).get("completed_journeys", 0) if personalization else 0,
            current_session_id=active_session["session_id"] if active_session else None,
            personalization_level=user_data.get("personalization_level", 1)
        )
        
    except Exception as e:
        logger.error(f"Error getting user status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user status"
        )