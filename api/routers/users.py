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
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get current user profile"""
    try:
        # Data is already available from current_user (populated by ACT system)
        # Convert to response model
        profile = UserProfileResponse(
            user_id=current_user["user_id"],
            created_date=current_user["created_date"],
            psychometric_results=PsychometricResults(**current_user["psychometric_results"][-1]) if current_user.get("psychometric_results") else None,
            visual_preferences=VisualPreferences(**current_user.get("visual_preferences", {})),
            personalization_level=1,  # Default personalization level
            settings=UserSettings(language="ko", notifications=True)  # Default settings
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
    """Update user profile settings"""
    try:
        # For now, just return the current profile since settings are handled separately
        # In a full implementation, you would update user settings through ACT system
        logger.info(f"Profile update requested for user {current_user['user_id']}: {settings}")
        
        # Return current profile
        return await get_user_profile(current_user)
        
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
    act_system = Depends(get_act_therapy_system)
):
    """Update user's visual preferences"""
    try:
        # Convert API model to ACT system format
        visual_prefs = {}
        if preferences.preferred_styles is not None:
            # Map API styles to ACT system styles
            if "painting" in preferences.preferred_styles:
                visual_prefs["art_style"] = "painting"
            elif "photography" in preferences.preferred_styles:
                visual_prefs["art_style"] = "photography"
            elif "abstract" in preferences.preferred_styles:
                visual_prefs["art_style"] = "abstract"
            else:
                visual_prefs["art_style"] = "painting"
        
        if preferences.color_preferences is not None:
            # Map color preferences
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
        
        # Update through ACT system
        if visual_prefs:
            act_system.user_manager.set_visual_preferences(current_user["user_id"], visual_prefs)
        
        # Return updated profile
        return await get_user_profile(current_user)
        
    except Exception as e:
        logger.error(f"Error updating visual preferences: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update visual preferences"
        )


@router.get("/status", response_model=UserStatusResponse)
async def get_user_status(
    current_user: Dict[str, Any] = Depends(get_current_user),
    act_system = Depends(get_act_therapy_system)
):
    """Get user status and activity summary"""
    try:
        # Get user stats through ACT system
        user_stats = act_system.user_manager.get_user_stats(current_user["user_id"])
        
        # Get gallery data through ACT system
        gallery_data = act_system.get_user_gallery(current_user["user_id"], limit=1)
        
        # Get incomplete journeys
        incomplete_journeys = act_system.gallery_manager.get_incomplete_journeys(current_user["user_id"])
        
        return UserStatusResponse(
            user_id=current_user["user_id"],
            is_active=True,
            last_activity=gallery_data.get("items", [{}])[0].get("created_date") if gallery_data.get("items") else None,
            completed_journeys=gallery_data.get("total_items", 0),
            current_session_id=incomplete_journeys[0].item_id if incomplete_journeys else None,
            personalization_level=1  # Default personalization level
        )
        
    except Exception as e:
        logger.error(f"Error getting user status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user status"
        )