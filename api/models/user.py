# api/models/user.py

from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, Field, validator
from enum import Enum


class CopingStyle(str, Enum):
    TASK_ORIENTED = "task_oriented"
    EMOTION_ORIENTED = "emotion_oriented"
    AVOIDANCE_ORIENTED = "avoidance_oriented"


class SeverityLevel(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class PsychometricResults(BaseModel):
    phq9_score: int = Field(..., ge=0, le=27, description="PHQ-9 score (0-27)")
    cesd_score: int = Field(..., ge=0, le=60, description="CES-D score (0-60)")
    meaq_score: int = Field(..., ge=0, le=62, description="MEAQ score (0-62)")
    ciss_score: int = Field(..., ge=0, le=96, description="CISS score (0-96)")
    coping_style: CopingStyle
    severity_level: SeverityLevel
    assessment_date: datetime


class VisualPreferences(BaseModel):
    preferred_styles: List[str] = Field(default_factory=list)
    color_preferences: List[str] = Field(default_factory=list)
    complexity_level: str = Field(default="medium")
    art_movements: List[str] = Field(default_factory=list)


class UserSettings(BaseModel):
    language: str = Field(default="en")
    notifications: bool = Field(default=True)


# Request models
class UserRegistrationRequest(BaseModel):
    user_id: str = Field(..., min_length=3, max_length=50)
    
    @validator('user_id')
    def validate_user_id(cls, v):
        if not v.isalnum() and '_' not in v:
            raise ValueError('User ID must be alphanumeric or contain underscores')
        return v


class PsychometricAssessmentRequest(BaseModel):
    phq9_score: int = Field(..., ge=0, le=27)
    cesd_score: int = Field(..., ge=0, le=60)
    meaq_score: int = Field(..., ge=0, le=62)
    ciss_score: int = Field(..., ge=0, le=96)


class UpdateVisualPreferencesRequest(BaseModel):
    preferred_styles: Optional[List[str]] = None
    color_preferences: Optional[List[str]] = None
    complexity_level: Optional[str] = None
    art_movements: Optional[List[str]] = None


class UpdateUserSettingsRequest(BaseModel):
    language: Optional[str] = None
    notifications: Optional[bool] = None


# Response models
class UserProfileResponse(BaseModel):
    user_id: str
    created_date: datetime
    psychometric_results: Optional[PsychometricResults] = None
    visual_preferences: VisualPreferences
    personalization_level: int = Field(default=1, ge=1, le=3)
    settings: UserSettings


class UserStatusResponse(BaseModel):
    user_id: str
    is_active: bool
    last_activity: Optional[datetime] = None
    completed_journeys: int = 0
    current_session_id: Optional[str] = None
    personalization_level: int = Field(default=1, ge=1, le=3)


class PsychometricResultResponse(BaseModel):
    coping_style: CopingStyle
    severity_level: SeverityLevel
    recommendation: str
    assessment_date: datetime