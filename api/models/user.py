# api/models/user.py

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, EmailStr, validator
from .common import BaseResponse, BaseTimestampModel, CopingStyle, SeverityLevel


class UserRegisterRequest(BaseModel):
    """사용자 회원가입 요청 모델"""

    user_id: str = Field(
        ...,
        min_length=3,
        max_length=50,
        pattern="^[a-zA-Z0-9_-]+$",
        description="Unique user identifier (alphanumeric, underscore, hyphen only)",
    )
    email: Optional[EmailStr] = Field(None, description="User email address")
    password: str = Field(
        ...,
        min_length=6,
        max_length=100,
        description="User password (minimum 6 characters)",
    )

    @validator("user_id")
    def validate_user_id(cls, v):
        if v.lower() in ["admin", "system", "api", "test"]:
            raise ValueError("Reserved user ID not allowed")
        return v.lower()


class UserLoginRequest(BaseModel):
    """사용자 로그인 요청 모델"""

    user_id: str = Field(..., description="User identifier")
    password: str = Field(..., description="User password")


class TokenResponse(BaseResponse):
    """JWT 토큰 응답 모델"""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    user_id: str = Field(..., description="Associated user ID")


class UserProfile(BaseTimestampModel):
    """사용자 프로필 모델"""

    user_id: str = Field(..., description="Unique user identifier")
    email: Optional[EmailStr] = Field(None, description="User email")

    # 심리검사 관련
    current_coping_style: Optional[CopingStyle] = Field(
        None, description="Current coping style"
    )
    current_severity_level: Optional[SeverityLevel] = Field(
        None, description="Current severity level"
    )
    last_assessment_date: Optional[datetime] = Field(
        None, description="Last psychometric assessment date"
    )

    # 사용 통계
    total_journeys: int = Field(0, ge=0, description="Total therapy journeys completed")
    completed_journeys: int = Field(
        0, ge=0, description="Successfully completed journeys"
    )
    total_images_generated: int = Field(0, ge=0, description="Total images generated")
    last_activity_date: Optional[datetime] = Field(
        None, description="Last activity timestamp"
    )

    # 개인화 레벨
    personalization_level: int = Field(
        1, ge=1, le=3, description="Current personalization level (1-3)"
    )

    @validator("completed_journeys")
    def validate_completed_journeys(cls, v, values):
        if "total_journeys" in values and v > values["total_journeys"]:
            raise ValueError("Completed journeys cannot exceed total journeys")
        return v


class UserResponse(BaseResponse):
    """사용자 정보 응답 모델"""

    user_id: str = Field(..., description="User identifier")
    email: Optional[str] = Field(None, description="User email")
    created_date: datetime = Field(..., description="Account creation date")
    profile: UserProfile = Field(..., description="User profile information")


class UserUpdateRequest(BaseModel):
    """사용자 정보 수정 요청 모델"""

    email: Optional[EmailStr] = Field(None, description="New email address")
    password: Optional[str] = Field(
        None,
        min_length=6,
        max_length=100,
        description="New password (minimum 6 characters)",
    )


class UserStats(BaseModel):
    """사용자 통계 모델"""

    user_id: str = Field(..., description="User identifier")

    # 활동 통계
    total_journeys: int = Field(..., ge=0, description="Total therapy journeys")
    completed_journeys: int = Field(..., ge=0, description="Completed journeys")
    completion_rate: float = Field(
        ..., ge=0, le=1, description="Journey completion rate"
    )

    # 감정 분석 통계
    dominant_emotions: List[str] = Field(..., description="Most frequent emotions")
    emotion_trend: Dict[str, float] = Field(..., description="Emotion trend over time")
    vad_average: Dict[str, float] = Field(..., description="Average VAD scores")

    # 이미지 생성 통계
    total_images: int = Field(..., ge=0, description="Total images generated")
    preferred_styles: List[str] = Field(..., description="Preferred art styles")

    # 비용 통계
    total_cost: float = Field(..., ge=0, description="Total API costs incurred")
    cost_breakdown: Dict[str, float] = Field(
        ..., description="Cost breakdown by service"
    )

    # 시간 통계
    first_journey_date: Optional[datetime] = Field(
        None, description="First journey date"
    )
    last_journey_date: Optional[datetime] = Field(None, description="Last journey date")
    average_session_duration: Optional[float] = Field(
        None, description="Average session duration in minutes"
    )


class UserPreferences(BaseModel):
    """사용자 기본 설정 모델"""

    language: str = Field("en", description="Preferred language")
    timezone: str = Field("UTC", description="User timezone")
    notifications_enabled: bool = Field(True, description="Email notifications enabled")
    data_retention_days: int = Field(
        365, ge=30, le=1095, description="Data retention period in days"
    )

    # 치료 관련 설정
    journey_reminders: bool = Field(True, description="Journey completion reminders")
    daily_check_in: bool = Field(False, description="Daily emotional check-in")

    # 개인정보 설정
    analytics_opt_in: bool = Field(True, description="Opt-in for anonymized analytics")
    research_participation: bool = Field(
        False, description="Participate in research studies"
    )


class UserActivity(BaseTimestampModel):
    """사용자 활동 로그 모델"""

    user_id: str = Field(..., description="User identifier")
    activity_type: str = Field(..., description="Type of activity")
    activity_details: Dict[str, Any] = Field(..., description="Activity details")
    ip_address: Optional[str] = Field(None, description="User IP address")
    user_agent: Optional[str] = Field(None, description="User agent string")
    session_id: Optional[str] = Field(None, description="Session identifier")


class PasswordResetRequest(BaseModel):
    """비밀번호 재설정 요청 모델"""

    email: EmailStr = Field(..., description="User email address")


class PasswordResetConfirm(BaseModel):
    """비밀번호 재설정 확인 모델"""

    token: str = Field(..., description="Password reset token")
    new_password: str = Field(
        ..., min_length=6, max_length=100, description="New password"
    )


class UserDeleteRequest(BaseModel):
    """사용자 계정 삭제 요청 모델"""

    password: str = Field(..., description="Current password for verification")
    confirmation: str = Field(..., description="Deletion confirmation phrase")

    @validator("confirmation")
    def validate_confirmation(cls, v):
        if v.lower() != "delete my account":
            raise ValueError('Must type "delete my account" to confirm')
        return v
