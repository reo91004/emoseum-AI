# api/models/__init__.py

"""
Pydantic Models Package

이 패키지는 FastAPI에서 사용되는 모든 Pydantic 모델들을 정의한다.
- common.py: 공통 응답 및 기본 모델
- user.py: 사용자 관련 모델 (회원가입, 로그인, 프로필)
- assessment.py: 심리검사 관련 모델 (PHQ-9, CES-D, MEAQ, CISS)
- therapy.py: 치료 여정 관련 모델 (일기, 이미지, 방명록, 큐레이터)
"""

from .common import *
from .user import *
from .assessment import *
from .therapy import *

__all__ = [
    # Common models
    "APIResponse",
    "ErrorResponse",
    "SuccessResponse",
    "PaginationParams",
    "PaginatedResponse",
    "BaseResponse",
    "BaseTimestampModel",
    "CopingStyle",
    "SeverityLevel",
    "CompletionStatus",
    "ServiceType",
    "ImageBackend",
    "CostSummary",
    "SystemStatus",
    "HealthCheck",
    # User models
    "UserRegisterRequest",
    "UserLoginRequest",
    "TokenResponse",
    "UserProfile",
    "UserResponse",
    "UserUpdateRequest",
    "UserStats",
    "UserPreferences",
    "UserActivity",
    "PasswordResetRequest",
    "PasswordResetConfirm",
    "UserDeleteRequest",
    # Assessment models
    "PsychometricTestRequest",
    "PsychometricResult",
    "PsychometricResponse",
    "VisualPreferencesRequest",
    "VisualPreferences",
    "VisualPreferencesResponse",
    "AssessmentSummary",
    "ProgressTracker",
    "PersonalizationMetrics",
    # Therapy models
    "VADScores",
    "EmotionAnalysis",
    "JourneyStartRequest",
    "JourneyResponse",
    "ReflectionRequest",
    "ImageGenerationResult",
    "ReflectionResponse",
    "DefusionRequest",
    "DefusionResponse",
    "CuratorMessage",
    "ClosureResponse",
    "GalleryItem",
    "GalleryListResponse",
    "TherapyAnalytics",
    "EmergencyResponse",
    "BatchJourneyRequest",
    "JourneyExport",
]
