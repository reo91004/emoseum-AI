# api/models/common.py

from typing import Any, Dict, List, Optional, Generic, TypeVar
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum

T = TypeVar("T")


class BaseResponse(BaseModel):
    """기본 응답 모델"""

    model_config = ConfigDict(
        from_attributes=True, validate_assignment=True, arbitrary_types_allowed=True
    )


class APIResponse(BaseResponse, Generic[T]):
    """표준 API 응답 모델"""

    success: bool = Field(..., description="Request success status")
    message: str = Field(..., description="Response message")
    data: Optional[T] = Field(None, description="Response data")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )
    request_id: Optional[str] = Field(None, description="Request tracking ID")


class ErrorResponse(BaseResponse):
    """에러 응답 모델"""

    success: bool = Field(False, description="Always false for errors")
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="User-friendly error message")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = Field(None, description="Request tracking ID")


class SuccessResponse(BaseResponse, Generic[T]):
    """성공 응답 모델"""

    success: bool = Field(True, description="Always true for success")
    message: str = Field(
        "Operation completed successfully", description="Success message"
    )
    data: T = Field(..., description="Response data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PaginationParams(BaseModel):
    """페이지네이션 파라미터"""

    page: int = Field(1, ge=1, description="Page number (starts from 1)")
    limit: int = Field(20, ge=1, le=100, description="Items per page (max 100)")
    sort_by: Optional[str] = Field(None, description="Sort field")
    sort_order: Optional[str] = Field(
        "desc", pattern="^(asc|desc)$", description="Sort order"
    )


class PaginatedResponse(BaseResponse, Generic[T]):
    """페이지네이션 응답 모델"""

    items: List[T] = Field(..., description="Paginated items")
    total: int = Field(..., ge=0, description="Total number of items")
    page: int = Field(..., ge=1, description="Current page number")
    limit: int = Field(..., ge=1, description="Items per page")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_prev: bool = Field(..., description="Whether there is a previous page")


class CopingStyle(str, Enum):
    """대처 스타일 열거형"""

    AVOIDANT = "avoidant"
    CONFRONTATIONAL = "confrontational"
    BALANCED = "balanced"


class SeverityLevel(str, Enum):
    """심각도 수준 열거형"""

    MINIMAL = "minimal"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class CompletionStatus(str, Enum):
    """완료 상태 열거형"""

    STARTED = "started"
    REFLECTING = "reflecting"
    DEFUSING = "defusing"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class ServiceType(str, Enum):
    """서비스 타입 열거형 (비용 추적용)"""

    GPT_PROMPT = "gpt_prompt"
    GPT_CURATOR = "gpt_curator"
    IMAGE_GENERATION = "image_generation"
    EMOTION_ANALYSIS = "emotion_analysis"


class ImageBackend(str, Enum):
    """이미지 생성 백엔드 열거형"""

    LOCAL = "local"
    REMOTE = "remote"
    COLAB = "colab"


class BaseTimestampModel(BaseResponse):
    """타임스탬프가 포함된 기본 모델"""

    created_date: datetime = Field(default_factory=datetime.utcnow)
    updated_date: Optional[datetime] = Field(None)


class FileUpload(BaseModel):
    """파일 업로드 모델"""

    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME type")
    size: int = Field(..., ge=0, description="File size in bytes")
    url: Optional[str] = Field(None, description="Generated file URL")


class CostSummary(BaseModel):
    """비용 요약 모델"""

    user_id: Optional[str] = Field(None, description="User ID (None for total)")
    total_cost: float = Field(..., ge=0, description="Total cost in USD")
    total_tokens: int = Field(..., ge=0, description="Total tokens used")
    service_breakdown: Dict[str, float] = Field(
        ..., description="Cost breakdown by service"
    )
    period_days: int = Field(..., ge=1, description="Analysis period in days")


class SystemStatus(BaseModel):
    """시스템 상태 모델"""

    status: str = Field(..., description="Overall system status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Environment (dev/prod)")
    services: Dict[str, str] = Field(..., description="Individual service statuses")
    configuration: Dict[str, Any] = Field(..., description="System configuration")


class HealthCheck(BaseModel):
    """헬스 체크 모델"""

    status: str = Field("healthy", description="Health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    uptime: Optional[float] = Field(None, description="Server uptime in seconds")
    dependencies: Dict[str, bool] = Field(..., description="Dependency health status")
