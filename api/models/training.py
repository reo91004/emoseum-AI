# api/models/training.py

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class TrainingType(str, Enum):
    LORA = "lora"
    DRAFT = "draft"


class TrainingStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TrainingEligibility(BaseModel):
    lora_ready: bool
    draft_ready: bool
    positive_interactions: int
    completed_journeys: int
    eligibility_message: str


class TrainingProgress(BaseModel):
    current_step: int
    total_steps: int
    percentage: float = Field(..., ge=0.0, le=100.0)
    estimated_time_remaining: Optional[int] = None  # seconds


# Request models
class StartTrainingRequest(BaseModel):
    training_type: TrainingType
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)


# Response models
class TrainingEligibilityResponse(BaseModel):
    user_id: str
    eligibility: TrainingEligibility
    recommendation: str


class TrainingStartResponse(BaseModel):
    training_id: str
    training_type: TrainingType
    status: TrainingStatus
    started_at: datetime
    estimated_completion: datetime


class TrainingStatusResponse(BaseModel):
    training_id: str
    training_type: TrainingType
    status: TrainingStatus
    progress: Optional[TrainingProgress] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result_metrics: Optional[Dict[str, Any]] = None