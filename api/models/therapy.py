# api/models/therapy.py

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class JourneyStage(str, Enum):
    THE_MOMENT = "the_moment"
    REFLECTION = "reflection"
    DEFUSION = "defusion"
    CLOSURE = "closure"


class EmotionAnalysis(BaseModel):
    keywords: List[str]
    vad_scores: List[float] = Field(..., min_items=3, max_items=3)
    primary_emotion: str
    intensity: float = Field(..., ge=0.0, le=1.0)


class ImageGenerationMetadata(BaseModel):
    service_used: str = Field(default="local_gpu")
    generation_time: float
    model_version: str = Field(default="stable-diffusion-v1-5")


class GeneratedImage(BaseModel):
    image_path: str
    prompt_used: str
    generation_metadata: ImageGenerationMetadata


class GuestbookEntry(BaseModel):
    title: str
    tags: List[str]
    reflection: str


class CuratorMessage(BaseModel):
    message: str
    message_type: str
    personalization_data: Dict[str, Any] = Field(default_factory=dict)


# Request models
class StartSessionRequest(BaseModel):
    user_id: str


class DiaryEntryRequest(BaseModel):
    diary_text: str = Field(..., min_length=10, max_length=1000)


class GuestbookRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)
    tags: List[str] = Field(..., min_items=1, max_items=5)
    reflection: Optional[str] = Field(None, max_length=500)


# Response models
class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    created_date: datetime
    journey_stage: JourneyStage
    is_completed: bool = False


class DiaryAnalysisResponse(BaseModel):
    session_id: str
    emotion_analysis: EmotionAnalysis
    next_stage: JourneyStage


class ImageGenerationResponse(BaseModel):
    session_id: str
    image_url: str
    prompt_used: str
    generation_time: float
    next_stage: JourneyStage


class GuestbookResponse(BaseModel):
    session_id: str
    guestbook_entry: GuestbookEntry
    next_stage: JourneyStage


class CuratorMessageResponse(BaseModel):
    session_id: str
    curator_message: CuratorMessage
    journey_completed: bool = True


class TherapySessionDetailResponse(BaseModel):
    session_id: str
    user_id: str
    created_date: datetime
    journey_stage: JourneyStage
    is_completed: bool
    diary_text: Optional[str] = None
    emotion_analysis: Optional[EmotionAnalysis] = None
    generated_image: Optional[GeneratedImage] = None
    guestbook_entry: Optional[GuestbookEntry] = None
    curator_message: Optional[CuratorMessage] = None