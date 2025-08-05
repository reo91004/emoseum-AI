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


class ArtworkTitle(BaseModel):
    title: str
    description: Optional[str] = ""
    reflection: str


class DocentMessage(BaseModel):
    message: str
    message_type: str
    personalization_data: Dict[str, Any] = Field(default_factory=dict)


# Request models
class StartSessionRequest(BaseModel):
    pass


class DiaryEntryRequest(BaseModel):
    diary_text: str = Field(..., min_length=10, max_length=1000)
    diary_id: str = Field(..., description="중앙 서버의 일기 ID")


class DiaryExplorationRequest(BaseModel):
    diary_text: str = Field(..., min_length=10, max_length=1000, description="탐색할 일기 내용")
    emotion_keywords: Optional[List[str]] = Field(None, description="감정 키워드 (선택사항)")


class DiaryFollowUpRequest(BaseModel):
    diary_text: str = Field(..., min_length=10, max_length=1000, description="원본 일기 내용")
    previous_question: str = Field(..., min_length=5, max_length=500, description="이전 질문")
    user_response: str = Field(..., min_length=1, max_length=2000, description="사용자의 답변")
    emotion_keywords: Optional[List[str]] = Field(None, description="감정 키워드 (선택사항)")


class ArtworkTitleRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)
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


class ArtworkTitleResponse(BaseModel):
    session_id: str
    artwork_title: ArtworkTitle
    next_stage: JourneyStage


class DocentMessageResponse(BaseModel):
    session_id: str
    docent_message: DocentMessage
    journey_completed: bool = True


class ExplorationQuestion(BaseModel):
    question: str
    category: str
    explanation: str


class DiaryExplorationResponse(BaseModel):
    success: bool
    questions: List[ExplorationQuestion]
    exploration_theme: str
    encouragement: str
    emotion_analysis: Optional[EmotionAnalysis] = None
    generation_timestamp: Optional[str] = None


class TherapySessionDetailResponse(BaseModel):
    session_id: str
    user_id: str
    created_date: datetime
    journey_stage: JourneyStage
    is_completed: bool
    diary_text: Optional[str] = None
    emotion_analysis: Optional[EmotionAnalysis] = None
    generated_image: Optional[GeneratedImage] = None
    artwork_title: Optional[ArtworkTitle] = None
    docent_message: Optional[DocentMessage] = None