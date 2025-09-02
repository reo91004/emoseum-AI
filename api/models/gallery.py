# api/models/gallery.py

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from .therapy import EmotionAnalysis, GeneratedImage, ArtworkTitle, DocentMessage, JourneyStage


class GalleryItemSummary(BaseModel):
    item_id: str
    session_id: str
    created_date: datetime
    thumbnail_url: Optional[str] = None
    primary_emotion: str
    is_completed: bool


class GalleryItemDetail(BaseModel):
    item_id: str
    session_id: str
    user_id: str
    created_date: datetime
    diary_text: str
    emotion_analysis: EmotionAnalysis
    generated_image: GeneratedImage
    artwork_title: Optional[ArtworkTitle] = None
    docent_message: Optional[DocentMessage] = None
    journey_stage: JourneyStage
    is_completed: bool


class EmotionTrend(BaseModel):
    date: datetime
    valence: float
    arousal: float
    dominance: float
    primary_emotion: str


class GalleryAnalytics(BaseModel):
    total_items: int
    completed_journeys: int
    emotion_trends: List[EmotionTrend]
    most_common_emotions: Dict[str, int]
    average_vad_scores: Dict[str, float]


class GalleryExportData(BaseModel):
    user_id: str
    export_date: datetime
    items: List[GalleryItemDetail]
    analytics: GalleryAnalytics


# Request models
class GalleryFilterRequest(BaseModel):
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    emotions: Optional[List[str]] = None
    completed_only: bool = False
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


# Response models
class GalleryListResponse(BaseModel):
    items: List[GalleryItemSummary]
    total_count: int
    has_more: bool


class GalleryItemResponse(BaseModel):
    item: GalleryItemDetail


class GalleryAnalyticsResponse(BaseModel):
    analytics: GalleryAnalytics
    period_start: datetime
    period_end: datetime


class GalleryExportResponse(BaseModel):
    export_url: str
    expires_at: datetime
    file_format: str = "json"