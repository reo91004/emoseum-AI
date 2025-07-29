# api/models/therapy.py

from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
from .common import (
    BaseResponse,
    BaseTimestampModel,
    CompletionStatus,
    ServiceType,
    ImageBackend,
)


class VADScores(BaseModel):
    """VAD (Valence-Arousal-Dominance) 감정 점수 모델"""

    valence: float = Field(
        ..., ge=-1.0, le=1.0, description="Emotional valence (-1=negative, 1=positive)"
    )
    arousal: float = Field(
        ..., ge=-1.0, le=1.0, description="Emotional arousal (-1=calm, 1=excited)"
    )
    dominance: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Emotional dominance (-1=submissive, 1=dominant)",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Analysis confidence score"
    )


class EmotionAnalysis(BaseModel):
    """감정 분석 결과 모델"""

    emotion_keywords: List[str] = Field(..., description="Extracted emotion keywords")
    vad_scores: VADScores = Field(..., description="VAD emotion scores")
    dominant_emotion: str = Field(..., description="Primary detected emotion")
    emotion_intensity: float = Field(
        ..., ge=0.0, le=1.0, description="Overall emotion intensity"
    )

    # 추가 분석
    emotion_categories: Dict[str, float] = Field(
        ..., description="Emotion category weights"
    )
    sentiment_polarity: float = Field(
        ..., ge=-1.0, le=1.0, description="Overall sentiment"
    )
    complexity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Emotional complexity"
    )


class JourneyStartRequest(BaseModel):
    """치료 여정 시작 요청 모델 (The Moment)"""

    diary_text: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Emotional diary entry in English",
    )

    # 선택적 메타데이터
    mood_rating: Optional[int] = Field(
        None, ge=1, le=10, description="Self-reported mood (1-10)"
    )
    trigger_events: Optional[List[str]] = Field(
        None, description="Triggering events or situations"
    )
    physical_symptoms: Optional[List[str]] = Field(
        None, description="Physical symptoms experienced"
    )

    @validator("diary_text")
    def validate_diary_text(cls, v):
        # 기본적인 내용 검증
        if len(v.strip()) < 10:
            raise ValueError("Diary entry must be at least 10 characters long")

        # 금지된 콘텐츠 기본 필터링
        forbidden_words = ["suicide", "kill myself", "end it all", "die"]
        if any(word in v.lower() for word in forbidden_words):
            raise ValueError(
                "Diary contains content requiring immediate professional attention"
            )

        return v.strip()


class JourneyResponse(BaseResponse):
    """치료 여정 응답 모델"""

    journey_id: str = Field(..., description="Unique journey identifier")
    user_id: str = Field(..., description="User identifier")
    status: CompletionStatus = Field(..., description="Current journey status")
    created_date: datetime = Field(..., description="Journey creation timestamp")

    # The Moment 단계 결과
    diary_text: str = Field(..., description="Original diary text")
    emotion_analysis: EmotionAnalysis = Field(
        ..., description="Emotion analysis results"
    )

    # 다음 단계 정보
    next_step: str = Field(..., description="Next step in the journey")
    estimated_completion_time: int = Field(
        ..., description="Estimated time to complete (minutes)"
    )


class ReflectionRequest(BaseModel):
    """성찰 단계 요청 모델 (Reflection)"""

    journey_id: str = Field(..., description="Journey identifier")

    # 이미지 생성 옵션
    image_style_override: Optional[str] = Field(
        None, description="Override default style preference"
    )
    custom_prompt_additions: Optional[str] = Field(
        None, max_length=200, description="Additional prompt elements"
    )

    # 생성 파라미터
    image_count: int = Field(1, ge=1, le=3, description="Number of images to generate")
    use_personalization: bool = Field(True, description="Use personalized prompts")


class ImageGenerationResult(BaseModel):
    """이미지 생성 결과 모델"""

    image_url: str = Field(..., description="Generated image URL")
    image_prompt: str = Field(..., description="Used generation prompt")
    backend_used: ImageBackend = Field(..., description="Backend used for generation")
    generation_time: float = Field(..., ge=0, description="Generation time in seconds")

    # 메타데이터
    model_version: Optional[str] = Field(None, description="Model version used")
    seed_used: Optional[int] = Field(None, description="Random seed used")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Generation parameters"
    )


class ReflectionResponse(BaseResponse):
    """성찰 단계 응답 모델"""

    journey_id: str = Field(..., description="Journey identifier")
    images: List[ImageGenerationResult] = Field(..., description="Generated images")

    # GPT 생성 프롬프트 정보
    prompt_reasoning: str = Field(
        ..., description="GPT's reasoning for prompt creation"
    )
    personalization_applied: List[str] = Field(
        ..., description="Applied personalization factors"
    )

    # 비용 정보
    generation_cost: float = Field(
        ..., ge=0, description="Total generation cost in USD"
    )
    token_usage: Dict[str, int] = Field(..., description="Token usage breakdown")


class DefusionRequest(BaseModel):
    """탈융합 단계 요청 모델 (Defusion)"""

    journey_id: str = Field(..., description="Journey identifier")
    selected_image_url: str = Field(
        ..., description="URL of selected image for guestbook"
    )

    # 방명록 내용
    guestbook_title: str = Field(
        ..., min_length=3, max_length=100, description="Guestbook entry title"
    )
    guestbook_content: str = Field(
        ..., min_length=10, max_length=1000, description="Guestbook entry content"
    )
    guestbook_tags: List[str] = Field(
        ..., min_items=1, max_items=10, description="Guestbook tags for categorization"
    )

    # 선택적 피드백
    image_satisfaction: Optional[int] = Field(
        None, ge=1, le=5, description="Image satisfaction rating"
    )
    emotional_shift: Optional[str] = Field(
        None, description="Perceived emotional shift"
    )

    @validator("guestbook_tags")
    def validate_tags(cls, v):
        # 태그 정리 및 검증
        cleaned_tags = []
        for tag in v:
            cleaned = tag.strip().lower()
            if len(cleaned) >= 2 and cleaned not in cleaned_tags:
                cleaned_tags.append(cleaned)

        if len(cleaned_tags) == 0:
            raise ValueError("At least one valid tag is required")

        return cleaned_tags[:10]  # 최대 10개로 제한


class DefusionResponse(BaseResponse):
    """탈융합 단계 응답 모델"""

    journey_id: str = Field(..., description="Journey identifier")
    guestbook_saved: bool = Field(
        ..., description="Whether guestbook was successfully saved"
    )

    # 탈융합 분석
    cognitive_distance_score: float = Field(
        ..., ge=0.0, le=1.0, description="Cognitive distance achieved"
    )
    reframing_quality: float = Field(
        ..., ge=0.0, le=1.0, description="Reframing quality score"
    )

    # 다음 단계 준비
    closure_ready: bool = Field(..., description="Ready for closure phase")
    estimated_wait_time: int = Field(
        ..., description="Estimated wait time for curator message (minutes)"
    )


class CuratorMessage(BaseModel):
    """큐레이터 메시지 모델"""

    # 5섹션 구조 (ACT 기반)
    opening: str = Field(
        ..., description="Opening section - greeting and acknowledgment"
    )
    recognition: str = Field(
        ..., description="Recognition section - validating emotions"
    )
    personal_note: str = Field(..., description="Personal note - individual insights")
    guidance: str = Field(..., description="Guidance section - therapeutic direction")
    closing: str = Field(..., description="Closing section - hope and encouragement")

    # 메타데이터
    full_message: str = Field(..., description="Complete message text")
    generation_method: str = Field("gpt", description="Message generation method")
    personalization_level: int = Field(
        ..., ge=1, le=3, description="Personalization level used"
    )

    # 품질 메트릭
    therapeutic_appropriateness: float = Field(
        ..., ge=0.0, le=1.0, description="Therapeutic appropriateness score"
    )
    personalization_relevance: float = Field(
        ..., ge=0.0, le=1.0, description="Personalization relevance score"
    )


class ClosureResponse(BaseResponse):
    """완료 단계 응답 모델 (Closure)"""

    journey_id: str = Field(..., description="Journey identifier")
    curator_message: CuratorMessage = Field(
        ..., description="Generated curator message"
    )

    # 여정 완료 통계
    journey_duration: float = Field(
        ..., ge=0, description="Total journey duration in minutes"
    )
    total_cost: float = Field(..., ge=0, description="Total journey cost in USD")

    # 성취 및 진행
    milestones_achieved: List[str] = Field(..., description="Achieved milestones")
    next_journey_suggestion: Optional[str] = Field(
        None, description="Suggestion for next journey"
    )

    # 학습 데이터 (Level 2 개인화용)
    learning_data_collected: bool = Field(
        ..., description="Whether learning data was collected"
    )


class GalleryItem(BaseTimestampModel):
    """갤러리 아이템 모델 (완성된 치료 여정)"""

    id: str = Field(..., description="Gallery item identifier")
    user_id: str = Field(..., description="User identifier")
    journey_id: str = Field(..., description="Associated journey identifier")

    # 4단계 데이터
    diary_text: str = Field(..., description="Original diary text")
    emotion_analysis: EmotionAnalysis = Field(..., description="Emotion analysis")

    image_url: str = Field(..., description="Generated image URL")
    image_prompt: str = Field(..., description="Image generation prompt")

    guestbook_title: str = Field(..., description="Guestbook title")
    guestbook_content: str = Field(..., description="Guestbook content")
    guestbook_tags: List[str] = Field(..., description="Guestbook tags")

    curator_message: CuratorMessage = Field(..., description="Curator message")

    # 메타데이터
    completion_status: CompletionStatus = Field(..., description="Completion status")
    total_duration: float = Field(..., ge=0, description="Total journey duration")
    total_cost: float = Field(..., ge=0, description="Total cost incurred")

    # 사용자 피드백
    user_rating: Optional[int] = Field(None, ge=1, le=5, description="User rating")
    user_feedback: Optional[str] = Field(None, description="User feedback text")
    is_favorite: bool = Field(False, description="Marked as favorite")

    # 분석 데이터
    therapeutic_effectiveness: Optional[float] = Field(
        None, description="Therapeutic effectiveness score"
    )
    emotion_shift_analysis: Optional[Dict[str, float]] = Field(
        None, description="Emotion shift metrics"
    )


class GalleryListResponse(BaseResponse):
    """갤러리 목록 응답 모델"""

    items: List[GalleryItem] = Field(..., description="Gallery items")
    total_count: int = Field(..., ge=0, description="Total number of items")

    # 필터링 정보
    applied_filters: Dict[str, Any] = Field(
        default_factory=dict, description="Applied filters"
    )
    available_tags: List[str] = Field(..., description="All available tags")

    # 통계
    completion_stats: Dict[str, int] = Field(..., description="Completion statistics")
    emotion_distribution: Dict[str, int] = Field(
        ..., description="Emotion distribution"
    )


class TherapyAnalytics(BaseModel):
    """치료 분석 모델"""

    user_id: str = Field(..., description="User identifier")
    analysis_period: int = Field(..., description="Analysis period in days")

    # 감정 변화 분석
    emotion_trend: Dict[str, List[float]] = Field(
        ..., description="Emotion trends over time"
    )
    vad_evolution: List[VADScores] = Field(..., description="VAD score evolution")
    dominant_emotion_changes: List[str] = Field(
        ..., description="Dominant emotion changes"
    )

    # 치료 효과 분석
    therapeutic_progress: float = Field(
        ..., ge=-1.0, le=1.0, description="Overall therapeutic progress"
    )
    breakthrough_moments: List[datetime] = Field(
        ..., description="Identified breakthrough moments"
    )
    regression_periods: List[Dict[str, Any]] = Field(
        ..., description="Identified regression periods"
    )

    # 개인화 효과
    personalization_impact: Dict[str, float] = Field(
        ..., description="Personalization impact metrics"
    )
    preferred_journey_patterns: List[str] = Field(
        ..., description="Preferred journey patterns"
    )

    # 예측 및 권장사항
    next_session_recommendation: str = Field(
        ..., description="Recommendation for next session"
    )
    risk_assessment: Dict[str, float] = Field(..., description="Risk assessment scores")
    intervention_suggestions: List[str] = Field(
        ..., description="Suggested interventions"
    )


class EmergencyResponse(BaseModel):
    """응급 상황 응답 모델"""

    alert_level: str = Field(..., description="Alert level (low/medium/high/critical)")
    risk_factors: List[str] = Field(..., description="Identified risk factors")
    immediate_actions: List[str] = Field(
        ..., description="Immediate recommended actions"
    )

    # 전문가 연결
    referral_recommended: bool = Field(
        ..., description="Professional referral recommended"
    )
    crisis_resources: List[Dict[str, str]] = Field(
        ..., description="Crisis support resources"
    )
    emergency_contacts: List[Dict[str, str]] = Field(
        ..., description="Emergency contact information"
    )

    # 시스템 조치
    journey_suspended: bool = Field(..., description="Whether journey was suspended")
    admin_notified: bool = Field(..., description="Whether admin was notified")
    followup_scheduled: Optional[datetime] = Field(
        None, description="Scheduled followup time"
    )


class BatchJourneyRequest(BaseModel):
    """배치 여정 생성 요청 (관리자용)"""

    user_ids: List[str] = Field(
        ..., min_items=1, max_items=50, description="User IDs for batch processing"
    )
    template_diary: str = Field(..., description="Template diary text")
    processing_options: Dict[str, Any] = Field(
        default_factory=dict, description="Processing options"
    )


class JourneyExport(BaseModel):
    """여정 내보내기 모델"""

    user_id: str = Field(..., description="User identifier")
    export_format: str = Field(
        ..., pattern="^(json|pdf|csv)$", description="Export format"
    )
    date_range: Optional[Dict[str, datetime]] = Field(
        None, description="Date range filter"
    )
    include_images: bool = Field(True, description="Include generated images")
    include_analytics: bool = Field(False, description="Include analytics data")

    # 개인정보 설정
    anonymize_data: bool = Field(False, description="Anonymize personal data")
    include_metadata: bool = Field(True, description="Include technical metadata")
