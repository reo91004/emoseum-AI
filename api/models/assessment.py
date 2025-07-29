# api/models/assessment.py

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
from .common import BaseResponse, BaseTimestampModel, CopingStyle, SeverityLevel


class PsychometricTestRequest(BaseModel):
    """심리검사 요청 모델"""

    # PHQ-9 (Patient Health Questionnaire-9) - 우울증 선별
    phq9_answers: List[int] = Field(
        ...,
        min_items=9,
        max_items=9,
        description="PHQ-9 answers (9 items, 0-3 scale each)",
    )

    # CES-D (Center for Epidemiologic Studies Depression Scale) - 우울 증상
    cesd_answers: List[int] = Field(
        ...,
        min_items=20,
        max_items=20,
        description="CES-D answers (20 items, 0-3 scale each)",
    )

    # MEAQ (Multidimensional Emotional Avoidance Questionnaire) - 감정 회피
    meaq_answers: List[int] = Field(
        ...,
        min_items=25,
        max_items=25,
        description="MEAQ answers (25 items, 1-5 scale each)",
    )

    # CISS (Coping Inventory for Stressful Situations) - 대처 스타일
    ciss_answers: List[int] = Field(
        ...,
        min_items=48,
        max_items=48,
        description="CISS answers (48 items, 1-5 scale each)",
    )

    @validator("phq9_answers")
    def validate_phq9(cls, v):
        if not all(0 <= answer <= 3 for answer in v):
            raise ValueError("PHQ-9 answers must be between 0 and 3")
        return v

    @validator("cesd_answers")
    def validate_cesd(cls, v):
        if not all(0 <= answer <= 3 for answer in v):
            raise ValueError("CES-D answers must be between 0 and 3")
        return v

    @validator("meaq_answers")
    def validate_meaq(cls, v):
        if not all(1 <= answer <= 5 for answer in v):
            raise ValueError("MEAQ answers must be between 1 and 5")
        return v

    @validator("ciss_answers")
    def validate_ciss(cls, v):
        if not all(1 <= answer <= 5 for answer in v):
            raise ValueError("CISS answers must be between 1 and 5")
        return v


class PsychometricResult(BaseTimestampModel):
    """심리검사 결과 모델"""

    user_id: str = Field(..., description="User identifier")

    # 점수들
    phq9_score: int = Field(..., ge=0, le=27, description="PHQ-9 total score (0-27)")
    cesd_score: int = Field(..., ge=0, le=60, description="CES-D total score (0-60)")
    meaq_score: int = Field(..., ge=25, le=125, description="MEAQ total score (25-125)")
    ciss_score: int = Field(..., ge=48, le=240, description="CISS total score (48-240)")

    # 해석 결과
    coping_style: CopingStyle = Field(..., description="Determined coping style")
    severity_level: SeverityLevel = Field(..., description="Depression severity level")

    # 상세 분석
    interpretation: Dict[str, Any] = Field(
        ..., description="Detailed test interpretation"
    )
    recommendations: List[str] = Field(..., description="Personalized recommendations")

    # CISS 하위 척도
    ciss_task_focused: float = Field(..., description="Task-focused coping score")
    ciss_emotion_focused: float = Field(..., description="Emotion-focused coping score")
    ciss_avoidance_focused: float = Field(
        ..., description="Avoidance-focused coping score"
    )

    # MEAQ 하위 요인
    meaq_behavioral_avoidance: float = Field(
        ..., description="Behavioral avoidance score"
    )
    meaq_distress_aversion: float = Field(..., description="Distress aversion score")
    meaq_procrastination: float = Field(..., description="Procrastination score")
    meaq_distraction: float = Field(..., description="Distraction score")
    meaq_repression: float = Field(..., description="Repression score")
    meaq_denial: float = Field(..., description="Denial score")


class PsychometricResponse(BaseResponse):
    """심리검사 응답 모델"""

    result: PsychometricResult = Field(..., description="Assessment results")
    risk_level: str = Field(..., description="Overall risk level assessment")
    next_steps: List[str] = Field(..., description="Recommended next steps")
    referral_needed: bool = Field(
        ..., description="Whether professional referral is needed"
    )


class VisualPreferencesRequest(BaseModel):
    """시각 선호도 설정 요청 모델"""

    # 기본 스타일 선호도
    preferred_art_styles: List[str] = Field(
        ...,
        min_items=1,
        description="Preferred art styles (painting, photography, abstract, etc.)",
    )

    # 색상 선호도
    color_preferences: Dict[str, float] = Field(
        ..., description="Color preference weights (warm, cool, pastel, etc.)"
    )

    # 복잡도 선호도
    complexity_level: int = Field(
        ..., ge=1, le=5, description="Preferred complexity level (1=simple, 5=complex)"
    )

    # 밝기/채도 선호도
    brightness_preference: float = Field(
        ..., ge=0.0, le=1.0, description="Brightness preference (0.0=dark, 1.0=bright)"
    )
    saturation_preference: float = Field(
        ..., ge=0.0, le=1.0, description="Saturation preference (0.0=muted, 1.0=vivid)"
    )

    # 테마 선호도
    preferred_themes: List[str] = Field(
        default_factory=list,
        description="Preferred visual themes (nature, urban, abstract, etc.)",
    )

    # 회피할 요소들
    avoided_elements: List[str] = Field(
        default_factory=list,
        description="Visual elements to avoid (darkness, crowds, etc.)",
    )

    @validator("color_preferences")
    def validate_color_preferences(cls, v):
        total_weight = sum(v.values())
        if not 0.8 <= total_weight <= 1.2:  # Allow some tolerance
            raise ValueError("Color preference weights should sum to approximately 1.0")
        return v


class VisualPreferences(BaseTimestampModel):
    """시각 선호도 모델"""

    user_id: str = Field(..., description="User identifier")

    # 스타일 선호도 (가중치)
    style_preferences: Dict[str, float] = Field(
        ..., description="Style preference weights"
    )
    color_preferences: Dict[str, float] = Field(
        ..., description="Color preference weights"
    )

    # 설정값들
    complexity_level: int = Field(..., ge=1, le=5, description="Complexity preference")
    brightness_preference: float = Field(
        ..., ge=0.0, le=1.0, description="Brightness preference"
    )
    saturation_preference: float = Field(
        ..., ge=0.0, le=1.0, description="Saturation preference"
    )

    # 테마 및 회피 요소
    preferred_themes: List[str] = Field(
        default_factory=list, description="Preferred themes"
    )
    avoided_elements: List[str] = Field(
        default_factory=list, description="Avoided elements"
    )

    # 학습된 선호도 (Level 2 개인화)
    learned_preferences: Optional[Dict[str, float]] = Field(
        None, description="AI-learned preferences"
    )
    learning_confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Learning confidence score"
    )


class VisualPreferencesResponse(BaseResponse):
    """시각 선호도 응답 모델"""

    preferences: VisualPreferences = Field(..., description="Visual preferences")
    personalization_level: int = Field(..., description="Current personalization level")
    recommendations: List[str] = Field(..., description="Style recommendations")


class AssessmentSummary(BaseModel):
    """심리검사 요약 모델"""

    user_id: str = Field(..., description="User identifier")
    last_assessment_date: Optional[datetime] = Field(
        None, description="Last assessment date"
    )
    current_coping_style: Optional[CopingStyle] = Field(
        None, description="Current coping style"
    )
    current_severity: Optional[SeverityLevel] = Field(
        None, description="Current severity level"
    )

    # 변화 추이
    phq9_trend: List[Dict[str, Any]] = Field(
        default_factory=list, description="PHQ-9 score trend"
    )
    cesd_trend: List[Dict[str, Any]] = Field(
        default_factory=list, description="CES-D score trend"
    )

    # 재검사 필요성
    reassessment_needed: bool = Field(..., description="Whether reassessment is needed")
    reassessment_reason: Optional[str] = Field(
        None, description="Reason for reassessment"
    )
    days_since_last: Optional[int] = Field(
        None, description="Days since last assessment"
    )


class ProgressTracker(BaseModel):
    """진행 상황 추적 모델"""

    user_id: str = Field(..., description="User identifier")
    baseline_scores: Dict[str, int] = Field(
        ..., description="Initial assessment scores"
    )
    current_scores: Optional[Dict[str, int]] = Field(
        None, description="Most recent scores"
    )

    # 개선 지표
    improvement_percentage: Optional[float] = Field(
        None, description="Overall improvement percentage"
    )
    areas_of_improvement: List[str] = Field(
        default_factory=list, description="Improved areas"
    )
    areas_needing_attention: List[str] = Field(
        default_factory=list, description="Areas needing attention"
    )

    # 목표 설정
    target_scores: Optional[Dict[str, int]] = Field(None, description="Target scores")
    estimated_timeline: Optional[int] = Field(
        None, description="Estimated days to reach targets"
    )


class PersonalizationMetrics(BaseModel):
    """개인화 메트릭 모델"""

    user_id: str = Field(..., description="User identifier")
    current_level: int = Field(
        ..., ge=1, le=3, description="Current personalization level"
    )

    # Level 1 (기본)
    profile_completeness: float = Field(
        ..., ge=0.0, le=1.0, description="Profile completion rate"
    )

    # Level 2 (GPT 학습)
    interaction_count: int = Field(0, ge=0, description="Total interactions")
    positive_feedback_rate: float = Field(
        0.0, ge=0.0, le=1.0, description="Positive feedback rate"
    )

    # Level 3 (고급 AI)
    lora_training_eligible: bool = Field(
        False, description="Eligible for LoRA training"
    )
    draft_training_eligible: bool = Field(
        False, description="Eligible for DRaFT+ training"
    )

    # 업그레이드 조건
    level_2_requirements: Dict[str, bool] = Field(
        ..., description="Level 2 upgrade requirements"
    )
    level_3_requirements: Dict[str, bool] = Field(
        ..., description="Level 3 upgrade requirements"
    )

    next_milestone: Optional[str] = Field(
        None, description="Next personalization milestone"
    )
