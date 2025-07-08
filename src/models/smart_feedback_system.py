#!/usr/bin/env python3
"""
스마트 피드백 시스템 - 우울증 치료제를 위한 부담 없는 구체적 피드백
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json

from config import logger
from models.emotion import EmotionEmbedding


@dataclass
class SmartFeedback:
    """구조화된 스마트 피드백"""
    overall_score: float  # 1-5 기본 점수
    
    # 간접 추출 정보
    interaction_time: float  # 이미지 보는 시간 (초)
    click_behavior: Dict[str, Any]  # 클릭/터치 패턴
    emotional_response: Optional[str]  # 자동 감지된 반응
    
    # 선택적 구체적 피드백 (부담 없음)
    quick_tags: List[str]  # 간단한 태그 선택
    mood_change: Optional[str]  # 기분 변화 감지
    comfort_level: Optional[float]  # 편안함 정도
    
    # 시스템 추론 정보
    inferred_preferences: Dict[str, float]  # AI가 추론한 선호도
    contextual_factors: Dict[str, Any]  # 컨텍스트 요인들
    
    timestamp: datetime


class SmartFeedbackCollector:
    """부담 없는 스마트 피드백 수집기"""
    
    def __init__(self):
        # 간단한 감정 태그들 (부담 없는 선택지)
        self.emotion_tags = {
            "positive": ["편안함", "따뜻함", "희망적", "평온함", "안정감"],
            "neutral": ["괜찮음", "보통", "무난함", "적당함"],
            "negative": ["답답함", "불편함", "어둠", "차갑다", "무거움"]
        }
        
        # 시각적 요소 태그들
        self.visual_tags = {
            "colors": ["따뜻한 색", "차가운 색", "밝은 색", "어두운 색", "자연스러운 색"],
            "mood": ["밝은 분위기", "차분한 분위기", "신비로운 분위기", "단순한 분위기"],
            "elements": ["자연", "도시", "사람", "동물", "추상적"]
        }
        
        # 부담 없는 질문 패턴들
        self.gentle_questions = [
            "이 이미지가 어떤 기분을 주나요?",
            "어떤 색감이 눈에 들어오시나요?",
            "이미지에서 가장 인상적인 부분은?",
            "이런 분위기는 어떠세요?",
            "편안함을 느끼시나요?"
        ]
        
        logger.info("✅ 스마트 피드백 수집기 초기화 완료")
    
    def create_gentle_feedback_interface(
        self, 
        image_metadata: Dict[str, Any],
        user_emotion: EmotionEmbedding,
        user_history: List[Dict]
    ) -> Dict[str, Any]:
        """부담 없는 피드백 인터페이스 생성"""
        
        # 1. 기본 점수는 항상 요청 (가장 부담 없음)
        interface = {
            "primary_question": "이 이미지는 어떠세요? (1: 별로 ~ 5: 좋음)",
            "score_required": True
        }
        
        # 2. 사용자 상태에 따른 적응적 추가 질문
        additional_questions = []
        
        # 우울 상태가 심하면 최소한의 질문만
        if user_emotion.valence < -0.5:
            if random.random() < 0.3:  # 30% 확률로만 추가 질문
                additional_questions.append({
                    "type": "optional_mood",
                    "question": "혹시 기분이 조금이라도 나아지셨나요?",
                    "options": ["조금 나아짐", "그대로", "모르겠음"],
                    "optional": True
                })
        
        # 보통 상태면 부드러운 선택형 질문
        elif user_emotion.valence > -0.2:
            if random.random() < 0.7:  # 70% 확률
                # 이미지 특성에 따른 맞춤 질문
                if image_metadata.get("brightness", 0.5) > 0.6:
                    additional_questions.append({
                        "type": "visual_element",
                        "question": "밝은 느낌이 어떠신가요?",
                        "options": ["좋음", "너무 밝음", "괜찮음", "잘 모르겠음"],
                        "optional": True
                    })
                
                if image_metadata.get("saturation", 0.5) > 0.6:
                    additional_questions.append({
                        "type": "color_preference",
                        "question": "색감은 어떠신가요?",
                        "options": ["선명해서 좋음", "너무 선명함", "적당함", "상관없음"],
                        "optional": True
                    })
        
        # 3. 사용자 히스토리 기반 개인화 질문
        if len(user_history) > 5:  # 어느 정도 사용한 사용자
            # 이전 선호도 패턴 분석
            prev_preferences = self._analyze_user_pattern(user_history)
            
            if "prefers_nature" in prev_preferences:
                if "nature" in image_metadata.get("detected_objects", []):
                    additional_questions.append({
                        "type": "content_match",
                        "question": "자연 요소가 마음에 드시나요?",
                        "options": ["네, 좋아요", "아니요", "상관없어요"],
                        "optional": True
                    })
        
        # 4. 간단한 태그 선택 (최대 1-2개만)
        if random.random() < 0.5 and user_emotion.valence > -0.3:
            suggested_tags = self._get_contextual_tags(image_metadata, user_emotion)
            if suggested_tags:
                additional_questions.append({
                    "type": "quick_tags",
                    "question": "이 중에서 느끼는 게 있다면? (선택사항)",
                    "options": suggested_tags + ["없음"],
                    "multiple": True,
                    "optional": True,
                    "max_selections": 2
                })
        
        interface["additional_questions"] = additional_questions
        
        # 5. 격려 메시지 (중요!)
        interface["encouragement"] = self._get_encouragement_message(user_emotion)
        
        return interface
    
    def _analyze_user_pattern(self, user_history: List[Dict]) -> Dict[str, Any]:
        """사용자 패턴 분석 (비침습적)"""
        
        patterns = {}
        
        # 최근 5개 기록만 보기
        recent_history = user_history[-5:]
        
        # 자연 요소 선호도 체크
        nature_mentions = sum(
            1 for record in recent_history 
            if "nature" in record.get("generated_prompt", "").lower() or
               "landscape" in record.get("generated_prompt", "").lower()
        )
        
        if nature_mentions >= 3:
            patterns["prefers_nature"] = True
        
        # 색감 선호도 패턴
        bright_preferences = sum(
            1 for record in recent_history
            if record.get("feedback_score", 3) > 3.5 and
               "bright" in record.get("generated_prompt", "").lower()
        )
        
        if bright_preferences >= 2:
            patterns["prefers_bright"] = True
        
        return patterns
    
    def _get_contextual_tags(
        self, 
        image_metadata: Dict[str, Any], 
        user_emotion: EmotionEmbedding
    ) -> List[str]:
        """컨텍스트에 맞는 태그 제안"""
        
        suggested = []
        
        # 이미지 특성 기반 태그
        brightness = image_metadata.get("brightness", 0.5)
        saturation = image_metadata.get("saturation", 0.5)
        
        if brightness > 0.6:
            suggested.extend(["밝은 느낌", "활기찬"])
        elif brightness < 0.4:
            suggested.extend(["차분한", "조용한"])
        
        if saturation > 0.6:
            suggested.extend(["선명한 색", "생동감"])
        elif saturation < 0.4:
            suggested.extend(["부드러운 색", "은은한"])
        
        # 감정 상태 기반 필터링
        if user_emotion.valence < -0.3:
            # 우울한 상태에서는 부정적 태그 제거
            suggested = [tag for tag in suggested if tag not in ["어두운", "무거운", "답답한"]]
            suggested.extend(["평온한", "안정적인"])  # 긍정적 대안 제공
        
        # 최대 4개까지만
        return suggested[:4]
    
    def _get_encouragement_message(self, user_emotion: EmotionEmbedding) -> str:
        """사용자 상태에 맞는 격려 메시지"""
        
        if user_emotion.valence < -0.5:
            messages = [
                "천천히 생각해보세요. 정답은 없어요.",
                "지금 느끼는 대로 솔직하게 답해주세요.",
                "부담 갖지 마시고 편하게 선택하세요."
            ]
        elif user_emotion.valence < 0:
            messages = [
                "어떤 느낌이든 괜찮습니다.",
                "편안하게 생각나는 대로 선택해주세요.",
                "선택하기 어려우면 패스하셔도 돼요."
            ]
        else:
            messages = [
                "이미지에 대한 솔직한 생각을 들려주세요.",
                "어떤 부분이 인상적이었나요?",
                "느낌을 자유롭게 표현해주세요."
            ]
        
        return random.choice(messages)
    
    def collect_smart_feedback(
        self,
        user_responses: Dict[str, Any],
        interaction_metadata: Dict[str, Any]
    ) -> SmartFeedback:
        """사용자 응답과 메타데이터로 스마트 피드백 구성"""
        
        # 1. 기본 점수 추출
        overall_score = float(user_responses.get("score", 3.0))
        
        # 2. 상호작용 메타데이터 분석
        interaction_time = interaction_metadata.get("viewing_time", 5.0)
        click_behavior = interaction_metadata.get("click_patterns", {})
        
        # 3. 선택적 응답 처리
        quick_tags = user_responses.get("selected_tags", [])
        mood_change = user_responses.get("mood_change")
        
        # 4. 간접 피드백 추론
        inferred_preferences = self._infer_preferences_from_behavior(
            overall_score, interaction_time, click_behavior, quick_tags
        )
        
        # 5. 편안함 수준 추론
        comfort_level = self._infer_comfort_level(
            overall_score, interaction_time, user_responses
        )
        
        # 6. 컨텍스트 요인 수집
        contextual_factors = {
            "response_time": interaction_metadata.get("response_time", 10.0),
            "hesitation_pattern": interaction_metadata.get("hesitation_count", 0),
            "skip_count": sum(1 for v in user_responses.values() if v == "skip"),
            "enthusiasm_level": self._detect_enthusiasm(user_responses)
        }
        
        return SmartFeedback(
            overall_score=overall_score,
            interaction_time=interaction_time,
            click_behavior=click_behavior,
            emotional_response=mood_change,
            quick_tags=quick_tags,
            mood_change=mood_change,
            comfort_level=comfort_level,
            inferred_preferences=inferred_preferences,
            contextual_factors=contextual_factors,
            timestamp=datetime.now()
        )
    
    def _infer_preferences_from_behavior(
        self,
        score: float,
        viewing_time: float,
        clicks: Dict[str, Any],
        tags: List[str]
    ) -> Dict[str, float]:
        """행동 패턴으로 선호도 추론"""
        
        preferences = {}
        
        # 시청 시간 기반 추론
        if viewing_time > 10.0 and score >= 4.0:
            preferences["engagement_level"] = 0.8  # 높은 관심
        elif viewing_time < 3.0:
            preferences["engagement_level"] = 0.2  # 낮은 관심
        else:
            preferences["engagement_level"] = 0.5
        
        # 태그 기반 추론
        if "밝은 느낌" in tags or "활기찬" in tags:
            preferences["brightness_preference"] = 0.7
        elif "차분한" in tags or "조용한" in tags:
            preferences["brightness_preference"] = 0.3
        else:
            preferences["brightness_preference"] = 0.5
        
        if "선명한 색" in tags or "생동감" in tags:
            preferences["saturation_preference"] = 0.7
        elif "부드러운 색" in tags or "은은한" in tags:
            preferences["saturation_preference"] = 0.3
        else:
            preferences["saturation_preference"] = 0.5
        
        # 점수와 행동의 일치성 검사
        if score >= 4.0 and viewing_time > 7.0:
            preferences["consistency_score"] = 0.9  # 일치함
        elif score <= 2.0 and viewing_time < 3.0:
            preferences["consistency_score"] = 0.9  # 일치함
        else:
            preferences["consistency_score"] = 0.5  # 보통
        
        return preferences
    
    def _infer_comfort_level(
        self,
        score: float,
        viewing_time: float,
        responses: Dict[str, Any]
    ) -> float:
        """편안함 수준 추론"""
        
        comfort_indicators = []
        
        # 점수 기반
        if score >= 4.0:
            comfort_indicators.append(0.8)
        elif score >= 3.0:
            comfort_indicators.append(0.6)
        else:
            comfort_indicators.append(0.3)
        
        # 시청 시간 기반 (너무 짧으면 불편했을 수도)
        if 5.0 <= viewing_time <= 15.0:
            comfort_indicators.append(0.7)  # 적절한 시간
        elif viewing_time > 15.0:
            comfort_indicators.append(0.6)  # 오래 봤지만 고민이 많았을 수도
        else:
            comfort_indicators.append(0.4)  # 너무 짧음
        
        # 응답 패턴 기반
        skip_ratio = sum(1 for v in responses.values() if v == "skip") / max(len(responses), 1)
        if skip_ratio < 0.3:
            comfort_indicators.append(0.7)  # 적극적 참여
        else:
            comfort_indicators.append(0.4)  # 많이 넘김
        
        return np.mean(comfort_indicators)
    
    def _detect_enthusiasm(self, responses: Dict[str, Any]) -> str:
        """응답에서 열정/관심도 감지"""
        
        # 선택된 태그 수
        tag_count = len(responses.get("selected_tags", []))
        
        # 추가 질문 응답률
        additional_responses = sum(
            1 for k, v in responses.items() 
            if k.startswith("additional_") and v and v != "skip"
        )
        
        if tag_count >= 2 and additional_responses >= 2:
            return "high"
        elif tag_count >= 1 or additional_responses >= 1:
            return "medium"
        else:
            return "low"


class FeedbackEnhancer:
    """기존 1-5점 피드백을 풍부하게 만드는 시스템"""
    
    def __init__(self):
        self.feedback_collector = SmartFeedbackCollector()
        
    def enhance_simple_feedback(
        self,
        simple_score: float,
        user_emotion: EmotionEmbedding,
        image_metadata: Dict[str, Any],
        user_history: List[Dict],
        interaction_time: float = 5.0
    ) -> Dict[str, Any]:
        """단순한 1-5점 피드백을 풍부한 정보로 확장"""
        
        enhanced_feedback = {
            "original_score": simple_score,
            "enhanced_insights": {},
            "inferred_aspects": {}
        }
        
        # 1. 점수와 감정 상태의 관계 분석
        emotion_score_gap = self._analyze_emotion_score_gap(simple_score, user_emotion)
        enhanced_feedback["enhanced_insights"]["emotion_alignment"] = emotion_score_gap
        
        # 2. 이미지 특성과 점수의 상관관계 분석
        image_score_correlation = self._analyze_image_score_correlation(
            simple_score, image_metadata
        )
        enhanced_feedback["enhanced_insights"]["visual_preferences"] = image_score_correlation
        
        # 3. 사용자 히스토리 패턴 분석
        if user_history:
            pattern_analysis = self._analyze_user_patterns(simple_score, user_history)
            enhanced_feedback["enhanced_insights"]["personal_patterns"] = pattern_analysis
        
        # 4. 상호작용 시간 기반 추론
        engagement_analysis = self._analyze_engagement(simple_score, interaction_time)
        enhanced_feedback["enhanced_insights"]["engagement"] = engagement_analysis
        
        # 5. 구체적 선호도 추론
        specific_preferences = self._infer_specific_preferences(
            simple_score, user_emotion, image_metadata, user_history
        )
        enhanced_feedback["inferred_aspects"] = specific_preferences
        
        return enhanced_feedback
    
    def _analyze_emotion_score_gap(
        self, 
        score: float, 
        emotion: EmotionEmbedding
    ) -> Dict[str, Any]:
        """감정과 점수의 괴리 분석"""
        
        # 감정 상태로 예측되는 점수
        predicted_score = 3.0 + emotion.valence * 1.5  # 기본 3점에서 감정에 따라 조정
        predicted_score = max(1.0, min(5.0, predicted_score))
        
        gap = score - predicted_score
        
        analysis = {
            "predicted_score": predicted_score,
            "actual_score": score,
            "gap": gap,
            "interpretation": ""
        }
        
        if abs(gap) < 0.5:
            analysis["interpretation"] = "감정 상태와 일치하는 반응"
        elif gap > 0.5:
            analysis["interpretation"] = "감정 대비 긍정적 반응 (이미지가 기분 개선에 도움)"
        else:
            analysis["interpretation"] = "감정 대비 부정적 반응 (이미지가 부담스러웠을 수 있음)"
        
        return analysis
    
    def _analyze_image_score_correlation(
        self, 
        score: float, 
        metadata: Dict[str, Any]
    ) -> Dict[str, float]:
        """이미지 특성과 점수의 상관관계"""
        
        correlations = {}
        
        # 밝기와 점수 관계
        brightness = metadata.get("brightness", 0.5)
        if score >= 4.0 and brightness > 0.6:
            correlations["brightness_positive"] = 0.8
        elif score <= 2.0 and brightness > 0.7:
            correlations["brightness_negative"] = 0.8  # 너무 밝아서 부담
        
        # 채도와 점수 관계
        saturation = metadata.get("saturation", 0.5)
        if score >= 4.0 and saturation > 0.6:
            correlations["saturation_positive"] = 0.7
        elif score <= 2.0 and saturation > 0.8:
            correlations["saturation_negative"] = 0.7  # 너무 선명해서 부담
        
        # 대비와 점수 관계
        contrast = metadata.get("contrast", 0.5)
        if score >= 4.0 and 0.4 <= contrast <= 0.7:
            correlations["contrast_optimal"] = 0.8
        elif score <= 2.0 and contrast > 0.8:
            correlations["contrast_excessive"] = 0.7
        
        return correlations
    
    def _analyze_user_patterns(
        self, 
        score: float, 
        history: List[Dict]
    ) -> Dict[str, Any]:
        """사용자 패턴 분석"""
        
        recent_scores = [
            record.get("feedback_score", 3.0) 
            for record in history[-5:] 
            if record.get("feedback_score")
        ]
        
        if not recent_scores:
            return {"pattern": "insufficient_data"}
        
        avg_score = np.mean(recent_scores)
        score_trend = recent_scores[-1] - recent_scores[0] if len(recent_scores) > 1 else 0
        
        pattern_analysis = {
            "average_score": avg_score,
            "current_vs_average": score - avg_score,
            "score_trend": score_trend,
            "consistency": np.std(recent_scores),
            "interpretation": ""
        }
        
        if abs(score - avg_score) < 0.5:
            pattern_analysis["interpretation"] = "일관된 선호도 패턴"
        elif score > avg_score + 0.5:
            pattern_analysis["interpretation"] = "평소보다 긍정적 반응"
        else:
            pattern_analysis["interpretation"] = "평소보다 부정적 반응"
        
        return pattern_analysis
    
    def _analyze_engagement(self, score: float, interaction_time: float) -> Dict[str, Any]:
        """참여도 분석"""
        
        # 점수 대비 적절한 시청 시간 예측
        if score >= 4.0:
            expected_time_range = (7.0, 20.0)  # 좋으면 오래 볼 것
        elif score >= 3.0:
            expected_time_range = (5.0, 12.0)  # 보통이면 적당히
        else:
            expected_time_range = (2.0, 8.0)   # 나쁘면 짧게 또는 고민하며 길게
        
        engagement = {
            "viewing_time": interaction_time,
            "expected_range": expected_time_range,
            "engagement_level": "",
            "interpretation": ""
        }
        
        if expected_time_range[0] <= interaction_time <= expected_time_range[1]:
            engagement["engagement_level"] = "appropriate"
            engagement["interpretation"] = "점수와 일치하는 참여도"
        elif interaction_time > expected_time_range[1]:
            engagement["engagement_level"] = "high"
            engagement["interpretation"] = "높은 관심도 또는 신중한 고려"
        else:
            engagement["engagement_level"] = "low"
            engagement["interpretation"] = "빠른 판단 또는 낮은 관심도"
        
        return engagement
    
    def _infer_specific_preferences(
        self,
        score: float,
        emotion: EmotionEmbedding,
        metadata: Dict[str, Any],
        history: List[Dict]
    ) -> Dict[str, Any]:
        """구체적 선호도 요소 추론"""
        
        preferences = {}
        
        # 색감 선호도
        if score >= 4.0:
            brightness = metadata.get("brightness", 0.5)
            saturation = metadata.get("saturation", 0.5)
            
            if brightness > 0.6:
                preferences["preferred_brightness"] = "bright"
            elif brightness < 0.4:
                preferences["preferred_brightness"] = "dim"
            
            if saturation > 0.6:
                preferences["preferred_saturation"] = "vibrant"
            elif saturation < 0.4:
                preferences["preferred_saturation"] = "muted"
        
        # 기분별 선호도 패턴
        if emotion.valence < -0.3 and score >= 4.0:
            preferences["therapeutic_preference"] = "comforting_when_sad"
            # 우울할 때 좋아한 이미지 스타일 기록
            
        # 시간대별 패턴 (히스토리 기반)
        if history:
            hour = datetime.now().hour
            if 6 <= hour <= 10:  # 아침
                preferences["time_context"] = "morning_preference"
            elif 18 <= hour <= 22:  # 저녁
                preferences["time_context"] = "evening_preference"
        
        return preferences