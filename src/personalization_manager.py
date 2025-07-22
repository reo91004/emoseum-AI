# src/personalization_manager.py

import re
from typing import Dict, List, Tuple, Any
import logging
from textblob import TextBlob

logger = logging.getLogger(__name__)


class PersonalizationManager:
    """룰 기반 개인화 엔진"""

    def __init__(self, user_manager):
        self.user_manager = user_manager

        # 감정 극성 키워드 사전
        self.positive_keywords = {
            "joy",
            "peace",
            "calm",
            "hope",
            "light",
            "bright",
            "warm",
            "gentle",
            "beautiful",
            "serene",
            "harmony",
            "balance",
            "comfort",
            "relief",
            "sunshine",
            "bloom",
            "growth",
            "healing",
            "renewal",
            "awakening",
        }

        self.negative_keywords = {
            "dark",
            "heavy",
            "storm",
            "cold",
            "empty",
            "void",
            "chaos",
            "turbulent",
            "shadow",
            "grey",
            "bitter",
            "harsh",
            "struggle",
            "burden",
            "weight",
            "alone",
            "isolated",
            "distant",
            "broken",
            "torn",
            "shattered",
        }

        # 시각적 요소와 선호도 매핑
        self.visual_element_mapping = {
            "brightness": {
                "positive_indicators": [
                    "bright",
                    "light",
                    "sunny",
                    "golden",
                    "radiant",
                ],
                "negative_indicators": ["dark", "dim", "shadow", "grey", "dull"],
            },
            "saturation": {
                "positive_indicators": ["vibrant", "colorful", "vivid", "rich", "bold"],
                "negative_indicators": ["muted", "pale", "washed", "faded", "grey"],
            },
            "complexity": {
                "positive_indicators": [
                    "detailed",
                    "intricate",
                    "rich",
                    "layered",
                    "complex",
                ],
                "negative_indicators": [
                    "simple",
                    "minimal",
                    "clean",
                    "sparse",
                    "basic",
                ],
            },
            "warmth": {
                "positive_indicators": ["warm", "golden", "orange", "red", "cozy"],
                "negative_indicators": ["cool", "blue", "cold", "icy", "distant"],
            },
        }

        # 아트 스타일 키워드 매핑
        self.style_keywords = {
            "painting": ["brush", "canvas", "artistic", "painted", "oil", "watercolor"],
            "photography": [
                "photo",
                "realistic",
                "natural",
                "captured",
                "lens",
                "light",
            ],
            "abstract": [
                "abstract",
                "conceptual",
                "symbolic",
                "metaphor",
                "dream",
                "surreal",
            ],
        }

    def update_preferences_from_guestbook(
        self,
        user_id: str,
        guestbook_title: str,
        guestbook_tags: List[str],
        image_prompt: str,
        image_metadata: Dict[str, Any] = None,
    ) -> Dict[str, float]:
        """방명록 데이터 분석하여 선호도 업데이트"""

        # 1. 제목과 태그의 감정 극성 분석
        title_sentiment = self._analyze_sentiment(guestbook_title)
        tags_sentiment = self._analyze_tags_sentiment(guestbook_tags)
        overall_sentiment = (title_sentiment + tags_sentiment) / 2

        logger.info(
            f"감정 극성 분석: 제목({title_sentiment:.2f}), 태그({tags_sentiment:.2f}), 전체({overall_sentiment:.2f})"
        )

        # 2. 긍정적 반응일 때만 선호도 강화
        if overall_sentiment > 0.1:  # 약간 긍정적 이상
            weight_updates = self._calculate_preference_updates(
                image_prompt, guestbook_title, guestbook_tags, overall_sentiment
            )

            # 3. 사용자 선호도 업데이트
            self.user_manager.update_preference_weights(user_id, weight_updates)

            logger.info(
                f"사용자 {user_id}의 선호도가 업데이트되었습니다: {weight_updates}"
            )
            return weight_updates
        else:
            logger.info("중성적 또는 부정적 반응으로 선호도 업데이트를 건너뜁니다.")
            return {}

    def _analyze_sentiment(self, text: str) -> float:
        """텍스트 감정 극성 분석"""
        if not text:
            return 0.0

        text_lower = text.lower()

        # 키워드 기반 분석
        positive_count = sum(1 for word in self.positive_keywords if word in text_lower)
        negative_count = sum(1 for word in self.negative_keywords if word in text_lower)

        # TextBlob을 사용한 보조 분석 (영어만 지원)
        try:
            blob_sentiment = TextBlob(text).sentiment.polarity
        except:
            blob_sentiment = 0.0

        # 키워드 기반과 TextBlob 결합
        keyword_sentiment = (positive_count - negative_count) / max(
            1, len(text_lower.split())
        )

        # 가중 평균 (키워드 기반에 더 높은 가중치)
        final_sentiment = 0.7 * keyword_sentiment + 0.3 * blob_sentiment

        return max(-1.0, min(1.0, final_sentiment))

    def _analyze_tags_sentiment(self, tags: List[str]) -> float:
        """태그들의 감정 극성 분석"""
        if not tags:
            return 0.0

        sentiments = []
        for tag in tags:
            sentiment = self._analyze_sentiment(tag)
            sentiments.append(sentiment)

        return sum(sentiments) / len(sentiments)

    def _calculate_preference_updates(
        self, prompt: str, title: str, tags: List[str], sentiment_strength: float
    ) -> Dict[str, float]:
        """선호도 업데이트 값 계산"""

        updates = {}

        # 감정 강도에 따른 학습률 조절
        learning_rate = min(0.1, sentiment_strength * 0.05)  # 최대 0.1

        # 1. 시각적 요소 분석
        visual_updates = self._analyze_visual_preferences(
            prompt, title, tags, learning_rate
        )
        updates.update(visual_updates)

        # 2. 스타일 선호도 분석
        style_updates = self._analyze_style_preferences(
            prompt, title, tags, learning_rate
        )
        updates.update(style_updates)

        # 3. 색감 선호도 분석
        color_updates = self._analyze_color_preferences(
            prompt, title, tags, learning_rate
        )
        updates.update(color_updates)

        return updates

    def _analyze_visual_preferences(
        self, prompt: str, title: str, tags: List[str], lr: float
    ) -> Dict[str, float]:
        """시각적 요소 선호도 분석"""
        updates = {}
        combined_text = f"{prompt} {title} {' '.join(tags)}".lower()

        for element, indicators in self.visual_element_mapping.items():
            positive_matches = sum(
                1 for word in indicators["positive_indicators"] if word in combined_text
            )
            negative_matches = sum(
                1 for word in indicators["negative_indicators"] if word in combined_text
            )

            if positive_matches > negative_matches:
                # 해당 요소에 대한 선호도 증가
                if element == "brightness":
                    updates["brightness"] = lr * positive_matches
                elif element == "saturation":
                    updates["saturation"] = lr * positive_matches
                elif element == "complexity":
                    # 복잡도는 스타일 가중치에 영향
                    updates.setdefault("style_complexity_preference", 0)
                    updates["style_complexity_preference"] += lr * positive_matches
                elif element == "warmth":
                    # 색온도 선호도 (시각 선호도 파일의 color_tone에 영향)
                    updates.setdefault("warm_preference", 0)
                    updates["warm_preference"] += lr * positive_matches

        return updates

    def _analyze_style_preferences(
        self, prompt: str, title: str, tags: List[str], lr: float
    ) -> Dict[str, float]:
        """아트 스타일 선호도 분석"""
        updates = {}
        combined_text = f"{prompt} {title} {' '.join(tags)}".lower()

        for style, keywords in self.style_keywords.items():
            matches = sum(1 for word in keywords if word in combined_text)
            if matches > 0:
                # style_weights 딕셔너리의 해당 스타일 가중치 증가
                updates[style] = lr * matches * 0.1  # 스타일 업데이트는 더 보수적

        return updates

    def _analyze_color_preferences(
        self, prompt: str, title: str, tags: List[str], lr: float
    ) -> Dict[str, float]:
        """색감 선호도 분석"""
        updates = {}
        combined_text = f"{prompt} {title} {' '.join(tags)}".lower()

        # 색상별 키워드 매핑
        color_mappings = {
            "warm_colors": [
                "warm",
                "orange",
                "red",
                "yellow",
                "golden",
                "sunset",
                "fire",
            ],
            "cool_colors": ["cool", "blue", "green", "purple", "ocean", "sky", "ice"],
            "pastel_colors": [
                "pastel",
                "soft",
                "gentle",
                "light",
                "pale",
                "subtle",
                "muted",
            ],
        }

        for color_type, keywords in color_mappings.items():
            matches = sum(1 for word in keywords if word in combined_text)
            if matches > 0:
                updates[color_type] = lr * matches * 0.05  # 색감 업데이트도 보수적

        return updates

    def get_personalization_insights(self, user_id: str) -> Dict[str, Any]:
        """개인화 인사이트 제공"""
        user = self.user_manager.get_user(user_id)
        if not user:
            return {}

        prefs = user.visual_preferences

        insights = {
            "user_id": user_id,
            "dominant_style": max(prefs.style_weights.items(), key=lambda x: x[1])[0],
            "style_distribution": prefs.style_weights,
            "visual_preferences": {
                "art_style": prefs.art_style,
                "color_tone": prefs.color_tone,
                "complexity": prefs.complexity,
                "brightness": prefs.brightness,
                "saturation": prefs.saturation,
            },
            "personalization_level": self._calculate_personalization_level(prefs),
        }

        return insights

    def _calculate_personalization_level(self, preferences) -> str:
        """개인화 수준 계산"""
        # 스타일 가중치의 분산을 통해 개인화 수준 측정
        weights = list(preferences.style_weights.values())
        variance = sum((w - (1 / len(weights))) ** 2 for w in weights) / len(weights)

        if variance > 0.1:
            return "high"  # 특정 스타일에 강한 선호
        elif variance > 0.05:
            return "medium"
        else:
            return "low"  # 고른 선호도

    def simulate_preference_learning(
        self, user_id: str, simulation_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """선호도 학습 시뮬레이션 (테스트/데모용)"""

        results = {
            "initial_preferences": self.get_personalization_insights(user_id),
            "learning_steps": [],
            "final_preferences": None,
        }

        for i, data in enumerate(simulation_data):
            # 각 스텝별 학습 수행
            weight_updates = self.update_preferences_from_guestbook(
                user_id=user_id,
                guestbook_title=data.get("title", ""),
                guestbook_tags=data.get("tags", []),
                image_prompt=data.get("prompt", ""),
                image_metadata=data.get("metadata", {}),
            )

            step_result = {
                "step": i + 1,
                "input_data": data,
                "weight_updates": weight_updates,
                "current_preferences": self.get_personalization_insights(user_id),
            }

            results["learning_steps"].append(step_result)

        results["final_preferences"] = self.get_personalization_insights(user_id)

        return results

    def export_user_learning_data(self, user_id: str) -> Dict[str, Any]:
        """사용자 학습 데이터 내보내기 (Level 3 학습용)"""
        user = self.user_manager.get_user(user_id)
        if not user:
            return {}

        # 이 데이터는 나중에 LoRA/DRaFT 학습에 사용될 수 있음
        learning_data = {
            "user_id": user_id,
            "visual_preferences": user.visual_preferences.__dict__,
            "coping_style": self.user_manager.get_current_coping_style(user_id),
            "personalization_insights": self.get_personalization_insights(user_id),
            "ready_for_advanced_learning": len(user.psychometric_results)
            >= 2,  # 최소 2회 검사
        }

        return learning_data

    def recommend_content_adjustments(self, user_id: str) -> Dict[str, str]:
        """사용자별 컨텐츠 조정 권장사항"""
        insights = self.get_personalization_insights(user_id)
        coping_style = self.user_manager.get_current_coping_style(user_id)

        recommendations = {}

        # 대처 스타일 기반 권장사항
        if coping_style == "avoidant":
            recommendations["prompt_style"] = "더 부드럽고 은유적인 표현 사용"
            recommendations["image_guidance"] = (
                "직접적이지 않은, 간접적인 감정 표현 권장"
            )
        elif coping_style == "confrontational":
            recommendations["prompt_style"] = "더 직설적이고 명확한 감정 표현"
            recommendations["image_guidance"] = "감정을 선명하게 드러내는 시각화"

        # 개인화 수준 기반 권장사항
        personalization_level = insights.get("personalization_level", "low")
        if personalization_level == "low":
            recommendations["learning_focus"] = "더 구체적인 선호도 수집 필요"
        elif personalization_level == "high":
            recommendations["learning_focus"] = "Level 3 고급 학습 모델 적용 고려"

        return recommendations
