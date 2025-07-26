# src/managers/personalization_manager.py

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

        # 메시지 반응 타입별 가중치
        self.reaction_weights = {
            "like": 0.1,
            "save": 0.2,
            "share": 0.3,
            "dismiss": -0.1,
            "skip": -0.05,
        }

        # 큐레이터 메시지 요소별 키워드
        self.message_element_keywords = {
            "encouragement": ["용기", "대단", "감동", "아름답", "훌륭", "인상적"],
            "growth_recognition": ["성장", "발전", "향상", "변화", "깊이", "이해"],
            "future_guidance": ["행동", "실천", "계획", "도전", "에너지", "변화"],
            "connection": ["함께", "응원", "지지", "연결", "기다리", "믿고"],
            "personal_strength": ["지혜", "용기", "힘", "능력", "강점", "재능"],
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

    def update_preferences_from_message_reaction(
        self,
        user_id: str,
        reaction_type: str,
        curator_message: Dict[str, Any],
        guestbook_data: Dict[str, Any],
        additional_context: Dict[str, Any] = None,
    ) -> Dict[str, float]:
        """큐레이터 메시지 반응 기반 선호도 업데이트"""

        logger.info(f"사용자 {user_id}의 메시지 반응 기반 학습: {reaction_type}")

        # 1. 반응 유형별 가중치 확인
        reaction_weight = self.reaction_weights.get(reaction_type, 0.0)

        if reaction_weight <= 0:
            logger.info(
                f"부정적 또는 중성적 반응으로 학습을 건너뜁니다: {reaction_type}"
            )
            return {}

        # 2. 메시지 내용 분석
        message_analysis = self._analyze_curator_message(curator_message)

        # 3. 방명록 컨텍스트 분석
        guestbook_analysis = self._analyze_guestbook_context(guestbook_data)

        # 4. 선호도 업데이트 계산
        weight_updates = self._calculate_message_based_updates(
            message_analysis, guestbook_analysis, reaction_weight
        )

        # 5. 사용자 선호도 업데이트
        if weight_updates:
            self.user_manager.update_preference_weights(user_id, weight_updates)
            logger.info(f"메시지 반응 기반 선호도 업데이트 완료: {weight_updates}")

        return weight_updates

    def _analyze_curator_message(
        self, curator_message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """큐레이터 메시지 내용 분석"""

        content = curator_message.get("content", {})
        if not content:
            return {}

        analysis = {
            "message_elements": [],
            "tone_indicators": [],
            "personalization_level": 0,
        }

        # 메시지 각 부분 분석
        for section_name, section_content in content.items():
            if isinstance(section_content, str) and section_content:
                # 메시지 요소 식별
                identified_elements = self._identify_message_elements(section_content)
                analysis["message_elements"].extend(identified_elements)

                # 톤 지표 분석
                tone = self._analyze_message_tone(section_content)
                if tone:
                    analysis["tone_indicators"].append(tone)

        # 개인화 수준 계산
        personalization_data = curator_message.get("personalization_data", {})
        analysis["personalization_level"] = self._calculate_personalization_level(
            personalization_data
        )

        return analysis

    def _identify_message_elements(self, text: str) -> List[str]:
        """메시지에서 요소 식별"""
        elements = []
        text_lower = text.lower()

        for element_type, keywords in self.message_element_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                elements.append(element_type)

        return elements

    def _analyze_message_tone(self, text: str) -> str:
        """메시지 톤 분석"""
        text_lower = text.lower()

        if any(word in text_lower for word in ["부드럽", "따뜻", "온화", "섬세"]):
            return "gentle"
        elif any(word in text_lower for word in ["강한", "확고", "명확", "직접"]):
            return "direct"
        elif any(word in text_lower for word in ["균형", "조화", "안정", "성숙"]):
            return "balanced"
        else:
            return "neutral"

    def _calculate_personalization_level(
        self, personalization_data: Dict[str, Any]
    ) -> float:
        """개인화 수준 계산"""
        if not personalization_data:
            return 0.0

        factors = []

        # 대처 스타일 고려 여부
        if "coping_style" in personalization_data:
            factors.append(0.3)

        # 개인화 요소 포함 여부
        personalized_elements = personalization_data.get("personalized_elements", {})
        if personalized_elements:
            factors.append(0.4)

        # 성장 단계 고려 여부
        if "growth_stage" in personalization_data:
            factors.append(0.3)

        return sum(factors)

    def _analyze_guestbook_context(
        self, guestbook_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """방명록 컨텍스트 분석"""

        title = guestbook_data.get("title", "")
        tags = guestbook_data.get("tags", [])

        return {
            "title_sentiment": self._analyze_sentiment(title),
            "tags_sentiment": self._analyze_tags_sentiment(tags),
            "title_characteristics": self._analyze_title_characteristics(title),
            "tag_themes": self._analyze_tag_themes(tags),
        }

    def _analyze_title_characteristics(self, title: str) -> Dict[str, bool]:
        """제목 특성 분석"""
        title_lower = title.lower()

        return {
            "has_metaphor": any(
                word in title_lower
                for word in ["같은", "처럼", "마치", "듯한", "속의", "안의"]
            ),
            "has_color": any(
                word in title_lower
                for word in ["빨간", "파란", "노란", "검은", "하얀", "회색"]
            ),
            "has_nature": any(
                word in title_lower
                for word in ["바다", "하늘", "나무", "꽃", "바람", "비", "해", "달"]
            ),
            "has_emotion": any(
                word in title_lower
                for word in ["joy", "sadness", "anger", "peace", "anxiety", "loneliness"]
            ),
        }

    def _analyze_tag_themes(self, tags: List[str]) -> List[str]:
        """태그 테마 분석"""
        themes = []

        combined_tags = " ".join(tags).lower()

        theme_keywords = {
            "nature": ["자연", "바다", "하늘", "숲", "꽃", "나무"],
            "emotion": ["감정", "마음", "기분", "느낌", "생각"],
            "time": ["시간", "순간", "하루", "오늘", "내일", "과거"],
            "space": ["공간", "장소", "집", "길", "여행", "방"],
            "relationship": ["사람", "친구", "가족", "연인", "관계"],
            "abstract": ["꿈", "희망", "의미", "가치", "철학", "생각"],
        }

        for theme, keywords in theme_keywords.items():
            if any(keyword in combined_tags for keyword in keywords):
                themes.append(theme)

        return themes

    def _calculate_message_based_updates(
        self,
        message_analysis: Dict[str, Any],
        guestbook_analysis: Dict[str, Any],
        reaction_weight: float,
    ) -> Dict[str, float]:
        """메시지 반응 기반 선호도 업데이트 계산"""

        updates = {}

        # 1. 메시지 요소별 선호도 업데이트
        message_elements = message_analysis.get("message_elements", [])
        for element in message_elements:
            # 각 메시지 요소에 대한 선호도 강화
            updates[f"message_{element}_preference"] = reaction_weight * 0.1

        # 2. 톤 선호도 업데이트
        tone_indicators = message_analysis.get("tone_indicators", [])
        for tone in tone_indicators:
            updates[f"tone_{tone}_preference"] = reaction_weight * 0.15

        # 3. 개인화 수준 기반 업데이트
        personalization_level = message_analysis.get("personalization_level", 0)
        if personalization_level > 0.5:
            updates["high_personalization_preference"] = reaction_weight * 0.2

        # 4. 방명록 컨텍스트 기반 업데이트
        title_characteristics = guestbook_analysis.get("title_characteristics", {})
        for characteristic, present in title_characteristics.items():
            if present:
                updates[f"title_{characteristic}_preference"] = reaction_weight * 0.05

        # 5. 태그 테마 기반 업데이트
        tag_themes = guestbook_analysis.get("tag_themes", [])
        for theme in tag_themes:
            updates[f"theme_{theme}_preference"] = reaction_weight * 0.08

        return updates

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
            "personalization_level": self._calculate_personalization_level_user(prefs),
            "message_preferences": self._get_message_preferences(user_id),
        }

        return insights

    def _calculate_personalization_level_user(self, preferences) -> str:
        """사용자 개인화 수준 계산"""
        # 스타일 가중치의 분산을 통해 개인화 수준 측정
        weights = list(preferences.style_weights.values())
        variance = sum((w - (1 / len(weights))) ** 2 for w in weights) / len(weights)

        if variance > 0.1:
            return "high"  # 특정 스타일에 강한 선호
        elif variance > 0.05:
            return "medium"
        else:
            return "low"  # 고른 선호도

    def _get_message_preferences(self, user_id: str) -> Dict[str, Any]:
        """메시지 선호도 분석"""
        # 실제 구현에서는 사용자의 메시지 반응 히스토리를 분석
        # 여기서는 간단한 더미 데이터 반환
        return {
            "preferred_tone": "gentle",
            "preferred_elements": ["encouragement", "growth_recognition"],
            "personalization_preference": "high",
        }

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
            if data.get("type") == "guestbook":
                weight_updates = self.update_preferences_from_guestbook(
                    user_id=user_id,
                    guestbook_title=data.get("title", ""),
                    guestbook_tags=data.get("tags", []),
                    image_prompt=data.get("prompt", ""),
                    image_metadata=data.get("metadata", {}),
                )
            elif data.get("type") == "message_reaction":
                weight_updates = self.update_preferences_from_message_reaction(
                    user_id=user_id,
                    reaction_type=data.get("reaction_type", "like"),
                    curator_message=data.get("curator_message", {}),
                    guestbook_data=data.get("guestbook_data", {}),
                )
            else:
                weight_updates = {}

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
            recommendations["message_guidance"] = "간접적이고 보호적인 큐레이터 메시지"
        elif coping_style == "confrontational":
            recommendations["prompt_style"] = "더 직설적이고 명확한 감정 표현"
            recommendations["message_guidance"] = "직접적이고 용기를 강조하는 메시지"

        # 개인화 수준 기반 권장사항
        personalization_level = insights.get("personalization_level", "low")
        if personalization_level == "low":
            recommendations["learning_focus"] = "더 구체적인 선호도 수집 필요"
        elif personalization_level == "high":
            recommendations["learning_focus"] = "Level 3 고급 학습 모델 적용 고려"

        # 메시지 선호도 기반 권장사항
        message_prefs = insights.get("message_preferences", {})
        preferred_tone = message_prefs.get("preferred_tone", "neutral")
        recommendations["curator_tone"] = f"{preferred_tone} 톤의 메시지 우선 사용"

        return recommendations
