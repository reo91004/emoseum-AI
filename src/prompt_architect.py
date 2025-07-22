# src/prompt_architect.py

from typing import Dict, List, Tuple, Any
import random
import re
import logging

logger = logging.getLogger(__name__)


class PromptArchitect:
    """ACT 기반 프롬프트 생성 시스템"""

    def __init__(self):
        self.style_templates = {
            "painting": "oil painting style, artistic brushstrokes, canvas texture",
            "photography": "photographic style, natural lighting, realistic",
            "abstract": "abstract art style, conceptual, non-representational",
        }

        self.color_templates = {
            "warm": "warm colors, golden tones, orange and red hues",
            "cool": "cool colors, blue and green tones, calming palette",
            "pastel": "pastel colors, soft tones, gentle hues",
        }

        self.complexity_templates = {
            "simple": "minimalist, clean composition, simple elements",
            "balanced": "balanced composition, moderate detail",
            "complex": "detailed, intricate, rich composition",
        }

        # 감정 순화 규칙 (회피 성향 사용자용)
        self.emotion_softening_rules = {
            "overwhelming": "gentle",
            "crushing": "heavy",
            "devastating": "profound",
            "terrifying": "uncertain",
            "hopeless": "challenging",
            "desperate": "yearning",
            "agonizing": "difficult",
            "unbearable": "intense",
            "suffocating": "confining",
            "nightmare": "dream-like",
        }

        # 희망적 전환 키워드
        self.hope_keywords = [
            "warm light",
            "gentle breeze",
            "opening path",
            "blooming flower",
            "sunrise",
            "bridge",
            "open door",
            "guiding star",
            "peaceful stream",
            "growing tree",
            "soft glow",
            "healing rain",
            "gentle hands",
            "welcoming space",
            "quiet strength",
        ]

    def create_reflection_prompt(
        self,
        emotion_keywords: List[str],
        vad_scores: Tuple[float, float, float],  # valence, arousal, dominance
        coping_style: str,
        visual_preferences: Dict[str, Any],
    ) -> str:
        """감정 반영 프롬프트 생성 (ACT 2단계: Acceptance)"""

        valence, arousal, dominance = vad_scores

        # 감정 키워드 전처리 (대처 스타일에 따라)
        processed_keywords = self._process_emotion_keywords(
            emotion_keywords, coping_style
        )

        # 기본 감정 묘사
        emotion_description = self._create_emotion_description(
            processed_keywords, valence, arousal, dominance, coping_style
        )

        # 시각적 스타일 적용
        style_part = self._apply_visual_style(visual_preferences)

        # 중립적이고 수용적인 분위기 조성
        acceptance_modifiers = self._get_acceptance_modifiers(coping_style)

        prompt = f"{emotion_description}, {style_part}, {acceptance_modifiers}"

        # 품질 향상 키워드 추가
        prompt += ", high quality, detailed, masterpiece"

        logger.info(f"Reflection 프롬프트가 생성되었습니다 ({coping_style} 스타일)")
        return prompt

    def create_hope_prompt(
        self,
        original_prompt: str,
        guestbook_title: str,
        guestbook_tags: List[str],
        visual_preferences: Dict[str, Any],
    ) -> str:
        """희망 이미지 프롬프트 생성 (ACT 4단계: Values & Committed Action)"""

        # 원본 프롬프트에서 부정적 요소 식별 및 제거
        base_prompt = self._neutralize_negative_elements(original_prompt)

        # 사용자가 명명한 제목을 바탕으로 희망적 요소 선택
        hope_elements = self._select_hope_elements(guestbook_title, guestbook_tags)

        # 시각적 스타일 재적용
        style_part = self._apply_visual_style(visual_preferences)

        # 희망적 서사 구성
        hope_narrative = self._create_hope_narrative(hope_elements)

        # 최종 프롬프트 조합
        prompt = f"{base_prompt}, {hope_narrative}, {style_part}"
        prompt += ", uplifting, inspiring, peaceful, high quality, masterpiece"

        logger.info(f"Hope 프롬프트가 생성되었습니다: {guestbook_title}")
        return prompt

    def _process_emotion_keywords(
        self, keywords: List[str], coping_style: str
    ) -> List[str]:
        """대처 스타일에 따른 감정 키워드 전처리"""
        processed = []

        for keyword in keywords:
            keyword_lower = keyword.lower()

            if coping_style == "avoidant":
                # 회피 성향: 부정적 키워드를 부드럽게 순화
                softened = self._soften_emotion_keyword(keyword_lower)
                processed.append(softened)
            elif coping_style == "confrontational":
                # 직면 성향: 키워드를 그대로 사용하되 더 구체적으로
                intensified = self._intensify_emotion_keyword(keyword_lower)
                processed.append(intensified)
            else:  # balanced
                # 균형: 적당한 수준으로 조절
                balanced = self._balance_emotion_keyword(keyword_lower)
                processed.append(balanced)

        return processed

    def _soften_emotion_keyword(self, keyword: str) -> str:
        """감정 키워드 순화"""
        for harsh, soft in self.emotion_softening_rules.items():
            if harsh in keyword:
                return keyword.replace(harsh, soft)

        # 추가 순화 규칙
        if any(neg in keyword for neg in ["angry", "rage", "furious"]):
            return "unsettled feeling"
        elif any(neg in keyword for neg in ["depressed", "sad", "miserable"]):
            return "gentle melancholy"
        elif any(neg in keyword for neg in ["anxious", "worried", "fearful"]):
            return "thoughtful contemplation"

        return keyword

    def _intensify_emotion_keyword(self, keyword: str) -> str:
        """감정 키워드 강화 (직면 성향용)"""
        intensity_map = {
            "sad": "deeply sorrowful",
            "angry": "intense anger",
            "worried": "sharp anxiety",
            "tired": "profound exhaustion",
            "lonely": "acute solitude",
            "confused": "mental turbulence",
        }

        return intensity_map.get(keyword, f"vivid {keyword}")

    def _balance_emotion_keyword(self, keyword: str) -> str:
        """감정 키워드 균형 조절"""
        # 중성적 표현으로 전환
        balanced_map = {
            "overwhelming": "encompassing",
            "crushing": "weighty",
            "terrifying": "unsettling",
            "devastating": "impactful",
        }

        return balanced_map.get(keyword, keyword)

    def _create_emotion_description(
        self,
        keywords: List[str],
        valence: float,
        arousal: float,
        dominance: float,
        coping_style: str,
    ) -> str:
        """감정 묘사 생성"""

        # VAD 점수를 바탕으로 시각적 은유 선택
        if valence < -0.3:  # 부정적
            if arousal > 0.3:  # 높은 각성
                base_metaphor = "stormy emotional landscape"
            else:  # 낮은 각성
                base_metaphor = "quiet, contemplative scene"
        else:  # 긍정적 또는 중성
            if arousal > 0.3:
                base_metaphor = "dynamic, energetic scene"
            else:
                base_metaphor = "peaceful, serene landscape"

        # 키워드 통합
        emotion_elements = ", ".join(keywords[:3])  # 주요 키워드 3개만 사용

        return f"{base_metaphor} reflecting {emotion_elements}"

    def _apply_visual_style(self, preferences: Dict[str, Any]) -> str:
        """시각적 스타일 적용"""
        components = []

        # 예술 스타일
        art_style = preferences.get("art_style", "painting")
        components.append(self.style_templates.get(art_style, "artistic style"))

        # 색감
        color_tone = preferences.get("color_tone", "warm")
        components.append(self.color_templates.get(color_tone, "balanced colors"))

        # 복잡도
        complexity = preferences.get("complexity", "balanced")
        components.append(
            self.complexity_templates.get(complexity, "balanced composition")
        )

        return ", ".join(components)

    def _get_acceptance_modifiers(self, coping_style: str) -> str:
        """수용적 분위기 조성 수정자"""
        base_modifiers = ["non-judgmental", "accepting atmosphere", "gentle presence"]

        if coping_style == "avoidant":
            base_modifiers.extend(["soft, safe space", "protective environment"])
        elif coping_style == "confrontational":
            base_modifiers.extend(["honest clarity", "authentic expression"])
        else:
            base_modifiers.extend(["balanced perspective", "mindful awareness"])

        return ", ".join(random.sample(base_modifiers, 3))

    def _neutralize_negative_elements(self, prompt: str) -> str:
        """프롬프트에서 부정적 요소 중성화"""
        # 부정적 키워드 제거 또는 중성화
        negative_patterns = [
            (r"\b(dark|gloomy|depressing|sad|angry|hopeless)\b", "contemplative"),
            (r"\b(stormy|turbulent|chaotic)\b", "dynamic"),
            (r"\b(heavy|crushing|overwhelming)\b", "meaningful"),
            (r"\b(isolated|lonely|empty)\b", "spacious"),
        ]

        neutralized = prompt
        for pattern, replacement in negative_patterns:
            neutralized = re.sub(pattern, replacement, neutralized, flags=re.IGNORECASE)

        return neutralized

    def _select_hope_elements(self, title: str, tags: List[str]) -> List[str]:
        """제목과 태그를 바탕으로 희망 요소 선택"""
        selected_elements = []

        # 제목 분석을 통한 희망 요소 선택
        title_lower = title.lower()

        if any(word in title_lower for word in ["blue", "grey", "dark"]):
            selected_elements.extend(["warm light", "gentle sunrise"])
        elif any(word in title_lower for word in ["cold", "winter", "frozen"]):
            selected_elements.extend(["cozy warmth", "spring awakening"])
        elif any(word in title_lower for word in ["empty", "void", "hollow"]):
            selected_elements.extend(["growing garden", "filling space"])
        elif any(word in title_lower for word in ["storm", "rain", "clouds"]):
            selected_elements.extend(["clearing sky", "rainbow"])

        # 태그 기반 희망 요소
        for tag in tags:
            tag_lower = tag.lower()
            if "calm" in tag_lower:
                selected_elements.append("peaceful stream")
            elif "solitude" in tag_lower:
                selected_elements.append("connecting bridge")
            elif "heavy" in tag_lower:
                selected_elements.append("lifting breeze")

        # 기본 희망 요소 (선택된 것이 없을 때)
        if not selected_elements:
            selected_elements = random.sample(self.hope_keywords, 2)

        return selected_elements[:3]  # 최대 3개

    def _create_hope_narrative(self, hope_elements: List[str]) -> str:
        """희망적 서사 구성"""
        if len(hope_elements) == 1:
            return f"touched by {hope_elements[0]}"
        elif len(hope_elements) == 2:
            return f"graced with {hope_elements[0]} and {hope_elements[1]}"
        else:
            return (
                f"embraced by {', '.join(hope_elements[:-1])} and {hope_elements[-1]}"
            )

    def create_guided_question(
        self, guestbook_title: str, emotion_keywords: List[str]
    ) -> str:
        """안내 질문 생성 (ACT 3-4단계 전환)"""
        questions = [
            f"당신이 '{guestbook_title}'라고 이름 붙인 감정 곁에, '따스한 햇살' 한 줌을 더한다면 어떤 모습일까요?",
            f"'{guestbook_title}'의 풍경에 희망의 빛을 비춘다면 무엇이 보일까요?",
            f"이 '{guestbook_title}' 순간에 작은 위로가 찾아온다면 어떤 모습일까요?",
            f"'{guestbook_title}'라는 감정 너머로 새로운 길이 보인다면?",
            f"당신의 '{guestbook_title}' 경험에 부드러운 변화의 바람이 분다면?",
        ]

        # 감정 키워드 기반으로 질문 선택
        if any(kw in emotion_keywords for kw in ["lonely", "isolated", "alone"]):
            return f"'{guestbook_title}'의 고요 속에서 따뜻한 연결의 다리가 놓인다면 어떤 모습일까요?"
        elif any(kw in emotion_keywords for kw in ["dark", "heavy", "overwhelming"]):
            return f"'{guestbook_title}'의 무게감 위로 가벼운 빛의 손길이 닿는다면?"
        else:
            return random.choice(questions)

    def get_prompt_analysis(self, prompt: str) -> Dict[str, Any]:
        """프롬프트 분석 정보 반환"""
        analysis = {
            "word_count": len(prompt.split()),
            "emotion_keywords": [],
            "visual_elements": [],
            "style_indicators": [],
            "hope_elements": [],
        }

        prompt_lower = prompt.lower()

        # 감정 키워드 식별
        emotion_words = [
            "melancholy",
            "peaceful",
            "turbulent",
            "serene",
            "intense",
            "gentle",
        ]
        analysis["emotion_keywords"] = [
            word for word in emotion_words if word in prompt_lower
        ]

        # 시각적 요소 식별
        visual_words = [
            "landscape",
            "scene",
            "light",
            "color",
            "composition",
            "texture",
        ]
        analysis["visual_elements"] = [
            word for word in visual_words if word in prompt_lower
        ]

        # 스타일 지표 식별
        for style, template in self.style_templates.items():
            if any(word in prompt_lower for word in template.split(", ")):
                analysis["style_indicators"].append(style)

        # 희망 요소 식별
        for hope_word in self.hope_keywords:
            if hope_word in prompt_lower:
                analysis["hope_elements"].append(hope_word)

        return analysis
