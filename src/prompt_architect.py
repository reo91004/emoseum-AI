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

        # 큐레이터 메시지로의 전환 안내 문구들
        self.curator_transition_phrases = [
            "이제 당신의 여정을 함께 돌아보며, 작은 용기에 박수를 보내드리고 싶습니다.",
            "이 감정의 탐험에서 보여준 당신의 진정성이 아름다웠습니다.",
            "방금 완성한 이 이야기에 대해 함께 이야기해보시겠어요?",
            "당신이 방금 마주한 감정에 대해 작은 격려를 전하고 싶습니다.",
            "이번 여정에서 발견한 것들에 대해 함께 나누고 싶은 마음입니다.",
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

    def create_guided_question(
        self, guestbook_title: str, emotion_keywords: List[str]
    ) -> str:
        """큐레이터 메시지로의 전환을 위한 안내 질문 생성"""

        # 기본 전환 문구 선택
        transition_phrase = random.choice(self.curator_transition_phrases)

        # 방명록 제목 기반 맞춤 안내
        title_based_guidance = self._create_title_based_guidance(guestbook_title)

        # 감정 키워드 기반 격려
        emotion_encouragement = self._create_emotion_encouragement(emotion_keywords)

        # 큐레이터 메시지 예고
        curator_preview = self._create_curator_preview(
            guestbook_title, emotion_keywords
        )

        # 최종 안내 질문 구성
        guided_question = f"""
        {transition_phrase}

        {title_based_guidance}

        {emotion_encouragement}

        {curator_preview}
        """.strip()

        logger.info(f"큐레이터 전환 안내 질문이 생성되었습니다: {guestbook_title}")
        return guided_question

    def _create_title_based_guidance(self, title: str) -> str:
        """방명록 제목 기반 안내 생성"""
        if not title:
            return "당신이 이 감정에 부여한 의미가 특별합니다."

        # 제목의 톤에 따른 안내 메시지
        title_lower = title.lower()

        if any(word in title_lower for word in ["dark", "heavy", "storm", "cold"]):
            return f"'{title}'라는 이름 속에서도 당신만의 깊이 있는 관점이 느껴집니다."
        elif any(word in title_lower for word in ["light", "warm", "gentle", "soft"]):
            return f"'{title}'에서 당신의 섬세한 감수성이 빛납니다."
        elif any(word in title_lower for word in ["quiet", "still", "peaceful"]):
            return f"'{title}'라는 표현에서 내면의 고요함을 찾는 지혜가 보입니다."
        else:
            return f"'{title}'라고 명명하신 선택에서 당신만의 독특한 시각이 돋보입니다."

    def _create_emotion_encouragement(self, keywords: List[str]) -> str:
        """감정 키워드 기반 격려 메시지"""
        if not keywords:
            return "이런 솔직한 감정 탐구가 정말 소중합니다."

        primary_emotion = keywords[0]

        encouragement_map = {
            "슬픔": "슬픔을 온전히 받아들이신 용기가 아름답습니다.",
            "기쁨": "이 기쁨을 우리와 나눠주셔서 감사합니다.",
            "화남": "분노를 건설적으로 표현해내신 모습이 인상적입니다.",
            "불안": "불안을 솔직하게 마주하신 용기에 박수를 보냅니다.",
            "평온": "내면의 평화를 찾아가는 여정이 지혜롭습니다.",
            "외로움": "외로움을 용기 있게 인정하신 정직함이 대단합니다.",
            "피곤": "지친 마음을 돌보려는 당신의 배려가 아름답습니다.",
        }

        return encouragement_map.get(
            primary_emotion, "이런 진솔한 감정 표현이 정말 의미 있습니다."
        )

    def _create_curator_preview(self, title: str, keywords: List[str]) -> str:
        """큐레이터 메시지 예고"""

        preview_messages = [
            "잠시 후, 이 여정을 마무리하는 특별한 메시지를 준비해드리겠습니다.",
            "당신의 이야기에 대한 작은 감사와 격려를 전해드리고 싶습니다.",
            "이제 이 경험을 소중히 간직할 수 있도록 마무리해드리겠습니다.",
            "당신이 보여준 용기에 대한 인정과 응원을 준비했습니다.",
            "이 특별한 순간을 기념하는 메시지를 전달해드리겠습니다.",
        ]

        return random.choice(preview_messages)

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

    def get_prompt_analysis(self, prompt: str) -> Dict[str, Any]:
        """프롬프트 분석 정보 반환"""
        analysis = {
            "word_count": len(prompt.split()),
            "emotion_keywords": [],
            "visual_elements": [],
            "style_indicators": [],
            "acceptance_elements": [],
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

        # 수용 요소 식별
        acceptance_words = [
            "accepting",
            "non-judgmental",
            "gentle",
            "safe",
            "protective",
            "balanced",
            "mindful",
        ]
        analysis["acceptance_elements"] = [
            word for word in acceptance_words if word in prompt_lower
        ]

        return analysis

    def create_personalized_transition_message(
        self, user_coping_style: str, emotion_keywords: List[str], guestbook_title: str
    ) -> str:
        """개인화된 큐레이터 전환 메시지 생성"""

        # 대처 스타일별 기본 톤 설정
        if user_coping_style == "avoidant":
            base_tone = "부드럽고 따뜻한 격려를 준비했습니다."
        elif user_coping_style == "confrontational":
            base_tone = "당신의 용기 있는 도전에 대한 진심 어린 응원을 전하고 싶습니다."
        else:  # balanced
            base_tone = "균형 잡힌 당신의 접근에 대한 인정과 격려를 드리고 싶습니다."

        # 감정과 제목을 고려한 맞춤 메시지
        personalized_elements = [
            self._create_title_based_guidance(guestbook_title),
            self._create_emotion_encouragement(emotion_keywords),
            base_tone,
        ]

        return " ".join(personalized_elements)

    def validate_prompt_safety(self, prompt: str) -> Dict[str, Any]:
        """프롬프트 안전성 검증"""

        safety_issues = []

        # 너무 부정적인 키워드 체크
        extreme_negative_words = [
            "suicide",
            "death",
            "kill",
            "harm",
            "violence",
            "blood",
            "weapon",
        ]

        for word in extreme_negative_words:
            if word in prompt.lower():
                safety_issues.append(f"극단적 부정 키워드 발견: {word}")

        # 부적절한 성적 내용 체크
        inappropriate_words = ["sexual", "nude", "naked", "porn", "explicit"]

        for word in inappropriate_words:
            if word in prompt.lower():
                safety_issues.append(f"부적절한 내용 발견: {word}")

        return {
            "is_safe": len(safety_issues) == 0,
            "safety_issues": safety_issues,
            "recommendation": (
                "안전한 프롬프트입니다."
                if len(safety_issues) == 0
                else "프롬프트 수정이 필요합니다."
            ),
        }
