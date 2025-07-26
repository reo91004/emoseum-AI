# src/therapy/prompt_architect.py

from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PromptArchitect:
    """ACT 기반 프롬프트 생성 시스템"""

    def __init__(self):
        self.prompt_engineer = None  # GPT 프롬프트 엔지니어 주입받을 예정
        self.current_diary_text = ""  # 일기 텍스트 저장용

        # 큐레이터 메시지로의 전환 안내 문구들 (GPT 독립적)
        self.curator_transition_phrases = [
            "이제 당신의 여정을 함께 돌아보며, 작은 용기에 박수를 보내드리고 싶습니다.",
            "이 감정의 탐험에서 보여준 당신의 진정성이 아름다웠습니다.",
            "방금 완성한 이 이야기에 대해 함께 이야기해보시겠어요?",
            "당신이 방금 마주한 감정에 대해 작은 격려를 전하고 싶습니다.",
            "이번 여정에서 발견한 것들에 대해 함께 나누고 싶은 마음입니다.",
        ]

    def set_prompt_engineer(self, prompt_engineer):
        """GPT 프롬프트 엔지니어 주입"""
        self.prompt_engineer = prompt_engineer
        logger.info("PromptEngineer가 PromptArchitect에 주입되었습니다.")

    def set_diary_context(self, diary_text: str):
        """일기 텍스트 컨텍스트 설정"""
        self.current_diary_text = diary_text
        logger.debug(f"일기 컨텍스트 설정됨: {len(diary_text)} 문자")

    def create_reflection_prompt(
        self,
        emotion_keywords: List[str],
        vad_scores: Tuple[float, float, float],
        coping_style: str,
        visual_preferences: Dict[str, Any],
    ) -> str:
        """GPT 기반 감정 반영 프롬프트 생성 (ACT 2단계: Acceptance)

        기존 하드코딩된 템플릿 시스템을 완전히 제거하고
        PromptEngineer의 GPT 기반 생성으로 대체
        """

        if not self.prompt_engineer:
            raise RuntimeError(
                "PromptEngineer가 주입되지 않았습니다. set_prompt_engineer()를 먼저 호출하세요."
            )

        if not self.current_diary_text:
            raise RuntimeError(
                "일기 텍스트가 설정되지 않았습니다. set_diary_context()를 먼저 호출하세요."
            )

        logger.info(f"GPT 기반 Reflection 프롬프트 생성 시작 ({coping_style} 스타일)")

        # GPT 프롬프트 엔지니어로 프롬프트 생성
        result = self.prompt_engineer.enhance_diary_to_prompt(
            diary_text=self.current_diary_text,
            emotion_keywords=emotion_keywords,
            coping_style=coping_style,
            visual_preferences=visual_preferences,
        )

        if not result["success"]:
            logger.error(
                f"GPT 프롬프트 생성 실패: {result.get('error', 'Unknown error')}"
            )
            raise RuntimeError(f"GPT 프롬프트 생성 실패: {result['error']}")

        generated_prompt = result["prompt"]
        logger.info(
            f"GPT 기반 Reflection 프롬프트 생성 완료: {len(generated_prompt)} 문자"
        )

        return generated_prompt

    def create_guided_question(
        self, guestbook_title: str, emotion_keywords: List[str]
    ) -> str:
        """큐레이터 메시지로의 전환을 위한 안내 질문 생성

        이 기능은 GPT와 독립적으로 유지 (단순 전환 안내)
        """
        import random

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

        logger.info(f"큐레이터 전환 안내 질문 생성 완료: {guestbook_title}")
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
        import random

        preview_messages = [
            "잠시 후, 이 여정을 마무리하는 특별한 메시지를 준비해드리겠습니다.",
            "당신의 이야기에 대한 작은 감사와 격려를 전해드리고 싶습니다.",
            "이제 이 경험을 소중히 간직할 수 있도록 마무리해드리겠습니다.",
            "당신이 보여준 용기에 대한 인정과 응원을 준비했습니다.",
            "이 특별한 순간을 기념하는 메시지를 전달해드리겠습니다.",
        ]

        return random.choice(preview_messages)

    def get_prompt_analysis(self, prompt: str) -> Dict[str, Any]:
        """프롬프트 분석 정보 반환

        GPT 생성된 프롬프트에 대한 메타 분석
        """
        analysis = {
            "word_count": len(prompt.split()),
            "character_count": len(prompt),
            "generation_method": "gpt",
            "prompt_type": "reflection",
            "estimated_quality": "high" if len(prompt) > 50 else "low",
        }

        prompt_lower = prompt.lower()

        # GPT 생성 품질 지표
        quality_indicators = [
            "detailed",
            "artistic",
            "style",
            "emotion",
            "atmosphere",
            "composition",
            "lighting",
            "color",
            "mood",
            "feeling",
        ]

        quality_score = sum(
            1 for indicator in quality_indicators if indicator in prompt_lower
        )
        analysis["quality_indicators_found"] = quality_score
        analysis["estimated_effectiveness"] = (
            "high" if quality_score >= 5 else "medium" if quality_score >= 3 else "low"
        )

        return analysis

    def validate_prompt_safety(self, prompt: str) -> Dict[str, Any]:
        """프롬프트 안전성 검증"""
        safety_issues = []

        # 극단적 부정 키워드 체크
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
            "generation_method": "gpt",
        }

    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 확인"""
        return {
            "prompt_engineer_injected": self.prompt_engineer is not None,
            "diary_context_set": bool(self.current_diary_text),
            "generation_method": "gpt_only",
            "fallback_available": False,  # 풀백 시스템 완전 제거
            "hardcoded_templates": False,  # 하드코딩 템플릿 완전 제거
        }
