# src/therapy/prompt_architect.py

from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PromptArchitect:
    """ACT 기반 프롬프트 생성 시스템"""

    def __init__(self):
        self.prompt_engineer = None  # GPT 프롬프트 엔지니어 주입받을 예정
        self.current_diary_text = ""  # 일기 텍스트 저장용

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
        user_id: str = "anonymous",
    ) -> str:
        """감정 반영 프롬프트 생성 (ACT 2단계: Acceptance)"""

        if not self.prompt_engineer:
            raise RuntimeError(
                "PromptEngineer가 주입되지 않았습니다. set_prompt_engineer()를 먼저 호출하세요."
            )

        if not self.current_diary_text:
            raise RuntimeError(
                "일기 텍스트가 설정되지 않았습니다. set_diary_context()를 먼저 호출하세요."
            )

        logger.info(f"Reflection 프롬프트 생성 시작 ({coping_style} 스타일)")

        # GPT 프롬프트 엔지니어로 프롬프트 생성
        result = self.prompt_engineer.enhance_diary_to_prompt(
            diary_text=self.current_diary_text,
            emotion_keywords=emotion_keywords,
            coping_style=coping_style,
            visual_preferences=visual_preferences,
            user_id=user_id,
        )

        if not result["success"]:
            logger.error(
                f"GPT 프롬프트 생성 실패: {result.get('error', 'Unknown error')}"
            )
            raise RuntimeError(f"GPT 프롬프트 생성 실패: {result['error']}")

        generated_prompt = result["prompt"]
        logger.info(
            f"Reflection 프롬프트 생성 완료: {len(generated_prompt)} 문자"
        )

        return generated_prompt

    def create_guided_question(
        self,
        guestbook_title: str,
        emotion_keywords: List[str],
        user_id: str = "anonymous",
    ) -> str:
        """큐레이터 전환 안내 질문 생성"""

        if not self.prompt_engineer:
            raise RuntimeError("PromptEngineer가 주입되지 않았습니다.")

        logger.info(f"큐레이터 전환 안내 질문 생성 시작: {guestbook_title}")

        # GPT에 전환 질문 생성 요청
        result = self.prompt_engineer.generate_transition_guidance(
            guestbook_title=guestbook_title,
            emotion_keywords=emotion_keywords,
            user_id=user_id,
        )

        if not result.get("success", False):
            logger.error(f"전환 안내 생성 실패: {result.get('error')}")
            raise RuntimeError(f"전환 안내 생성 실패: {result.get('error')}")

        logger.info(f"큐레이터 전환 안내 질문 생성 완료: {guestbook_title}")
        return result["content"]

    def get_prompt_analysis(self, prompt: str) -> Dict[str, Any]:
        """프롬프트 분석 정보 반환"""
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
            "fallback_available": False,
            "hardcoded_templates": False,
            "simulation_mode": False,
            "gpt_integration": "complete",
            "language_consistency": "english_only",
            "error_handling": "graceful_failure",
            "handover_status": "completed",
        }
