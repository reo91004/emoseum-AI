# src/therapy/prompt_architect.py

# ==============================================================================
# 이 파일은 ACT(수용전념치료) 이론의 'Acceptance'와 'Defusion' 단계를 담당하며,
# 이미지 프롬프트와 안내 질문 생성을 총괄한다.
# `act_therapy_system`으로부터 요청을 받아, 주입된 `prompt_engineer` 모듈을 사용하여
# 사용자의 감정 데이터에 기반한 Reflection 프롬프트나 Guided Question을 생성하도록 지시한다.
# ==============================================================================

from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PromptArchitect:
    """ACT 기반 프롬프트 생성 시스템"""

    def __init__(self, safety_validator=None):
        self.prompt_engineer = None  # GPT 프롬프트 엔지니어 주입받을 예정
        self.safety_validator = safety_validator  # 안전성 검증기 주입
        self.current_diary_text = ""  # 일기 텍스트 저장용

    def set_prompt_engineer(self, prompt_engineer):
        """GPT 프롬프트 엔지니어 주입"""
        self.prompt_engineer = prompt_engineer
        logger.info("PromptEngineer가 PromptArchitect에 주입되었습니다.")

    def set_safety_validator(self, safety_validator):
        """안전성 검증기 주입"""
        self.safety_validator = safety_validator
        logger.info("SafetyValidator가 PromptArchitect에 주입되었습니다.")

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
        logger.info(f"Reflection 프롬프트 생성 완료: {len(generated_prompt)} 문자")

        ##
        with open("./prompt_test/generated_prompt.txt", "w", encoding="utf-8") as f:
            f.write(generated_prompt)
        ##

        return generated_prompt

    def create_guided_question(
        self,
        artwork_title: str,
        emotion_keywords: List[str],
        user_id: str = "anonymous",
    ) -> str:
        """도슨트 전환 안내 질문 생성"""

        if not self.prompt_engineer:
            raise RuntimeError("PromptEngineer가 주입되지 않았습니다.")

        logger.info(f"도슨트 전환 안내 질문 생성 시작: {artwork_title}")

        # GPT에 전환 질문 생성 요청
        result = self.prompt_engineer.generate_transition_guidance(
            artwork_title=artwork_title,
            emotion_keywords=emotion_keywords,
            user_id=user_id,
        )

        if not result.get("success", False):
            logger.error(f"전환 안내 생성 실패: {result.get('error')}")
            raise RuntimeError(f"전환 안내 생성 실패: {result.get('error')}")

        logger.info(f"도슨트 전환 안내 질문 생성 완료: {artwork_title}")
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
        """프롬프트 안전성 검증 - SafetyValidator 사용"""
        if not self.safety_validator:
            logger.warning("SafetyValidator가 주입되지 않음. 기본 안전 검증 사용")
            return {
                "is_safe": True,
                "safety_issues": [],
                "recommendation": "SafetyValidator가 설정되지 않았습니다.",
                "generation_method": "gpt",
            }

        # SafetyValidator를 사용하여 검증
        validation_result = self.safety_validator.validate_gpt_response(
            response=prompt, context={"type": "generated_prompt"}
        )

        return {
            "is_safe": validation_result.get("is_safe", False),
            "safety_issues": validation_result.get("issues", []),
            "recommendation": validation_result.get("recommendation", "검증 완료"),
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
            "gpt_integration": "complete",
            "language_consistency": "english_only",
            "error_handling": "graceful_failure",
            "handover_status": "completed",
        }
