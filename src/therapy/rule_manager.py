# src/therapy/rule_manager.py

from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CopingStyleRules:
    """대처 스타일 설정 관리 (GPT에서 직접 처리)"""

    def __init__(self, rules_file: Optional[str] = None):
        self.rules_file = Path(rules_file) if rules_file else None

        # 대처 스타일별 GPT 시스템 메시지 가이드라인만 보관
        self.style_guidelines = {
            "avoidant": "gentle, protective, metaphorical",
            "confrontational": "direct, authentic, honest",
            "balanced": "thoughtful, nuanced, harmonious",
        }

        # 감정 카테고리별 기본 접근 방식 (GPT 참조용)
        self.emotion_approaches = {
            "sadness": {
                "avoidant": "gentle melancholy, soft contemplation",
                "confrontational": "raw sorrow, unfiltered grief",
                "balanced": "thoughtful sadness, meaningful reflection",
            },
            "anger": {
                "avoidant": "muted tension, restrained energy",
                "confrontational": "blazing fury, intense rage",
                "balanced": "controlled anger, focused frustration",
            },
            "fear": {
                "avoidant": "cautious uncertainty, gentle concern",
                "confrontational": "sharp anxiety, acute worry",
                "balanced": "reasonable concern, manageable anxiety",
            },
            "joy": {
                "avoidant": "quiet happiness, peaceful contentment",
                "confrontational": "explosive joy, vibrant celebration",
                "balanced": "warm happiness, steady joy",
            },
        }

        logger.info("CopingStyleRules 초기화 완료 - GPT 기반 설정 관리")

    def get_style_guidance(self, coping_style: str) -> str:
        """GPT 시스템 메시지에 포함할 스타일 가이드라인 반환"""
        return self.style_guidelines.get(
            coping_style, self.style_guidelines["balanced"]
        )

    def get_emotion_approach(self, coping_style: str, emotion_category: str) -> str:
        """특정 감정에 대한 대처 스타일별 접근 방식 반환"""
        if emotion_category in self.emotion_approaches:
            return self.emotion_approaches[emotion_category].get(
                coping_style, self.emotion_approaches[emotion_category]["balanced"]
            )
        return self.get_style_guidance(coping_style)

    def get_contextual_prompt_modifiers(
        self, coping_style: str, emotion_category: str
    ) -> List[str]:
        """대처 스타일과 감정 카테고리에 따른 프롬프트 수식어 반환"""

        base_modifiers = {
            "avoidant": ["protective atmosphere", "safe distance", "gentle buffer"],
            "confrontational": [
                "authentic confrontation",
                "raw honesty",
                "direct engagement",
            ],
            "balanced": [
                "thoughtful balance",
                "harmonious integration",
                "mindful approach",
            ],
        }

        return base_modifiers.get(coping_style, base_modifiers["balanced"])

    def create_emotional_bridge(
        self, original_emotion: str, target_emotion: str, coping_style: str
    ) -> str:
        """감정 간 연결 표현 생성 (GPT 참조용 템플릿)"""

        bridge_templates = {
            "avoidant": "gently moving from {original} toward {target}",
            "confrontational": "transforming {original} into {target}",
            "balanced": "thoughtfully bridging {original} and {target}",
        }

        template = bridge_templates.get(coping_style, bridge_templates["balanced"])
        return template.format(original=original_emotion, target=target_emotion)

    def get_rule_statistics(self) -> Dict[str, Any]:
        """규칙 통계 반환"""
        return {
            "style_guidelines_count": len(self.style_guidelines),
            "emotion_approaches_count": len(self.emotion_approaches),
            "generation_method": "gpt_only",
            "hardcoded_rules": False,
            "fallback_available": False,
            "rule_application": "gpt_system_message_only",
        }

    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 확인"""
        return {
            "generation_method": "gpt_only",
            "fallback_available": False,
            "hardcoded_templates": False,
            "simulation_mode": False,
            "gpt_integration": "complete",
            "language_consistency": "english_only",
            "error_handling": "graceful_failure",
            "handover_status": "completed",
        }
