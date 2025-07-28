# src/therapy/curator_message.py

# ==============================================================================
# 이 파일은 ACT(수용전념치료) 이론의 'Closure' 단계를 담당하며, 큐레이터 메시지 생성을 총괄한다.
# `act_therapy_system`으로부터 요청을 받아, 주입된 `curator_gpt` 모듈을 사용하여
# 사용자 프로필과 갤러리 아이템에 기반한 개인화된 큐레이터 메시지를 생성하도록 지시한다.
# 생성된 메시지는 다시 `act_therapy_system`으로 반환되어 사용자에게 전달된다.
# ==============================================================================

from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CuratorMessageSystem:
    """ACT 기반 큐레이터 메시지 생성 시스템"""

    def __init__(self, user_manager):
        self.user_manager = user_manager
        self.curator_gpt = None  # GPT 큐레이터 주입받을 예정

        logger.info("CuratorMessageSystem 초기화 완료 (GPT)")

    def set_curator_gpt(self, curator_gpt):
        """GPT 큐레이터 주입"""
        self.curator_gpt = curator_gpt
        logger.info("CuratorGPT가 CuratorMessageSystem에 주입되었습니다.")

    def create_personalized_message(
        self, user, gallery_item, message_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """개인화된 큐레이터 메시지 생성"""

        if not self.curator_gpt:
            raise RuntimeError(
                "CuratorGPT가 주입되지 않았습니다. set_curator_gpt()를 먼저 호출하세요."
            )

        logger.info(
            f"큐레이터 메시지 생성 시작: 사용자 {user.user_id}, 아이템 {gallery_item.item_id}"
        )

        # GPT 큐레이터로 개인화된 메시지 생성
        result = self.curator_gpt.generate_personalized_message(
            user_profile=user,
            gallery_item=gallery_item,
            personalization_context=message_context,
        )

        # GPT 생성 결과 검증
        if not result or not isinstance(result, dict):
            logger.error("CuratorGPT에서 유효하지 않은 결과를 반환했습니다.")
            raise RuntimeError("GPT 큐레이터 메시지 생성에 실패했습니다.")

        # 풀백 사용 여부 확인 및 경고
        metadata = result.get("metadata", {})
        if metadata.get("fallback_used", False):
            logger.warning(
                f"큐레이터 메시지에서 풀백 생성이 사용되었습니다: 사용자 {user.user_id}"
            )
            # 풀백 사용시에도 계속 진행 (하지만 로그로 추적)

        # GPT 품질 메트릭 로깅
        if "quality_metrics" in metadata:
            quality = metadata["quality_metrics"]
            logger.info(
                f"GPT 메시지 품질: safety={quality.get('safety_level', 'unknown')}, "
                f"personalization={quality.get('personalization_score', 0):.2f}"
            )

        # 생성 완료 로깅
        logger.info(f"큐레이터 메시지 생성 완료: 사용자 {user.user_id}")

        return result


    def validate_message_quality(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """GPT 생성 메시지 품질 검증"""

        validation_result = {
            "is_valid": True,
            "quality_score": 0.5,  # 기본 점수
            "issues": [],
            "recommendations": [],
        }

        # 기본 구조 확인
        required_fields = ["content", "metadata"]
        for field in required_fields:
            if field not in message:
                validation_result["issues"].append(f"필수 필드가 없습니다: {field}")
                validation_result["is_valid"] = False

        if not validation_result["is_valid"]:
            return validation_result

        # content 세부 검증
        content = message.get("content", {})
        expected_sections = [
            "opening",
            "recognition",
            "personal_note",
            "guidance",
            "closing",
        ]

        missing_sections = [
            section for section in expected_sections if not content.get(section)
        ]
        if missing_sections:
            validation_result["issues"].append(
                f"빠진 섹션: {', '.join(missing_sections)}"
            )
            validation_result["quality_score"] -= 0.2

        # 메시지 길이 확인
        total_length = sum(len(str(section)) for section in content.values())
        if total_length < 100:
            validation_result["issues"].append("메시지가 너무 짧습니다.")
            validation_result["quality_score"] -= 0.1
        elif total_length > 1000:
            validation_result["issues"].append("메시지가 너무 깁니다.")
            validation_result["quality_score"] -= 0.1
        else:
            validation_result["quality_score"] += 0.2

        # 안전성 확인
        metadata = message.get("metadata", {})
        safety_level = metadata.get("quality_metrics", {}).get(
            "safety_level", "unknown"
        )
        if safety_level == "safe":
            validation_result["quality_score"] += 0.3
        elif safety_level == "warning":
            validation_result["quality_score"] -= 0.1
            validation_result["recommendations"].append("안전성 검토가 필요합니다.")
        elif safety_level == "critical":
            validation_result["is_valid"] = False
            validation_result["issues"].append("안전성 문제가 발견되었습니다.")

        # 최종 품질 점수 정규화
        validation_result["quality_score"] = max(
            0.0, min(1.0, validation_result["quality_score"] + 0.5)
        )

        # 전체 유효성 판단
        if validation_result["quality_score"] < 0.3:
            validation_result["is_valid"] = False

        return validation_result

    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 확인"""
        return {
            "curator_gpt_injected": self.curator_gpt is not None,
            "generation_method": "gpt_only",
            "fallback_available": False,
            "hardcoded_templates": False,
            "message_variations_supported": True,
            "quality_validation_enabled": True,
            "handover_status": "completed",
        }


    def get_message_analytics(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """메시지 분석 정보 반환"""
        content = message.get("content", {})

        analytics = {
            "total_word_count": sum(
                len(str(section).split()) for section in content.values()
            ),
            "total_character_count": sum(
                len(str(section)) for section in content.values()
            ),
            "section_count": len([v for v in content.values() if v]),
            "message_type": message.get("message_type", "unknown"),
            "generation_method": message.get("metadata", {}).get(
                "generation_method", "unknown"
            ),
        }

        # 섹션별 분석
        for section_name, section_content in content.items():
            if section_content:
                analytics[f"{section_name}_word_count"] = len(
                    str(section_content).split()
                )
                analytics[f"{section_name}_char_count"] = len(str(section_content))

        return analytics

    def get_user_performance_metrics(self, user_id: str) -> Dict[str, Any]:
        """사용자별 성능 메트릭 반환"""
        # 이 메서드는 실제로는 데이터베이스에서 사용자의 메시지 히스토리를 분석해야 함
        # 현재는 기본 구조만 제공
        return {
            "user_id": user_id,
            "total_messages_generated": 0,
            "average_quality_score": 0.0,
            "successful_generations": 0,
            "failed_generations": 0,
            "emergency_messages": 0,
            "preferred_message_style": "unknown",
            "last_generation_timestamp": None,
        }

    def validate_message_basic_quality(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """메시지 기본 품질 검증"""
        basic_validation = {
            "has_required_structure": False,
            "minimum_length_met": False,
            "has_content": False,
            "overall_valid": False,
        }

        # 기본 구조 확인
        if "content" in message and "metadata" in message:
            basic_validation["has_required_structure"] = True

        # 내용 확인
        content = message.get("content", {})
        if content and any(content.values()):
            basic_validation["has_content"] = True

        # 최소 길이 확인
        total_length = sum(len(str(section)) for section in content.values())
        if total_length >= 50:  # 최소 50자
            basic_validation["minimum_length_met"] = True

        # 전체 유효성
        basic_validation["overall_valid"] = all(
            [
                basic_validation["has_required_structure"],
                basic_validation["minimum_length_met"],
                basic_validation["has_content"],
            ]
        )

        return basic_validation
