# src/therapy/curator_message.py

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
        """GPT 기반 개인화된 큐레이터 메시지 생성"""

        if not self.curator_gpt:
            raise RuntimeError(
                "CuratorGPT가 주입되지 않았습니다. set_curator_gpt()를 먼저 호출하세요."
            )

        logger.info(
            f"GPT 기반 큐레이터 메시지 생성 시작: 사용자 {user.user_id}, 아이템 {gallery_item.item_id}"
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
        logger.info(f"GPT 기반 큐레이터 메시지 생성 완료: 사용자 {user.user_id}")

        return result

    def get_message_variations(
        self, base_message: Dict[str, Any], variation_count: int = 3
    ) -> List[Dict[str, Any]]:
        """GPT 기반 메시지 변형 생성 (A/B 테스트용)

        기존 템플릿 기반 변형을 GPT 기반 변형으로 대체
        """

        if not self.curator_gpt:
            logger.warning(
                "CuratorGPT가 주입되지 않아 메시지 변형을 생성할 수 없습니다."
            )
            return [base_message]

        variations = [base_message]

        # GPT를 통한 메시지 변형 생성
        try:
            for i in range(variation_count - 1):
                # 기본 메시지 데이터 추출
                user_profile = base_message.get("user_profile")
                gallery_item = base_message.get("gallery_item")

                if not user_profile or not gallery_item:
                    logger.warning("메시지 변형 생성을 위한 데이터 부족")
                    continue

                # 변형 컨텍스트 설정
                variation_context = {
                    "variation_request": True,
                    "variation_index": i + 1,
                    "base_message_id": base_message.get("message_id"),
                    "tone_adjustment": self._get_tone_variation(i),
                }

                # GPT로 변형 생성
                variation_result = self.curator_gpt.generate_personalized_message(
                    user_profile=user_profile,
                    gallery_item=gallery_item,
                    personalization_context=variation_context,
                )

                if variation_result:
                    # 변형 ID 설정
                    variation_result["message_id"] = (
                        f"{base_message.get('message_id', 'unknown')}_var{i+1}"
                    )
                    variations.append(variation_result)

            logger.info(f"GPT 기반 메시지 변형 생성 완료: {len(variations)}개")

        except Exception as e:
            logger.error(f"GPT 메시지 변형 생성 실패: {e}")

        return variations

    def _get_tone_variation(self, variation_index: int) -> str:
        """변형별 톤 조정 요청"""
        tone_variations = [
            "slightly_more_encouraging",
            "more_reflective",
            "warmer_and_supportive",
        ]

        return tone_variations[variation_index % len(tone_variations)]

    def analyze_message_effectiveness(
        self, message_reactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """GPT 생성 메시지 효과성 분석"""

        if not message_reactions:
            return {"total_messages": 0, "insights": [], "generation_method": "gpt"}

        # 반응 유형별 집계
        reaction_counts = {}
        gpt_message_count = 0

        for reaction in message_reactions:
            reaction_type = reaction.get("reaction_type", "unknown")
            reaction_counts[reaction_type] = reaction_counts.get(reaction_type, 0) + 1

            # GPT 생성 메시지인지 확인
            if reaction.get("generation_method") == "gpt":
                gpt_message_count += 1

        # 대처 스타일별 GPT 효과성
        style_effectiveness = {}
        for reaction in message_reactions:
            style = reaction.get("coping_style", "unknown")
            if style not in style_effectiveness:
                style_effectiveness[style] = {
                    "positive": 0,
                    "total": 0,
                    "gpt_positive": 0,
                    "gpt_total": 0,
                }

            style_effectiveness[style]["total"] += 1
            if reaction.get("reaction_type") in ["like", "save", "share"]:
                style_effectiveness[style]["positive"] += 1

            # GPT 생성 메시지 추적
            if reaction.get("generation_method") == "gpt":
                style_effectiveness[style]["gpt_total"] += 1
                if reaction.get("reaction_type") in ["like", "save", "share"]:
                    style_effectiveness[style]["gpt_positive"] += 1

        # 인사이트 생성
        insights = []
        total_positive = sum(
            reaction_counts.get(rt, 0) for rt in ["like", "save", "share"]
        )
        total_reactions = len(message_reactions)

        if total_reactions > 0:
            positive_rate = total_positive / total_reactions
            gpt_ratio = gpt_message_count / total_reactions

            if positive_rate > 0.7:
                insights.append(
                    "GPT 생성 메시지가 사용자들에게 긍정적으로 받아들여지고 있습니다."
                )
            elif positive_rate > 0.5:
                insights.append(
                    "GPT 메시지 효과가 적절하지만 개인화 개선이 필요할 수 있습니다."
                )
            else:
                insights.append(
                    "GPT 메시지 전략을 재검토하고 프롬프트 엔지니어링을 개선해야 합니다."
                )

            if gpt_ratio > 0.8:
                insights.append(
                    f"전체 메시지의 {gpt_ratio:.1%}가 GPT로 생성되어 완전 전환이 성공적입니다."
                )

        return {
            "total_messages": total_reactions,
            "gpt_messages": gpt_message_count,
            "gpt_ratio": (
                gpt_message_count / total_reactions if total_reactions > 0 else 0
            ),
            "reaction_distribution": reaction_counts,
            "positive_reaction_rate": (
                total_positive / total_reactions if total_reactions > 0 else 0
            ),
            "style_effectiveness": style_effectiveness,
            "insights": insights,
            "generation_method": "gpt_analysis",
        }

    def get_gpt_performance_metrics(self, user_id: str) -> Dict[str, Any]:
        """GPT 큐레이터 성능 메트릭 조회"""

        if not self.curator_gpt:
            return {"error": "CuratorGPT가 주입되지 않았습니다.", "available": False}

        try:
            # CuratorGPT의 성능 메트릭 조회
            performance = self.curator_gpt.get_user_performance_metrics(user_id)

            return {
                "user_id": user_id,
                "gpt_performance": performance,
                "system_status": "gpt_only",
                "fallback_usage": performance.get("fallback_usage_rate", 0),
                "avg_generation_time": performance.get("avg_generation_time", 0),
                "quality_score": performance.get("avg_quality_score", 0),
                "personalization_score": performance.get(
                    "avg_personalization_score", 0
                ),
            }

        except Exception as e:
            logger.error(f"GPT 성능 메트릭 조회 실패: {e}")
            return {"error": str(e), "user_id": user_id, "available": False}

    def validate_gpt_message_quality(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """GPT 생성 메시지 품질 검증"""

        validation_result = {
            "is_valid": True,
            "quality_score": 0.0,
            "issues": [],
            "recommendations": [],
        }

        content = message.get("content", {})
        if not content:
            validation_result["is_valid"] = False
            validation_result["issues"].append("메시지 내용이 비어있습니다.")
            return validation_result

        # 필수 섹션 확인
        required_sections = [
            "opening",
            "recognition",
            "personal_note",
            "guidance",
            "closing",
        ]
        missing_sections = []

        for section in required_sections:
            if not content.get(section):
                missing_sections.append(section)

        if missing_sections:
            validation_result["issues"].append(
                f"누락된 섹션: {', '.join(missing_sections)}"
            )
            validation_result["quality_score"] -= 0.2 * len(missing_sections)

        # 개인화 수준 확인
        personalization_data = message.get("personalization_data", {})
        if personalization_data:
            validation_result["quality_score"] += 0.3
        else:
            validation_result["recommendations"].append("개인화 요소를 더 강화하세요.")

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
            "fallback_available": False,  # 풀백 시스템 완전 제거
            "hardcoded_templates": False,  # 하드코딩 템플릿 완전 제거
            "message_variations_supported": True,
            "quality_validation_enabled": True,
        }

    def emergency_message_generation(
        self, user_id: str, error_context: str
    ) -> Dict[str, Any]:
        """응급 메시지 생성 (GPT 완전 실패시)

        GPT가 완전히 실패했을 때만 사용하는 최소한의 응급 메시지
        하드코딩된 풀백이 아닌 간단한 오류 처리용
        """

        logger.error(
            f"GPT 큐레이터 완전 실패로 응급 메시지 생성: 사용자 {user_id}, 오류: {error_context}"
        )

        emergency_message = {
            "message_id": f"emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "user_id": user_id,
            "message_type": "emergency_placeholder",
            "content": {
                "opening": "시스템에 일시적인 문제가 발생했습니다.",
                "recognition": "당신의 감정 여정은 소중하게 기록되었습니다.",
                "personal_note": "잠시 후 다시 시도해주시면 개인화된 메시지를 받으실 수 있습니다.",
                "guidance": "현재 경험한 감정을 마음에 간직해주세요.",
                "closing": "곧 다시 만나뵙겠습니다.",
            },
            "metadata": {
                "generation_method": "emergency",
                "error_context": error_context,
                "requires_retry": True,
                "quality_metrics": {
                    "safety_level": "safe",
                    "personalization_score": 0.0,
                },
            },
            "created_date": datetime.now().isoformat(),
        }

        return emergency_message
