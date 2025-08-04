# src/utils/safety_validator.py

# ==============================================================================
# 이 파일은 시스템의 치료적 안전성을 보장하기 위한 핵심 검증 도구이다.
# `config/safety_rules.yaml`에 정의된 규칙(유해 키워드, 부적절한 응답 패턴 등)을 기반으로
# 사용자의 입력과 GPT가 생성한 모든 콘텐츠(프롬프트, 도슨트 메시지)를 검증한다.
# 검증 결과에 따라 콘텐츠를 그대로 사용하거나, 수정하거나, 또는 비상 대응 절차를 실행한다.
# ==============================================================================

import re
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging
import yaml

logger = logging.getLogger(__name__)


class SafetyValidator:
    """치료적 안전성 검증 시스템"""

    def __init__(self, safety_rules_path: Optional[str] = None):
        self.safety_rules_path = Path(safety_rules_path) if safety_rules_path else None

        # YAML 파일에서 안전성 규칙 로드
        if self.safety_rules_path and self.safety_rules_path.exists():
            self.default_safety_rules = self._load_safety_rules_from_yaml()
            logger.info(
                f"안전성 규칙을 YAML 파일에서 로드했습니다: {self.safety_rules_path}"
            )
        else:
            # YAML 파일 로드 실패 시 에러 발생
            if self.safety_rules_path:
                logger.error(
                    f"안전성 규칙 파일을 찾을 수 없습니다: {self.safety_rules_path}"
                )
                raise FileNotFoundError(
                    f"Safety rules file not found: {self.safety_rules_path}"
                )
            else:
                logger.error("안전성 규칙 파일 경로가 제공되지 않았습니다")
                raise ValueError("Safety rules file path is required")

        # 치료적 품질 지표와 전문가 상담 권유 조건은 YAML에서 로드됨

        # YAML 파일에서 나머지 설정도 로드
        self._load_additional_settings_from_yaml()

    def _load_safety_rules_from_yaml(self) -> Dict[str, Any]:
        """YAML 파일에서 안전성 규칙 로드"""
        try:
            with open(self.safety_rules_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)

            # 필요한 섹션들 추출
            safety_rules = {}

            for section in [
                "critical_keywords",
                "warning_keywords",
                "inappropriate_responses",
            ]:
                if section in yaml_data:
                    safety_rules[section] = yaml_data[section]

            return safety_rules

        except Exception as e:
            logger.error(f"YAML 안전성 규칙 로드 실패: {e}")
            raise

    def _load_additional_settings_from_yaml(self):
        """YAML 파일에서 추가 설정 로드"""
        try:
            with open(self.safety_rules_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)

            # therapeutic_quality_indicators 로드
            if "therapeutic_quality" in yaml_data:
                therapeutic = yaml_data["therapeutic_quality"]
                self.therapeutic_quality_indicators = {}

                # positive_indicators 처리
                if "positive_indicators" in therapeutic:
                    self.therapeutic_quality_indicators["positive"] = []
                    for subcategory in therapeutic["positive_indicators"].values():
                        self.therapeutic_quality_indicators["positive"].extend(
                            subcategory
                        )

                # empathetic_indicators 처리
                if "empathetic_indicators" in therapeutic:
                    self.therapeutic_quality_indicators["empathetic"] = []
                    for subcategory in therapeutic["empathetic_indicators"].values():
                        self.therapeutic_quality_indicators["empathetic"].extend(
                            subcategory
                        )

                # empowerment는 positive_indicators 안의 하위 카테고리로 별도 처리
                if (
                    "positive_indicators" in therapeutic
                    and "empowerment" in therapeutic["positive_indicators"]
                ):
                    self.therapeutic_quality_indicators["empowering"] = therapeutic[
                        "positive_indicators"
                    ]["empowerment"]

                # 기본값 설정 (키가 없을 경우)
                for key in ["positive", "empathetic", "empowering"]:
                    if key not in self.therapeutic_quality_indicators:
                        self.therapeutic_quality_indicators[key] = []

            # professional_referral_triggers 로드
            if "professional_referral" in yaml_data:
                referral_data = yaml_data["professional_referral"]
                self.professional_referral_triggers = {}

                for level in ["immediate", "urgent", "recommended"]:
                    if level in referral_data and "keywords" in referral_data[level]:
                        self.professional_referral_triggers[level] = referral_data[
                            level
                        ]["keywords"]

                # 기본값 설정 (키가 없을 경우)
                for level in ["immediate", "urgent", "recommended"]:
                    if level not in self.professional_referral_triggers:
                        self.professional_referral_triggers[level] = []

        except Exception as e:
            logger.error(f"추가 YAML 설정 로드 실패: {e}")
            raise

    def validate_gpt_response(
        self, response: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """GPT 응답 안전성 검증"""

        validation_result = {
            "is_safe": True,
            "safety_level": "safe",
            "issues": [],
            "referral_recommendation": None,
            "content_warnings": [],
            "therapeutic_quality": "unknown",
            "recommendations": [],
        }

        if not response or not response.strip():
            validation_result["is_safe"] = False
            validation_result["issues"].append("Empty or invalid response")
            return validation_result

        try:
            # 1. 치명적 키워드 검사
            critical_issues = self._check_critical_keywords(response)
            if critical_issues:
                validation_result["is_safe"] = False
                validation_result["safety_level"] = "critical"
                validation_result["issues"].extend(critical_issues)
                validation_result["referral_recommendation"] = "immediate"

            # 2. 경고 레벨 키워드 검사
            warning_issues = self._check_warning_keywords(response)
            if warning_issues and validation_result["safety_level"] == "safe":
                validation_result["safety_level"] = "warning"
                validation_result["content_warnings"].extend(warning_issues)

                # 경고 수준에 따른 상담 권유
                if len(warning_issues) >= 3:
                    validation_result["referral_recommendation"] = "urgent"
                elif len(warning_issues) >= 2:
                    validation_result["referral_recommendation"] = "recommended"

            # 3. 부적절한 응답 패턴 검사
            inappropriate_patterns = self._check_inappropriate_responses(response)
            if inappropriate_patterns:
                validation_result["issues"].extend(inappropriate_patterns)
                if validation_result["safety_level"] == "safe":
                    validation_result["safety_level"] = "warning"

            # 4. 치료적 품질 평가
            therapeutic_quality = self._assess_therapeutic_quality(response)
            validation_result["therapeutic_quality"] = therapeutic_quality["level"]

            if therapeutic_quality["level"] == "poor":
                validation_result["issues"].append("Low therapeutic quality")
                validation_result["recommendations"].append(
                    "Consider regenerating response"
                )

            # 5. 컨텍스트 기반 추가 검증
            context_issues = self._validate_context_appropriateness(response, context)
            if context_issues:
                validation_result["issues"].extend(context_issues)

            # 6. 전문가 상담 권유 필요성 재평가
            if not validation_result["referral_recommendation"]:
                referral_need = self._assess_professional_referral_need(
                    response, context
                )
                validation_result["referral_recommendation"] = referral_need

            # 7. 최종 안전성 판단
            if (
                validation_result["safety_level"] == "critical"
                or len(validation_result["issues"]) >= 3
            ):
                validation_result["is_safe"] = False

            # 8. 권장사항 생성
            validation_result["recommendations"].extend(
                self._generate_safety_recommendations(validation_result)
            )

            logger.info(
                f"안전성 검증 완료: 레벨={validation_result['safety_level']}, "
                f"이슈={len(validation_result['issues'])}개, "
                f"상담권유={validation_result['referral_recommendation']}"
            )

            return validation_result

        except Exception as e:
            logger.error(f"안전성 검증 중 오류 발생: {e}")
            return {
                "is_safe": False,
                "safety_level": "error",
                "issues": [f"Validation error: {str(e)}"],
                "referral_recommendation": "recommended",
                "content_warnings": [],
                "therapeutic_quality": "unknown",
                "recommendations": ["Manual review required due to validation error"],
            }

    def _check_critical_keywords(self, text: str) -> List[str]:
        """치명적 키워드 검사"""

        issues = []
        text_lower = text.lower()

        for category, keywords in self.default_safety_rules[
            "critical_keywords"
        ].items():
            found_keywords = []
            for keyword in keywords:
                if keyword in text_lower:
                    found_keywords.append(keyword)

            if found_keywords:
                issues.append(
                    f"Critical {category} indicators found: {', '.join(found_keywords)}"
                )

        return issues

    def _check_warning_keywords(self, text: str) -> List[str]:
        """경고 레벨 키워드 검사"""

        warnings = []
        text_lower = text.lower()

        for category, keywords in self.default_safety_rules["warning_keywords"].items():
            found_keywords = []
            for keyword in keywords:
                if keyword in text_lower:
                    found_keywords.append(keyword)

            if found_keywords:
                warnings.append(f"Warning {category}: {', '.join(found_keywords)}")

        return warnings

    def _check_inappropriate_responses(self, text: str) -> List[str]:
        """부적절한 응답 패턴 검사"""

        issues = []
        text_lower = text.lower()

        for category, patterns in self.default_safety_rules[
            "inappropriate_responses"
        ].items():
            found_patterns = []
            for pattern in patterns:
                if pattern in text_lower:
                    found_patterns.append(pattern)

            if found_patterns:
                issues.append(f"Inappropriate {category}: {', '.join(found_patterns)}")

        return issues

    def _assess_therapeutic_quality(self, text: str) -> Dict[str, Any]:
        """치료적 품질 평가"""

        text_lower = text.lower()

        # 긍정적 지표 계산
        positive_count = sum(
            1
            for indicator in self.therapeutic_quality_indicators.get("positive", [])
            if indicator in text_lower
        )

        empathetic_count = sum(
            1
            for indicator in self.therapeutic_quality_indicators.get("empathetic", [])
            if indicator in text_lower
        )

        empowering_count = sum(
            1
            for indicator in self.therapeutic_quality_indicators.get("empowering", [])
            if indicator in text_lower
        )

        total_positive = positive_count + empathetic_count + empowering_count

        # 품질 레벨 결정
        if total_positive >= 5:
            quality_level = "excellent"
        elif total_positive >= 3:
            quality_level = "good"
        elif total_positive >= 1:
            quality_level = "acceptable"
        else:
            quality_level = "poor"

        return {
            "level": quality_level,
            "positive_indicators": positive_count,
            "empathetic_indicators": empathetic_count,
            "empowering_indicators": empowering_count,
            "total_score": total_positive,
        }

    def _validate_context_appropriateness(
        self, response: str, context: Dict[str, Any]
    ) -> List[str]:
        """컨텍스트 적절성 검증"""

        issues = []

        # 대처 스타일 일치성 검사
        coping_style = context.get("coping_style", "balanced")
        if not self._matches_coping_style(response, coping_style):
            issues.append(f"Response doesn't match {coping_style} coping style")

        # 사용자 상황 고려 여부 검사
        emotion_keywords = context.get("emotion_keywords", [])
        if emotion_keywords and not self._acknowledges_emotions(
            response, emotion_keywords
        ):
            issues.append("Response doesn't acknowledge user's specific emotions")

        # 개인화 수준 검사
        guestbook_data = context.get("guestbook_data", {})
        if guestbook_data.get("title") and not self._references_user_content(
            response, guestbook_data
        ):
            issues.append(
                "Response lacks personalization despite available user content"
            )

        return issues

    def _matches_coping_style(self, response: str, coping_style: str) -> bool:
        """응답이 대처 스타일에 맞는지 확인"""

        response_lower = response.lower()

        style_indicators = {
            "avoidant": [
                "gentle",
                "soft",
                "gradually",
                "slowly",
                "carefully",
                "tenderly",
            ],
            "confrontational": [
                "direct",
                "honest",
                "authentic",
                "bold",
                "courageously",
                "strength",
            ],
            "balanced": [
                "thoughtful",
                "balanced",
                "considered",
                "mature",
                "wisely",
                "harmonious",
            ],
        }

        if coping_style not in style_indicators:
            return True  # 알 수 없는 스타일은 통과

        indicators = style_indicators[coping_style]
        return any(indicator in response_lower for indicator in indicators)

    def _acknowledges_emotions(
        self, response: str, emotion_keywords: List[str]
    ) -> bool:
        """응답이 사용자 감정을 인정하는지 확인"""

        if not emotion_keywords:
            return True

        response_lower = response.lower()

        # 직접적인 감정 언급 확인
        direct_mentions = sum(
            1 for emotion in emotion_keywords if emotion.lower() in response_lower
        )

        # 간접적인 감정 인정 표현 확인
        acknowledgment_phrases = [
            "feeling",
            "emotion",
            "experience",
            "going through",
            "facing",
            "dealing with",
        ]
        indirect_acknowledgment = any(
            phrase in response_lower for phrase in acknowledgment_phrases
        )

        return direct_mentions > 0 or indirect_acknowledgment

    def _references_user_content(
        self, response: str, guestbook_data: Dict[str, Any]
    ) -> bool:
        """응답이 사용자 콘텐츠를 참조하는지 확인"""

        response_lower = response.lower()

        # 방명록 제목 참조 확인
        title = guestbook_data.get("title", "")
        if title and title.lower() in response_lower:
            return True

        # 태그 참조 확인
        tags = guestbook_data.get("tags", [])
        if tags and any(tag.lower() in response_lower for tag in tags):
            return True

        # 일반적인 개인화 표현 확인
        personalization_phrases = [
            "you named",
            "you called",
            "you titled",
            "your choice",
            "your words",
            "your expression",
        ]

        return any(phrase in response_lower for phrase in personalization_phrases)

    def _assess_professional_referral_need(
        self, response: str, context: Dict[str, Any]
    ) -> Optional[str]:
        """전문가 상담 권유 필요성 평가"""

        text_lower = response.lower()

        # 즉시 상담 필요
        for trigger in self.professional_referral_triggers.get("immediate", []):
            if trigger in text_lower:
                return "immediate"

        # 긴급 상담 권유
        for trigger in self.professional_referral_triggers.get("urgent", []):
            if trigger in text_lower:
                return "urgent"

        # 상담 권장
        urgent_count = sum(
            1
            for trigger in self.professional_referral_triggers.get("recommended", [])
            if trigger in text_lower
        )

        if urgent_count >= 2:
            return "recommended"

        # 컨텍스트 기반 추가 평가
        user_journey = context.get("user_journey", {})
        if user_journey:
            # 장기간 사용자이지만 지속적인 부정적 패턴
            gallery_count = user_journey.get("gallery_items_count", 0)
            if gallery_count > 20:  # 많은 사용 이력이 있는 경우
                negative_pattern_indicators = [
                    "still struggling",
                    "getting worse",
                    "no improvement",
                ]
                if any(
                    indicator in text_lower for indicator in negative_pattern_indicators
                ):
                    return "recommended"

        return None

    def _generate_safety_recommendations(
        self, validation_result: Dict[str, Any]
    ) -> List[str]:
        """안전성 검증 결과 기반 권장사항 생성"""

        recommendations = []

        if validation_result["safety_level"] == "critical":
            recommendations.extend(
                [
                    "즉시 전문가 개입이 필요합니다",
                    "응답 생성을 중단하고 위기 개입 프로토콜을 활성화하세요",
                    "사용자에게 즉시 전문 상담 리소스를 제공하세요",
                ]
            )

        elif validation_result["safety_level"] == "warning":
            recommendations.extend(
                [
                    "응답을 수정하여 더 안전하고 지지적인 내용으로 변경하세요",
                    "전문가 상담 정보를 함께 제공하는 것을 고려하세요",
                ]
            )

        if validation_result["therapeutic_quality"] == "poor":
            recommendations.append("치료적 품질을 향상시키기 위해 응답을 재생성하세요")

        if validation_result["referral_recommendation"]:
            referral_messages = {
                "immediate": "즉시 전문가 상담 연결이 필요합니다",
                "urgent": "빠른 시일 내에 전문가 상담을 권유하세요",
                "recommended": "전문가 상담 정보를 제공하는 것을 고려하세요",
            }
            rec_message = referral_messages.get(
                validation_result["referral_recommendation"]
            )
            if rec_message:
                recommendations.append(rec_message)

        if len(validation_result["issues"]) > 0:
            recommendations.append("식별된 안전성 이슈를 해결하세요")

        return recommendations

    def check_therapeutic_safety(self, content: str) -> bool:
        """치료적 맥락에서의 안전성 체크 (간단한 버전)"""

        if not content or not content.strip():
            return False

        # 기본 안전성 검사
        validation_result = self.validate_gpt_response(content, {})

        return (
            validation_result["is_safe"]
            and validation_result["safety_level"] != "critical"
        )

    def analyze_safety_trends(
        self, validation_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """안전성 검증 이력 분석"""

        if not validation_history:
            return {"total_validations": 0}

        analysis = {
            "total_validations": len(validation_history),
            "safety_level_distribution": {},
            "common_issues": {},
            "referral_rate": 0,
            "improvement_trend": "stable",
            "recommendations": [],
        }

        # 안전성 레벨 분포
        for validation in validation_history:
            level = validation.get("safety_level", "unknown")
            analysis["safety_level_distribution"][level] = (
                analysis["safety_level_distribution"].get(level, 0) + 1
            )

        # 일반적인 이슈 분석
        all_issues = []
        referral_count = 0

        for validation in validation_history:
            all_issues.extend(validation.get("issues", []))
            if validation.get("referral_recommendation"):
                referral_count += 1

        # 이슈 빈도 계산
        issue_counts = {}
        for issue in all_issues:
            issue_type = issue.split(":")[0] if ":" in issue else issue
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

        analysis["common_issues"] = dict(
            sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        )
        analysis["referral_rate"] = (referral_count / len(validation_history)) * 100

        # 개선 트렌드 분석 (최근 10개 vs 이전 기록)
        if len(validation_history) >= 10:
            recent_critical = sum(
                1
                for v in validation_history[-10:]
                if v.get("safety_level") == "critical"
            )
            older_critical = sum(
                1
                for v in validation_history[:-10]
                if v.get("safety_level") == "critical"
            )

            if recent_critical < older_critical:
                analysis["improvement_trend"] = "improving"
            elif recent_critical > older_critical:
                analysis["improvement_trend"] = "declining"

        # 권장사항 생성
        if analysis["referral_rate"] > 20:
            analysis["recommendations"].append("높은 상담 권유율 - 시스템 조정 필요")

        critical_rate = (
            analysis["safety_level_distribution"].get("critical", 0)
            / len(validation_history)
            * 100
        )
        if critical_rate > 5:
            analysis["recommendations"].append(
                "치명적 안전성 이슈 비율이 높음 - 긴급 검토 필요"
            )

        return analysis

    def export_safety_rules(self, file_path: str) -> bool:
        """안전성 규칙 내보내기"""

        try:
            export_data = {
                "safety_rules": self.default_safety_rules,
                "therapeutic_quality_indicators": self.therapeutic_quality_indicators,
                "professional_referral_triggers": self.professional_referral_triggers,
                "export_date": datetime.now().isoformat(),
                "version": "1.0",
            }

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"안전성 규칙이 내보내졌습니다: {file_path}")
            return True

        except Exception as e:
            logger.error(f"안전성 규칙 내보내기 실패: {e}")
            return False
