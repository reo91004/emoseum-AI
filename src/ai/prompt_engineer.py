# src/ai/prompt_engineer.py

# ==============================================================================
# 이 파일은 사용자의 일기 내용, 감정, 대처 스타일을 바탕으로 Stable Diffusion을 위한
# 이미지 생성 프롬프트를 전문적으로 설계(Engineering)하는 역할을 한다.
# `src.services.gpt_service.GPTService`를 통해 GPT API를 호출하며,
# `config/gpt_prompts.yaml`의 'prompt_engineering' 섹션의 가이드라인을 사용하여
# 치료적이면서도 예술적인 프롬프트를 생성한다.
# ==============================================================================

import re
from typing import Dict, List, Tuple, Any, Optional
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class PromptEngineer:
    """일기 내용을 이미지 프롬프트로 변환"""

    def __init__(self, gpt_service, gpt_prompts_path: Optional[str] = None):
        self.gpt_service = gpt_service
        self.gpt_prompts_path = Path(gpt_prompts_path) if gpt_prompts_path else None

        # YAML 파일에서 모든 설정 로드
        if self.gpt_prompts_path and self.gpt_prompts_path.exists():
            self._load_all_settings_from_yaml()
            logger.info(
                f"모든 설정을 YAML 파일에서 로드했습니다: {self.gpt_prompts_path}"
            )
        else:
            # YAML 파일 로드 실패 시 에러 발생
            if self.gpt_prompts_path:
                logger.error(
                    f"GPT 프롬프트 파일을 찾을 수 없습니다: {self.gpt_prompts_path}"
                )
                raise FileNotFoundError(
                    f"GPT prompts file not found: {self.gpt_prompts_path}"
                )
            else:
                logger.error("GPT 프롬프트 파일 경로가 제공되지 않았습니다")
                raise ValueError("GPT prompts file path is required")

        logger.info("PromptEngineer 초기화 완료")

    def _load_all_settings_from_yaml(self):
        """YAML 파일에서 모든 설정 로드"""
        try:
            with open(self.gpt_prompts_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)

            # 시스템 메시지 로드
            if "prompt_engineering" in yaml_data:
                prompt_data = yaml_data["prompt_engineering"]
                self.system_messages = {}
                for style in ["avoidant", "confrontational", "balanced"]:
                    if style in prompt_data and "system_message" in prompt_data[style]:
                        self.system_messages[style] = prompt_data[style][
                            "system_message"
                        ]

            # 시각적 매핑 로드
            if "visual_mappings" in yaml_data:
                mappings = yaml_data["visual_mappings"]
                self.visual_style_mappings = mappings.get("style_mappings", {})
                self.color_tone_mappings = mappings.get("color_tone_mappings", {})
                self.complexity_mappings = mappings.get("complexity_mappings", {})

            # 안전 키워드 로드
            if "unsafe_keywords" in yaml_data:
                self.unsafe_keywords = yaml_data["unsafe_keywords"]

        except Exception as e:
            logger.error(f"YAML 설정 로드 실패: {e}")
            raise

    def enhance_diary_to_prompt(
        self,
        diary_text: str,
        emotion_keywords: List[str],
        coping_style: str,
        visual_preferences: Dict[str, Any],
        user_id: str = "anonymous",
    ) -> Dict[str, Any]:
        """일기 내용을 이미지 프롬프트로 변환 (메인 메서드)"""

        try:
            # 1. 입력 데이터 검증
            if not diary_text or not diary_text.strip():
                raise ValueError("일기 텍스트가 비어있습니다")

            # 2. 안전성 사전 검증
            safety_check = self.validate_prompt_safety(diary_text)
            if not safety_check["is_safe"]:
                logger.warning(
                    f"안전하지 않은 일기 내용 감지: {safety_check['issues']}"
                )
                return {
                    "success": False,
                    "error": "unsafe_content",
                    "safety_issues": safety_check["issues"],
                    "prompt": "",
                    "metadata": {},
                }

            # 3. GPT API 호출 (GPTService에서 시스템/사용자 메시지 구성)
            gpt_response = self.gpt_service.generate_prompt_engineering_response(
                diary_text=diary_text,
                emotion_keywords=emotion_keywords,
                coping_style=coping_style,
                visual_preferences=visual_preferences,
                user_id=user_id,
                max_tokens=100,
                temperature=0.7,
            )

            if not gpt_response.get("success", False):
                logger.error(
                    f"GPT 프롬프트 생성 실패: {gpt_response.get('error', 'Unknown error')}"
                )
                return {
                    "success": False,
                    "error": "gpt_generation_failed",
                    "gpt_error": gpt_response.get("error", "Unknown error"),
                    "prompt": "",
                    "metadata": {},
                }

            # 4. 생성된 프롬프트 후처리
            raw_prompt = gpt_response.get("prompt", gpt_response.get("content", ""))
            processed_prompt = self._post_process_prompt(raw_prompt, visual_preferences)

            # 5. 최종 안전성 검증
            final_safety_check = self.validate_prompt_safety(processed_prompt)
            if not final_safety_check["is_safe"]:
                logger.warning(
                    f"생성된 프롬프트가 안전하지 않음: {final_safety_check['issues']}"
                )
                return {
                    "success": False,
                    "error": "generated_unsafe_content",
                    "safety_issues": final_safety_check["issues"],
                    "prompt": "",
                    "metadata": {},
                }

            result = {
                "success": True,
                "prompt": processed_prompt,
                "metadata": {
                    "original_diary_length": len(diary_text),
                    "emotion_keywords": emotion_keywords,
                    "coping_style": coping_style,
                    "visual_preferences": visual_preferences,
                    "prompt_length": len(processed_prompt),
                    "gpt_model": gpt_response.get("model", "unknown"),
                    "gpt_tokens": gpt_response.get("token_usage", {}),
                    "generation_method": "gpt_based",
                    "safety_validated": True,
                },
            }

            logger.info(
                f"프롬프트 생성 완료: {len(diary_text)}자 일기 → {len(processed_prompt)}자 프롬프트 "
                f"({coping_style} 스타일)"
            )

            return result

        except Exception as e:
            logger.error(f"프롬프트 생성 중 오류 발생: {e}")
            return {"success": False, "error": str(e), "prompt": "", "metadata": {}}


    def _post_process_prompt(
        self, raw_prompt: str, visual_preferences: Dict[str, Any]
    ) -> str:
        """프롬프트 후처리"""

        if not raw_prompt:
            return ""

        # 기본 정리
        processed_prompt = raw_prompt.strip()

        # 불필요한 따옴표나 특수문자 제거
        processed_prompt = re.sub(r'^["\']|["\']$', "", processed_prompt)

        # 연속된 공백 정리
        processed_prompt = re.sub(r"\s+", " ", processed_prompt)

        # 길이 제한 (150자)
        if len(processed_prompt) > 150:
            truncated_prompt = processed_prompt[:147] + "..."

            # 단어 중간에서 자르지 않도록 조정
            if truncated_prompt.endswith((" ,", " .", " ;", ",")):
                # 마지막 쉼표나 완전한 구문을 찾아서 그 지점에서 자르기
                last_comma = truncated_prompt.rfind(", ")
                if last_comma > len(truncated_prompt) * 0.7:  # 너무 많이 자르지 않도록
                    truncated_prompt = truncated_prompt[:last_comma]

            processed_prompt = truncated_prompt.strip()

        return processed_prompt

    def validate_prompt_safety(self, text: str) -> Dict[str, Any]:
        """프롬프트 안전성 검증"""

        safety_issues = []
        text_lower = text.lower()

        # 카테고리별 안전하지 않은 키워드 검사
        for category, keywords in self.unsafe_keywords.items():
            found_keywords = [keyword for keyword in keywords if keyword in text_lower]
            if found_keywords:
                safety_issues.append(
                    {
                        "category": category,
                        "found_keywords": found_keywords,
                        "severity": self._get_severity_level(category),
                    }
                )

        # 추가 휴리스틱 검사
        additional_checks = self._additional_safety_checks(text)
        safety_issues.extend(additional_checks)

        is_safe = len(safety_issues) == 0

        return {
            "is_safe": is_safe,
            "issues": safety_issues,
            "severity_level": max(
                [issue["severity"] for issue in safety_issues], default="safe"
            ),
            "recommendation": self._get_safety_recommendation(safety_issues),
        }

    def _get_severity_level(self, category: str) -> str:
        """카테고리별 심각도 반환"""

        severity_map = {
            "self_harm": "critical",
            "violence": "high",
            "sexual": "high",
            "extreme_negative": "medium",
        }

        return severity_map.get(category, "low")

    def _additional_safety_checks(self, text: str) -> List[Dict[str, Any]]:
        """추가 안전성 검사"""

        additional_issues = []
        text_lower = text.lower()

        # 과도하게 어두운 표현 체크
        dark_indicators = [
            "complete darkness",
            "total despair",
            "endless void",
            "absolute emptiness",
        ]
        found_dark = [
            indicator for indicator in dark_indicators if indicator in text_lower
        ]
        if found_dark:
            additional_issues.append(
                {
                    "category": "excessive_darkness",
                    "found_keywords": found_dark,
                    "severity": "medium",
                }
            )

        # 치료적 부적절성 체크
        inappropriate_therapeutic = ["fix me", "cure my depression", "make me normal"]
        found_inappropriate = [
            phrase for phrase in inappropriate_therapeutic if phrase in text_lower
        ]
        if found_inappropriate:
            additional_issues.append(
                {
                    "category": "inappropriate_therapeutic_language",
                    "found_keywords": found_inappropriate,
                    "severity": "low",
                }
            )

        return additional_issues

    def _get_safety_recommendation(self, safety_issues: List[Dict[str, Any]]) -> str:
        """안전성 검사 결과에 따른 권장사항"""

        if not safety_issues:
            return "Prompt is safe to proceed."

        critical_issues = [
            issue for issue in safety_issues if issue["severity"] == "critical"
        ]
        if critical_issues:
            return "Immediate professional consultation recommended. Prompt generation halted."

        high_issues = [issue for issue in safety_issues if issue["severity"] == "high"]
        if high_issues:
            return "Inappropriate content detected. Please modify prompt or try different approach."

        return "Some content requires attention. Please review the content."



    def generate_transition_guidance(
        self,
        guestbook_title: str,
        emotion_keywords: List[str],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """큐레이터 전환 안내 질문 생성"""

        try:
            logger.info(f"큐레이터 전환 안내 질문 생성 시작: {guestbook_title}")

            # GPT 서비스를 통해 전환 안내 질문 생성
            gpt_response = self.gpt_service.generate_transition_guidance(
                guestbook_title=guestbook_title,
                emotion_keywords=emotion_keywords,
                user_id=user_id,
                max_tokens=150,
                temperature=0.7,
            )

            if not gpt_response.get("success", False):
                logger.error(f"전환 안내 생성 실패: {gpt_response.get('error')}")
                return {
                    "success": False,
                    "error": gpt_response.get("error", "Unknown error"),
                    "content": "",
                }

            result = {
                "success": True,
                "content": gpt_response.get("content", ""),
                "metadata": {
                    "guestbook_title": guestbook_title,
                    "emotion_keywords": emotion_keywords,
                    "generation_method": "gpt_based",
                    "gpt_model": gpt_response.get("model", "unknown"),
                    "gpt_tokens": gpt_response.get("token_usage", {}),
                },
            }

            logger.info(f"큐레이터 전환 안내 질문 생성 완료: {guestbook_title}")
            return result

        except Exception as e:
            logger.error(f"전환 안내 질문 생성 중 오류 발생: {e}")
            return {"success": False, "error": str(e), "content": ""}

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

    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 확인"""
        return {
            "generation_method": "gpt_only",
            "fallback_available": False,
            "hardcoded_templates": False,
            "gpt_integration": "complete",
            "safety_validation_enabled": True,
            "coping_style_support": ["avoidant", "confrontational", "balanced"],
            "handover_status": "completed",
        }
