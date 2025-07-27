# src/ai/prompt_engineer.py

import re
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PromptEngineer:
    """일기 내용을 이미지 프롬프트로 변환"""

    def __init__(self, gpt_service):
        self.gpt_service = gpt_service

        # 대처 스타일별 시스템 메시지 템플릿
        self.system_messages = {
            "avoidant": """You are an expert AI image prompt engineer specializing in therapeutic art generation for sensitive users who prefer gentle, indirect emotional expression.

Your task is to transform personal diary entries into artistic image prompts that:
- Use soft, metaphorical language instead of direct emotional terms
- Create a protective, safe visual atmosphere
- Employ gentle, soothing imagery and colors
- Avoid harsh or confrontational visual elements
- Focus on abstract, non-threatening representations of emotions

Guidelines:
- Maximum 150 characters for the final prompt
- Use words like "gentle", "soft", "peaceful", "distant", "dreamy"
- Transform negative emotions into softer metaphors (storm → gentle rain, darkness → twilight)
- Prioritize comfort and emotional safety over direct expression""",
            "confrontational": """You are an expert AI image prompt engineer specializing in therapeutic art generation for users who prefer direct, honest emotional expression.

Your task is to transform personal diary entries into artistic image prompts that:
- Use clear, direct language that doesn't avoid difficult emotions
- Create authentic, unfiltered visual representations
- Employ bold colors and strong contrasts when appropriate
- Allow for intense imagery that matches emotional reality
- Focus on honest, confrontational depictions of inner states

Guidelines:
- Maximum 150 characters for the final prompt
- Use words like "bold", "intense", "raw", "authentic", "unmasked"
- Maintain emotional honesty while keeping imagery constructive
- Transform emotions into powerful, direct visual metaphors""",
            "balanced": """You are an expert AI image prompt engineer specializing in therapeutic art generation for users who prefer balanced emotional expression.

Your task is to transform personal diary entries into artistic image prompts that:
- Balance emotional honesty with visual harmony
- Create thoughtful, considered representations
- Use moderate intensity in colors and imagery
- Combine direct and metaphorical elements appropriately
- Focus on mature, nuanced depictions of emotional states

Guidelines:
- Maximum 150 characters for the final prompt
- Use words like "balanced", "harmonious", "thoughtful", "considered", "nuanced"
- Create prompts that are neither too soft nor too harsh
- Maintain emotional authenticity with visual sophistication""",
        }

        # 시각적 요소 매핑
        self.visual_style_mappings = {
            "painting": "oil painting style, artistic brushstrokes, canvas texture",
            "photography": "photographic style, natural lighting, realistic",
            "abstract": "abstract art style, conceptual, non-representational",
        }

        self.color_tone_mappings = {
            "warm": "warm colors, golden tones, orange and red hues",
            "cool": "cool colors, blue and green tones, calming palette",
            "pastel": "pastel colors, soft tones, gentle hues",
        }

        self.complexity_mappings = {
            "simple": "minimalist, clean composition, simple elements",
            "balanced": "balanced composition, moderate detail",
            "complex": "detailed, intricate, rich composition",
        }

        # 안전하지 않은 키워드 분류
        self.unsafe_keywords = {
            "self_harm": [
                "suicide",
                "kill myself",
                "end it all",
                "cutting",
                "self harm",
                "self-harm",
            ],
            "violence": [
                "kill",
                "murder",
                "weapon",
                "blood",
                "violence",
                "death",
                "harm others",
            ],
            "sexual": ["sexual", "nude", "naked", "porn", "explicit"],
            "extreme_negative": [
                "hopeless",
                "worthless",
                "hate myself",
                "complete failure",
            ],
        }

        logger.info("PromptEngineer 초기화 완료")

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

    def generate_image_prompt(
        self,
        diary_text: str,
        emotion_keywords: List[str],
        coping_style: str = "balanced",
        visual_preferences: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """별칭 메서드 - enhance_diary_to_prompt와 동일"""
        return self.enhance_diary_to_prompt(
            diary_text=diary_text,
            emotion_keywords=emotion_keywords,
            coping_style=coping_style,
            visual_preferences=visual_preferences or {},
            user_id=user_id or "anonymous",
        )

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

    def get_style_recommendations(self, coping_style: str) -> Dict[str, Any]:
        """대처 스타일별 권장사항 제공"""

        recommendations = {
            "avoidant": {
                "recommended_keywords": [
                    "gentle",
                    "soft",
                    "peaceful",
                    "dreamy",
                    "distant",
                    "subtle",
                ],
                "avoid_keywords": [
                    "harsh",
                    "sharp",
                    "intense",
                    "bold",
                    "confrontational",
                ],
                "visual_approach": "은유적이고 간접적인 표현 선호",
                "color_preferences": ["pastel", "warm"],
                "style_preferences": ["abstract", "painting"],
            },
            "confrontational": {
                "recommended_keywords": [
                    "bold",
                    "intense",
                    "authentic",
                    "raw",
                    "direct",
                    "honest",
                ],
                "avoid_keywords": ["soft", "gentle", "hiding", "avoiding", "distant"],
                "visual_approach": "직접적이고 솔직한 표현 선호",
                "color_preferences": ["cool", "warm"],
                "style_preferences": ["photography", "painting"],
            },
            "balanced": {
                "recommended_keywords": [
                    "balanced",
                    "harmonious",
                    "thoughtful",
                    "nuanced",
                    "considered",
                ],
                "avoid_keywords": ["extreme", "overwhelming", "excessive"],
                "visual_approach": "균형잡힌 감정 표현",
                "color_preferences": ["warm", "cool", "pastel"],
                "style_preferences": ["painting", "photography", "abstract"],
            },
        }

        return recommendations.get(coping_style, recommendations["balanced"])

    def analyze_prompt_effectiveness(self, prompt: str) -> Dict[str, Any]:
        """프롬프트 효과성 분석"""

        analysis = {
            "word_count": len(prompt.split()),
            "character_count": len(prompt),
            "estimated_effectiveness": "medium",
            "style_indicators": [],
            "emotion_indicators": [],
            "technical_quality": "good",
        }

        prompt_lower = prompt.lower()

        # 스타일 지표 분석
        style_keywords = [
            "painting",
            "photography",
            "abstract",
            "realistic",
            "artistic",
            "detailed",
        ]
        found_styles = [
            keyword for keyword in style_keywords if keyword in prompt_lower
        ]
        analysis["style_indicators"] = found_styles

        # 감정 지표 분석
        emotion_keywords = [
            "happy",
            "sad",
            "peaceful",
            "intense",
            "gentle",
            "bold",
            "melancholic",
            "joyful",
        ]
        found_emotions = [
            keyword for keyword in emotion_keywords if keyword in prompt_lower
        ]
        analysis["emotion_indicators"] = found_emotions

        # 효과성 점수 계산
        effectiveness_score = 0
        if len(prompt.split()) >= 10:
            effectiveness_score += 1
        if found_styles:
            effectiveness_score += 1
        if found_emotions:
            effectiveness_score += 1

        if effectiveness_score >= 3:
            analysis["estimated_effectiveness"] = "high"
        elif effectiveness_score >= 2:
            analysis["estimated_effectiveness"] = "medium"
        else:
            analysis["estimated_effectiveness"] = "low"

        return analysis

    def generate_transition_guidance(
        self,
        guestbook_title: str,
        emotion_keywords: List[str],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """큐레이터 전환 안내 질문 생성"""

        try:
            logger.info(
                f"큐레이터 전환 안내 질문 생성 시작: {guestbook_title}"
            )

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
            "simulation_mode": False,
            "gpt_integration": "complete",
            "safety_validation_enabled": True,
            "coping_style_support": ["avoidant", "confrontational", "balanced"],
            "handover_status": "completed",
        }
