# src/ai/prompt_engineer.py

import re
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PromptEngineer:
    """일기 내용을 GPT 기반 이미지 프롬프트로 변환"""

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

        # 안전성 검증 키워드
        self.unsafe_keywords = {
            "violence": [
                "violence",
                "blood",
                "weapon",
                "gun",
                "knife",
                "kill",
                "murder",
                "harm",
            ],
            "sexual": ["sexual", "nude", "naked", "porn", "explicit", "sex"],
            "self_harm": [
                "suicide",
                "self-harm",
                "cutting",
                "overdose",
                "hanging",
                "jump",
            ],
            "extreme_negative": [
                "worthless",
                "hopeless",
                "hate myself",
                "want to die",
                "end it all",
            ],
        }

    def enhance_diary_to_prompt(
        self,
        diary_text: str,
        emotion_keywords: List[str],
        coping_style: str,
        visual_preferences: Dict[str, Any],
    ) -> Dict[str, Any]:
        """일기 내용을 GPT 기반 이미지 프롬프트로 변환"""

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

            # 3. GPT 시스템 메시지 구성
            system_message = self._build_system_message(
                coping_style, visual_preferences
            )

            # 4. 사용자 메시지 구성
            user_message = self._build_user_message(
                diary_text, emotion_keywords, visual_preferences
            )

            # 5. GPT API 호출
            gpt_response = self.gpt_service.generate_prompt_engineering_response(
                system_message=system_message,
                user_message=user_message,
                max_tokens=100,  # 프롬프트는 짧게
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

            # 6. 생성된 프롬프트 후처리
            raw_prompt = gpt_response["content"]
            processed_prompt = self._post_process_prompt(raw_prompt, visual_preferences)

            # 7. 최종 안전성 검증
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
                    "gpt_tokens": gpt_response.get("usage", {}),
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

    def _build_system_message(
        self, coping_style: str, visual_preferences: Dict[str, Any]
    ) -> str:
        """대처 스타일별 시스템 메시지 구성"""

        base_system = self.system_messages.get(
            coping_style, self.system_messages["balanced"]
        )

        # 시각적 선호도 정보 추가
        visual_context = f"""
Visual Style Preferences:
- Art Style: {visual_preferences.get('art_style', 'painting')}
- Color Tone: {visual_preferences.get('color_tone', 'warm')}
- Complexity: {visual_preferences.get('complexity', 'balanced')}
- Brightness: {visual_preferences.get('brightness', 0.5)}
- Saturation: {visual_preferences.get('saturation', 0.5)}

Incorporate these preferences naturally into your prompt generation."""

        return base_system + "\n\n" + visual_context

    def _build_user_message(
        self,
        diary_text: str,
        emotion_keywords: List[str],
        visual_preferences: Dict[str, Any],
    ) -> str:
        """사용자 메시지 구성"""

        # 감정 키워드 정리
        emotions_text = ", ".join(emotion_keywords) if emotion_keywords else "neutral"

        user_message = f"""Transform this diary entry into an artistic image prompt:

DIARY ENTRY:
"{diary_text}"

DETECTED EMOTIONS: {emotions_text}

REQUIREMENTS:
1. Create a visual metaphor that captures the emotional essence
2. Maximum 150 characters for the final prompt
3. Include artistic style elements naturally
4. Focus on therapeutic and constructive imagery
5. Ensure the prompt is safe and appropriate

Please generate only the image prompt, nothing else."""

        return user_message

    def _post_process_prompt(
        self, raw_prompt: str, visual_preferences: Dict[str, Any]
    ) -> str:
        """생성된 프롬프트 후처리"""

        # 1. 불필요한 텍스트 제거 (GPT가 추가한 설명 등)
        cleaned_prompt = self._clean_gpt_response(raw_prompt)

        # 2. 시각적 요소 보강
        enhanced_prompt = self._enhance_visual_elements(
            cleaned_prompt, visual_preferences
        )

        # 3. 150자 제한 적용
        final_prompt = self._apply_character_limit(enhanced_prompt, 150)

        return final_prompt

    def _clean_gpt_response(self, raw_prompt: str) -> str:
        """GPT 응답에서 순수 프롬프트만 추출"""

        # 줄바꿈 제거
        cleaned = raw_prompt.replace("\n", " ").replace("\r", " ")

        # 연속된 공백 제거
        cleaned = re.sub(r"\s+", " ", cleaned)

        # 따옴표 제거 (GPT가 프롬프트를 따옴표로 감쌀 수 있음)
        cleaned = cleaned.strip().strip('"').strip("'")

        # "Image prompt:" 같은 접두사 제거
        prefixes_to_remove = [
            "image prompt:",
            "prompt:",
            "art prompt:",
            "visual prompt:",
            "here's the prompt:",
            "here is the prompt:",
            "the prompt is:",
        ]

        cleaned_lower = cleaned.lower()
        for prefix in prefixes_to_remove:
            if cleaned_lower.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
                break

        return cleaned.strip()

    def _enhance_visual_elements(
        self, prompt: str, visual_preferences: Dict[str, Any]
    ) -> str:
        """시각적 요소 보강"""

        enhanced_elements = []

        # 기존 프롬프트
        enhanced_elements.append(prompt)

        # 아트 스타일 추가 (프롬프트에 없는 경우)
        art_style = visual_preferences.get("art_style", "painting")
        style_text = self.visual_style_mappings.get(art_style, "")
        if style_text and not any(
            word in prompt.lower() for word in ["painting", "photo", "abstract"]
        ):
            enhanced_elements.append(style_text.split(",")[0])  # 첫 번째 키워드만 사용

        # 색감 보강 (필요시)
        color_tone = visual_preferences.get("color_tone", "warm")
        if not any(
            color in prompt.lower() for color in ["warm", "cool", "pastel", "color"]
        ):
            color_hint = self.color_tone_mappings.get(color_tone, "").split(",")[0]
            if color_hint:
                enhanced_elements.append(color_hint)

        # 품질 키워드 추가
        enhanced_elements.append("high quality")

        return ", ".join(enhanced_elements)

    def _apply_character_limit(self, prompt: str, max_chars: int) -> str:
        """150자 제한 적용 및 문장 완성"""

        if len(prompt) <= max_chars:
            return prompt

        # 단어 경계에서 자르기
        words = prompt.split()
        truncated_words = []
        current_length = 0

        for word in words:
            # 다음 단어를 추가했을 때의 길이 계산 (공백 포함)
            additional_length = len(word) + (1 if truncated_words else 0)

            if current_length + additional_length <= max_chars:
                truncated_words.append(word)
                current_length += additional_length
            else:
                break

        truncated_prompt = " ".join(truncated_words)

        # 문장이 완전하지 않은 경우 마지막 불완전한 구문 제거
        if truncated_prompt and not truncated_prompt.endswith((".", ",", ";")):
            # 마지막 쉼표나 완전한 구문을 찾아서 그 지점에서 자르기
            last_comma = truncated_prompt.rfind(", ")
            if last_comma > len(truncated_prompt) * 0.7:  # 너무 많이 자르지 않도록
                truncated_prompt = truncated_prompt[:last_comma]

        return truncated_prompt.strip()

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
            return "프롬프트가 안전합니다."

        critical_issues = [
            issue for issue in safety_issues if issue["severity"] == "critical"
        ]
        if critical_issues:
            return "즉시 전문가 상담을 권장합니다. 프롬프트 생성을 중단합니다."

        high_issues = [issue for issue in safety_issues if issue["severity"] == "high"]
        if high_issues:
            return "부적절한 내용이 감지되었습니다. 프롬프트를 수정하거나 다른 접근을 시도하세요."

        return "일부 주의가 필요한 내용이 있습니다. 내용을 검토해주세요."

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
        """생성된 프롬프트의 효과성 분석"""

        analysis = {
            "length_check": {
                "character_count": len(prompt),
                "within_limit": len(prompt) <= 150,
                "optimal_range": 100 <= len(prompt) <= 150,
            },
            "keyword_analysis": {
                "emotional_keywords": self._count_emotional_keywords(prompt),
                "visual_keywords": self._count_visual_keywords(prompt),
                "quality_keywords": self._count_quality_keywords(prompt),
            },
            "structure_analysis": {
                "has_main_subject": self._has_main_subject(prompt),
                "has_style_elements": self._has_style_elements(prompt),
                "has_mood_descriptors": self._has_mood_descriptors(prompt),
            },
            "therapeutic_suitability": {
                "constructive_language": self._uses_constructive_language(prompt),
                "avoids_negative_triggers": self._avoids_negative_triggers(prompt),
                "encourages_reflection": self._encourages_reflection(prompt),
            },
        }

        # 전체 점수 계산
        analysis["overall_score"] = self._calculate_overall_score(analysis)

        return analysis

    def _count_emotional_keywords(self, prompt: str) -> int:
        """감정 관련 키워드 개수"""
        emotional_words = [
            "calm",
            "peaceful",
            "serene",
            "gentle",
            "warm",
            "bright",
            "hopeful",
            "contemplative",
            "reflective",
            "thoughtful",
            "melancholy",
            "dreamy",
        ]
        return sum(1 for word in emotional_words if word in prompt.lower())

    def _count_visual_keywords(self, prompt: str) -> int:
        """시각적 요소 키워드 개수"""
        visual_words = [
            "painting",
            "photography",
            "abstract",
            "color",
            "light",
            "shadow",
            "texture",
            "composition",
            "style",
            "artistic",
        ]
        return sum(1 for word in visual_words if word in prompt.lower())

    def _count_quality_keywords(self, prompt: str) -> int:
        """품질 관련 키워드 개수"""
        quality_words = [
            "high quality",
            "detailed",
            "masterpiece",
            "artistic",
            "beautiful",
        ]
        return sum(1 for word in quality_words if word in prompt.lower())

    def _has_main_subject(self, prompt: str) -> bool:
        """주요 주제가 있는지 확인"""
        subjects = [
            "landscape",
            "portrait",
            "scene",
            "figure",
            "object",
            "space",
            "environment",
            "setting",
            "place",
            "moment",
        ]
        return any(subject in prompt.lower() for subject in subjects)

    def _has_style_elements(self, prompt: str) -> bool:
        """스타일 요소가 포함되어 있는지 확인"""
        return any(
            style in prompt.lower()
            for style in ["painting", "photography", "abstract", "artistic", "style"]
        )

    def _has_mood_descriptors(self, prompt: str) -> bool:
        """분위기 서술어가 있는지 확인"""
        moods = [
            "calm",
            "peaceful",
            "gentle",
            "warm",
            "cool",
            "bright",
            "soft",
            "intense",
            "dramatic",
            "subtle",
            "vibrant",
            "muted",
        ]
        return any(mood in prompt.lower() for mood in moods)

    def _uses_constructive_language(self, prompt: str) -> bool:
        """건설적인 언어를 사용하는지 확인"""
        constructive_words = [
            "hope",
            "growth",
            "beauty",
            "harmony",
            "balance",
            "peace",
            "reflection",
            "contemplation",
            "understanding",
            "acceptance",
        ]
        return any(word in prompt.lower() for word in constructive_words)

    def _avoids_negative_triggers(self, prompt: str) -> bool:
        """부정적 트리거를 피하는지 확인"""
        triggers = [
            "death",
            "violence",
            "blood",
            "pain",
            "suffering",
            "hopeless",
            "worthless",
            "hate",
            "destroy",
            "broken",
        ]
        return not any(trigger in prompt.lower() for trigger in triggers)

    def _encourages_reflection(self, prompt: str) -> bool:
        """성찰을 장려하는지 확인"""
        reflective_elements = [
            "contemplative",
            "thoughtful",
            "reflective",
            "introspective",
            "meditative",
            "peaceful",
            "quiet",
            "serene",
        ]
        return any(element in prompt.lower() for element in reflective_elements)

    def _calculate_overall_score(self, analysis: Dict[str, Any]) -> float:
        """전체 효과성 점수 계산 (0-100)"""

        score = 0

        # 길이 점수 (20점)
        if analysis["length_check"]["within_limit"]:
            score += 15
            if analysis["length_check"]["optimal_range"]:
                score += 5

        # 키워드 점수 (30점)
        keyword_score = min(
            30,
            analysis["keyword_analysis"]["emotional_keywords"] * 5
            + analysis["keyword_analysis"]["visual_keywords"] * 5
            + analysis["keyword_analysis"]["quality_keywords"] * 5,
        )
        score += keyword_score

        # 구조 점수 (25점)
        structure_elements = [
            analysis["structure_analysis"]["has_main_subject"],
            analysis["structure_analysis"]["has_style_elements"],
            analysis["structure_analysis"]["has_mood_descriptors"],
        ]
        score += sum(structure_elements) * 8.33  # 25/3

        # 치료적 적합성 점수 (25점)
        therapeutic_elements = [
            analysis["therapeutic_suitability"]["constructive_language"],
            analysis["therapeutic_suitability"]["avoids_negative_triggers"],
            analysis["therapeutic_suitability"]["encourages_reflection"],
        ]
        score += sum(therapeutic_elements) * 8.33  # 25/3

        return round(min(100, score), 1)
