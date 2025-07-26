# src/ai/curator_gpt.py

import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CuratorGPT:
    """GPT 기반 개인화된 큐레이터 메시지 생성"""

    def __init__(self, gpt_service, safety_validator):
        self.gpt_service = gpt_service
        self.safety_validator = safety_validator

        # 대처 스타일별 큐레이터 시스템 메시지
        self.curator_system_messages = {
            "avoidant": """You are a gentle, supportive digital art curator specializing in therapeutic communication for sensitive individuals.

Your role is to create personalized, encouraging messages for users who prefer gentle, indirect emotional support. You help them reflect on their emotional journey through art in a safe, non-threatening way.

Communication Style:
- Use soft, nurturing language that feels protective and safe
- Employ metaphors and indirect expressions rather than direct confrontation
- Focus on gradual acceptance and gentle self-compassion
- Avoid overwhelming emotions or intensity
- Create a sense of being understood without pressure

Message Structure:
- Opening: Warm, gentle acknowledgment
- Recognition: Soft appreciation of their courage in sharing
- Personal Note: Indirect validation using metaphors or gentle imagery
- Guidance: Subtle, non-directive suggestions for moving forward
- Closing: Supportive presence and ongoing availability

Tone: Nurturing, protective, patient, understanding, non-judgmental""",
            "confrontational": """You are a direct, authentic digital art curator specializing in honest therapeutic communication for individuals who value straightforward emotional engagement.

Your role is to create personalized, empowering messages for users who prefer clear, direct emotional support. You help them confront and process their emotions authentically through art reflection.

Communication Style:
- Use clear, honest language that respects their emotional reality
- Address difficult emotions directly without sugarcoating
- Focus on empowerment and authentic self-expression
- Encourage brave emotional exploration
- Validate their strength in facing challenges head-on

Message Structure:
- Opening: Strong, direct acknowledgment of their courage
- Recognition: Clear appreciation of their emotional honesty
- Personal Note: Direct validation of their experience and strength
- Guidance: Clear, actionable suggestions for continued growth
- Closing: Confident support and respect for their journey

Tone: Honest, empowering, respectful, straightforward, encouraging""",
            "balanced": """You are a wise, balanced digital art curator specializing in thoughtful therapeutic communication for individuals who appreciate nuanced emotional support.

Your role is to create personalized, harmonious messages that balance emotional honesty with gentle guidance. You help users reflect on their emotional journey with both courage and compassion.

Communication Style:
- Use mature, considered language that honors complexity
- Balance directness with sensitivity appropriately
- Focus on thoughtful reflection and wise perspective
- Encourage both courage and self-compassion
- Provide guidance that respects their autonomy

Message Structure:
- Opening: Thoughtful, balanced acknowledgment
- Recognition: Mature appreciation of their emotional work
- Personal Note: Nuanced validation that honors their complexity
- Guidance: Wise, balanced suggestions for continued growth
- Closing: Respectful support that honors their journey

Tone: Wise, balanced, thoughtful, respectful, encouraging""",
        }


    def generate_personalized_message(
        self,
        user_profile: Any,
        gallery_item: Any,
        personalization_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """개인화된 큐레이터 메시지 생성"""

        try:
            # 1. 개인화 컨텍스트 구성
            context = self._build_personalization_context(
                user_profile, gallery_item, personalization_context
            )

            # 2. GPT API 호출 (GPTService에서 메시지 구성)
            gpt_response = self.gpt_service.generate_curator_message(
                user_profile=context,
                gallery_item=context,
                personalization_context=context,
                user_id=user_profile.user_id,
                max_tokens=300,
                temperature=0.8,
            )

            if not gpt_response.get("success", False):
                logger.error(
                    f"GPT 큐레이터 메시지 생성 실패: {gpt_response.get('error', 'Unknown error')}"
                )
                return self._create_fallback_message(context)

            # 5. 생성된 메시지 구조화 (GPTService에서 이미 구조화됨)
            structured_message = gpt_response.get("message", {})
            raw_content = gpt_response.get("raw_content", "")

            # 6. 안전성 검증 (원본 텍스트 사용)
            safety_check = self.safety_validator.validate_gpt_response(
                response=raw_content, context=context
            )

            if not safety_check.get("is_safe", False):
                logger.warning(
                    f"생성된 큐레이터 메시지가 안전하지 않음: {safety_check.get('issues', [])}"
                )
                return self._create_fallback_message(context)

            # 7. 최종 메시지 구성
            final_message = {
                "message_id": f"curator_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "user_id": user_profile.user_id,
                "gallery_item_id": gallery_item.item_id,
                "message_type": "curator_closure",
                "content": structured_message,
                "personalization_data": {
                    "coping_style": context["coping_style"],
                    "emotion_keywords": context["emotion_keywords"],
                    "guestbook_data": context["guestbook_data"],
                    "personalization_level": context.get(
                        "personalization_level", "medium"
                    ),
                    "generation_method": "gpt_based",
                },
                "metadata": {
                    "gpt_model": gpt_response.get("model", "unknown"),
                    "gpt_tokens": gpt_response.get("token_usage", {}),
                    "safety_validated": True,
                    "generation_time": gpt_response.get("processing_time", 0),
                    "fallback_used": False,
                },
                "created_date": datetime.now().isoformat(),
            }

            logger.info(
                f"큐레이터 메시지 생성 완료: 사용자={user_profile.user_id}, "
                f"스타일={context['coping_style']}, 토큰={gpt_response.get('token_usage', {}).get('total_tokens', 0)}"
            )

            return final_message

        except Exception as e:
            logger.error(f"큐레이터 메시지 생성 중 오류 발생: {e}")
            return self._create_fallback_message(
                self._build_personalization_context(
                    user_profile, gallery_item, personalization_context
                )
            )

    def _build_personalization_context(
        self,
        user_profile: Any,
        gallery_item: Any,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """개인화 컨텍스트 구성"""

        # 기본 컨텍스트
        context = {
            "user_id": user_profile.user_id,
            "coping_style": self._get_current_coping_style(user_profile),
            "emotion_keywords": gallery_item.emotion_keywords or [],
            "diary_excerpt": self._create_diary_excerpt(gallery_item.diary_text),
            "guestbook_data": {
                "title": gallery_item.guestbook_title or "",
                "tags": gallery_item.guestbook_tags or [],
                "guided_question": gallery_item.guided_question or "",
            },
            "user_journey": {
                "member_since": user_profile.created_date,
                "psychometric_tests": (
                    len(user_profile.psychometric_results)
                    if user_profile.psychometric_results
                    else 0
                ),
                "gallery_items_count": self._estimate_gallery_items(user_profile),
            },
            "session_context": {
                "reflection_prompt": gallery_item.reflection_prompt or "",
                "vad_scores": (
                    gallery_item.vad_scores
                    if hasattr(gallery_item, "vad_scores")
                    else [0, 0, 0]
                ),
            },
        }

        # 추가 컨텍스트 병합
        if additional_context:
            context.update(additional_context)

        # 개인화 수준 계산
        context["personalization_level"] = self._calculate_personalization_level(
            context
        )

        return context

    def _get_current_coping_style(self, user_profile: Any) -> str:
        """현재 대처 스타일 추출"""
        if user_profile.psychometric_results:
            return user_profile.psychometric_results[0].coping_style
        return "balanced"

    def _create_diary_excerpt(self, diary_text: str, max_length: int = 100) -> str:
        """일기 내용 요약 추출"""
        if not diary_text:
            return ""

        # 첫 문장 또는 지정된 길이까지 추출
        if len(diary_text) <= max_length:
            return diary_text

        # 문장 단위로 자르기 시도
        sentences = diary_text.split(". ")
        if sentences and len(sentences[0]) <= max_length:
            return sentences[0] + ("." if not sentences[0].endswith(".") else "")

        # 단어 단위로 자르기
        words = diary_text.split()
        excerpt_words = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 <= max_length:
                excerpt_words.append(word)
                current_length += len(word) + 1
            else:
                break

        return " ".join(excerpt_words) + ("..." if len(diary_text) > max_length else "")

    def _estimate_gallery_items(self, user_profile: Any) -> int:
        """갤러리 아이템 수 추정 (실제 구현에서는 gallery_manager를 통해 조회)"""
        # 임시로 심리검사 횟수 기반 추정
        if user_profile.psychometric_results:
            return (
                len(user_profile.psychometric_results) * 5
            )  # 검사당 평균 5개 아이템 추정
        return 1

    def _calculate_personalization_level(self, context: Dict[str, Any]) -> str:
        """개인화 수준 계산"""

        score = 0

        # 사용자 데이터 풍부도 (40점)
        if context["user_journey"]["psychometric_tests"] > 0:
            score += 15
        if context["user_journey"]["gallery_items_count"] > 5:
            score += 15
        if context["guestbook_data"]["title"]:
            score += 10

        # 감정 데이터 품질 (30점)
        if context["emotion_keywords"]:
            score += 15
        if len(context["diary_excerpt"]) > 50:
            score += 15

        # 상호작용 데이터 (30점)
        if context["guestbook_data"]["tags"]:
            score += 15
        if context["guestbook_data"]["guided_question"]:
            score += 15

        if score >= 70:
            return "high"
        elif score >= 40:
            return "medium"
        else:
            return "low"

    def _build_curator_system_message(self, context: Dict[str, Any]) -> str:
        """큐레이터 시스템 메시지 구성"""

        coping_style = context["coping_style"]
        base_system = self.curator_system_messages.get(
            coping_style, self.curator_system_messages["balanced"]
        )

        # 개인화 정보 추가
        personalization_info = f"""
User Context:
- Coping Style: {coping_style}
- Emotions Explored: {', '.join(context['emotion_keywords'])}
- Personalization Level: {context.get('personalization_level', 'medium')}
- User Journey: {context['user_journey']['psychometric_tests']} assessments, {context['user_journey']['gallery_items_count']} gallery items

Guestbook Information:
- Title Given: "{context['guestbook_data']['title']}"
- Tags Used: {', '.join(context['guestbook_data']['tags'])}

Guidelines for This User:
- Refer to their guestbook title meaningfully
- Acknowledge their specific emotions: {', '.join(context['emotion_keywords'])}
- Match the {coping_style} communication style consistently
- Provide exactly 5 sections: opening, recognition, personal_note, guidance, closing
- Keep each section 1-2 sentences, total message under 250 words"""

        return base_system + "\n\n" + personalization_info

    def _build_curator_user_message(self, context: Dict[str, Any]) -> str:
        """큐레이터 사용자 메시지 구성"""

        user_message = f"""Create a personalized curator message for this user's emotional art journey:

DIARY EXCERPT:
"{context['diary_excerpt']}"

EMOTIONS EXPLORED:
{', '.join(context['emotion_keywords']) if context['emotion_keywords'] else 'Mixed emotions'}

GUESTBOOK ENTRY:
- Title: "{context['guestbook_data']['title']}"
- Tags: {', '.join(context['guestbook_data']['tags']) if context['guestbook_data']['tags'] else 'None'}

USER CONTEXT:
- Coping Style: {context['coping_style']}
- Experience Level: {context['user_journey']['gallery_items_count']} previous explorations
- Personalization: {context.get('personalization_level', 'medium')} level

REQUIREMENTS:
1. Create exactly 5 sections: opening, recognition, personal_note, guidance, closing
2. Reference their guestbook title "{context['guestbook_data']['title']}" meaningfully
3. Acknowledge emotions: {', '.join(context['emotion_keywords'])}
4. Use {context['coping_style']} communication style
5. Be therapeutic, supportive, and encouraging
6. Keep total message under 250 words

Generate the curator message now:"""

        return user_message

    def structure_message_content(self, raw_content: str) -> Dict[str, str]:
        """GPT 응답을 구조화된 메시지로 변환"""

        # GPT 응답 정리
        cleaned_content = self._clean_gpt_message(raw_content)

        # 섹션 구분 시도
        sections = self._parse_message_sections(cleaned_content)

        # 구조화된 형태로 변환
        structured_content = {
            "opening": sections.get("opening", ""),
            "recognition": sections.get("recognition", ""),
            "personal_note": sections.get("personal_note", ""),
            "guidance": sections.get("guidance", ""),
            "closing": sections.get("closing", ""),
        }

        # 빈 섹션 보완
        structured_content = self._fill_missing_sections(
            structured_content, cleaned_content
        )

        return structured_content

    def _clean_gpt_message(self, raw_content: str) -> str:
        """GPT 메시지 정리"""

        # 줄바꿈 정규화
        cleaned = re.sub(r"\r\n|\r|\n", " ", raw_content)

        # 연속 공백 제거
        cleaned = re.sub(r"\s+", " ", cleaned)

        # 불필요한 접두사 제거
        prefixes_to_remove = [
            "here's the curator message:",
            "curator message:",
            "message:",
            "here is the message:",
            "the message is:",
        ]

        cleaned_lower = cleaned.lower().strip()
        for prefix in prefixes_to_remove:
            if cleaned_lower.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
                break

        return cleaned.strip()

    def _parse_message_sections(self, content: str) -> Dict[str, str]:
        """메시지를 섹션별로 파싱"""

        sections = {}

        # 명시적 섹션 구분자가 있는 경우
        section_patterns = {
            "opening": r"(?:opening|start|beginning):\s*([^.]+\.)",
            "recognition": r"(?:recognition|acknowledgment):\s*([^.]+\.)",
            "personal_note": r"(?:personal_note|personal):\s*([^.]+\.)",
            "guidance": r"(?:guidance|advice|suggestion):\s*([^.]+\.)",
            "closing": r"(?:closing|end|conclusion):\s*([^.]+\.)",
        }

        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                sections[section_name] = match.group(1).strip()

        # 명시적 구분자가 없는 경우 문장 기반 분할
        if not sections:
            sentences = self._split_into_sentences(content)
            if len(sentences) >= 3:
                # 문장을 5개 섹션으로 분배
                sections = self._distribute_sentences_to_sections(sentences)

        return sections

    def _split_into_sentences(self, content: str) -> List[str]:
        """내용을 문장으로 분할"""

        # 마침표, 느낌표, 물음표로 문장 분할
        sentences = re.split(r"[.!?]+", content)

        # 빈 문장 제거 및 정리
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # 너무 짧은 문장 제외
                cleaned_sentences.append(sentence)

        return cleaned_sentences

    def _distribute_sentences_to_sections(self, sentences: List[str]) -> Dict[str, str]:
        """문장들을 5개 섹션에 분배"""

        sections = {
            "opening": "",
            "recognition": "",
            "personal_note": "",
            "guidance": "",
            "closing": "",
        }

        num_sentences = len(sentences)

        if num_sentences == 3:
            # 3문장인 경우
            sections["opening"] = sentences[0]
            sections["personal_note"] = sentences[1]
            sections["closing"] = sentences[2]
        elif num_sentences == 4:
            # 4문장인 경우
            sections["opening"] = sentences[0]
            sections["recognition"] = sentences[1]
            sections["guidance"] = sentences[2]
            sections["closing"] = sentences[3]
        elif num_sentences >= 5:
            # 5문장 이상인 경우
            sections["opening"] = sentences[0]
            sections["recognition"] = sentences[1]
            sections["personal_note"] = sentences[2]
            sections["guidance"] = sentences[3]
            sections["closing"] = sentences[4]
        else:
            # 2문장 이하인 경우
            sections["opening"] = sentences[0] if num_sentences > 0 else ""
            sections["closing"] = sentences[1] if num_sentences > 1 else ""

        return sections

    def _fill_missing_sections(
        self, sections: Dict[str, str], full_content: str
    ) -> Dict[str, str]:
        """빈 섹션 보완"""

        # 빈 섹션이 있으면 기본 내용으로 채우기
        empty_sections = [key for key, value in sections.items() if not value.strip()]

        if empty_sections:
            # 전체 내용을 빈 섹션에 균등 분배
            if not any(sections.values()):
                # 모든 섹션이 비어있으면 전체 내용을 opening에 배치
                sections["opening"] = full_content
            else:
                # 일부 섹션만 비어있으면 기본 문구 사용
                default_phrases = {
                    "opening": "Thank you for sharing this meaningful moment with me.",
                    "recognition": "Your emotional exploration shows real courage.",
                    "personal_note": "This experience reflects your growing self-awareness.",
                    "guidance": "Continue to honor your emotional journey with kindness.",
                    "closing": "I'm here to support you on this path of discovery.",
                }

                for empty_section in empty_sections:
                    if empty_section in default_phrases:
                        sections[empty_section] = default_phrases[empty_section]

        return sections

    def _create_fallback_message(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """GPT 실패 시 기본 메시지 생성"""

        # 단순한 기본 메시지
        content = {
            "opening": "Thank you for sharing your emotional journey with me.",
            "recognition": "Your courage in exploring these feelings is meaningful.",
            "personal_note": "Every step in emotional understanding is valuable.",
            "guidance": "Continue to be patient and compassionate with yourself.",
            "closing": "I'm here to support you on this path of growth.",
        }

        return {
            "message_id": f"curator_fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "user_id": context.get("user_id", "unknown"),
            "gallery_item_id": context.get("gallery_item_id", 0),
            "message_type": "curator_closure",
            "content": content,
            "personalization_data": {
                "coping_style": context.get("coping_style", "balanced"),
                "emotion_keywords": context.get("emotion_keywords", []),
                "guestbook_data": context.get("guestbook_data", {}),
                "personalization_level": "low",
                "generation_method": "fallback",
            },
            "metadata": {
                "gpt_model": "fallback",
                "gpt_tokens": {"total_tokens": 0},
                "safety_validated": True,
                "generation_time": 0,
                "fallback_used": True,
            },
            "created_date": datetime.now().isoformat(),
        }

    def validate_message_therapeutic_quality(
        self, message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """큐레이터 메시지의 치료적 품질 검증"""

        content = message.get("content", {})

        validation_result = {
            "is_therapeutic": True,
            "quality_score": 0,
            "issues": [],
            "recommendations": [],
        }

        # 구조 검증 (20점)
        required_sections = [
            "opening",
            "recognition",
            "personal_note",
            "guidance",
            "closing",
        ]
        present_sections = [
            section for section in required_sections if content.get(section, "").strip()
        ]
        structure_score = (len(present_sections) / len(required_sections)) * 20
        validation_result["quality_score"] += structure_score

        if len(present_sections) < len(required_sections):
            missing_sections = set(required_sections) - set(present_sections)
            validation_result["issues"].append(
                f"Missing sections: {', '.join(missing_sections)}"
            )

        # 내용 품질 검증 (40점)
        therapeutic_keywords = [
            "courage",
            "strength",
            "journey",
            "growth",
            "understanding",
            "healing",
            "support",
            "kindness",
            "wisdom",
            "progress",
            "reflection",
            "awareness",
        ]

        all_content = " ".join(content.values()).lower()
        therapeutic_keyword_count = sum(
            1 for keyword in therapeutic_keywords if keyword in all_content
        )
        content_score = min(40, therapeutic_keyword_count * 5)
        validation_result["quality_score"] += content_score

        # 개인화 수준 검증 (25점)
        personalization_data = message.get("personalization_data", {})
        guestbook_title = personalization_data.get("guestbook_data", {}).get(
            "title", ""
        )
        emotion_keywords = personalization_data.get("emotion_keywords", [])

        personalization_score = 0
        if guestbook_title and guestbook_title.lower() in all_content:
            personalization_score += 10
        if emotion_keywords and any(
            emotion.lower() in all_content for emotion in emotion_keywords
        ):
            personalization_score += 10
        if personalization_data.get("coping_style") != "balanced":  # 특화된 스타일 사용
            personalization_score += 5

        validation_result["quality_score"] += personalization_score

        # 안전성 검증 (15점)
        safety_score = 15  # 기본 점수
        unsafe_patterns = ["fix yourself", "you should", "you must", "wrong with you"]

        for pattern in unsafe_patterns:
            if pattern in all_content:
                safety_score -= 5
                validation_result["issues"].append(
                    f"Potentially directive language: {pattern}"
                )

        validation_result["quality_score"] += max(0, safety_score)

        # 최종 평가
        validation_result["quality_score"] = round(
            validation_result["quality_score"], 1
        )

        if validation_result["quality_score"] < 60:
            validation_result["is_therapeutic"] = False
            validation_result["recommendations"].append(
                "Message quality below therapeutic standard"
            )

        if validation_result["quality_score"] < 40:
            validation_result["recommendations"].append("Consider regenerating message")

        return validation_result

    def get_message_analytics(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """큐레이터 메시지 분석"""

        if not messages:
            return {"total_messages": 0}

        analytics = {
            "total_messages": len(messages),
            "by_coping_style": {},
            "avg_personalization_level": 0,
            "gpt_vs_fallback": {"gpt": 0, "fallback": 0},
            "quality_distribution": {"high": 0, "medium": 0, "low": 0},
            "common_themes": [],
            "user_engagement_indicators": {},
        }

        # 대처 스타일별 분석
        for message in messages:
            coping_style = message.get("personalization_data", {}).get(
                "coping_style", "unknown"
            )
            analytics["by_coping_style"][coping_style] = (
                analytics["by_coping_style"].get(coping_style, 0) + 1
            )

        # 생성 방법 분석
        for message in messages:
            generation_method = message.get("personalization_data", {}).get(
                "generation_method", "unknown"
            )
            if generation_method == "fallback":
                analytics["gpt_vs_fallback"]["fallback"] += 1
            else:
                analytics["gpt_vs_fallback"]["gpt"] += 1

        # 개인화 수준 평균
        personalization_levels = []
        for message in messages:
            level = message.get("personalization_data", {}).get(
                "personalization_level", "medium"
            )
            level_score = {"low": 1, "medium": 2, "high": 3}.get(level, 2)
            personalization_levels.append(level_score)

        if personalization_levels:
            analytics["avg_personalization_level"] = sum(personalization_levels) / len(
                personalization_levels
            )

        return analytics
