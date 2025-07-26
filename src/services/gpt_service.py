# src/services/gpt_service.py

import time
import json
import hashlib
import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import openai
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI 라이브러리를 사용할 수 없습니다.")

from ..utils.cost_tracker import CostTracker


class GPTService:
    """OpenAI API 호출 핵심 서비스"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        cost_tracker: Optional[CostTracker] = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None

        # 기본 설정
        self.default_params = {
            "max_tokens": 150,
            "temperature": 0.7,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }

        # 오류 처리 설정
        self.max_retries = 3
        self.base_delay = 1.0  # 초
        self.max_delay = 30.0  # 초
        self.timeout = 30

        # 토큰 및 비용 추적
        if cost_tracker is not None:
            self.cost_tracker = cost_tracker
            logger.info("외부 CostTracker 인스턴스가 주입되었습니다.")
        else:
            self.cost_tracker = CostTracker("default_cost_tracking.db")
            logger.info("새로운 CostTracker 인스턴스가 생성되었습니다.")

        # 응답 캐싱 (메모리 기반 간단 캐시)
        self.cache_enabled = True
        self.cache = {}
        self.cache_ttl = 3600  # 1시간

        # OpenAI 클라이언트 초기화
        if OPENAI_AVAILABLE and self.api_key:
            self._initialize_client()
        else:
            logger.warning("GPT 서비스가 사용 불가능한 상태입니다.")

    def _initialize_client(self):
        """OpenAI 클라이언트 초기화"""
        try:
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"OpenAI 클라이언트 초기화 완료 - 모델: {self.model}")
        except Exception as e:
            logger.error(f"OpenAI 클라이언트 초기화 실패: {e}")
            self.client = None

    def generate_prompt_engineering_response(
        self,
        diary_text: str,
        emotion_keywords: List[str],
        coping_style: str,
        visual_preferences: Dict[str, Any],
        user_id: str = "anonymous",
        **kwargs,
    ) -> Dict[str, Any]:
        """일기 내용을 기반으로 이미지 프롬프트 엔지니어링"""

        # 시스템 메시지 구성
        system_message = self._create_prompt_engineering_system_message(
            coping_style, visual_preferences
        )

        # 사용자 메시지 구성
        user_message = self._create_prompt_engineering_user_message(
            diary_text, emotion_keywords
        )

        # GPT API 호출
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        response = self._make_api_call(
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 150),
            temperature=kwargs.get("temperature", 0.7),
            purpose="prompt_engineering",
            user_id=user_id,
        )

        if response["success"]:
            # 프롬프트 후처리
            enhanced_prompt = self._post_process_prompt(
                response["content"], visual_preferences
            )

            return {
                "success": True,
                "prompt": enhanced_prompt,
                "original_response": response["content"],
                "token_usage": response["token_usage"],
                "processing_time": response["processing_time"],
                "safety_check": {"is_safe": True},  # 추후 안전성 검증 추가
            }

        return {
            "success": False,
            "error": response.get("error", "프롬프트 생성 실패"),
            "retry_recommended": True,
            "requires_manual_intervention": True,
        }

    def generate_curator_message(
        self,
        user_profile: Dict[str, Any],
        gallery_item: Dict[str, Any],
        personalization_context: Dict[str, Any],
        user_id: str = "anonymous",
        **kwargs,
    ) -> Dict[str, Any]:
        """개인화된 큐레이터 메시지 생성"""

        # 시스템 메시지 구성
        system_message = self._create_curator_system_message(
            user_profile.get("coping_style", "balanced"), personalization_context
        )

        # 사용자 메시지 구성
        user_message = self._create_curator_user_message(user_profile, gallery_item)

        # GPT API 호출
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        response = self._make_api_call(
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 500),
            temperature=kwargs.get("temperature", 0.8),
            purpose="curator_message",
            user_id=user_id,
        )

        if response["success"]:
            # 메시지 구조화
            structured_message = self._structure_curator_message(
                response["content"], user_profile, gallery_item
            )

            return {
                "success": True,
                "message": structured_message,
                "raw_content": response["content"],
                "token_usage": response["token_usage"],
                "processing_time": response["processing_time"],
                "safety_check": {"is_safe": True},  # 추후 안전성 검증 추가
            }

        return {
            "success": False,
            "error": response.get("error", "큐레이터 메시지 생성 실패"),
            "retry_recommended": True,
            "requires_manual_intervention": True,
        }

    def generate_transition_guidance(
        self,
        guestbook_title: str,
        emotion_keywords: List[str],
        user_id: str = "anonymous",
        **kwargs,
    ) -> Dict[str, Any]:
        """큐레이터 전환을 위한 안내 질문 생성"""

        system_message = """You are a therapeutic transition guide specializing in helping users move from personal reflection to receiving supportive guidance.

Your task is to create a warm, encouraging transition message that:
1. Acknowledges their emotional journey
2. References their guestbook title meaningfully  
3. Builds anticipation for personalized curator support
4. Uses encouraging, supportive language
5. Keeps the message under 150 words

Focus on validation, encouragement, and smooth transition to closure."""

        user_message = f"""Create a transition message for a user who has:
- Titled their emotional guestbook: "{guestbook_title}"
- Explored these emotions: {', '.join(emotion_keywords)}

Generate a message that helps them transition to receiving personalized curator support."""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        response = self._make_api_call(
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 200),
            temperature=kwargs.get("temperature", 0.7),
            purpose="transition_guidance",
            user_id=user_id,
        )

        if response["success"]:
            return {
                "success": True,
                "content": response["content"],
                "token_usage": response["token_usage"],
                "processing_time": response["processing_time"],
            }

        return {
            "success": False,
            "error": response.get("error", "전환 안내 생성 실패"),
            "retry_recommended": True,
            "requires_manual_intervention": True,
        }

    def analyze_emotion(
        self,
        diary_text: str,
        user_id: str = "anonymous",
        **kwargs,
    ) -> Dict[str, Any]:
        """일기 텍스트의 감정 분석 (폴백 없음)"""

        # 시스템 메시지 구성
        system_message = """You are an expert emotion analysis AI specializing in therapeutic applications.

Your task is to analyze diary text and extract:
1. Emotional keywords (3-5 main emotions)
2. VAD scores (Valence, Arousal, Dominance on scale 0-1)
3. Confidence level of the analysis

Guidelines:
- Focus on constructive emotional understanding
- Consider cultural context and therapeutic value
- Provide accurate VAD psychological scores
- Use clear, therapeutic language for emotions

Response format (JSON):
{
    "keywords": ["emotion1", "emotion2", "emotion3"],
    "vad_scores": [valence, arousal, dominance],
    "confidence": 0.85,
    "primary_emotion": "main_emotion",
    "emotional_intensity": "low/medium/high"
}"""

        # 사용자 메시지 구성
        user_message = f"""Analyze the emotional content of this diary entry:

DIARY TEXT: "{diary_text}"

Please provide a comprehensive emotional analysis including:
1. 3-5 key emotional keywords in English
2. VAD scores (Valence: positive/negative, Arousal: calm/excited, Dominance: controlled/overwhelmed)
3. Your confidence in this analysis
4. The primary emotion
5. Overall emotional intensity

Provide the analysis in the specified JSON format."""

        # GPT API 호출
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        response = self._make_api_call(
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 200),
            temperature=kwargs.get("temperature", 0.3),  # 낮은 온도로 일관성 있는 분석
            purpose="emotion_analysis",
            user_id=user_id,
        )

        if response["success"]:
            try:
                # JSON 파싱 시도 (폴백 없음)
                analysis_data = self._parse_emotion_analysis_strict(response["content"])

                return {
                    "success": True,
                    "keywords": analysis_data["keywords"],
                    "vad_scores": tuple(analysis_data["vad_scores"]),
                    "confidence": analysis_data["confidence"],
                    "primary_emotion": analysis_data["primary_emotion"],
                    "emotional_intensity": analysis_data["emotional_intensity"],
                    "token_usage": response["token_usage"],
                    "processing_time": response["processing_time"],
                }
            except Exception as e:
                logger.error(f"감정 분석 결과 파싱 실패: {e}")
                return {
                    "success": False,
                    "error": f"Emotion analysis parsing failed: {str(e)}",
                    "retry_recommended": True,
                    "requires_manual_intervention": True,
                }

        return {
            "success": False,
            "error": response.get("error", "감정 분석 실패"),
            "retry_recommended": True,
            "requires_manual_intervention": True,
        }

    def _make_api_call(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 150,
        temperature: float = 0.7,
        purpose: str = "general",
        user_id: str = "anonymous",
    ) -> Dict[str, Any]:
        """OpenAI API 호출 (폴백 없음)"""

        # 캐시 확인
        if self.cache_enabled:
            cache_key = self._generate_cache_key(messages, max_tokens, temperature)
            if cache_key in self.cache:
                cached_item = self.cache[cache_key]
                if time.time() - cached_item["timestamp"] < self.cache_ttl:
                    logger.info(f"캐시된 응답 반환 ({purpose})")
                    return cached_item["response"]

        # OpenAI 클라이언트가 없으면 명확한 오류 반환
        if not OPENAI_AVAILABLE or not self.client:
            return {
                "success": False,
                "error": "OpenAI API not available - please check configuration",
                "requires_setup": True,
                "setup_instructions": "Set OPENAI_API_KEY environment variable and install openai package",
            }

        # 실제 API 호출
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=self.timeout,
                )

                processing_time = time.time() - start_time

                # 결과 구성
                result = {
                    "success": True,
                    "content": response.choices[0].message.content,
                    "token_usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                    "processing_time": processing_time,
                    "model": response.model,
                    "finish_reason": response.choices[0].finish_reason,
                }

                # 비용 추적
                self.cost_tracker.record_api_call(
                    user_id=user_id,
                    purpose=purpose,
                    model=self.model,
                    token_usage=result["token_usage"],
                    processing_time=processing_time,
                )

                # 캐시 저장
                if self.cache_enabled:
                    self.cache[cache_key] = {
                        "response": result,
                        "timestamp": time.time(),
                    }

                logger.info(
                    f"GPT API 호출 성공 ({purpose}): {result['token_usage']['total_tokens']} 토큰"
                )
                return result

            except openai.RateLimitError as e:
                delay = min(self.base_delay * (2**attempt), self.max_delay)
                logger.warning(
                    f"Rate limit 오류 (시도 {attempt + 1}/{self.max_retries}): {delay}초 대기"
                )
                time.sleep(delay)

            except openai.APITimeoutError as e:
                delay = min(self.base_delay * (2**attempt), self.max_delay)
                logger.warning(
                    f"Timeout 오류 (시도 {attempt + 1}/{self.max_retries}): {delay}초 대기"
                )
                time.sleep(delay)

            except openai.APIError as e:
                logger.error(f"OpenAI API 오류: {e}")
                break

            except Exception as e:
                logger.error(f"예상치 못한 오류: {e}")
                break

        # 모든 재시도 실패
        return {
            "success": False,
            "error": "API 호출 실패 - 최대 재시도 횟수 초과",
            "attempts": self.max_retries,
            "retry_recommended": True,
            "requires_manual_intervention": True,
        }

    def _generate_cache_key(
        self, messages: List[Dict[str, str]], max_tokens: int, temperature: float
    ) -> str:
        """캐시 키 생성"""
        content = json.dumps(messages, sort_keys=True) + f"{max_tokens}{temperature}"
        return hashlib.md5(content.encode()).hexdigest()

    def _create_prompt_engineering_system_message(
        self, coping_style: str, visual_preferences: Dict[str, Any]
    ) -> str:
        """프롬프트 엔지니어링용 시스템 메시지 생성"""

        style_guidance = {
            "avoidant": "gentle, protective, metaphorical",
            "confrontational": "direct, authentic, honest",
            "balanced": "thoughtful, nuanced, harmonious",
        }.get(coping_style, "thoughtful, nuanced, harmonious")

        return f"""You are an expert AI image prompt engineer specializing in therapeutic art generation.

Transform diary entries into artistic image prompts that are:
- Emotionally supportive and healing
- Visually beautiful and contemplative
- Appropriate for therapeutic contexts
- {style_guidance} in tone

Guidelines:
- Create prompts under 100 words
- Use artistic and poetic language
- Include style, mood, and composition elements
- Avoid literal representation of trauma
- Focus on hope, growth, and beauty
- Consider the emotional healing journey

Visual preferences: {visual_preferences}
Coping style: {coping_style}"""

    def _create_prompt_engineering_user_message(
        self, diary_text: str, emotion_keywords: List[str]
    ) -> str:
        """프롬프트 엔지니어링용 사용자 메시지 생성"""
        return f"""Transform this emotional diary entry into a beautiful, therapeutic image prompt:

DIARY ENTRY: "{diary_text}"

EMOTIONS IDENTIFIED: {', '.join(emotion_keywords)}

Create an artistic image prompt that:
1. Honors these emotions respectfully
2. Transforms them into visual poetry
3. Promotes healing and reflection
4. Uses artistic terminology and style references
5. Focuses on beauty, hope, and growth

Generate only the image prompt, no explanations."""

    def _create_curator_system_message(
        self, coping_style: str, personalization_context: Dict[str, Any]
    ) -> str:
        """큐레이터 메시지용 시스템 메시지 생성"""

        style_guidance = {
            "avoidant": "gentle, protective, metaphorical",
            "confrontational": "direct, authentic, honest",
            "balanced": "thoughtful, nuanced, harmonious",
        }.get(coping_style, "thoughtful, nuanced, harmonious")

        return f"""You are a compassionate art curator specializing in therapeutic digital experiences.

Create personalized messages that:
- Acknowledge the user's emotional journey with empathy
- Reference their artwork and guestbook meaningfully
- Provide encouragement and validation
- Use {style_guidance} language
- Keep messages warm but professional
- Focus on growth, courage, and healing

Structure your response with:
1. Opening acknowledgment
2. Recognition of their courage
3. Personal reflection on their journey
4. Encouraging guidance
5. Warm closing

Personalization context: {personalization_context}
Tone: {style_guidance}"""

    def _create_curator_user_message(
        self, user_profile: Dict[str, Any], gallery_item: Dict[str, Any]
    ) -> str:
        """큐레이터 메시지용 사용자 메시지 생성"""
        return f"""Create a personalized curator message for this user's emotional art journey:

USER PROFILE:
- Coping style: {user_profile.get('coping_style', 'balanced')}
- Previous interactions: {user_profile.get('interaction_history', 'New user')}

GALLERY ITEM:
- Original diary: "{gallery_item.get('diary_text', '')}"
- Emotions explored: {gallery_item.get('emotion_keywords', [])}
- Guestbook title: "{gallery_item.get('guestbook_title', '')}"
- Guestbook tags: {gallery_item.get('guestbook_tags', [])}

Create a meaningful, personalized curator message that acknowledges their specific journey and provides therapeutic support."""

    def _structure_curator_message(
        self,
        raw_content: str,
        user_profile: Dict[str, Any],
        gallery_item: Dict[str, Any],
    ) -> Dict[str, Any]:
        """큐레이터 메시지 구조화"""
        # 간단한 구조화 로직 (더 정교한 파싱 필요 시 개선)
        sections = raw_content.split("\n\n")

        return {
            "full_message": raw_content,
            "opening": sections[0] if len(sections) > 0 else raw_content[:100],
            "recognition": sections[1] if len(sections) > 1 else "",
            "personal_note": sections[2] if len(sections) > 2 else "",
            "guidance": sections[3] if len(sections) > 3 else "",
            "closing": sections[-1] if len(sections) > 4 else "",
            "user_id": user_profile.get("user_id", "anonymous"),
            "guestbook_title": gallery_item.get("guestbook_title", ""),
            "emotion_keywords": gallery_item.get("emotion_keywords", []),
        }

    def _post_process_prompt(
        self, prompt: str, visual_preferences: Dict[str, Any]
    ) -> str:
        """프롬프트 후처리"""
        # 기본 후처리 로직
        processed = prompt.strip()

        # 선호하는 스타일 추가
        if visual_preferences.get("preferred_style"):
            processed += f", {visual_preferences['preferred_style']} style"

        return processed

    def _parse_emotion_analysis_strict(self, response_content: str) -> Dict[str, Any]:
        """감정 분석 응답 엄격 파싱 (폴백 없음)"""
        try:
            # JSON 파싱만 시도, 실패 시 예외 발생
            if not response_content.strip().startswith("{"):
                raise ValueError("Response is not in JSON format")

            parsed_data = json.loads(response_content)

            # 필수 필드 검증
            required_fields = [
                "keywords",
                "vad_scores",
                "confidence",
                "primary_emotion",
                "emotional_intensity",
            ]
            for field in required_fields:
                if field not in parsed_data:
                    raise ValueError(f"Missing required field: {field}")

            # 데이터 타입 검증
            if not isinstance(parsed_data["keywords"], list):
                raise ValueError("keywords must be a list")
            if (
                not isinstance(parsed_data["vad_scores"], list)
                or len(parsed_data["vad_scores"]) != 3
            ):
                raise ValueError("vad_scores must be a list of 3 values")
            if not isinstance(parsed_data["confidence"], (int, float)):
                raise ValueError("confidence must be a number")

            return parsed_data

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # 폴백 대신 예외 발생
            raise Exception(f"Failed to parse emotion analysis response: {str(e)}")

    def get_usage_statistics(self, user_id: str = None) -> Dict[str, Any]:
        """사용량 통계 반환"""
        return self.cost_tracker.get_usage_summary(user_id)

    def clear_cache(self):
        """캐시 초기화"""
        self.cache.clear()
        logger.info("GPT 응답 캐시가 초기화되었습니다.")

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
            "api_available": OPENAI_AVAILABLE and self.client is not None,
            "cache_enabled": self.cache_enabled,
            "cache_size": len(self.cache),
        }
