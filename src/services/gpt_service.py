# src/services/gpt_service.py

import time
import json
import hashlib
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
    logger.warning(
        "OpenAI 라이브러리를 사용할 수 없습니다. 시뮬레이션 모드로 실행됩니다."
    )

from ..utils.cost_tracker import CostTracker


class GPTService:
    """OpenAI API 호출 핵심 서비스"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key
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
        self.cost_tracker = CostTracker()

        # 응답 캐싱 (메모리 기반 간단 캐시)
        self.cache_enabled = True
        self.cache = {}
        self.cache_ttl = 3600  # 1시간

        # OpenAI 클라이언트 초기화
        if OPENAI_AVAILABLE and api_key:
            self._initialize_client()
        else:
            logger.warning("GPT 서비스가 시뮬레이션 모드로 실행됩니다.")

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
        )

        if response["success"]:
            # 프롬프트 후처리
            enhanced_prompt = self._post_process_prompt(
                response["content"], visual_preferences
            )

            return {
                "success": True,
                "enhanced_prompt": enhanced_prompt,
                "original_response": response["content"],
                "token_usage": response.get("token_usage", {}),
                "processing_time": response.get("processing_time", 0),
                "metadata": {
                    "coping_style": coping_style,
                    "emotion_keywords": emotion_keywords,
                    "visual_preferences": visual_preferences,
                },
            }
        else:
            return {
                "success": False,
                "error": response["error"],
                "fallback_needed": True,
            }

    def generate_curator_message(
        self,
        user_profile: Dict[str, Any],
        gallery_item: Dict[str, Any],
        personalization_context: Dict[str, Any],
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
            max_tokens=kwargs.get("max_tokens", 300),
            temperature=kwargs.get("temperature", 0.8),
            purpose="curator_message",
        )

        if response["success"]:
            # 큐레이터 메시지 구조화
            structured_message = self._structure_curator_message(
                response["content"], user_profile, gallery_item
            )

            return {
                "success": True,
                "curator_message": structured_message,
                "original_response": response["content"],
                "token_usage": response.get("token_usage", {}),
                "processing_time": response.get("processing_time", 0),
                "personalization_data": personalization_context,
            }
        else:
            return {
                "success": False,
                "error": response["error"],
                "fallback_needed": True,
            }

    def _make_api_call(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 150,
        temperature: float = 0.7,
        purpose: str = "general",
        **kwargs,
    ) -> Dict[str, Any]:
        """OpenAI API 호출 (재시도 로직 포함)"""

        if not self.client:
            return self._simulate_api_call(messages, purpose)

        # 캐시 확인
        cache_key = self._generate_cache_key(messages, max_tokens, temperature)
        if self.cache_enabled and cache_key in self.cache:
            cached_response = self.cache[cache_key]
            if time.time() - cached_response["timestamp"] < self.cache_ttl:
                logger.info("캐시된 응답을 반환합니다.")
                return cached_response["response"]

        # API 호출 준비
        api_params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": kwargs.get("top_p", self.default_params["top_p"]),
            "frequency_penalty": kwargs.get(
                "frequency_penalty", self.default_params["frequency_penalty"]
            ),
            "presence_penalty": kwargs.get(
                "presence_penalty", self.default_params["presence_penalty"]
            ),
        }

        start_time = time.time()

        for attempt in range(self.max_retries):
            try:
                # API 호출
                response = self.client.chat.completions.create(**api_params)

                processing_time = time.time() - start_time

                # 응답 구성
                result = {
                    "success": True,
                    "content": response.choices[0].message.content.strip(),
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
        }

    def _simulate_api_call(
        self, messages: List[Dict[str, str]], purpose: str
    ) -> Dict[str, Any]:
        """API 호출 시뮬레이션 (라이브러리 부족시)"""

        logger.info(f"GPT API 호출을 시뮬레이션합니다 ({purpose})")

        # 시뮬레이션 지연
        time.sleep(1)

        # 목적에 따른 시뮬레이션 응답
        if purpose == "prompt_engineering":
            simulated_content = self._generate_simulated_prompt()
        elif purpose == "curator_message":
            simulated_content = self._generate_simulated_curator_message()
        else:
            simulated_content = "This is a simulated GPT response for testing purposes."

        return {
            "success": True,
            "content": simulated_content,
            "token_usage": {
                "prompt_tokens": 50,
                "completion_tokens": 30,
                "total_tokens": 80,
            },
            "processing_time": 1.0,
            "model": f"{self.model}-simulated",
            "finish_reason": "stop",
            "simulated": True,
        }

    def _generate_simulated_prompt(self) -> str:
        """프롬프트 엔지니어링 시뮬레이션 응답"""
        return "peaceful emotional landscape reflecting gentle contemplation, oil painting style, warm colors, balanced composition, accepting atmosphere"

    def _generate_simulated_curator_message(self) -> str:
        """큐레이터 메시지 시뮬레이션 응답"""
        return """Thank you for sharing this emotional journey with us. Your courage in exploring these feelings is truly admirable. 

This experience you've created shows a deep understanding of your inner world. The way you've expressed your emotions through this artistic process demonstrates real growth.

Continue to honor your feelings with this same gentle honesty. Your journey of self-discovery is valuable and meaningful."""

    def _create_prompt_engineering_system_message(
        self, coping_style: str, visual_preferences: Dict[str, Any]
    ) -> str:
        """프롬프트 엔지니어링용 시스템 메시지 생성"""

        base_instruction = """You are an expert AI prompt engineer specializing in therapeutic art generation. Your task is to enhance emotional diary entries into optimized Stable Diffusion prompts for therapeutic image generation.

Key Guidelines:
1. Transform emotional content into visual metaphors
2. Maintain therapeutic safety and positivity
3. Keep prompts under 150 characters
4. Ensure complete sentences
5. Focus on artistic and healing imagery"""

        # 대처 스타일별 조정
        if coping_style == "avoidant":
            style_guidance = "\n- Use gentle, soft, and non-threatening imagery\n- Prefer abstract and metaphorical representations\n- Avoid direct emotional confrontation"
        elif coping_style == "confrontational":
            style_guidance = "\n- Use direct, clear, and honest imagery\n- Allow for emotional intensity while maintaining hope\n- Emphasize transformation and growth"
        else:  # balanced
            style_guidance = "\n- Balance between gentle and direct approaches\n- Use harmonious and integrated imagery\n- Emphasize emotional equilibrium"

        # 시각적 선호도 통합
        visual_guidance = f"""
Visual Style Preferences:
- Art style: {visual_preferences.get('art_style', 'painting')}
- Color tone: {visual_preferences.get('color_tone', 'warm')}
- Complexity: {visual_preferences.get('complexity', 'balanced')}"""

        return f"{base_instruction}{style_guidance}{visual_guidance}"

    def _create_prompt_engineering_user_message(
        self, diary_text: str, emotion_keywords: List[str]
    ) -> str:
        """프롬프트 엔지니어링용 사용자 메시지 생성"""

        return f"""Transform this emotional diary entry into an enhanced Stable Diffusion prompt:

Diary Text: "{diary_text}"

Identified Emotions: {', '.join(emotion_keywords)}

Please create a therapeutic art prompt that:
1. Captures the emotional essence
2. Uses healing visual metaphors
3. Maintains hope and growth potential
4. Stays under 150 characters
5. Forms complete sentences

Enhanced Prompt:"""

    def _create_curator_system_message(
        self, coping_style: str, personalization_context: Dict[str, Any]
    ) -> str:
        """큐레이터 메시지용 시스템 메시지 생성"""

        base_instruction = """You are a wise, empathetic art curator specializing in therapeutic emotional art. You provide personalized, healing messages that acknowledge the user's emotional journey and encourage continued growth.

Message Structure:
1. Acknowledgment of courage/effort
2. Recognition of growth or insight
3. Personal connection to their specific experience
4. Gentle guidance for moving forward
5. Warm, supportive closing

Guidelines:
- Be genuine and specific, not generic
- Acknowledge both struggle and strength
- Maintain hope without dismissing difficulty
- Use warm, professional therapeutic language
- Keep messages to 3-4 sentences per section"""

        # 대처 스타일별 톤 조정
        if coping_style == "avoidant":
            tone_guidance = "\nTone: Gentle, soft, protective. Use indirect language and metaphors. Emphasize safety and gradual progress."
        elif coping_style == "confrontational":
            tone_guidance = "\nTone: Direct, honest, empowering. Use clear language and acknowledge strength. Emphasize courage and transformation."
        else:  # balanced
            tone_guidance = "\nTone: Balanced, wise, harmonious. Use both gentle and clear language. Emphasize integration and wisdom."

        # 개인화 컨텍스트
        personalization = f"""
Personalization Context:
- Previous positive reactions: {personalization_context.get('positive_reactions', 0)}
- Engagement level: {personalization_context.get('engagement_level', 'new')}
- Preferred message elements: {', '.join(personalization_context.get('preferred_elements', []))}"""

        return f"{base_instruction}{tone_guidance}{personalization}"

    def _create_curator_user_message(
        self, user_profile: Dict[str, Any], gallery_item: Dict[str, Any]
    ) -> str:
        """큐레이터 메시지용 사용자 메시지 생성"""

        return f"""Create a personalized curator message for this user's emotional art journey:

User Information:
- Coping style: {user_profile.get('coping_style', 'balanced')}
- Time with system: {user_profile.get('member_days', 'new user')}
- Previous artworks: {user_profile.get('total_artworks', 0)}

Current Artwork:
- Diary text: "{gallery_item.get('diary_text', '')}"
- Emotions explored: {', '.join(gallery_item.get('emotion_keywords', []))}
- Guestbook title: "{gallery_item.get('guestbook_title', '')}"
- Tags: {', '.join(gallery_item.get('guestbook_tags', []))}

Please create a message that:
1. Acknowledges their specific emotional journey
2. Recognizes the courage shown in this artwork
3. Connects to their unique way of processing emotions
4. Offers gentle guidance for continued growth
5. Feels personal and meaningful

Curator Message:"""

    def _post_process_prompt(
        self, raw_prompt: str, visual_preferences: Dict[str, Any]
    ) -> str:
        """프롬프트 후처리 및 완성"""

        # 기본 정리
        prompt = raw_prompt.strip()

        # 150자 제한 확인
        if len(prompt) > 150:
            # 마지막 완전한 문장까지만 유지
            sentences = prompt.split(".")
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence + ".") <= 150:
                    truncated += sentence + "."
                else:
                    break
            prompt = truncated.rstrip(".")

        # 문장 완성 확인
        if not prompt.endswith((".", "!", "?")):
            prompt += "."

        # 시각적 요소 강화
        if len(prompt) < 130:  # 여유가 있으면 시각적 요소 추가
            art_style = visual_preferences.get("art_style", "painting")
            if art_style not in prompt.lower():
                prompt = f"{prompt.rstrip('.')} in {art_style} style."

        # 품질 키워드 추가
        if len(prompt) < 140:
            prompt = f"{prompt.rstrip('.')}, high quality."

        return prompt

    def _structure_curator_message(
        self,
        raw_content: str,
        user_profile: Dict[str, Any],
        gallery_item: Dict[str, Any],
    ) -> Dict[str, str]:
        """큐레이터 메시지를 구조화된 형태로 변환"""

        # 간단한 구조화 (실제로는 더 정교한 파싱이 필요할 수 있음)
        lines = [line.strip() for line in raw_content.split("\n") if line.strip()]

        structured = {
            "opening": "",
            "recognition": "",
            "personal_note": "",
            "guidance": "",
            "closing": "",
        }

        # 간단한 휴리스틱으로 구조화
        if len(lines) >= 3:
            structured["opening"] = lines[0]
            structured["recognition"] = lines[1] if len(lines) > 1 else ""
            structured["personal_note"] = lines[2] if len(lines) > 2 else ""
            structured["guidance"] = lines[3] if len(lines) > 3 else ""
            structured["closing"] = lines[4] if len(lines) > 4 else ""
        else:
            # 짧은 응답의 경우 전체를 opening으로
            structured["opening"] = raw_content

        return structured

    def _generate_cache_key(
        self, messages: List[Dict[str, str]], max_tokens: int, temperature: float
    ) -> str:
        """캐시 키 생성"""
        content = json.dumps(messages, sort_keys=True) + f"{max_tokens}{temperature}"
        return hashlib.md5(content.encode()).hexdigest()

    def clear_cache(self):
        """캐시 초기화"""
        self.cache.clear()
        logger.info("GPT 응답 캐시가 초기화되었습니다.")

    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        total_size = len(self.cache)
        current_time = time.time()

        expired_count = sum(
            1
            for item in self.cache.values()
            if current_time - item["timestamp"] > self.cache_ttl
        )

        return {
            "total_cached": total_size,
            "expired_items": expired_count,
            "valid_items": total_size - expired_count,
            "cache_ttl_hours": self.cache_ttl / 3600,
        }

    def get_service_status(self) -> Dict[str, Any]:
        """서비스 상태 반환"""
        return {
            "client_initialized": self.client is not None,
            "openai_available": OPENAI_AVAILABLE,
            "model": self.model,
            "cache_enabled": self.cache_enabled,
            "cache_stats": self.get_cache_stats(),
            "cost_tracking": self.cost_tracker.get_daily_summary(),
            "default_params": self.default_params,
        }
