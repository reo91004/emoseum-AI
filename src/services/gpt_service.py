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
    logger.warning(
        "OpenAI 라이브러리를 사용할 수 없습니다. 시뮬레이션 모드로 실행됩니다."
    )

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

        # 토큰 및 비용 추적 (외부에서 주입받거나 자체 생성)
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
            "fallback": False,
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
            "fallback": False,
        }

    def analyze_emotion(
        self,
        diary_text: str,
        user_id: str = "anonymous",
        **kwargs,
    ) -> Dict[str, Any]:
        """일기 텍스트의 감정 분석"""

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
                # JSON 파싱 시도
                analysis_data = self._parse_emotion_analysis(response["content"])

                return {
                    "success": True,
                    "keywords": analysis_data.get("keywords", ["neutral"]),
                    "vad_scores": tuple(
                        analysis_data.get("vad_scores", [0.5, 0.5, 0.5])
                    ),
                    "confidence": analysis_data.get("confidence", 0.8),
                    "primary_emotion": analysis_data.get("primary_emotion", "neutral"),
                    "emotional_intensity": analysis_data.get(
                        "emotional_intensity", "medium"
                    ),
                    "token_usage": response["token_usage"],
                    "processing_time": response["processing_time"],
                }
            except Exception as e:
                logger.error(f"감정 분석 결과 파싱 실패: {e}")
                return {
                    "success": False,
                    "error": f"emotion_analysis_parsing_failed: {str(e)}",
                    "keywords": ["neutral"],
                    "vad_scores": (0.5, 0.5, 0.5),
                    "confidence": 0.0,
                }

        return {
            "success": False,
            "error": response.get("error", "감정 분석 실패"),
            "keywords": ["neutral"],
            "vad_scores": (0.5, 0.5, 0.5),
            "confidence": 0.0,
        }

    def _make_api_call(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 150,
        temperature: float = 0.7,
        purpose: str = "general",
        user_id: str = "anonymous",
    ) -> Dict[str, Any]:
        """OpenAI API 호출 (재시도 로직 포함)"""

        # 캐시 확인
        if self.cache_enabled:
            cache_key = self._generate_cache_key(messages, max_tokens, temperature)
            if cache_key in self.cache:
                cached_item = self.cache[cache_key]
                if time.time() - cached_item["timestamp"] < self.cache_ttl:
                    logger.info(f"캐시된 응답 반환 ({purpose})")
                    return cached_item["response"]

        # OpenAI 클라이언트가 없으면 시뮬레이션
        if not self.client:
            return self._simulate_api_call(messages, purpose, user_id)

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
        }

    def _simulate_api_call(
        self, messages: List[Dict[str, str]], purpose: str, user_id: str = "anonymous"
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
        elif purpose == "emotion_analysis":
            simulated_content = self._generate_simulated_emotion_analysis()
        else:
            simulated_content = "This is a simulated GPT response for testing purposes."

        # 시뮬레이션 토큰 사용량
        simulated_token_usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

        # 시뮬레이션 비용 추적
        self.cost_tracker.record_api_call(
            user_id=user_id,
            purpose=purpose,
            model=self.model,
            token_usage=simulated_token_usage,
            processing_time=1.0,
        )

        return {
            "success": True,
            "content": simulated_content,
            "token_usage": simulated_token_usage,
            "processing_time": 1.0,
            "model": self.model,
            "finish_reason": "stop",
            "simulated": True,
        }

    def _generate_simulated_prompt(self) -> str:
        """시뮬레이션 프롬프트 생성"""
        return "A serene landscape with soft clouds and gentle lighting, representing inner peace and emotional healing."

    def _generate_simulated_curator_message(self) -> str:
        """시뮬레이션 큐레이터 메시지 생성"""
        return """Thank you for sharing your emotional journey with us today.

Your courage in exploring these feelings shows tremendous strength. Through this artwork, you've created something beautiful from your experience.

This reflection demonstrates your growing ability to transform difficult emotions into meaningful expression.

Continue to trust in your inner wisdom as you move forward on this healing path."""

    def _generate_simulated_emotion_analysis(self) -> str:
        """시뮬레이션 감정 분석 JSON 생성"""
        return """{
    "keywords": ["contentment", "satisfaction", "joy"],
    "vad_scores": [0.7, 0.5, 0.6],
    "confidence": 0.85,
    "primary_emotion": "contentment",
    "emotional_intensity": "medium"
}"""

    def _create_prompt_engineering_system_message(
        self, coping_style: str, visual_preferences: Dict[str, Any]
    ) -> str:
        """프롬프트 엔지니어링용 시스템 메시지 생성"""

        base_instruction = """You are an expert image prompt engineer specializing in therapeutic art creation.
Your role is to transform emotional diary entries into powerful, healing visual prompts for image generation.

Guidelines:
- Keep prompts under 150 characters
- Focus on visual metaphors that promote healing
- Incorporate artistic elements that support emotional processing
- Ensure the imagery is appropriate for therapeutic contexts
- Use descriptive, artistic language that evokes emotion and beauty"""

        # 대처 스타일별 지침
        if coping_style == "avoidant":
            style_guidance = "\nStyle: Create gentle, soft imagery with protective elements. Use metaphors of safety, gradual light, and nurturing environments."
        elif coping_style == "confrontational":
            style_guidance = "\nStyle: Create bold, transformative imagery. Use metaphors of strength, breakthrough moments, and powerful natural forces."
        else:  # balanced
            style_guidance = "\nStyle: Create harmonious imagery that balances challenge and comfort. Use metaphors of growth, balance, and integrated wholeness."

        # 시각적 선호도 통합
        art_style = visual_preferences.get("art_style", "painting")
        color_tone = visual_preferences.get("color_tone", "warm")
        visual_elements = (
            f"\nPreferred visual elements: {art_style} style, {color_tone} colors"
        )

        return f"{base_instruction}{style_guidance}{visual_elements}"

    def _create_prompt_engineering_user_message(
        self, diary_text: str, emotion_keywords: List[str]
    ) -> str:
        """프롬프트 엔지니어링용 사용자 메시지 생성"""

        return f"""Transform this emotional diary entry into a therapeutic image prompt:

Diary Entry: "{diary_text}"

Key Emotions: {', '.join(emotion_keywords)}

Create a visual prompt that:
1. Captures the emotional essence metaphorically
2. Promotes healing and reflection
3. Is suitable for therapeutic art creation
4. Uses artistic, descriptive language
5. Stays under 150 characters

Visual Prompt:"""

    def _create_curator_system_message(
        self, coping_style: str, personalization_context: Dict[str, Any]
    ) -> str:
        """큐레이터 메시지용 시스템 메시지 생성"""

        base_instruction = """You are a compassionate art curator and therapeutic guide.
You provide personalized, healing messages that acknowledge the user's emotional journey and encourage continued growth.

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

    def get_usage_statistics(self, user_id: str = None) -> Dict[str, Any]:
        """사용량 통계 조회"""
        return self.cost_tracker.get_usage_statistics(user_id)

    def get_cost_summary(self, user_id: str = None) -> Dict[str, Any]:
        """비용 요약 조회"""
        return self.cost_tracker.get_cost_summary(user_id)

    def _parse_emotion_analysis(self, response_content: str) -> Dict[str, Any]:
        """감정 분석 응답 파싱"""
        try:
            # JSON 블록 찾기
            import re

            json_match = re.search(r"\{.*\}", response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed_data = json.loads(json_str)

                # 데이터 검증 및 정규화
                validated_data = {
                    "keywords": parsed_data.get("keywords", ["neutral"])[:5],  # 최대 5개
                    "vad_scores": self._validate_vad_scores(
                        parsed_data.get("vad_scores", [0.5, 0.5, 0.5])
                    ),
                    "confidence": max(
                        0.0, min(1.0, float(parsed_data.get("confidence", 0.8)))
                    ),
                    "primary_emotion": parsed_data.get("primary_emotion", "neutral"),
                    "emotional_intensity": parsed_data.get(
                        "emotional_intensity", "medium"
                    ),
                }

                return validated_data

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"JSON 파싱 실패: {e}")

        # 파싱 실패시 기본값 반환
        logger.warning("JSON 파싱 실패, 기본값 반환")
        return {
            "keywords": ["neutral"],
            "vad_scores": [0.5, 0.5, 0.5],
            "confidence": 0.1,
            "primary_emotion": "neutral",
            "emotional_intensity": "medium",
        }

    def _validate_vad_scores(self, vad_scores) -> List[float]:
        """VAD 점수 검증 및 정규화"""
        if not isinstance(vad_scores, (list, tuple)) or len(vad_scores) != 3:
            return [0.5, 0.5, 0.5]

        try:
            validated = []
            for score in vad_scores:
                # 0-1 범위로 정규화
                normalized = max(0.0, min(1.0, float(score)))
                validated.append(normalized)
            return validated
        except (ValueError, TypeError):
            return [0.5, 0.5, 0.5]

