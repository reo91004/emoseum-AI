# src/services/gpt_service.py

# ==============================================================================
# 이 파일은 OpenAI의 GPT API와의 모든 통신을 담당하는 핵심 서비스이다.
# `prompt_engineer`, `docent_gpt` 등 다른 AI 관련 모듈로부터 요청을 받아,
# `config/gpt_prompts.yaml`에서 정의된 템플릿을 사용하여 API에 전달할 최종 메시지를 구성하고 API를 호출한다.
# 또한, `cost_tracker`를 통해 API 사용량과 비용을 기록하는 역할도 수행한다.
# ==============================================================================

import time
import json
import hashlib
import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from pathlib import Path
import yaml

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
        gpt_prompts_path: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None
        self.gpt_prompts_path = Path(gpt_prompts_path) if gpt_prompts_path else None

        # YAML 파일에서 프롬프트 템플릿 로드
        if self.gpt_prompts_path and self.gpt_prompts_path.exists():
            self.prompt_templates = self._load_prompt_templates_from_yaml()
            logger.info(
                f"프롬프트 템플릿을 YAML 파일에서 로드했습니다: {self.gpt_prompts_path}"
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

    def _load_prompt_templates_from_yaml(self) -> Dict[str, Any]:
        """YAML 파일에서 프롬프트 템플릿 로드"""
        try:
            with open(self.gpt_prompts_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)

            return yaml_data

        except Exception as e:
            logger.error(f"YAML 프롬프트 템플릿 로드 실패: {e}")
            raise

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

    def generate_docent_message(
        self,
        user_profile: Dict[str, Any],
        gallery_item: Dict[str, Any],
        personalization_context: Dict[str, Any],
        user_id: str = "anonymous",
        **kwargs,
    ) -> Dict[str, Any]:
        """개인화된 도슨트 메시지 생성"""

        # 시스템 메시지 구성
        system_message = self._create_docent_system_message(
            user_profile.get("coping_style", "balanced"), personalization_context
        )

        # 사용자 메시지 구성
        user_message = self._create_docent_user_message(user_profile, gallery_item)

        # GPT API 호출
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        response = self._make_api_call(
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 500),
            temperature=kwargs.get("temperature", 0.8),
            purpose="docent_message",
            user_id=user_id,
        )

        if response["success"]:
            # 메시지 구조화
            structured_message = self._structure_docent_message(
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
            "error": response.get("error", "도슨트 메시지 생성 실패"),
            "retry_recommended": True,
            "requires_manual_intervention": True,
        }

    def generate_transition_guidance(
        self,
        artwork_title: str,
        emotion_keywords: List[str],
        user_id: str = "anonymous",
        **kwargs,
    ) -> Dict[str, Any]:
        """도슨트 전환을 위한 안내 질문 생성"""

        system_message = self.prompt_templates["system_message_templates"][
            "transition_guidance"
        ]

        user_template = self.prompt_templates["user_message_templates"][
            "transition_guidance"
        ]
        user_message = user_template.format(
            artwork_title=artwork_title,
            emotion_keywords=", ".join(emotion_keywords),
        )

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
        """일기 텍스트의 감정 분석"""

        # YAML에서 메시지 템플릿 가져오기
        system_message = self.prompt_templates["system_message_templates"]["emotion_analysis"]
        
        user_template = self.prompt_templates["user_message_templates"]["emotion_analysis"]
        user_message = user_template.format(diary_text=diary_text)

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
                logger.info(f"GPT 감정 분석 응답: {response['content']}")
                
                # JSON 파싱 시도
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
                logger.error(f"GPT 응답 내용: {response['content']}")
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
        """OpenAI API 호출"""

        # 캐시 확인
        if self.cache_enabled:
            cache_key = self._generate_cache_key(messages, max_tokens, temperature)
            if cache_key in self.cache:
                cached_item = self.cache[cache_key]
                if time.time() - cached_item["timestamp"] < self.cache_ttl:
                    logger.info(f"캐시된 응답 반환 ({purpose})")
                    return cached_item["response"]

        # OpenAI 클라이언트가 없으면 오류 반환
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

                # 비용 추적 (오류가 발생해도 API 결과는 반환)
                try:
                    self.cost_tracker.record_api_call(
                        user_id=user_id,
                        purpose=purpose,
                        model=self.model,
                        prompt_tokens=result["token_usage"]["prompt_tokens"],
                        completion_tokens=result["token_usage"]["completion_tokens"],
                        processing_time=processing_time,
                        success=True
                    )
                except Exception as cost_error:
                    logger.warning(f"비용 추적 실패 (API 결과는 정상): {cost_error}")

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
                # 실패한 API 호출도 추적
                if self.cost_tracker:
                    self.cost_tracker.record_api_call(
                        user_id=user_id,
                        purpose=purpose,
                        model=self.model,
                        prompt_tokens=0,
                        completion_tokens=0,
                        processing_time=processing_time,
                        success=False,
                        error_message=str(e)
                    )
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

        # YAML에서 프롬프트 엔지니어링 시스템 메시지 가져오기
        prompt_data = self.prompt_templates["prompt_engineering"]
        base_message = prompt_data[coping_style]["system_message"]

        # 추가 컨텍스트 정보 첨부
        return f"""{base_message}

Visual preferences: {visual_preferences}
Coping style: {coping_style}"""

    def _create_prompt_engineering_user_message(
        self, diary_text: str, emotion_keywords: List[str]
    ) -> str:
        """프롬프트 엔지니어링용 사용자 메시지 생성"""
        template = self.prompt_templates["user_message_templates"]["prompt_engineering"]
        return template.format(
            diary_text=diary_text, emotion_keywords=", ".join(emotion_keywords)
        )

    def _create_docent_system_message(
        self, coping_style: str, personalization_context: Dict[str, Any]
    ) -> str:
        """도슨트 메시지용 시스템 메시지 생성"""

        # YAML에서 도슨트 시스템 메시지 가져오기
        docent_data = self.prompt_templates["docent_messages"]
        base_message = docent_data[coping_style]["system_message"]

        # 추가 컨텍스트 정보 첨부
        return f"""{base_message}

Personalization context: {personalization_context}
Tone: {docent_data[coping_style].get('tone', 'balanced')}"""

    def _create_docent_user_message(
        self, user_profile: Dict[str, Any], gallery_item: Dict[str, Any]
    ) -> str:
        """도슨트 메시지용 사용자 메시지 생성"""
        template = self.prompt_templates["user_message_templates"]["docent_message"]
        return template.format(
            user_nickname=user_profile.get("user_id", "friend"),
            coping_style=user_profile.get("coping_style", "balanced"),
            interaction_history=user_profile.get("interaction_history", "New user"),
            diary_text=gallery_item.get("diary_text", ""),
            emotion_keywords=gallery_item.get("emotion_keywords", []),
            artwork_title=gallery_item.get("artwork_title", ""),
        )

    def _structure_docent_message(
        self,
        raw_content: str,
        user_profile: Dict[str, Any],
        gallery_item: Dict[str, Any],
    ) -> Dict[str, Any]:
        """도슨트 메시지 구조화 - YAML 설정 기반"""
        # YAML에서 메시지 구조 설정 가져오기
        message_structure = self.prompt_templates.get("message_structure", {})
        five_section_format = message_structure.get("five_section_format", {})
        response_format = message_structure.get("response_format", {})
        
        # 구조화 방법 결정
        delimiter = response_format.get("section_delimiter", "\n\n")
        expected_sections = response_format.get("expected_sections", 5)
        parsing_method = response_format.get("parsing_method", "paragraph_split")
        
        if parsing_method == "paragraph_split":
            sections = raw_content.split(delimiter)
        else:
            # 폴백 방법: 단일 블록으로 처리
            sections = [raw_content]

        # 섹션 매핑 (YAML에서 정의된 구조 순서대로)
        section_names = list(five_section_format.keys()) if five_section_format else [
            "opening", "recognition", "personal_note", "guidance", "closing"
        ]
        
        result = {
            "full_message": raw_content,
            "user_id": user_profile.get("user_id", "anonymous"),
            "artwork_title": gallery_item.get("artwork_title", ""),
            "emotion_keywords": gallery_item.get("emotion_keywords", []),
        }
        
        # 구조화된 섹션 매핑
        for i, section_name in enumerate(section_names):
            if i < len(sections):
                result[section_name] = sections[i].strip()
            else:
                result[section_name] = ""
        
        # 마지막 섹션이 closing인 경우 특별 처리
        if len(sections) > len(section_names):
            result["closing"] = sections[-1].strip()
            
        return result

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
        """감정 분석 응답 엄격 파싱"""
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
            "gpt_integration": "complete",
            "language_consistency": "english_only",
            "error_handling": "graceful_failure",
            "handover_status": "completed",
            "api_available": OPENAI_AVAILABLE and self.client is not None,
            "cache_enabled": self.cache_enabled,
            "cache_size": len(self.cache),
        }

    def generate_artwork_description(
        self,
        diary_text: str,
        emotion_keywords: List[str],
        image_prompt: str,
        artwork_title: str = "",
        user_id: str = "anonymous",
        max_tokens: int = 50,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """작품 설명 생성 (미술관 스타일)"""
        try:
            logger.info(f"작품 설명 생성 시작: 제목='{artwork_title}', 키워드={emotion_keywords}")
            
            # YAML에서 artwork description 설정 로드
            artwork_config = self.prompt_templates.get("artwork_description", {})
            system_message = artwork_config.get("system_message", "")

            if not system_message:
                logger.error("YAML에서 artwork_description system_message를 찾을 수 없음")
                raise ValueError("Artwork description system message not found in YAML")
            
            logger.info(f"System message 로드 성공: {len(system_message)} 문자")

            # 사용자 메시지 구성
            user_message = f"""Based on the following information, create a museum-style description for this artwork:

Diary excerpt: "{diary_text[:200]}..."
Emotion keywords: {', '.join(emotion_keywords)}
Image prompt: "{image_prompt[:150]}..."
Artwork title: "{artwork_title}"

Generate a single, elegant sentence (15-25 words) that captures the emotional essence and artistic elements of this work."""

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]

            # API 호출
            logger.info(f"GPT API 호출 시작: max_tokens={max_tokens}, temperature={temperature}")
            response = self._make_api_call(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                purpose="artwork_description"
            )

            logger.info(f"GPT API 응답: success={response.get('success', False)}")
            if not response.get("success", False):
                logger.error(f"GPT API 호출 실패: {response.get('error', 'API call failed')}")
                raise Exception(response.get("error", "API call failed"))

            description = response.get("content", "").strip()
            logger.info(f"생성된 설명: '{description}' (길이: {len(description)})")
            
            # 응답 검증
            if not description:
                raise ValueError("Empty artwork description generated")

            # 길이 검증 (단어 수 15-25개)
            word_count = len(description.split())
            if word_count > 30:  # 약간의 여유를 둠
                # 첫 번째 문장만 사용
                first_sentence = description.split('.')[0] + '.'
                description = first_sentence

            logger.info(f"작품 설명 생성 완료: {len(description)}자, {word_count}단어")

            return {
                "success": True,
                "description": description,
                "word_count": len(description.split()),
                "character_count": len(description),
                "metadata": {
                    "model": response.get("model", "unknown"),
                    "token_usage": response.get("token_usage", {}),
                    "user_id": user_id,
                    "generation_method": "gpt_artwork_description"
                }
            }

        except Exception as e:
            logger.error(f"작품 설명 생성 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "description": "",
                "metadata": {}
            }
