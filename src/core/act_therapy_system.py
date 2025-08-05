# src/core/act_therapy_system.py

# ==============================================================================
# 이 파일은 Emoseum 시스템의 모든 구성 요소를 통합하고 관리하는 중앙 컨트롤 타워이다.
# 사용자 관리, 이미지 생성, 갤러리 관리, GPT 서비스 등 모든 하위 모듈들을 초기화하고,
# 사용자의 요청에 따라 각 모듈을 조율하여 ACT 치료 프로세스의 전체 흐름을 실행한다.
# main.py의 CLI는 이 클래스의 메서드를 호출하여 시스템과 상호작용한다.
# ==============================================================================

##
from PIL import Image
##
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from ..managers.user_manager import UserManager, PsychometricResult
from ..therapy.prompt_architect import PromptArchitect
from ..managers.personalization_manager import PersonalizationManager
from ..services.image_generator import ImageGenerator
from ..services.image_service_wrapper import ColabImageGenerator, ExternalImageGenerator
from ..services.emotion_analyzer import get_emotion_analyzer
from ..services.diary_exploration_service import get_diary_exploration_service
from ..managers.gallery_manager import GalleryManager, GalleryItem
from ..therapy.docent_message import DocentMessageSystem

import os
import requests
from io import BytesIO
import uuid

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
BUCKET = "emoseum-images"

logger = logging.getLogger(__name__)


class ACTTherapySystem:
    """ACT 기반 디지털 치료 시스템 통합 클래스"""

    def __init__(
        self, mongodb_client=None, model_path: str = "runwayml/stable-diffusion-v1-5"
    ):
        logger.info("ACT 치료 시스템 초기화 시작...")

        # MongoDB 클라이언트 초기화
        if mongodb_client is None:
            from ..dependencies import get_mongodb_client
            mongodb_client = get_mongodb_client()
        
        self.mongodb_client = mongodb_client

        # MongoDB 기반 컴포넌트 초기화
        self.user_manager = UserManager(mongodb_client)
        self.personalization_manager = PersonalizationManager(self.user_manager)
        
        # 이미지 생성 서비스 초기화 (환경변수에 따라 선택)
        self._initialize_image_service(model_path)
        
        self.gallery_manager = GalleryManager(mongodb_client)

        # GPT 서비스들 초기화
        self._initialize_gpt_services()

        # 컴포넌트들 초기화
        self.prompt_architect = PromptArchitect()
        self.docent_message_system = DocentMessageSystem(self.user_manager)

        # GPT 서비스들 주입
        self._inject_gpt_services()
        
        # 일기 심화 탐색 서비스 초기화 (GPT 서비스 주입 후)
        self.diary_exploration_service = get_diary_exploration_service(gpt_service=self.gpt_service)

        logger.info("ACT 치료 시스템 초기화 완료")

    def _initialize_image_service(self, model_path: str):
        """이미지 생성 서비스 초기화 (환경변수에 따라 선택)"""
        
        # 환경변수에서 이미지 생성 서비스 타입 확인
        service_type = os.getenv("IMAGE_GENERATION_SERVICE", "local")
        
        try:
            if service_type == "colab":
                # Colab 서비스 사용
                logger.info("Colab 이미지 생성 서비스를 초기화합니다...")
                colab_notebook_url = os.getenv("COLAB_NOTEBOOK_URL")
                if not colab_notebook_url:
                    logger.warning("COLAB_NOTEBOOK_URL이 설정되지 않았습니다. local로 대체합니다.")
                    service_type = "local"
                else:
                    # Colab용 래퍼 클래스 사용
                    self.image_generator = ColabImageGenerator(colab_notebook_url)
                    logger.info("Colab 이미지 생성 서비스 초기화 완료")
                    return
            
            # local 또는 기본값
            logger.info("로컬 이미지 생성 서비스를 초기화합니다...")
            self.image_generator = ImageGenerator(model_path)
            logger.info("로컬 이미지 생성 서비스 초기화 완료")
                
        except Exception as e:
            logger.error(f"이미지 생성 서비스 초기화 실패: {e}")
            logger.info("기본 로컬 GPU 서비스로 대체합니다...")
            self.image_generator = ImageGenerator(model_path)

    def _initialize_gpt_services(self):
        """GPT 서비스들 초기화"""

        try:
            from ..services.gpt_service import GPTService
            from ..ai.prompt_engineer import PromptEngineer
            from ..ai.docent_gpt import DocentGPT
            from ..utils.safety_validator import SafetyValidator
            from ..utils.cost_tracker import CostTracker

            logger.info("GPT 서비스 컴포넌트들을 초기화합니다...")

            # YAML 파일 경로 설정
            config_dir = Path("config")
            safety_rules_path = str(config_dir / "safety_rules.yaml")
            gpt_prompts_path = str(config_dir / "gpt_prompts.yaml")

            # 컴포넌트들 초기화 (MongoDB 클라이언트 전달)
            self.cost_tracker = CostTracker(self.mongodb_client)
            self.gpt_service = GPTService(
                cost_tracker=self.cost_tracker, gpt_prompts_path=gpt_prompts_path
            )
            self.safety_validator = SafetyValidator(safety_rules_path=safety_rules_path)
            self.prompt_engineer = PromptEngineer(
                self.gpt_service, gpt_prompts_path=gpt_prompts_path
            )
            self.docent_gpt = DocentGPT(
                self.gpt_service,
                self.safety_validator,
                gpt_prompts_path=gpt_prompts_path,
            )

            logger.info("모든 GPT 서비스 컴포넌트가 성공적으로 초기화되었습니다.")

        except ImportError as e:
            logger.error(f"GPT 서비스 모듈 import 실패: {e}")
            raise RuntimeError(f"필수 GPT 서비스 모듈을 찾을 수 없습니다: {e}")
        except Exception as e:
            logger.error(f"GPT 서비스 초기화 실패: {e}")
            raise RuntimeError(f"GPT 서비스 초기화에 실패했습니다: {e}")

    def _inject_gpt_services(self):
        """기존 컴포넌트들에 GPT 서비스 주입"""

        try:
            self.prompt_architect.set_prompt_engineer(self.prompt_engineer)
            self.prompt_architect.set_safety_validator(self.safety_validator)
            self.docent_message_system.set_docent_gpt(self.docent_gpt)
            logger.info("모든 GPT 서비스 주입이 완료되었습니다.")

        except Exception as e:
            logger.error(f"GPT 서비스 주입 실패: {e}")
            raise RuntimeError(f"GPT 서비스 주입에 실패했습니다: {e}")

    def onboard_new_user(self, user_id: str) -> Dict[str, Any]:
        """신규 사용자 온보딩"""

        logger.info(f"신규 사용자 온보딩 시작: {user_id}")

        user = self.user_manager.create_user(user_id)

        onboarding_info = {
            "user_id": user_id,
            "created_date": user.created_date,
            "next_steps": [
                "1. 심리검사 실시 (PHQ-9, CES-D, MEAQ, CISS)",
                "2. 시각적 선호도 설정",
                "3. 첫 감정 일기 작성",
            ],
            "default_preferences": user.visual_preferences.__dict__,
            "status": "onboarding_complete",
            "gpt_system_ready": True,
        }

        logger.info(f"사용자 {user_id} 온보딩 완료")
        return onboarding_info

    def conduct_psychometric_assessment(
        self,
        user_id: str,
        phq9_score: int,
        cesd_score: int,
        meaq_score: int,
        ciss_score: int,
    ) -> Dict[str, Any]:
        """심리검사 실시 및 결과 분석"""

        logger.info(f"사용자 {user_id} 심리검사 실시")

        result = self.user_manager.conduct_psychometric_test(
            user_id, phq9_score, cesd_score, meaq_score, ciss_score
        )

        assessment_result = {
            "user_id": user_id,
            "test_date": result.test_date,
            "scores": {
                "PHQ-9": phq9_score,
                "CES-D": cesd_score,
                "MEAQ": meaq_score,
                "CISS": ciss_score,
            },
            "coping_style": result.coping_style,
            "severity_level": result.severity_level,
            "interpretation": self._interpret_assessment_results(result),
            "recommendations": self._generate_assessment_recommendations(result),
            "gpt_personalization_ready": True,
        }

        logger.info(f"심리검사 완료: {result.coping_style}, {result.severity_level}")
        return assessment_result

    def set_visual_preferences(
        self,
        user_id: str,
        art_style: str,
        color_tone: str,
        complexity: str,
        brightness: float = 0.5,
        saturation: float = 0.5,
    ) -> Dict[str, Any]:
        """시각적 선호도 설정"""

        preferences = {
            "art_style": art_style,
            "color_tone": color_tone,
            "complexity": complexity,
            "brightness": brightness,
            "saturation": saturation,
        }

        self.user_manager.set_visual_preferences(user_id, preferences)

        logger.info(f"사용자 {user_id} 시각적 선호도 설정 완료")
        return {
            "user_id": user_id,
            "preferences_set": preferences,
            "status": "preferences_saved",
            "gpt_prompt_generation_ready": True,
        }

    def process_emotion_journey(self, user_id: str, diary_text: str) -> Dict[str, Any]:
        """ACT 4단계 감정 여정 처리 (Step 1-2: The Moment → Reflection)"""

        logger.info(f"사용자 {user_id} 감정 여정 시작")

        try:
            # GPT를 통한 감정 분석
            emotion_analysis = self._analyze_emotion_with_gpt(diary_text, user_id)

            user = self.user_manager.get_user(user_id)
            if not user:
                raise ValueError(f"사용자를 찾을 수 없습니다: {user_id}")

            # GPT를 통한 이미지 생성
            reflection_result = self._create_gpt_reflection_image(
                user, emotion_analysis, diary_text
            )

            gallery_item_id = self.gallery_manager.create_gallery_item(
                user_id=user_id,
                diary_text=diary_text,
                emotion_keywords=emotion_analysis["keywords"],
                vad_scores=emotion_analysis["vad_scores"],
                reflection_prompt=reflection_result["prompt"],
                reflection_image=reflection_result["image"],
                coping_style=(
                    user.psychometric_results[0].coping_style
                    if user.psychometric_results
                    else "balanced"
                ),
                gpt_prompt_tokens=reflection_result.get("prompt_tokens", 0),
                prompt_generation_time=reflection_result.get("generation_time", 0.0),
            )

            journey_result = {
                "user_id": user_id,
                "gallery_item_id": gallery_item_id,
                "step": "reflection_complete",
                "emotion_analysis": emotion_analysis,
                "reflection_image": {
                    "prompt": reflection_result["prompt"],
                    "image_path": str(reflection_result["image_path"]),
                    "generation_time": reflection_result["generation_time"],
                    "generation_method": "gpt",
                },
                "next_step": "artwork_title",
                "guided_message": "생성된 이미지를 보며 떠오르는 감정이나 생각을 자유롭게 표현해보세요.",
                "gpt_metadata": {
                    "prompt_tokens": reflection_result.get("prompt_tokens", 0),
                    "generation_successful": True,
                    "safety_validated": reflection_result.get("safety_validated", True),
                },
            }

            logger.info(f"Reflection 단계 완료: 아이템 {gallery_item_id}")
            return journey_result

        except Exception as e:
            logger.error(f"감정 여정 처리 실패: {e}")
            raise RuntimeError(f"감정 여정 처리에 실패했습니다: {e}")

    def _analyze_emotion_with_gpt(
        self, diary_text: str, user_id: str = "anonymous"
    ) -> Dict[str, Any]:
        """감정 분석 (환경변수에 따라 GoEmotions 또는 GPT 사용)"""
        
        # 환경변수에서 감정 분석 서비스 확인
        analysis_service = os.getenv("EMOTION_ANALYSIS_SERVICE", "local")
        
        try:
            if analysis_service in ["local", "colab"]:
                # GoEmotions 분석기 사용 (로컬 또는 Colab)
                service_type = "local_goEmotions" if analysis_service == "local" else "colab_goEmotions"
                emotion_analyzer = get_emotion_analyzer(service_type)
                analysis_result = emotion_analyzer.analyze_emotions(diary_text)
                
                logger.info(f"{analysis_service} GoEmotions 감정 분석 결과: {analysis_result}")
                
                return {
                    "diary_text": diary_text,
                    "keywords": analysis_result["keywords"],
                    "vad_scores": analysis_result["vad_scores"],
                    "analysis_confidence": analysis_result["confidence"],
                    "generation_method": f"{analysis_service}_goEmotions",
                    "primary_emotion": analysis_result["primary_emotion"],
                    "emotional_intensity": analysis_result["emotional_intensity"],
                    "top_emotions": analysis_result.get("top_emotions", {})
                }
            
            else:
                # GPT 분석기 사용 (기존 방식)
                analysis_result = self.gpt_service.analyze_emotion(
                    diary_text, user_id=user_id
                )

                if analysis_result["success"]:
                    return {
                        "diary_text": diary_text,
                        "keywords": analysis_result.get("keywords", ["neutral"]),
                        "vad_scores": analysis_result.get("vad_scores", (0.0, 0.0, 0.0)),
                        "analysis_confidence": analysis_result.get("confidence", 0.8),
                        "generation_method": "gpt",
                    }
                else:
                    raise RuntimeError(
                        f"GPT 감정 분석 실패: {analysis_result.get('error', 'Unknown error')}"
                    )

        except Exception as e:
            logger.error(f"감정 분석 실패 (서비스: {analysis_service}): {e}")
            raise

    def upload_image_to_supabase(self, image, user_id, prefix="generated/"):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        filename = f"{prefix}diary_{user_id}_{uuid.uuid4().hex}.png"

        response = requests.post(
            f"{SUPABASE_URL}/storage/v1/object/emoseum-images/{filename}",
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": "image/png"
            },
            data=buffer
        )

        if response.status_code >= 400:
            logger.warning(f"Supabase 업로드 실패: {response.text}")
            return None

        public_url = f"{SUPABASE_URL}/storage/v1/object/public/emoseum-images/{filename}"
        return public_url
    
    def _create_gpt_reflection_image(
        self, user, emotion_analysis: Dict[str, Any], diary_text: str
    ) -> Dict[str, Any]:
        """GPT를 통한 Reflection 이미지 생성 (이미지 생성 생략하고 fallback으로만 처리)"""

        coping_style = "balanced"
        if user.psychometric_results:
            coping_style = user.psychometric_results[0].coping_style

        self.prompt_architect.set_diary_context(diary_text)

        try:
            reflection_prompt = self.prompt_architect.create_reflection_prompt(
                emotion_keywords=emotion_analysis["keywords"],
                vad_scores=emotion_analysis["vad_scores"],
                coping_style=coping_style,
                visual_preferences=user.visual_preferences.__dict__,
                user_id=user.user_id,
            )


            # 환경변수에 따라 설정된 이미지 생성 서비스 사용
            service_type = os.getenv("IMAGE_GENERATION_SERVICE", "local")
            
            if service_type == "local":
                # 로컬 서비스 (ImageGenerator)
                generation_result = self.image_generator.generate_image(
                    prompt=reflection_prompt,
                    width=512,
                    height=512,
                    num_inference_steps=20,
                    guidance_scale=7.5
                )
                
                if generation_result.get("success"):
                    final_image = generation_result["image"]
                    image_url = self.upload_image_to_supabase(final_image, user.user_id, prefix="generated/")
                    print(f"이미지 생성 성공 (서비스: local)")
                else:
                    raise Exception(generation_result.get('error', 'Unknown error'))
                    
            else:
                # Colab 서비스 (래퍼 클래스)
                generation_result = self.image_generator.generate_image(
                    prompt=reflection_prompt,
                    output_dir="data/gallery_images/reflection",
                    filename=f"reflection_{user.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                
                if generation_result.get("success"):
                    # 생성된 이미지를 Supabase에 업로드
                    image_path = generation_result["image_path"]
                    with Image.open(image_path) as final_image:
                        image_url = self.upload_image_to_supabase(final_image, user.user_id, prefix="generated/")
                    
                    print(f"이미지 생성 성공 (서비스: {generation_result.get('service', service_type)})")
                else:
                    raise Exception(generation_result.get('error', 'Unknown error'))

            return {
                "prompt": reflection_prompt,
                "image": final_image if 'final_image' in locals() else None,
                "image_path": image_url,
                "generation_time": generation_result.get("generation_time", 30.0) if 'generation_result' in locals() else 30.0,
                "prompt_tokens": 0,
                "safety_validated": True,
            }

        except Exception as e:
            logger.error(f"이미지 생성 중 예외 발생: {e}")
            raise



    def complete_artwork_title(
        self,
        user_id: str,
        gallery_item_id: str,
        artwork_title: str,
    ) -> Dict[str, Any]:
        """ACT 3단계: Defusion (작품 제목 작성)"""

        logger.info(f"사용자 {user_id} 작품 제목 작성: 아이템 {gallery_item_id}")

        gallery_item = self.gallery_manager.get_gallery_item(gallery_item_id)
        if not gallery_item or gallery_item.user_id != user_id:
            raise ValueError("갤러리 아이템을 찾을 수 없습니다.")

        success = self.gallery_manager.complete_artwork_title(
            gallery_item_id, artwork_title, ""
        )

        if not success:
            raise RuntimeError("작품 제목 저장에 실패했습니다.")

        personalization_updates = (
            self.personalization_manager.update_preferences_from_artwork_title(
                user_id=user_id,
                artwork_title=artwork_title,
                image_prompt=gallery_item.reflection_prompt,
            )
        )

        artwork_title_result = {
            "user_id": user_id,
            "gallery_item_id": gallery_item_id,
            "step": "defusion_complete",
            "artwork_title": {
                "title": artwork_title,
            },
            "personalization_updates": personalization_updates,
            "next_step": "docent_message",
            "gpt_docent_ready": True,
        }

        logger.info(f"작품 제목 작성 완료: {artwork_title}")
        return artwork_title_result

    def create_docent_message(
        self, user_id: str, gallery_item_id: str
    ) -> Dict[str, Any]:
        """ACT 4단계: Closure (도슨트 메시지 생성)"""

        logger.info(f"사용자 {user_id} 도슨트 메시지 생성: 아이템 {gallery_item_id}")

        try:
            gallery_item = self.gallery_manager.get_gallery_item(gallery_item_id)
            if not gallery_item or gallery_item.user_id != user_id:
                raise ValueError("갤러리 아이템을 찾을 수 없습니다.")

            user = self.user_manager.get_user(user_id)

            docent_message = self.docent_message_system.create_personalized_message(
                user=user,
                gallery_item=gallery_item,
            )

            success = self.gallery_manager.add_docent_message(
                gallery_item_id, docent_message
            )

            if not success:
                raise RuntimeError("도슨트 메시지 저장에 실패했습니다.")

            docent_result = {
                "user_id": user_id,
                "gallery_item_id": gallery_item_id,
                "step": "closure_complete",
                "docent_message": docent_message,
                "journey_complete": True,
                "completion_message": "감정의 여정이 완성되었습니다.",
                "next_recommendations": [
                    "미술관에서 이전 작품들을 돌아보기",
                    "새로운 감정 일기 작성하기",
                    "치료 진행도 확인하기",
                ],
                "gpt_metadata": {
                    "generation_method": "gpt",
                    "personalization_level": docent_message.get("metadata", {}).get(
                        "personalization_level", 0
                    ),
                    "safety_validated": True,
                },
            }

            logger.info(f"도슨트 메시지 생성 완료: 아이템 {gallery_item_id}")
            return docent_result

        except Exception as e:
            logger.error(f"도슨트 메시지 생성 실패: {e}")
            raise

    def generate_diary_exploration_questions(
        self, user_id: str, diary_text: str
    ) -> Dict[str, Any]:
        """일기 심화 탐색 질문 생성"""
        
        logger.info(f"사용자 {user_id} 일기 심화 탐색 질문 생성")
        
        try:
            # 먼저 감정 분석 수행
            emotion_analysis = self._analyze_emotion_with_gpt(diary_text, user_id)
            emotion_keywords = emotion_analysis.get("keywords", [])
            
            # 일기 심화 탐색 서비스를 통해 질문 생성
            exploration_result = self.diary_exploration_service.generate_exploration_questions(
                diary_text=diary_text,
                emotion_keywords=emotion_keywords
            )
            
            # 결과에 사용자 정보와 감정 분석 결과 추가
            exploration_result.update({
                "user_id": user_id,
                "emotion_analysis": emotion_analysis,
                "diary_text_length": len(diary_text),
                "generation_timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"일기 심화 탐색 질문 생성 완료: {len(exploration_result.get('questions', []))}개 질문")
            return exploration_result
            
        except Exception as e:
            logger.error(f"일기 심화 탐색 질문 생성 실패: {e}")
            # 실패 시 기본 응답 반환
            return {
                "success": False,
                "user_id": user_id,
                "questions": [],
                "exploration_theme": "Exploration Error",
                "encouragement": "Please try again in a moment.",
                "error": str(e)
            }

    def generate_follow_up_exploration_question(
        self, 
        user_id: str, 
        diary_text: str, 
        previous_question: str,
        user_response: str
    ) -> Dict[str, Any]:
        """이전 답변을 바탕으로 후속 일기 심화 탐색 질문 생성"""
        
        logger.info(f"사용자 {user_id} 후속 일기 심화 탐색 질문 생성")
        
        try:
            # 먼저 감정 분석 수행 (기존 데이터 활용 또는 새로 분석)
            emotion_analysis = self._analyze_emotion_with_gpt(diary_text, user_id)
            emotion_keywords = emotion_analysis.get("keywords", [])
            
            # 후속 질문 생성
            exploration_result = self.diary_exploration_service.generate_follow_up_question(
                diary_text=diary_text,
                previous_question=previous_question,
                user_response=user_response,
                emotion_keywords=emotion_keywords
            )
            
            # 결과에 사용자 정보와 메타데이터 추가
            exploration_result.update({
                "user_id": user_id,
                "emotion_analysis": emotion_analysis,
                "previous_question": previous_question,
                "user_response_length": len(user_response),
                "generation_timestamp": datetime.now().isoformat(),
                "is_follow_up": True
            })
            
            logger.info(f"후속 일기 심화 탐색 질문 생성 완료: {len(exploration_result.get('questions', []))}개 질문")
            return exploration_result
            
        except Exception as e:
            logger.error(f"후속 일기 심화 탐색 질문 생성 실패: {e}")
            # 실패 시 기본 응답 반환
            return {
                "success": False,
                "user_id": user_id,
                "questions": [],
                "exploration_theme": "Follow-up Error",
                "encouragement": "Thank you for sharing. Please try again in a moment.",
                "error": str(e),
                "is_follow_up": True
            }

    def record_message_reaction(
        self,
        user_id: str,
        gallery_item_id: str,
        reaction_type: str,
        reaction_data: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """도슨트 메시지에 대한 사용자 반응 기록"""

        logger.info(f"사용자 {user_id} 메시지 반응 기록: {reaction_type}")

        gallery_item = self.gallery_manager.get_gallery_item(gallery_item_id)
        if not gallery_item or gallery_item.user_id != user_id:
            raise ValueError("갤러리 아이템을 찾을 수 없습니다.")

        success = self.gallery_manager.record_message_reaction(
            gallery_item_id, reaction_type, reaction_data or {}
        )

        if not success:
            raise RuntimeError("메시지 반응 저장에 실패했습니다.")

        personalization_updates = {}
        if reaction_type in ["like", "save", "share"]:
            personalization_updates = (
                self.personalization_manager.update_preferences_from_message_reaction(
                    user_id=user_id,
                    reaction_type=reaction_type,
                    docent_message=gallery_item.docent_message,
                    artwork_title_data={
                        "title": gallery_item.artwork_title,
                    },
                )
            )

        reaction_result = {
            "user_id": user_id,
            "gallery_item_id": gallery_item_id,
            "reaction_type": reaction_type,
            "reaction_recorded": True,
            "personalization_updates": personalization_updates,
            "gpt_learning_enabled": True,
        }

        logger.info(f"메시지 반응 기록 완료: {reaction_type}")
        return reaction_result

    def get_user_gallery(
        self, user_id: str, limit: int = 20, date_from: Optional[str] = None
    ) -> Dict[str, Any]:
        """사용자 미술관 조회"""

        gallery_items = self.gallery_manager.get_user_gallery(
            user_id, limit=limit, date_from=date_from
        )

        gallery_data = {
            "user_id": user_id,
            "total_items": len(gallery_items),
            "items": [item.to_dict() for item in gallery_items],
            "analytics": self.gallery_manager.get_gallery_analytics(user_id),
            "gpt_usage_stats": self._get_gpt_usage_stats(user_id),
        }

        return gallery_data

    def _get_gpt_usage_stats(self, user_id: str) -> Dict[str, Any]:
        """사용자별 GPT 사용 통계"""
        try:
            cost_stats = self.cost_tracker.get_user_usage_summary(user_id)
            return {
                "total_tokens": cost_stats.get("total_tokens", 0),
                "total_cost": cost_stats.get("total_cost", 0.0),
                "prompt_generations": cost_stats.get("prompt_calls", 0),
                "docent_generations": cost_stats.get("docent_calls", 0),
                "avg_generation_time": cost_stats.get("avg_generation_time", 0.0),
            }
        except Exception as e:
            logger.warning(f"GPT 사용 통계 조회 실패: {e}")
            return {}

    def get_therapeutic_insights(self, user_id: str) -> Dict[str, Any]:
        """치료적 인사이트 제공"""

        user_stats = self.user_manager.get_user_stats(user_id)
        gallery_analytics = self.gallery_manager.get_gallery_analytics(user_id)
        personalization_insights = (
            self.personalization_manager.get_personalization_insights(user_id)
        )
        content_recommendations = (
            self.personalization_manager.recommend_content_adjustments(user_id)
        )
        message_analytics = self.gallery_manager.get_message_reaction_analytics(user_id)
        gpt_performance = self._analyze_gpt_performance(user_id)

        insights = {
            "user_id": user_id,
            "assessment_date": datetime.now().isoformat(),
            "user_profile": user_stats,
            "emotional_journey": gallery_analytics.get("emotion_trends", {}),
            "growth_insights": gallery_analytics.get("growth_insights", []),
            "personalization_status": personalization_insights,
            "message_engagement": message_analytics,
            "gpt_performance": gpt_performance,
            "recommendations": {
                "content_adjustments": content_recommendations,
                "next_actions": self._generate_next_action_recommendations(
                    user_stats, gallery_analytics
                ),
                "gpt_optimization": gpt_performance.get("optimization_suggestions", []),
            },
            "summary": self._generate_therapeutic_summary(
                user_stats, gallery_analytics
            ),
        }

        return insights

    def _analyze_gpt_performance(self, user_id: str) -> Dict[str, Any]:
        """GPT 성능 분석"""
        try:
            docent_performance = (
                self.docent_message_system.get_gpt_performance_metrics(user_id)
            )
            cost_data = self.cost_tracker.get_user_usage_summary(user_id)

            return {
                "docent_effectiveness": docent_performance.get("quality_score", 0),
                "personalization_level": docent_performance.get(
                    "personalization_score", 0
                ),
                "cost_efficiency": cost_data.get("cost_per_generation", 0),
                "generation_success_rate": docent_performance.get("success_rate", 1.0),
                "optimization_suggestions": self._generate_gpt_optimization_suggestions(
                    docent_performance
                ),
            }
        except Exception as e:
            logger.warning(f"GPT 성능 분석 실패: {e}")
            return {"error": str(e)}

    def _generate_gpt_optimization_suggestions(
        self, performance: Dict[str, Any]
    ) -> List[str]:
        """GPT 개선 제안"""
        suggestions = []

        quality_score = performance.get("quality_score", 0)
        if quality_score < 0.7:
            suggestions.append(
                "도슨트 메시지 품질을 위한 프롬프트 엔지니어링 필요"
            )

        personalization_score = performance.get("personalization_score", 0)
        if personalization_score < 0.6:
            suggestions.append("개인화 수준을 위한 사용자 데이터 활용 필요")

        fallback_usage = performance.get("fallback_usage", 0)
        if fallback_usage > 0.1:
            suggestions.append("GPT 실패율 감소를 위한 시스템 안정성 필요")

        return suggestions

    def trigger_advanced_training(
        self, user_id: str, training_type: str = "both"
    ) -> Dict[str, Any]:
        """Level 3 모델 훈련 실행"""

        logger.info(f"사용자 {user_id}의 Level 3 훈련 시작: {training_type}")

        try:
            gallery_items = self.gallery_manager.get_user_gallery(user_id, limit=1000)

            complete_journeys = [
                item.to_dict()
                for item in gallery_items
                if item.artwork_title and item.docent_message
            ]

            results = {}

            if training_type in ["lora", "both"]:
                from ..training.lora_trainer import PersonalizedLoRATrainer

                lora_trainer = PersonalizedLoRATrainer()

                lora_data = lora_trainer.prepare_training_data(complete_journeys)
                lora_result = lora_trainer.train_user_lora(user_id, lora_data)
                results["lora"] = lora_result

            if training_type in ["draft", "both"]:
                from ..training.draft_trainer import DRaFTPlusTrainer

                draft_trainer = DRaFTPlusTrainer()

                draft_data = draft_trainer.prepare_training_data(complete_journeys)
                draft_result = draft_trainer.train_user_draft(user_id, draft_data)
                results["draft"] = draft_result

            overall_success = all(
                result.get("success", False) for result in results.values()
            )

            return {
                "success": overall_success,
                "user_id": user_id,
                "training_type": training_type,
                "results": results,
                "data_used": len(complete_journeys),
                "gpt_integration": "GPT 데이터 활용",
            }

        except Exception as e:
            logger.error(f"Level 3 훈련 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "user_id": user_id,
                "training_type": training_type,
            }

    def check_advanced_training_readiness(self, user_id: str) -> Dict[str, Any]:
        """Level 3 모델 훈련 준비 상태 확인"""

        gallery_items = self.gallery_manager.get_user_gallery(user_id, limit=1000)

        complete_journeys = [
            item
            for item in gallery_items
            if item.artwork_title and item.docent_message
        ]

        from ..training.lora_trainer import PersonalizedLoRATrainer
        from ..training.draft_trainer import DRaFTPlusTrainer

        lora_trainer = PersonalizedLoRATrainer()
        lora_data = lora_trainer.prepare_training_data(
            [item.to_dict() for item in complete_journeys]
        )
        lora_requirements = lora_trainer.get_training_requirements(len(lora_data))

        draft_trainer = DRaFTPlusTrainer()
        draft_data = draft_trainer.prepare_training_data(
            [item.to_dict() for item in complete_journeys]
        )
        draft_requirements = draft_trainer.get_training_requirements(len(draft_data))

        readiness = {
            "user_id": user_id,
            "total_gallery_items": len(gallery_items),
            "complete_journeys": len(complete_journeys),
            "lora_training": {
                "ready": lora_requirements["can_train"],
                "data_size": len(lora_data),
                "requirements": lora_requirements,
                "estimated_performance": (
                    "high"
                    if len(lora_data) >= 100
                    else "medium" if len(lora_data) >= 50 else "low"
                ),
            },
            "draft_training": {
                "ready": draft_requirements["can_train"],
                "data_size": len(draft_data),
                "requirements": draft_requirements,
                "estimated_performance": (
                    "high"
                    if len(draft_data) >= 50
                    else "medium" if len(draft_data) >= 30 else "low"
                ),
            },
            "overall_readiness": (
                "ready"
                if lora_requirements["can_train"] and draft_requirements["can_train"]
                else "not_ready"
            ),
            "recommendations": self._get_advanced_training_recommendations(
                lora_requirements, draft_requirements
            ),
            "gpt_data_quality": "high",
        }

        return readiness

    def _interpret_assessment_results(
        self, result: PsychometricResult
    ) -> Dict[str, str]:
        """심리검사 결과 해석"""
        interpretations = {
            "coping_style_description": {
                "avoidance_oriented": "감정적 상황을 회피하거나 우회하는 경향이 있습니다.",
                "task_oriented": "감정적 상황에 직면하고 적극적으로 대처하는 경향이 있습니다.",
                "balanced": "상황에 따라 유연하게 대처하는 균형잡힌 스타일을 보입니다.",
            }[result.coping_style],
            "severity_description": {
                "mild": "가벼운 수준의 우울 증상이 관찰됩니다.",
                "moderate": "중등도의 우울 증상이 있어 주의가 필요합니다.",
                "severe": "심한 우울 증상이 있어 상담이 권장됩니다.",
            }[result.severity_level],
        }

        return interpretations

    def _generate_assessment_recommendations(
        self, result: PsychometricResult
    ) -> List[str]:
        """심리검사 기반 권장사항"""
        recommendations = []

        if result.coping_style == "avoidance_oriented":
            recommendations.extend(
                [
                    "부드럽고 은유적인 감정 표현을 통해 점진적으로 감정에 다가가보세요.",
                    "GPT가 생성하는 이미지를 통해 안전한 거리에서 감정을 관찰해보세요.",
                    "감정을 안전한 거리에서 관찰하고 수용하는 연습을 해보세요.",
                ]
            )
        elif result.coping_style == "task_oriented":
            recommendations.extend(
                [
                    "감정을 직접적이고 명확하게 표현하는 것이 도움이 될 것 같습니다.",
                    "GPT 생성 이미지를 통해 감정의 본질을 깊이 탐구해보세요.",
                    "감정의 강도와 복잡성을 있는 그대로 받아들여보세요.",
                ]
            )
        else:
            recommendations.extend(
                [
                    "상황에 맞는 유연한 감정 표현을 시도해보세요.",
                    "GPT가 생성하는 다양한 스타일의 이미지를 통해 감정의 여러 면을 탐색해보세요.",
                ]
            )

        if result.severity_level == "severe":
            recommendations.append("상담사와의 상담을 병행하시기를 권장합니다.")

        return recommendations

    def _generate_next_action_recommendations(
        self, user_stats: Dict[str, Any], gallery_analytics: Dict[str, Any]
    ) -> List[str]:
        """다음 행동 권장사항"""
        recommendations = []

        total_items = gallery_analytics.get("total_items", 0)

        if total_items == 0:
            recommendations.append("첫 번째 감정 일기를 작성해보세요.")
        elif total_items < 5:
            recommendations.append("더 많은 감정 경험을 기록해보세요.")
        elif total_items >= 10:
            recommendations.extend(
                [
                    "이전 작품들을 돌아보며 감정의 변화를 관찰해보세요.",
                    "특별히 의미 있었던 작품에 대해 다시 생각해보세요.",
                ]
            )

        needs_test = user_stats.get("needs_periodic_test", False)
        if needs_test:
            recommendations.append("2주가 지났습니다. 주기적 심리검사를 받아보세요.")

        return recommendations

    def _generate_therapeutic_summary(
        self, user_stats: Dict[str, Any], gallery_analytics: Dict[str, Any]
    ) -> str:
        """치료적 요약"""
        total_items = gallery_analytics.get("total_items", 0)
        member_days = gallery_analytics.get("date_range", {}).get("span_days", 0)

        if total_items == 0:
            return "감정 여정을 시작할 준비가 되어있습니다."

        summary = f"{member_days}일 동안 {total_items}개의 감정 여정을 완성했습니다. "

        emotion_trends = gallery_analytics.get("emotion_trends", {})
        if emotion_trends:
            valence_trend = emotion_trends.get("valence", {}).get("trend", "stable")
            if valence_trend == "improving":
                summary += "감정 상태가 긍정적으로 변화하고 있습니다."
            else:
                summary += "꾸준한 감정 탐색을 통해 자기 이해가 깊어지고 있습니다."

        return summary

    def _get_advanced_training_recommendations(
        self, lora_req: Dict[str, Any], draft_req: Dict[str, Any]
    ) -> List[str]:
        """훈련 권장사항"""
        recommendations = []

        if not lora_req["can_train"]:
            recommendations.append(
                f"LoRA 개인화를 위해 {lora_req['data_shortage']}개의 긍정적 반응이 더 필요합니다."
            )

        if not draft_req["can_train"]:
            recommendations.append(
                f"DRaFT+ 학습을 위해 {draft_req['data_shortage']}개의 완성된 여정이 더 필요합니다."
            )

        if lora_req["can_train"] and draft_req["can_train"]:
            recommendations.append("개인화 모델을 훈련할 준비가 되었습니다!")

        return recommendations

    def cleanup(self):
        """시스템 리소스 정리"""
        if hasattr(self.image_generator, "cleanup"):
            self.image_generator.cleanup()

        # MongoDB 연결은 자동으로 관리됨
        logger.info("ACT 치료 시스템 리소스 정리 완료")

    def get_system_status(self) -> Dict[str, Any]:
        """전체 시스템 상태 확인"""
        return {
            "system_version": "mongodb_integrated",
            "database": {
                "type": "MongoDB",
                "connection_ready": self.mongodb_client.test_connection(),
                "migration_complete": True
            },
            "components": {
                "user_manager": self.user_manager is not None,
                "gallery_manager": self.gallery_manager is not None,
                "cost_tracker": hasattr(self, "cost_tracker") and self.cost_tracker is not None,
                "prompt_architect": self.prompt_architect.get_system_status(),
                "docent_message": self.docent_message_system.get_system_status(),
                "gpt_service": hasattr(self, "gpt_service") and self.gpt_service is not None,
                "prompt_engineer": hasattr(self, "prompt_engineer") and self.prompt_engineer is not None,
                "docent_gpt": hasattr(self, "docent_gpt") and self.docent_gpt is not None,
                "safety_validator": hasattr(self, "safety_validator") and self.safety_validator is not None,
            },
            "sqlite_files": 0,
            "mongodb_collections": ["users", "gallery_items", "cost_tracking"],
            "fallback_systems": False,
            "hardcoded_templates": False,
            "gpt_integration_complete": True,
            "mongodb_migration_complete": True,
        }
    

