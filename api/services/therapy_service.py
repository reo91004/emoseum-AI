# api/services/therapy_service.py

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
import tempfile
from pathlib import Path

from api.services.database import db
from api.services.image_service import ImageService
from api.config import settings

logger = logging.getLogger(__name__)


class TherapyService:
    """기존 ACTTherapySystem을 API용으로 래핑하는 서비스"""

    def __init__(self):
        self.image_service = ImageService()
        self.therapy_system = None
        self._initialize_therapy_system()

        logger.info("🏥 치료 서비스 초기화 완료")

    def _initialize_therapy_system(self):
        """기존 ACTTherapySystem 초기화"""
        try:
            # 임시 데이터 디렉토리 생성 (실제로는 Supabase 사용)
            temp_dir = Path(tempfile.gettempdir()) / "emoseum_api"
            temp_dir.mkdir(exist_ok=True)

            from src.core.act_therapy_system import ACTTherapySystem

            # 이미지 생성은 별도 서비스 사용하므로 모델 경로는 None
            self.therapy_system = ACTTherapySystem(
                data_dir=str(temp_dir), model_path="dummy"  # 실제로는 사용하지 않음
            )

            logger.info("✅ ACTTherapySystem 초기화 완료")

        except Exception as e:
            logger.error(f"❌ ACTTherapySystem 초기화 실패: {e}")
            raise

    # === 사용자 관리 ===
    async def create_user(self, user_id: str) -> Dict[str, Any]:
        """신규 사용자 생성"""
        try:
            # 기존 사용자 확인
            existing_user = await db.get_user(user_id)
            if existing_user:
                return {
                    "success": False,
                    "error": "User already exists",
                    "user_id": user_id,
                }

            # Supabase에 사용자 생성
            user_data = {
                "user_id": user_id,
                "created_date": datetime.utcnow().isoformat(),
            }

            new_user = await db.create_user(user_data)

            # 기존 시스템의 온보딩 로직 활용
            onboarding_info = self.therapy_system.onboard_new_user(user_id)

            logger.info(f"✅ 신규 사용자 생성 완료: {user_id}")

            return {"success": True, "user": new_user, "onboarding": onboarding_info}

        except Exception as e:
            logger.error(f"❌ 사용자 생성 실패: {e}")
            raise

    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """사용자 프로필 조회"""
        try:
            user = await db.get_user(user_id)
            if not user:
                return None

            # 추가 정보 수집
            assessment = await db.get_assessment(user_id)
            visual_prefs = await db.get_visual_preferences(user_id)
            gallery_count = len(await db.get_gallery_items(user_id, limit=1000))

            profile = {
                "user_info": user,
                "assessment": assessment,
                "visual_preferences": visual_prefs,
                "statistics": {
                    "total_journeys": gallery_count,
                    "member_since": user.get("created_date"),
                },
            }

            logger.debug(f"✅ 사용자 프로필 조회 완료: {user_id}")
            return profile

        except Exception as e:
            logger.error(f"❌ 사용자 프로필 조회 실패: {e}")
            raise

    # === 심리검사 ===
    async def conduct_assessment(
        self, user_id: str, phq9: int, cesd: int, meaq: int, ciss: int
    ) -> Dict[str, Any]:
        """심리검사 실시"""
        try:
            # 점수 유효성 검증
            if not all(0 <= score <= 27 for score in [phq9, cesd, meaq, ciss]):
                return {
                    "success": False,
                    "error": "Invalid scores. All scores must be between 0 and 27",
                }

            # 기존 로직 활용
            result = self.therapy_system.conduct_psychometric_assessment(
                user_id, phq9, cesd, meaq, ciss
            )

            # Supabase에 저장
            await db.save_assessment(user_id, result)

            logger.info(f"✅ 심리검사 완료: {user_id} - {result['coping_style']}")

            return {"success": True, "assessment": result}

        except Exception as e:
            logger.error(f"❌ 심리검사 실시 실패: {e}")
            raise

    async def get_assessment_result(self, user_id: str) -> Optional[Dict[str, Any]]:
        """심리검사 결과 조회"""
        try:
            assessment = await db.get_assessment(user_id)

            if assessment:
                logger.debug(f"✅ 심리검사 결과 조회 완료: {user_id}")
            else:
                logger.debug(f"🔍 심리검사 결과 없음: {user_id}")

            return assessment

        except Exception as e:
            logger.error(f"❌ 심리검사 결과 조회 실패: {e}")
            raise

    # === 시각 선호도 ===
    async def set_visual_preferences(
        self, user_id: str, preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """시각 선호도 설정"""
        try:
            # 기존 로직으로 선호도 처리
            processed_prefs = self.therapy_system.set_visual_preferences(
                user_id,
                preferences.get("style_preferences", {}),
                preferences.get("color_preferences", {}),
                preferences.get("complexity_level", 5),
            )

            # Supabase에 저장
            saved_prefs = await db.save_visual_preferences(user_id, processed_prefs)

            logger.info(f"✅ 시각 선호도 설정 완료: {user_id}")

            return {"success": True, "preferences": saved_prefs}

        except Exception as e:
            logger.error(f"❌ 시각 선호도 설정 실패: {e}")
            raise

    # === 치료 여정 ===
    async def start_journey(self, user_id: str, diary_text: str) -> Dict[str, Any]:
        """감정 여정 시작 (The Moment 단계)"""
        try:
            # 일기 텍스트 유효성 검증
            if not diary_text or len(diary_text.strip()) < 10:
                return {
                    "success": False,
                    "error": "Diary text must be at least 10 characters long",
                }

            # 기존 로직으로 감정 분석
            journey_result = self.therapy_system.start_emotional_journey(
                user_id, diary_text
            )

            if not journey_result["success"]:
                return {
                    "success": False,
                    "error": journey_result.get("error", "Failed to start journey"),
                }

            # Supabase에 갤러리 아이템 생성
            gallery_data = {
                "user_id": user_id,
                "diary_text": diary_text,
                "emotion_keywords": journey_result["emotion_analysis"]["keywords"],
                "vad_scores": journey_result["emotion_analysis"]["vad_scores"],
                "completion_status": "moment_completed",
            }

            gallery_item = await db.create_gallery_item(gallery_data)

            result = {
                "success": True,
                "journey_id": gallery_item["id"],
                "emotion_analysis": journey_result["emotion_analysis"],
                "stage": "moment_completed",
                "next_stage": "reflection",
            }

            logger.info(f"✅ 감정 여정 시작 완료: {gallery_item['id']}")
            return result

        except Exception as e:
            logger.error(f"❌ 감정 여정 시작 실패: {e}")
            raise

    async def generate_reflection(
        self, user_id: str, journey_id: str
    ) -> Dict[str, Any]:
        """이미지 생성 (Reflection 단계)"""
        try:
            # 갤러리 아이템 조회
            item = await db.get_gallery_item(journey_id)
            if not item:
                return {"success": False, "error": "Journey not found"}

            # 권한 확인
            if item["user_id"] != user_id:
                return {"success": False, "error": "Access denied"}

            # 단계 확인
            if item["completion_status"] != "moment_completed":
                return {
                    "success": False,
                    "error": f"Invalid stage: {item['completion_status']}",
                }

            # 사용자 정보 조회
            assessment = await db.get_assessment(user_id)
            visual_prefs = await db.get_visual_preferences(user_id)

            # 기존 로직으로 프롬프트 생성
            self.therapy_system.prompt_architect.set_diary_context(item["diary_text"])

            prompt = self.therapy_system.prompt_architect.create_reflection_prompt(
                emotion_keywords=item["emotion_keywords"],
                vad_scores=item["vad_scores"],
                coping_style=assessment["coping_style"] if assessment else "balanced",
                visual_preferences=visual_prefs or {},
                user_id=user_id,
            )

            # 새로운 이미지 서비스로 이미지 생성
            image_result = await self.image_service.generate_image(
                prompt=prompt,
                width=512,
                height=512,
                num_inference_steps=20,
                guidance_scale=7.5,
            )

            # 결과 업데이트
            updates = {
                "image_prompt": prompt,
                "completion_status": "reflection_completed",
            }

            if image_result["success"]:
                # 이미지 URL 또는 base64 저장
                if image_result.get("image_url"):
                    updates["image_path"] = image_result["image_url"]
                elif image_result.get("image_b64"):
                    # base64를 데이터 URL로 변환
                    updates["image_path"] = (
                        f"data:image/png;base64,{image_result['image_b64']}"
                    )

            updated_item = await db.update_gallery_item(journey_id, updates)

            result = {
                "success": True,
                "journey_id": journey_id,
                "prompt": prompt,
                "image_result": image_result,
                "stage": "reflection_completed",
                "next_stage": "defusion",
            }

            logger.info(f"✅ 이미지 생성 완료: {journey_id}")
            return result

        except Exception as e:
            logger.error(f"❌ 이미지 생성 실패: {e}")
            raise

    async def create_defusion(
        self,
        user_id: str,
        journey_id: str,
        guestbook_title: str,
        guestbook_content: str,
        guestbook_tags: List[str],
    ) -> Dict[str, Any]:
        """방명록 작성 (Defusion 단계)"""
        try:
            # 갤러리 아이템 조회
            item = await db.get_gallery_item(journey_id)
            if not item:
                return {"success": False, "error": "Journey not found"}

            # 권한 확인
            if item["user_id"] != user_id:
                return {"success": False, "error": "Access denied"}

            # 단계 확인
            if item["completion_status"] != "reflection_completed":
                return {
                    "success": False,
                    "error": f"Invalid stage: {item['completion_status']}",
                }

            # 입력 유효성 검증
            if not guestbook_title or not guestbook_content:
                return {"success": False, "error": "Title and content are required"}

            # 기존 로직으로 방명록 처리 (개인화 학습)
            defusion_result = self.therapy_system.create_defusion_entry(
                user_id, journey_id, guestbook_title, guestbook_content, guestbook_tags
            )

            # 개인화 데이터 저장
            personalization_data = {
                "interaction_type": "defusion",
                "feedback_data": {
                    "title": guestbook_title,
                    "content": guestbook_content,
                    "tags": guestbook_tags,
                    "sentiment": defusion_result.get("sentiment_analysis", {}),
                },
                "learning_weights": defusion_result.get("learning_updates", {}),
            }
            await db.save_personalization_data(user_id, personalization_data)

            # 갤러리 아이템 업데이트
            updates = {
                "guestbook_title": guestbook_title,
                "guestbook_content": guestbook_content,
                "guestbook_tags": guestbook_tags,
                "completion_status": "defusion_completed",
            }

            updated_item = await db.update_gallery_item(journey_id, updates)

            result = {
                "success": True,
                "journey_id": journey_id,
                "guestbook": {
                    "title": guestbook_title,
                    "content": guestbook_content,
                    "tags": guestbook_tags,
                },
                "defusion_analysis": defusion_result,
                "stage": "defusion_completed",
                "next_stage": "closure",
            }

            logger.info(f"✅ 방명록 작성 완료: {journey_id}")
            return result

        except Exception as e:
            logger.error(f"❌ 방명록 작성 실패: {e}")
            raise

    async def generate_closure(self, user_id: str, journey_id: str) -> Dict[str, Any]:
        """큐레이터 메시지 생성 (Closure 단계)"""
        try:
            # 갤러리 아이템 조회
            item = await db.get_gallery_item(journey_id)
            if not item:
                return {"success": False, "error": "Journey not found"}

            # 권한 확인
            if item["user_id"] != user_id:
                return {"success": False, "error": "Access denied"}

            # 단계 확인
            if item["completion_status"] != "defusion_completed":
                return {
                    "success": False,
                    "error": f"Invalid stage: {item['completion_status']}",
                }

            # 사용자 프로필 수집
            user_profile = await self.get_user_profile(user_id)

            # 기존 로직으로 큐레이터 메시지 생성
            curator_result = self.therapy_system.generate_curator_message(
                user_id, journey_id, user_profile, item
            )

            if not curator_result["success"]:
                return {
                    "success": False,
                    "error": curator_result.get(
                        "error", "Failed to generate curator message"
                    ),
                }

            # 갤러리 아이템 업데이트
            updates = {
                "curator_message": curator_result["message"],
                "completion_status": "completed",
            }

            updated_item = await db.update_gallery_item(journey_id, updates)

            result = {
                "success": True,
                "journey_id": journey_id,
                "curator_message": curator_result["message"],
                "message_analysis": curator_result.get("analysis", {}),
                "stage": "completed",
                "journey_completed": True,
            }

            logger.info(f"✅ 큐레이터 메시지 생성 완료: {journey_id}")
            return result

        except Exception as e:
            logger.error(f"❌ 큐레이터 메시지 생성 실패: {e}")
            raise

    # === 갤러리 ===
    async def get_journey(
        self, user_id: str, journey_id: str
    ) -> Optional[Dict[str, Any]]:
        """특정 여정 조회"""
        try:
            item = await db.get_gallery_item(journey_id)

            if not item:
                return None

            # 권한 확인
            if item["user_id"] != user_id:
                logger.warning(
                    f"⚠️  권한 없는 갤러리 접근 시도: {user_id} -> {journey_id}"
                )
                return None

            logger.debug(f"✅ 여정 조회 완료: {journey_id}")
            return item

        except Exception as e:
            logger.error(f"❌ 여정 조회 실패: {e}")
            raise

    async def get_user_journeys(
        self, user_id: str, limit: int = 20, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """사용자의 모든 여정 목록"""
        try:
            items = await db.get_gallery_items(user_id, limit, offset)

            logger.debug(f"✅ 사용자 여정 목록 조회 완료: {user_id} ({len(items)}개)")
            return items

        except Exception as e:
            logger.error(f"❌ 사용자 여정 목록 조회 실패: {e}")
            raise

    async def delete_journey(self, user_id: str, journey_id: str) -> bool:
        """여정 삭제"""
        try:
            item = await db.get_gallery_item(journey_id)

            if not item:
                return False

            # 권한 확인
            if item["user_id"] != user_id:
                return False

            await db.delete_gallery_item(journey_id)

            logger.info(f"✅ 여정 삭제 완료: {journey_id}")
            return True

        except Exception as e:
            logger.error(f"❌ 여정 삭제 실패: {e}")
            raise

    # === 시스템 상태 ===
    def get_service_status(self) -> Dict[str, Any]:
        """서비스 상태 확인"""
        try:
            image_status = self.image_service.get_backend_status()
            therapy_status = self.therapy_system.get_system_status()

            return {
                "status": "operational",
                "timestamp": datetime.utcnow().isoformat(),
                "components": {
                    "therapy_system": therapy_status,
                    "image_service": image_status,
                    "database": "supabase",
                },
                "features": {
                    "act_therapy": True,
                    "image_generation": image_status.get("available", False),
                    "personalization": True,
                    "safety_validation": True,
                },
            }

        except Exception as e:
            logger.error(f"❌ 서비스 상태 확인 실패: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
