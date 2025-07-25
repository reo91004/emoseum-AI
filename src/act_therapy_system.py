# src/act_therapy_system.py

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from .user_manager import UserManager, PsychometricResult
from .prompt_architect import PromptArchitect
from .personalization_manager import PersonalizationManager
from .image_generator import ImageGenerator
from .gallery_manager import GalleryManager, GalleryItem
from .rule_manager import CopingStyleRules
from .curator_message import CuratorMessageSystem

logger = logging.getLogger(__name__)


class ACTTherapySystem:
    """ACT 기반 디지털 치료 시스템 통합 클래스"""

    def __init__(
        self, data_dir: str = "data", model_path: str = "runwayml/stable-diffusion-v1-5"
    ):

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 핵심 컴포넌트 초기화
        logger.info("ACT 치료 시스템 초기화 시작...")

        self.user_manager = UserManager(
            db_path=str(self.data_dir / "users.db"),
            preferences_dir=str(self.data_dir / "preferences"),
        )

        self.prompt_architect = PromptArchitect()
        self.personalization_manager = PersonalizationManager(self.user_manager)
        self.image_generator = ImageGenerator(model_path)
        self.gallery_manager = GalleryManager(
            db_path=str(self.data_dir / "gallery.db"),
            images_dir=str(self.data_dir / "gallery_images"),
        )
        self.rule_manager = CopingStyleRules()
        self.curator_message_system = CuratorMessageSystem(self.user_manager)

        logger.info("ACT 치료 시스템 초기화 완료")

    def onboard_new_user(self, user_id: str) -> Dict[str, Any]:
        """신규 사용자 온보딩 (Level 1: 초기 프로파일링)"""

        logger.info(f"신규 사용자 온보딩 시작: {user_id}")

        # 1. 기본 사용자 생성
        user = self.user_manager.create_user(user_id)

        # 2. 온보딩 완료 정보 반환
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

        # 심리검사 수행
        result = self.user_manager.conduct_psychometric_test(
            user_id, phq9_score, cesd_score, meaq_score, ciss_score
        )

        # 결과 해석
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
        }

    def process_emotion_journey(self, user_id: str, diary_text: str) -> Dict[str, Any]:
        """ACT 4단계 감정 여정 처리 (Step 1-2: The Moment → Reflection)"""

        logger.info(f"사용자 {user_id} 감정 여정 시작")

        # 1. The Moment: 감정 분석
        emotion_analysis = self._analyze_emotion_moment(diary_text)

        # 2. 사용자 프로필 조회
        user = self.user_manager.get_user(user_id)
        if not user:
            raise ValueError(f"사용자를 찾을 수 없습니다: {user_id}")

        # 3. Reflection: 감정 이미지 생성
        reflection_result = self._create_reflection_image(
            user, emotion_analysis, diary_text
        )

        # 4. 미술관 아이템 생성
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
            },
            "next_step": "guestbook",
            "guided_message": "생성된 이미지를 보며 떠오르는 감정이나 생각을 자유롭게 표현해보세요.",
        }

        logger.info(f"Reflection 단계 완료: 아이템 {gallery_item_id}")
        return journey_result

    def complete_guestbook(
        self,
        user_id: str,
        gallery_item_id: int,
        guestbook_title: str,
        guestbook_tags: List[str],
    ) -> Dict[str, Any]:
        """ACT 3단계: Defusion (방명록 작성)"""

        logger.info(f"사용자 {user_id} 방명록 작성: 아이템 {gallery_item_id}")

        # 1. 갤러리 아이템 조회
        gallery_item = self.gallery_manager.get_gallery_item(gallery_item_id)
        if not gallery_item or gallery_item.user_id != user_id:
            raise ValueError("갤러리 아이템을 찾을 수 없습니다.")

        # 2. 안내 질문 생성
        guided_question = self.prompt_architect.create_guided_question(
            guestbook_title, gallery_item.emotion_keywords
        )

        # 3. 방명록 완료
        success = self.gallery_manager.complete_guestbook(
            gallery_item_id, guestbook_title, guestbook_tags, guided_question
        )

        if not success:
            raise RuntimeError("방명록 저장에 실패했습니다.")

        # 4. 개인화 업데이트 (Level 2)
        personalization_updates = (
            self.personalization_manager.update_preferences_from_guestbook(
                user_id=user_id,
                guestbook_title=guestbook_title,
                guestbook_tags=guestbook_tags,
                image_prompt=gallery_item.reflection_prompt,
            )
        )

        guestbook_result = {
            "user_id": user_id,
            "gallery_item_id": gallery_item_id,
            "step": "defusion_complete",
            "guestbook": {
                "title": guestbook_title,
                "tags": guestbook_tags,
                "guided_question": guided_question,
            },
            "personalization_updates": personalization_updates,
            "next_step": "curator_message",
            "guided_question": guided_question,
        }

        logger.info(f"방명록 작성 완료: {guestbook_title}")
        return guestbook_result

    def create_curator_message(
        self, user_id: str, gallery_item_id: int
    ) -> Dict[str, Any]:
        """ACT 4단계: Closure (큐레이터 메시지 생성)"""

        logger.info(f"사용자 {user_id} 큐레이터 메시지 생성: 아이템 {gallery_item_id}")

        # 1. 갤러리 아이템 조회
        gallery_item = self.gallery_manager.get_gallery_item(gallery_item_id)
        if not gallery_item or gallery_item.user_id != user_id:
            raise ValueError("갤러리 아이템을 찾을 수 없습니다.")

        # 2. 사용자 프로필 조회
        user = self.user_manager.get_user(user_id)

        # 3. 개인화된 큐레이터 메시지 생성
        curator_message = self.curator_message_system.create_personalized_message(
            user=user,
            gallery_item=gallery_item,
        )

        # 4. 큐레이터 메시지 저장
        success = self.gallery_manager.add_curator_message(
            gallery_item_id, curator_message
        )

        if not success:
            raise RuntimeError("큐레이터 메시지 저장에 실패했습니다.")

        curator_result = {
            "user_id": user_id,
            "gallery_item_id": gallery_item_id,
            "step": "closure_complete",
            "curator_message": curator_message,
            "journey_complete": True,
            "completion_message": "감정의 여정이 완성되었습니다. 큐레이터가 당신의 용기를 인정합니다.",
            "next_recommendations": [
                "미술관에서 이전 작품들을 돌아보기",
                "새로운 감정 일기 작성하기",
                "치료 진행도 확인하기",
            ],
        }

        logger.info(f"큐레이터 메시지 생성 완료: 아이템 {gallery_item_id}")
        return curator_result

    def record_message_reaction(
        self,
        user_id: str,
        gallery_item_id: int,
        reaction_type: str,
        reaction_data: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """큐레이터 메시지에 대한 사용자 반응 기록"""

        logger.info(f"사용자 {user_id} 메시지 반응 기록: {reaction_type}")

        # 1. 갤러리 아이템 조회
        gallery_item = self.gallery_manager.get_gallery_item(gallery_item_id)
        if not gallery_item or gallery_item.user_id != user_id:
            raise ValueError("갤러리 아이템을 찾을 수 없습니다.")

        # 2. 반응 데이터 저장
        success = self.gallery_manager.record_message_reaction(
            gallery_item_id, reaction_type, reaction_data or {}
        )

        if not success:
            raise RuntimeError("메시지 반응 저장에 실패했습니다.")

        # 3. Level 2 개인화 업데이트 (긍정적 반응시)
        personalization_updates = {}
        if reaction_type in ["like", "save", "share"]:
            personalization_updates = (
                self.personalization_manager.update_preferences_from_message_reaction(
                    user_id=user_id,
                    reaction_type=reaction_type,
                    curator_message=gallery_item.curator_message,
                    guestbook_data={
                        "title": gallery_item.guestbook_title,
                        "tags": gallery_item.guestbook_tags,
                    },
                )
            )

        reaction_result = {
            "user_id": user_id,
            "gallery_item_id": gallery_item_id,
            "reaction_type": reaction_type,
            "reaction_recorded": True,
            "personalization_updates": personalization_updates,
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
        }

        return gallery_data

    def get_therapeutic_insights(self, user_id: str) -> Dict[str, Any]:
        """치료적 인사이트 제공"""

        # 1. 기본 사용자 통계
        user_stats = self.user_manager.get_user_stats(user_id)

        # 2. 미술관 분석
        gallery_analytics = self.gallery_manager.get_gallery_analytics(user_id)

        # 3. 개인화 인사이트
        personalization_insights = (
            self.personalization_manager.get_personalization_insights(user_id)
        )

        # 4. 컨텐츠 조정 권장사항
        content_recommendations = (
            self.personalization_manager.recommend_content_adjustments(user_id)
        )

        # 5. 큐레이터 메시지 반응 분석
        message_analytics = self.gallery_manager.get_message_reaction_analytics(user_id)

        # 6. 종합 인사이트
        insights = {
            "user_id": user_id,
            "assessment_date": datetime.now().isoformat(),
            "user_profile": user_stats,
            "emotional_journey": gallery_analytics.get("emotion_trends", {}),
            "growth_insights": gallery_analytics.get("growth_insights", []),
            "personalization_status": personalization_insights,
            "message_engagement": message_analytics,
            "recommendations": {
                "content_adjustments": content_recommendations,
                "next_actions": self._generate_next_action_recommendations(
                    user_stats, gallery_analytics
                ),
            },
            "summary": self._generate_therapeutic_summary(
                user_stats, gallery_analytics
            ),
        }

        return insights

    def check_advanced_training_readiness(self, user_id: str) -> Dict[str, Any]:
        """Level 3 고급 모델 훈련 준비 상태 확인"""

        # 1. 갤러리 데이터 조회
        gallery_items = self.gallery_manager.get_user_gallery(user_id, limit=1000)

        # 2. 완성된 여정 수 계산 (reflection + guestbook + curator_message)
        complete_journeys = [
            item
            for item in gallery_items
            if item.guestbook_title and item.curator_message
        ]

        # 3. LoRA 훈련 데이터 준비
        from ..training.lora_trainer import PersonalizedLoRATrainer

        lora_trainer = PersonalizedLoRATrainer()
        lora_data = lora_trainer.prepare_training_data(
            [item.to_dict() for item in complete_journeys]
        )
        lora_requirements = lora_trainer.get_training_requirements(len(lora_data))

        # 4. DRaFT+ 훈련 데이터 준비
        from ..training.draft_trainer import DRaFTPlusTrainer

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
        }

        return readiness

    # [이전과 동일한 헬퍼 메서드들...]
    def _analyze_emotion_moment(self, diary_text: str) -> Dict[str, Any]:
        """감정 분석 (The Moment 단계)"""
        emotion_keywords = self._extract_emotion_keywords(diary_text)
        vad_scores = self._estimate_vad_scores(emotion_keywords, diary_text)

        return {
            "diary_text": diary_text,
            "keywords": emotion_keywords,
            "vad_scores": vad_scores,
            "analysis_confidence": 0.8,
        }

    def _extract_emotion_keywords(self, text: str) -> List[str]:
        """간단한 감정 키워드 추출"""
        emotion_dict = {
            "기쁨": ["기쁨", "기쁘", "즐거", "행복", "좋", "웃", "신나"],
            "슬픔": ["슬프", "우울", "울", "눈물", "아프", "힘들"],
            "화남": ["화", "짜증", "분노", "답답", "억울", "빡"],
            "불안": ["불안", "걱정", "두렵", "무서", "긴장", "스트레스"],
            "평온": ["평온", "차분", "조용", "편안", "안정", "평화"],
            "외로움": ["외로", "혼자", "쓸쓸", "고립", "소외"],
            "피곤": ["피곤", "지쳐", "힘없", "무기력", "귀찮"],
        }

        found_keywords = []
        text_lower = text.lower()

        for emotion, keywords in emotion_dict.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_keywords.append(emotion)
                    break

        return found_keywords if found_keywords else ["중성"]

    def _estimate_vad_scores(
        self, keywords: List[str], text: str
    ) -> Tuple[float, float, float]:
        """VAD 점수 추정"""
        vad_mapping = {
            "기쁨": (0.8, 0.6, 0.4),
            "슬픔": (-0.7, -0.3, -0.5),
            "화남": (-0.6, 0.8, 0.7),
            "불안": (-0.5, 0.7, -0.5),
            "평온": (0.4, -0.7, 0.2),
            "외로움": (-0.6, -0.2, -0.6),
            "피곤": (-0.3, -0.8, -0.4),
            "중성": (0.0, 0.0, 0.0),
        }

        if not keywords:
            return (0.0, 0.0, 0.0)

        valences, arousals, dominances = zip(
            *[vad_mapping.get(k, (0, 0, 0)) for k in keywords]
        )

        return (
            sum(valences) / len(valences),
            sum(arousals) / len(arousals),
            sum(dominances) / len(dominances),
        )

    def _create_reflection_image(
        self, user, emotion_analysis: Dict[str, Any], diary_text: str
    ) -> Dict[str, Any]:
        """Reflection 이미지 생성"""
        coping_style = "balanced"
        if user.psychometric_results:
            coping_style = user.psychometric_results[0].coping_style

        reflection_prompt = self.prompt_architect.create_reflection_prompt(
            emotion_keywords=emotion_analysis["keywords"],
            vad_scores=emotion_analysis["vad_scores"],
            coping_style=coping_style,
            visual_preferences=user.visual_preferences.__dict__,
        )

        generation_result = self.image_generator.generate_image(
            prompt=reflection_prompt,
            width=512,
            height=512,
            num_inference_steps=20,
            guidance_scale=7.5,
        )

        if not generation_result["success"]:
            raise RuntimeError(f"이미지 생성 실패: {generation_result['error']}")

        return {
            "prompt": reflection_prompt,
            "image": generation_result["image"],
            "image_path": generation_result["metadata"]["timestamp"],
            "generation_time": generation_result["metadata"]["generation_time"],
        }

    def _interpret_assessment_results(
        self, result: PsychometricResult
    ) -> Dict[str, str]:
        """심리검사 결과 해석"""
        interpretations = {
            "coping_style_description": {
                "avoidant": "감정적 상황을 회피하거나 우회하는 경향이 있습니다.",
                "confrontational": "감정적 상황에 직면하고 적극적으로 대처하는 경향이 있습니다.",
                "balanced": "상황에 따라 유연하게 대처하는 균형잡힌 스타일을 보입니다.",
            }[result.coping_style],
            "severity_description": {
                "mild": "가벼운 수준의 우울 증상이 관찰됩니다.",
                "moderate": "중등도의 우울 증상이 있어 주의가 필요합니다.",
                "severe": "심한 우울 증상이 있어 전문적 도움이 권장됩니다.",
            }[result.severity_level],
        }

        return interpretations

    def _generate_assessment_recommendations(
        self, result: PsychometricResult
    ) -> List[str]:
        """심리검사 기반 권장사항"""
        recommendations = []

        if result.coping_style == "avoidant":
            recommendations.extend(
                [
                    "부드럽고 은유적인 감정 표현을 통해 점진적으로 감정에 다가가보세요.",
                    "이미지 생성 시 직접적이지 않은 추상적 표현을 활용하겠습니다.",
                    "감정을 안전한 거리에서 관찰하고 수용하는 연습을 해보세요.",
                ]
            )
        elif result.coping_style == "confrontational":
            recommendations.extend(
                [
                    "감정을 직접적이고 명확하게 표현하는 것이 도움이 될 것 같습니다.",
                    "이미지를 통해 감정의 본질을 깊이 탐구해보세요.",
                    "감정의 강도와 복잡성을 있는 그대로 받아들여보세요.",
                ]
            )
        else:  # balanced
            recommendations.extend(
                [
                    "상황에 맞는 유연한 감정 표현을 시도해보세요.",
                    "다양한 스타일의 이미지를 통해 감정의 여러 면을 탐색해보세요.",
                ]
            )

        if result.severity_level == "severe":
            recommendations.append("전문 상담사와의 상담을 병행하시기를 권장합니다.")

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
        """고급 훈련 권장사항"""
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
            recommendations.append("고급 개인화 모델을 훈련할 준비가 되었습니다!")

        return recommendations

    def cleanup(self):
        """시스템 리소스 정리"""
        if hasattr(self.image_generator, "cleanup"):
            self.image_generator.cleanup()

        logger.info("ACT 치료 시스템 리소스 정리 완료")
