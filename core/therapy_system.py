#!/usr/bin/env python3
"""
EmotionalImageTherapySystem - 감정 기반 이미지 치료 시스템
"""

import os
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import torch
from PIL import Image

from config import (
    device,
    logger,
    TRANSFORMERS_AVAILABLE,
    DIFFUSERS_AVAILABLE,
    PEFT_AVAILABLE,
)
from models.emotion import EmotionEmbedding
from models.emotion_mapper import AdvancedEmotionMapper
from models.user_profile import UserEmotionProfile
from models.lora_manager import PersonalizedLoRAManager
from models.reward_model import DRaFTPlusRewardModel
from training.trainer import DRaFTPlusTrainer

# 경고 메시지 억제
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 선택적 임포트
if DIFFUSERS_AVAILABLE:
    from diffusers import (
        StableDiffusionPipeline,
        EulerDiscreteScheduler,
    )


class EmotionalImageTherapySystem:
    """감정 기반 이미지 치료 시스템"""

    def __init__(self, model_path: str = "runwayml/stable-diffusion-v1-5"):
        self.model_path = model_path
        self.device = device

        # 출력 디렉토리 생성
        self.output_dir = Path("generated_images")
        self.output_dir.mkdir(exist_ok=True)

        # 컴포넌트 초기화
        logger.info("🚀 시스템 초기화 시작...")

        # 1. 감정 매퍼 초기화
        self.emotion_mapper = AdvancedEmotionMapper()

        # 2. LoRA 매니저 초기화
        self.lora_manager = PersonalizedLoRAManager(model_path)

        # 3. SD 파이프라인 로드
        self.pipeline = self._load_pipeline()

        # 4. 보상 모델 및 트레이너 초기화
        if self.pipeline:
            self.reward_model = DRaFTPlusRewardModel(self.device)
            self.trainer = DRaFTPlusTrainer(self.pipeline, self.reward_model)
        else:
            self.reward_model = None
            self.trainer = None

        # 5. 사용자 프로파일 캐시
        self.user_profiles = {}

        logger.info("✅ 시스템 초기화 완료!")

    def _load_pipeline(self):
        """SD 파이프라인 로드"""
        if not DIFFUSERS_AVAILABLE:
            logger.error("❌ Diffusers 라이브러리가 필요합니다")
            return None

        try:
            logger.info(f"📦 Stable Diffusion 파이프라인 로드 중: {self.model_path}")

            pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=(
                    # 유효하지 않은 숫자가 들어가는 오류가 발생하므로 모두 float32로 설정
                    torch.float32
                    if self.device.type == "mps"
                    else torch.float32
                ),
                use_safetensors=True,
                safety_checker=None,  # 빠른 생성을 위해 비활성화
                requires_safety_checker=False,
            )

            # 최적화 설정
            pipeline = pipeline.to(self.device)

            # 메모리 최적화
            pipeline.enable_attention_slicing()

            if self.device.type == "cuda":
                pipeline.enable_sequential_cpu_offload()

            # 빠른 스케줄러로 변경
            pipeline.scheduler = EulerDiscreteScheduler.from_config(
                pipeline.scheduler.config
            )

            logger.info("✅ SD 파이프라인 로드 및 최적화 완료")
            return pipeline

        except Exception as e:
            logger.error(f"❌ SD 파이프라인 로드 실패: {e}")
            return None

    def get_user_profile(self, user_id: str) -> UserEmotionProfile:
        """사용자 프로파일 가져오기 또는 생성"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserEmotionProfile(user_id)
            logger.info(f"✅ 새 사용자 프로파일 생성: {user_id}")
        return self.user_profiles[user_id]

    def generate_therapeutic_image(
        self,
        user_id: str,
        input_text: str,
        base_prompt: str = "",
        num_inference_steps: int = 15,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
    ) -> Dict[str, Any]:
        """치료용 이미지 생성"""

        try:
            logger.info(f"🎨 사용자 {user_id}의 이미지 생성 시작")
            logger.info(f"📝 입력 텍스트: {input_text}")

            # 1. 사용자 프로파일 로드
            user_profile = self.get_user_profile(user_id)

            # 2. 감정 분석
            emotion = self.emotion_mapper.extract_emotion_from_text(input_text)
            logger.info(
                f"😊 감정 분석: V={emotion.valence:.3f}, A={emotion.arousal:.3f}, D={emotion.dominance:.3f}"
            )

            # 3. 프롬프트 생성
            emotion_modifiers = self.emotion_mapper.emotion_to_prompt_modifiers(emotion)
            personal_modifiers = user_profile.get_personalized_style_modifiers()

            # 기본 프롬프트가 없으면 생성
            if not base_prompt:
                base_prompt = "digital art, beautiful scene"

            final_prompt = f"{base_prompt}, {emotion_modifiers}, {personal_modifiers}"
            final_prompt += ", high quality, detailed, masterpiece"

            logger.info(f"🎯 최종 프롬프트: {final_prompt}")

            # 4. 이미지 생성
            if self.pipeline:
                # SD 파이프라인 사용
                with torch.autocast(
                    self.device.type if self.device.type != "mps" else "cpu"
                ):
                    result = self.pipeline(
                        prompt=final_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height,
                        output_type="pil",
                    )

                generated_image = result.images[0]
                logger.info("✅ SD 파이프라인으로 이미지 생성 완료")
            else:
                # 폴백: 간단한 이미지 생성
                generated_image = self._generate_fallback_image(emotion, width, height)
                logger.info("⚠️ 폴백 이미지 생성기 사용")

            # 5. 이미지 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"{user_id}_{timestamp}.png"
            image_path = self.output_dir / image_filename
            generated_image.save(image_path)

            # 6. 데이터베이스에 기록
            emotion_id = user_profile.add_emotion_record(
                input_text=input_text,
                emotion=emotion,
                generated_prompt=final_prompt,
                image_path=str(image_path),
            )

            # 7. 메타데이터 구성
            metadata = {
                "emotion_id": emotion_id,
                "user_id": user_id,
                "input_text": input_text,
                "emotion": emotion.to_dict(),
                "final_prompt": final_prompt,
                "image_path": str(image_path),
                "image_filename": image_filename,
                "generation_params": {
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "width": width,
                    "height": height,
                },
                "timestamp": timestamp,
                "device": str(self.device),
            }

            logger.info(f"✅ 이미지 생성 완료: {image_path}")
            return {"success": True, "image": generated_image, "metadata": metadata}

        except Exception as e:
            logger.error(f"❌ 이미지 생성 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "metadata": {"user_id": user_id, "input_text": input_text},
            }

    def _generate_fallback_image(
        self, emotion: EmotionEmbedding, width: int = 512, height: int = 512
    ) -> Image.Image:
        """폴백 이미지 생성 (SD 파이프라인 실패시)"""

        # 감정 기반 색상 생성
        if emotion.valence > 0.3:
            # 긍정적 감정 - 따뜻한 색상
            base_color = [0.9, 0.8, 0.6]  # 따뜻한 노란색
        elif emotion.valence < -0.3:
            # 부정적 감정 - 차가운 색상
            base_color = [0.6, 0.7, 0.9]  # 차가운 파란색
        else:
            # 중성 감정 - 중간 색상
            base_color = [0.7, 0.7, 0.8]  # 회색빛

        # 각성도 기반 강도 조정
        intensity = 0.5 + abs(emotion.arousal) * 0.5
        base_color = [c * intensity for c in base_color]

        # 그라데이션 이미지 생성
        image_array = np.zeros((height, width, 3))

        for i in range(height):
            for j in range(width):
                # 중심에서의 거리 기반 그라데이션
                center_x, center_y = width // 2, height // 2
                distance = np.sqrt((j - center_x) ** 2 + (i - center_y) ** 2)
                max_distance = np.sqrt(center_x**2 + center_y**2)

                # 감정 기반 그라데이션 패턴
                if emotion.dominance > 0:
                    # 지배적 감정 - 중심에서 바깥으로
                    factor = 1.0 - (distance / max_distance) * 0.5
                else:
                    # 수동적 감정 - 바깥에서 중심으로
                    factor = 0.5 + (distance / max_distance) * 0.5

                image_array[i, j] = [c * factor for c in base_color]

        # numpy 배열을 PIL 이미지로 변환
        image_array = np.clip(image_array * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(image_array)

    def process_feedback(
        self,
        user_id: str,
        emotion_id: int,
        feedback_score: float,
        feedback_type: str = "rating",
        comments: str = None,
        enable_training: bool = True,
    ) -> Dict[str, Any]:
        """사용자 피드백 처리 및 개인화 학습"""

        try:
            logger.info(f"📝 사용자 {user_id} 피드백 처리: 점수 {feedback_score}")

            # 1. 사용자 프로파일 로드
            user_profile = self.get_user_profile(user_id)

            # 2. 피드백 저장
            user_profile.add_feedback(
                emotion_id=emotion_id,
                feedback_score=feedback_score,
                feedback_type=feedback_type,
                comments=comments,
            )

            # 3. 강화학습 수행 (옵션)
            training_result = None
            if (
                enable_training and self.trainer and feedback_score != 3.0
            ):  # 중성 피드백 제외

                # 해당 감정 기록 찾기
                emotion_record = None
                for record in user_profile.emotion_history:
                    if record.get("id") == emotion_id:
                        emotion_record = record
                        break

                if emotion_record:
                    logger.info("🤖 개인화 학습 시작...")
                    training_result = self.trainer.train_step(
                        prompt=emotion_record["generated_prompt"],
                        target_emotion=emotion_record["emotion"],
                        user_profile=user_profile,
                        num_inference_steps=8,  # 빠른 학습
                    )
                    logger.info(
                        f"✅ 학습 완료: 보상 {training_result.get('total_reward', 0):.3f}"
                    )

            # 4. LoRA 어댑터 저장 (주기적)
            if len(user_profile.feedback_history) % 5 == 0:  # 5번째 피드백마다 저장
                self._save_user_lora_if_needed(user_id, user_profile)

            # 5. 치료 인사이트 업데이트
            insights = user_profile.get_therapeutic_insights()

            result = {
                "success": True,
                "feedback_recorded": True,
                "training_performed": training_result is not None,
                "training_result": training_result,
                "therapeutic_insights": insights,
                "total_interactions": len(user_profile.emotion_history),
                "total_feedbacks": len(user_profile.feedback_history),
            }

            logger.info("✅ 피드백 처리 완료")
            return result

        except Exception as e:
            logger.error(f"❌ 피드백 처리 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "feedback_recorded": False,
                "training_performed": False,
            }

    def _save_user_lora_if_needed(self, user_id: str, user_profile: UserEmotionProfile):
        """필요시 사용자 LoRA 어댑터 저장"""
        try:
            if self.pipeline and hasattr(self.pipeline, "unet"):
                # 현재 모델 상태를 LoRA로 저장
                model_state = {
                    "unet_state_dict": self.pipeline.unet.state_dict(),
                    "user_preferences": user_profile.preference_weights,
                    "training_metadata": user_profile.learning_metadata,
                }

                self.lora_manager.save_user_lora(user_id, model_state)
                logger.info(f"💾 사용자 {user_id} LoRA 어댑터 저장")
        except Exception as e:
            logger.warning(f"⚠️ LoRA 저장 실패: {e}")

    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """사용자 치료 인사이트 제공"""
        user_profile = self.get_user_profile(user_id)
        return user_profile.get_therapeutic_insights()

    def get_emotion_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """사용자 감정 히스토리 조회"""
        user_profile = self.get_user_profile(user_id)
        return user_profile.emotion_history[-limit:]

    def cleanup_old_images(self, days_old: int = 30):
        """오래된 이미지 파일 정리"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            cleaned_count = 0

            for image_file in self.output_dir.glob("*.png"):
                if image_file.stat().st_mtime < cutoff_date.timestamp():
                    image_file.unlink()
                    cleaned_count += 1

            logger.info(f"🧹 오래된 이미지 {cleaned_count}개 정리 완료")
            return cleaned_count

        except Exception as e:
            logger.error(f"❌ 이미지 정리 실패: {e}")
            return 0