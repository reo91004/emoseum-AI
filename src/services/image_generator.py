# src/services/image_generator.py

# ==============================================================================
# 이 파일은 Stable Diffusion 모델을 사용하여 텍스트 프롬프트로부터 이미지를 생성하는 역할을 한다.
# `diffusers` 라이브러리를 사용하여 모델을 로드하고, `prompt_architect`로부터 전달받은 프롬프트를 기반으로
# 이미지를 생성한다. 생성된 이미지는 `ACTTherapySystem`을 통해 `gallery_manager`에 저장된다.
# ==============================================================================

import torch
from PIL import Image
import numpy as np
from typing import Dict, Optional, Tuple, Any
import logging
import warnings
from datetime import datetime

# 경고 메시지 억제
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

try:
    from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logger.warning("Diffusers 라이브러리를 사용할 수 없습니다.")


class ImageGenerator:
    """이미지 생성 래퍼 클래스"""

    def __init__(self, model_path: str = "runwayml/stable-diffusion-v1-5"):
        self.model_path = model_path
        self.device = self._get_device()
        self.pipeline = None

        # 기본 생성 설정
        self.default_config = {
            "width": 512,
            "height": 512,
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "negative_prompt": "photorealistic, 3d render, harsh lighting, ugly, deformed, noisy, blurry, low quality",
        }

        self._load_pipeline()

    def _get_device(self) -> torch.device:
        """디바이스 결정"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _load_pipeline(self):
        """SD 파이프라인 로드"""
        if not DIFFUSERS_AVAILABLE:
            logger.warning(
                "Diffusers 라이브러리가 설치되지 않아 이미지 생성이 불가능합니다."
            )
            return

        try:
            logger.info(f"Stable Diffusion 파이프라인 로드 중: {self.model_path}")

            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,  # MPS 호환성을 위해 float32 사용
                use_safetensors=True,
                safety_checker=None,
                requires_safety_checker=False,
            )

            self.pipeline = self.pipeline.to(self.device)

            # 메모리 정리
            self.pipeline.enable_attention_slicing()

            if self.device.type == "cuda":
                self.pipeline.enable_sequential_cpu_offload()

            # 빠른 스케줄러로 변경
            self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
                self.pipeline.scheduler.config
            )

            logger.info(f"SD 파이프라인 로드 완료 (디바이스: {self.device})")

        except Exception as e:
            logger.error(f"SD 파이프라인 로드 실패: {e}")
            self.pipeline = None

    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """이미지 생성"""

        generation_start = datetime.now()

        # 라이브러리 가용성 확인
        if not DIFFUSERS_AVAILABLE:
            return {
                "success": False,
                "error": "Stable Diffusion libraries not available",
                "requires_setup": True,
                "setup_instructions": "Install diffusers, transformers, and accelerate packages",
                "image": None,
                "metadata": {"prompt": prompt},
            }

        # 파이프라인 로드 확인
        if self.pipeline is None:
            return {
                "success": False,
                "error": "Stable Diffusion pipeline failed to load",
                "requires_setup": True,
                "retry_recommended": True,
                "image": None,
                "metadata": {"prompt": prompt},
            }

        # 기본값 설정
        if negative_prompt is None:
            negative_prompt = self.default_config["negative_prompt"]

        # 시드 설정
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        try:
            # Stable Diffusion으로 이미지 생성
            image = self._generate_with_sd(
                prompt,
                negative_prompt,
                width,
                height,
                num_inference_steps,
                guidance_scale,
                generator,
            )
            generation_method = "stable_diffusion"

            generation_time = (datetime.now() - generation_start).total_seconds()

            result = {
                "success": True,
                "image": image,
                "metadata": {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "seed": seed,
                    "device": str(self.device),
                    "generation_method": generation_method,
                    "generation_time": generation_time,
                    "timestamp": generation_start.isoformat(),
                },
            }

            logger.info(
                f"이미지 생성 완료 ({generation_method}, {generation_time:.2f}초)"
            )
            return result

        except Exception as e:
            logger.error(f"이미지 생성 실패: {e}")
            return {
                "success": False,
                "error": f"Image generation failed: {str(e)}",
                "retry_recommended": True,
                "image": None,
                "metadata": {"prompt": prompt},
            }

    def _generate_with_sd(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        generator,
    ) -> Image.Image:
        """Stable Diffusion으로 이미지 생성"""

        with torch.autocast(self.device.type if self.device.type != "mps" else "cpu"):
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="pil",
            )

        return result.images[0]

    def batch_generate(self, prompts: list, **kwargs) -> list:
        """배치 이미지 생성"""
        results = []

        for i, prompt in enumerate(prompts):
            logger.info(f"배치 생성 진행중: {i+1}/{len(prompts)}")
            result = self.generate_image(prompt, **kwargs)
            results.append(result)

        return results

    def get_pipeline_info(self) -> Dict[str, Any]:
        """파이프라인 정보 반환"""
        info = {
            "model_path": self.model_path,
            "device": str(self.device),
            "pipeline_loaded": self.pipeline is not None,
            "diffusers_available": DIFFUSERS_AVAILABLE,
            "default_config": self.default_config,
            "fallback_system": False,  # 폴백 시스템 없음 명시
        }

        if self.pipeline is not None:
            info.update(
                {
                    "scheduler_type": type(self.pipeline.scheduler).__name__,
                    "vae_available": hasattr(self.pipeline, "vae"),
                    "text_encoder_available": hasattr(self.pipeline, "text_encoder"),
                }
            )

        return info

    def update_default_config(self, **kwargs):
        """기본 설정 업데이트"""
        for key, value in kwargs.items():
            if key in self.default_config:
                self.default_config[key] = value
                logger.info(f"기본 설정 업데이트: {key} = {value}")

    def cleanup(self):
        """메모리 정리"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            logger.info("파이프라인 메모리가 정리되었습니다.")

    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 확인"""
        return {
            "generation_method": "stable_diffusion_only",
            "fallback_available": False,
            "diffusers_available": DIFFUSERS_AVAILABLE,
            "pipeline_loaded": self.pipeline is not None,
            "device": str(self.device),
            "model_path": self.model_path,
            "graceful_failure": True,
            "handover_status": "completed",
        }

    def __del__(self):
        """소멸자"""
        self.cleanup()
