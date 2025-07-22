# src/image_generator.py

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
    logger.warning(
        "Diffusers 라이브러리를 사용할 수 없습니다. 폴백 이미지 생성을 사용합니다."
    )


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
            logger.info("Diffusers를 사용할 수 없어 폴백 모드로 실행합니다.")
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

            # 메모리 최적화
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

        # 기본값 설정
        if negative_prompt is None:
            negative_prompt = self.default_config["negative_prompt"]

        # 시드 설정
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        try:
            if self.pipeline is not None:
                # Stable Diffusion으로 생성
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
            else:
                # 폴백 이미지 생성
                image = self._generate_fallback_image(prompt, width, height)
                generation_method = "fallback"

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
                "error": str(e),
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

    def _generate_fallback_image(
        self, prompt: str, width: int = 512, height: int = 512
    ) -> Image.Image:
        """폴백 이미지 생성 (프롬프트 기반 색상 패턴)"""

        # 프롬프트에서 색상 힌트 추출
        color_hints = self._extract_color_hints(prompt)
        mood_hints = self._extract_mood_hints(prompt)

        # 기본 색상 설정
        if "warm" in color_hints or "golden" in color_hints:
            base_color = [0.9, 0.7, 0.4]  # 따뜻한 골든
        elif "cool" in color_hints or "blue" in color_hints:
            base_color = [0.4, 0.6, 0.9]  # 차가운 블루
        elif "green" in color_hints or "nature" in color_hints:
            base_color = [0.5, 0.8, 0.5]  # 자연 그린
        elif "peaceful" in mood_hints or "calm" in mood_hints:
            base_color = [0.7, 0.7, 0.9]  # 평온한 라벤더
        elif "melancholy" in mood_hints or "sad" in mood_hints:
            base_color = [0.6, 0.6, 0.7]  # 차분한 그레이
        else:
            base_color = [0.7, 0.7, 0.8]  # 기본 중성색

        # 그라데이션 패턴 생성
        image_array = self._create_gradient_pattern(
            base_color, width, height, mood_hints
        )

        # numpy 배열을 PIL 이미지로 변환
        image_array = np.clip(image_array * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(image_array)

    def _extract_color_hints(self, prompt: str) -> set:
        """프롬프트에서 색상 힌트 추출"""
        color_keywords = {
            "warm",
            "cool",
            "golden",
            "blue",
            "red",
            "green",
            "purple",
            "orange",
            "pastel",
            "vibrant",
            "muted",
            "bright",
            "dark",
            "light",
        }

        prompt_lower = prompt.lower()
        return {keyword for keyword in color_keywords if keyword in prompt_lower}

    def _extract_mood_hints(self, prompt: str) -> set:
        """프롬프트에서 분위기 힌트 추출"""
        mood_keywords = {
            "peaceful",
            "calm",
            "serene",
            "melancholy",
            "joyful",
            "energetic",
            "contemplative",
            "dynamic",
            "gentle",
            "intense",
            "quiet",
            "vibrant",
            "sad",
            "happy",
            "turbulent",
            "stormy",
            "sunny",
            "cloudy",
        }

        prompt_lower = prompt.lower()
        return {keyword for keyword in mood_keywords if keyword in prompt_lower}

    def _create_gradient_pattern(
        self, base_color: list, width: int, height: int, mood_hints: set
    ) -> np.ndarray:
        """그라데이션 패턴 생성"""

        image_array = np.zeros((height, width, 3))
        center_x, center_y = width // 2, height // 2

        for i in range(height):
            for j in range(width):
                # 중심으로부터의 거리 계산
                distance = np.sqrt((j - center_x) ** 2 + (i - center_y) ** 2)
                max_distance = np.sqrt(center_x**2 + center_y**2)
                normalized_distance = distance / max_distance

                # 분위기에 따른 패턴 조절
                if "turbulent" in mood_hints or "stormy" in mood_hints:
                    # 소용돌이 패턴
                    angle = np.arctan2(i - center_y, j - center_x)
                    factor = 0.7 + 0.3 * np.sin(angle * 3 + normalized_distance * 10)
                elif "peaceful" in mood_hints or "calm" in mood_hints:
                    # 부드러운 원형 그라데이션
                    factor = 1.0 - normalized_distance * 0.3
                elif "dynamic" in mood_hints or "energetic" in mood_hints:
                    # 방사형 패턴
                    factor = 0.8 + 0.2 * np.cos(normalized_distance * 8)
                else:
                    # 기본 중심 그라데이션
                    factor = 1.0 - normalized_distance * 0.4

                # 색상 적용
                factor = max(0.3, min(1.0, factor))  # 최소/최대값 제한
                image_array[i, j] = [c * factor for c in base_color]

        return image_array

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

    def __del__(self):
        """소멸자"""
        self.cleanup()
