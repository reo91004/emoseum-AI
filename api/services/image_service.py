# api/services/image_service.py

import httpx
import base64
import asyncio
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
import logging

from api.config import settings

logger = logging.getLogger(__name__)


class ImageGenerationStrategy(ABC):
    """이미지 생성 전략 인터페이스"""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """이미지 생성 추상 메서드"""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """백엔드 상태 확인"""
        pass


class LocalGPUStrategy(ImageGenerationStrategy):
    """로컬 GPU를 사용한 Stable Diffusion"""

    def __init__(self):
        self.generator = None
        self.model_path = settings.local_model_path
        self.is_initialized = False

        # 초기화는 첫 사용 시 지연 로딩
        logger.info(f"🖼️  LocalGPU 전략 초기화 (모델: {self.model_path})")

    def _initialize_generator(self):
        """지연 초기화: 실제 사용 시점에 모델 로드"""
        if self.is_initialized:
            return

        try:
            # 기존 ImageGenerator 클래스 활용
            from src.services.image_generator import ImageGenerator

            self.generator = ImageGenerator(self.model_path)
            self.is_initialized = True
            logger.info("✅ 로컬 GPU 이미지 생성기 초기화 완료")

        except Exception as e:
            logger.error(f"❌ 로컬 GPU 이미지 생성기 초기화 실패: {e}")
            raise

    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """로컬 GPU로 이미지 생성"""
        try:
            # 지연 초기화
            if not self.is_initialized:
                self._initialize_generator()

            # 비동기 실행을 위해 executor 사용
            loop = asyncio.get_event_loop()

            # 기본 설정
            generation_params = {
                "width": kwargs.get("width", 512),
                "height": kwargs.get("height", 512),
                "num_inference_steps": kwargs.get("num_inference_steps", 20),
                "guidance_scale": kwargs.get("guidance_scale", 7.5),
                "seed": kwargs.get("seed", None),
            }

            # 블로킹 호출을 비동기로 실행
            result = await loop.run_in_executor(
                None,
                self.generator.generate_image,
                prompt,
                generation_params.get("width"),
                generation_params.get("height"),
                generation_params.get("num_inference_steps"),
                generation_params.get("guidance_scale"),
                generation_params.get("seed"),
            )

            if result["success"]:
                logger.info(f"✅ 로컬 GPU 이미지 생성 완료: {prompt[:50]}...")

                # 이미지를 base64로 인코딩하여 반환
                if result.get("image"):
                    import io

                    img_bytes = io.BytesIO()
                    result["image"].save(img_bytes, format="PNG")
                    img_bytes.seek(0)

                    image_b64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")

                    return {
                        "success": True,
                        "image_b64": image_b64,
                        "image_url": None,  # 로컬에서는 base64 사용
                        "prompt": prompt,
                        "backend": "local_gpu",
                        "generation_time": result.get("metadata", {}).get(
                            "generation_time", 0
                        ),
                        "metadata": result.get("metadata", {}),
                    }
                else:
                    return {
                        "success": False,
                        "error": "No image generated",
                        "backend": "local_gpu",
                    }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                    "backend": "local_gpu",
                }

        except Exception as e:
            logger.error(f"❌ 로컬 GPU 이미지 생성 실패: {e}")
            return {
                "success": False,
                "error": f"Local GPU generation failed: {str(e)}",
                "backend": "local_gpu",
                "retry_recommended": True,
            }

    def get_status(self) -> Dict[str, Any]:
        """로컬 GPU 상태 확인"""
        try:
            import torch

            status = {
                "backend": "local_gpu",
                "available": True,
                "model_path": self.model_path,
                "initialized": self.is_initialized,
                "cuda_available": torch.cuda.is_available(),
                "mps_available": hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available(),
            }

            if torch.cuda.is_available():
                status["gpu_info"] = {
                    "device_name": torch.cuda.get_device_name(0),
                    "memory_allocated": torch.cuda.memory_allocated(0),
                    "memory_reserved": torch.cuda.memory_reserved(0),
                }
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                status["device"] = "Apple Silicon MPS"
            else:
                status["device"] = "CPU"
                status["warning"] = "GPU not available, using CPU (slow)"

            return status

        except ImportError:
            return {
                "backend": "local_gpu",
                "available": False,
                "error": "PyTorch not installed",
            }
        except Exception as e:
            return {"backend": "local_gpu", "available": False, "error": str(e)}


class RemoteGPUStrategy(ImageGenerationStrategy):
    """외부 GPU 서버 API 호출"""

    def __init__(self):
        self.api_url = settings.remote_gpu_url
        self.api_token = settings.remote_gpu_token
        self.timeout = 120.0  # 2분 타임아웃

        logger.info(f"🌐 RemoteGPU 전략 초기화 (URL: {self.api_url})")

    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """외부 GPU 서버로 이미지 생성"""
        if not self.api_url or not self.api_token:
            return {
                "success": False,
                "error": "Remote GPU settings not configured",
                "backend": "remote_gpu",
            }

        try:
            generation_params = {
                "prompt": prompt,
                "width": kwargs.get("width", 512),
                "height": kwargs.get("height", 512),
                "num_inference_steps": kwargs.get("num_inference_steps", 20),
                "guidance_scale": kwargs.get("guidance_scale", 7.5),
                "seed": kwargs.get("seed", None),
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.api_url}/generate",
                    json=generation_params,
                    headers={"Authorization": f"Bearer {self.api_token}"},
                )

                if response.status_code == 200:
                    result = response.json()

                    logger.info(f"✅ 원격 GPU 이미지 생성 완료: {prompt[:50]}...")

                    return {
                        "success": True,
                        "image_b64": result.get("image_b64"),
                        "image_url": result.get("image_url"),
                        "prompt": prompt,
                        "backend": "remote_gpu",
                        "generation_time": result.get("generation_time", 0),
                        "metadata": result.get("metadata", {}),
                    }

                else:
                    logger.error(f"❌ 원격 GPU API 오류: {response.status_code}")
                    return {
                        "success": False,
                        "error": f"Remote API error: {response.status_code} - {response.text}",
                        "backend": "remote_gpu",
                        "retry_recommended": response.status_code >= 500,
                    }

        except httpx.TimeoutException:
            logger.error("❌ 원격 GPU API 타임아웃")
            return {
                "success": False,
                "error": "Remote API timeout",
                "backend": "remote_gpu",
                "retry_recommended": True,
            }

        except Exception as e:
            logger.error(f"❌ 원격 GPU 이미지 생성 실패: {e}")
            return {
                "success": False,
                "error": f"Remote GPU generation failed: {str(e)}",
                "backend": "remote_gpu",
                "retry_recommended": True,
            }

    def get_status(self) -> Dict[str, Any]:
        """원격 GPU 상태 확인"""
        if not self.api_url or not self.api_token:
            return {
                "backend": "remote_gpu",
                "available": False,
                "error": "Remote GPU settings not configured",
            }

        # 실제로는 /health 엔드포인트 호출해야 함
        return {
            "backend": "remote_gpu",
            "available": True,
            "api_url": self.api_url,
            "configured": bool(self.api_token),
            "note": "Status check requires actual API call",
        }


class ColabStrategy(ImageGenerationStrategy):
    """Google Colab 런타임 활용"""

    def __init__(self):
        self.notebook_url = settings.colab_notebook_url
        self.access_token = settings.colab_access_token
        self.timeout = 180.0  # 3분 타임아웃

        logger.info(f"📔 Colab 전략 초기화 (URL: {self.notebook_url})")

    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Google Colab을 통한 이미지 생성"""
        if not self.notebook_url:
            return {
                "success": False,
                "error": "Colab settings not configured",
                "backend": "colab",
            }

        try:
            # Colab 노트북과 통신하는 로직
            # 실제 구현에서는 ngrok URL이나 Colab API 활용

            generation_params = {
                "prompt": prompt,
                "width": kwargs.get("width", 512),
                "height": kwargs.get("height", 512),
                "num_inference_steps": kwargs.get("num_inference_steps", 20),
                "guidance_scale": kwargs.get("guidance_scale", 7.5),
            }

            # 임시 구현 - 실제로는 Colab 런타임과 통신
            logger.warning("⚠️  Colab 통합은 아직 구현되지 않았습니다")

            return {
                "success": False,
                "error": "Colab integration not implemented yet",
                "backend": "colab",
                "note": "Colab integration requires ngrok tunnel or direct API",
            }

        except Exception as e:
            logger.error(f"❌ Colab 이미지 생성 실패: {e}")
            return {
                "success": False,
                "error": f"Colab generation failed: {str(e)}",
                "backend": "colab",
            }

    def get_status(self) -> Dict[str, Any]:
        """Colab 상태 확인"""
        return {
            "backend": "colab",
            "available": False,
            "notebook_url": self.notebook_url,
            "configured": bool(self.notebook_url),
            "note": "Colab integration not implemented yet",
        }


class ImageService:
    """이미지 생성 서비스 (Strategy Pattern)"""

    def __init__(self):
        self.strategy = self._get_strategy()
        self.supported_backends = ["local", "remote", "colab"]

        logger.info(f"🎨 이미지 서비스 초기화 - 백엔드: {settings.image_backend}")

    def _get_strategy(self) -> ImageGenerationStrategy:
        """설정에 따른 전략 선택"""
        backend = settings.image_backend.lower()

        if backend == "local":
            return LocalGPUStrategy()
        elif backend == "remote":
            return RemoteGPUStrategy()
        elif backend == "colab":
            return ColabStrategy()
        else:
            logger.warning(f"⚠️  지원하지 않는 이미지 백엔드: {backend}, local로 대체")
            return LocalGPUStrategy()

    async def generate_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """이미지 생성 (비동기)"""
        start_time = datetime.utcnow()

        try:
            # 프롬프트 유효성 검증
            if not prompt or not prompt.strip():
                return {
                    "success": False,
                    "error": "Empty prompt provided",
                    "backend": settings.image_backend,
                }

            # 프롬프트 길이 제한
            if len(prompt) > 500:
                prompt = prompt[:500]
                logger.warning("⚠️  프롬프트가 500자로 잘렸습니다")

            # 이미지 생성
            result = await self.strategy.generate(prompt, **kwargs)

            # 생성 시간 계산
            generation_time = (datetime.utcnow() - start_time).total_seconds()

            if "generation_time" not in result:
                result["generation_time"] = generation_time

            result["requested_at"] = start_time.isoformat()
            result["completed_at"] = datetime.utcnow().isoformat()

            if result["success"]:
                logger.info(
                    f"✅ 이미지 생성 성공 ({settings.image_backend}): {generation_time:.2f}s"
                )
            else:
                logger.error(
                    f"❌ 이미지 생성 실패 ({settings.image_backend}): {result.get('error')}"
                )

            return result

        except Exception as e:
            generation_time = (datetime.utcnow() - start_time).total_seconds()

            logger.error(f"❌ 이미지 생성 서비스 오류: {e}")

            return {
                "success": False,
                "error": f"Image service error: {str(e)}",
                "backend": settings.image_backend,
                "generation_time": generation_time,
                "retry_recommended": True,
            }

    def get_backend_status(self) -> Dict[str, Any]:
        """현재 백엔드 상태 확인"""
        try:
            status = self.strategy.get_status()
            status.update(
                {
                    "service_info": {
                        "current_backend": settings.image_backend,
                        "supported_backends": self.supported_backends,
                        "service_status": "operational",
                    },
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

            return status

        except Exception as e:
            logger.error(f"❌ 백엔드 상태 확인 실패: {e}")
            return {
                "backend": settings.image_backend,
                "available": False,
                "error": str(e),
                "service_info": {
                    "current_backend": settings.image_backend,
                    "supported_backends": self.supported_backends,
                    "service_status": "error",
                },
                "timestamp": datetime.utcnow().isoformat(),
            }

    def switch_backend(self, new_backend: str) -> Dict[str, Any]:
        """백엔드 동적 변경 (개발/테스트용)"""
        if new_backend not in self.supported_backends:
            return {
                "success": False,
                "error": f"Unsupported backend: {new_backend}",
                "supported": self.supported_backends,
            }

        try:
            old_backend = settings.image_backend
            settings.image_backend = new_backend
            self.strategy = self._get_strategy()

            logger.info(f"🔄 이미지 백엔드 변경: {old_backend} → {new_backend}")

            return {
                "success": True,
                "old_backend": old_backend,
                "new_backend": new_backend,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"❌ 백엔드 변경 실패: {e}")
            return {"success": False, "error": str(e)}

    async def batch_generate(self, prompts: list, **kwargs) -> List[Dict[str, Any]]:
        """배치 이미지 생성"""
        if len(prompts) > 10:
            logger.warning("⚠️  배치 생성은 최대 10개까지 지원됩니다")
            prompts = prompts[:10]

        results = []

        for i, prompt in enumerate(prompts):
            logger.info(f"🎨 배치 생성 중 ({i+1}/{len(prompts)}): {prompt[:30]}...")
            result = await self.generate_image(prompt, **kwargs)
            result["batch_index"] = i
            results.append(result)

            # 연속 생성 간 짧은 대기 (시스템 부하 방지)
            if i < len(prompts) - 1:
                await asyncio.sleep(1)

        successful = sum(1 for r in results if r["success"])
        logger.info(f"✅ 배치 생성 완료: {successful}/{len(prompts)} 성공")

        return results
