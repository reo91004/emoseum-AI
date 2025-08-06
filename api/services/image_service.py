# api/services/image_service.py

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
import sys

# Add parent directory to path to import existing services
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.services.image_generator import ImageGenerator

logger = logging.getLogger(__name__)


class ImageGenerationService(ABC):
    """Abstract base class for image generation services"""
    
    @abstractmethod
    async def generate_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate image from prompt"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if service is healthy"""
        pass


class LocalGPUService(ImageGenerationService):
    """Local GPU image generation service using existing ImageGenerator"""
    
    def __init__(self, model_path: str = "runwayml/stable-diffusion-v1-5"):
        self.image_generator = ImageGenerator(model_path)
        logger.info("Local GPU service initialized")
    
    async def generate_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate image using local GPU"""
        try:
            # Extract parameters
            output_dir = kwargs.get("output_dir", "data/gallery_images/reflection")
            filename = kwargs.get("filename", "generated_image.png")
            
            # Use existing image generator
            result = self.image_generator.generate_image(
                prompt=prompt,
                output_dir=output_dir,
                filename=filename
            )
            
            if result and result.get("success"):
                return {
                    "success": True,
                    "image_path": result["image_path"],
                    "prompt": prompt,
                    "generation_time": result.get("generation_time", 30.0),
                    "service": "local_gpu",
                    "model_version": "stable-diffusion-v1-5"
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                    "service": "local_gpu"
                }
                
        except Exception as e:
            logger.error(f"Local GPU generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "service": "local_gpu"
            }
    
    async def health_check(self) -> bool:
        """Check if local GPU service is healthy"""
        try:
            # Simple check - verify model is loaded
            return hasattr(self.image_generator, 'pipe') and self.image_generator.pipe is not None
        except Exception:
            return False



class ColabService(ImageGenerationService):
    """Google Colab image generation service"""
    
    def __init__(self, notebook_url: str, runtime_token: Optional[str] = None):
        self.notebook_url = notebook_url
        self.runtime_token = runtime_token
        logger.info(f"Colab service initialized: {notebook_url}")
    
    async def generate_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate image using Google Colab"""
        try:
            import httpx
            
            payload = {"prompt": prompt}
            logger.info(f"Sending to Colab: {self.notebook_url}/generate")
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.notebook_url}/generate",
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        # Base64 이미지를 파일로 저장
                        import base64
                        from PIL import Image
                        from io import BytesIO
                        
                        img_data = base64.b64decode(result["image"])
                        image = Image.open(BytesIO(img_data))
                        
                        # 파일명 생성
                        filename = kwargs.get("filename", "colab_generated.png")
                        output_dir = kwargs.get("output_dir", "data/gallery_images/reflection")
                        image_path = f"{output_dir}/{filename}"
                        
                        # 이미지 저장
                        image.save(image_path)
                        
                        return {
                            "success": True,
                            "image_path": image_path,
                            "prompt": prompt,
                            "generation_time": 30.0,
                            "service": "colab"
                        }
                
                logger.error(f"Colab request failed: {response.status_code}")
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "service": "colab"
                }
                    
        except Exception as e:
            logger.error(f"Colab generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "service": "colab"
            }
    
    async def health_check(self) -> bool:
        """Check if Colab service is healthy"""
        # 기본 구현
        return False


class ImageServiceFactory:
    """Factory for creating image generation services"""
    
    @staticmethod
    def create_service(service_type: str, **kwargs) -> ImageGenerationService:
        """Create image generation service based on type"""
        
        if service_type == "local":
            model_path = kwargs.get("model_path", "runwayml/stable-diffusion-v1-5")
            return LocalGPUService(model_path)
        
        elif service_type == "colab":
            notebook_url = kwargs.get("notebook_url")
            if not notebook_url:
                raise ValueError("Colab notebook URL is required")
            runtime_token = kwargs.get("runtime_token")
            return ColabService(notebook_url, runtime_token)
        
        else:
            raise ValueError(f"Unsupported service type: {service_type}")


# Global service instance
_image_service: Optional[ImageGenerationService] = None


def get_image_service() -> ImageGenerationService:
    """Get the global image generation service instance"""
    global _image_service
    
    if _image_service is None:
        # Import here to avoid circular imports
        from ..config import settings
        
        _image_service = ImageServiceFactory.create_service(
            service_type=settings.image_generation_service,
            notebook_url=settings.colab_notebook_url
        )
    
    return _image_service