# src/services/image_service_wrapper.py

# ==============================================================================
# CLI에서 사용할 이미지 생성 서비스 래퍼 클래스들
# 기존 ImageGenerator 인터페이스와 호환되도록 설계된 래퍼들
# ==============================================================================

import logging
import requests
import base64
from PIL import Image
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional
import os

logger = logging.getLogger(__name__)


class ColabImageGenerator:
    """Colab 노트북을 사용한 이미지 생성 래퍼"""
    
    def __init__(self, notebook_url: str):
        self.notebook_url = notebook_url.rstrip('/')
        self.device = "colab"  # ImageGenerator와 호환성을 위해
        
    def generate_image(
        self, 
        prompt: str, 
        output_dir: str = "data/gallery_images/reflection",
        filename: str = "generated_image.png",
        width: int = 512,
        height: int = 512,
        **kwargs
    ) -> Dict[str, Any]:
        """Colab을 통해 이미지 생성"""
        
        try:
            logger.info(f"Colab을 통해 이미지 생성 중: {prompt[:50]}...")
            
            # Colab 서버에 요청 (1:1 비율 강제)
            payload = {
                "prompt": prompt,
                "width": width,
                "height": height
            }
            response = requests.post(
                f"{self.notebook_url}/generate",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    # Base64 이미지를 파일로 저장
                    img_data = base64.b64decode(result["image"])
                    image = Image.open(BytesIO(img_data))
                    
                    # 출력 디렉토리 생성
                    os.makedirs(output_dir, exist_ok=True)
                    image_path = os.path.join(output_dir, filename)
                    
                    # 이미지 저장
                    image.save(image_path)
                    
                    return {
                        "success": True,
                        "image_path": image_path,
                        "prompt": prompt,
                        "generation_time": result.get("generation_time", 30.0),
                        "service": "colab"
                    }
            
            logger.error(f"Colab 요청 실패: {response.status_code}")
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}",
                "service": "colab"
            }
            
        except Exception as e:
            logger.error(f"Colab 이미지 생성 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "service": "colab"
            }

    def cleanup(self):
        """정리 작업 (ImageGenerator 호환성)"""
        pass

