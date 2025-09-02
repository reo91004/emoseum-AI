# api/services/emotion_service.py

import logging
import sys
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

# Add parent directory to path to import existing services
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.services.emotion_analyzer import get_emotion_analyzer

logger = logging.getLogger(__name__)


class EmotionAnalysisService(ABC):
    """Abstract base class for emotion analysis services"""
    
    @abstractmethod
    async def analyze_emotions(self, text: str, **kwargs) -> Dict[str, Any]:
        """Analyze emotions from text"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if service is healthy"""
        pass


class LocalGoEmotionsService(EmotionAnalysisService):
    """Local GoEmotions emotion analysis service"""
    
    def __init__(self):
        self.analyzer = get_emotion_analyzer("local_goEmotions")
        logger.info("Local GoEmotions service initialized")
    
    async def analyze_emotions(self, text: str, **kwargs) -> Dict[str, Any]:
        """Analyze emotions using local GoEmotions model"""
        try:
            threshold = kwargs.get("threshold", 0.3)
            
            result = self.analyzer.analyze_emotions(text, threshold)
            
            return {
                "success": True,
                "service": "local_goEmotions",
                **result
            }
                
        except Exception as e:
            logger.error(f"Local GoEmotions analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "service": "local_goEmotions"
            }
    
    async def health_check(self) -> bool:
        """Check if local GoEmotions service is healthy"""
        try:
            # Simple test analysis
            result = self.analyzer.analyze_emotions("test")
            return result.get("keywords") is not None
        except Exception:
            return False


class ColabGoEmotionsService(EmotionAnalysisService):
    """Colab GoEmotions emotion analysis service"""
    
    def __init__(self):
        self.analyzer = get_emotion_analyzer("colab_goEmotions")
        logger.info("Colab GoEmotions service initialized")
    
    async def analyze_emotions(self, text: str, **kwargs) -> Dict[str, Any]:
        """Analyze emotions using Colab GoEmotions server"""
        try:
            threshold = kwargs.get("threshold", 0.3)
            
            result = self.analyzer.analyze_emotions(text, threshold)
            
            if "error" in result:
                return {
                    "success": False,
                    "error": result["error"],
                    "service": "colab_goEmotions"
                }
            
            return {
                "success": True,
                "service": "colab_goEmotions",
                **result
            }
                
        except Exception as e:
            logger.error(f"Colab GoEmotions analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "service": "colab_goEmotions"
            }
    
    async def health_check(self) -> bool:
        """Check if Colab GoEmotions service is healthy"""
        try:
            # Simple test analysis
            result = self.analyzer.analyze_emotions("test")
            return "error" not in result
        except Exception:
            return False


class EmotionServiceFactory:
    """Factory for creating emotion analysis services"""
    
    @staticmethod
    def create_service(service_type: str, **kwargs) -> EmotionAnalysisService:
        """Create emotion analysis service based on type"""
        
        if service_type == "local_goEmotions":
            return LocalGoEmotionsService()
        
        elif service_type == "colab_goEmotions":
            return ColabGoEmotionsService()
        
        else:
            raise ValueError(f"Unsupported service type: {service_type}")


# Global service instance
_emotion_service: Optional[EmotionAnalysisService] = None


def get_emotion_service() -> EmotionAnalysisService:
    """Get the global emotion analysis service instance"""
    global _emotion_service
    
    if _emotion_service is None:
        # Import here to avoid circular imports
        from ..config import settings
        
        # 환경변수에서 감정 분석 서비스 타입 가져오기
        service_type = os.getenv("EMOTION_ANALYSIS_SERVICE", "local")
        
        if service_type == "local":
            _emotion_service = EmotionServiceFactory.create_service("local_goEmotions")
        elif service_type == "colab":
            _emotion_service = EmotionServiceFactory.create_service("colab_goEmotions")
        else:
            # 기본값으로 로컬 서비스 사용
            _emotion_service = EmotionServiceFactory.create_service("local_goEmotions")
    
    return _emotion_service