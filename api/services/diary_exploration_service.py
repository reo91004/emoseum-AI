# api/services/diary_exploration_service.py

import logging
from typing import Dict, List, Any, Optional
import sys
import os

# Add parent directory to path to import existing services
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.services.diary_exploration_service import DiaryExplorationService

logger = logging.getLogger(__name__)


class APIDiaryExplorationService:
    """API용 일기 심화 탐색 서비스 래퍼"""
    
    def __init__(self):
        self.core_service = DiaryExplorationService()
        logger.info("API 일기 심화 탐색 서비스 초기화 완료")
    
    async def generate_exploration_questions(
        self, 
        diary_text: str, 
        emotion_keywords: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        일기 심화 탐색 질문 생성 (비동기 래퍼)
        
        Args:
            diary_text: 사용자의 일기 내용
            emotion_keywords: 감정 키워드 (선택사항)
            
        Returns:
            생성된 질문들과 메타데이터
        """
        try:
            # 감정 키워드가 없으면 기본값 설정
            if emotion_keywords is None:
                emotion_keywords = []
            
            # 동기 서비스 호출
            result = self.core_service.generate_exploration_questions(
                diary_text=diary_text,
                emotion_keywords=emotion_keywords
            )
            
            # API 응답 형식에 맞게 메타데이터 추가
            result.update({
                "service": "diary_exploration",
                "api_version": "1.0.0",
                "request_processed": True
            })
            
            return result
            
        except Exception as e:
            logger.error(f"API 일기 심화 탐색 질문 생성 실패: {e}")
            return {
                "success": False,
                "questions": [],
                "exploration_theme": "서비스 오류",
                "encouragement": "잠시 후 다시 시도해주세요.",
                "error": str(e),
                "service": "diary_exploration",
                "api_version": "1.0.0",
                "request_processed": False
            }
    
    async def get_question_categories(self) -> Dict[str, Any]:
        """질문 카테고리 정보 반환"""
        try:
            categories = self.core_service.get_question_categories_info()
            return {
                "success": True,
                "categories": categories,
                "service": "diary_exploration",
                "api_version": "1.0.0"
            }
        except Exception as e:
            logger.error(f"질문 카테고리 조회 실패: {e}")
            return {
                "success": False,
                "categories": {},
                "error": str(e),
                "service": "diary_exploration",
                "api_version": "1.0.0"
            }
    
    async def get_safety_guidelines(self) -> Dict[str, Any]:
        """안전 가이드라인 반환"""
        try:
            guidelines = self.core_service.get_safety_guidelines()
            return {
                "success": True,
                "safety_guidelines": guidelines,
                "service": "diary_exploration",
                "api_version": "1.0.0"
            }
        except Exception as e:
            logger.error(f"안전 가이드라인 조회 실패: {e}")
            return {
                "success": False,
                "safety_guidelines": [],
                "error": str(e),
                "service": "diary_exploration",
                "api_version": "1.0.0"
            }
    
    async def generate_follow_up_question(
        self,
        diary_text: str,
        previous_question: str,
        user_response: str,
        emotion_keywords: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        후속 질문 생성 (비동기 래퍼)
        
        Args:
            diary_text: 원본 일기 내용
            previous_question: 이전 질문
            user_response: 사용자의 답변
            emotion_keywords: 감정 키워드 (선택사항)
            
        Returns:
            생성된 후속 질문과 메타데이터
        """
        try:
            # 감정 키워드가 없으면 기본값 설정
            if emotion_keywords is None:
                emotion_keywords = []
            
            # 동기 서비스 호출
            result = self.core_service.generate_follow_up_question(
                diary_text=diary_text,
                previous_question=previous_question,
                user_response=user_response,
                emotion_keywords=emotion_keywords
            )
            
            # API 응답 형식에 맞게 메타데이터 추가
            result.update({
                "service": "diary_exploration_follow_up",
                "api_version": "1.0.0",
                "request_processed": True
            })
            
            return result
            
        except Exception as e:
            logger.error(f"API 후속 질문 생성 실패: {e}")
            return {
                "success": False,
                "questions": [],
                "exploration_theme": "Follow-up Error",
                "encouragement": "Thank you for sharing. Please try again in a moment.",
                "error": str(e),
                "service": "diary_exploration_follow_up",
                "api_version": "1.0.0",
                "request_processed": False,
                "is_follow_up": True
            }

    async def health_check(self) -> bool:
        """서비스 상태 확인"""
        try:
            # 간단한 상태 확인
            return hasattr(self.core_service, 'generate_exploration_questions')
        except Exception:
            return False


# 전역 서비스 인스턴스
_api_diary_exploration_service: Optional[APIDiaryExplorationService] = None


def get_api_diary_exploration_service() -> APIDiaryExplorationService:
    """전역 API 일기 심화 탐색 서비스 인스턴스 반환"""
    global _api_diary_exploration_service
    
    if _api_diary_exploration_service is None:
        _api_diary_exploration_service = APIDiaryExplorationService()
    
    return _api_diary_exploration_service