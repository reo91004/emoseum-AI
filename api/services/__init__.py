# api/services/__init__.py

"""
API Services Package

이 패키지는 외부 서비스와의 연동을 담당하는 서비스 계층이다.
- database.py: Supabase 데이터베이스 연동
- image_service.py: 멀티 백엔드 이미지 생성 서비스
- therapy_service.py: 기존 ACTTherapySystem을 API용으로 래핑
"""

from .database import db, SupabaseService
from .image_service import ImageService
from .therapy_service import TherapyService

__all__ = ["db", "SupabaseService", "ImageService", "TherapyService"]
