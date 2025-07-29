# api/routers/__init__.py

"""
API Routers Package

이 패키지는 FastAPI 라우터들을 정의한다.
- auth.py: 인증 관련 엔드포인트 (회원가입, 로그인, JWT)
- users.py: 사용자 프로필 관리 엔드포인트
- assessment.py: 심리검사 관련 엔드포인트 (PHQ-9, CES-D, MEAQ, CISS)
- therapy.py: ACT 4단계 치료 여정 엔드포인트
- gallery.py: 디지털 갤러리 관리 엔드포인트
- admin.py: 시스템 관리 및 모니터링 엔드포인트
"""

from .auth import router as auth_router
from .users import router as users_router
from .assessment import router as assessment_router
from .therapy import router as therapy_router
from .gallery import router as gallery_router
from .admin import router as admin_router

__all__ = [
    "auth_router",
    "users_router",
    "assessment_router",
    "therapy_router",
    "gallery_router",
    "admin_router",
]
