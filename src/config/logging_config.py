#!/usr/bin/env python3
"""
로깅 설정
"""

import os
import logging
import sys
from pathlib import Path

# 로그 디렉토리 생성
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# 로그 레벨 설정
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# 로깅 설정
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "therapy.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("emoseum")

# 라이브러리 가용성 로깅
def log_library_status():
    """라이브러리 상태 로깅"""
    from .settings import TRANSFORMERS_AVAILABLE, DIFFUSERS_AVAILABLE, PEFT_AVAILABLE
    
    if TRANSFORMERS_AVAILABLE:
        logger.info("✅ Transformers 라이브러리 로드 완료")
    else:
        logger.error("❌ transformers 라이브러리가 필요합니다: pip install transformers")
    
    if DIFFUSERS_AVAILABLE:
        logger.info("✅ Diffusers 라이브러리 로드 완료")
    else:
        logger.error("❌ diffusers 라이브러리가 필요합니다: pip install diffusers")
    
    if PEFT_AVAILABLE:
        logger.info("✅ PEFT 라이브러리 로드 완료")
    else:
        logger.error("❌ peft 라이브러리가 필요합니다: pip install peft")

# 초기화 시 라이브러리 상태 로깅
log_library_status()