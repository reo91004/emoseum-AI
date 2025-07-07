#!/usr/bin/env python3
"""
Emoseum 설정 및 상수 정의
"""

import torch
import logging
import sys

# 디바이스 설정
if torch.backends.mps.is_available():
    device_type = "mps"
    torch.mps.set_per_process_memory_fraction(0.8)
elif torch.cuda.is_available():
    device_type = "cuda"
    torch.backends.cudnn.benchmark = True
else:
    device_type = "cpu"

device = torch.device(device_type)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("emotion_therapy.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# 라이브러리 가용성 확인
try:
    from transformers import (
        CLIPTextModel,
        CLIPTokenizer,
        AutoTokenizer,
        AutoModel,
        pipeline,
        RobertaTokenizer,
        RobertaModel,
    )
    TRANSFORMERS_AVAILABLE = True
    logger.info("✅ Transformers 라이브러리 로드 완료")
except ImportError:
    logger.error("❌ transformers 라이브러리가 필요합니다: pip install transformers")
    TRANSFORMERS_AVAILABLE = False

try:
    from diffusers import (
        StableDiffusionPipeline,
        UNet2DConditionModel,
        DDPMScheduler,
        AutoencoderKL,
        DPMSolverMultistepScheduler,
        EulerDiscreteScheduler,
    )
    DIFFUSERS_AVAILABLE = True
    logger.info("✅ Diffusers 라이브러리 로드 완료")
except ImportError:
    logger.error("❌ diffusers 라이브러리가 필요합니다: pip install diffusers")
    DIFFUSERS_AVAILABLE = False

try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    PEFT_AVAILABLE = True
    logger.info("✅ PEFT 라이브러리 로드 완료")
except ImportError:
    logger.error("❌ peft 라이브러리가 필요합니다: pip install peft")
    PEFT_AVAILABLE = False

# 기본 설정 값들
DEFAULT_DIFFUSION_MODEL = "runwayml/stable-diffusion-v1-5"
DEFAULT_STEPS = 15
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_SEED = None

# 데이터베이스 설정
DATABASE_NAME = "user_profiles.db"
GENERATED_IMAGES_DIR = "generated_images"
USER_LORAS_DIR = "user_loras"