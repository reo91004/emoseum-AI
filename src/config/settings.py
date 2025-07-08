#!/usr/bin/env python3
"""
Emoseum 설정 및 상수 정의
"""

import os
import torch

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

# 환경변수에서 설정 읽기
DEFAULT_DIFFUSION_MODEL = os.getenv("DIFFUSION_MODEL", "runwayml/stable-diffusion-v1-5")
DEFAULT_STEPS = int(os.getenv("DIFFUSION_STEPS", "15"))
DEFAULT_GUIDANCE_SCALE = float(os.getenv("GUIDANCE_SCALE", "7.5"))
DEFAULT_WIDTH = int(os.getenv("IMAGE_WIDTH", "512"))
DEFAULT_HEIGHT = int(os.getenv("IMAGE_HEIGHT", "512"))
DEFAULT_SEED = None

# 디렉토리 설정
DATABASE_NAME = os.getenv("DATABASE_NAME", "data/user_profiles.db")
GENERATED_IMAGES_DIR = os.getenv("GENERATED_IMAGES_DIR", "data/generated_images")
USER_LORAS_DIR = os.getenv("USER_LORAS_DIR", "data/user_loras")

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
except ImportError:
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
except ImportError:
    DIFFUSERS_AVAILABLE = False

try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False