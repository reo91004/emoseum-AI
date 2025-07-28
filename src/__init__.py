# src/__init__.py

# Core 시스템
from .core.act_therapy_system import ACTTherapySystem

# Managers
from .managers.user_manager import (
    UserManager,
    User,
    PsychometricResult,
    VisualPreferences,
)
from .managers.gallery_manager import GalleryManager, GalleryItem
from .managers.personalization_manager import PersonalizationManager

# Services
from .services.image_generator import ImageGenerator

# Therapy 모듈들
from .therapy.prompt_architect import PromptArchitect
from .therapy.curator_message import CuratorMessageSystem

# AI 모듈들

# Training 모듈들
from .training.lora_trainer import PersonalizedLoRATrainer
from .training.draft_trainer import DRaFTPlusTrainer

__version__ = "2.0.0"  # GPT 통합 버전
__all__ = [
    # Core
    "ACTTherapySystem",
    # Managers
    "UserManager",
    "User",
    "PsychometricResult",
    "VisualPreferences",
    "GalleryManager",
    "GalleryItem",
    "PersonalizationManager",
    # Services
    "ImageGenerator",
    # Therapy
    "PromptArchitect",
    "CuratorMessageSystem",
    # AI
    # Training
    "PersonalizedLoRATrainer",
    "DRaFTPlusTrainer",
]
