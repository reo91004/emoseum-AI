#!/usr/bin/env python3
"""
PersonalizedLoRAManager - 개인화된 LoRA 어댑터 관리 시스템
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import torch

from config import device, logger, PEFT_AVAILABLE

if PEFT_AVAILABLE:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel


class PersonalizedLoRAManager:
    """개인화된 LoRA 어댑터 관리 시스템"""

    def __init__(
        self,
        base_model_path: str = "runwayml/stable-diffusion-v1-5",
        lora_rank: int = 16,
        lora_dir: str = "data/user_loras",
    ):
        self.base_model_path = base_model_path
        self.lora_rank = lora_rank
        self.device = device
        self.lora_dir = Path(lora_dir)
        self.lora_dir.mkdir(parents=True, exist_ok=True)
        self.user_adapters = {}
        self.adapter_configs = {}

        if not PEFT_AVAILABLE:
            logger.warning("⚠️ PEFT 라이브러리가 없어 LoRA 기능이 제한됩니다")

    def create_user_lora_config(self, user_id: str) -> Optional[LoraConfig]:
        """사용자별 LoRA 설정 생성"""
        if not PEFT_AVAILABLE:
            return None

        try:
            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=32,
                target_modules=[
                    "to_k",
                    "to_q",
                    "to_v",
                    "to_out.0",
                    "proj_in",
                    "proj_out",
                    "ff.net.0.proj",
                    "ff.net.2",
                ],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.DIFFUSION,
            )

            self.adapter_configs[user_id] = lora_config
            logger.info(f"✅ 사용자 {user_id}의 LoRA 설정 생성 완료")
            return lora_config

        except Exception as e:
            logger.error(f"❌ LoRA 설정 생성 실패: {e}")
            return None

    def save_user_lora(self, user_id: str, model_state_dict: Dict[str, torch.Tensor]):
        """사용자 LoRA 어댑터 저장"""
        try:
            user_lora_path = self.lora_dir / f"{user_id}_lora.pt"
            torch.save(model_state_dict, user_lora_path)
            logger.info(f"✅ 사용자 {user_id} LoRA 저장: {user_lora_path}")
        except Exception as e:
            logger.error(f"❌ LoRA 저장 실패: {e}")

    def load_user_lora(self, user_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """사용자 LoRA 어댑터 로드"""
        try:
            user_lora_path = self.lora_dir / f"{user_id}_lora.pt"
            if user_lora_path.exists():
                state_dict = torch.load(user_lora_path, map_location=self.device)
                logger.info(f"✅ 사용자 {user_id} LoRA 로드: {user_lora_path}")
                return state_dict
        except Exception as e:
            logger.error(f"❌ LoRA 로드 실패: {e}")
        return None

    def get_user_adapter_info(self, user_id: str) -> Dict[str, Any]:
        """사용자 어댑터 정보 반환"""
        return {
            "user_id": user_id,
            "lora_rank": self.lora_rank,
            "config": self.adapter_configs.get(user_id),
            "path": self.lora_dir / f"{user_id}_lora.pt",
        }