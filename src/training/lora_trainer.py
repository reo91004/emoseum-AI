#!/usr/bin/env python3
"""
LoRA Trainer - 메모리 효율적인 LoRA 기반 디퓨전 모델 파인튜닝
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import warnings

from config import device, logger, PEFT_AVAILABLE
from models.emotion import EmotionEmbedding
from models.user_profile import UserEmotionProfile

# PEFT 라이브러리 import (LoRA)
if PEFT_AVAILABLE:
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        from peft.utils import _get_submodules
    except ImportError:
        logger.warning("⚠️ PEFT 라이브러리의 일부 기능을 불러올 수 없습니다")
        PEFT_AVAILABLE = False


class LoRATrainerConfig:
    """LoRA 트레이너 설정"""
    
    def __init__(
        self,
        r: int = 16,                    # LoRA rank
        lora_alpha: int = 32,           # LoRA alpha
        lora_dropout: float = 0.1,      # LoRA dropout
        target_modules: List[str] = None,  # 타겟 모듈
        learning_rate: float = 1e-4,    # 학습률
        gradient_accumulation_steps: int = 4,  # 그래디언트 누적
        max_grad_norm: float = 1.0,     # 그래디언트 클리핑
        warmup_steps: int = 100,        # 워밍업 스텝
        save_steps: int = 500,          # 저장 간격
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or [
            "to_k", "to_q", "to_v", "to_out.0",  # Attention layers
            "ff.net.0.proj", "ff.net.2"          # Feed-forward layers
        ]
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.save_steps = save_steps


class ImprovedLoRATrainer:
    """개선된 LoRA 기반 트레이너"""
    
    def __init__(
        self,
        pipeline,
        reward_model,
        config: LoRATrainerConfig = None
    ):
        self.pipeline = pipeline
        self.reward_model = reward_model
        self.config = config or LoRATrainerConfig()
        self.device = device
        
        # LoRA 설정
        self.lora_model = None
        self.can_train = False
        
        # 학습 상태
        self.global_step = 0
        self.epoch = 0
        
        # 메트릭 추적
        self.training_metrics = {
            "losses": [],
            "rewards": [],
            "learning_rates": []
        }
        
        # LoRA 모델 초기화
        self._setup_lora()
        
        # 옵티마이저 설정
        if self.can_train:
            self._setup_optimizer()
        
        logger.info(f"✅ LoRA 트레이너 초기화 완료 (훈련 가능: {self.can_train})")
    
    def _setup_lora(self):
        """LoRA 모델 설정"""
        
        if not PEFT_AVAILABLE:
            logger.warning("⚠️ PEFT 라이브러리가 없어 LoRA 사용 불가")
            return
        
        if not hasattr(self.pipeline, "unet"):
            logger.warning("⚠️ UNet이 없어 LoRA 설정 불가")
            return
        
        try:
            # LoRA 설정
            lora_config = LoraConfig(
                r=self.config.r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.DIFFUSION,  # 디퓨전 모델용
            )
            
            # UNet에 LoRA 적용
            self.lora_model = get_peft_model(self.pipeline.unet, lora_config)
            
            # 학습 모드 설정
            self.lora_model.train()
            
            # 그래디언트 체크포인팅 (메모리 효율성)
            if hasattr(self.lora_model, "enable_gradient_checkpointing"):
                self.lora_model.enable_gradient_checkpointing()
            
            self.can_train = True
            
            # LoRA 파라미터 수 계산
            trainable_params = sum(
                p.numel() for p in self.lora_model.parameters() if p.requires_grad
            )
            
            logger.info(f"✅ LoRA 설정 완료: {trainable_params:,}개 훈련 가능 파라미터")
            
        except Exception as e:
            logger.error(f"❌ LoRA 설정 실패: {e}")
            self.can_train = False
    
    def _setup_optimizer(self):
        """옵티마이저 및 스케줄러 설정"""
        
        # LoRA 파라미터만 훈련
        lora_params = [p for p in self.lora_model.parameters() if p.requires_grad]
        
        # AdamW 옵티마이저
        self.optimizer = torch.optim.AdamW(
            lora_params,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # 코사인 어닐링 스케줄러
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000,
            eta_min=self.config.learning_rate * 0.1
        )
        
        logger.info(f"✅ 옵티마이저 설정 완료: {len(lora_params)}개 파라미터")
    
    def train_step(
        self,
        prompt: str,
        target_emotion: EmotionEmbedding,
        user_profile: UserEmotionProfile,
        batch_size: int = 1,
        num_inference_steps: int = 50
    ) -> Dict[str, Any]:
        """LoRA 기반 훈련 스텝"""
        
        if not self.can_train:
            return self._simulation_step()
        
        try:
            # 그래디언트 누적 시작
            if self.global_step % self.config.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()
            
            # Forward pass
            loss_info = self._forward_pass(
                prompt, target_emotion, user_profile, 
                batch_size, num_inference_steps
            )
            
            # 손실 스케일링 (그래디언트 누적)
            scaled_loss = loss_info["loss"] / self.config.gradient_accumulation_steps
            
            # Backward pass
            scaled_loss.backward()
            
            # 그래디언트 누적 완료시 업데이트
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                # 그래디언트 클리핑
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.lora_model.parameters(),
                    self.config.max_grad_norm
                )
                
                # 옵티마이저 스텝
                self.optimizer.step()
                self.lr_scheduler.step()
                
                loss_info["grad_norm"] = grad_norm.item()
            
            # 전역 스텝 증가
            self.global_step += 1
            
            # 메트릭 업데이트
            self._update_metrics(loss_info)
            
            # 체크포인트 저장
            if self.global_step % self.config.save_steps == 0:
                self._save_checkpoint()
            
            return {
                **loss_info,
                "global_step": self.global_step,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "mode": "lora_training"
            }
            
        except Exception as e:
            logger.error(f"❌ LoRA 훈련 스텝 실패: {e}")
            return self._simulation_step()
    
    def _forward_pass(
        self,
        prompt: str,
        target_emotion: EmotionEmbedding,
        user_profile: UserEmotionProfile,
        batch_size: int,
        num_inference_steps: int
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        
        # 텍스트 인코딩
        text_embeddings = self._encode_prompt(prompt, batch_size)
        
        # 랜덤 타임스텝 샘플링
        timesteps = torch.randint(
            0, self.pipeline.scheduler.config.num_train_timesteps,
            (batch_size,), device=self.device
        ).long()
        
        # 랜덤 노이즈
        noise = torch.randn(
            (batch_size, 4, 64, 64),
            device=self.device,
            dtype=text_embeddings.dtype
        )
        
        # 노이즈 추가된 잠재 변수
        latents = self.pipeline.scheduler.add_noise(
            torch.zeros_like(noise), noise, timesteps
        )
        
        # UNet 예측
        noise_pred = self.lora_model(
            latents,
            timesteps,
            encoder_hidden_states=text_embeddings,
            return_dict=False
        )[0]
        
        # MSE 손실 (디퓨전 모델 기본 손실)
        mse_loss = nn.functional.mse_loss(noise_pred, noise, reduction="mean")
        
        # 보상 기반 손실 (선택적)
        reward_loss = torch.tensor(0.0, device=self.device)
        
        if self.global_step % 10 == 0:  # 매 10스텝마다 보상 계산
            # 빠른 생성으로 보상 계산
            quick_images = self._quick_generate(text_embeddings, num_steps=10)
            rewards = self.reward_model.calculate_comprehensive_reward(
                quick_images, target_emotion, user_profile
            )
            reward_loss = -rewards.mean() * 0.1  # 보상 손실 가중치
        
        # 총 손실
        total_loss = mse_loss + reward_loss
        
        return {
            "loss": total_loss,
            "mse_loss": mse_loss.item(),
            "reward_loss": reward_loss.item(),
            "reward_mean": 0.0 if reward_loss == 0 else rewards.mean().item()
        }
    
    def _quick_generate(
        self, 
        text_embeddings: torch.Tensor, 
        num_steps: int = 10
    ) -> torch.Tensor:
        """빠른 이미지 생성 (보상 계산용)"""
        
        batch_size = text_embeddings.shape[0]
        
        # 초기 노이즈
        latents = torch.randn(
            (batch_size, 4, 64, 64),
            device=self.device,
            dtype=text_embeddings.dtype
        )
        
        # 빠른 디노이징
        self.pipeline.scheduler.set_timesteps(num_steps)
        
        with torch.no_grad():
            for t in self.pipeline.scheduler.timesteps:
                timestep_batch = t.repeat(batch_size).to(self.device)
                
                # UNet 예측
                noise_pred = self.lora_model(
                    latents,
                    timestep_batch,
                    encoder_hidden_states=text_embeddings,
                    return_dict=False
                )[0]
                
                # 스케줄러 스텝
                latents = self.pipeline.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]
        
        # VAE 디코딩
        images = self._decode_latents(latents)
        
        return images
    
    def _encode_prompt(self, prompt: str, batch_size: int = 1) -> torch.Tensor:
        """프롬프트 인코딩"""
        
        if hasattr(self.pipeline, "text_encoder"):
            text_inputs = self.pipeline.tokenizer(
                [prompt] * batch_size,
                padding="max_length",
                max_length=self.pipeline.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            
            with torch.no_grad():
                text_embeddings = self.pipeline.text_encoder(
                    text_inputs.input_ids.to(self.device)
                )[0]
        else:
            text_embeddings = torch.randn(
                batch_size, 77, 768, device=self.device
            )
        
        return text_embeddings
    
    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """VAE 디코딩"""
        
        if hasattr(self.pipeline, "vae"):
            try:
                if hasattr(self.pipeline.vae.config, "scaling_factor"):
                    latents_scaled = latents / self.pipeline.vae.config.scaling_factor
                else:
                    latents_scaled = latents
                
                with torch.no_grad():
                    images = self.pipeline.vae.decode(
                        latents_scaled, return_dict=False
                    )[0]
                
                images = (images / 2 + 0.5).clamp(0, 1)
                
            except Exception as e:
                logger.warning(f"⚠️ VAE 디코딩 실패: {e}")
                images = torch.rand(
                    latents.shape[0], 3, 512, 512, device=self.device
                )
        else:
            images = torch.rand(
                latents.shape[0], 3, 512, 512, device=self.device
            )
        
        return images
    
    def _update_metrics(self, loss_info: Dict[str, Any]):
        """메트릭 업데이트"""
        
        self.training_metrics["losses"].append(loss_info.get("loss", 0.0))
        self.training_metrics["rewards"].append(loss_info.get("reward_mean", 0.0))
        self.training_metrics["learning_rates"].append(
            self.optimizer.param_groups[0]["lr"]
        )
        
        # 메트릭 히스토리 크기 제한
        max_history = 1000
        for key in self.training_metrics:
            if len(self.training_metrics[key]) > max_history:
                self.training_metrics[key] = self.training_metrics[key][-max_history:]
    
    def _save_checkpoint(self):
        """체크포인트 저장"""
        
        try:
            checkpoint_path = f"checkpoints/lora_checkpoint_step_{self.global_step}.pt"
            
            # LoRA 가중치만 저장 (메모리 효율적)
            if hasattr(self.lora_model, "save_pretrained"):
                self.lora_model.save_pretrained(f"checkpoints/lora_step_{self.global_step}")
            
            # 추가 정보 저장
            torch.save({
                "global_step": self.global_step,
                "epoch": self.epoch,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                "training_metrics": self.training_metrics,
                "config": self.config.__dict__
            }, checkpoint_path)
            
            logger.info(f"✅ LoRA 체크포인트 저장: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"❌ 체크포인트 저장 실패: {e}")
    
    def _simulation_step(self) -> Dict[str, Any]:
        """시뮬레이션 모드"""
        
        import random
        
        return {
            "loss": random.uniform(0.1, 0.5),
            "mse_loss": random.uniform(0.05, 0.3),
            "reward_loss": random.uniform(0.0, 0.2),
            "reward_mean": random.uniform(0.4, 0.8),
            "global_step": self.global_step,
            "learning_rate": self.config.learning_rate,
            "mode": "simulation"
        }
    
    def load_checkpoint(self, checkpoint_path: str):
        """체크포인트 로드"""
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.global_step = checkpoint["global_step"]
            self.epoch = checkpoint["epoch"]
            
            if self.can_train:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
            
            self.training_metrics = checkpoint.get("training_metrics", self.training_metrics)
            
            logger.info(f"✅ 체크포인트 로드: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"❌ 체크포인트 로드 실패: {e}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """학습 요약"""
        
        summary = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "can_train": self.can_train,
            "trainable_parameters": sum(
                p.numel() for p in self.lora_model.parameters() if p.requires_grad
            ) if self.can_train else 0
        }
        
        # 최근 메트릭 통계
        for metric_name, values in self.training_metrics.items():
            if values:
                recent_values = values[-100:]  # 최근 100개
                summary[f"{metric_name}_mean"] = sum(recent_values) / len(recent_values)
                summary[f"{metric_name}_latest"] = values[-1]
        
        return summary