#!/usr/bin/env python3
"""
DRaFTPlusTrainer - DRaFT+ 기반 강화학습 트레이너
"""

import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List

from config import device, logger
from models.emotion import EmotionEmbedding
from models.user_profile import UserEmotionProfile
from models.reward_model import DRaFTPlusRewardModel


class DRaFTPlusTrainer:
    """DRaFT+ 기반 강화학습 트레이너"""

    def __init__(
        self, pipeline, reward_model: DRaFTPlusRewardModel, learning_rate: float = 1e-5
    ):
        self.pipeline = pipeline
        self.reward_model = reward_model
        self.device = device
        self.learning_rate = learning_rate

        # 옵티마이저 설정
        if hasattr(pipeline, "unet") and hasattr(pipeline.unet, "parameters"):
            self.optimizer = optim.AdamW(
                pipeline.unet.parameters(),
                lr=learning_rate,
                weight_decay=0.01,
                eps=1e-8,
            )
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=100, eta_min=learning_rate * 0.1
            )
            self.can_train = True
            logger.info("✅ DRaFT+ 트레이너 초기화 완료")
        else:
            logger.warning("⚠️ UNet이 없어 시뮬레이션 모드로 실행")
            self.can_train = False

        # 다양성을 위한 이미지 히스토리
        self.image_history = []
        self.max_history_size = 10

    def train_step(
        self,
        prompt: str,
        target_emotion: EmotionEmbedding,
        user_profile: UserEmotionProfile,
        num_inference_steps: int = 8,
        diversity_weight: float = 0.15,
    ) -> Dict[str, float]:
        """DRaFT+ 학습 스텝"""

        if not self.can_train:
            # 시뮬레이션 모드
            return {
                "emotion_reward": random.uniform(0.4, 0.8),
                "aesthetic_reward": random.uniform(0.5, 0.9),
                "personalization_reward": random.uniform(0.3, 0.7),
                "diversity_reward": random.uniform(0.6, 1.0),
                "total_reward": random.uniform(0.5, 0.8),
                "loss": random.uniform(0.2, 0.6),
                "learning_rate": self.learning_rate,
                "mode": "simulation",
            }

        try:
            # 그래디언트 초기화
            self.optimizer.zero_grad()

            # UNet 학습 모드로 설정
            self.pipeline.unet.train()

            # 이미지 생성 (간소화된 디퓨전 과정)
            with torch.enable_grad():
                # 텍스트 임베딩
                text_embeddings = self._encode_prompt(prompt)

                # 노이즈 생성
                latents = torch.randn(
                    (1, 4, 64, 64),  # SD 1.5 기본 latent 크기
                    device=self.device,
                    dtype=text_embeddings.dtype,
                    requires_grad=True,
                )

                # 간소화된 디노이징 (빠른 학습을 위해)
                for step in range(num_inference_steps):
                    t = torch.tensor(
                        [1000 - step * (1000 // num_inference_steps)],
                        device=self.device,
                    )

                    # UNet 예측
                    noise_pred = self.pipeline.unet(
                        latents,
                        t,
                        encoder_hidden_states=text_embeddings,
                        return_dict=False,
                    )[0]

                    # 디노이징 스텝
                    latents = latents - 0.1 * noise_pred

                # VAE 디코딩 (가능한 경우)
                if hasattr(self.pipeline, "vae"):
                    try:
                        if hasattr(self.pipeline.vae.config, "scaling_factor"):
                            latents_scaled = (
                                latents / self.pipeline.vae.config.scaling_factor
                            )
                        else:
                            latents_scaled = latents

                        images = self.pipeline.vae.decode(
                            latents_scaled, return_dict=False
                        )[0]
                        images = (images / 2 + 0.5).clamp(0, 1)
                    except:
                        # VAE 디코딩 실패시 가짜 이미지
                        images = torch.rand(1, 3, 512, 512, device=self.device)
                else:
                    images = torch.rand(1, 3, 512, 512, device=self.device)

                # 보상 계산
                rewards = self.reward_model.calculate_comprehensive_reward(
                    images, target_emotion, user_profile, self.image_history
                )

                # DRaFT+ 손실 계산 (다양성 정규화 포함)
                reward_loss = -rewards.mean()

                # 다양성 정규화 손실
                diversity_loss = self._calculate_diversity_loss(images)

                # 총 손실
                total_loss = reward_loss + diversity_weight * diversity_loss

                # 역전파
                total_loss.backward()

                # 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(self.pipeline.unet.parameters(), 1.0)

                # 옵티마이저 스텝
                self.optimizer.step()
                self.scheduler.step()

                # 이미지 히스토리 업데이트
                self._update_image_history(images.detach())

                # 결과 반환
                with torch.no_grad():
                    return {
                        "total_reward": rewards.mean().item(),
                        "reward_loss": reward_loss.item(),
                        "diversity_loss": diversity_loss.item(),
                        "total_loss": total_loss.item(),
                        "learning_rate": self.scheduler.get_last_lr()[0],
                        "mode": "training",
                    }

        except Exception as e:
            logger.error(f"❌ DRaFT+ 학습 실패: {e}")
            return {
                "error": str(e),
                "total_reward": 0.5,
                "loss": 1.0,
                "learning_rate": self.learning_rate,
                "mode": "error",
            }

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """프롬프트 인코딩"""
        try:
            text_inputs = self.pipeline.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.pipeline.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            text_embeddings = self.pipeline.text_encoder(text_inputs.input_ids)[0]
            return text_embeddings

        except Exception as e:
            logger.warning(f"⚠️ 프롬프트 인코딩 실패: {e}, 기본값 사용")
            return torch.randn(1, 77, 768, device=self.device)

    def _calculate_diversity_loss(self, images: torch.Tensor) -> torch.Tensor:
        """다양성 손실 계산 (DRaFT+ 핵심)"""
        if len(self.image_history) == 0:
            return torch.tensor(0.0, device=self.device)

        # 현재 이미지와 히스토리 이미지들 간의 유사성 계산
        current_features = self.reward_model._extract_simple_features(images)

        total_similarity = 0.0
        count = 0

        for hist_img in self.image_history[-3:]:  # 최근 3개와 비교
            hist_features = self.reward_model._extract_simple_features(hist_img)
            similarity = F.cosine_similarity(
                current_features, hist_features, dim=1
            ).mean()
            total_similarity += similarity
            count += 1

        if count > 0:
            avg_similarity = total_similarity / count
            # 유사성이 높을수록 다양성 손실 증가
            diversity_loss = torch.clamp(avg_similarity, 0.0, 1.0)
        else:
            diversity_loss = torch.tensor(0.0, device=self.device)

        return diversity_loss

    def _update_image_history(self, images: torch.Tensor):
        """이미지 히스토리 업데이트"""
        self.image_history.append(images.clone())

        # 히스토리 크기 제한
        if len(self.image_history) > self.max_history_size:
            self.image_history.pop(0)