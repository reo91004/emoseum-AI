# training/draft_trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import random

logger = logging.getLogger(__name__)

try:
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler

    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logger.warning("Diffusers 라이브러리를 사용할 수 없습니다.")


class DRaFTRewardModel:
    """DRaFT+ 보상 모델"""

    def __init__(self, device: torch.device):
        self.device = device

        # 간단한 보상 계산 네트워크
        self.reward_net = nn.Sequential(
            nn.Linear(1024, 512),  # 이미지 특성 입력
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        ).to(device)

        # 보상 계산 가중치
        self.reward_weights = {
            "sentiment_score": 0.6,  # 방명록 감정 점수
            "visual_quality": 0.2,  # 시각적 품질
            "user_preference": 0.2,  # 사용자 선호도 일치
        }

    def calculate_reward(
        self,
        image_features: torch.Tensor,
        sentiment_score: float,
        user_preferences: Dict[str, float],
    ) -> torch.Tensor:
        """종합 보상 계산"""

        # 1. 감정 점수 기반 보상 (1-5 -> 0-1)
        sentiment_reward = (sentiment_score - 1) / 4

        # 2. 시각적 품질 보상 (신경망으로 추정)
        visual_reward = self.reward_net(image_features).squeeze()

        # 3. 사용자 선호도 일치 보상 (간단한 휴리스틱)
        preference_reward = self._calculate_preference_reward(user_preferences)

        # 가중 평균
        total_reward = (
            self.reward_weights["sentiment_score"] * sentiment_reward
            + self.reward_weights["visual_quality"] * visual_reward
            + self.reward_weights["user_preference"] * preference_reward
        )

        return torch.clamp(total_reward, 0.0, 1.0)

    def _calculate_preference_reward(self, preferences: Dict[str, float]) -> float:
        """사용자 선호도 일치 보상"""
        # 실제로는 더 복잡한 로직이 필요하지만, 여기서는 간단히
        avg_preference = np.mean(list(preferences.values()))
        return max(0.0, min(1.0, avg_preference))


class DRaFTPlusTrainer:
    """DRaFT+ 기반 강화학습 트레이너 (Level 3)"""

    def __init__(
        self,
        model_path: str = "runwayml/stable-diffusion-v1-5",
        save_dir: str = "data/draft_models",
        device: Optional[torch.device] = None,
    ):

        self.model_path = model_path
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or self._get_device()

        # 훈련 설정
        self.training_config = {
            "learning_rate": 1e-6,
            "batch_size": 1,
            "num_epochs": 5,
            "gradient_accumulation_steps": 4,
            "max_grad_norm": 1.0,
            "clip_range": 0.2,  # PPO 클리핑
            "save_steps": 20,
            "warmup_steps": 5,
        }

        self.pipeline = None
        self.reward_model = None
        self.can_train = DIFFUSERS_AVAILABLE

        if self.can_train:
            self._initialize_components()

    def _get_device(self) -> torch.device:
        """디바이스 결정"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _initialize_components(self):
        """컴포넌트 초기화"""
        if not self.can_train:
            logger.warning("DRaFT+ 훈련을 위한 라이브러리가 부족합니다.")
            return

        try:
            logger.info(f"DRaFT+ 훈련용 파이프라인 로드 중: {self.model_path}")

            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                use_safetensors=True,
                safety_checker=None,
                requires_safety_checker=False,
            )

            self.pipeline = self.pipeline.to(self.device)

            # 보상 모델 초기화
            self.reward_model = DRaFTRewardModel(self.device)

            logger.info("DRaFT+ 컴포넌트 초기화 완료")

        except Exception as e:
            logger.error(f"DRaFT+ 컴포넌트 초기화 실패: {e}")
            self.can_train = False

    def prepare_training_data(
        self, gallery_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """방명록 데이터를 DRaFT+ 훈련 데이터로 변환"""

        training_data = []

        for item in gallery_items:
            # 완성된 아이템만 사용 (reflection + hope 모두 있음)
            if (
                item.get("guestbook_title")
                and item.get("reflection_image_path")
                and item.get("hope_image_path")
            ):

                # 방명록 감정 점수 분석
                sentiment_score = self._analyze_sentiment_score(item["guestbook_title"])

                # 모든 데이터를 사용하되, 감정 점수를 보상으로 활용
                training_sample = {
                    "reflection_prompt": item["reflection_prompt"],
                    "hope_prompt": item["hope_prompt"],
                    "sentiment_score": sentiment_score,
                    "guestbook_title": item["guestbook_title"],
                    "guestbook_tags": item.get("guestbook_tags", []),
                    "emotion_keywords": item.get("emotion_keywords", []),
                    "vad_scores": item.get("vad_scores", [0, 0, 0]),
                    "user_id": item["user_id"],
                    "coping_style": item.get("coping_style", "balanced"),
                }

                training_data.append(training_sample)

        logger.info(f"DRaFT+ 훈련 데이터 준비 완료: {len(training_data)}개 샘플")
        return training_data

    def _analyze_sentiment_score(self, title: str) -> float:
        """방명록 제목의 감정 점수 분석"""
        positive_words = {
            "light",
            "bright",
            "hope",
            "peace",
            "joy",
            "calm",
            "beautiful",
            "warm",
            "gentle",
            "soft",
            "serene",
            "harmony",
            "balance",
            "comfort",
        }

        negative_words = {
            "dark",
            "heavy",
            "storm",
            "sad",
            "grey",
            "empty",
            "void",
            "chaos",
            "cold",
            "harsh",
            "bitter",
            "struggle",
            "burden",
            "alone",
            "broken",
        }

        title_lower = title.lower()
        words = title_lower.split()

        if not words:
            return 3.0

        positive_count = sum(
            1 for word in words if any(pos in word for pos in positive_words)
        )
        negative_count = sum(
            1 for word in words if any(neg in word for neg in negative_words)
        )

        # 1-5 척도 계산
        base_score = 3.0
        sentiment_adjustment = (positive_count - negative_count) / len(words) * 2.0

        return max(1.0, min(5.0, base_score + sentiment_adjustment))

    def train_user_draft(
        self, user_id: str, training_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """사용자별 DRaFT+ 모델 훈련"""

        if not self.can_train:
            return self._simulate_draft_training(user_id, len(training_data))

        min_required = 30  # DRaFT+는 더 적은 데이터로도 가능
        if len(training_data) < min_required:
            logger.warning(
                f"DRaFT+ 훈련 데이터 부족: {len(training_data)}개 (최소 {min_required}개 필요)"
            )
            return {
                "success": False,
                "error": "insufficient_data",
                "required": min_required,
                "available": len(training_data),
            }

        try:
            # UNet을 훈련 모드로 설정
            self.pipeline.unet.train()

            # 옵티마이저 설정 (매우 작은 학습률 사용)
            optimizer = optim.AdamW(
                self.pipeline.unet.parameters(),
                lr=self.training_config["learning_rate"],
                weight_decay=0.01,
                eps=1e-8,
            )

            # 학습률 스케줄러
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.training_config["num_epochs"] * len(training_data)
            )

            # 훈련 메트릭
            training_metrics = {
                "policy_losses": [],
                "rewards": [],
                "sentiment_scores": [],
                "kl_divergences": [],
            }

            total_steps = 0

            for epoch in range(self.training_config["num_epochs"]):
                epoch_loss = 0
                epoch_reward = 0

                # 데이터를 랜덤하게 섞음
                shuffled_data = training_data.copy()
                random.shuffle(shuffled_data)

                for step, sample in enumerate(shuffled_data):
                    loss, reward = self._draft_training_step(sample)

                    # 그래디언트 누적
                    loss = loss / self.training_config["gradient_accumulation_steps"]
                    loss.backward()

                    if (step + 1) % self.training_config[
                        "gradient_accumulation_steps"
                    ] == 0:
                        # 그래디언트 클리핑
                        torch.nn.utils.clip_grad_norm_(
                            self.pipeline.unet.parameters(),
                            self.training_config["max_grad_norm"],
                        )

                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                        # 메트릭 기록
                        training_metrics["policy_losses"].append(loss.item())
                        training_metrics["rewards"].append(reward)
                        training_metrics["sentiment_scores"].append(
                            sample["sentiment_score"]
                        )

                        total_steps += 1

                        if total_steps % self.training_config["save_steps"] == 0:
                            logger.info(
                                f"Step {total_steps}: Loss = {loss.item():.4f}, "
                                f"Reward = {reward:.4f}"
                            )

                    epoch_loss += loss.item()
                    epoch_reward += reward

                avg_epoch_loss = epoch_loss / len(shuffled_data)
                avg_epoch_reward = epoch_reward / len(shuffled_data)

                logger.info(
                    f"Epoch {epoch + 1}: Loss = {avg_epoch_loss:.4f}, "
                    f"Reward = {avg_epoch_reward:.4f}"
                )

            # 모델 저장
            save_path = self.save_dir / f"{user_id}_draft"
            self.pipeline.save_pretrained(save_path)

            # 메타데이터 저장
            metadata = {
                "user_id": user_id,
                "training_date": datetime.now().isoformat(),
                "training_data_size": len(training_data),
                "num_epochs": self.training_config["num_epochs"],
                "total_steps": total_steps,
                "final_loss": (
                    training_metrics["policy_losses"][-1]
                    if training_metrics["policy_losses"]
                    else 0
                ),
                "avg_reward": (
                    np.mean(training_metrics["rewards"])
                    if training_metrics["rewards"]
                    else 0
                ),
                "avg_sentiment": (
                    np.mean(training_metrics["sentiment_scores"])
                    if training_metrics["sentiment_scores"]
                    else 0
                ),
                "training_config": self.training_config,
            }

            metadata_path = self.save_dir / f"{user_id}_draft_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            result = {
                "success": True,
                "user_id": user_id,
                "save_path": str(save_path),
                "metadata": metadata,
                "training_metrics": {
                    "final_loss": metadata["final_loss"],
                    "avg_reward": metadata["avg_reward"],
                    "avg_sentiment": metadata["avg_sentiment"],
                    "total_steps": total_steps,
                },
            }

            logger.info(f"사용자 {user_id}의 DRaFT+ 모델 훈련 완료: {save_path}")
            return result

        except Exception as e:
            logger.error(f"DRaFT+ 훈련 실패: {e}")
            return {"success": False, "error": str(e), "user_id": user_id}

    def _draft_training_step(
        self, sample: Dict[str, Any]
    ) -> Tuple[torch.Tensor, float]:
        """DRaFT+ 훈련 스텝"""

        # 희망 프롬프트 사용 (더 긍정적인 결과를 목표)
        prompt = sample["hope_prompt"]

        # 프롬프트 인코딩
        text_inputs = self.pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            text_embeddings = self.pipeline.text_encoder(text_inputs.input_ids)[0]

        # 랜덤 타임스텝과 노이즈
        timesteps = torch.randint(
            0,
            self.pipeline.scheduler.config.num_train_timesteps,
            (1,),
            device=self.device,
        ).long()

        noise = torch.randn((1, 4, 64, 64), device=self.device)

        # 초기 잠재 변수
        latents = torch.randn((1, 4, 64, 64), device=self.device)

        # 노이즈가 추가된 잠재 변수
        noisy_latents = self.pipeline.scheduler.add_noise(latents, noise, timesteps)

        # UNet 예측
        noise_pred = self.pipeline.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embeddings,
            return_dict=False,
        )[0]

        # 기본 MSE 손실
        mse_loss = nn.functional.mse_loss(noise_pred, noise, reduction="mean")

        # 보상 계산 (간단한 특성 추출)
        with torch.no_grad():
            # 이미지 특성을 간단히 노이즈 예측에서 추출
            image_features = noise_pred.flatten().unsqueeze(0)
            if image_features.shape[1] > 1024:
                image_features = image_features[:, :1024]
            elif image_features.shape[1] < 1024:
                padding = torch.zeros(
                    1, 1024 - image_features.shape[1], device=self.device
                )
                image_features = torch.cat([image_features, padding], dim=1)

            # 사용자 선호도 (간단한 더미 데이터)
            user_preferences = {"overall": 0.5}

            reward = self.reward_model.calculate_reward(
                image_features, sample["sentiment_score"], user_preferences
            ).item()

        # DRaFT+ 정책 손실: -log π(a|s) * (R - baseline)
        baseline = 0.5  # 간단한 베이스라인
        advantage = reward - baseline

        # 정책 그래디언트 손실
        policy_loss = mse_loss * (-advantage)  # 보상이 높으면 손실 감소

        return policy_loss, reward

    def _simulate_draft_training(self, user_id: str, data_size: int) -> Dict[str, Any]:
        """DRaFT+ 훈련 시뮬레이션"""

        logger.info(f"사용자 {user_id}의 DRaFT+ 훈련을 시뮬레이션합니다...")

        import time

        time.sleep(3)  # 훈련 시간 시뮬레이션

        # 시뮬레이션된 결과
        simulated_loss = random.uniform(0.05, 0.2)
        simulated_reward = random.uniform(0.6, 0.8)
        simulated_sentiment = random.uniform(3.5, 4.2)

        # 시뮬레이션 정보 저장
        save_path = self.save_dir / f"{user_id}_draft_simulated"
        save_path.mkdir(exist_ok=True)

        metadata = {
            "user_id": user_id,
            "training_date": datetime.now().isoformat(),
            "training_data_size": data_size,
            "simulation": True,
            "simulated_final_loss": simulated_loss,
            "simulated_avg_reward": simulated_reward,
            "simulated_avg_sentiment": simulated_sentiment,
            "note": "실제 라이브러리가 없어 시뮬레이션으로 실행됨",
        }

        metadata_path = save_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return {
            "success": True,
            "user_id": user_id,
            "save_path": str(save_path),
            "simulation": True,
            "metadata": metadata,
            "training_metrics": {
                "final_loss": simulated_loss,
                "avg_reward": simulated_reward,
                "avg_sentiment": simulated_sentiment,
                "total_steps": data_size * 5,
            },
        }

    def get_training_requirements(self, current_data_size: int) -> Dict[str, Any]:
        """DRaFT+ 훈련 요구사항"""

        min_required = 30
        recommended = 50

        return {
            "current_data_size": current_data_size,
            "min_required": min_required,
            "recommended": recommended,
            "can_train": current_data_size >= min_required,
            "data_shortage": max(0, min_required - current_data_size),
            "recommendation": self._get_draft_recommendation(current_data_size),
        }

    def _get_draft_recommendation(self, data_size: int) -> str:
        """DRaFT+ 훈련 권장사항"""

        if data_size < 10:
            return "더 많은 감정 일기 작성과 희망 이미지 생성이 필요합니다."
        elif data_size < 30:
            return f"DRaFT+ 훈련까지 {30 - data_size}개의 완성된 여정이 더 필요합니다."
        elif data_size < 50:
            return "훈련 가능하지만, 더 많은 데이터로 성능을 향상시킬 수 있습니다."
        else:
            return "충분한 데이터가 있어 고품질 DRaFT+ 모델을 훈련할 수 있습니다."

    def load_user_draft(self, user_id: str) -> bool:
        """사용자 DRaFT+ 모델 로드"""

        draft_path = self.save_dir / f"{user_id}_draft"

        if not draft_path.exists():
            logger.warning(f"사용자 {user_id}의 DRaFT+ 모델을 찾을 수 없습니다.")
            return False

        if not self.can_train:
            logger.info(f"시뮬레이션 모드: 사용자 {user_id}의 DRaFT+ 모델 로드됨")
            return True

        try:
            # 파이프라인 재로드
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                draft_path,
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )
            self.pipeline = self.pipeline.to(self.device)

            logger.info(f"사용자 {user_id}의 DRaFT+ 모델 로드 완료")
            return True

        except Exception as e:
            logger.error(f"DRaFT+ 모델 로드 실패: {e}")
            return False

    def cleanup(self):
        """리소스 정리"""

        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

        if self.reward_model is not None:
            del self.reward_model
            self.reward_model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("DRaFT+ 트레이너 리소스가 정리되었습니다.")

    def get_user_draft_info(self, user_id: str) -> Dict[str, Any]:
        """사용자 DRaFT+ 정보 조회"""

        draft_path = self.save_dir / f"{user_id}_draft"
        metadata_path = self.save_dir / f"{user_id}_draft_metadata.json"

        info = {
            "user_id": user_id,
            "draft_exists": draft_path.exists(),
            "metadata_exists": metadata_path.exists(),
        }

        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                info["metadata"] = metadata
            except Exception as e:
                logger.warning(f"DRaFT+ 메타데이터 로드 실패: {e}")

        return info
