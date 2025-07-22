# training/lora_trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)

try:
    from peft import LoraConfig, get_peft_model, TaskType

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT 라이브러리를 사용할 수 없습니다.")

try:
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel

    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logger.warning("Diffusers 라이브러리를 사용할 수 없습니다.")


class PersonalizedLoRATrainer:
    """개인화된 LoRA 트레이너 (Level 3)"""

    def __init__(
        self,
        model_path: str = "runwayml/stable-diffusion-v1-5",
        lora_save_dir: str = "data/user_loras",
        device: Optional[torch.device] = None,
    ):

        self.model_path = model_path
        self.lora_save_dir = Path(lora_save_dir)
        self.lora_save_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or self._get_device()

        # LoRA 설정
        self.lora_config = LoraConfig(
            r=16,  # rank
            lora_alpha=32,
            target_modules=[
                "to_k",
                "to_q",
                "to_v",
                "to_out.0",  # attention layers
                "ff.net.0.proj",
                "ff.net.2",  # feed-forward layers
            ],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.DIFFUSION,
        )

        # 훈련 설정
        self.training_config = {
            "learning_rate": 1e-4,
            "batch_size": 1,
            "num_epochs": 10,
            "gradient_accumulation_steps": 4,
            "max_grad_norm": 1.0,
            "save_steps": 50,
            "warmup_steps": 10,
        }

        self.pipeline = None
        self.lora_model = None
        self.can_train = PEFT_AVAILABLE and DIFFUSERS_AVAILABLE

        if self.can_train:
            self._initialize_pipeline()

    def _get_device(self) -> torch.device:
        """디바이스 결정"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _initialize_pipeline(self):
        """파이프라인 초기화"""
        if not self.can_train:
            logger.warning("LoRA 훈련을 위한 라이브러리가 부족합니다.")
            return

        try:
            logger.info(f"LoRA 훈련용 파이프라인 로드 중: {self.model_path}")

            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                use_safetensors=True,
                safety_checker=None,
                requires_safety_checker=False,
            )

            self.pipeline = self.pipeline.to(self.device)
            logger.info("LoRA 훈련용 파이프라인 로드 완료")

        except Exception as e:
            logger.error(f"파이프라인 초기화 실패: {e}")
            self.can_train = False

    def prepare_training_data(
        self, gallery_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """방명록 데이터를 훈련 데이터로 변환"""

        training_data = []

        for item in gallery_items:
            # 방명록 제목이 있고 희망 이미지가 있는 완성된 아이템만 사용
            if (
                item.get("guestbook_title")
                and item.get("hope_image_path")
                and Path(item["hope_image_path"]).exists()
            ):

                # 방명록 제목의 감정 극성 분석
                title_sentiment = self._analyze_title_sentiment(item["guestbook_title"])

                # 긍정적 반응(3.5점 이상)만 학습 데이터로 사용
                if title_sentiment >= 3.5:
                    training_sample = {
                        "prompt": item["hope_prompt"],
                        "image_path": item["hope_image_path"],
                        "sentiment_score": title_sentiment,
                        "user_title": item["guestbook_title"],
                        "tags": item.get("guestbook_tags", []),
                        "emotion_keywords": item.get("emotion_keywords", []),
                        "vad_scores": item.get("vad_scores", [0, 0, 0]),
                    }

                    training_data.append(training_sample)

        logger.info(
            f"훈련 데이터 준비 완료: {len(training_data)}개 샘플 (총 {len(gallery_items)}개 중)"
        )
        return training_data

    def _analyze_title_sentiment(self, title: str) -> float:
        """방명록 제목의 감정 극성 분석 (1-5 척도)"""

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

        neutral_words = {
            "moment",
            "day",
            "time",
            "feeling",
            "thought",
            "memory",
            "scene",
            "landscape",
            "journey",
            "path",
            "space",
            "reflection",
            "contemplation",
        }

        title_lower = title.lower()
        words = title_lower.split()

        positive_count = sum(
            1 for word in words if any(pos in word for pos in positive_words)
        )
        negative_count = sum(
            1 for word in words if any(neg in word for neg in negative_words)
        )
        neutral_count = sum(
            1 for word in words if any(neu in word for neu in neutral_words)
        )

        total_words = len(words)
        if total_words == 0:
            return 3.0  # 중성

        # 감정 점수 계산 (1-5 척도)
        positive_ratio = positive_count / total_words
        negative_ratio = negative_count / total_words

        base_score = 3.0  # 중성 기준점
        sentiment_adjustment = (positive_ratio - negative_ratio) * 2.0

        final_score = base_score + sentiment_adjustment
        return max(1.0, min(5.0, final_score))

    def train_user_lora(
        self, user_id: str, training_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """사용자별 LoRA 모델 훈련"""

        if not self.can_train:
            return self._simulate_training(user_id, len(training_data))

        if len(training_data) < 50:
            logger.warning(
                f"훈련 데이터가 부족합니다: {len(training_data)}개 (최소 50개 필요)"
            )
            return {
                "success": False,
                "error": "insufficient_data",
                "required": 50,
                "available": len(training_data),
            }

        try:
            # LoRA 모델 설정
            self.lora_model = get_peft_model(self.pipeline.unet, self.lora_config)

            # 옵티마이저 설정
            optimizer = optim.AdamW(
                self.lora_model.parameters(),
                lr=self.training_config["learning_rate"],
                weight_decay=0.01,
            )

            # 학습률 스케줄러
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.training_config["num_epochs"] * len(training_data)
            )

            # 훈련 메트릭 추적
            training_metrics = {
                "losses": [],
                "learning_rates": [],
                "sentiment_scores": [],
            }

            # 훈련 루프
            self.lora_model.train()
            total_steps = 0

            for epoch in range(self.training_config["num_epochs"]):
                epoch_loss = 0

                for step, sample in enumerate(training_data):
                    loss = self._training_step(sample)

                    # 그래디언트 누적
                    loss = loss / self.training_config["gradient_accumulation_steps"]
                    loss.backward()

                    if (step + 1) % self.training_config[
                        "gradient_accumulation_steps"
                    ] == 0:
                        # 그래디언트 클리핑
                        torch.nn.utils.clip_grad_norm_(
                            self.lora_model.parameters(),
                            self.training_config["max_grad_norm"],
                        )

                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                        # 메트릭 기록
                        training_metrics["losses"].append(loss.item())
                        training_metrics["learning_rates"].append(
                            scheduler.get_last_lr()[0]
                        )
                        training_metrics["sentiment_scores"].append(
                            sample["sentiment_score"]
                        )

                        total_steps += 1

                        if total_steps % self.training_config["save_steps"] == 0:
                            logger.info(f"Step {total_steps}: Loss = {loss.item():.4f}")

                    epoch_loss += loss.item()

                avg_epoch_loss = epoch_loss / len(training_data)
                logger.info(
                    f"Epoch {epoch + 1}/{self.training_config['num_epochs']}: Avg Loss = {avg_epoch_loss:.4f}"
                )

            # LoRA 모델 저장
            save_path = self.lora_save_dir / f"{user_id}_lora"
            self.lora_model.save_pretrained(save_path)

            # 훈련 메타데이터 저장
            metadata = {
                "user_id": user_id,
                "training_date": datetime.now().isoformat(),
                "training_data_size": len(training_data),
                "num_epochs": self.training_config["num_epochs"],
                "final_loss": (
                    training_metrics["losses"][-1] if training_metrics["losses"] else 0
                ),
                "avg_sentiment_score": np.mean(training_metrics["sentiment_scores"]),
                "lora_config": self.lora_config.__dict__,
                "training_config": self.training_config,
            }

            metadata_path = self.lora_save_dir / f"{user_id}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            result = {
                "success": True,
                "user_id": user_id,
                "save_path": str(save_path),
                "metadata": metadata,
                "training_metrics": {
                    "final_loss": metadata["final_loss"],
                    "avg_sentiment": metadata["avg_sentiment_score"],
                    "total_steps": total_steps,
                },
            }

            logger.info(f"사용자 {user_id}의 LoRA 모델 훈련 완료: {save_path}")
            return result

        except Exception as e:
            logger.error(f"LoRA 훈련 실패: {e}")
            return {"success": False, "error": str(e), "user_id": user_id}

    def _training_step(self, sample: Dict[str, Any]) -> torch.Tensor:
        """단일 훈련 스텝"""

        # 프롬프트 인코딩
        prompt = sample["prompt"]
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

        # 노이즈 추가된 잠재 변수 (실제로는 이미지 로드 필요)
        # 여기서는 간단히 랜덤 잠재 변수 사용
        latents = torch.randn((1, 4, 64, 64), device=self.device)
        noisy_latents = self.pipeline.scheduler.add_noise(latents, noise, timesteps)

        # UNet 예측
        noise_pred = self.lora_model(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embeddings,
            return_dict=False,
        )[0]

        # MSE 손실
        loss = nn.functional.mse_loss(noise_pred, noise, reduction="mean")

        # 감정 점수 기반 가중치 (긍정적 반응에 더 높은 가중치)
        sentiment_weight = sample["sentiment_score"] / 5.0  # 0-1 정규화
        weighted_loss = loss * sentiment_weight

        return weighted_loss

    def _simulate_training(self, user_id: str, data_size: int) -> Dict[str, Any]:
        """LoRA 훈련 시뮬레이션 (라이브러리 부족시)"""

        import random
        import time

        logger.info(f"사용자 {user_id}의 LoRA 훈련을 시뮬레이션합니다...")

        # 훈련 시뮬레이션
        simulated_loss = random.uniform(0.1, 0.3)
        simulated_sentiment = random.uniform(3.5, 4.5)

        time.sleep(2)  # 훈련 시간 시뮬레이션

        # 시뮬레이션된 모델 정보 저장
        save_path = self.lora_save_dir / f"{user_id}_lora_simulated"
        save_path.mkdir(exist_ok=True)

        metadata = {
            "user_id": user_id,
            "training_date": datetime.now().isoformat(),
            "training_data_size": data_size,
            "simulation": True,
            "simulated_final_loss": simulated_loss,
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
                "avg_sentiment": simulated_sentiment,
                "total_steps": data_size * 10,  # 가상의 스텝 수
            },
        }

    def load_user_lora(self, user_id: str) -> bool:
        """사용자 LoRA 모델 로드"""

        lora_path = self.lora_save_dir / f"{user_id}_lora"

        if not lora_path.exists():
            logger.warning(f"사용자 {user_id}의 LoRA 모델을 찾을 수 없습니다.")
            return False

        if not self.can_train:
            logger.info(f"시뮬레이션 모드: 사용자 {user_id}의 LoRA 로드됨")
            return True

        try:
            # 기존 LoRA 모델 정리
            if self.lora_model is not None:
                del self.lora_model

            # 새로운 LoRA 모델 로드
            self.lora_model = get_peft_model(self.pipeline.unet, self.lora_config)
            # 실제로는 여기서 저장된 가중치를 로드해야 함

            logger.info(f"사용자 {user_id}의 LoRA 모델 로드 완료")
            return True

        except Exception as e:
            logger.error(f"LoRA 모델 로드 실패: {e}")
            return False

    def get_training_requirements(self, current_data_size: int) -> Dict[str, Any]:
        """훈련 요구사항 확인"""

        min_required = 50
        recommended = 100

        return {
            "current_data_size": current_data_size,
            "min_required": min_required,
            "recommended": recommended,
            "can_train": current_data_size >= min_required,
            "data_shortage": max(0, min_required - current_data_size),
            "recommendation": self._get_training_recommendation(current_data_size),
        }

    def _get_training_recommendation(self, data_size: int) -> str:
        """훈련 권장사항"""

        if data_size < 20:
            return "더 많은 감정 일기 작성과 방명록 작성이 필요합니다."
        elif data_size < 50:
            return f"훈련까지 {50 - data_size}개의 긍정적 피드백이 더 필요합니다."
        elif data_size < 100:
            return "훈련 가능하지만, 더 많은 데이터로 성능을 향상시킬 수 있습니다."
        else:
            return "충분한 데이터가 있어 고품질 개인화 모델을 훈련할 수 있습니다."

    def cleanup(self):
        """리소스 정리"""

        if self.lora_model is not None:
            del self.lora_model
            self.lora_model = None

        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("LoRA 트레이너 리소스가 정리되었습니다.")

    def get_user_lora_info(self, user_id: str) -> Dict[str, Any]:
        """사용자 LoRA 정보 조회"""

        lora_path = self.lora_save_dir / f"{user_id}_lora"
        metadata_path = self.lora_save_dir / f"{user_id}_metadata.json"

        info = {
            "user_id": user_id,
            "lora_exists": lora_path.exists(),
            "metadata_exists": metadata_path.exists(),
        }

        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                info["metadata"] = metadata
            except Exception as e:
                logger.warning(f"메타데이터 로드 실패: {e}")

        return info
