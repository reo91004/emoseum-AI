# src/training/draft_trainer.py

# ==============================================================================
# 이 파일은 Level 3 고급 개인화 중 하나인 DRaFT+ 강화학습 모델의 훈련을 담당한다.
# 사용자의 작품 제목 피드백과 도슨트 메시지 반응을 '보상(reward)'으로 사용하여
# Stable Diffusion 모델의 UNet을 직접 미세 조정(fine-tuning)한다.
# 이를 통해 사용자의 선호도에 더 부합하는 이미지를 생성하도록 모델을 점진적으로 개선한다.
# ==============================================================================

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

        # 보상 계산 가중치 (GPT 요소 추가)
        self.reward_weights = {
            "message_reaction_score": 0.35,  # 도슨트 메시지 반응 점수
            "artwork_title_sentiment": 0.25,  # 방명록 감정 점수
            "gpt_quality_score": 0.20,  # GPT 생성 품질 점수
            "personalization_score": 0.10,  # 개인화 수준
            "visual_quality": 0.05,  # 시각적 품질
            "user_preference": 0.05,  # 사용자 선호도 일치
        }

    def calculate_gpt_enhanced_reward(
        self,
        image_features: torch.Tensor,
        message_reaction_score: float,
        artwork_title_sentiment: float,
        gpt_metadata: Dict[str, Any],
        user_preferences: Dict[str, float],
    ) -> torch.Tensor:
        """GPT 메타데이터를 활용한 종합 보상 계산"""

        # 1. 메시지 반응 기반 보상 (1-5 -> 0-1)
        message_reward = (message_reaction_score - 1) / 4

        # 2. 방명록 감정 점수 기반 보상 (1-5 -> 0-1)
        sentiment_reward = (artwork_title_sentiment - 1) / 4

        # 3. GPT 품질 점수 보상
        gpt_quality_reward = self._calculate_gpt_quality_reward(gpt_metadata)

        # 4. 개인화 점수 보상
        personalization_reward = gpt_metadata.get("personalization_score", 0.0)

        # 5. 시각적 품질 보상 (신경망으로 추정)
        visual_reward = self.reward_net(image_features).squeeze()

        # 6. 사용자 선호도 일치 보상
        preference_reward = self._calculate_preference_reward(user_preferences)

        # 가중 평균
        total_reward = (
            self.reward_weights["message_reaction_score"] * message_reward
            + self.reward_weights["artwork_title_sentiment"] * sentiment_reward
            + self.reward_weights["gpt_quality_score"] * gpt_quality_reward
            + self.reward_weights["personalization_score"] * personalization_reward
            + self.reward_weights["visual_quality"] * visual_reward
            + self.reward_weights["user_preference"] * preference_reward
        )

        return torch.clamp(total_reward, 0.0, 1.0)

    def calculate_reward(
        self,
        image_features: torch.Tensor,
        message_reaction_score: float,
        artwork_title_sentiment: float,
        user_preferences: Dict[str, float],
    ) -> torch.Tensor:
        """기존 보상 계산 (하위 호환성)"""

        # 기본 GPT 메타데이터로 호출
        default_gpt_metadata = {
            "prompt_quality_score": 0.5,
            "curator_quality_score": 0.5,
            "personalization_score": 0.0,
        }

        return self.calculate_gpt_enhanced_reward(
            image_features,
            message_reaction_score,
            artwork_title_sentiment,
            default_gpt_metadata,
            user_preferences,
        )

    def _calculate_gpt_quality_reward(self, gpt_metadata: Dict[str, Any]) -> float:
        """GPT 품질 기반 보상 계산"""

        prompt_quality = gpt_metadata.get("prompt_quality_score", 0.5)
        curator_quality = gpt_metadata.get("curator_quality_score", 0.5)

        # 치료적 품질 고려
        therapeutic_quality = gpt_metadata.get("therapeutic_quality", "medium")
        therapeutic_multiplier = {"high": 1.2, "medium": 1.0, "low": 0.8}.get(
            therapeutic_quality, 1.0
        )

        # 안전성 수준 고려
        safety_level = gpt_metadata.get("safety_level", "safe")
        safety_multiplier = {"safe": 1.0, "warning": 0.8, "critical": 0.3}.get(
            safety_level, 1.0
        )

        # 종합 GPT 품질 점수
        combined_quality = (prompt_quality + curator_quality) / 2
        final_quality = combined_quality * therapeutic_multiplier * safety_multiplier

        return min(1.0, max(0.0, final_quality))

    def _calculate_preference_reward(self, preferences: Dict[str, float]) -> float:
        """사용자 선호도 일치 보상"""
        # 실제로는 더 복잡한 로직이 필요하지만, 여기서는 간단히
        avg_preference = np.mean(list(preferences.values()))
        return max(0.0, min(1.0, avg_preference))


class DRaFTPlusTrainer:
    """DRaFT+ 기반 강화학습 트레이너 (Level 3 - GPT 연동)"""

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

    def prepare_gpt_reaction_training_data(
        self, gallery_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """GPT 메시지 반응 기반 훈련 데이터 준비"""

        training_data = []

        for item in gallery_items:
            # 완성된 아이템만 사용 (reflection + artwork_title + docent_message)
            if (
                item.get("artwork_title")
                and item.get("reflection_image_path")
                and item.get("docent_message")
            ):

                # GPT 메타데이터 추출
                gpt_metadata = self._extract_gpt_metadata(item)

                # 방명록 감정 점수 분석
                artwork_title_sentiment = self._analyze_artwork_title_sentiment(
                    item["artwork_title"]
                )

                # 도슨트 메시지 반응 점수 계산
                message_reactions = item.get("message_reactions", [])
                reaction_score = self._calculate_message_reaction_score(
                    message_reactions
                )

                # GPT 품질 기반 필터링 - DRaFT+는 모든 데이터를 사용하되 보상 차등화
                training_sample = {
                    "reflection_prompt": item["reflection_prompt"],
                    "docent_message": item["docent_message"],
                    "artwork_title_sentiment": artwork_title_sentiment,
                    "message_reaction_score": reaction_score,
                    "message_reactions": message_reactions,
                    "artwork_title": item["artwork_title"],
                    "emotion_keywords": item.get("emotion_keywords", []),
                    "vad_scores": item.get("vad_scores", [0, 0, 0]),
                    "user_id": item["user_id"],
                    "coping_style": item.get("coping_style", "balanced"),
                    # GPT 관련 데이터 추가
                    "gpt_metadata": gpt_metadata,
                    "gpt_prompt_used": item.get("gpt_prompt_used", True),
                    "gpt_curator_used": item.get("gpt_curator_used", True),
                    "prompt_generation_time": item.get("prompt_generation_time", 0.0),
                    "curator_generation_method": item.get(
                        "curator_generation_method", "gpt"
                    ),
                }

                training_data.append(training_sample)

        logger.info(f"DRaFT+ 훈련 데이터 준비 완료: {len(training_data)}개 샘플")
        return training_data

    def prepare_training_data(
        self, gallery_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """갤러리 아이템을 DRaFT+ 훈련 데이터로 변환 (GPT 연동)"""

        # GPT 반응 데이터 활용 추가
        return self.prepare_gpt_reaction_training_data(gallery_items)

    def _extract_gpt_metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """갤러리 아이템에서 GPT 메타데이터 추출"""

        metadata = {
            "prompt_quality_score": 0.5,
            "curator_quality_score": 0.5,
            "gpt_tokens_used": 0,
            "gpt_processing_time": 0.0,
            "personalization_score": 0.0,
            "safety_level": "safe",
            "therapeutic_quality": "medium",
            "curator_engagement_score": 0.5,
        }

        # GPT 프롬프트 토큰 수 추출
        if item.get("gpt_prompt_tokens"):
            metadata["gpt_tokens_used"] += item["gpt_prompt_tokens"]

        # GPT 도슨트 토큰 수 추출
        if item.get("gpt_curator_tokens"):
            metadata["gpt_tokens_used"] += item["gpt_curator_tokens"]

        # 생성 시간 정보
        if item.get("prompt_generation_time"):
            metadata["gpt_processing_time"] = item["prompt_generation_time"]

        # 도슨트 메시지의 개인화 수준 분석
        docent_message = item.get("docent_message", {})
        if docent_message and isinstance(docent_message, dict):
            personalization_data = docent_message.get("personalization_data", {})
            if personalization_data:
                # 개인화 요소 수에 따른 점수 계산
                elements = personalization_data.get("personalized_elements", {})
                if elements:
                    metadata["personalization_score"] = min(1.0, len(elements) * 0.2)

                # 대처 스타일 맞춤화 여부
                if personalization_data.get("coping_style"):
                    metadata["personalization_score"] += 0.2

            # 도슨트 메시지 참여도 점수 계산
            metadata["curator_engagement_score"] = (
                self._calculate_docent_engagement_score(docent_message)
            )

        # 프롬프트 품질 점수 추정
        prompt = item.get("reflection_prompt", "")
        if prompt:
            metadata["prompt_quality_score"] = self._estimate_prompt_quality(prompt)

        # 도슨트 메시지 품질 점수 추정
        if docent_message:
            metadata["curator_quality_score"] = self._estimate_curator_quality(
                docent_message
            )

        # 치료적 품질 추정 (메시지 반응 기반)
        message_reactions = item.get("message_reactions", [])
        metadata["therapeutic_quality"] = self._estimate_therapeutic_quality(
            message_reactions
        )

        return metadata

    def _estimate_prompt_quality(self, prompt: str) -> float:
        """프롬프트 품질 추정"""
        quality_score = 0.3  # 기본 점수

        word_count = len(prompt.split())
        if 10 <= word_count <= 50:  # 적절한 길이
            quality_score += 0.2

        # 스타일 지시어 포함 여부
        if any(
            style in prompt.lower() for style in ["style", "tone", "mood", "atmosphere"]
        ):
            quality_score += 0.2

        # 감정 키워드 포함 여부
        if any(
            emotion in prompt.lower()
            for emotion in ["calm", "peaceful", "gentle", "vibrant", "intense"]
        ):
            quality_score += 0.2

        # 시각적 요소 포함 여부
        if any(
            visual in prompt.lower()
            for visual in ["color", "light", "composition", "texture"]
        ):
            quality_score += 0.1

        return min(1.0, quality_score)

    def _estimate_docent_quality(self, docent_message: Dict[str, Any]) -> float:
        """도슨트 메시지 품질 추정"""
        quality_score = 0.2  # 기본 점수

        content = docent_message.get("content", {})
        if content:
            sections = [v for v in content.values() if isinstance(v, str) and v.strip()]
            if len(sections) >= 3:  # 충분한 섹션 수
                quality_score += 0.3

            # 개인화 언급 여부
            combined_text = " ".join(sections).lower()
            if any(
                word in combined_text for word in ["당신의", "당신이", "용기", "성장"]
            ):
                quality_score += 0.3

            # 감정적 지원 요소
            if any(word in combined_text for word in ["함께", "응원", "지지", "이해"]):
                quality_score += 0.2

        return min(1.0, quality_score)

    def _calculate_curator_engagement_score(
        self, docent_message: Dict[str, Any]
    ) -> float:
        """도슨트 메시지 참여도 점수 계산"""

        base_score = 0.3

        # 메시지 구조 복잡성
        content = docent_message.get("content", {})
        if content:
            sections_count = len(
                [v for v in content.values() if isinstance(v, str) and v.strip()]
            )
            base_score += min(0.3, sections_count * 0.1)

        # 개인화 데이터 존재 여부
        personalization_data = docent_message.get("personalization_data", {})
        if personalization_data:
            base_score += 0.2

            # 개인화 요소 다양성
            elements = personalization_data.get("personalized_elements", {})
            if elements:
                base_score += min(0.2, len(elements) * 0.05)

        return min(1.0, base_score)

    def _estimate_therapeutic_quality(self, message_reactions: List[str]) -> str:
        """메시지 반응 기반 치료적 품질 추정"""

        if not message_reactions:
            return "medium"

        positive_reactions = sum(
            1 for reaction in message_reactions if reaction in ["like", "save", "share"]
        )
        total_reactions = len(message_reactions)

        if total_reactions == 0:
            return "medium"

        positive_ratio = positive_reactions / total_reactions

        if positive_ratio >= 0.8:
            return "high"
        elif positive_ratio >= 0.5:
            return "medium"
        else:
            return "low"

    def calculate_gpt_message_reward(
        self, docent_message: Dict[str, Any], user_reactions: List[str]
    ) -> float:
        """GPT 도슨트 메시지 기반 보상 계산"""

        base_reward = 0.5

        # 사용자 반응 기반 보상
        if user_reactions:
            positive_reactions = sum(
                1
                for reaction in user_reactions
                if reaction in ["like", "save", "share"]
            )
            total_reactions = len(user_reactions)

            if total_reactions > 0:
                positive_ratio = positive_reactions / total_reactions
                reaction_bonus = positive_ratio * 0.4
                base_reward += reaction_bonus

        # 메시지 품질 기반 보상
        content = docent_message.get("content", {})
        if content:
            # 메시지 완성도
            sections = [v for v in content.values() if isinstance(v, str) and v.strip()]
            completeness_bonus = min(0.2, len(sections) * 0.04)
            base_reward += completeness_bonus

        # 개인화 수준 기반 보상
        personalization_data = docent_message.get("personalization_data", {})
        if personalization_data:
            personalization_bonus = 0.1

            # 개인화 요소 다양성 보너스
            elements = personalization_data.get("personalized_elements", {})
            if elements:
                personalization_bonus += min(0.1, len(elements) * 0.02)

            base_reward += personalization_bonus

        return min(1.0, max(0.0, base_reward))

    def _analyze_artwork_title_sentiment(self, title: str) -> float:
        """작품 제목의 감정 점수 분석"""

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

        # 제목 분석
        title_lower = title.lower() if title else ""
        title_words = title_lower.split()

        # 태그 분석
        tags_text = " ".join(tags).lower() if tags else ""
        tag_words = tags_text.split()

        all_words = title_words + tag_words

        if not all_words:
            return 3.0

        positive_count = sum(
            1 for word in all_words if any(pos in word for pos in positive_words)
        )
        negative_count = sum(
            1 for word in all_words if any(neg in word for neg in negative_words)
        )

        # 1-5 척도 계산
        base_score = 3.0
        sentiment_adjustment = (positive_count - negative_count) / len(all_words) * 2.0

        return max(1.0, min(5.0, base_score + sentiment_adjustment))

    def _calculate_message_reaction_score(self, reactions: List[str]) -> float:
        """도슨트 메시지 반응 점수 계산"""

        if not reactions:
            return 3.0  # 기본 중성 점수

        # 반응 유형별 점수
        reaction_scores = {
            "like": 4.2,
            "save": 4.5,
            "share": 5.0,
            "dismiss": 2.0,
            "skip": 2.5,
        }

        scores = []
        for reaction in reactions:
            score = reaction_scores.get(reaction, 3.0)
            scores.append(score)

        # 최근 반응에 더 높은 가중치를 주는 가중 평균
        if len(scores) == 1:
            return scores[0]

        weights = [1.0 + i * 0.3 for i in range(len(scores))]
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        weight_sum = sum(weights)

        return weighted_sum / weight_sum

    def train_user_draft(
        self, user_id: str, training_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """사용자별 DRaFT+ 모델 훈련 (GPT 데이터 활용)"""

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
            # GPT 품질 분석
            gpt_analysis = self._analyze_gpt_data_quality(training_data)

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

            # 훈련 메트릭 (GPT 관련 메트릭 추가)
            training_metrics = {
                "policy_losses": [],
                "rewards": [],
                "artwork_title_sentiments": [],
                "message_reaction_scores": [],
                "gpt_quality_scores": [],
                "personalization_scores": [],
                "curator_engagement_scores": [],
                "therapeutic_quality_scores": [],
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
                    loss, reward = self._gpt_enhanced_draft_training_step(sample)

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

                        # 메트릭 기록 (GPT 관련 메트릭 포함)
                        training_metrics["policy_losses"].append(loss.item())
                        training_metrics["rewards"].append(reward)
                        training_metrics["artwork_title_sentiments"].append(
                            sample["artwork_title_sentiment"]
                        )
                        training_metrics["message_reaction_scores"].append(
                            sample["message_reaction_score"]
                        )

                        # GPT 메타데이터 메트릭
                        gpt_metadata = sample.get("gpt_metadata", {})

                        training_metrics["gpt_quality_scores"].append(
                            (
                                gpt_metadata.get("prompt_quality_score", 0.5)
                                + gpt_metadata.get("curator_quality_score", 0.5)
                            )
                            / 2
                        )
                        training_metrics["personalization_scores"].append(
                            gpt_metadata.get("personalization_score", 0.0)
                        )
                        training_metrics["curator_engagement_scores"].append(
                            gpt_metadata.get("curator_engagement_score", 0.5)
                        )

                        # 치료적 품질을 점수로 변환
                        therapeutic_quality = gpt_metadata.get(
                            "therapeutic_quality", "medium"
                        )
                        quality_score = {"high": 1.0, "medium": 0.6, "low": 0.2}.get(
                            therapeutic_quality, 0.6
                        )
                        training_metrics["therapeutic_quality_scores"].append(
                            quality_score
                        )

                        total_steps += 1

                        if total_steps % self.training_config["save_steps"] == 0:
                            logger.info(
                                f"Step {total_steps}: Loss = {loss.item():.4f}, "
                                f"Reward = {reward:.4f}, "
                                f"Message Score = {sample['message_reaction_score']:.2f}, "
                                f"GPT Quality = {training_metrics['gpt_quality_scores'][-1]:.2f}"
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

            # 메타데이터 저장 (GPT 관련 정보 포함)
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
                "avg_artwork_title_sentiment": (
                    np.mean(training_metrics["artwork_title_sentiments"])
                    if training_metrics["artwork_title_sentiments"]
                    else 0
                ),
                "avg_message_reaction_score": (
                    np.mean(training_metrics["message_reaction_scores"])
                    if training_metrics["message_reaction_scores"]
                    else 0
                ),
                "avg_gpt_quality_score": (
                    np.mean(training_metrics["gpt_quality_scores"])
                    if training_metrics["gpt_quality_scores"]
                    else 0
                ),
                "avg_personalization_score": (
                    np.mean(training_metrics["personalization_scores"])
                    if training_metrics["personalization_scores"]
                    else 0
                ),
                "avg_curator_engagement_score": (
                    np.mean(training_metrics["curator_engagement_scores"])
                    if training_metrics["curator_engagement_scores"]
                    else 0
                ),
                "avg_therapeutic_quality_score": (
                    np.mean(training_metrics["therapeutic_quality_scores"])
                    if training_metrics["therapeutic_quality_scores"]
                    else 0
                ),
                "gpt_data_analysis": gpt_analysis,
                "training_config": self.training_config,
                "gpt_integration": {
                    "gpt_enhanced_training": True,
                    "reward_weights": self.reward_model.reward_weights,
                    "gpt_quality_correlation": gpt_analysis.get(
                        "quality_correlation", 0.0
                    ),
                    "personalization_impact": gpt_analysis.get(
                        "personalization_impact", 0.0
                    ),
                },
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
                    "avg_artwork_title_sentiment": metadata["avg_artwork_title_sentiment"],
                    "avg_message_reaction_score": metadata[
                        "avg_message_reaction_score"
                    ],
                    "avg_gpt_quality_score": metadata["avg_gpt_quality_score"],
                    "avg_personalization_score": metadata["avg_personalization_score"],
                    "total_steps": total_steps,
                },
            }

            logger.info(f"사용자 {user_id}의 DRaFT+ 모델 훈련 완료: {save_path}")
            return result

        except Exception as e:
            logger.error(f"DRaFT+ 훈련 실패: {e}")
            return {"success": False, "error": str(e), "user_id": user_id}

    def _analyze_gpt_data_quality(
        self, training_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """GPT 데이터 품질 분석"""

        if not training_data:
            return {"quality_correlation": 0.0, "personalization_impact": 0.0}

        # 품질 점수와 사용자 반응 수집
        quality_scores = []
        reaction_scores = []
        personalization_scores = []

        for sample in training_data:
            gpt_metadata = sample.get("gpt_metadata", {})

            # 종합 품질 점수
            prompt_quality = gpt_metadata.get("prompt_quality_score", 0.5)
            curator_quality = gpt_metadata.get("curator_quality_score", 0.5)
            combined_quality = (prompt_quality + curator_quality) / 2
            quality_scores.append(combined_quality)

            # 사용자 반응 점수
            reaction_scores.append(sample.get("message_reaction_score", 3.0))

            # 개인화 점수
            personalization_scores.append(
                gpt_metadata.get("personalization_score", 0.0)
            )

        # 상관관계 분석
        quality_correlation = 0.0
        personalization_impact = 0.0

        if len(quality_scores) > 1:
            # 품질-반응 상관관계
            correlation_matrix = np.corrcoef(quality_scores, reaction_scores)
            quality_correlation = (
                correlation_matrix[0, 1]
                if not np.isnan(correlation_matrix[0, 1])
                else 0.0
            )

            # 개인화 영향도
            high_personalization = [
                r for p, r in zip(personalization_scores, reaction_scores) if p >= 0.5
            ]
            low_personalization = [
                r for p, r in zip(personalization_scores, reaction_scores) if p < 0.3
            ]

            if high_personalization and low_personalization:
                personalization_impact = np.mean(high_personalization) - np.mean(
                    low_personalization
                )

        return {
            "sample_size": len(training_data),
            "avg_quality_score": (
                float(np.mean(quality_scores)) if quality_scores else 0.0
            ),
            "avg_reaction_score": (
                float(np.mean(reaction_scores)) if reaction_scores else 0.0
            ),
            "avg_personalization_score": (
                float(np.mean(personalization_scores))
                if personalization_scores
                else 0.0
            ),
            "quality_correlation": float(quality_correlation),
            "personalization_impact": float(personalization_impact),
            "high_quality_samples": sum(1 for q in quality_scores if q >= 0.7),
            "high_personalization_samples": sum(
                1 for p in personalization_scores if p >= 0.5
            ),
        }

    def _gpt_enhanced_draft_training_step(
        self, sample: Dict[str, Any]
    ) -> Tuple[torch.Tensor, float]:
        """GPT 메타데이터를 활용한 DRaFT+ 훈련 스텝"""

        # 기본 훈련 스텝
        loss, reward = self._draft_training_step(sample)

        # GPT 메타데이터 기반 손실 조정
        gpt_metadata = sample.get("gpt_metadata", {})

        # 품질 기반 학습률 조정
        combined_quality = (
            gpt_metadata.get("prompt_quality_score", 0.5)
            + gpt_metadata.get("curator_quality_score", 0.5)
        ) / 2

        # 개인화 수준 기반 조정
        personalization_score = gpt_metadata.get("personalization_score", 0.0)

        # 치료적 품질 기반 조정
        therapeutic_quality = gpt_metadata.get("therapeutic_quality", "medium")
        therapeutic_multiplier = {"high": 1.2, "medium": 1.0, "low": 0.8}.get(
            therapeutic_quality, 1.0
        )

        # 종합 조정 계수
        quality_multiplier = 0.7 + (combined_quality * 0.6)  # 0.7 ~ 1.3 범위
        personalization_multiplier = 1.0 + (
            personalization_score * 0.3
        )  # 1.0 ~ 1.3 범위

        # 최종 손실 조정
        adjusted_loss = (
            loss
            * quality_multiplier
            * personalization_multiplier
            * therapeutic_multiplier
        )

        # 보상도 조정
        adjusted_reward = reward * therapeutic_multiplier

        return adjusted_loss, adjusted_reward

    def _draft_training_step(
        self, sample: Dict[str, Any]
    ) -> Tuple[torch.Tensor, float]:
        """DRaFT+ 훈련 스텝"""

        # Reflection 프롬프트 사용 (도슨트 메시지 맥락 포함)
        prompt = sample["reflection_prompt"]

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

        # 보상 계산
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

            # GPT 메타데이터 활용
            gpt_metadata = sample.get("gpt_metadata", {})

            reward = self.reward_model.calculate_gpt_enhanced_reward(
                image_features=image_features,
                message_reaction_score=sample["message_reaction_score"],
                artwork_title_sentiment=sample["artwork_title_sentiment"],
                gpt_metadata=gpt_metadata,
                user_preferences=user_preferences,
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
        simulated_artwork_title_sentiment = random.uniform(3.2, 4.5)
        simulated_message_score = random.uniform(3.5, 4.8)
        simulated_gpt_quality = random.uniform(0.6, 0.8)
        simulated_personalization = random.uniform(0.4, 0.7)
        simulated_curator_engagement = random.uniform(0.5, 0.8)
        simulated_therapeutic_quality = random.uniform(0.6, 0.9)

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
            "simulated_avg_artwork_title_sentiment": simulated_artwork_title_sentiment,
            "simulated_avg_message_reaction_score": simulated_message_score,
            "simulated_avg_gpt_quality_score": simulated_gpt_quality,
            "simulated_avg_personalization_score": simulated_personalization,
            "simulated_avg_curator_engagement_score": simulated_curator_engagement,
            "simulated_avg_therapeutic_quality_score": simulated_therapeutic_quality,
            "gpt_integration": {
                "gpt_enhanced_training": True,
                "simulation_mode": True,
                "quality_correlation": random.uniform(0.4, 0.7),
                "personalization_impact": random.uniform(0.2, 0.5),
            },
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
                "avg_artwork_title_sentiment": simulated_artwork_title_sentiment,
                "avg_message_reaction_score": simulated_message_score,
                "avg_gpt_quality_score": simulated_gpt_quality,
                "avg_personalization_score": simulated_personalization,
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
            return "더 많은 감정 일기 작성과 도슨트 메시지 상호작용이 필요합니다."
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

                # GPT 관련 정보 추출
                gpt_integration = metadata.get("gpt_integration", {})
                if gpt_integration:
                    info["gpt_enhanced"] = gpt_integration.get(
                        "gpt_enhanced_training", False
                    )
                    info["quality_correlation"] = gpt_integration.get(
                        "quality_correlation", 0.0
                    )
                    info["personalization_impact"] = gpt_integration.get(
                        "personalization_impact", 0.0
                    )

            except Exception as e:
                logger.warning(f"DRaFT+ 메타데이터 로드 실패: {e}")

        return info
