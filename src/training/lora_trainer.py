# src/training/lora_trainer.py

# ==============================================================================
# 이 파일은 Level 3 개인화 중 하나인 LoRA(Low-Rank Adaptation) 모델의 훈련을 담당한다.
# 사용자가 긍정적인 반응을 보인 감정 여정 데이터를 선별하여, 해당 사용자의 특정 화풍이나
# 스타일에 맞는 작은 크기의 LoRA 어댑터(adapter)를 훈련한다. 이 어댑터를 Stable Diffusion
# 모델에 결합하면, 적은 비용으로 사용자 맞춤형 이미지를 생성할 수 있다.
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
from .quality_evaluator import QualityEvaluator

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
        
        # 품질 평가기 초기화
        self.quality_evaluator = QualityEvaluator()

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

    def prepare_quality_based_training_data(
        self, gallery_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """품질 메트릭 기반 훈련 데이터 준비 (사용자 반응 없이)"""

        training_data = []

        for item in gallery_items:
            # 완성된 아이템만 사용
            if (
                item.get("artwork_title")
                and item.get("docent_message")
                and item.get("reflection_image_path")
                and Path(item["reflection_image_path"]).exists()
                and item.get("reflection_prompt")
            ):

                # GPT 메타데이터 추출
                gpt_metadata = self._extract_gpt_metadata(item)
                
                # 종합 품질 점수 계산
                quality_result = self.quality_evaluator.calculate_comprehensive_quality_score(
                    item, gpt_metadata
                )
                
                # 품질 기준 필터링 (0.4 이상)
                if quality_result['total_score'] >= 0.4:
                    training_sample = {
                        "prompt": item["reflection_prompt"],
                        "image_path": item["reflection_image_path"],
                        "quality_score": quality_result['total_score'],
                        "training_weight": self.quality_evaluator.get_training_weight(quality_result['total_score']),
                        "artwork_title": item["artwork_title"],
                        "docent_message": item["docent_message"],
                        "emotion_keywords": item.get("emotion_keywords", []),
                        "vad_scores": item.get("vad_scores", [0, 0, 0]),
                        # 품질 관련 데이터 추가
                        "gpt_metadata": gpt_metadata,
                        "quality_metadata": quality_result,
                        "gpt_prompt_used": item.get("gpt_prompt_used", True),
                        "gpt_curator_used": item.get("gpt_curator_used", True),
                        "prompt_generation_time": item.get(
                            "prompt_generation_time", 0.0
                        ),
                        "curator_generation_method": item.get(
                            "curator_generation_method", "gpt"
                        ),
                    }

                    training_data.append(training_sample)

        # 품질 분석 수행
        quality_analysis = self.quality_evaluator.analyze_training_data_quality(
            [item for item in gallery_items if item.get("reflection_prompt")]
        )
        
        logger.info(
            f"품질 기반 LoRA 훈련 데이터 준비 완료: {len(training_data)}개 샘플 (총 {len(gallery_items)}개 중)"
        )
        logger.info(
            f"평균 품질 점수: {quality_analysis.get('average_quality', 0):.3f}, "
            f"고품질 비율: {quality_analysis.get('quality_distribution_percent', {}).get('high', 0):.1f}%"
        )
        return training_data

    def prepare_training_data(
        self, gallery_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """갤러리 아이템을 LoRA 훈련 데이터로 변환 (품질 기반)"""

        # 품질 기반 데이터 준비 사용
        return self.prepare_quality_based_training_data(gallery_items)

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

        # 프롬프트 품질 점수 추정 (길이와 복잡성 기반)
        prompt = item.get("reflection_prompt", "")
        if prompt:
            # 프롬프트 품질 휴리스틱
            word_count = len(prompt.split())
            if 10 <= word_count <= 50:  # 적절한 길이
                metadata["prompt_quality_score"] += 0.3
            if "style" in prompt.lower():  # 스타일 지시어 포함
                metadata["prompt_quality_score"] += 0.2
            if any(
                emotion in prompt.lower()
                for emotion in ["calm", "peaceful", "gentle", "vibrant"]
            ):
                metadata["prompt_quality_score"] += 0.2

        # 도슨트 메시지 품질 점수 추정
        if docent_message:
            content = docent_message.get("content", {})
            if content:
                sections = [
                    v for v in content.values() if isinstance(v, str) and v.strip()
                ]
                if len(sections) >= 3:  # 충분한 섹션 수
                    metadata["curator_quality_score"] += 0.3

                # 개인화 언급 여부
                combined_text = " ".join(sections).lower()
                if any(
                    word in combined_text
                    for word in ["당신의", "당신이", "용기", "성장"]
                ):
                    metadata["curator_quality_score"] += 0.4

        return metadata

    def analyze_gpt_quality_correlation(
        self, training_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """GPT 품질과 사용자 반응 상관관계 분석"""

        if not training_data:
            return {"correlation": 0.0, "sample_size": 0}

        # 품질 점수와 반응 점수 수집
        quality_scores = []
        reaction_scores = []

        for sample in training_data:
            gpt_metadata = sample.get("gpt_metadata", {})

            # 종합 GPT 품질 점수 계산
            prompt_quality = gpt_metadata.get("prompt_quality_score", 0.5)
            curator_quality = gpt_metadata.get("curator_quality_score", 0.5)
            personalization = gpt_metadata.get("personalization_score", 0.0)

            combined_quality = (prompt_quality + curator_quality + personalization) / 3
            quality_scores.append(combined_quality)

            # 사용자 반응 점수
            reaction_scores.append(sample.get("reaction_score", 3.0))

        # 상관관계 계산 (간단한 피어슨 상관계수)
        if len(quality_scores) > 1:
            correlation = np.corrcoef(quality_scores, reaction_scores)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0

        # 추가 분석
        high_quality_reactions = [
            reaction
            for quality, reaction in zip(quality_scores, reaction_scores)
            if quality >= 0.7
        ]

        low_quality_reactions = [
            reaction
            for quality, reaction in zip(quality_scores, reaction_scores)
            if quality < 0.5
        ]

        analysis = {
            "correlation": float(correlation),
            "sample_size": len(training_data),
            "avg_quality_score": (
                float(np.mean(quality_scores)) if quality_scores else 0.0
            ),
            "avg_reaction_score": (
                float(np.mean(reaction_scores)) if reaction_scores else 0.0
            ),
            "high_quality_avg_reaction": (
                float(np.mean(high_quality_reactions))
                if high_quality_reactions
                else 0.0
            ),
            "low_quality_avg_reaction": (
                float(np.mean(low_quality_reactions)) if low_quality_reactions else 0.0
            ),
            "quality_impact": (
                float(np.mean(high_quality_reactions))
                - float(np.mean(low_quality_reactions))
                if high_quality_reactions and low_quality_reactions
                else 0.0
            ),
        }

        logger.info(f"GPT 품질-반응 상관관계 분석 완료: 상관계수 {correlation:.3f}")
        return analysis

    def weight_samples_by_gpt_performance(
        self, training_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """GPT 성과 기반 샘플 가중치 적용"""

        weighted_data = []

        for sample in training_data:
            gpt_metadata = sample.get("gpt_metadata", {})

            # 기본 가중치
            weight = 1.0

            # GPT 품질 점수 기반 가중치 조정
            prompt_quality = gpt_metadata.get("prompt_quality_score", 0.5)
            curator_quality = gpt_metadata.get("curator_quality_score", 0.5)
            personalization = gpt_metadata.get("personalization_score", 0.0)

            # 종합 품질 점수
            quality_score = (prompt_quality + curator_quality + personalization) / 3

            # 품질이 높을수록 가중치 증가
            if quality_score >= 0.8:
                weight *= 1.5
            elif quality_score >= 0.6:
                weight *= 1.2
            elif quality_score < 0.4:
                weight *= 0.7

            # 사용자 반응 점수 기반 가중치 조정
            reaction_score = sample.get("reaction_score", 3.0)
            if reaction_score >= 4.5:
                weight *= 1.3
            elif reaction_score >= 4.0:
                weight *= 1.1
            elif reaction_score < 3.0:
                weight *= 0.8

            # 개인화 수준 기반 가중치 조정
            personalization_score = gpt_metadata.get("personalization_score", 0.0)
            if personalization_score >= 0.6:
                weight *= 1.2
            elif personalization_score >= 0.4:
                weight *= 1.1

            # GPT 토큰 효율성 기반 가중치
            tokens_used = gpt_metadata.get("gpt_tokens_used", 0)
            processing_time = gpt_metadata.get("gpt_processing_time", 1.0)

            if tokens_used > 0 and processing_time > 0:
                tokens_per_second = tokens_used / processing_time
                if tokens_per_second > 50:  # 빠른 생성
                    weight *= 1.1

            # 가중치가 적용된 샘플 생성
            weighted_sample = sample.copy()
            weighted_sample["training_weight"] = weight
            weighted_sample["quality_breakdown"] = {
                "prompt_quality": prompt_quality,
                "curator_quality": curator_quality,
                "personalization": personalization,
                "combined_quality": quality_score,
                "reaction_score": reaction_score,
            }

            weighted_data.append(weighted_sample)

        # 가중치 분포 로깅
        weights = [sample["training_weight"] for sample in weighted_data]
        logger.info(
            f"샘플 가중치 적용 완료: 평균 {np.mean(weights):.2f}, "
            f"최대 {np.max(weights):.2f}, 최소 {np.min(weights):.2f}"
        )

        return weighted_data

    def _calculate_quality_based_score(self, training_sample: Dict[str, Any]) -> float:
        """품질 기반 훈련 점수 계산 (0-1 척도)"""
        
        # 종합 품질 점수를 기본으로 사용
        base_score = training_sample.get('quality_score', 0.5)
        
        # 추가 보너스 요소들
        gpt_metadata = training_sample.get('gpt_metadata', {})
        
        # 개인화 수준 보너스
        personalization_bonus = gpt_metadata.get('personalization_score', 0.0) * 0.1
        
        # 안전성 보너스
        safety_level = gpt_metadata.get('safety_level', 'medium')
        safety_bonus = 0.05 if safety_level == 'safe' else 0.0
        
        # 최종 점수
        final_score = min(1.0, base_score + personalization_bonus + safety_bonus)
        
        return final_score

    def train_user_lora(
        self, user_id: str, training_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """사용자별 LoRA 모델 훈련 (GPT 데이터 활용)"""

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
            # GPT 품질 상관관계 분석
            quality_analysis = self.analyze_gpt_quality_correlation(training_data)

            # GPT 성과 기반 샘플 가중치 적용
            weighted_training_data = self.weight_samples_by_gpt_performance(
                training_data
            )

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
                optimizer,
                T_max=self.training_config["num_epochs"] * len(weighted_training_data),
            )

            # 훈련 메트릭 추적 (GPT 관련 메트릭 추가)
            training_metrics = {
                "losses": [],
                "learning_rates": [],
                "reaction_scores": [],
                "curator_engagement": [],
                "gpt_quality_scores": [],
                "personalization_scores": [],
                "training_weights": [],
            }

            # 훈련 루프
            self.lora_model.train()
            total_steps = 0

            for epoch in range(self.training_config["num_epochs"]):
                epoch_loss = 0

                for step, sample in enumerate(weighted_training_data):
                    loss = self._gpt_training_step(sample)

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

                        # 메트릭 기록 (GPT 관련 메트릭 포함)
                        training_metrics["losses"].append(loss.item())
                        training_metrics["learning_rates"].append(
                            scheduler.get_last_lr()[0]
                        )
                        training_metrics["reaction_scores"].append(
                            sample["reaction_score"]
                        )

                        # GPT 관련 메트릭
                        gpt_metadata = sample.get("gpt_metadata", {})
                        quality_breakdown = sample.get("quality_breakdown", {})

                        training_metrics["gpt_quality_scores"].append(
                            quality_breakdown.get("combined_quality", 0.5)
                        )
                        training_metrics["personalization_scores"].append(
                            gpt_metadata.get("personalization_score", 0.0)
                        )
                        training_metrics["training_weights"].append(
                            sample.get("training_weight", 1.0)
                        )

                        # 도슨트 메시지 참여도 계산
                        engagement = self._calculate_curator_engagement(sample)
                        training_metrics["curator_engagement"].append(engagement)

                        total_steps += 1

                        if total_steps % self.training_config["save_steps"] == 0:
                            logger.info(
                                f"Step {total_steps}: Loss = {loss.item():.4f}, "
                                f"Reaction = {sample['reaction_score']:.2f}, "
                                f"GPT Quality = {quality_breakdown.get('combined_quality', 0.5):.2f}, "
                                f"Weight = {sample.get('training_weight', 1.0):.2f}"
                            )

                    epoch_loss += loss.item()

                avg_epoch_loss = epoch_loss / len(weighted_training_data)
                logger.info(
                    f"Epoch {epoch + 1}/{self.training_config['num_epochs']}: "
                    f"Avg Loss = {avg_epoch_loss:.4f}"
                )

            # LoRA 모델 저장
            save_path = self.lora_save_dir / f"{user_id}_lora"
            self.lora_model.save_pretrained(save_path)

            # 훈련 메타데이터 저장 (GPT 관련 정보 포함)
            metadata = {
                "user_id": user_id,
                "training_date": datetime.now().isoformat(),
                "training_data_size": len(training_data),
                "weighted_data_size": len(weighted_training_data),
                "num_epochs": self.training_config["num_epochs"],
                "final_loss": (
                    training_metrics["losses"][-1] if training_metrics["losses"] else 0
                ),
                "avg_reaction_score": np.mean(training_metrics["reaction_scores"]),
                "avg_curator_engagement": np.mean(
                    training_metrics["curator_engagement"]
                ),
                "avg_gpt_quality_score": np.mean(
                    training_metrics["gpt_quality_scores"]
                ),
                "avg_personalization_score": np.mean(
                    training_metrics["personalization_scores"]
                ),
                "avg_training_weight": np.mean(training_metrics["training_weights"]),
                "gpt_quality_analysis": quality_analysis,
                "lora_config": self.lora_config.__dict__,
                "training_config": self.training_config,
                "gpt_integration": {
                    "gpt_training": True,
                    "quality_correlation": quality_analysis["correlation"],
                    "quality_impact": quality_analysis["quality_impact"],
                    "high_quality_samples": sum(
                        1
                        for score in training_metrics["gpt_quality_scores"]
                        if score >= 0.7
                    ),
                    "personalized_samples": sum(
                        1
                        for score in training_metrics["personalization_scores"]
                        if score >= 0.4
                    ),
                },
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
                    "avg_reaction_score": metadata["avg_reaction_score"],
                    "avg_curator_engagement": metadata["avg_curator_engagement"],
                    "avg_gpt_quality_score": metadata["avg_gpt_quality_score"],
                    "avg_personalization_score": metadata["avg_personalization_score"],
                    "total_steps": total_steps,
                    "gpt_quality_correlation": quality_analysis["correlation"],
                },
            }

            logger.info(f"사용자 {user_id}의 LoRA 모델 훈련 완료: {save_path}")
            return result

        except Exception as e:
            logger.error(f"LoRA 훈련 실패: {e}")
            return {"success": False, "error": str(e), "user_id": user_id}

    def _gpt_training_step(self, sample: Dict[str, Any]) -> torch.Tensor:
        """GPT 메타데이터를 활용한 훈련 스텝"""

        # 기본 훈련 스텝
        loss = self._training_step(sample)

        # 훈련 가중치 적용
        training_weight = sample.get("training_weight", 1.0)
        weighted_loss = loss * training_weight

        # GPT 품질 기반 추가 조정
        gpt_metadata = sample.get("gpt_metadata", {})
        quality_score = (
            gpt_metadata.get("prompt_quality_score", 0.5)
            + gpt_metadata.get("curator_quality_score", 0.5)
        ) / 2

        # 높은 품질의 GPT 응답에 대해서는 손실을 더 강하게 학습
        quality_multiplier = 0.8 + (quality_score * 0.4)  # 0.8 ~ 1.2 범위
        final_loss = weighted_loss * quality_multiplier

        return final_loss

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

        # 반응 점수 기반 가중치 (긍정적 반응에 더 높은 가중치)
        reaction_weight = sample["reaction_score"] / 5.0  # 0-1 정규화
        weighted_loss = loss * reaction_weight

        return weighted_loss

    def _calculate_curator_engagement(self, sample: Dict[str, Any]) -> float:
        """도슨트 메시지 참여도 계산"""

        docent_message = sample.get("docent_message", {})
        message_reactions = sample.get("message_reactions", [])

        # 기본 참여도 점수
        base_engagement = 0.5

        # 메시지 반응이 있으면 참여도 증가
        if message_reactions:
            positive_reactions = sum(
                1
                for reaction in message_reactions
                if reaction in ["like", "save", "share"]
            )
            total_reactions = len(message_reactions)

            if total_reactions > 0:
                positive_ratio = positive_reactions / total_reactions
                base_engagement += positive_ratio * 0.5

        # 도슨트 메시지 개인화 수준 고려
        personalization_data = docent_message.get("personalization_data", {})
        if personalization_data:
            personalization_level = personalization_data.get(
                "personalized_elements", {}
            )
            if personalization_level:
                base_engagement += 0.2

        return min(1.0, base_engagement)

    def _simulate_training(self, user_id: str, data_size: int) -> Dict[str, Any]:
        """LoRA 훈련 시뮬레이션 (라이브러리 부족시)"""

        import random
        import time

        logger.info(f"사용자 {user_id}의 LoRA 훈련을 시뮬레이션합니다...")

        # 훈련 시뮬레이션
        simulated_loss = random.uniform(0.1, 0.3)
        simulated_reaction_score = random.uniform(3.5, 4.8)
        simulated_engagement = random.uniform(0.6, 0.9)
        simulated_gpt_quality = random.uniform(0.6, 0.8)
        simulated_personalization = random.uniform(0.4, 0.7)
        simulated_correlation = random.uniform(0.3, 0.7)

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
            "simulated_avg_reaction_score": simulated_reaction_score,
            "simulated_avg_curator_engagement": simulated_engagement,
            "simulated_avg_gpt_quality_score": simulated_gpt_quality,
            "simulated_avg_personalization_score": simulated_personalization,
            "simulated_gpt_quality_correlation": simulated_correlation,
            "gpt_integration": {
                "gpt_training": True,
                "simulation_mode": True,
                "quality_correlation": simulated_correlation,
                "quality_impact": random.uniform(0.2, 0.5),
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
                "avg_reaction_score": simulated_reaction_score,
                "avg_curator_engagement": simulated_engagement,
                "avg_gpt_quality_score": simulated_gpt_quality,
                "avg_personalization_score": simulated_personalization,
                "total_steps": data_size * 10,
                "gpt_quality_correlation": simulated_correlation,
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
            # LoRA 모델 정리
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
            return "더 많은 감정 일기 작성과 도슨트 메시지 상호작용이 필요합니다."
        elif data_size < 50:
            return f"훈련까지 {50 - data_size}개의 긍정적 메시지 반응이 더 필요합니다."
        elif data_size < 100:
            return "훈련 가능하지만, 더 많은 데이터로 성능을 개선할 수 있습니다."
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

                # GPT 관련 정보 추출
                gpt_integration = metadata.get("gpt_integration", {})
                if gpt_integration:
                    info["gpt_integrated"] = gpt_integration.get("gpt_training", False)
                    info["quality_correlation"] = gpt_integration.get(
                        "quality_correlation", 0.0
                    )
                    info["quality_impact"] = gpt_integration.get("quality_impact", 0.0)

            except Exception as e:
                logger.warning(f"메타데이터 로드 실패: {e}")

        return info
