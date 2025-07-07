#!/usr/bin/env python3
"""
DRaFTPlusRewardModel - DRaFT+ 방식의 개선된 보상 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from config import device, logger
from models.emotion import EmotionEmbedding
from models.user_profile import UserEmotionProfile
from models.improved_evaluators import (
    ImprovedAestheticEvaluator,
    ImprovedEmotionEvaluator,
    ImprovedPersonalizationEvaluator
)
from models.improved_diversity_evaluator import ImprovedDiversityEvaluator


class DRaFTPlusRewardModel:
    """DRaFT+ 방식의 개선된 보상 모델"""

    def __init__(self, device: torch.device = None):
        self.device = device if device else torch.device("cpu")

        # 개선된 평가기들
        self.aesthetic_evaluator = ImprovedAestheticEvaluator()
        self.emotion_evaluator = ImprovedEmotionEvaluator()
        self.personalization_evaluator = ImprovedPersonalizationEvaluator()

        # 다양성 평가기 (개선된 버전)
        self.diversity_evaluator = ImprovedDiversityEvaluator()

        logger.info("✅ 개선된 DRaFT+ 보상 모델 초기화 완료")

    # 기존 단순한 평가기 빌더들은 모두 개선된 평가기로 대체됨

    def calculate_comprehensive_reward(
        self,
        generated_images: torch.Tensor,
        target_emotion: EmotionEmbedding,
        user_profile: UserEmotionProfile,
        previous_images: List[torch.Tensor] = None,
    ) -> torch.Tensor:
        """종합적인 보상 계산 (DRaFT+ 방식)"""
        batch_size = generated_images.shape[0]

        try:
            with torch.no_grad():
                # 1. 감정 정확도 보상 (개선된 평가기)
                emotion_reward = self.emotion_evaluator.evaluate(
                    generated_images, target_emotion
                )

                # 2. 미적 품질 보상 (개선된 평가기)
                aesthetic_reward = self.aesthetic_evaluator.evaluate(generated_images)

                # 3. 개인화 보상 (개선된 평가기)
                personalization_reward = self.personalization_evaluator.evaluate(
                    generated_images, user_profile
                )

                # 4. 다양성 보상 (DRaFT+ 추가 요소) - 개선된 평가기
                diversity_reward = self.diversity_evaluator.evaluate(
                    generated_images, previous_images
                )

                # 5. 가중 합계 (DRaFT+ 가중치)
                total_reward = (
                    0.35 * emotion_reward
                    + 0.25 * aesthetic_reward
                    + 0.25 * personalization_reward
                    + 0.15 * diversity_reward
                )

                # 6. 정규화 및 스무딩
                total_reward = torch.clamp(total_reward, 0.0, 1.0)

            return total_reward

        except Exception as e:
            logger.warning(f"⚠️ 보상 계산 실패: {e}, 기본값 반환")
            return torch.tensor([0.5] * batch_size, device=self.device)

    # 모든 개별 평가 메서드들은 개선된 평가기로 완전히 대체됨