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


class DRaFTPlusRewardModel:
    """DRaFT+ 방식의 개선된 보상 모델"""

    def __init__(self, device: torch.device = None):
        self.device = device if device else torch.device("cpu")

        # 개선된 평가기들
        self.aesthetic_evaluator = ImprovedAestheticEvaluator()
        self.emotion_evaluator = ImprovedEmotionEvaluator()
        self.personalization_evaluator = ImprovedPersonalizationEvaluator()

        # 다양성 평가기
        self.diversity_evaluator = self._build_diversity_evaluator()

        logger.info("✅ 개선된 DRaFT+ 보상 모델 초기화 완료")

    # 기존 단순한 평가기 빌더들은 개선된 평가기로 대체됨

    def _build_diversity_evaluator(self) -> nn.Module:
        """다양성 평가기"""
        return nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        ).to(self.device)

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

                # 4. 다양성 보상 (DRaFT+ 추가 요소)
                diversity_reward = self._calculate_diversity_reward(
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

    # 기존 개별 평가 메서드들은 개선된 평가기로 대체됨

    def _calculate_diversity_reward(
        self, images: torch.Tensor, previous_images: List[torch.Tensor] = None
    ) -> torch.Tensor:
        """다양성 보상 (DRaFT+ 핵심 요소)"""
        batch_size = images.shape[0]

        if previous_images is None or len(previous_images) == 0:
            # 이전 이미지가 없으면 최대 다양성 보상
            return torch.ones(batch_size, device=self.device)

        # 현재 이미지 특성 추출
        current_features = self._extract_simple_features(images)

        # 이전 이미지들과의 거리 계산
        diversity_scores = []

        for img_features in current_features:
            min_distance = float("inf")

            for prev_img in previous_images[-5:]:  # 최근 5개 이미지와 비교
                if prev_img.shape[0] == 1:  # 배치 크기 1인 경우
                    prev_features = self._extract_simple_features(prev_img).squeeze(0)
                    distance = F.pairwise_distance(
                        img_features.unsqueeze(0), prev_features.unsqueeze(0)
                    )
                    min_distance = min(min_distance, distance.item())

            # 거리 기반 다양성 점수 (거리가 클수록 다양성 높음)
            diversity_score = min(1.0, min_distance / 10.0)  # 정규화
            diversity_scores.append(diversity_score)

        return torch.tensor(diversity_scores, device=self.device)

    def _extract_simple_features(self, images: torch.Tensor) -> torch.Tensor:
        """간단한 이미지 특성 추출"""
        batch_size = images.shape[0]

        # 기본적인 통계적 특성들
        features = []

        for i in range(batch_size):
            img = images[i]

            # 색상 통계
            mean_rgb = img.mean(dim=[1, 2])  # RGB 평균
            std_rgb = img.std(dim=[1, 2])  # RGB 표준편차

            # 밝기 및 대비
            gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
            brightness = gray.mean()
            contrast = gray.std()

            # 에지 밀도 (간단한 근사)
            grad_x = torch.abs(gray[1:, :] - gray[:-1, :]).mean()
            grad_y = torch.abs(gray[:, 1:] - gray[:, :-1]).mean()
            edge_density = (grad_x + grad_y) / 2

            # 특성 벡터 구성
            feature_vector = torch.cat(
                [
                    mean_rgb,
                    std_rgb,
                    brightness.unsqueeze(0),
                    contrast.unsqueeze(0),
                    edge_density.unsqueeze(0),
                ]
            )

            # 512차원으로 패딩 (실제로는 더 정교한 특성 추출 필요)
            if feature_vector.shape[0] < 512:
                padding = torch.zeros(512 - feature_vector.shape[0], device=self.device)
                feature_vector = torch.cat([feature_vector, padding])

            features.append(feature_vector[:512])

        return torch.stack(features)