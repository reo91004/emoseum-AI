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

        # 감정 정확도 평가기
        self.emotion_evaluator = self._build_emotion_evaluator()

        # 미적 품질 평가기
        self.aesthetic_evaluator = self._build_aesthetic_evaluator()

        # 개인화 점수 평가기
        self.personalization_evaluator = self._build_personalization_evaluator()

        # 다양성 평가기
        self.diversity_evaluator = self._build_diversity_evaluator()

        logger.info("✅ DRaFT+ 보상 모델 초기화 완료")

    def _build_emotion_evaluator(self) -> nn.Module:
        """감정 정확도 평가기"""
        return nn.Sequential(
            nn.Linear(768, 512),  # CLIP 임베딩 크기 가정
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 3),  # VAD 예측
            nn.Tanh(),
        ).to(self.device)

    def _build_aesthetic_evaluator(self) -> nn.Module:
        """미적 품질 평가기"""
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        ).to(self.device)

    def _build_personalization_evaluator(self) -> nn.Module:
        """개인화 점수 평가기"""
        return nn.Sequential(
            nn.Linear(512 + 7, 256),  # 이미지 특성 + 개인화 선호도
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        ).to(self.device)

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
                # 1. 감정 정확도 보상
                emotion_reward = self._calculate_emotion_reward(
                    generated_images, target_emotion
                )

                # 2. 미적 품질 보상
                aesthetic_reward = self._calculate_aesthetic_reward(generated_images)

                # 3. 개인화 보상
                personalization_reward = self._calculate_personalization_reward(
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

    def _calculate_emotion_reward(
        self, images: torch.Tensor, target_emotion: EmotionEmbedding
    ) -> torch.Tensor:
        """감정 정확도 기반 보상"""
        batch_size = images.shape[0]

        # 간단한 이미지 특성 추출 (실제로는 CLIP 등 사용)
        image_features = self._extract_simple_features(images)

        # 목표 감정과의 일치도 계산
        target_vector = torch.tensor(
            [target_emotion.valence, target_emotion.arousal, target_emotion.dominance],
            device=self.device,
        ).repeat(batch_size, 1)

        predicted_emotions = self.emotion_evaluator(image_features)
        emotion_distance = F.mse_loss(
            predicted_emotions, target_vector, reduction="none"
        ).mean(dim=1)

        # 거리를 보상으로 변환 (거리가 클수록 보상 낮음)
        emotion_reward = torch.exp(-emotion_distance * 2.0)

        return emotion_reward

    def _calculate_aesthetic_reward(self, images: torch.Tensor) -> torch.Tensor:
        """미적 품질 보상"""
        # 이미지 크기 조정 (필요시)
        if images.shape[-1] != 64:  # 예시 크기
            images_resized = F.interpolate(
                images, size=(64, 64), mode="bilinear", align_corners=False
            )
        else:
            images_resized = images

        aesthetic_scores = self.aesthetic_evaluator(images_resized).squeeze()

        # 배치 차원 보장
        if aesthetic_scores.dim() == 0:
            aesthetic_scores = aesthetic_scores.unsqueeze(0)

        return aesthetic_scores

    def _calculate_personalization_reward(
        self, images: torch.Tensor, user_profile: UserEmotionProfile
    ) -> torch.Tensor:
        """개인화 보상"""
        batch_size = images.shape[0]

        # 이미지 특성 추출
        image_features = self._extract_simple_features(images)

        # 사용자 선호도 벡터 생성
        preference_vector = torch.tensor(
            [
                user_profile.preference_weights["color_temperature"],
                user_profile.preference_weights["brightness"],
                user_profile.preference_weights["saturation"],
                user_profile.preference_weights["contrast"],
                user_profile.preference_weights["complexity"],
                (
                    1.0
                    if user_profile.preference_weights["art_style"] == "realistic"
                    else 0.0
                ),
                (
                    1.0
                    if user_profile.preference_weights["composition"] == "balanced"
                    else 0.0
                ),
            ],
            device=self.device,
        ).repeat(batch_size, 1)

        # 개인화 특성과 결합
        combined_features = torch.cat([image_features, preference_vector], dim=1)
        personalization_scores = self.personalization_evaluator(
            combined_features
        ).squeeze()

        if personalization_scores.dim() == 0:
            personalization_scores = personalization_scores.unsqueeze(0)

        return personalization_scores

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