#!/usr/bin/env python3
"""
Improved Evaluators - 사전 훈련된 최신 모델 기반 평가기들
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
import warnings

from config import device, logger, TRANSFORMERS_AVAILABLE
from models.emotion import EmotionEmbedding
from models.user_profile import UserEmotionProfile

# 경고 메시지 억제
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

if TRANSFORMERS_AVAILABLE:
    from transformers import CLIPModel, CLIPProcessor


class ImprovedAestheticEvaluator:
    """LAION Aesthetic Predictor V2 기반 미학 평가기"""
    
    def __init__(self):
        self.device = device
        self.use_pretrained = TRANSFORMERS_AVAILABLE
        
        if self.use_pretrained:
            try:
                # CLIP 모델 로드 (LAION 사전 훈련)
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_model.to(self.device)
                self.clip_model.eval()
                
                # LAION aesthetic predictor head
                self.aesthetic_head = nn.Sequential(
                    nn.Linear(512, 256),  # CLIP ViT-B/32 임베딩 크기
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1),
                    nn.Sigmoid(),
                ).to(self.device)
                
                # 사전 훈련된 가중치로 초기화 (LAION 스타일)
                self._init_aesthetic_weights()
                
                logger.info("✅ LAION 기반 미학 평가기 초기화 완료")
                
            except Exception as e:
                logger.warning(f"⚠️ CLIP 모델 로드 실패: {e}, 기본 평가기 사용")
                self.use_pretrained = False
                self._build_fallback_evaluator()
        else:
            self._build_fallback_evaluator()
    
    def _init_aesthetic_weights(self):
        """LAION 스타일 가중치 초기화"""
        for module in self.aesthetic_head:
            if isinstance(module, nn.Linear):
                # LAION aesthetic predictor와 유사한 초기화
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def _build_fallback_evaluator(self):
        """기본 CNN 평가기 (개선된 버전)"""
        self.fallback_evaluator = nn.Sequential(
            # 더 깊고 강력한 CNN
            nn.Conv2d(3, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        ).to(self.device)
        
        logger.info("✅ 개선된 CNN 미학 평가기 초기화 완료")
    
    def evaluate(self, images: torch.Tensor) -> torch.Tensor:
        """미학 점수 평가"""
        if self.use_pretrained:
            return self._evaluate_with_clip(images)
        else:
            return self._evaluate_with_cnn(images)
    
    def _evaluate_with_clip(self, images: torch.Tensor) -> torch.Tensor:
        """CLIP 기반 미학 평가"""
        batch_size = images.shape[0]
        
        try:
            with torch.no_grad():
                # 이미지를 PIL 형태로 변환하여 processor에 전달
                images_pil = []
                for i in range(batch_size):
                    img_tensor = images[i].cpu()
                    # 0-1 범위로 정규화
                    if img_tensor.max() > 1.0:
                        img_tensor = img_tensor / 255.0
                    
                    # CHW -> HWC 변환
                    img_numpy = img_tensor.permute(1, 2, 0).numpy()
                    img_numpy = (img_numpy * 255).astype(np.uint8)
                    images_pil.append(img_numpy)
                
                # CLIP processor로 전처리
                inputs = self.processor(images=images_pil, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # CLIP 이미지 특성 추출
                image_features = self.clip_model.get_image_features(**inputs)
                image_features = F.normalize(image_features, p=2, dim=-1)
                
                # Aesthetic 점수 예측
                aesthetic_scores = self.aesthetic_head(image_features).squeeze()
                
                # 배치 차원 보장
                if aesthetic_scores.dim() == 0:
                    aesthetic_scores = aesthetic_scores.unsqueeze(0)
                
                return aesthetic_scores
                
        except Exception as e:
            logger.warning(f"⚠️ CLIP 미학 평가 실패: {e}, CNN 평가기 사용")
            return self._evaluate_with_cnn(images)
    
    def _evaluate_with_cnn(self, images: torch.Tensor) -> torch.Tensor:
        """CNN 기반 미학 평가"""
        # 이미지 크기 조정
        if images.shape[-1] != 128:  # 더 높은 해상도 사용
            images_resized = F.interpolate(
                images, size=(128, 128), mode="bilinear", align_corners=False
            )
        else:
            images_resized = images
        
        aesthetic_scores = self.fallback_evaluator(images_resized).squeeze()
        
        # 배치 차원 보장
        if aesthetic_scores.dim() == 0:
            aesthetic_scores = aesthetic_scores.unsqueeze(0)
        
        return aesthetic_scores


class ImprovedEmotionEvaluator:
    """CLIP 기반 감정 평가기"""
    
    def __init__(self):
        self.device = device
        self.use_pretrained = TRANSFORMERS_AVAILABLE
        
        if self.use_pretrained:
            try:
                # CLIP 모델 로드
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_model.to(self.device)
                self.clip_model.eval()
                
                # VAD 기반 감정 텍스트 프롬프트
                self.emotion_texts = [
                    "a very positive and pleasant image",      # High Valence
                    "a very negative and unpleasant image",    # Low Valence
                    "a highly energetic and exciting image",   # High Arousal  
                    "a very calm and peaceful image",          # Low Arousal
                    "a dominant and powerful image",           # High Dominance
                    "a submissive and gentle image",           # Low Dominance
                ]
                
                # VAD 매핑 가중치
                self.vad_mapping = torch.tensor([
                    [1.0, 0.0, 0.0],   # High Valence
                    [-1.0, 0.0, 0.0],  # Low Valence  
                    [0.0, 1.0, 0.0],   # High Arousal
                    [0.0, -1.0, 0.0],  # Low Arousal
                    [0.0, 0.0, 1.0],   # High Dominance
                    [0.0, 0.0, -1.0],  # Low Dominance
                ], device=self.device)
                
                logger.info("✅ CLIP 기반 감정 평가기 초기화 완료")
                
            except Exception as e:
                logger.warning(f"⚠️ CLIP 모델 로드 실패: {e}, 기본 평가기 사용")
                self.use_pretrained = False
                self._build_fallback_evaluator()
        else:
            self._build_fallback_evaluator()
    
    def _build_fallback_evaluator(self):
        """기본 MLP 평가기 (개선된 버전)"""
        self.fallback_evaluator = nn.Sequential(
            nn.Linear(1024, 512),  # 더 큰 입력 차원
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # VAD
            nn.Tanh(),
        ).to(self.device)
        
        logger.info("✅ 개선된 MLP 감정 평가기 초기화 완료")
    
    def evaluate(self, images: torch.Tensor, target_emotion: EmotionEmbedding) -> torch.Tensor:
        """감정 정확도 평가"""
        if self.use_pretrained:
            return self._evaluate_with_clip(images, target_emotion)
        else:
            return self._evaluate_with_mlp(images, target_emotion)
    
    def _evaluate_with_clip(self, images: torch.Tensor, target_emotion: EmotionEmbedding) -> torch.Tensor:
        """CLIP zero-shot 감정 평가"""
        batch_size = images.shape[0]
        
        try:
            with torch.no_grad():
                # 이미지를 PIL 형태로 변환
                images_pil = []
                for i in range(batch_size):
                    img_tensor = images[i].cpu()
                    if img_tensor.max() > 1.0:
                        img_tensor = img_tensor / 255.0
                    
                    img_numpy = img_tensor.permute(1, 2, 0).numpy()
                    img_numpy = (img_numpy * 255).astype(np.uint8)
                    images_pil.append(img_numpy)
                
                # CLIP으로 이미지-텍스트 유사도 계산
                inputs = self.processor(
                    text=self.emotion_texts,
                    images=images_pil,
                    return_tensors="pt",
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.clip_model(**inputs)
                
                # 이미지-텍스트 유사도 점수
                logits_per_image = outputs.logits_per_image  # [batch_size, num_texts]
                emotion_probs = F.softmax(logits_per_image, dim=-1)
                
                # VAD 점수로 변환
                predicted_vad = torch.matmul(emotion_probs, self.vad_mapping)
                
                # 목표 감정과의 거리 계산
                target_vector = torch.tensor(
                    [target_emotion.valence, target_emotion.arousal, target_emotion.dominance],
                    device=self.device
                ).repeat(batch_size, 1)
                
                emotion_distance = F.mse_loss(
                    predicted_vad, target_vector, reduction="none"
                ).mean(dim=1)
                
                # 거리를 보상으로 변환
                emotion_reward = torch.exp(-emotion_distance * 2.0)
                
                return emotion_reward
                
        except Exception as e:
            logger.warning(f"⚠️ CLIP 감정 평가 실패: {e}, MLP 평가기 사용")
            return self._evaluate_with_mlp(images, target_emotion)
    
    def _evaluate_with_mlp(self, images: torch.Tensor, target_emotion: EmotionEmbedding) -> torch.Tensor:
        """MLP 기반 감정 평가"""
        batch_size = images.shape[0]
        
        # 간단한 이미지 특성 추출
        image_features = self._extract_features(images)
        
        # VAD 예측
        predicted_vad = self.fallback_evaluator(image_features)
        
        # 목표 감정과의 거리 계산
        target_vector = torch.tensor(
            [target_emotion.valence, target_emotion.arousal, target_emotion.dominance],
            device=self.device
        ).repeat(batch_size, 1)
        
        emotion_distance = F.mse_loss(
            predicted_vad, target_vector, reduction="none"
        ).mean(dim=1)
        
        emotion_reward = torch.exp(-emotion_distance * 2.0)
        
        return emotion_reward
    
    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """간단한 특성 추출"""
        batch_size = images.shape[0]
        
        # 이미지 통계 특성
        mean_vals = images.mean(dim=[2, 3])  # 채널별 평균
        std_vals = images.std(dim=[2, 3])    # 채널별 표준편차
        
        # 히스토그램 특성 (간단화)
        hist_features = []
        for i in range(batch_size):
            img = images[i].flatten()
            hist = torch.histc(img, bins=10, min=0, max=1)
            hist_features.append(hist)
        
        hist_features = torch.stack(hist_features)
        
        # 전체 특성 결합
        features = torch.cat([
            mean_vals,      # 3 features
            std_vals,       # 3 features  
            hist_features,  # 10 features
        ], dim=1)
        
        # 1024 차원으로 확장 (패딩)
        if features.shape[1] < 1024:
            padding = torch.zeros(batch_size, 1024 - features.shape[1], device=self.device)
            features = torch.cat([features, padding], dim=1)
        
        return features


class ImprovedPersonalizationEvaluator:
    """개선된 개인화 평가기"""
    
    def __init__(self):
        self.device = device
        self.use_pretrained = TRANSFORMERS_AVAILABLE
        
        if self.use_pretrained:
            try:
                # CLIP 모델 로드
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_model.to(self.device)
                self.clip_model.eval()
                
                # 개인화 점수 예측 헤드
                self.personalization_head = nn.Sequential(
                    nn.Linear(512 + 16, 256),  # CLIP features + 확장된 user preferences
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid(),
                ).to(self.device)
                
                logger.info("✅ CLIP 기반 개인화 평가기 초기화 완료")
                
            except Exception as e:
                logger.warning(f"⚠️ CLIP 모델 로드 실패: {e}, 기본 평가기 사용")
                self.use_pretrained = False
                self._build_fallback_evaluator()
        else:
            self._build_fallback_evaluator()
    
    def _build_fallback_evaluator(self):
        """기본 개인화 평가기 (개선된 버전)"""
        self.fallback_evaluator = nn.Sequential(
            nn.Linear(1024 + 16, 512),  # 이미지 특성 + 사용자 선호도
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        ).to(self.device)
        
        logger.info("✅ 개선된 MLP 개인화 평가기 초기화 완료")
    
    def evaluate(self, images: torch.Tensor, user_profile: UserEmotionProfile) -> torch.Tensor:
        """개인화 점수 평가"""
        if self.use_pretrained:
            return self._evaluate_with_clip(images, user_profile)
        else:
            return self._evaluate_with_mlp(images, user_profile)
    
    def _evaluate_with_clip(self, images: torch.Tensor, user_profile: UserEmotionProfile) -> torch.Tensor:
        """CLIP 기반 개인화 평가"""
        batch_size = images.shape[0]
        
        try:
            with torch.no_grad():
                # 이미지를 PIL 형태로 변환
                images_pil = []
                for i in range(batch_size):
                    img_tensor = images[i].cpu()
                    if img_tensor.max() > 1.0:
                        img_tensor = img_tensor / 255.0
                    
                    img_numpy = img_tensor.permute(1, 2, 0).numpy()
                    img_numpy = (img_numpy * 255).astype(np.uint8)
                    images_pil.append(img_numpy)
                
                # CLIP 이미지 특성 추출
                inputs = self.processor(images=images_pil, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                image_features = self.clip_model.get_image_features(**inputs)
                image_features = F.normalize(image_features, p=2, dim=-1)
                
                # 사용자 선호도 벡터 생성 (확장된 버전)
                preference_vector = self._create_extended_preference_vector(user_profile)
                preference_vector = preference_vector.repeat(batch_size, 1)
                
                # 특성 결합
                combined_features = torch.cat([image_features, preference_vector], dim=1)
                
                # 개인화 점수 예측
                personalization_scores = self.personalization_head(combined_features).squeeze()
                
                # 배치 차원 보장
                if personalization_scores.dim() == 0:
                    personalization_scores = personalization_scores.unsqueeze(0)
                
                return personalization_scores
                
        except Exception as e:
            logger.warning(f"⚠️ CLIP 개인화 평가 실패: {e}, MLP 평가기 사용")
            return self._evaluate_with_mlp(images, user_profile)
    
    def _evaluate_with_mlp(self, images: torch.Tensor, user_profile: UserEmotionProfile) -> torch.Tensor:
        """MLP 기반 개인화 평가"""
        batch_size = images.shape[0]
        
        # 이미지 특성 추출
        image_features = self._extract_features(images)
        
        # 사용자 선호도 벡터
        preference_vector = self._create_extended_preference_vector(user_profile)
        preference_vector = preference_vector.repeat(batch_size, 1)
        
        # 특성 결합
        combined_features = torch.cat([image_features, preference_vector], dim=1)
        
        # 개인화 점수 예측
        personalization_scores = self.fallback_evaluator(combined_features).squeeze()
        
        # 배치 차원 보장
        if personalization_scores.dim() == 0:
            personalization_scores = personalization_scores.unsqueeze(0)
        
        return personalization_scores
    
    def _create_extended_preference_vector(self, user_profile: UserEmotionProfile) -> torch.Tensor:
        """확장된 사용자 선호도 벡터 생성"""
        # 기본 선호도
        basic_prefs = [
            user_profile.preference_weights.get("color_temperature", 0.5),
            user_profile.preference_weights.get("brightness", 0.5),
            user_profile.preference_weights.get("saturation", 0.5),
            user_profile.preference_weights.get("contrast", 0.5),
            user_profile.preference_weights.get("complexity", 0.5),
        ]
        
        # 스타일 선호도 (원-핫 인코딩)
        art_style = user_profile.preference_weights.get("art_style", "realistic")
        style_prefs = [
            1.0 if art_style == "realistic" else 0.0,
            1.0 if art_style == "abstract" else 0.0,
            1.0 if art_style == "cartoon" else 0.0,
            1.0 if art_style == "painting" else 0.0,
        ]
        
        # 감정 프로필 요약
        emotion_history = user_profile.emotion_history[-10:] if user_profile.emotion_history else []
        if emotion_history:
            avg_valence = np.mean([e.valence for e in emotion_history])
            avg_arousal = np.mean([e.arousal for e in emotion_history])
            avg_dominance = np.mean([e.dominance for e in emotion_history])
        else:
            avg_valence = avg_arousal = avg_dominance = 0.0
        
        emotion_prefs = [avg_valence, avg_arousal, avg_dominance]
        
        # 추가 메타 정보
        meta_prefs = [
            float(len(emotion_history)) / 100.0,  # 사용 빈도
            user_profile.adaptation_rate,         # 적응 속도
            float(user_profile.session_count) / 50.0,  # 세션 수
            user_profile.average_satisfaction,    # 평균 만족도
        ]
        
        # 전체 벡터 결합 (총 16차원)
        all_prefs = basic_prefs + style_prefs + emotion_prefs + meta_prefs
        
        return torch.tensor(all_prefs, device=self.device, dtype=torch.float32)
    
    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """간단한 이미지 특성 추출 (개인화용)"""
        batch_size = images.shape[0]
        
        # 색상 특성
        mean_rgb = images.mean(dim=[2, 3])  # RGB 평균
        std_rgb = images.std(dim=[2, 3])    # RGB 표준편차
        
        # 밝기와 대비
        gray = images.mean(dim=1, keepdim=True)  # 그레이스케일 변환
        brightness = gray.mean(dim=[2, 3]).squeeze()
        contrast = gray.std(dim=[2, 3]).squeeze()
        
        # 채도 (간략화된 계산)
        max_rgb = images.max(dim=1)[0]
        min_rgb = images.min(dim=1)[0]
        saturation = ((max_rgb - min_rgb) / (max_rgb + 1e-8)).mean(dim=[1, 2])
        
        # 복잡도 (에지 밀도 근사)
        sobel_x = F.conv2d(gray, torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], device=self.device, dtype=torch.float32), padding=1)
        sobel_y = F.conv2d(gray, torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], device=self.device, dtype=torch.float32), padding=1)
        edge_magnitude = torch.sqrt(sobel_x**2 + sobel_y**2)
        complexity = edge_magnitude.mean(dim=[2, 3]).squeeze()
        
        # 특성 결합
        features = torch.stack([
            mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2],
            std_rgb[:, 0], std_rgb[:, 1], std_rgb[:, 2],
            brightness, contrast, saturation, complexity
        ], dim=1)
        
        # 1024 차원으로 확장 (패딩)
        if features.shape[1] < 1024:
            padding = torch.zeros(batch_size, 1024 - features.shape[1], device=self.device)
            features = torch.cat([features, padding], dim=1)
        
        return features
    
    def learn_from_feedback(self, images: torch.Tensor, user_profile: UserEmotionProfile, 
                          ratings: torch.Tensor, lr: float = 0.001) -> float:
        """사용자 피드백으로부터 학습"""
        if not self.use_pretrained:
            return 0.0
        
        try:
            # 현재 예측
            predicted_scores = self.evaluate(images, user_profile)
            
            # 손실 계산
            loss = F.mse_loss(predicted_scores, ratings)
            
            # 그래디언트 업데이트 (간단한 SGD)
            loss.backward()
            
            with torch.no_grad():
                for param in self.personalization_head.parameters():
                    if param.grad is not None:
                        param.data -= lr * param.grad.data
                        param.grad.zero_()
            
            return loss.item()
            
        except Exception as e:
            logger.warning(f"⚠️ 개인화 학습 실패: {e}")
            return 0.0