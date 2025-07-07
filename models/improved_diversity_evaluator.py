#!/usr/bin/env python3
"""
Improved Diversity Evaluator - CLIP 기반 의미론적 다양성 평가기
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
import warnings

from config import device, logger, TRANSFORMERS_AVAILABLE

# 경고 메시지 억제
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

if TRANSFORMERS_AVAILABLE:
    from transformers import CLIPModel, CLIPProcessor


class ImprovedDiversityEvaluator:
    """CLIP 기반 의미론적 다양성 평가기"""
    
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
                
                logger.info("✅ CLIP 기반 다양성 평가기 초기화 완료")
                
            except Exception as e:
                logger.warning(f"⚠️ CLIP 모델 로드 실패: {e}, 기본 평가기 사용")
                self.use_pretrained = False
                self._build_fallback_evaluator()
        else:
            self._build_fallback_evaluator()
    
    def _build_fallback_evaluator(self):
        """기본 다양성 평가기 (개선된 버전)"""
        self.fallback_evaluator = nn.Sequential(
            nn.Linear(1024, 512),
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
        
        logger.info("✅ 개선된 CNN 다양성 평가기 초기화 완료")
    
    def evaluate(self, images: torch.Tensor, previous_images: List[torch.Tensor] = None) -> torch.Tensor:
        """다양성 점수 평가"""
        if self.use_pretrained:
            return self._evaluate_with_clip(images, previous_images)
        else:
            return self._evaluate_with_cnn(images, previous_images)
    
    def _evaluate_with_clip(self, images: torch.Tensor, previous_images: List[torch.Tensor] = None) -> torch.Tensor:
        """CLIP 기반 의미론적 다양성 평가"""
        batch_size = images.shape[0]
        
        if not previous_images:
            # 이전 이미지가 없으면 최대 다양성 점수
            return torch.ones(batch_size, device=self.device)
        
        try:
            with torch.no_grad():
                # 현재 이미지들의 CLIP 특성 추출
                current_features = self._extract_clip_features(images)
                
                # 이전 이미지들의 CLIP 특성 추출
                all_prev_features = []
                for prev_img_batch in previous_images:
                    if prev_img_batch.numel() > 0:  # 빈 텐서 체크
                        prev_features = self._extract_clip_features(prev_img_batch)
                        all_prev_features.append(prev_features)
                
                if not all_prev_features:
                    return torch.ones(batch_size, device=self.device)
                
                # 모든 이전 특성들을 결합
                prev_features_combined = torch.cat(all_prev_features, dim=0)
                
                # 다양성 점수 계산
                diversity_scores = []
                for i in range(batch_size):
                    current_feat = current_features[i:i+1]  # [1, 512]
                    
                    # 현재 이미지와 모든 이전 이미지들 간의 코사인 유사도
                    similarities = F.cosine_similarity(
                        current_feat, prev_features_combined, dim=1
                    )
                    
                    # 최대 유사도 (가장 유사한 이전 이미지와의 유사도)
                    max_similarity = similarities.max()
                    
                    # 다양성 = 1 - 최대유사도 (유사도가 낮을수록 다양성 높음)
                    diversity = 1.0 - max_similarity
                    
                    # 추가 보정: 배치 내 다양성도 고려
                    if batch_size > 1:
                        # 배치 내 다른 이미지들과의 유사도
                        other_indices = [j for j in range(batch_size) if j != i]
                        if other_indices:
                            other_features = current_features[other_indices]
                            intra_similarities = F.cosine_similarity(
                                current_feat, other_features, dim=1
                            )
                            avg_intra_similarity = intra_similarities.mean()
                            intra_diversity = 1.0 - avg_intra_similarity
                            
                            # 전체 다양성 = 0.7 * 이전대비 + 0.3 * 배치내
                            diversity = 0.7 * diversity + 0.3 * intra_diversity
                    
                    diversity_scores.append(diversity)
                
                return torch.stack(diversity_scores)
                
        except Exception as e:
            logger.warning(f"⚠️ CLIP 다양성 평가 실패: {e}, CNN 평가기 사용")
            return self._evaluate_with_cnn(images, previous_images)
    
    def _extract_clip_features(self, images: torch.Tensor) -> torch.Tensor:
        """CLIP 특성 추출"""
        batch_size = images.shape[0]
        
        # 이미지를 PIL 형태로 변환
        images_pil = []
        for i in range(batch_size):
            img_tensor = images[i].cpu()
            if img_tensor.max() > 1.0:
                img_tensor = img_tensor / 255.0
            
            img_numpy = img_tensor.permute(1, 2, 0).numpy()
            img_numpy = (img_numpy * 255).astype(np.uint8)
            images_pil.append(img_numpy)
        
        # CLIP processor로 전처리
        inputs = self.processor(images=images_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # CLIP 이미지 특성 추출
        image_features = self.clip_model.get_image_features(**inputs)
        image_features = F.normalize(image_features, p=2, dim=-1)
        
        return image_features
    
    def _evaluate_with_cnn(self, images: torch.Tensor, previous_images: List[torch.Tensor] = None) -> torch.Tensor:
        """CNN 기반 다양성 평가"""
        batch_size = images.shape[0]
        
        if not previous_images:
            return torch.ones(batch_size, device=self.device)
        
        # 현재 이미지들의 특성 추출
        current_features = self._extract_advanced_features(images)
        
        # 이전 이미지들의 특성 추출
        all_prev_features = []
        for prev_img_batch in previous_images:
            if prev_img_batch.numel() > 0:
                prev_features = self._extract_advanced_features(prev_img_batch)
                all_prev_features.append(prev_features)
        
        if not all_prev_features:
            return torch.ones(batch_size, device=self.device)
        
        prev_features_combined = torch.cat(all_prev_features, dim=0)
        
        # 유클리드 거리 기반 다양성 계산
        diversity_scores = []
        for i in range(batch_size):
            current_feat = current_features[i:i+1]
            
            # L2 거리 계산
            distances = torch.norm(
                current_feat - prev_features_combined, p=2, dim=1
            )
            
            # 최소 거리 (가장 유사한 이미지와의 거리)
            min_distance = distances.min()
            
            # 거리를 0-1 범위로 정규화 (경험적 스케일링)
            diversity = torch.tanh(min_distance / 10.0)
            diversity_scores.append(diversity)
        
        return torch.stack(diversity_scores)
    
    def _extract_advanced_features(self, images: torch.Tensor) -> torch.Tensor:
        """고급 이미지 특성 추출 (개선된 버전)"""
        batch_size = images.shape[0]
        
        # 기본 색상 특성
        mean_rgb = images.mean(dim=[2, 3])  # RGB 평균
        std_rgb = images.std(dim=[2, 3])    # RGB 표준편차
        
        # 밝기와 대비
        gray = images.mean(dim=1, keepdim=True)  # 그레이스케일 변환
        brightness = gray.mean(dim=[2, 3]).squeeze()
        contrast = gray.std(dim=[2, 3]).squeeze()
        
        # 채도 계산
        max_rgb = images.max(dim=1)[0]
        min_rgb = images.min(dim=1)[0]
        saturation = ((max_rgb - min_rgb) / (max_rgb + 1e-8)).mean(dim=[1, 2])
        
        # 복잡도 (에지 밀도)
        sobel_x = F.conv2d(
            gray, 
            torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], 
                        device=self.device, dtype=torch.float32), 
            padding=1
        )
        sobel_y = F.conv2d(
            gray, 
            torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], 
                        device=self.device, dtype=torch.float32), 
            padding=1
        )
        edge_magnitude = torch.sqrt(sobel_x**2 + sobel_y**2)
        complexity = edge_magnitude.mean(dim=[2, 3]).squeeze()
        
        # 텍스처 특성 (Local Binary Pattern 근사)
        texture_features = []
        for i in range(batch_size):
            img_gray = gray[i, 0]  # [H, W]
            
            # 간단한 LBP 근사 (8방향)
            center = img_gray[1:-1, 1:-1]
            neighbors = [
                img_gray[:-2, :-2],   # 좌상
                img_gray[:-2, 1:-1],  # 상
                img_gray[:-2, 2:],    # 우상
                img_gray[1:-1, 2:],   # 우
                img_gray[2:, 2:],     # 우하
                img_gray[2:, 1:-1],   # 하
                img_gray[2:, :-2],    # 좌하
                img_gray[1:-1, :-2],  # 좌
            ]
            
            # 이진 패턴 계산
            lbp_sum = 0
            for j, neighbor in enumerate(neighbors):
                lbp_sum += (neighbor > center).float() * (2 ** j)
            
            # 텍스처 히스토그램
            texture_hist = torch.histc(lbp_sum.flatten(), bins=16, min=0, max=255)
            texture_hist = texture_hist / (texture_hist.sum() + 1e-8)  # 정규화
            texture_features.append(texture_hist)
        
        texture_features = torch.stack(texture_features)
        
        # 색상 히스토그램 (개선된 버전)
        color_hist_features = []
        for i in range(batch_size):
            img = images[i]
            hist_r = torch.histc(img[0].flatten(), bins=8, min=0, max=1)
            hist_g = torch.histc(img[1].flatten(), bins=8, min=0, max=1)
            hist_b = torch.histc(img[2].flatten(), bins=8, min=0, max=1)
            
            # 정규화
            hist_r = hist_r / (hist_r.sum() + 1e-8)
            hist_g = hist_g / (hist_g.sum() + 1e-8)
            hist_b = hist_b / (hist_b.sum() + 1e-8)
            
            color_hist = torch.cat([hist_r, hist_g, hist_b])  # 24차원
            color_hist_features.append(color_hist)
        
        color_hist_features = torch.stack(color_hist_features)
        
        # 모든 특성 결합
        basic_features = torch.stack([
            mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2],  # 3
            std_rgb[:, 0], std_rgb[:, 1], std_rgb[:, 2],     # 3
            brightness, contrast, saturation, complexity      # 4
        ], dim=1)  # [batch_size, 10]
        
        # 전체 특성 벡터: 기본(10) + 텍스처(16) + 색상히스토그램(24) = 50차원
        features = torch.cat([
            basic_features,      # 10차원
            texture_features,    # 16차원
            color_hist_features, # 24차원
        ], dim=1)  # [batch_size, 50]
        
        # 1024차원으로 확장 (패딩)
        if features.shape[1] < 1024:
            padding = torch.zeros(batch_size, 1024 - features.shape[1], device=self.device)
            features = torch.cat([features, padding], dim=1)
        
        return features
    
    def calculate_semantic_diversity(self, images: torch.Tensor, 
                                   semantic_concepts: List[str] = None) -> torch.Tensor:
        """의미론적 개념 기반 다양성 계산"""
        if not self.use_pretrained or not semantic_concepts:
            return self.evaluate(images)
        
        try:
            with torch.no_grad():
                batch_size = images.shape[0]
                
                # 이미지 특성 추출
                image_features = self._extract_clip_features(images)
                
                # 의미론적 개념들의 텍스트 임베딩
                concept_inputs = self.processor(
                    text=semantic_concepts, return_tensors="pt", padding=True
                )
                concept_inputs = {k: v.to(self.device) for k, v in concept_inputs.items()}
                concept_features = self.clip_model.get_text_features(**concept_inputs)
                concept_features = F.normalize(concept_features, p=2, dim=-1)
                
                # 각 이미지가 각 개념과 얼마나 유사한지 계산
                diversity_scores = []
                for i in range(batch_size):
                    img_feat = image_features[i:i+1]  # [1, 512]
                    
                    # 이미지와 모든 개념 간 유사도
                    concept_similarities = F.cosine_similarity(
                        img_feat, concept_features, dim=1
                    )
                    
                    # 가장 높은 유사도 (주요 개념)
                    max_concept_sim = concept_similarities.max()
                    
                    # 개념 분포의 엔트로피 (다양성 측정)
                    concept_probs = F.softmax(concept_similarities * 10, dim=0)  # 온도 스케일링
                    entropy = -(concept_probs * torch.log(concept_probs + 1e-8)).sum()
                    
                    # 정규화된 엔트로피 (0-1 범위)
                    max_entropy = torch.log(torch.tensor(len(semantic_concepts), dtype=torch.float32))
                    normalized_entropy = entropy / max_entropy
                    
                    diversity_scores.append(normalized_entropy)
                
                return torch.stack(diversity_scores)
                
        except Exception as e:
            logger.warning(f"⚠️ 의미론적 다양성 계산 실패: {e}")
            return self.evaluate(images)