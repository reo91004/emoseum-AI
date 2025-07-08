#!/usr/bin/env python3
"""
적응형 개인화 시스템 - 머신러닝 기반 사용자 선호도 학습
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
from datetime import datetime
import warnings

from config import device, logger
from models.emotion import EmotionEmbedding

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class FeedbackContext:
    """피드백 컨텍스트 정보"""
    emotion: EmotionEmbedding
    image_features: Dict[str, float]  # 이미지 특성 (밝기, 채도, 대비 등)
    prompt_features: Dict[str, Any]   # 프롬프트 특성
    user_state: Dict[str, float]      # 사용자 상태 (시간대, 기분 이력 등)
    feedback_score: float
    timestamp: datetime


class AdaptivePreferenceNet(nn.Module):
    """적응형 선호도 예측 신경망"""
    
    def __init__(
        self,
        emotion_dim: int = 3,        # VAD 차원
        image_feature_dim: int = 8,  # 이미지 특성 차원
        prompt_feature_dim: int = 10, # 프롬프트 특성 차원
        user_state_dim: int = 5,     # 사용자 상태 차원
        hidden_dim: int = 128,
        preference_dim: int = 7      # 선호도 차원 수
    ):
        super().__init__()
        
        self.emotion_dim = emotion_dim
        self.image_feature_dim = image_feature_dim
        self.prompt_feature_dim = prompt_feature_dim
        self.user_state_dim = user_state_dim
        self.preference_dim = preference_dim
        
        # 입력 차원 계산
        input_dim = emotion_dim + image_feature_dim + prompt_feature_dim + user_state_dim
        
        # 신경망 구조
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 선호도 예측 헤드
        self.preference_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, preference_dim),
            nn.Tanh()  # [-1, 1] 범위로 정규화
        )
        
        # 만족도 예측 헤드 (피드백 점수 예측)
        self.satisfaction_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # [0, 1] 범위
        )
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """가중치 초기화"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, context_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            context_features: [batch_size, input_dim] 컨텍스트 특성
            
        Returns:
            preferences: [batch_size, preference_dim] 예측된 선호도
            satisfaction: [batch_size, 1] 예측된 만족도
        """
        features = self.feature_encoder(context_features)
        
        preferences = self.preference_head(features)
        satisfaction = self.satisfaction_head(features)
        
        return preferences, satisfaction


class ContextualBandit:
    """컨텍스추얼 밴딧 기반 선호도 탐색"""
    
    def __init__(self, preference_dim: int = 7, exploration_rate: float = 0.1):
        self.preference_dim = preference_dim
        self.exploration_rate = exploration_rate
        self.action_counts = np.zeros(preference_dim)
        self.action_rewards = np.zeros(preference_dim)
        
    def select_exploration_direction(self, current_preferences: np.ndarray) -> np.ndarray:
        """탐색 방향 선택 (UCB 기반)"""
        
        total_counts = np.sum(self.action_counts) + 1
        
        # UCB (Upper Confidence Bound) 계산
        confidence_intervals = np.sqrt(
            2 * np.log(total_counts) / (self.action_counts + 1)
        )
        
        # 평균 보상 + 신뢰도 구간
        ucb_values = (self.action_rewards / (self.action_counts + 1)) + confidence_intervals
        
        # 가장 높은 UCB 값을 가진 차원 선택
        explore_dim = np.argmax(ucb_values)
        
        # 탐색 방향 생성
        exploration_vector = np.zeros(self.preference_dim)
        exploration_vector[explore_dim] = np.random.choice([-1, 1]) * self.exploration_rate
        
        return current_preferences + exploration_vector
    
    def update_rewards(self, action_dim: int, reward: float):
        """보상 업데이트"""
        self.action_counts[action_dim] += 1
        self.action_rewards[action_dim] += reward


class AdaptivePersonalizationSystem:
    """적응형 개인화 시스템"""
    
    def __init__(
        self,
        learning_rate: float = 1e-3,
        update_interval: int = 5,  # 몇 번의 피드백마다 모델 업데이트
        exploration_rate: float = 0.1
    ):
        self.learning_rate = learning_rate
        self.update_interval = update_interval
        self.exploration_rate = exploration_rate
        
        # 적응형 선호도 신경망
        self.preference_net = AdaptivePreferenceNet().to(device)
        self.optimizer = optim.Adam(self.preference_net.parameters(), lr=learning_rate)
        
        # 컨텍스추얼 밴딧
        self.bandit = ContextualBandit(exploration_rate=exploration_rate)
        
        # 학습 데이터 버퍼
        self.feedback_buffer: List[FeedbackContext] = []
        self.max_buffer_size = 1000
        
        # 개인별 선호도 저장
        self.user_preferences: Dict[str, np.ndarray] = {}
        self.user_contexts: Dict[str, List[FeedbackContext]] = {}
        
        # 학습 메트릭
        self.training_history = {
            "losses": [],
            "accuracies": [],
            "exploration_rates": []
        }
        
        # 선호도 차원 정의
        self.preference_names = [
            "color_temperature", "brightness", "saturation", 
            "contrast", "complexity", "artistic_style", "composition"
        ]
        
        logger.info("✅ 적응형 개인화 시스템 초기화 완료")
    
    def extract_context_features(
        self, 
        emotion: EmotionEmbedding,
        image_metadata: Dict[str, Any],
        prompt: str,
        user_id: str,
        timestamp: datetime = None
    ) -> Dict[str, float]:
        """컨텍스트 특성 추출"""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # 1. 감정 특성
        emotion_features = {
            "valence": emotion.valence,
            "arousal": emotion.arousal,
            "dominance": emotion.dominance
        }
        
        # 2. 이미지 특성 (메타데이터에서 추출)
        image_features = {
            "brightness": image_metadata.get("brightness", 0.0),
            "saturation": image_metadata.get("saturation", 0.0),
            "contrast": image_metadata.get("contrast", 0.0),
            "hue_variance": image_metadata.get("hue_variance", 0.0),
            "edge_density": image_metadata.get("edge_density", 0.0),
            "color_diversity": image_metadata.get("color_diversity", 0.0),
            "composition_balance": image_metadata.get("composition_balance", 0.0),
            "texture_complexity": image_metadata.get("texture_complexity", 0.0)
        }
        
        # 3. 프롬프트 특성
        prompt_features = {
            "length": len(prompt.split()) / 50.0,  # 정규화
            "positive_words": self._count_positive_words(prompt),
            "negative_words": self._count_negative_words(prompt),
            "color_mentions": self._count_color_mentions(prompt),
            "emotion_words": self._count_emotion_words(prompt),
            "art_style_mentions": self._count_art_style_mentions(prompt),
            "complexity_indicators": self._count_complexity_indicators(prompt),
            "nature_elements": self._count_nature_elements(prompt),
            "human_elements": self._count_human_elements(prompt),
            "abstract_elements": self._count_abstract_elements(prompt)
        }
        
        # 4. 사용자 상태 특성
        hour = timestamp.hour
        user_state_features = {
            "time_of_day": np.sin(2 * np.pi * hour / 24),  # 시간을 순환 특성으로
            "time_of_day_cos": np.cos(2 * np.pi * hour / 24),
            "recent_mood_trend": self._calculate_mood_trend(user_id),
            "session_length": self._get_session_length(user_id),
            "interaction_frequency": self._get_interaction_frequency(user_id)
        }
        
        return {
            **emotion_features,
            **image_features, 
            **prompt_features,
            **user_state_features
        }
    
    def _count_positive_words(self, prompt: str) -> float:
        """긍정적 단어 개수"""
        positive_words = {
            "beautiful", "bright", "cheerful", "happy", "joyful", "peaceful",
            "vibrant", "warm", "wonderful", "amazing", "stunning", "lovely"
        }
        words = prompt.lower().split()
        return sum(1 for word in words if word in positive_words) / len(words)
    
    def _count_negative_words(self, prompt: str) -> float:
        """부정적 단어 개수"""
        negative_words = {
            "dark", "sad", "gloomy", "depressing", "cold", "harsh",
            "ugly", "terrible", "awful", "nightmare", "scary", "disturbing"
        }
        words = prompt.lower().split()
        return sum(1 for word in words if word in negative_words) / len(words)
    
    def _count_color_mentions(self, prompt: str) -> float:
        """색상 언급 개수"""
        color_words = {
            "red", "blue", "green", "yellow", "purple", "orange", "pink",
            "black", "white", "brown", "gray", "violet", "cyan", "magenta"
        }
        words = prompt.lower().split()
        return sum(1 for word in words if word in color_words) / len(words)
    
    def _count_emotion_words(self, prompt: str) -> float:
        """감정 단어 개수"""
        emotion_words = {
            "calm", "excited", "relaxed", "energetic", "serene", "dynamic",
            "peaceful", "lively", "tranquil", "passionate", "soothing", "intense"
        }
        words = prompt.lower().split()
        return sum(1 for word in words if word in emotion_words) / len(words)
    
    def _count_art_style_mentions(self, prompt: str) -> float:
        """예술 스타일 언급 개수"""
        art_styles = {
            "realistic", "abstract", "impressionist", "minimalist", "surreal",
            "watercolor", "oil painting", "digital art", "sketch", "vintage"
        }
        words = prompt.lower().split()
        return sum(1 for word in words if word in art_styles) / len(words)
    
    def _count_complexity_indicators(self, prompt: str) -> float:
        """복잡성 지시어 개수"""
        complexity_words = {
            "detailed", "intricate", "complex", "elaborate", "simple", 
            "minimalist", "clean", "busy", "cluttered", "sparse"
        }
        words = prompt.lower().split()
        return sum(1 for word in words if word in complexity_words) / len(words)
    
    def _count_nature_elements(self, prompt: str) -> float:
        """자연 요소 개수"""
        nature_words = {
            "forest", "mountain", "ocean", "sky", "tree", "flower",
            "landscape", "nature", "garden", "beach", "sunset", "sunrise"
        }
        words = prompt.lower().split()
        return sum(1 for word in words if word in nature_words) / len(words)
    
    def _count_human_elements(self, prompt: str) -> float:
        """인간 요소 개수"""
        human_words = {
            "person", "people", "human", "face", "portrait", "family",
            "child", "woman", "man", "smile", "emotion", "expression"
        }
        words = prompt.lower().split()
        return sum(1 for word in words if word in human_words) / len(words)
    
    def _count_abstract_elements(self, prompt: str) -> float:
        """추상적 요소 개수"""
        abstract_words = {
            "abstract", "geometric", "pattern", "shape", "form", "concept",
            "idea", "emotion", "feeling", "mood", "atmosphere", "energy"
        }
        words = prompt.lower().split()
        return sum(1 for word in words if word in abstract_words) / len(words)
    
    def _calculate_mood_trend(self, user_id: str) -> float:
        """최근 기분 트렌드 계산"""
        if user_id not in self.user_contexts or len(self.user_contexts[user_id]) < 2:
            return 0.0
        
        recent_contexts = self.user_contexts[user_id][-10:]  # 최근 10개
        valences = [ctx.emotion.valence for ctx in recent_contexts]
        
        if len(valences) < 2:
            return 0.0
        
        # 선형 트렌드 계산
        x = np.arange(len(valences))
        slope = np.polyfit(x, valences, 1)[0]
        return float(slope)
    
    def _get_session_length(self, user_id: str) -> float:
        """현재 세션 길이 (정규화)"""
        if user_id not in self.user_contexts or not self.user_contexts[user_id]:
            return 0.0
        
        session_start = self.user_contexts[user_id][0].timestamp
        current_time = datetime.now()
        session_length = (current_time - session_start).total_seconds() / 3600  # 시간 단위
        
        return min(session_length / 2.0, 1.0)  # 2시간을 최대값으로 정규화
    
    def _get_interaction_frequency(self, user_id: str) -> float:
        """상호작용 빈도 (정규화)"""
        if user_id not in self.user_contexts:
            return 0.0
        
        interaction_count = len(self.user_contexts[user_id])
        return min(interaction_count / 50.0, 1.0)  # 50회를 최대값으로 정규화
    
    def add_feedback(
        self,
        user_id: str,
        emotion: EmotionEmbedding,
        image_metadata: Dict[str, Any],
        prompt: str,
        feedback_score: float,
        timestamp: datetime = None
    ):
        """피드백 추가 및 학습"""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # 컨텍스트 특성 추출
        context_features = self.extract_context_features(
            emotion, image_metadata, prompt, user_id, timestamp
        )
        
        # 피드백 컨텍스트 생성
        feedback_context = FeedbackContext(
            emotion=emotion,
            image_features={k: v for k, v in context_features.items() 
                          if k in ["brightness", "saturation", "contrast", "hue_variance", 
                                 "edge_density", "color_diversity", "composition_balance", "texture_complexity"]},
            prompt_features={k: v for k, v in context_features.items() 
                           if k in ["length", "positive_words", "negative_words", "color_mentions",
                                  "emotion_words", "art_style_mentions", "complexity_indicators",
                                  "nature_elements", "human_elements", "abstract_elements"]},
            user_state={k: v for k, v in context_features.items() 
                       if k in ["time_of_day", "time_of_day_cos", "recent_mood_trend",
                              "session_length", "interaction_frequency"]},
            feedback_score=feedback_score,
            timestamp=timestamp
        )
        
        # 사용자별 컨텍스트 저장
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = []
        self.user_contexts[user_id].append(feedback_context)
        
        # 전역 피드백 버퍼에 추가
        self.feedback_buffer.append(feedback_context)
        if len(self.feedback_buffer) > self.max_buffer_size:
            self.feedback_buffer.pop(0)
        
        # 주기적 모델 업데이트
        if len(self.feedback_buffer) % self.update_interval == 0:
            self._update_model()
        
        # 즉시 선호도 업데이트
        self._update_user_preferences(user_id, feedback_context)
        
        logger.info(f"✅ 사용자 {user_id} 피드백 추가 및 학습 완료")
    
    def _update_model(self):
        """신경망 모델 업데이트"""
        
        if len(self.feedback_buffer) < 10:  # 최소 데이터 요구량
            return
        
        self.preference_net.train()
        
        # 배치 데이터 준비
        batch_contexts = []
        batch_preferences = []
        batch_satisfactions = []
        
        for feedback_ctx in self.feedback_buffer[-50:]:  # 최근 50개
            # 컨텍스트 특성 벡터화
            context_vector = self._vectorize_context(feedback_ctx)
            batch_contexts.append(context_vector)
            
            # 타겟 선호도 (현재는 피드백 기반 휴리스틱 사용)
            target_prefs = self._feedback_to_preferences(feedback_ctx)
            batch_preferences.append(target_prefs)
            
            # 만족도 (정규화된 피드백 점수)
            satisfaction = (feedback_ctx.feedback_score - 1) / 4  # 1-5 -> 0-1
            batch_satisfactions.append([satisfaction])
        
        # 텐서 변환
        contexts_tensor = torch.FloatTensor(batch_contexts).to(device)
        prefs_tensor = torch.FloatTensor(batch_preferences).to(device)
        sats_tensor = torch.FloatTensor(batch_satisfactions).to(device)
        
        # Forward pass
        pred_prefs, pred_sats = self.preference_net(contexts_tensor)
        
        # 손실 계산
        pref_loss = nn.MSELoss()(pred_prefs, prefs_tensor)
        sat_loss = nn.MSELoss()(pred_sats, sats_tensor)
        total_loss = pref_loss + sat_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # 메트릭 업데이트
        self.training_history["losses"].append(total_loss.item())
        
        logger.info(f"🤖 모델 업데이트 완료: 손실 {total_loss.item():.4f}")
    
    def _vectorize_context(self, feedback_ctx: FeedbackContext) -> List[float]:
        """피드백 컨텍스트를 벡터로 변환"""
        
        vector = []
        
        # 감정 특성
        vector.extend([
            feedback_ctx.emotion.valence,
            feedback_ctx.emotion.arousal, 
            feedback_ctx.emotion.dominance
        ])
        
        # 이미지 특성
        for key in ["brightness", "saturation", "contrast", "hue_variance", 
                   "edge_density", "color_diversity", "composition_balance", "texture_complexity"]:
            vector.append(feedback_ctx.image_features.get(key, 0.0))
        
        # 프롬프트 특성
        for key in ["length", "positive_words", "negative_words", "color_mentions",
                   "emotion_words", "art_style_mentions", "complexity_indicators",
                   "nature_elements", "human_elements", "abstract_elements"]:
            vector.append(feedback_ctx.prompt_features.get(key, 0.0))
        
        # 사용자 상태
        for key in ["time_of_day", "time_of_day_cos", "recent_mood_trend",
                   "session_length", "interaction_frequency"]:
            vector.append(feedback_ctx.user_state.get(key, 0.0))
        
        return vector
    
    def _feedback_to_preferences(self, feedback_ctx: FeedbackContext) -> List[float]:
        """피드백을 선호도로 변환 (초기 휴리스틱)"""
        
        preferences = [0.0] * len(self.preference_names)
        feedback_strength = (feedback_ctx.feedback_score - 3) / 2  # -1 to 1
        
        if abs(feedback_strength) < 0.1:  # 중성 피드백
            return preferences
        
        # 다양한 요소를 고려한 동적 매핑
        emotion = feedback_ctx.emotion
        
        # 색온도 (감정 valence + 프롬프트 특성 기반)
        preferences[0] = feedback_strength * (
            0.3 * emotion.valence + 
            0.2 * feedback_ctx.prompt_features.get("positive_words", 0) -
            0.2 * feedback_ctx.prompt_features.get("negative_words", 0) +
            0.3 * (feedback_ctx.user_state.get("time_of_day", 0) > 0)  # 낮 시간 선호
        )
        
        # 밝기 (arousal + 시간대 고려)
        preferences[1] = feedback_strength * (
            0.4 * emotion.arousal +
            0.3 * feedback_ctx.prompt_features.get("positive_words", 0) +
            0.3 * np.cos(2 * np.pi * datetime.now().hour / 24)  # 낮에 더 밝게
        )
        
        # 채도 (arousal + 감정 강도)
        preferences[2] = feedback_strength * (
            0.5 * emotion.arousal +
            0.3 * abs(emotion.valence) +
            0.2 * feedback_ctx.prompt_features.get("color_mentions", 0)
        )
        
        # 대비 (dominance + 복잡성 선호)
        preferences[3] = feedback_strength * (
            0.4 * emotion.dominance +
            0.3 * feedback_ctx.prompt_features.get("complexity_indicators", 0) +
            0.3 * emotion.arousal
        )
        
        # 복잡성 (개인별 학습 이력 + 프롬프트)
        preferences[4] = feedback_strength * (
            0.3 * emotion.dominance +
            0.4 * feedback_ctx.prompt_features.get("complexity_indicators", 0) +
            0.3 * feedback_ctx.user_state.get("interaction_frequency", 0)
        )
        
        # 예술 스타일 (프롬프트 기반)
        preferences[5] = feedback_strength * (
            0.5 * feedback_ctx.prompt_features.get("art_style_mentions", 0) +
            0.3 * feedback_ctx.prompt_features.get("abstract_elements", 0) +
            0.2 * emotion.valence
        )
        
        # 구성 (dominance + 균형 선호)
        preferences[6] = feedback_strength * (
            0.4 * emotion.dominance +
            0.3 * feedback_ctx.image_features.get("composition_balance", 0) +
            0.3 * feedback_ctx.prompt_features.get("nature_elements", 0)
        )
        
        # 범위 제한
        preferences = [max(-1.0, min(1.0, p)) for p in preferences]
        
        return preferences
    
    def _update_user_preferences(self, user_id: str, feedback_ctx: FeedbackContext):
        """사용자별 선호도 업데이트"""
        
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = np.zeros(len(self.preference_names))
        
        # 컨텍스트 기반 선호도 예측
        if len(self.feedback_buffer) > 10:
            context_vector = torch.FloatTensor([self._vectorize_context(feedback_ctx)]).to(device)
            
            self.preference_net.eval()
            with torch.no_grad():
                predicted_prefs, _ = self.preference_net(context_vector)
                predicted_prefs = predicted_prefs.cpu().numpy()[0]
        else:
            # 초기에는 휴리스틱 사용
            predicted_prefs = np.array(self._feedback_to_preferences(feedback_ctx))
        
        # 적응형 학습률 (피드백 강도에 비례)
        feedback_strength = abs(feedback_ctx.feedback_score - 3) / 2
        adaptive_lr = 0.05 + 0.15 * feedback_strength  # 0.05 ~ 0.2
        
        # 탐색-활용 균형
        if np.random.random() < self.exploration_rate:
            # 탐색: 밴딧 기반 탐색 방향
            exploration_prefs = self.bandit.select_exploration_direction(
                self.user_preferences[user_id]
            )
            self.user_preferences[user_id] = (
                (1 - adaptive_lr) * self.user_preferences[user_id] + 
                adaptive_lr * exploration_prefs
            )
        else:
            # 활용: 예측된 선호도 사용
            self.user_preferences[user_id] = (
                (1 - adaptive_lr) * self.user_preferences[user_id] + 
                adaptive_lr * predicted_prefs
            )
        
        # 밴딧 보상 업데이트
        for i, pref_val in enumerate(predicted_prefs):
            if abs(pref_val) > 0.1:  # 유의미한 선호도 변화
                reward = feedback_strength if pref_val * feedback_strength > 0 else -feedback_strength
                self.bandit.update_rewards(i, reward)
        
        # 범위 제한
        self.user_preferences[user_id] = np.clip(self.user_preferences[user_id], -1.0, 1.0)
    
    def get_user_preferences(self, user_id: str) -> Dict[str, float]:
        """사용자 선호도 반환"""
        
        if user_id not in self.user_preferences:
            return {name: 0.0 for name in self.preference_names}
        
        prefs_array = self.user_preferences[user_id]
        return {
            name: float(prefs_array[i]) 
            for i, name in enumerate(self.preference_names)
        }
    
    def predict_satisfaction(
        self,
        user_id: str,
        emotion: EmotionEmbedding,
        image_metadata: Dict[str, Any],
        prompt: str
    ) -> float:
        """만족도 예측"""
        
        context_features = self.extract_context_features(
            emotion, image_metadata, prompt, user_id
        )
        
        feedback_ctx = FeedbackContext(
            emotion=emotion,
            image_features={k: v for k, v in context_features.items() 
                          if k in ["brightness", "saturation", "contrast", "hue_variance", 
                                 "edge_density", "color_diversity", "composition_balance", "texture_complexity"]},
            prompt_features={k: v for k, v in context_features.items() 
                           if k in ["length", "positive_words", "negative_words", "color_mentions",
                                  "emotion_words", "art_style_mentions", "complexity_indicators",
                                  "nature_elements", "human_elements", "abstract_elements"]},
            user_state={k: v for k, v in context_features.items() 
                       if k in ["time_of_day", "time_of_day_cos", "recent_mood_trend",
                              "session_length", "interaction_frequency"]},
            feedback_score=3.0,  # 더미값
            timestamp=datetime.now()
        )
        
        context_vector = torch.FloatTensor([self._vectorize_context(feedback_ctx)]).to(device)
        
        self.preference_net.eval()
        with torch.no_grad():
            _, predicted_satisfaction = self.preference_net(context_vector)
            satisfaction = predicted_satisfaction.cpu().item()
        
        return satisfaction * 4 + 1  # 0-1을 1-5로 변환
    
    def save_model(self, path: str):
        """모델 저장"""
        
        save_data = {
            "model_state_dict": self.preference_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "user_preferences": {k: v.tolist() for k, v in self.user_preferences.items()},
            "bandit_state": {
                "action_counts": self.bandit.action_counts.tolist(),
                "action_rewards": self.bandit.action_rewards.tolist()
            },
            "training_history": self.training_history,
            "preference_names": self.preference_names
        }
        
        torch.save(save_data, path)
        logger.info(f"✅ 적응형 개인화 모델 저장: {path}")
    
    def load_model(self, path: str):
        """모델 로드"""
        
        try:
            save_data = torch.load(path, map_location=device)
            
            self.preference_net.load_state_dict(save_data["model_state_dict"])
            self.optimizer.load_state_dict(save_data["optimizer_state_dict"])
            
            self.user_preferences = {
                k: np.array(v) for k, v in save_data["user_preferences"].items()
            }
            
            self.bandit.action_counts = np.array(save_data["bandit_state"]["action_counts"])
            self.bandit.action_rewards = np.array(save_data["bandit_state"]["action_rewards"])
            
            self.training_history = save_data["training_history"]
            self.preference_names = save_data["preference_names"]
            
            logger.info(f"✅ 적응형 개인화 모델 로드: {path}")
            
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {e}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """학습 요약 정보"""
        
        summary = {
            "total_users": len(self.user_preferences),
            "total_feedback": len(self.feedback_buffer),
            "training_iterations": len(self.training_history["losses"]),
            "average_loss": np.mean(self.training_history["losses"][-10:]) if self.training_history["losses"] else 0.0,
            "exploration_rate": self.exploration_rate,
            "model_parameters": sum(p.numel() for p in self.preference_net.parameters()),
            "bandit_total_actions": np.sum(self.bandit.action_counts)
        }
        
        return summary