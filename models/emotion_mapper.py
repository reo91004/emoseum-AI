#!/usr/bin/env python3
"""
AdvancedEmotionMapper - 고급 VAD 기반 감정 매핑 시스템
"""

import os
import json
import warnings
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

from config import device, logger, TRANSFORMERS_AVAILABLE
from models.emotion import EmotionEmbedding

# 경고 메시지 억제
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

if TRANSFORMERS_AVAILABLE:
    from transformers import AutoTokenizer, AutoModel


class AdvancedEmotionMapper:
    """고급 VAD 기반 감정 매핑 시스템"""

    def __init__(self, model_name="klue/roberta-large"):
        self.device = device
        self.model_name = model_name

        # 감정 어휘 사전 (한국어 + 영어)
        self.emotion_lexicon = {
            # 기본 감정들
            "기쁨": EmotionEmbedding(0.8, 0.6, 0.4),
            "행복": EmotionEmbedding(0.8, 0.5, 0.3),
            "즐거움": EmotionEmbedding(0.7, 0.7, 0.4),
            "신남": EmotionEmbedding(0.9, 0.8, 0.6),
            "만족": EmotionEmbedding(0.6, 0.2, 0.3),
            "뿌듯": EmotionEmbedding(0.7, 0.4, 0.7),
            "슬픔": EmotionEmbedding(-0.7, -0.3, -0.5),
            "우울": EmotionEmbedding(-0.8, -0.4, -0.6),
            "허무": EmotionEmbedding(-0.5, -0.6, -0.4),
            "절망": EmotionEmbedding(-0.9, 0.3, -0.8),
            "상실": EmotionEmbedding(-0.8, -0.2, -0.6),
            "외로움": EmotionEmbedding(-0.6, -0.2, -0.6),
            "화남": EmotionEmbedding(-0.6, 0.8, 0.7),
            "분노": EmotionEmbedding(-0.8, 0.9, 0.8),
            "짜증": EmotionEmbedding(-0.5, 0.6, 0.4),
            "답답": EmotionEmbedding(-0.4, 0.5, -0.2),
            "억울": EmotionEmbedding(-0.7, 0.6, -0.3),
            "두려움": EmotionEmbedding(-0.8, 0.7, -0.8),
            "걱정": EmotionEmbedding(-0.5, 0.6, -0.4),
            "불안": EmotionEmbedding(-0.5, 0.7, -0.5),
            "무서움": EmotionEmbedding(-0.8, 0.8, -0.7),
            "긴장": EmotionEmbedding(-0.2, 0.8, -0.3),
            "놀람": EmotionEmbedding(0.2, 0.9, 0.1),
            "당황": EmotionEmbedding(-0.2, 0.8, -0.4),
            "충격": EmotionEmbedding(-0.3, 0.9, -0.2),
            "평온": EmotionEmbedding(0.4, -0.7, 0.2),
            "차분": EmotionEmbedding(0.3, -0.8, 0.1),
            "편안": EmotionEmbedding(0.6, -0.5, 0.3),
            "안정": EmotionEmbedding(0.5, -0.6, 0.4),
            "스트레스": EmotionEmbedding(-0.6, 0.7, -0.3),
            "피곤": EmotionEmbedding(-0.3, -0.8, -0.4),
            "지침": EmotionEmbedding(-0.4, -0.7, -0.5),
            "권태": EmotionEmbedding(-0.2, -0.8, -0.3),
            "사랑": EmotionEmbedding(0.9, 0.5, 0.3),
            "애정": EmotionEmbedding(0.8, 0.4, 0.4),
            "그리움": EmotionEmbedding(0.3, 0.3, -0.2),
            "감사": EmotionEmbedding(0.8, 0.3, 0.3),
            "고마움": EmotionEmbedding(0.7, 0.2, 0.2),
            # 영어 감정들
            "joy": EmotionEmbedding(0.8, 0.6, 0.4),
            "happiness": EmotionEmbedding(0.8, 0.5, 0.3),
            "sadness": EmotionEmbedding(-0.7, -0.3, -0.5),
            "anger": EmotionEmbedding(-0.6, 0.8, 0.7),
            "fear": EmotionEmbedding(-0.8, 0.7, -0.8),
            "surprise": EmotionEmbedding(0.2, 0.9, 0.1),
            "love": EmotionEmbedding(0.9, 0.5, 0.3),
            "peace": EmotionEmbedding(0.4, -0.7, 0.2),
            "stress": EmotionEmbedding(-0.6, 0.7, -0.3),
            "tired": EmotionEmbedding(-0.3, -0.8, -0.4),
        }

        # 감정 강화 표현들
        self.emotion_intensifiers = {
            "매우": 1.3,
            "정말": 1.2,
            "엄청": 1.4,
            "너무": 1.3,
            "완전": 1.4,
            "조금": 0.7,
            "약간": 0.6,
            "살짝": 0.5,
            "좀": 0.7,
            "extremely": 1.4,
            "very": 1.3,
            "really": 1.2,
            "quite": 1.1,
            "slightly": 0.6,
            "somewhat": 0.7,
            "a bit": 0.6,
        }

        # 부정 표현들
        self.negation_words = {
            "안",
            "못",
            "없",
            "아니",
            "not",
            "no",
            "never",
            "don't",
            "can't",
            "won't",
        }

        # Transformer 모델 로드
        self.use_transformer = TRANSFORMERS_AVAILABLE
        if self.use_transformer:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.text_encoder = AutoModel.from_pretrained(model_name).to(
                    self.device
                )
                self.text_encoder.eval()

                # VAD 예측 헤드
                hidden_size = self.text_encoder.config.hidden_size
                self.vad_predictor = nn.Sequential(
                    nn.Linear(hidden_size, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 3),  # valence, arousal, dominance
                    nn.Tanh(),
                ).to(self.device)

                # 간단한 가중치 초기화
                self._init_vad_predictor()

                logger.info(f"✅ 고급 감정 분석 모델 로드 완료: {model_name}")
            except Exception as e:
                logger.warning(
                    f"⚠️ Transformer 모델 로드 실패: {e}, 규칙 기반 시스템 사용"
                )
                self.use_transformer = False

    def _init_vad_predictor(self):
        """VAD 예측기 초기화"""
        for module in self.vad_predictor:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def extract_emotion_from_text(self, text: str) -> EmotionEmbedding:
        """텍스트에서 감정 추출 (다중 방법론 융합)"""
        # 1. 규칙 기반 감정 분석
        rule_based_emotion = self._rule_based_emotion_analysis(text)

        # 2. Transformer 기반 분석 (가능한 경우)
        if self.use_transformer:
            try:
                transformer_emotion = self._transformer_emotion_analysis(text)
                # 두 결과를 가중 평균
                final_emotion = self._combine_emotions(
                    rule_based_emotion,
                    transformer_emotion,
                    rule_weight=0.4,
                    transformer_weight=0.6,
                )
            except Exception as e:
                logger.warning(f"Transformer 분석 실패: {e}, 규칙 기반 결과 사용")
                final_emotion = rule_based_emotion
        else:
            final_emotion = rule_based_emotion

        # 3. 후처리 및 정규화
        final_emotion = self._post_process_emotion(final_emotion, text)

        logger.info(
            f"감정 분석 결과: V={final_emotion.valence:.3f}, A={final_emotion.arousal:.3f}, D={final_emotion.dominance:.3f}"
        )
        return final_emotion

    def _rule_based_emotion_analysis(self, text: str) -> EmotionEmbedding:
        """규칙 기반 감정 분석"""
        text_lower = text.lower()
        words = text_lower.split()

        detected_emotions = []
        emotion_weights = []

        # 감정 단어 탐지
        for i, word in enumerate(words):
            # 감정 어휘 매칭
            for emotion_word, emotion_emb in self.emotion_lexicon.items():
                if emotion_word in word or word in emotion_word:
                    # 강화 표현 체크
                    intensity = 1.0
                    if i > 0 and words[i - 1] in self.emotion_intensifiers:
                        intensity = self.emotion_intensifiers[words[i - 1]]

                    # 부정 표현 체크
                    negated = False
                    for j in range(max(0, i - 2), i):
                        if words[j] in self.negation_words:
                            negated = True
                            break

                    # 감정 임베딩 조정
                    adjusted_emotion = EmotionEmbedding(
                        valence=emotion_emb.valence
                        * intensity
                        * (-1 if negated else 1),
                        arousal=emotion_emb.arousal * intensity,
                        dominance=emotion_emb.dominance
                        * intensity
                        * (-1 if negated else 1),
                    )

                    detected_emotions.append(adjusted_emotion)
                    emotion_weights.append(intensity)

        if not detected_emotions:
            # 기본 중성 감정
            return EmotionEmbedding(0.0, 0.0, 0.0, confidence=0.3)

        # 가중 평균 계산
        total_weight = sum(emotion_weights)
        avg_valence = (
            sum(e.valence * w for e, w in zip(detected_emotions, emotion_weights))
            / total_weight
        )
        avg_arousal = (
            sum(e.arousal * w for e, w in zip(detected_emotions, emotion_weights))
            / total_weight
        )
        avg_dominance = (
            sum(e.dominance * w for e, w in zip(detected_emotions, emotion_weights))
            / total_weight
        )

        confidence = min(
            1.0, len(detected_emotions) / 3.0
        )  # 감정 단어 개수 기반 신뢰도

        return EmotionEmbedding(avg_valence, avg_arousal, avg_dominance, confidence)

    def _transformer_emotion_analysis(self, text: str) -> EmotionEmbedding:
        """Transformer 기반 감정 분석"""
        if not self.use_transformer:
            return EmotionEmbedding(0.0, 0.0, 0.0, confidence=0.0)

        # 토큰화 및 인코딩
        inputs = self.tokenizer(
            text, return_tensors="pt", max_length=512, truncation=True, padding=True
        ).to(self.device)

        with torch.no_grad():
            # 텍스트 특성 추출
            outputs = self.text_encoder(**inputs)
            # CLS 토큰 또는 평균 풀링 사용
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                text_features = outputs.pooler_output
            else:
                text_features = outputs.last_hidden_state.mean(dim=1)

            # VAD 예측
            vad_scores = self.vad_predictor(text_features)
            vad_scores = vad_scores.squeeze().cpu().numpy()

        return EmotionEmbedding(
            valence=float(vad_scores[0]),
            arousal=float(vad_scores[1]),
            dominance=float(vad_scores[2]),
            confidence=0.8,
        )

    def _combine_emotions(
        self,
        emotion1: EmotionEmbedding,
        emotion2: EmotionEmbedding,
        rule_weight: float = 0.5,
        transformer_weight: float = 0.5,
    ) -> EmotionEmbedding:
        """두 감정 분석 결과를 결합"""
        total_weight = rule_weight + transformer_weight
        rule_weight /= total_weight
        transformer_weight /= total_weight

        # 신뢰도 기반 가중치 조정
        rule_conf_weight = rule_weight * emotion1.confidence
        trans_conf_weight = transformer_weight * emotion2.confidence
        total_conf_weight = rule_conf_weight + trans_conf_weight

        if total_conf_weight > 0:
            rule_conf_weight /= total_conf_weight
            trans_conf_weight /= total_conf_weight
        else:
            rule_conf_weight = trans_conf_weight = 0.5

        return EmotionEmbedding(
            valence=emotion1.valence * rule_conf_weight
            + emotion2.valence * trans_conf_weight,
            arousal=emotion1.arousal * rule_conf_weight
            + emotion2.arousal * trans_conf_weight,
            dominance=emotion1.dominance * rule_conf_weight
            + emotion2.dominance * trans_conf_weight,
            confidence=min(1.0, emotion1.confidence + emotion2.confidence),
        )

    def _post_process_emotion(
        self, emotion: EmotionEmbedding, text: str
    ) -> EmotionEmbedding:
        """감정 후처리 및 정규화"""
        # 범위 제한
        valence = np.clip(emotion.valence, -1.0, 1.0)
        arousal = np.clip(emotion.arousal, -1.0, 1.0)
        dominance = np.clip(emotion.dominance, -1.0, 1.0)

        # 텍스트 길이 기반 신뢰도 조정
        text_length_factor = min(1.0, len(text.split()) / 10.0)
        confidence = emotion.confidence * text_length_factor

        return EmotionEmbedding(valence, arousal, dominance, confidence)

    def emotion_to_prompt_modifiers(self, emotion: EmotionEmbedding) -> str:
        """감정을 프롬프트 수정자로 변환"""
        modifiers = []

        # Valence 기반 수정자
        if emotion.valence > 0.6:
            modifiers.extend(["bright", "cheerful", "uplifting", "positive"])
        elif emotion.valence > 0.2:
            modifiers.extend(["pleasant", "mild", "gentle"])
        elif emotion.valence < -0.6:
            modifiers.extend(["dark", "melancholic", "somber", "moody"])
        elif emotion.valence < -0.2:
            modifiers.extend(["subdued", "quiet", "contemplative"])

        # Arousal 기반 수정자
        if emotion.arousal > 0.6:
            modifiers.extend(["dynamic", "energetic", "vibrant", "intense"])
        elif emotion.arousal > 0.2:
            modifiers.extend(["lively", "animated"])
        elif emotion.arousal < -0.6:
            modifiers.extend(["calm", "peaceful", "serene", "tranquil"])
        elif emotion.arousal < -0.2:
            modifiers.extend(["relaxed", "soft"])

        # Dominance 기반 수정자
        if emotion.dominance > 0.4:
            modifiers.extend(["bold", "confident", "strong"])
        elif emotion.dominance < -0.4:
            modifiers.extend(["delicate", "subtle", "gentle"])

        # 감정 강도 기반 수정자
        intensity = np.sqrt(
            emotion.valence**2 + emotion.arousal**2 + emotion.dominance**2
        ) / np.sqrt(3)
        if intensity > 0.8:
            modifiers.append("highly detailed")
        elif intensity < 0.3:
            modifiers.append("minimalist")

        return ", ".join(modifiers[:6])  # 최대 6개 수정자