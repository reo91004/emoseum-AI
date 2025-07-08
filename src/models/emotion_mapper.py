#!/usr/bin/env python3
"""
AdvancedEmotionMapper - 고급 VAD 기반 감정 매핑 시스템
"""

import os
import json
import warnings
from typing import Dict, List, Tuple, Optional
import numpy as np

from config import logger
from models.emotion import EmotionEmbedding

# 경고 메시지 억제
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class AdvancedEmotionMapper:
    """고급 VAD 기반 감정 매핑 시스템"""

    def __init__(self):

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

        logger.info("✅ 규칙 기반 감정 분석 시스템 초기화 완료")


    def extract_emotion_from_text(self, text: str) -> EmotionEmbedding:
        """텍스트에서 감정 추출 (규칙 기반 시스템)"""
        # 규칙 기반 감정 분석
        final_emotion = self._rule_based_emotion_analysis(text)

        # 후처리 및 정규화
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