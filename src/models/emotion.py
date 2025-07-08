#!/usr/bin/env python3
"""
EmotionEmbedding 클래스 - VAD 기반 감정 임베딩
"""

import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class EmotionEmbedding:
    """Valence-Arousal-Dominance 기반 감정 임베딩"""

    valence: float  # -1.0 (부정) to 1.0 (긍정)
    arousal: float  # -1.0 (차분) to 1.0 (흥분)
    dominance: float = 0.0  # -1.0 (수동) to 1.0 (지배적)
    confidence: float = 1.0  # 감정 예측 신뢰도

    def to_vector(self) -> np.ndarray:
        """감정을 numpy 벡터로 변환"""
        return np.array([self.valence, self.arousal, self.dominance])

    def to_dict(self) -> Dict[str, float]:
        """감정을 딕셔너리로 변환"""
        return asdict(self)

    @classmethod
    def from_vector(cls, vector: np.ndarray, confidence: float = 1.0):
        """numpy 벡터로부터 감정 임베딩 생성"""
        return cls(
            valence=float(vector[0]),
            arousal=float(vector[1]),
            dominance=float(vector[2]) if len(vector) > 2 else 0.0,
            confidence=confidence,
        )

    def distance_to(self, other: "EmotionEmbedding") -> float:
        """다른 감정과의 유클리드 거리"""
        return np.linalg.norm(self.to_vector() - other.to_vector())

    def similarity_to(self, other: "EmotionEmbedding") -> float:
        """다른 감정과의 유사도 (0-1)"""
        max_distance = np.sqrt(3 * 4)  # 최대 거리 (각 차원 -2 to 2)
        return 1.0 - (self.distance_to(other) / max_distance)