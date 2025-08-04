# src/services/emotion_analyzer.py

# ==============================================================================
# GoEmotions 모델을 사용한 감정 분석 서비스
# Google Research의 GoEmotions 데이터셋으로 학습된 모델을 사용하여
# 텍스트에서 28가지 세밀한 감정을 추출합니다.
# ==============================================================================

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import warnings
import requests
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class GoEmotionsAnalyzer:
    """GoEmotions 모델을 사용한 감정 분석기"""
    
    # GoEmotions 감정 레이블 (28개)
    EMOTION_LABELS = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness',
        'optimism', 'pride', 'realization', 'relief', 'remorse',
        'sadness', 'surprise', 'neutral'
    ]
    
    # VAD 매핑 (각 감정에 대한 Valence, Arousal, Dominance 값)
    EMOTION_VAD_MAPPING = {
        'admiration': (0.8, 0.5, 0.4),
        'amusement': (0.8, 0.6, 0.7),
        'anger': (0.1, 0.8, 0.7),
        'annoyance': (0.2, 0.6, 0.5),
        'approval': (0.7, 0.3, 0.6),
        'caring': (0.8, 0.4, 0.6),
        'confusion': (0.3, 0.5, 0.2),
        'curiosity': (0.6, 0.6, 0.5),
        'desire': (0.7, 0.7, 0.6),
        'disappointment': (0.2, 0.3, 0.3),
        'disapproval': (0.2, 0.5, 0.6),
        'disgust': (0.1, 0.5, 0.5),
        'embarrassment': (0.2, 0.6, 0.2),
        'excitement': (0.9, 0.9, 0.7),
        'fear': (0.1, 0.8, 0.1),
        'gratitude': (0.9, 0.4, 0.4),
        'grief': (0.1, 0.3, 0.1),
        'joy': (0.9, 0.7, 0.8),
        'love': (0.9, 0.6, 0.6),
        'nervousness': (0.3, 0.7, 0.2),
        'optimism': (0.8, 0.5, 0.7),
        'pride': (0.8, 0.6, 0.8),
        'realization': (0.6, 0.5, 0.6),
        'relief': (0.8, 0.3, 0.5),
        'remorse': (0.2, 0.4, 0.2),
        'sadness': (0.1, 0.3, 0.2),
        'surprise': (0.6, 0.8, 0.4),
        'neutral': (0.5, 0.5, 0.5)
    }
    
    # 감정 그룹핑 (유사한 감정들을 그룹화)
    EMOTION_GROUPS = {
        'positive': ['admiration', 'amusement', 'approval', 'caring', 'excitement', 
                    'gratitude', 'joy', 'love', 'optimism', 'pride', 'relief'],
        'negative': ['anger', 'annoyance', 'disappointment', 'disapproval', 'disgust',
                    'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness'],
        'ambiguous': ['confusion', 'curiosity', 'desire', 'realization', 'surprise', 'neutral']
    }
    
    def __init__(self, model_name: str = None):
        """
        GoEmotions 분석기 초기화
        
        Args:
            model_name: HuggingFace 모델 이름 (기본값: 환경변수에서 가져옴)
        """
        import os
        self.model_name = model_name or os.getenv("GOEMOTION_MODEL", "joeddav/distilbert-base-uncased-go-emotions-student")
        self.device = self._get_device()
        self.classifier = None
        self._load_model()
        
    def _get_device(self) -> torch.device:
        """사용 가능한 디바이스 확인"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def _load_model(self):
        """GoEmotions 모델 로드"""
        try:
            logger.info(f"GoEmotions 모델 로드 중: {self.model_name}")
            
            # HuggingFace pipeline 사용
            self.classifier = pipeline(
                "text-classification",
                model=self.model_name,
                device=0 if self.device.type == "cuda" else -1,
                top_k=None  # 모든 레이블의 점수를 반환
            )
            
            logger.info(f"GoEmotions 모델 로드 완료 (디바이스: {self.device})")
            
        except Exception as e:
            logger.error(f"GoEmotions 모델 로드 실패: {e}")
            self.classifier = None
    
    def analyze_emotions(self, text: str, threshold: float = 0.3) -> Dict[str, Any]:
        """
        텍스트에서 감정 분석
        
        Args:
            text: 분석할 텍스트
            threshold: 감정 선택 임계값 (0-1)
            
        Returns:
            감정 분석 결과 딕셔너리
        """
        if not self.classifier:
            raise RuntimeError("GoEmotions 모델이 로드되지 않았습니다.")
        
        try:
            # 모델 예측
            results = self.classifier(text)
            
            # 점수가 높은 감정들 선택
            emotions = []
            scores = {}
            
            for result in results[0]:  # pipeline은 리스트 안에 결과를 반환
                label = result['label']
                score = result['score']
                scores[label] = score
                
                if score >= threshold:
                    emotions.append((label, score))
            
            # 점수 기준으로 정렬
            emotions.sort(key=lambda x: x[1], reverse=True)
            
            # 상위 5개 감정 선택
            top_emotions = emotions[:5]
            emotion_keywords = [em[0] for em in top_emotions]
            
            # 감정이 없으면 neutral 추가
            if not emotion_keywords:
                emotion_keywords = ['neutral']
                top_emotions = [('neutral', 1.0)]
            
            # VAD 점수 계산
            vad_scores = self._calculate_vad_scores(top_emotions)
            
            # 주요 감정 결정
            primary_emotion = emotion_keywords[0] if emotion_keywords else 'neutral'
            
            # 감정 강도 계산
            emotional_intensity = self._calculate_intensity(top_emotions)
            
            return {
                "keywords": emotion_keywords,
                "vad_scores": vad_scores,
                "confidence": float(np.mean([score for _, score in top_emotions])),
                "primary_emotion": primary_emotion,
                "emotional_intensity": emotional_intensity,
                "all_scores": scores,
                "top_emotions": dict(top_emotions)
            }
            
        except Exception as e:
            logger.error(f"감정 분석 실패: {e}")
            # 오류 시 기본값 반환
            return {
                "keywords": ["neutral"],
                "vad_scores": [0.5, 0.5, 0.5],
                "confidence": 0.0,
                "primary_emotion": "neutral",
                "emotional_intensity": "low",
                "error": str(e)
            }
    
    def _calculate_vad_scores(self, emotions: List[Tuple[str, float]]) -> List[float]:
        """가중 평균으로 VAD 점수 계산"""
        if not emotions:
            return [0.5, 0.5, 0.5]
        
        total_weight = sum(score for _, score in emotions)
        if total_weight == 0:
            return [0.5, 0.5, 0.5]
        
        weighted_vad = [0.0, 0.0, 0.0]
        
        for emotion, score in emotions:
            vad = self.EMOTION_VAD_MAPPING.get(emotion, (0.5, 0.5, 0.5))
            for i in range(3):
                weighted_vad[i] += vad[i] * score
        
        # 정규화
        vad_scores = [v / total_weight for v in weighted_vad]
        
        # 0-1 범위로 클리핑
        vad_scores = [max(0.0, min(1.0, v)) for v in vad_scores]
        
        return vad_scores
    
    def _calculate_intensity(self, emotions: List[Tuple[str, float]]) -> str:
        """감정 강도 계산"""
        if not emotions:
            return "low"
        
        # 상위 감정의 평균 점수
        avg_score = np.mean([score for _, score in emotions[:3]])
        
        # Arousal 기반 강도 조정
        arousal_boost = 0.0
        for emotion, score in emotions[:3]:
            vad = self.EMOTION_VAD_MAPPING.get(emotion, (0.5, 0.5, 0.5))
            arousal_boost += vad[1] * score
        
        if emotions:
            arousal_boost /= min(3, len(emotions))
        
        # 최종 강도 계산
        intensity_score = (avg_score + arousal_boost) / 2
        
        if intensity_score >= 0.7:
            return "high"
        elif intensity_score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def get_emotion_info(self, emotion: str) -> Dict[str, Any]:
        """특정 감정에 대한 상세 정보 반환"""
        if emotion not in self.EMOTION_LABELS:
            return {"error": f"Unknown emotion: {emotion}"}
        
        vad = self.EMOTION_VAD_MAPPING.get(emotion, (0.5, 0.5, 0.5))
        
        # 감정이 속한 그룹 찾기
        emotion_group = "unknown"
        for group, emotions in self.EMOTION_GROUPS.items():
            if emotion in emotions:
                emotion_group = group
                break
        
        return {
            "emotion": emotion,
            "vad": {
                "valence": vad[0],
                "arousal": vad[1], 
                "dominance": vad[2]
            },
            "group": emotion_group
        }
    
    def batch_analyze(self, texts: List[str], threshold: float = 0.3) -> List[Dict[str, Any]]:
        """여러 텍스트를 한 번에 분석"""
        results = []
        for text in texts:
            result = self.analyze_emotions(text, threshold)
            results.append(result)
        return results


class ColabGoEmotionsAnalyzer:
    """Colab에서 실행되는 GoEmotions 모델을 사용한 감정 분석기"""
    
    def __init__(self, colab_url: str = None):
        """
        Colab GoEmotions 분석기 초기화
        
        Args:
            colab_url: Colab 서버 URL (기본값: 환경변수에서 가져옴)
        """
        self.colab_url = colab_url or os.getenv("COLAB_NOTEBOOK_URL", "").rstrip('/')
        if not self.colab_url:
            raise ValueError("COLAB_NOTEBOOK_URL 환경변수가 설정되지 않았습니다.")
        
        logger.info(f"Colab GoEmotions 분석기 초기화: {self.colab_url}")
    
    def analyze_emotions(self, text: str, threshold: float = 0.3) -> Dict[str, Any]:
        """
        Colab 서버를 통해 텍스트 감정 분석
        
        Args:
            text: 분석할 텍스트
            threshold: 감정 선택 임계값 (0-1)
            
        Returns:
            감정 분석 결과 딕셔너리
        """
        try:
            # Colab 서버에 요청
            payload = {
                "text": text,
                "threshold": threshold
            }
            
            response = requests.post(
                f"{self.colab_url}/analyze_emotion",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    # 응답에서 success 키를 제거하고 반환
                    result_data = {k: v for k, v in result.items() if k != "success"}
                    logger.info(f"Colab 감정 분석 성공: {result_data.get('keywords', [])}")
                    return result_data
                else:
                    raise Exception(result.get("error", "Unknown error"))
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            logger.error(f"Colab 감정 분석 실패: {e}")
            # 오류 시 기본값 반환
            return {
                "keywords": ["neutral"],
                "vad_scores": [0.5, 0.5, 0.5],
                "confidence": 0.0,
                "primary_emotion": "neutral",
                "emotional_intensity": "low",
                "error": str(e)
            }
    
    def batch_analyze(self, texts: List[str], threshold: float = 0.3) -> List[Dict[str, Any]]:
        """여러 텍스트를 한 번에 분석"""
        results = []
        for text in texts:
            result = self.analyze_emotions(text, threshold)
            results.append(result)
        return results
    
    def get_emotion_info(self, emotion: str) -> Dict[str, Any]:
        """특정 감정에 대한 상세 정보 반환"""
        # 로컬 정보 사용
        local_analyzer = GoEmotionsAnalyzer()
        return local_analyzer.get_emotion_info(emotion)


# 싱글톤 인스턴스들
_local_analyzer_instance: Optional[GoEmotionsAnalyzer] = None
_colab_analyzer_instance: Optional[ColabGoEmotionsAnalyzer] = None


def get_emotion_analyzer(service_type: str = None) -> Any:
    """
    감정 분석기 인스턴스 반환 (환경변수 또는 명시적 타입에 따라)
    
    Args:
        service_type: 분석기 타입 ("local_goEmotions", "colab_goEmotions", None)
                     None인 경우 환경변수 EMOTION_ANALYSIS_SERVICE 사용
    
    Returns:
        GoEmotionsAnalyzer 또는 ColabGoEmotionsAnalyzer 인스턴스
    """
    global _local_analyzer_instance, _colab_analyzer_instance
    
    if service_type is None:
        service_type = os.getenv("EMOTION_ANALYSIS_SERVICE", "local_goEmotions")
    
    if service_type == "colab_goEmotions":
        if _colab_analyzer_instance is None:
            _colab_analyzer_instance = ColabGoEmotionsAnalyzer()
        return _colab_analyzer_instance
    else:
        # local_goEmotions 또는 기타
        if _local_analyzer_instance is None:
            _local_analyzer_instance = GoEmotionsAnalyzer()
        return _local_analyzer_instance