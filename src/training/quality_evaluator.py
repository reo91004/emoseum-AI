# src/training/quality_evaluator.py

# ==============================================================================
# GPT 품질 메트릭 기반 훈련 품질 평가기
# 사용자 반응 없이도 객관적으로 이미지 생성 품질을 평가하여 LoRA/DRaFT 훈련에 활용
# ==============================================================================

import re
import logging
from typing import Dict, List, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class QualityEvaluator:
    """GPT 메트릭 기반 품질 평가기"""

    def __init__(self):
        # 감정-스타일 매핑
        self.emotion_style_mapping = {
            # 긍정 감정 스타일
            'joy': ['bright', 'vibrant', 'warm', 'uplifting', 'energetic', 'radiant'],
            'excitement': ['dynamic', 'bold', 'colorful', 'lively', 'spirited'],
            'gratitude': ['soft', 'gentle', 'warm', 'peaceful', 'harmonious'],
            'love': ['warm', 'tender', 'glowing', 'romantic', 'intimate'],
            'optimism': ['bright', 'hopeful', 'ascending', 'light', 'positive'],
            'pride': ['strong', 'confident', 'bold', 'dignified', 'elevated'],
            'relief': ['calm', 'soothing', 'peaceful', 'gentle', 'relaxing'],
            
            # 부정 감정 스타일 (치료적 접근)
            'sadness': ['soft', 'muted', 'gentle', 'contemplative', 'healing'],
            'anger': ['controlled', 'structured', 'channeled', 'transformative'],
            'fear': ['protective', 'enclosed', 'safe', 'grounded', 'secure'],
            'disappointment': ['reflective', 'growth-oriented', 'learning', 'patient'],
            'grief': ['respectful', 'memorial', 'honoring', 'peaceful', 'accepting'],
            
            # 모호한 감정 스타일
            'confusion': ['maze-like', 'questioning', 'exploratory', 'seeking'],
            'curiosity': ['detailed', 'intricate', 'discovering', 'investigating'],
            'surprise': ['unexpected', 'revealing', 'illuminating', 'awakening'],
            'realization': ['clarifying', 'enlightening', 'revealing', 'understanding'],
            
            # 중립
            'neutral': ['balanced', 'centered', 'stable', 'harmonious']
        }
        
        # 치료적 키워드 (점수가 높을수록 치료적 가치 높음)
        self.therapeutic_keywords = {
            # 높은 치료적 가치 (1.0)
            'healing': 1.0, 'growth': 1.0, 'transformation': 1.0, 'renewal': 1.0,
            'hope': 1.0, 'peace': 1.0, 'balance': 1.0, 'harmony': 1.0,
            'acceptance': 1.0, 'mindfulness': 1.0, 'compassion': 1.0,
            
            # 중간 치료적 가치 (0.7)
            'calm': 0.7, 'gentle': 0.7, 'soft': 0.7, 'warm': 0.7,
            'light': 0.7, 'flowing': 0.7, 'serene': 0.7, 'tranquil': 0.7,
            'nurturing': 0.7, 'supportive': 0.7, 'comforting': 0.7,
            
            # 기본 치료적 가치 (0.5)
            'reflection': 0.5, 'journey': 0.5, 'path': 0.5, 'bridge': 0.5,
            'landscape': 0.5, 'nature': 0.5, 'organic': 0.5, 'natural': 0.5,
            
            # 예술적 가치 (0.3)
            'artistic': 0.3, 'creative': 0.3, 'expressive': 0.3, 'beautiful': 0.3,
            'aesthetic': 0.3, 'composition': 0.3, 'style': 0.3
        }
        
        # 부정적 키워드 (치료적으로 부적절한 요소들)
        self.negative_keywords = {
            'violent', 'aggressive', 'harsh', 'destructive', 'chaotic',
            'overwhelming', 'suffocating', 'oppressive', 'threatening',
            'disturbing', 'frightening', 'terrifying'
        }

    def calculate_comprehensive_quality_score(
        self, 
        gallery_item: Dict[str, Any],
        gpt_metadata: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """종합 품질 점수 계산 (사용자 반응 없이)"""
        
        # 1. GPT 품질 점수 (50%)
        gpt_quality = self._calculate_gpt_quality_score(gallery_item, gpt_metadata)
        
        # 2. 감정-스타일 일치도 (30%)
        emotion_style_match = self._calculate_emotion_style_consistency(gallery_item)
        
        # 3. 치료적 키워드 점수 (20%)
        therapeutic_score = self._calculate_therapeutic_keyword_score(gallery_item)
        
        # 종합 점수 계산
        total_score = (
            gpt_quality * 0.5 +
            emotion_style_match * 0.3 + 
            therapeutic_score * 0.2
        )
        
        return {
            'total_score': total_score,
            'gpt_quality_score': gpt_quality,
            'emotion_style_match': emotion_style_match,
            'therapeutic_score': therapeutic_score,
            'components': {
                'gpt_quality_weight': 0.5,
                'emotion_style_weight': 0.3,
                'therapeutic_weight': 0.2
            }
        }

    def _calculate_gpt_quality_score(
        self, 
        gallery_item: Dict[str, Any],
        gpt_metadata: Dict[str, Any] = None
    ) -> float:
        """GPT 품질 점수 계산"""
        
        if not gpt_metadata:
            gpt_metadata = gallery_item.get('gpt_metadata', {})
        
        # 기본 GPT 품질 메트릭들 (이미 구현된 것들 활용)
        prompt_quality = gpt_metadata.get('prompt_quality_score', 0.5)
        curator_quality = gpt_metadata.get('curator_quality_score', 0.5)
        personalization = gpt_metadata.get('personalization_score', 0.0)
        safety_level = gpt_metadata.get('safety_level', 'medium')
        
        # 안전성 점수 변환
        safety_score = {
            'safe': 1.0,
            'medium': 0.7,
            'warning': 0.3,
            'unsafe': 0.0
        }.get(safety_level, 0.7)
        
        # GPT 종합 품질 점수
        gpt_score = (
            prompt_quality * 0.4 +
            curator_quality * 0.3 +
            personalization * 0.2 +
            safety_score * 0.1
        )
        
        return min(1.0, gpt_score)

    def _calculate_emotion_style_consistency(self, gallery_item: Dict[str, Any]) -> float:
        """감정-스타일 일치도 계산"""
        
        emotion_keywords = gallery_item.get('emotion_keywords', [])
        reflection_prompt = gallery_item.get('reflection_prompt', '')
        
        if not emotion_keywords or not reflection_prompt:
            return 0.5  # 기본 점수
        
        prompt_lower = reflection_prompt.lower()
        total_consistency = 0.0
        matched_emotions = 0
        
        for emotion in emotion_keywords[:3]:  # 상위 3개 감정만 평가
            if emotion in self.emotion_style_mapping:
                expected_styles = self.emotion_style_mapping[emotion]
                
                # 프롬프트에서 해당 감정에 맞는 스타일 키워드 찾기
                style_matches = 0
                for style in expected_styles:
                    if style in prompt_lower:
                        style_matches += 1
                
                # 매칭 비율 계산
                emotion_consistency = min(1.0, style_matches / len(expected_styles))
                total_consistency += emotion_consistency
                matched_emotions += 1
        
        if matched_emotions == 0:
            return 0.5
        
        return total_consistency / matched_emotions

    def _calculate_therapeutic_keyword_score(self, gallery_item: Dict[str, Any]) -> float:
        """치료적 키워드 포함도 점수 계산"""
        
        reflection_prompt = gallery_item.get('reflection_prompt', '')
        docent_message = gallery_item.get('docent_message', {})
        
        # 분석할 텍스트 수집
        texts_to_analyze = [reflection_prompt]
        
        if isinstance(docent_message, dict):
            content = docent_message.get('content', {})
            if isinstance(content, dict):
                for section in content.values():
                    if isinstance(section, str):
                        texts_to_analyze.append(section)
        
        combined_text = ' '.join(texts_to_analyze).lower()
        
        if not combined_text.strip():
            return 0.3  # 기본 점수
        
        # 치료적 키워드 점수 계산
        therapeutic_score = 0.0
        keyword_count = 0
        
        for keyword, value in self.therapeutic_keywords.items():
            if keyword in combined_text:
                therapeutic_score += value
                keyword_count += 1
        
        # 부정적 키워드 페널티
        negative_penalty = 0.0
        for neg_keyword in self.negative_keywords:
            if neg_keyword in combined_text:
                negative_penalty += 0.2  # 부정적 키워드당 0.2 감점
        
        # 최종 점수 계산
        if keyword_count > 0:
            avg_therapeutic_score = therapeutic_score / keyword_count
        else:
            avg_therapeutic_score = 0.3  # 키워드 없으면 기본 점수
        
        # 페널티 적용
        final_score = max(0.0, avg_therapeutic_score - negative_penalty)
        
        return min(1.0, final_score)

    def analyze_training_data_quality(
        self, 
        gallery_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """훈련 데이터 전체의 품질 분석"""
        
        if not gallery_items:
            return {'error': 'No training data provided'}
        
        quality_scores = []
        quality_distribution = {'high': 0, 'medium': 0, 'low': 0}
        component_averages = {
            'gpt_quality': [],
            'emotion_style_match': [],
            'therapeutic_score': []
        }
        
        for item in gallery_items:
            quality_result = self.calculate_comprehensive_quality_score(item)
            total_score = quality_result['total_score']
            quality_scores.append(total_score)
            
            # 품질 분포 계산
            if total_score >= 0.7:
                quality_distribution['high'] += 1
            elif total_score >= 0.4:
                quality_distribution['medium'] += 1
            else:
                quality_distribution['low'] += 1
            
            # 컴포넌트별 점수 수집
            component_averages['gpt_quality'].append(quality_result['gpt_quality_score'])
            component_averages['emotion_style_match'].append(quality_result['emotion_style_match'])
            component_averages['therapeutic_score'].append(quality_result['therapeutic_score'])
        
        # 분석 결과
        analysis = {
            'total_items': len(gallery_items),
            'average_quality': np.mean(quality_scores),
            'quality_std': np.std(quality_scores),
            'min_quality': np.min(quality_scores),
            'max_quality': np.max(quality_scores),
            'quality_distribution': quality_distribution,
            'quality_distribution_percent': {
                'high': quality_distribution['high'] / len(gallery_items) * 100,
                'medium': quality_distribution['medium'] / len(gallery_items) * 100,
                'low': quality_distribution['low'] / len(gallery_items) * 100
            },
            'component_averages': {
                'gpt_quality': np.mean(component_averages['gpt_quality']),
                'emotion_style_match': np.mean(component_averages['emotion_style_match']),
                'therapeutic_score': np.mean(component_averages['therapeutic_score'])
            },
            'high_quality_threshold': 0.7,
            'minimum_quality_threshold': 0.4,
            'recommended_for_training': quality_distribution['high'] + quality_distribution['medium']
        }
        
        logger.info(f"Training data quality analysis: {analysis['average_quality']:.3f} average, "
                   f"{analysis['quality_distribution_percent']['high']:.1f}% high quality")
        
        return analysis

    def get_training_weight(self, quality_score: float) -> float:
        """품질 점수를 기반으로 훈련 가중치 계산"""
        
        if quality_score >= 0.8:
            return 1.5  # 최고 품질
        elif quality_score >= 0.6:
            return 1.2  # 고품질
        elif quality_score >= 0.4:
            return 1.0  # 표준
        elif quality_score >= 0.2:
            return 0.7  # 낮은 품질
        else:
            return 0.3  # 매우 낮은 품질

    def filter_training_data_by_quality(
        self, 
        gallery_items: List[Dict[str, Any]], 
        min_quality_threshold: float = 0.4
    ) -> List[Dict[str, Any]]:
        """품질 기준으로 훈련 데이터 필터링"""
        
        filtered_items = []
        
        for item in gallery_items:
            quality_result = self.calculate_comprehensive_quality_score(item)
            
            if quality_result['total_score'] >= min_quality_threshold:
                # 품질 메타데이터 추가
                item['quality_metadata'] = quality_result
                item['training_weight'] = self.get_training_weight(quality_result['total_score'])
                filtered_items.append(item)
        
        logger.info(f"Filtered training data: {len(filtered_items)}/{len(gallery_items)} items "
                   f"meet quality threshold {min_quality_threshold}")
        
        return filtered_items