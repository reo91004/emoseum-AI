#!/usr/bin/env python3
"""
UserEmotionProfile - 사용자 감정 프로파일 및 개인화 데이터 관리
"""

import os
import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np

from config import logger
from models.emotion import EmotionEmbedding
from models.adaptive_personalization import AdaptivePersonalizationSystem


class UserEmotionProfile:
    """사용자 감정 프로파일 및 개인화 데이터 관리"""

    def __init__(self, user_id: str, db_path: str = "user_profiles.db"):
        self.user_id = user_id
        self.db_path = db_path
        self.emotion_history: List[Dict] = []
        self.feedback_history: List[Dict] = []
        
        # 적응형 개인화 시스템 (클래스 레벨에서 공유)
        if not hasattr(UserEmotionProfile, '_adaptive_system'):
            UserEmotionProfile._adaptive_system = AdaptivePersonalizationSystem()
        self.adaptive_system = UserEmotionProfile._adaptive_system

        # 개인화 선호도 가중치
        self.preference_weights = {
            "color_temperature": 0.0,  # -1.0 (차가운) to 1.0 (따뜻한)
            "brightness": 0.0,  # -1.0 (어두운) to 1.0 (밝은)
            "saturation": 0.0,  # -1.0 (무채색) to 1.0 (채도 높은)
            "contrast": 0.0,  # -1.0 (낮은 대비) to 1.0 (높은 대비)
            "complexity": 0.0,  # -1.0 (단순) to 1.0 (복잡)
            "art_style": "realistic",  # realistic, abstract, impressionist, minimalist
            "composition": "balanced",  # minimal, balanced, complex
        }

        # 치료 진행도 지표
        self.therapeutic_progress = {
            "mood_trend": 0.0,  # 감정 변화 트렌드
            "stability_score": 0.0,  # 감정 안정성
            "engagement_level": 0.0,  # 참여도
            "recovery_indicator": 0.0,  # 회복 지표
        }

        # 학습 메타데이터
        self.learning_metadata = {
            "total_interactions": 0,
            "positive_feedback_rate": 0.0,
            "last_training_date": None,
            "model_version": 1,
        }

        self._init_database()
        self._load_profile()

    def _init_database(self):
        """SQLite 데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 감정 히스토리 테이블
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS emotion_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                input_text TEXT,
                valence REAL,
                arousal REAL,
                dominance REAL,
                confidence REAL,
                generated_prompt TEXT,
                image_path TEXT
            )
        """
        )

        # 피드백 히스토리 테이블
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                emotion_id INTEGER,
                timestamp TEXT NOT NULL,
                feedback_score REAL,
                feedback_type TEXT,
                comments TEXT,
                FOREIGN KEY (emotion_id) REFERENCES emotion_history (id)
            )
        """
        )

        # 사용자 프로파일 테이블
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                preference_weights TEXT,
                therapeutic_progress TEXT,
                learning_metadata TEXT,
                last_updated TEXT
            )
        """
        )

        conn.commit()
        conn.close()
        logger.info(f"✅ 사용자 {self.user_id} 데이터베이스 초기화 완료")

    def _load_profile(self):
        """프로파일 데이터 로드"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # 감정 히스토리 로드 (최근 50개)
            cursor.execute(
                """
                SELECT * FROM emotion_history 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 50
            """,
                (self.user_id,),
            )

            emotion_rows = cursor.fetchall()
            for row in emotion_rows:
                self.emotion_history.append(
                    {
                        "id": row[0],
                        "timestamp": row[2],
                        "input_text": row[3],
                        "emotion": EmotionEmbedding(row[4], row[5], row[6], row[7]),
                        "generated_prompt": row[8],
                        "image_path": row[9],
                    }
                )

            # 피드백 히스토리 로드
            cursor.execute(
                """
                SELECT * FROM feedback_history 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 50
            """,
                (self.user_id,),
            )

            feedback_rows = cursor.fetchall()
            for row in feedback_rows:
                self.feedback_history.append(
                    {
                        "id": row[0],
                        "emotion_id": row[2],
                        "timestamp": row[3],
                        "feedback_score": row[4],
                        "feedback_type": row[5],
                        "comments": row[6],
                    }
                )

            # 프로파일 설정 로드
            cursor.execute(
                "SELECT * FROM user_profiles WHERE user_id = ?", (self.user_id,)
            )
            profile_row = cursor.fetchone()

            if profile_row:
                self.preference_weights.update(json.loads(profile_row[1]))
                self.therapeutic_progress.update(json.loads(profile_row[2]))
                self.learning_metadata.update(json.loads(profile_row[3]))

            logger.info(
                f"✅ 사용자 {self.user_id} 프로파일 로드: 감정 {len(self.emotion_history)}개, 피드백 {len(self.feedback_history)}개"
            )

        except Exception as e:
            logger.error(f"❌ 프로파일 로드 실패: {e}")
        finally:
            conn.close()

    def add_emotion_record(
        self,
        input_text: str,
        emotion: EmotionEmbedding,
        generated_prompt: str,
        image_path: str = None,
    ) -> int:
        """감정 기록 추가"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            timestamp = datetime.now().isoformat()
            cursor.execute(
                """
                INSERT INTO emotion_history 
                (user_id, timestamp, input_text, valence, arousal, dominance, confidence, generated_prompt, image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    self.user_id,
                    timestamp,
                    input_text,
                    emotion.valence,
                    emotion.arousal,
                    emotion.dominance,
                    emotion.confidence,
                    generated_prompt,
                    image_path,
                ),
            )

            emotion_id = cursor.lastrowid
            conn.commit()

            # 메모리에도 추가
            self.emotion_history.append(
                {
                    "id": emotion_id,
                    "timestamp": timestamp,
                    "input_text": input_text,
                    "emotion": emotion,
                    "generated_prompt": generated_prompt,
                    "image_path": image_path,
                }
            )

            logger.info(f"✅ 감정 기록 추가: ID {emotion_id}")
            return emotion_id

        except Exception as e:
            logger.error(f"❌ 감정 기록 추가 실패: {e}")
            return -1
        finally:
            conn.close()

    def add_feedback(
        self,
        emotion_id: int,
        feedback_score: float,
        feedback_type: str = "rating",
        comments: str = None,
    ):
        """피드백 추가"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            timestamp = datetime.now().isoformat()
            cursor.execute(
                """
                INSERT INTO feedback_history 
                (user_id, emotion_id, timestamp, feedback_score, feedback_type, comments)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    self.user_id,
                    emotion_id,
                    timestamp,
                    feedback_score,
                    feedback_type,
                    comments,
                ),
            )

            conn.commit()

            # 메모리에도 추가
            self.feedback_history.append(
                {
                    "emotion_id": emotion_id,
                    "timestamp": timestamp,
                    "feedback_score": feedback_score,
                    "feedback_type": feedback_type,
                    "comments": comments,
                }
            )

            # 개인화 선호도 업데이트
            self._update_preferences_from_feedback(feedback_score)

            # 치료 진행도 업데이트
            self._update_therapeutic_progress()

            # 프로파일 저장
            self._save_profile()

            logger.info(f"✅ 피드백 추가: 감정 ID {emotion_id}, 점수 {feedback_score}")

        except Exception as e:
            logger.error(f"❌ 피드백 추가 실패: {e}")
        finally:
            conn.close()

    def _update_preferences_from_feedback(self, feedback_score: float):
        """적응형 개인화 시스템을 사용한 선호도 업데이트"""
        
        if not self.emotion_history:
            logger.warning("감정 히스토리가 없어 선호도 업데이트를 건너뜁니다")
            return
        
        try:
            # 최근 감정 기록 가져오기
            recent_record = self.emotion_history[-1]
            recent_emotion = recent_record["emotion"]
            
            # 이미지 메타데이터 준비 (실제 이미지 분석 결과가 있다면 사용)
            image_metadata = recent_record.get("image_metadata", {
                "brightness": 0.5,
                "saturation": 0.5, 
                "contrast": 0.5,
                "hue_variance": 0.3,
                "edge_density": 0.4,
                "color_diversity": 0.6,
                "composition_balance": 0.5,
                "texture_complexity": 0.4
            })
            
            # 프롬프트 정보
            prompt = recent_record.get("generated_prompt", "")
            
            # 적응형 시스템에 피드백 추가
            self.adaptive_system.add_feedback(
                user_id=self.user_id,
                emotion=recent_emotion,
                image_metadata=image_metadata,
                prompt=prompt,
                feedback_score=feedback_score
            )
            
            # 업데이트된 선호도 가져오기
            updated_preferences = self.adaptive_system.get_user_preferences(self.user_id)
            
            # 기존 선호도 구조 업데이트
            for key in self.preference_weights:
                if isinstance(self.preference_weights[key], (int, float)):
                    if key in updated_preferences:
                        self.preference_weights[key] = updated_preferences[key]
            
            logger.info(f"✅ 적응형 개인화 시스템으로 사용자 {self.user_id} 선호도 업데이트 완료")
            
        except Exception as e:
            logger.error(f"❌ 적응형 선호도 업데이트 실패: {e}")
            # Fallback: 기존 하드코딩된 방식 사용
            self._fallback_preference_update(feedback_score)
    
    def _fallback_preference_update(self, feedback_score: float):
        """기존 하드코딩된 방식으로 폴백 (적응형 시스템 실패시)"""
        
        learning_rate = 0.05  # 더 보수적인 학습률
        
        if abs(feedback_score - 3.0) < 0.5:  # 중성 피드백은 무시
            return
        
        if feedback_score > 3.0:  # 긍정적 피드백
            weight = (feedback_score - 3.0) / 2.0 * learning_rate
            
            if self.emotion_history:
                recent_emotion = self.emotion_history[-1]["emotion"]
                
                # 더 세밀한 조정
                emotion_strength = abs(recent_emotion.valence) + abs(recent_emotion.arousal) + abs(recent_emotion.dominance)
                adaptive_weight = weight * min(emotion_strength, 1.0)
                
                # Valence 기반 조정 (더 보수적)
                if recent_emotion.valence > 0.2:
                    self.preference_weights["brightness"] += adaptive_weight * 0.05
                    self.preference_weights["color_temperature"] += adaptive_weight * 0.03
                elif recent_emotion.valence < -0.2:
                    self.preference_weights["brightness"] -= adaptive_weight * 0.03
                    self.preference_weights["saturation"] -= adaptive_weight * 0.02
                
                # Arousal 기반 조정
                if recent_emotion.arousal > 0.2:
                    self.preference_weights["contrast"] += adaptive_weight * 0.04
                    self.preference_weights["saturation"] += adaptive_weight * 0.03
                elif recent_emotion.arousal < -0.2:
                    self.preference_weights["contrast"] -= adaptive_weight * 0.02
                    self.preference_weights["complexity"] -= adaptive_weight * 0.03
                
                # Dominance 기반 조정
                if recent_emotion.dominance > 0.2:
                    self.preference_weights["complexity"] += adaptive_weight * 0.03
                elif recent_emotion.dominance < -0.2:
                    self.preference_weights["complexity"] -= adaptive_weight * 0.02
        
        # 범위 제한
        for key in self.preference_weights:
            if isinstance(self.preference_weights[key], (int, float)):
                self.preference_weights[key] = np.clip(
                    self.preference_weights[key], -1.0, 1.0
                )
        
        logger.info(f"⚠️ 폴백 방식으로 사용자 {self.user_id} 선호도 업데이트")

    def _update_therapeutic_progress(self):
        """치료 진행도 업데이트"""
        if len(self.emotion_history) < 3:
            return

        # 최근 감정들의 Valence 트렌드 분석
        recent_valences = [
            entry["emotion"].valence for entry in self.emotion_history[-10:]
        ]
        if len(recent_valences) >= 3:
            # 선형 회귀로 트렌드 계산
            x = np.arange(len(recent_valences))
            y = np.array(recent_valences)
            slope = np.corrcoef(x, y)[0, 1] if len(recent_valences) > 1 else 0
            self.therapeutic_progress["mood_trend"] = slope

        # 감정 안정성 (변동성의 역수)
        if len(recent_valences) >= 5:
            stability = 1.0 / (1.0 + np.std(recent_valences))
            self.therapeutic_progress["stability_score"] = stability

        # 참여도 (피드백 제공률)
        if self.feedback_history:
            recent_interactions = len(self.emotion_history[-20:])
            recent_feedbacks = len(
                [
                    f
                    for f in self.feedback_history[-20:]
                    if f["emotion_id"] in [e["id"] for e in self.emotion_history[-20:]]
                ]
            )
            engagement = recent_feedbacks / max(1, recent_interactions)
            self.therapeutic_progress["engagement_level"] = engagement

        # 회복 지표 (긍정적 피드백 비율 + 감정 트렌드)
        if self.feedback_history:
            positive_feedbacks = len(
                [f for f in self.feedback_history[-20:] if f["feedback_score"] > 3.0]
            )
            total_feedbacks = len(self.feedback_history[-20:])
            positive_rate = positive_feedbacks / max(1, total_feedbacks)

            recovery = (
                positive_rate
                + max(0, self.therapeutic_progress["mood_trend"])
                + self.therapeutic_progress["stability_score"]
            ) / 3.0
            self.therapeutic_progress["recovery_indicator"] = recovery

    def _save_profile(self):
        """프로파일 데이터 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            timestamp = datetime.now().isoformat()
            cursor.execute(
                """
                INSERT OR REPLACE INTO user_profiles 
                (user_id, preference_weights, therapeutic_progress, learning_metadata, last_updated)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    self.user_id,
                    json.dumps(self.preference_weights),
                    json.dumps(self.therapeutic_progress),
                    json.dumps(self.learning_metadata),
                    timestamp,
                ),
            )

            conn.commit()
            logger.info(f"✅ 사용자 {self.user_id} 프로파일 저장 완료")

        except Exception as e:
            logger.error(f"❌ 프로파일 저장 실패: {e}")
        finally:
            conn.close()

    def get_personalized_style_modifiers(self) -> str:
        """적응형 개인화 기반 스타일 수정자 생성"""
        modifiers = []
        
        # 더 세밀한 임계값과 다양한 표현 사용
        
        # 색온도 기반 (더 세밀한 구분)
        temp_val = self.preference_weights["color_temperature"]
        if temp_val > 0.5:
            modifiers.append("warm golden lighting")
        elif temp_val > 0.2:
            modifiers.append("warm lighting")
        elif temp_val < -0.5:
            modifiers.append("cool blue lighting")
        elif temp_val < -0.2:
            modifiers.append("cool lighting")

        # 밝기 기반 (감정 상태 고려)
        brightness_val = self.preference_weights["brightness"]
        if brightness_val > 0.5:
            modifiers.append("bright, luminous")
        elif brightness_val > 0.2:
            modifiers.append("well-lit")
        elif brightness_val < -0.5:
            modifiers.append("moody, dim lighting")
        elif brightness_val < -0.2:
            modifiers.append("soft lighting")

        # 채도 기반 (개인 취향 반영)
        saturation_val = self.preference_weights["saturation"]
        if saturation_val > 0.5:
            modifiers.append("vibrant, rich colors")
        elif saturation_val > 0.2:
            modifiers.append("colorful")
        elif saturation_val < -0.5:
            modifiers.append("desaturated, muted palette")
        elif saturation_val < -0.2:
            modifiers.append("subtle colors")

        # 대비 기반 (드라마틱함 조절)
        contrast_val = self.preference_weights["contrast"]
        if contrast_val > 0.5:
            modifiers.append("dramatic contrast")
        elif contrast_val > 0.2:
            modifiers.append("clear contrast")
        elif contrast_val < -0.5:
            modifiers.append("soft, gentle contrast")
        elif contrast_val < -0.2:
            modifiers.append("low contrast")

        # 복잡성 기반 (개인 선호도 반영)
        complexity_val = self.preference_weights["complexity"]
        if complexity_val > 0.5:
            modifiers.append("intricate, detailed")
        elif complexity_val > 0.2:
            modifiers.append("detailed")
        elif complexity_val < -0.5:
            modifiers.append("clean, minimalist")
        elif complexity_val < -0.2:
            modifiers.append("simple")

        # 아트 스타일 (다양화)
        art_style = self.preference_weights.get("art_style", "realistic")
        style_variations = {
            "realistic": ["photorealistic", "lifelike", "naturalistic"],
            "abstract": ["abstract", "conceptual", "non-representational"],
            "impressionist": ["impressionistic", "painterly", "expressive"],
            "minimalist": ["minimalist", "clean", "geometric"]
        }
        
        if art_style in style_variations:
            style_options = style_variations[art_style]
            # 다양성을 위해 첫 번째 옵션 사용 (나중에 랜덤화 가능)
            modifiers.append(f"{style_options[0]} style")
        else:
            modifiers.append(f"{art_style} style")

        # 구성 (composition) 반영
        composition = self.preference_weights.get("composition", "balanced")
        if composition == "complex":
            modifiers.append("dynamic composition")
        elif composition == "minimal":
            modifiers.append("simple composition")
        # balanced는 기본값이므로 추가하지 않음
        
        # 적응형 시스템의 예측 만족도 반영 (가능한 경우)
        try:
            if hasattr(self, 'adaptive_system') and self.emotion_history:
                recent_emotion = self.emotion_history[-1]["emotion"]
                recent_prompt = self.emotion_history[-1].get("generated_prompt", "")
                
                # 더미 이미지 메타데이터로 만족도 예측
                dummy_metadata = {"brightness": 0.5, "saturation": 0.5, "contrast": 0.5}
                predicted_satisfaction = self.adaptive_system.predict_satisfaction(
                    self.user_id, recent_emotion, dummy_metadata, recent_prompt
                )
                
                # 높은 만족도 예측시 품질 강조 수정자 추가
                if predicted_satisfaction > 4.0:
                    modifiers.append("high quality")
                elif predicted_satisfaction > 3.5:
                    modifiers.append("refined")
        except Exception:
            # 예측 실패시 무시
            pass

        return ", ".join(modifiers) if modifiers else "realistic style"

    def get_therapeutic_insights(self) -> Dict[str, Any]:
        """치료적 인사이트 제공"""
        if len(self.emotion_history) < 3:
            return {
                "message": "충분한 데이터가 수집되지 않았습니다.",
                "status": "insufficient_data",
            }

        insights = {
            "emotional_state": {
                "current_mood": self._get_current_mood_description(),
                "mood_trend": self.therapeutic_progress["mood_trend"],
                "stability": self.therapeutic_progress["stability_score"],
            },
            "progress_indicators": {
                "engagement_level": self.therapeutic_progress["engagement_level"],
                "recovery_indicator": self.therapeutic_progress["recovery_indicator"],
                "total_interactions": len(self.emotion_history),
                "feedback_count": len(self.feedback_history),
            },
            "recommendations": self._generate_recommendations(),
            "preference_summary": self.preference_weights,
        }

        return insights

    def _get_current_mood_description(self) -> str:
        """현재 기분 상태 설명"""
        if not self.emotion_history:
            return "데이터 없음"

        recent_emotions = [entry["emotion"] for entry in self.emotion_history[-5:]]
        avg_valence = np.mean([e.valence for e in recent_emotions])
        avg_arousal = np.mean([e.arousal for e in recent_emotions])

        if avg_valence > 0.3 and avg_arousal > 0.3:
            return "활기찬 긍정 상태"
        elif avg_valence > 0.3 and avg_arousal < -0.3:
            return "평온한 긍정 상태"
        elif avg_valence < -0.3 and avg_arousal > 0.3:
            return "불안한 부정 상태"
        elif avg_valence < -0.3 and avg_arousal < -0.3:
            return "우울한 상태"
        else:
            return "중성적 상태"

    def _generate_recommendations(self) -> List[str]:
        """개인화된 추천사항 생성"""
        recommendations = []

        # 감정 트렌드 기반
        if self.therapeutic_progress["mood_trend"] < -0.3:
            recommendations.append(
                "부정적인 감정 패턴이 감지되었습니다. 긍정적인 활동이나 이미지 생성을 시도해보세요."
            )
        elif self.therapeutic_progress["mood_trend"] > 0.3:
            recommendations.append(
                "감정 상태가 개선되고 있습니다. 현재 패턴을 유지하세요."
            )

        # 안정성 기반
        if self.therapeutic_progress["stability_score"] < 0.5:
            recommendations.append(
                "감정 변동이 큽니다. 규칙적인 사용과 일관된 피드백이 도움될 것입니다."
            )

        # 참여도 기반
        if self.therapeutic_progress["engagement_level"] < 0.3:
            recommendations.append(
                "더 자주 피드백을 제공하시면 개인화 효과가 향상됩니다."
            )

        # 기본 추천
        if not recommendations:
            recommendations.append(
                "현재 상태가 양호합니다. 지속적인 사용을 권장합니다."
            )

        return recommendations