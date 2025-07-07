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


class UserEmotionProfile:
    """사용자 감정 프로파일 및 개인화 데이터 관리"""

    def __init__(self, user_id: str, db_path: str = "user_profiles.db"):
        self.user_id = user_id
        self.db_path = db_path
        self.emotion_history: List[Dict] = []
        self.feedback_history: List[Dict] = []

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
        """피드백 점수를 기반으로 선호도 업데이트"""
        learning_rate = 0.1

        if feedback_score > 3.0:  # 긍정적 피드백 (1-5 척도)
            weight = (feedback_score - 3.0) / 2.0 * learning_rate

            # 최근 감정 기반 선호도 조정
            if self.emotion_history:
                recent_emotion = self.emotion_history[-1]["emotion"]

                # Valence 기반 밝기/채도 조정
                if recent_emotion.valence > 0:
                    self.preference_weights["brightness"] += weight * 0.1
                    self.preference_weights["saturation"] += weight * 0.1
                else:
                    self.preference_weights["brightness"] -= weight * 0.05
                    self.preference_weights["saturation"] -= weight * 0.05

                # Arousal 기반 대비/복잡성 조정
                if recent_emotion.arousal > 0:
                    self.preference_weights["contrast"] += weight * 0.1
                    self.preference_weights["complexity"] += weight * 0.05
                else:
                    self.preference_weights["contrast"] -= weight * 0.05
                    self.preference_weights["complexity"] -= weight * 0.1

        # 범위 제한
        for key in self.preference_weights:
            if isinstance(self.preference_weights[key], (int, float)):
                self.preference_weights[key] = np.clip(
                    self.preference_weights[key], -1.0, 1.0
                )

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
        """개인화된 스타일 수정자 생성"""
        modifiers = []

        # 색온도 기반
        if self.preference_weights["color_temperature"] > 0.3:
            modifiers.append("warm lighting")
        elif self.preference_weights["color_temperature"] < -0.3:
            modifiers.append("cool lighting")

        # 밝기 기반
        if self.preference_weights["brightness"] > 0.3:
            modifiers.append("bright")
        elif self.preference_weights["brightness"] < -0.3:
            modifiers.append("dim lighting")

        # 채도 기반
        if self.preference_weights["saturation"] > 0.3:
            modifiers.append("vibrant colors")
        elif self.preference_weights["saturation"] < -0.3:
            modifiers.append("muted colors")

        # 대비 기반
        if self.preference_weights["contrast"] > 0.3:
            modifiers.append("high contrast")
        elif self.preference_weights["contrast"] < -0.3:
            modifiers.append("soft contrast")

        # 복잡성 기반
        if self.preference_weights["complexity"] > 0.3:
            modifiers.append("detailed")
        elif self.preference_weights["complexity"] < -0.3:
            modifiers.append("minimalist")

        # 아트 스타일
        modifiers.append(f"{self.preference_weights['art_style']} style")

        return ", ".join(modifiers)

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