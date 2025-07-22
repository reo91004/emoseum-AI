# src/user_manager.py

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class PsychometricResult:
    """심리검사 결과"""

    phq9_score: int = 0
    cesd_score: int = 0
    meaq_score: int = 0  # 감정 회피 척도
    ciss_score: int = 0  # 대처 스타일
    coping_style: str = "balanced"  # confrontational, avoidant, balanced
    severity_level: str = "mild"  # mild, moderate, severe
    test_date: str = ""


@dataclass
class VisualPreferences:
    """시각적 선호도"""

    art_style: str = "painting"  # painting, photography, abstract
    color_tone: str = "warm"  # warm, cool, pastel
    complexity: str = "balanced"  # simple, balanced, complex
    brightness: float = 0.5  # 0.0-1.0
    saturation: float = 0.5  # 0.0-1.0
    style_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.style_weights is None:
            self.style_weights = {
                "painting": 0.33,
                "photography": 0.33,
                "abstract": 0.34,
            }


@dataclass
class User:
    """사용자 정보"""

    user_id: str
    created_date: str
    psychometric_results: List[PsychometricResult] = None
    visual_preferences: VisualPreferences = None
    last_updated: str = ""

    def __post_init__(self):
        if self.psychometric_results is None:
            self.psychometric_results = []
        if self.visual_preferences is None:
            self.visual_preferences = VisualPreferences()
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()


class UserManager:
    """사용자 프로필 관리자"""

    def __init__(
        self, db_path: str = "data/users.db", preferences_dir: str = "data/preferences"
    ):
        self.db_path = Path(db_path)
        self.preferences_dir = Path(preferences_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.preferences_dir.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 사용자 테이블
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                created_date TEXT NOT NULL,
                last_updated TEXT NOT NULL
            )
        """
        )

        # 심리검사 결과 테이블
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS psychometric_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                phq9_score INTEGER,
                cesd_score INTEGER,
                meaq_score INTEGER,
                ciss_score INTEGER,
                coping_style TEXT,
                severity_level TEXT,
                test_date TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        """
        )

        conn.commit()
        conn.close()
        logger.info("사용자 데이터베이스가 초기화되었습니다.")

    def create_user(self, user_id: str) -> User:
        """새 사용자 생성"""
        user = User(user_id=user_id, created_date=datetime.now().isoformat())

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO users (user_id, created_date, last_updated)
            VALUES (?, ?, ?)
        """,
            (user.user_id, user.created_date, user.last_updated),
        )

        conn.commit()
        conn.close()

        # 기본 선호도 파일 생성
        self._save_user_preferences(user)

        logger.info(f"새 사용자가 생성되었습니다: {user_id}")
        return user

    def get_user(self, user_id: str) -> Optional[User]:
        """사용자 정보 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 기본 사용자 정보
        cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        user_row = cursor.fetchone()

        if not user_row:
            conn.close()
            return None

        user = User(
            user_id=user_row[0], created_date=user_row[1], last_updated=user_row[2]
        )

        # 심리검사 결과들
        cursor.execute(
            """
            SELECT * FROM psychometric_results 
            WHERE user_id = ? 
            ORDER BY test_date DESC
        """,
            (user_id,),
        )

        results = []
        for row in cursor.fetchall():
            result = PsychometricResult(
                phq9_score=row[2],
                cesd_score=row[3],
                meaq_score=row[4],
                ciss_score=row[5],
                coping_style=row[6],
                severity_level=row[7],
                test_date=row[8],
            )
            results.append(result)

        user.psychometric_results = results
        conn.close()

        # 시각 선호도 로드
        user.visual_preferences = self._load_user_preferences(user_id)

        return user

    def conduct_psychometric_test(
        self,
        user_id: str,
        phq9_score: int,
        cesd_score: int,
        meaq_score: int,
        ciss_score: int,
    ) -> PsychometricResult:
        """심리검사 실시 및 결과 저장"""

        # 대처 스타일 결정
        coping_style = self._determine_coping_style(meaq_score, ciss_score)

        # 심각도 결정
        severity_level = self._determine_severity_level(phq9_score, cesd_score)

        result = PsychometricResult(
            phq9_score=phq9_score,
            cesd_score=cesd_score,
            meaq_score=meaq_score,
            ciss_score=ciss_score,
            coping_style=coping_style,
            severity_level=severity_level,
            test_date=datetime.now().isoformat(),
        )

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO psychometric_results 
            (user_id, phq9_score, cesd_score, meaq_score, ciss_score, 
             coping_style, severity_level, test_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                user_id,
                phq9_score,
                cesd_score,
                meaq_score,
                ciss_score,
                coping_style,
                severity_level,
                result.test_date,
            ),
        )

        conn.commit()
        conn.close()

        logger.info(
            f"사용자 {user_id}의 심리검사가 완료되었습니다: {coping_style}, {severity_level}"
        )
        return result

    def _determine_coping_style(self, meaq_score: int, ciss_score: int) -> str:
        """대처 스타일 결정"""
        # MEAQ: 감정회피척도 (높을수록 회피적)
        # CISS: 대처스타일척도 (높을수록 직면적)

        avoidance_tendency = meaq_score / 100.0  # 0-1 정규화
        confrontation_tendency = ciss_score / 100.0

        if avoidance_tendency > 0.7:
            return "avoidant"
        elif confrontation_tendency > 0.7:
            return "confrontational"
        else:
            return "balanced"

    def _determine_severity_level(self, phq9_score: int, cesd_score: int) -> str:
        """심각도 결정"""
        # PHQ-9: 0-4(minimal), 5-9(mild), 10-14(moderate), 15-19(moderate-severe), 20+(severe)
        # CES-D: 0-15(normal), 16-23(mild), 24+(moderate-severe)

        if phq9_score >= 15 or cesd_score >= 24:
            return "severe"
        elif phq9_score >= 10 or cesd_score >= 16:
            return "moderate"
        else:
            return "mild"

    def set_visual_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """시각적 선호도 설정"""
        visual_prefs = VisualPreferences(
            art_style=preferences.get("art_style", "painting"),
            color_tone=preferences.get("color_tone", "warm"),
            complexity=preferences.get("complexity", "balanced"),
            brightness=preferences.get("brightness", 0.5),
            saturation=preferences.get("saturation", 0.5),
            style_weights=preferences.get(
                "style_weights",
                {"painting": 0.33, "photography": 0.33, "abstract": 0.34},
            ),
        )

        # preferences.json 파일에 저장
        pref_file = self.preferences_dir / f"{user_id}_preferences.json"
        with open(pref_file, "w", encoding="utf-8") as f:
            json.dump(asdict(visual_prefs), f, indent=2, ensure_ascii=False)

        logger.info(f"사용자 {user_id}의 시각적 선호도가 설정되었습니다.")

    def _load_user_preferences(self, user_id: str) -> VisualPreferences:
        """사용자 선호도 로드"""
        pref_file = self.preferences_dir / f"{user_id}_preferences.json"

        if pref_file.exists():
            try:
                with open(pref_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return VisualPreferences(**data)
            except Exception as e:
                logger.warning(f"선호도 파일 로드 실패: {e}")

        return VisualPreferences()

    def _save_user_preferences(self, user: User):
        """사용자 선호도 저장"""
        pref_file = self.preferences_dir / f"{user.user_id}_preferences.json"
        with open(pref_file, "w", encoding="utf-8") as f:
            json.dump(asdict(user.visual_preferences), f, indent=2, ensure_ascii=False)

    def update_preference_weights(self, user_id: str, weight_updates: Dict[str, float]):
        """선호도 가중치 업데이트"""
        user = self.get_user(user_id)
        if not user:
            return

        # 기존 가중치 업데이트
        for key, delta in weight_updates.items():
            if key in user.visual_preferences.style_weights:
                user.visual_preferences.style_weights[key] += delta
                # 0-1 범위로 클리핑
                user.visual_preferences.style_weights[key] = max(
                    0.0, min(1.0, user.visual_preferences.style_weights[key])
                )

        # 가중치 정규화
        total = sum(user.visual_preferences.style_weights.values())
        if total > 0:
            for key in user.visual_preferences.style_weights:
                user.visual_preferences.style_weights[key] /= total

        self._save_user_preferences(user)
        logger.info(f"사용자 {user_id}의 선호도 가중치가 업데이트되었습니다.")

    def should_conduct_periodic_test(self, user_id: str) -> bool:
        """주기적 검사 필요 여부 확인"""
        user = self.get_user(user_id)
        if not user or not user.psychometric_results:
            return True

        last_test = user.psychometric_results[0]  # 가장 최근 검사
        last_test_date = datetime.fromisoformat(last_test.test_date)

        # 2주 경과 확인
        return datetime.now() - last_test_date >= timedelta(weeks=2)

    def get_current_coping_style(self, user_id: str) -> str:
        """현재 대처 스타일 반환"""
        user = self.get_user(user_id)
        if user and user.psychometric_results:
            return user.psychometric_results[0].coping_style
        return "balanced"

    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """사용자 통계 반환"""
        user = self.get_user(user_id)
        if not user:
            return {}

        stats = {
            "user_id": user_id,
            "member_since": user.created_date,
            "test_count": len(user.psychometric_results),
            "current_coping_style": self.get_current_coping_style(user_id),
            "needs_periodic_test": self.should_conduct_periodic_test(user_id),
        }

        if user.psychometric_results:
            latest = user.psychometric_results[0]
            stats.update(
                {
                    "latest_phq9": latest.phq9_score,
                    "latest_severity": latest.severity_level,
                    "last_test_date": latest.test_date,
                }
            )

        return stats
