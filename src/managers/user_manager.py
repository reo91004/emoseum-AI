# src/managers/user_manager.py

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

        # 기본 테이블 생성
        self._create_base_tables(cursor)

        # GPT 관련 마이그레이션 실행
        self._run_gpt_migrations(cursor)

        conn.commit()
        conn.close()
        logger.info("사용자 데이터베이스가 초기화되었습니다.")

    def _create_base_tables(self, cursor):
        """기본 테이블 생성"""

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

    def _run_gpt_migrations(self, cursor):
        """GPT 관련 마이그레이션 실행"""

        # 마이그레이션 추적 테이블
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS migrations (
                migration_id TEXT PRIMARY KEY,
                executed_date TEXT
            )
        """
        )

        # Migration 1: user_gpt_settings 테이블 생성
        cursor.execute(
            "SELECT migration_id FROM migrations WHERE migration_id = ?",
            ("create_user_gpt_settings",),
        )
        if not cursor.fetchone():
            logger.info("Migration: user_gpt_settings 테이블 생성")

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_gpt_settings (
                    user_id TEXT PRIMARY KEY,
                    daily_token_limit INTEGER DEFAULT 1000,
                    monthly_cost_limit REAL DEFAULT 10.0,
                    current_daily_usage INTEGER DEFAULT 0,
                    current_monthly_cost REAL DEFAULT 0.0,
                    last_reset_date TEXT,
                    gpt_preferences TEXT,  -- JSON
                    notification_settings TEXT,  -- JSON
                    created_date TEXT,
                    updated_date TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """
            )

            cursor.execute(
                "INSERT OR REPLACE INTO migrations (migration_id, executed_date) VALUES (?, ?)",
                ("create_user_gpt_settings", datetime.now().isoformat()),
            )

        # Migration 2: gpt_usage_log 테이블 생성
        cursor.execute(
            "SELECT migration_id FROM migrations WHERE migration_id = ?",
            ("create_gpt_usage_log",),
        )
        if not cursor.fetchone():
            logger.info("Migration: gpt_usage_log 테이블 생성")

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS gpt_usage_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    purpose TEXT NOT NULL,  -- prompt_engineering, curator_message
                    model TEXT NOT NULL,    -- gpt-4o-mini, gpt-4o, etc.
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER, 
                    total_tokens INTEGER,
                    processing_time REAL,
                    cost_estimate REAL,
                    success BOOLEAN,
                    error_message TEXT,
                    safety_level TEXT,      -- safe, warning, critical
                    created_date TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """
            )

            cursor.execute(
                "INSERT OR REPLACE INTO migrations (migration_id, executed_date) VALUES (?, ?)",
                ("create_gpt_usage_log", datetime.now().isoformat()),
            )

        # Migration 3: gpt_quality_log 테이블 생성
        cursor.execute(
            "SELECT migration_id FROM migrations WHERE migration_id = ?",
            ("create_gpt_quality_log",),
        )
        if not cursor.fetchone():
            logger.info("Migration: gpt_quality_log 테이블 생성")

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS gpt_quality_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    gallery_item_id INTEGER,
                    response_type TEXT,  -- prompt, curator_message
                    gpt_model TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    generation_time REAL,
                    safety_level TEXT,   -- safe, warning, critical
                    therapeutic_quality TEXT,  -- high, medium, low
                    personalization_score REAL,  -- 0.0-1.0
                    user_reaction TEXT,  -- like, save, share, dismiss
                    reaction_timestamp TEXT,
                    feedback_data TEXT,  -- JSON
                    created_date TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """
            )

            cursor.execute(
                "INSERT OR REPLACE INTO migrations (migration_id, executed_date) VALUES (?, ?)",
                ("create_gpt_quality_log", datetime.now().isoformat()),
            )

        # Migration 4: gpt_performance_stats 테이블 생성
        cursor.execute(
            "SELECT migration_id FROM migrations WHERE migration_id = ?",
            ("create_gpt_performance_stats",),
        )
        if not cursor.fetchone():
            logger.info("Migration: gpt_performance_stats 테이블 생성")

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS gpt_performance_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,  -- YYYY-MM-DD 형식
                    model TEXT NOT NULL,
                    purpose TEXT NOT NULL,  -- prompt_engineering, curator_message
                    total_requests INTEGER DEFAULT 0,
                    successful_requests INTEGER DEFAULT 0,
                    avg_response_time REAL DEFAULT 0.0,
                    avg_tokens_per_request REAL DEFAULT 0.0,
                    total_cost REAL DEFAULT 0.0,
                    safety_violations INTEGER DEFAULT 0,
                    positive_user_reactions INTEGER DEFAULT 0,
                    total_user_reactions INTEGER DEFAULT 0,
                    created_date TEXT,
                    UNIQUE(date, model, purpose)
                )
            """
            )

            cursor.execute(
                "INSERT OR REPLACE INTO migrations (migration_id, executed_date) VALUES (?, ?)",
                ("create_gpt_performance_stats", datetime.now().isoformat()),
            )

        # Migration 5: 기존 사용자들을 위한 기본 GPT 설정 생성
        cursor.execute(
            "SELECT migration_id FROM migrations WHERE migration_id = ?",
            ("init_existing_user_gpt_settings",),
        )
        if not cursor.fetchone():
            logger.info("Migration: 기존 사용자 GPT 설정 초기화")

            cursor.execute(
                """
                INSERT OR IGNORE INTO user_gpt_settings 
                (user_id, daily_token_limit, monthly_cost_limit, current_daily_usage, 
                 current_monthly_cost, last_reset_date, gpt_preferences, 
                 notification_settings, created_date, updated_date)
                SELECT user_id, 1000, 10.0, 0, 0.0, ?, '{}', '{}', ?, ?
                FROM users
            """,
                (
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                ),
            )

            cursor.execute(
                "INSERT OR REPLACE INTO migrations (migration_id, executed_date) VALUES (?, ?)",
                ("init_existing_user_gpt_settings", datetime.now().isoformat()),
            )

    def create_user_gpt_settings(self, user_id: str) -> None:
        """사용자 GPT 설정 초기화"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR IGNORE INTO user_gpt_settings
            (user_id, daily_token_limit, monthly_cost_limit, current_daily_usage,
             current_monthly_cost, last_reset_date, gpt_preferences, 
             notification_settings, created_date, updated_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                user_id,
                1000,  # 기본 일일 토큰 한도
                10.0,  # 기본 월간 비용 한도
                0,  # 현재 일일 사용량
                0.0,  # 현재 월간 비용
                datetime.now().isoformat(),
                json.dumps({}),  # 기본 GPT 선호도
                json.dumps({}),  # 기본 알림 설정
                datetime.now().isoformat(),
                datetime.now().isoformat(),
            ),
        )

        conn.commit()
        conn.close()
        logger.info(f"사용자 {user_id}의 GPT 설정이 초기화되었습니다.")

    def get_user_gpt_usage(self, user_id: str) -> Dict[str, Any]:
        """사용자 GPT 사용량 조회"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # GPT 설정 조회
        cursor.execute("SELECT * FROM user_gpt_settings WHERE user_id = ?", (user_id,))
        settings_row = cursor.fetchone()

        if not settings_row:
            # 설정이 없으면 생성
            self.create_user_gpt_settings(user_id)
            cursor.execute(
                "SELECT * FROM user_gpt_settings WHERE user_id = ?", (user_id,)
            )
            settings_row = cursor.fetchone()

        # 오늘의 사용량 조회
        today = datetime.now().strftime("%Y-%m-%d")
        cursor.execute(
            """
            SELECT COUNT(*), SUM(total_tokens), SUM(cost_estimate)
            FROM gpt_usage_log 
            WHERE user_id = ? AND DATE(created_date) = ?
        """,
            (user_id, today),
        )
        today_usage = cursor.fetchone()

        # 이번 달 사용량 조회
        this_month = datetime.now().strftime("%Y-%m")
        cursor.execute(
            """
            SELECT COUNT(*), SUM(total_tokens), SUM(cost_estimate)
            FROM gpt_usage_log 
            WHERE user_id = ? AND strftime('%Y-%m', created_date) = ?
        """,
            (user_id, this_month),
        )
        monthly_usage = cursor.fetchone()

        conn.close()

        usage_data = {
            "user_id": user_id,
            "daily_limit": settings_row[1],
            "monthly_cost_limit": settings_row[2],
            "current_daily_usage": today_usage[1] if today_usage[1] else 0,
            "current_monthly_cost": monthly_usage[2] if monthly_usage[2] else 0.0,
            "today_requests": today_usage[0] if today_usage[0] else 0,
            "monthly_requests": monthly_usage[0] if monthly_usage[0] else 0,
            "last_reset_date": settings_row[5],
            "usage_percentage": {
                "daily": (
                    (today_usage[1] / settings_row[1]) * 100
                    if today_usage[1] and settings_row[1] > 0
                    else 0
                ),
                "monthly": (
                    (monthly_usage[2] / settings_row[2]) * 100
                    if monthly_usage[2] and settings_row[2] > 0
                    else 0
                ),
            },
        }

        return usage_data

    def update_daily_usage(self, user_id: str, tokens_used: int, cost: float) -> None:
        """일일 사용량 업데이트"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 현재 설정 조회
        cursor.execute(
            "SELECT current_daily_usage, current_monthly_cost FROM user_gpt_settings WHERE user_id = ?",
            (user_id,),
        )
        current = cursor.fetchone()

        if current:
            new_daily_usage = current[0] + tokens_used
            new_monthly_cost = current[1] + cost

            cursor.execute(
                """
                UPDATE user_gpt_settings
                SET current_daily_usage = ?, current_monthly_cost = ?, updated_date = ?
                WHERE user_id = ?
            """,
                (
                    new_daily_usage,
                    new_monthly_cost,
                    datetime.now().isoformat(),
                    user_id,
                ),
            )

        conn.commit()
        conn.close()

    def check_usage_limits(self, user_id: str) -> Dict[str, bool]:
        """사용량 한도 확인"""

        usage_data = self.get_user_gpt_usage(user_id)

        return {
            "daily_limit_exceeded": usage_data["current_daily_usage"]
            >= usage_data["daily_limit"],
            "monthly_limit_exceeded": usage_data["current_monthly_cost"]
            >= usage_data["monthly_cost_limit"],
            "daily_warning": usage_data["usage_percentage"]["daily"] >= 80,
            "monthly_warning": usage_data["usage_percentage"]["monthly"] >= 80,
            "can_use_gpt": (
                usage_data["current_daily_usage"] < usage_data["daily_limit"]
                and usage_data["current_monthly_cost"]
                < usage_data["monthly_cost_limit"]
            ),
        }

    def reset_daily_usage(self, user_id: str) -> None:
        """일일 사용량 리셋"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE user_gpt_settings
            SET current_daily_usage = 0, last_reset_date = ?, updated_date = ?
            WHERE user_id = ?
        """,
            (datetime.now().isoformat(), datetime.now().isoformat(), user_id),
        )

        conn.commit()
        conn.close()
        logger.info(f"사용자 {user_id}의 일일 사용량이 리셋되었습니다.")

    def get_gpt_preferences(self, user_id: str) -> Dict[str, Any]:
        """GPT 개인화 선호도 조회"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT gpt_preferences FROM user_gpt_settings WHERE user_id = ?",
            (user_id,),
        )
        result = cursor.fetchone()
        conn.close()

        if result and result[0]:
            try:
                return json.loads(result[0])
            except json.JSONDecodeError:
                logger.warning(f"사용자 {user_id}의 GPT 선호도 파싱 실패")

        return {}

    def update_gpt_preferences(self, user_id: str, preferences: Dict[str, Any]) -> None:
        """GPT 개인화 선호도 업데이트"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE user_gpt_settings
            SET gpt_preferences = ?, updated_date = ?
            WHERE user_id = ?
        """,
            (json.dumps(preferences), datetime.now().isoformat(), user_id),
        )

        conn.commit()
        conn.close()
        logger.info(f"사용자 {user_id}의 GPT 선호도가 업데이트되었습니다.")

    def log_gpt_usage(
        self,
        user_id: str,
        purpose: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        processing_time: float,
        cost_estimate: float,
        success: bool,
        error_message: str = None,
        safety_level: str = "safe",
    ) -> None:
        """GPT 사용량 로그 기록"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        total_tokens = prompt_tokens + completion_tokens

        cursor.execute(
            """
            INSERT INTO gpt_usage_log
            (user_id, purpose, model, prompt_tokens, completion_tokens, total_tokens,
             processing_time, cost_estimate, success, error_message, safety_level, created_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                user_id,
                purpose,
                model,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                processing_time,
                cost_estimate,
                success,
                error_message,
                safety_level,
                datetime.now().isoformat(),
            ),
        )

        conn.commit()
        conn.close()

        # 일일 사용량도 업데이트
        if success:
            self.update_daily_usage(user_id, total_tokens, cost_estimate)

    def log_gpt_quality(
        self,
        user_id: str,
        gallery_item_id: int,
        response_type: str,
        gpt_model: str,
        input_tokens: int,
        output_tokens: int,
        generation_time: float,
        safety_level: str,
        therapeutic_quality: str,
        personalization_score: float,
        user_reaction: str = None,
        feedback_data: Dict[str, Any] = None,
    ) -> None:
        """GPT 응답 품질 로그 기록"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO gpt_quality_log
            (user_id, gallery_item_id, response_type, gpt_model, input_tokens, output_tokens,
             generation_time, safety_level, therapeutic_quality, personalization_score,
             user_reaction, reaction_timestamp, feedback_data, created_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                user_id,
                gallery_item_id,
                response_type,
                gpt_model,
                input_tokens,
                output_tokens,
                generation_time,
                safety_level,
                therapeutic_quality,
                personalization_score,
                user_reaction,
                datetime.now().isoformat() if user_reaction else None,
                json.dumps(feedback_data) if feedback_data else None,
                datetime.now().isoformat(),
            ),
        )

        conn.commit()
        conn.close()

    def get_gpt_usage_analytics(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """GPT 사용량 분석 데이터 조회"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 지정된 기간의 사용량 조회
        start_date = (datetime.now() - timedelta(days=days)).isoformat()

        # 목적별 사용량
        cursor.execute(
            """
            SELECT purpose, COUNT(*), SUM(total_tokens), SUM(cost_estimate)
            FROM gpt_usage_log
            WHERE user_id = ? AND created_date >= ?
            GROUP BY purpose
        """,
            (user_id, start_date),
        )
        purpose_stats = [
            {
                "purpose": row[0],
                "requests": row[1],
                "tokens": row[2],
                "cost": row[3],
            }
            for row in cursor.fetchall()
        ]

        # 일별 사용량
        cursor.execute(
            """
            SELECT DATE(created_date) as date, COUNT(*), SUM(total_tokens)
            FROM gpt_usage_log
            WHERE user_id = ? AND created_date >= ?
            GROUP BY DATE(created_date)
            ORDER BY date DESC
            LIMIT 30
        """,
            (user_id, start_date),
        )
        daily_usage = [
            {"date": row[0], "requests": row[1], "tokens": row[2]}
            for row in cursor.fetchall()
        ]

        # 성공률
        cursor.execute(
            """
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful
            FROM gpt_usage_log
            WHERE user_id = ? AND created_date >= ?
        """,
            (user_id, start_date),
        )
        success_stats = cursor.fetchone()
        success_rate = (
            success_stats[1] / success_stats[0] if success_stats[0] > 0 else 0
        )

        conn.close()

        return {
            "user_id": user_id,
            "period_days": days,
            "purpose_breakdown": purpose_stats,
            "daily_usage": daily_usage,
            "success_rate": success_rate,
            "total_requests": sum(stat["requests"] for stat in purpose_stats),
            "total_tokens": sum(stat["tokens"] for stat in purpose_stats),
            "total_cost": sum(stat["cost"] for stat in purpose_stats),
        }

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

        # GPT 설정 초기화
        self.create_user_gpt_settings(user_id)

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

        # GPT 사용량 정보 추가
        gpt_usage = self.get_user_gpt_usage(user_id)
        stats.update(
            {
                "gpt_daily_usage": gpt_usage["current_daily_usage"],
                "gpt_monthly_cost": gpt_usage["current_monthly_cost"],
                "gpt_usage_percentage": gpt_usage["usage_percentage"],
            }
        )

        return stats
