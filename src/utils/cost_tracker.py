# src/utils/cost_tracker.py

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class CostTracker:
    """GPT API 비용 추적 및 관리"""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # 모델별 토큰 비용 (USD per 1K tokens)
        self.model_costs = {
            "gpt-4o": {"input": 0.0025, "output": 0.01},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        }

        # 기본 한도 설정
        self.default_limits = {
            "daily_token_limit": 10000,
            "monthly_cost_limit": 50.0,
            "burst_limit": 5000,  # 시간당 버스트 한도
        }

        self._init_database()

    def _init_database(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # API 호출 기록 테이블
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS api_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                purpose TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                input_cost REAL NOT NULL,
                output_cost REAL NOT NULL,
                total_cost REAL NOT NULL,
                processing_time REAL NOT NULL,
                success BOOLEAN NOT NULL,
                error_message TEXT,
                timestamp TEXT NOT NULL,
                date TEXT NOT NULL
            )
        """
        )

        # 사용자별 한도 설정 테이블
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_limits (
                user_id TEXT PRIMARY KEY,
                daily_token_limit INTEGER DEFAULT 10000,
                monthly_cost_limit REAL DEFAULT 50.0,
                burst_limit INTEGER DEFAULT 5000,
                created_date TEXT NOT NULL,
                updated_date TEXT NOT NULL
            )
        """
        )

        # 일일 사용량 캐시 테이블
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_usage (
                user_id TEXT NOT NULL,
                date TEXT NOT NULL,
                total_tokens INTEGER DEFAULT 0,
                total_cost REAL DEFAULT 0.0,
                api_calls INTEGER DEFAULT 0,
                last_updated TEXT NOT NULL,
                PRIMARY KEY (user_id, date)
            )
        """
        )

        # 인덱스 생성
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_api_calls_user_date ON api_calls(user_id, date)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_api_calls_timestamp ON api_calls(timestamp)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_daily_usage_date ON daily_usage(date)"
        )

        conn.commit()
        conn.close()
        logger.info("비용 추적 데이터베이스가 초기화되었습니다.")

    def record_api_call(
        self,
        user_id: str,
        purpose: str,
        model: str,
        token_usage: Dict[str, int],
        processing_time: float,
        success: bool = True,
        error_message: str = None,
    ) -> Dict[str, Any]:
        """GPT API 호출 기록"""

        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens

        # 비용 계산
        costs = self._calculate_cost(model, prompt_tokens, completion_tokens)

        timestamp = datetime.now().isoformat()
        date = datetime.now().strftime("%Y-%m-%d")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # API 호출 기록 저장
            cursor.execute(
                """
                INSERT INTO api_calls 
                (user_id, purpose, model, prompt_tokens, completion_tokens, total_tokens,
                 input_cost, output_cost, total_cost, processing_time, success, error_message, timestamp, date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    user_id,
                    purpose,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    costs["input_cost"],
                    costs["output_cost"],
                    costs["total_cost"],
                    processing_time,
                    success,
                    error_message,
                    timestamp,
                    date,
                ),
            )

            # 일일 사용량 업데이트
            cursor.execute(
                """
                INSERT OR REPLACE INTO daily_usage 
                (user_id, date, total_tokens, total_cost, api_calls, last_updated)
                VALUES (?, ?, 
                    COALESCE((SELECT total_tokens FROM daily_usage WHERE user_id = ? AND date = ?), 0) + ?,
                    COALESCE((SELECT total_cost FROM daily_usage WHERE user_id = ? AND date = ?), 0) + ?,
                    COALESCE((SELECT api_calls FROM daily_usage WHERE user_id = ? AND date = ?), 0) + 1,
                    ?)
            """,
                (
                    user_id,
                    date,
                    user_id,
                    date,
                    total_tokens,
                    user_id,
                    date,
                    costs["total_cost"],
                    user_id,
                    date,
                    timestamp,
                ),
            )

            conn.commit()

            record_result = {
                "success": True,
                "call_id": cursor.lastrowid,
                "user_id": user_id,
                "purpose": purpose,
                "model": model,
                "tokens": {
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                    "total": total_tokens,
                },
                "costs": costs,
                "processing_time": processing_time,
                "timestamp": timestamp,
            }

            logger.info(
                f"API 호출 기록됨: 사용자={user_id}, 목적={purpose}, "
                f"토큰={total_tokens}, 비용=${costs['total_cost']:.4f}"
            )

            return record_result

        except Exception as e:
            conn.rollback()
            logger.error(f"API 호출 기록 실패: {e}")
            return {"success": False, "error": str(e)}
        finally:
            conn.close()

    def _calculate_cost(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> Dict[str, float]:
        """토큰 사용량 기반 비용 계산"""

        if model not in self.model_costs:
            logger.warning(f"알 수 없는 모델: {model}. 기본 요금 적용")
            # 기본적으로 gpt-4o-mini 요금 사용
            model = "gpt-4o-mini"

        cost_info = self.model_costs[model]

        input_cost = (prompt_tokens / 1000) * cost_info["input"]
        output_cost = (completion_tokens / 1000) * cost_info["output"]
        total_cost = input_cost + output_cost

        return {
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6),
        }

    def check_daily_limit(self, user_id: str) -> Dict[str, Any]:
        """일일 한도 체크"""

        user_limits = self._get_user_limits(user_id)
        current_usage = self._get_daily_usage(user_id)

        token_limit = user_limits["daily_token_limit"]
        current_tokens = current_usage["total_tokens"]

        check_result = {
            "user_id": user_id,
            "within_limit": current_tokens < token_limit,
            "current_tokens": current_tokens,
            "token_limit": token_limit,
            "remaining_tokens": max(0, token_limit - current_tokens),
            "usage_percentage": (
                (current_tokens / token_limit * 100) if token_limit > 0 else 0
            ),
            "current_cost": current_usage["total_cost"],
            "api_calls": current_usage["api_calls"],
        }

        # 경고 레벨 결정
        usage_pct = check_result["usage_percentage"]
        if usage_pct >= 100:
            check_result["warning_level"] = "exceeded"
            check_result["warning_message"] = "일일 토큰 한도를 초과했습니다."
        elif usage_pct >= 90:
            check_result["warning_level"] = "critical"
            check_result["warning_message"] = "일일 토큰 한도의 90%에 도달했습니다."
        elif usage_pct >= 75:
            check_result["warning_level"] = "warning"
            check_result["warning_message"] = "일일 토큰 한도의 75%에 도달했습니다."
        else:
            check_result["warning_level"] = "normal"
            check_result["warning_message"] = ""

        return check_result

    def check_monthly_limit(self, user_id: str) -> Dict[str, Any]:
        """월간 비용 한도 체크"""

        user_limits = self._get_user_limits(user_id)
        monthly_usage = self._get_monthly_usage(user_id)

        cost_limit = user_limits["monthly_cost_limit"]
        current_cost = monthly_usage["total_cost"]

        return {
            "user_id": user_id,
            "within_limit": current_cost < cost_limit,
            "current_cost": current_cost,
            "cost_limit": cost_limit,
            "remaining_budget": max(0, cost_limit - current_cost),
            "usage_percentage": (
                (current_cost / cost_limit * 100) if cost_limit > 0 else 0
            ),
            "days_in_month": monthly_usage["days_in_month"],
            "avg_daily_cost": monthly_usage["avg_daily_cost"],
        }

    def _get_user_limits(self, user_id: str) -> Dict[str, Any]:
        """사용자 한도 설정 조회"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM user_limits WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()

        if row:
            limits = {
                "daily_token_limit": row[1],
                "monthly_cost_limit": row[2],
                "burst_limit": row[3],
            }
        else:
            # 기본 한도 설정 및 저장
            limits = self.default_limits.copy()
            timestamp = datetime.now().isoformat()

            cursor.execute(
                """
                INSERT INTO user_limits 
                (user_id, daily_token_limit, monthly_cost_limit, burst_limit, created_date, updated_date)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    user_id,
                    limits["daily_token_limit"],
                    limits["monthly_cost_limit"],
                    limits["burst_limit"],
                    timestamp,
                    timestamp,
                ),
            )
            conn.commit()

        conn.close()
        return limits

    def _get_daily_usage(self, user_id: str, date: str = None) -> Dict[str, Any]:
        """일일 사용량 조회"""

        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT total_tokens, total_cost, api_calls FROM daily_usage WHERE user_id = ? AND date = ?",
            (user_id, date),
        )
        row = cursor.fetchone()

        if row:
            usage = {
                "total_tokens": row[0],
                "total_cost": row[1],
                "api_calls": row[2],
            }
        else:
            usage = {
                "total_tokens": 0,
                "total_cost": 0.0,
                "api_calls": 0,
            }

        conn.close()
        return usage

    def _get_monthly_usage(self, user_id: str) -> Dict[str, Any]:
        """월간 사용량 조회"""

        # 현재 월의 시작일과 종료일 계산
        now = datetime.now()
        start_date = now.replace(day=1).strftime("%Y-%m-%d")

        # 다음 달 첫날에서 하루 빼기
        if now.month == 12:
            next_month = now.replace(year=now.year + 1, month=1, day=1)
        else:
            next_month = now.replace(month=now.month + 1, day=1)
        end_date = (next_month - timedelta(days=1)).strftime("%Y-%m-%d")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT SUM(total_cost), COUNT(DISTINCT date), AVG(total_cost)
            FROM daily_usage 
            WHERE user_id = ? AND date >= ? AND date <= ?
        """,
            (user_id, start_date, end_date),
        )
        row = cursor.fetchone()

        total_cost = row[0] if row[0] else 0.0
        days_with_usage = row[1] if row[1] else 0
        avg_daily_cost = row[2] if row[2] else 0.0

        conn.close()

        return {
            "total_cost": total_cost,
            "days_in_month": now.day,
            "days_with_usage": days_with_usage,
            "avg_daily_cost": avg_daily_cost,
            "start_date": start_date,
            "end_date": end_date,
        }

    def get_daily_summary(self, date: str = None) -> Dict[str, Any]:
        """일일 사용량 요약"""

        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 전체 일일 통계
        cursor.execute(
            """
            SELECT COUNT(*), SUM(total_tokens), SUM(total_cost), AVG(processing_time)
            FROM api_calls 
            WHERE date = ? AND success = 1
        """,
            (date,),
        )
        overall_stats = cursor.fetchone()

        # 목적별 통계
        cursor.execute(
            """
            SELECT purpose, COUNT(*), SUM(total_tokens), SUM(total_cost)
            FROM api_calls 
            WHERE date = ? AND success = 1
            GROUP BY purpose
        """,
            (date,),
        )
        purpose_stats = cursor.fetchall()

        # 모델별 통계
        cursor.execute(
            """
            SELECT model, COUNT(*), SUM(total_tokens), SUM(total_cost)
            FROM api_calls 
            WHERE date = ? AND success = 1
            GROUP BY model
        """,
            (date,),
        )
        model_stats = cursor.fetchall()

        # 시간대별 사용량
        cursor.execute(
            """
            SELECT strftime('%H', timestamp) as hour, COUNT(*), SUM(total_tokens)
            FROM api_calls 
            WHERE date = ? AND success = 1
            GROUP BY hour
            ORDER BY hour
        """,
            (date,),
        )
        hourly_stats = cursor.fetchall()

        conn.close()

        summary = {
            "date": date,
            "overall": {
                "total_calls": overall_stats[0] or 0,
                "total_tokens": overall_stats[1] or 0,
                "total_cost": overall_stats[2] or 0.0,
                "avg_processing_time": overall_stats[3] or 0.0,
            },
            "by_purpose": [
                {
                    "purpose": row[0],
                    "calls": row[1],
                    "tokens": row[2],
                    "cost": row[3],
                }
                for row in purpose_stats
            ],
            "by_model": [
                {
                    "model": row[0],
                    "calls": row[1],
                    "tokens": row[2],
                    "cost": row[3],
                }
                for row in model_stats
            ],
            "by_hour": [
                {
                    "hour": int(row[0]),
                    "calls": row[1],
                    "tokens": row[2],
                }
                for row in hourly_stats
            ],
        }

        return summary

    def get_cost_optimization_recommendations(
        self, user_id: str
    ) -> List[Dict[str, Any]]:
        """비용 최적화 권장사항"""

        recommendations = []

        # 일일 및 월간 사용량 분석
        daily_check = self.check_daily_limit(user_id)
        monthly_check = self.check_monthly_limit(user_id)

        # 높은 사용량 경고
        if daily_check["usage_percentage"] > 80:
            recommendations.append(
                {
                    "type": "usage_warning",
                    "priority": "high",
                    "title": "높은 일일 사용량",
                    "description": f"오늘 토큰 사용량이 한도의 {daily_check['usage_percentage']:.1f}%에 도달했습니다.",
                    "action": "사용량을 모니터링하고 필요시 한도를 조정하세요.",
                }
            )

        if monthly_check["usage_percentage"] > 70:
            recommendations.append(
                {
                    "type": "budget_warning",
                    "priority": "medium",
                    "title": "월간 예산 주의",
                    "description": f"이번 달 비용이 예산의 {monthly_check['usage_percentage']:.1f}%에 도달했습니다.",
                    "action": "남은 기간 동안 사용량을 조절하거나 예산을 늘리는 것을 고려하세요.",
                }
            )

        # 모델 사용 최적화
        recent_usage = self._analyze_recent_model_usage(user_id)
        expensive_model_usage = sum(
            stats["cost"]
            for model, stats in recent_usage.items()
            if model in ["gpt-4", "gpt-4-turbo"]
        )
        total_cost = sum(stats["cost"] for stats in recent_usage.values())

        if expensive_model_usage > total_cost * 0.5 and total_cost > 1.0:
            recommendations.append(
                {
                    "type": "model_optimization",
                    "priority": "medium",
                    "title": "모델 사용 최적화",
                    "description": "비싼 모델(GPT-4)의 사용 비중이 높습니다.",
                    "action": "간단한 작업에는 gpt-4o-mini를 사용하는 것을 고려하세요.",
                }
            )

        # 사용 패턴 분석
        usage_pattern = self._analyze_usage_pattern(user_id)
        if usage_pattern["peak_hour_concentration"] > 0.6:
            recommendations.append(
                {
                    "type": "usage_pattern",
                    "priority": "low",
                    "title": "사용 시간 분산",
                    "description": "특정 시간대에 사용량이 집중되어 있습니다.",
                    "action": "가능하다면 사용 시간을 분산하여 시스템 부하를 줄이세요.",
                }
            )

        return recommendations

    def _analyze_recent_model_usage(
        self, user_id: str, days: int = 7
    ) -> Dict[str, Dict[str, Any]]:
        """최근 모델별 사용량 분석"""

        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT model, COUNT(*), SUM(total_tokens), SUM(total_cost)
            FROM api_calls 
            WHERE user_id = ? AND date >= ? AND success = 1
            GROUP BY model
        """,
            (user_id, start_date),
        )

        model_usage = {}
        for row in cursor.fetchall():
            model_usage[row[0]] = {
                "calls": row[1],
                "tokens": row[2],
                "cost": row[3],
            }

        conn.close()
        return model_usage

    def _analyze_usage_pattern(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """사용 패턴 분석"""

        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 시간대별 사용량
        cursor.execute(
            """
            SELECT strftime('%H', timestamp) as hour, COUNT(*)
            FROM api_calls 
            WHERE user_id = ? AND date >= ? AND success = 1
            GROUP BY hour
        """,
            (user_id, start_date),
        )

        hourly_usage = {int(row[0]): row[1] for row in cursor.fetchall()}
        total_calls = sum(hourly_usage.values())

        # 피크 시간 집중도 계산
        max_hourly_calls = max(hourly_usage.values()) if hourly_usage else 0
        peak_concentration = (max_hourly_calls / total_calls) if total_calls > 0 else 0

        conn.close()

        return {
            "peak_hour_concentration": peak_concentration,
            "total_calls": total_calls,
            "active_hours": len(hourly_usage),
            "hourly_distribution": hourly_usage,
        }

    def set_user_limits(
        self,
        user_id: str,
        daily_token_limit: int = None,
        monthly_cost_limit: float = None,
        burst_limit: int = None,
    ) -> bool:
        """사용자 한도 설정 업데이트"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # 현재 설정 조회
            current_limits = self._get_user_limits(user_id)

            # 새로운 값으로 업데이트 (None이 아닌 경우만)
            new_limits = {
                "daily_token_limit": daily_token_limit
                or current_limits["daily_token_limit"],
                "monthly_cost_limit": monthly_cost_limit
                or current_limits["monthly_cost_limit"],
                "burst_limit": burst_limit or current_limits["burst_limit"],
            }

            cursor.execute(
                """
                INSERT OR REPLACE INTO user_limits 
                (user_id, daily_token_limit, monthly_cost_limit, burst_limit, created_date, updated_date)
                VALUES (?, ?, ?, ?, 
                    COALESCE((SELECT created_date FROM user_limits WHERE user_id = ?), ?), ?)
            """,
                (
                    user_id,
                    new_limits["daily_token_limit"],
                    new_limits["monthly_cost_limit"],
                    new_limits["burst_limit"],
                    user_id,
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                ),
            )

            conn.commit()
            logger.info(f"사용자 {user_id}의 한도가 업데이트되었습니다: {new_limits}")
            return True

        except Exception as e:
            conn.rollback()
            logger.error(f"사용자 한도 업데이트 실패: {e}")
            return False
        finally:
            conn.close()

    def get_usage_statistics(
        self, user_id: str = None, days: int = 30
    ) -> Dict[str, Any]:
        """사용량 통계"""

        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        where_clause = "WHERE date >= ?"
        params = [start_date]

        if user_id:
            where_clause += " AND user_id = ?"
            params.append(user_id)

        # 전체 통계
        cursor.execute(
            f"""
            SELECT COUNT(*), SUM(total_tokens), SUM(total_cost), 
                   AVG(processing_time), COUNT(DISTINCT user_id)
            FROM api_calls 
            {where_clause} AND success = 1
        """,
            params,
        )
        overall_stats = cursor.fetchone()

        # 일별 통계
        cursor.execute(
            f"""
            SELECT date, COUNT(*), SUM(total_tokens), SUM(total_cost)
            FROM api_calls 
            {where_clause} AND success = 1
            GROUP BY date
            ORDER BY date
        """,
            params,
        )
        daily_stats = cursor.fetchall()

        conn.close()

        return {
            "period": {"start_date": start_date, "days": days},
            "overall": {
                "total_calls": overall_stats[0] or 0,
                "total_tokens": overall_stats[1] or 0,
                "total_cost": overall_stats[2] or 0.0,
                "avg_processing_time": overall_stats[3] or 0.0,
                "unique_users": overall_stats[4] or 0,
            },
            "daily_breakdown": [
                {
                    "date": row[0],
                    "calls": row[1],
                    "tokens": row[2],
                    "cost": row[3],
                }
                for row in daily_stats
            ],
        }

    def cleanup_old_records(self, days_old: int = 90) -> int:
        """오래된 기록 정리"""

        cutoff_date = (datetime.now() - timedelta(days=days_old)).strftime("%Y-%m-%d")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # API 호출 기록 정리
            cursor.execute("DELETE FROM api_calls WHERE date < ?", (cutoff_date,))
            api_deleted = cursor.rowcount

            # 일일 사용량 정리
            cursor.execute("DELETE FROM daily_usage WHERE date < ?", (cutoff_date,))
            daily_deleted = cursor.rowcount

            conn.commit()

            logger.info(
                f"오래된 기록 정리 완료: API 호출 {api_deleted}개, 일일 사용량 {daily_deleted}개"
            )
            return api_deleted + daily_deleted

        except Exception as e:
            conn.rollback()
            logger.error(f"기록 정리 실패: {e}")
            return 0
        finally:
            conn.close()
