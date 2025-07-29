# src/managers/user_manager.py

# ==============================================================================
# 이 파일은 사용자 프로필 정보를 관리하는 핵심 모듈이다.
# MongoDB를 사용하여 사용자 기본 정보와 심리검사 결과를 저장하고,
# 시각적 선호도도 MongoDB에 통합 관리한다.
# `ACTTherapySystem`은 이 매니저를 통해 사용자를 생성, 조회하고 심리검사, 선호도 설정을 처리한다.
# ==============================================================================

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging
from pymongo.database import Database
from bson import ObjectId

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
    """사용자 프로필 관리자 - MongoDB 기반"""

    def __init__(self, mongodb_client):
        self.db: Database = mongodb_client.sync_db
        self.users_collection = self.db.users
        self._ensure_indexes()

    def _ensure_indexes(self):
        """MongoDB 인덱스 확인 및 생성"""
        try:
            # users 컬렉션 인덱스
            self.users_collection.create_index("user_id", unique=True)
            self.users_collection.create_index("created_date")
            
            logger.info("MongoDB 인덱스가 확인되었습니다.")
        except Exception as e:
            logger.warning(f"인덱스 생성 중 오류: {e}")

    def create_user(self, user_id: str) -> User:
        """새 사용자 생성"""
        now = datetime.now().isoformat()
        
        # 사용자 문서 생성
        user_doc = {
            "user_id": user_id,
            "created_date": now,
            "last_updated": now,
            "psychometric_results": [],
            "visual_preferences": asdict(VisualPreferences()),
            "gpt_settings": {
                "daily_token_limit": 1000,
                "monthly_cost_limit": 10.0,
                "current_daily_usage": 0,
                "current_monthly_cost": 0.0,
                "last_reset_date": now,
                "gpt_preferences": {},
                "notification_settings": {}
            }
        }
        
        try:
            result = self.users_collection.insert_one(user_doc)
            logger.info(f"새 사용자가 생성되었습니다: {user_id}")
            
            # User 객체 반환
            user = User(
                user_id=user_id,
                created_date=now,
                visual_preferences=VisualPreferences(),
                last_updated=now
            )
            return user
            
        except Exception as e:
            logger.error(f"사용자 생성 실패: {e}")
            raise

    def get_user(self, user_id: str) -> Optional[User]:
        """사용자 정보 조회"""
        try:
            user_doc = self.users_collection.find_one({"user_id": user_id})
            
            if not user_doc:
                return None
            
            # 심리검사 결과 변환
            psychometric_results = []
            for result_data in user_doc.get("psychometric_results", []):
                result = PsychometricResult(**result_data)
                psychometric_results.append(result)
            
            # 시각적 선호도 변환
            visual_prefs_data = user_doc.get("visual_preferences", {})
            visual_preferences = VisualPreferences(**visual_prefs_data)
            
            # User 객체 생성
            user = User(
                user_id=user_doc["user_id"],
                created_date=user_doc["created_date"],
                psychometric_results=psychometric_results,
                visual_preferences=visual_preferences,
                last_updated=user_doc.get("last_updated", user_doc["created_date"])
            )
            
            return user
            
        except Exception as e:
            logger.error(f"사용자 조회 실패: {e}")
            return None

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
        
        try:
            # MongoDB에 결과 추가
            self.users_collection.update_one(
                {"user_id": user_id},
                {
                    "$push": {"psychometric_results": asdict(result)},
                    "$set": {"last_updated": datetime.now().isoformat()}
                }
            )
            
            logger.info(f"사용자 {user_id}의 심리검사가 완료되었습니다: {coping_style}, {severity_level}")
            return result
            
        except Exception as e:
            logger.error(f"심리검사 결과 저장 실패: {e}")
            raise

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
        
        try:
            self.users_collection.update_one(
                {"user_id": user_id},
                {
                    "$set": {
                        "visual_preferences": asdict(visual_prefs),
                        "last_updated": datetime.now().isoformat()
                    }
                }
            )
            
            logger.info(f"사용자 {user_id}의 시각적 선호도가 설정되었습니다.")
            
        except Exception as e:
            logger.error(f"시각적 선호도 설정 실패: {e}")
            raise

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

        # MongoDB 업데이트
        try:
            self.users_collection.update_one(
                {"user_id": user_id},
                {
                    "$set": {
                        "visual_preferences": asdict(user.visual_preferences),
                        "last_updated": datetime.now().isoformat()
                    }
                }
            )
            
            logger.info(f"사용자 {user_id}의 선호도 가중치가 업데이트되었습니다.")
            
        except Exception as e:
            logger.error(f"선호도 가중치 업데이트 실패: {e}")

    def should_conduct_periodic_test(self, user_id: str) -> bool:
        """주기적 검사 필요 여부 확인"""
        user = self.get_user(user_id)
        if not user or not user.psychometric_results:
            return True

        last_test = user.psychometric_results[-1]  # 가장 최근 검사
        last_test_date = datetime.fromisoformat(last_test.test_date)

        # 2주 경과 확인
        return datetime.now() - last_test_date >= timedelta(weeks=2)

    def get_current_coping_style(self, user_id: str) -> str:
        """현재 대처 스타일 반환"""
        user = self.get_user(user_id)
        if user and user.psychometric_results:
            return user.psychometric_results[-1].coping_style
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
            latest = user.psychometric_results[-1]
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

    # =================================================================================
    # GPT 관련 기능들
    # =================================================================================

    def create_user_gpt_settings(self, user_id: str) -> None:
        """사용자 GPT 설정 초기화"""
        now = datetime.now().isoformat()
        
        gpt_settings = {
            "daily_token_limit": 1000,
            "monthly_cost_limit": 10.0,
            "current_daily_usage": 0,
            "current_monthly_cost": 0.0,
            "last_reset_date": now,
            "gpt_preferences": {},
            "notification_settings": {}
        }
        
        try:
            self.users_collection.update_one(
                {"user_id": user_id},
                {
                    "$set": {
                        "gpt_settings": gpt_settings,
                        "last_updated": now
                    }
                },
                upsert=True
            )
            
            logger.info(f"사용자 {user_id}의 GPT 설정이 초기화되었습니다.")
            
        except Exception as e:
            logger.error(f"GPT 설정 초기화 실패: {e}")

    def get_user_gpt_usage(self, user_id: str) -> Dict[str, Any]:
        """사용자 GPT 사용량 조회"""
        try:
            user_doc = self.users_collection.find_one({"user_id": user_id})
            
            if not user_doc or "gpt_settings" not in user_doc:
                # 설정이 없으면 생성
                self.create_user_gpt_settings(user_id)
                user_doc = self.users_collection.find_one({"user_id": user_id})
            
            gpt_settings = user_doc["gpt_settings"]
            
            # 오늘과 이번 달의 실제 사용량 조회 (gpt_usage_log에서)
            today = datetime.now().strftime("%Y-%m-%d")
            this_month = datetime.now().strftime("%Y-%m")
            
            # 일일 사용량 조회
            today_usage = list(self.db.gpt_usage_log.aggregate([
                {"$match": {
                    "user_id": user_id,
                    "created_date": {"$regex": f"^{today}"}
                }},
                {"$group": {
                    "_id": None,
                    "count": {"$sum": 1},
                    "total_tokens": {"$sum": "$total_tokens"},
                    "total_cost": {"$sum": "$cost_estimate"}
                }}
            ]))
            
            # 월간 사용량 조회
            monthly_usage = list(self.db.gpt_usage_log.aggregate([
                {"$match": {
                    "user_id": user_id,
                    "created_date": {"$regex": f"^{this_month}"}
                }},
                {"$group": {
                    "_id": None,
                    "count": {"$sum": 1},
                    "total_tokens": {"$sum": "$total_tokens"},
                    "total_cost": {"$sum": "$cost_estimate"}
                }}
            ]))
            
            today_stats = today_usage[0] if today_usage else {"count": 0, "total_tokens": 0, "total_cost": 0.0}
            monthly_stats = monthly_usage[0] if monthly_usage else {"count": 0, "total_tokens": 0, "total_cost": 0.0}
            
            usage_data = {
                "user_id": user_id,
                "daily_limit": gpt_settings["daily_token_limit"],
                "monthly_cost_limit": gpt_settings["monthly_cost_limit"],
                "current_daily_usage": today_stats["total_tokens"],
                "current_monthly_cost": monthly_stats["total_cost"],
                "today_requests": today_stats["count"],
                "monthly_requests": monthly_stats["count"],
                "last_reset_date": gpt_settings["last_reset_date"],
                "usage_percentage": {
                    "daily": (
                        (today_stats["total_tokens"] / gpt_settings["daily_token_limit"]) * 100
                        if gpt_settings["daily_token_limit"] > 0 else 0
                    ),
                    "monthly": (
                        (monthly_stats["total_cost"] / gpt_settings["monthly_cost_limit"]) * 100
                        if gpt_settings["monthly_cost_limit"] > 0 else 0
                    ),
                },
            }
            
            return usage_data
            
        except Exception as e:
            logger.error(f"GPT 사용량 조회 실패: {e}")
            return {
                "user_id": user_id,
                "daily_limit": 1000,
                "monthly_cost_limit": 10.0,
                "current_daily_usage": 0,
                "current_monthly_cost": 0.0,
                "today_requests": 0,
                "monthly_requests": 0,
                "usage_percentage": {"daily": 0, "monthly": 0},
            }

    def check_usage_limits(self, user_id: str) -> Dict[str, bool]:
        """사용량 한도 확인"""
        usage_data = self.get_user_gpt_usage(user_id)

        return {
            "daily_limit_exceeded": usage_data["current_daily_usage"] >= usage_data["daily_limit"],
            "monthly_limit_exceeded": usage_data["current_monthly_cost"] >= usage_data["monthly_cost_limit"],
            "daily_warning": usage_data["usage_percentage"]["daily"] >= 80,
            "monthly_warning": usage_data["usage_percentage"]["monthly"] >= 80,
            "can_use_gpt": (
                usage_data["current_daily_usage"] < usage_data["daily_limit"]
                and usage_data["current_monthly_cost"] < usage_data["monthly_cost_limit"]
            ),
        }

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
        try:
            usage_log = {
                "user_id": user_id,
                "purpose": purpose,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "processing_time": processing_time,
                "cost_estimate": cost_estimate,
                "success": success,
                "error_message": error_message,
                "safety_level": safety_level,
                "created_date": datetime.now().isoformat()
            }
            
            self.db.gpt_usage_log.insert_one(usage_log)
            
        except Exception as e:
            logger.error(f"GPT 사용량 로그 기록 실패: {e}")

    def get_gpt_usage_analytics(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """GPT 사용량 분석 데이터 조회"""
        try:
            start_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            # 목적별 사용량
            purpose_stats = list(self.db.gpt_usage_log.aggregate([
                {"$match": {
                    "user_id": user_id,
                    "created_date": {"$gte": start_date}
                }},
                {"$group": {
                    "_id": "$purpose",
                    "requests": {"$sum": 1},
                    "tokens": {"$sum": "$total_tokens"},
                    "cost": {"$sum": "$cost_estimate"}
                }}
            ]))
            
            # 일별 사용량
            daily_usage = list(self.db.gpt_usage_log.aggregate([
                {"$match": {
                    "user_id": user_id,
                    "created_date": {"$gte": start_date}
                }},
                {"$group": {
                    "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": {"$dateFromString": {"dateString": "$created_date"}}}},
                    "requests": {"$sum": 1},
                    "tokens": {"$sum": "$total_tokens"}
                }},
                {"$sort": {"_id": -1}},
                {"$limit": 30}
            ]))
            
            # 성공률
            success_stats = list(self.db.gpt_usage_log.aggregate([
                {"$match": {
                    "user_id": user_id,
                    "created_date": {"$gte": start_date}
                }},
                {"$group": {
                    "_id": None,
                    "total": {"$sum": 1},
                    "successful": {"$sum": {"$cond": ["$success", 1, 0]}}
                }}
            ]))
            
            success_rate = 0
            if success_stats:
                total = success_stats[0]["total"]
                successful = success_stats[0]["successful"]
                success_rate = successful / total if total > 0 else 0
            
            # 결과 포맷팅
            purpose_breakdown = [
                {
                    "purpose": stat["_id"],
                    "requests": stat["requests"],
                    "tokens": stat["tokens"],
                    "cost": stat["cost"]
                }
                for stat in purpose_stats
            ]
            
            daily_data = [
                {
                    "date": stat["_id"],
                    "requests": stat["requests"],
                    "tokens": stat["tokens"]
                }
                for stat in daily_usage
            ]
            
            return {
                "user_id": user_id,
                "period_days": days,
                "purpose_breakdown": purpose_breakdown,
                "daily_usage": daily_data,
                "success_rate": success_rate,
                "total_requests": sum(stat["requests"] for stat in purpose_breakdown),
                "total_tokens": sum(stat["tokens"] for stat in purpose_breakdown),
                "total_cost": sum(stat["cost"] for stat in purpose_breakdown),
            }
            
        except Exception as e:
            logger.error(f"GPT 사용량 분석 조회 실패: {e}")
            return {
                "user_id": user_id,
                "period_days": days,
                "purpose_breakdown": [],
                "daily_usage": [],
                "success_rate": 0,
                "total_requests": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
            }