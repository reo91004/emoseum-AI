# src/utils/cost_tracker.py

# ==============================================================================
# 이 파일은 GPT API 사용량과 그에 따른 비용을 추적하고 관리하는 유틸리티이다.
# MongoDB를 사용하여 사용자 ID, 사용 목적, 모델, 토큰 사용량, 비용 등을 기록한다.
# 이를 통해 사용자별/시스템 전체의 API 비용을 모니터링하고 관리할 수 있다.
# ==============================================================================

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from pymongo.database import Database

logger = logging.getLogger(__name__)


class CostTracker:
    """GPT API 비용 추적 및 관리 - MongoDB 기반"""

    def __init__(self, mongodb_client):
        self.db: Database = mongodb_client.sync_db
        self.api_calls = self.db.cost_tracking
        
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

        self._ensure_indexes()

    def _ensure_indexes(self):
        """MongoDB 인덱스 확인 및 생성"""
        try:
            # cost_tracking 컬렉션 인덱스
            self.api_calls.create_index("user_id")
            self.api_calls.create_index("timestamp")
            self.api_calls.create_index("date")
            self.api_calls.create_index([("user_id", 1), ("timestamp", -1)])
            self.api_calls.create_index([("user_id", 1), ("date", 1)])
            
            logger.info("비용 추적 MongoDB 인덱스가 확인되었습니다.")
        except Exception as e:
            logger.warning(f"비용 추적 인덱스 생성 중 오류: {e}")

    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> Dict[str, float]:
        """토큰 사용량을 기반으로 비용 계산"""
        if model not in self.model_costs:
            logger.warning(f"알 수 없는 모델: {model}, 기본 비용 적용")
            model = "gpt-4o-mini"  # 기본 모델로 폴백

        costs = self.model_costs[model]
        
        input_cost = (prompt_tokens / 1000) * costs["input"]
        output_cost = (completion_tokens / 1000) * costs["output"]
        total_cost = input_cost + output_cost

        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
        }

    def record_api_call(
        self,
        user_id: str,
        purpose: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        processing_time: float,
        success: bool,
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """API 호출 기록"""
        
        total_tokens = prompt_tokens + completion_tokens
        costs = self.calculate_cost(model, prompt_tokens, completion_tokens)
        
        now = datetime.now()
        timestamp = now.isoformat()
        date = now.strftime("%Y-%m-%d")
        
        # API 호출 기록 생성
        api_call_doc = {
            "user_id": user_id,
            "purpose": purpose,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "input_cost": costs["input_cost"],
            "output_cost": costs["output_cost"],
            "total_cost": costs["total_cost"],
            "processing_time": processing_time,
            "success": success,
            "error_message": error_message,
            "timestamp": timestamp,
            "date": date
        }
        
        try:
            result = self.api_calls.insert_one(api_call_doc)
            
            # 반환 데이터
            record = {
                "id": str(result.inserted_id),
                "user_id": user_id,
                "purpose": purpose,
                "model": model,
                "total_tokens": total_tokens,
                "total_cost": costs["total_cost"],
                "success": success,
                "timestamp": timestamp,
            }
            
            logger.info(f"API 호출 기록됨: {user_id} - {purpose} - {model} - ${costs['total_cost']:.4f}")
            return record
            
        except Exception as e:
            logger.error(f"API 호출 기록 실패: {e}")
            raise

    def get_user_usage_summary(
        self, user_id: str, days: int = 30
    ) -> Dict[str, Any]:
        """사용자별 사용량 요약"""
        
        try:
            start_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            # 기본 통계 조회
            pipeline = [
                {
                    "$match": {
                        "user_id": user_id,
                        "timestamp": {"$gte": start_date}
                    }
                },
                {
                    "$group": {
                        "_id": None,
                        "total_calls": {"$sum": 1},
                        "successful_calls": {"$sum": {"$cond": ["$success", 1, 0]}},
                        "total_tokens": {"$sum": "$total_tokens"},
                        "total_cost": {"$sum": "$total_cost"},
                        "avg_processing_time": {"$avg": "$processing_time"},
                        "first_call": {"$min": "$timestamp"},
                        "last_call": {"$max": "$timestamp"}
                    }
                }
            ]
            
            basic_stats = list(self.api_calls.aggregate(pipeline))
            
            if not basic_stats:
                return {
                    "user_id": user_id,
                    "period_days": days,
                    "total_calls": 0,
                    "total_cost": 0.0,
                    "error": "데이터 없음"
                }
            
            stats = basic_stats[0]
            
            # 목적별 사용량
            purpose_pipeline = [
                {
                    "$match": {
                        "user_id": user_id,
                        "timestamp": {"$gte": start_date}
                    }
                },
                {
                    "$group": {
                        "_id": "$purpose",
                        "calls": {"$sum": 1},
                        "tokens": {"$sum": "$total_tokens"},
                        "cost": {"$sum": "$total_cost"}
                    }
                }
            ]
            
            purpose_breakdown = {}
            for doc in self.api_calls.aggregate(purpose_pipeline):
                purpose_breakdown[doc["_id"]] = {
                    "calls": doc["calls"],
                    "tokens": doc["tokens"],
                    "cost": doc["cost"]
                }
            
            # 모델별 사용량
            model_pipeline = [
                {
                    "$match": {
                        "user_id": user_id,
                        "timestamp": {"$gte": start_date}
                    }
                },
                {
                    "$group": {
                        "_id": "$model",
                        "calls": {"$sum": 1},
                        "tokens": {"$sum": "$total_tokens"},
                        "cost": {"$sum": "$total_cost"}
                    }
                }
            ]
            
            model_breakdown = {}
            for doc in self.api_calls.aggregate(model_pipeline):
                model_breakdown[doc["_id"]] = {
                    "calls": doc["calls"],
                    "tokens": doc["tokens"],
                    "cost": doc["cost"]
                }
            
            # 일별 사용량
            daily_pipeline = [
                {
                    "$match": {
                        "user_id": user_id,
                        "timestamp": {"$gte": start_date}
                    }
                },
                {
                    "$group": {
                        "_id": "$date",
                        "calls": {"$sum": 1},
                        "tokens": {"$sum": "$total_tokens"},
                        "cost": {"$sum": "$total_cost"}
                    }
                },
                {"$sort": {"_id": -1}},
                {"$limit": 30}
            ]
            
            daily_usage = {}
            for doc in self.api_calls.aggregate(daily_pipeline):
                daily_usage[doc["_id"]] = {
                    "calls": doc["calls"],
                    "tokens": doc["tokens"],
                    "cost": doc["cost"]
                }
            
            # 오늘과 이번 달 사용량
            today = datetime.now().strftime("%Y-%m-%d")
            this_month = datetime.now().strftime("%Y-%m")
            
            today_usage = daily_usage.get(today, {"calls": 0, "tokens": 0, "cost": 0.0})
            
            monthly_usage_pipeline = [
                {
                    "$match": {
                        "user_id": user_id,
                        "date": {"$regex": f"^{this_month}"}
                    }
                },
                {
                    "$group": {
                        "_id": None,
                        "calls": {"$sum": 1},
                        "tokens": {"$sum": "$total_tokens"},
                        "cost": {"$sum": "$total_cost"}
                    }
                }
            ]
            
            monthly_result = list(self.api_calls.aggregate(monthly_usage_pipeline))
            monthly_usage = monthly_result[0] if monthly_result else {"calls": 0, "tokens": 0, "cost": 0.0}
            
            return {
                "user_id": user_id,
                "period_days": days,
                "summary": {
                    "total_calls": stats["total_calls"],
                    "successful_calls": stats["successful_calls"],
                    "success_rate": stats["successful_calls"] / stats["total_calls"] if stats["total_calls"] > 0 else 0,
                    "total_tokens": stats["total_tokens"],
                    "total_cost": stats["total_cost"],
                    "avg_processing_time": stats["avg_processing_time"] or 0,
                    "first_call": stats["first_call"],
                    "last_call": stats["last_call"]
                },
                "current_usage": {
                    "today": today_usage,
                    "this_month": monthly_usage
                },
                "breakdowns": {
                    "by_purpose": purpose_breakdown,
                    "by_model": model_breakdown,
                    "by_day": daily_usage
                },
                "limits": self.default_limits,
                "limit_warnings": self._check_usage_limits(user_id, today_usage, monthly_usage)
            }
            
        except Exception as e:
            logger.error(f"사용자 사용량 요약 조회 실패: {e}")
            return {"user_id": user_id, "error": str(e)}

    def _check_usage_limits(self, user_id: str, today_usage: Dict, monthly_usage: Dict) -> Dict[str, Any]:
        """사용량 한도 확인"""
        
        warnings = {
            "daily_token_warning": False,
            "monthly_cost_warning": False,
            "daily_token_exceeded": False,
            "monthly_cost_exceeded": False,
        }
        
        # 일일 토큰 한도 체크
        daily_tokens = today_usage.get("tokens", 0)
        daily_limit = self.default_limits["daily_token_limit"]
        
        if daily_tokens >= daily_limit:
            warnings["daily_token_exceeded"] = True
        elif daily_tokens >= daily_limit * 0.8:
            warnings["daily_token_warning"] = True
        
        # 월간 비용 한도 체크
        monthly_cost = monthly_usage.get("cost", 0.0)
        monthly_limit = self.default_limits["monthly_cost_limit"]
        
        if monthly_cost >= monthly_limit:
            warnings["monthly_cost_exceeded"] = True
        elif monthly_cost >= monthly_limit * 0.8:
            warnings["monthly_cost_warning"] = True
        
        warnings.update({
            "daily_usage_percentage": (daily_tokens / daily_limit) * 100 if daily_limit > 0 else 0,
            "monthly_usage_percentage": (monthly_cost / monthly_limit) * 100 if monthly_limit > 0 else 0,
        })
        
        return warnings

    def get_system_usage_summary(self, days: int = 7) -> Dict[str, Any]:
        """시스템 전체 사용량 요약"""
        
        try:
            start_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            # 전체 통계
            overall_pipeline = [
                {
                    "$match": {
                        "timestamp": {"$gte": start_date}
                    }
                },
                {
                    "$group": {
                        "_id": None,
                        "total_calls": {"$sum": 1},
                        "successful_calls": {"$sum": {"$cond": ["$success", 1, 0]}},
                        "total_tokens": {"$sum": "$total_tokens"},
                        "total_cost": {"$sum": "$total_cost"},
                        "unique_users": {"$addToSet": "$user_id"}
                    }
                }
            ]
            
            overall_result = list(self.api_calls.aggregate(overall_pipeline))
            if not overall_result:
                return {"period_days": days, "total_calls": 0, "error": "데이터 없음"}
            
            overall = overall_result[0]
            
            # 상위 사용자
            top_users_pipeline = [
                {
                    "$match": {
                        "timestamp": {"$gte": start_date}
                    }
                },
                {
                    "$group": {
                        "_id": "$user_id",
                        "calls": {"$sum": 1},
                        "tokens": {"$sum": "$total_tokens"},
                        "cost": {"$sum": "$total_cost"}
                    }
                },
                {"$sort": {"cost": -1}},
                {"$limit": 10}
            ]
            
            top_users = []
            for doc in self.api_calls.aggregate(top_users_pipeline):
                top_users.append({
                    "user_id": doc["_id"],
                    "calls": doc["calls"],
                    "tokens": doc["tokens"],
                    "cost": doc["cost"]
                })
            
            # 모델별 사용량
            model_pipeline = [
                {
                    "$match": {
                        "timestamp": {"$gte": start_date}
                    }
                },
                {
                    "$group": {
                        "_id": "$model",
                        "calls": {"$sum": 1},
                        "tokens": {"$sum": "$total_tokens"},
                        "cost": {"$sum": "$total_cost"}
                    }
                },
                {"$sort": {"cost": -1}}
            ]
            
            model_usage = []
            for doc in self.api_calls.aggregate(model_pipeline):
                model_usage.append({
                    "model": doc["_id"],
                    "calls": doc["calls"],
                    "tokens": doc["tokens"],
                    "cost": doc["cost"]
                })
            
            # 일별 트렌드
            daily_trend_pipeline = [
                {
                    "$match": {
                        "timestamp": {"$gte": start_date}
                    }
                },
                {
                    "$group": {
                        "_id": "$date",
                        "calls": {"$sum": 1},
                        "tokens": {"$sum": "$total_tokens"},
                        "cost": {"$sum": "$total_cost"},
                        "unique_users": {"$addToSet": "$user_id"}
                    }
                },
                {"$sort": {"_id": 1}}
            ]
            
            daily_trend = []
            for doc in self.api_calls.aggregate(daily_trend_pipeline):
                daily_trend.append({
                    "date": doc["_id"],
                    "calls": doc["calls"],
                    "tokens": doc["tokens"],
                    "cost": doc["cost"],
                    "unique_users": len(doc["unique_users"])
                })
            
            return {
                "period_days": days,
                "overall": {
                    "total_calls": overall["total_calls"],
                    "successful_calls": overall["successful_calls"],
                    "success_rate": overall["successful_calls"] / overall["total_calls"] if overall["total_calls"] > 0 else 0,
                    "total_tokens": overall["total_tokens"],
                    "total_cost": overall["total_cost"],
                    "unique_users": len(overall["unique_users"]),
                    "avg_cost_per_user": overall["total_cost"] / len(overall["unique_users"]) if overall["unique_users"] else 0
                },
                "top_users": top_users,
                "model_usage": model_usage,
                "daily_trend": daily_trend
            }
            
        except Exception as e:
            logger.error(f"시스템 사용량 요약 조회 실패: {e}")
            return {"error": str(e)}

    def get_cost_analytics(self, user_id: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
        """비용 분석 (사용자별 또는 전체)"""
        
        try:
            start_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            match_filter = {"timestamp": {"$gte": start_date}}
            if user_id:
                match_filter["user_id"] = user_id
            
            # 비용 효율성 분석
            efficiency_pipeline = [
                {"$match": match_filter},
                {
                    "$group": {
                        "_id": "$model",
                        "total_calls": {"$sum": 1},
                        "total_cost": {"$sum": "$total_cost"},
                        "total_tokens": {"$sum": "$total_tokens"},
                        "avg_cost_per_call": {"$avg": "$total_cost"},
                        "avg_tokens_per_call": {"$avg": "$total_tokens"}
                    }
                },
                {
                    "$addFields": {
                        "cost_per_token": {"$divide": ["$total_cost", "$total_tokens"]}
                    }
                },
                {"$sort": {"cost_per_token": 1}}
            ]
            
            efficiency_data = []
            for doc in self.api_calls.aggregate(efficiency_pipeline):
                efficiency_data.append({
                    "model": doc["_id"],
                    "total_calls": doc["total_calls"],
                    "total_cost": doc["total_cost"],
                    "total_tokens": doc["total_tokens"],
                    "avg_cost_per_call": doc["avg_cost_per_call"],
                    "avg_tokens_per_call": doc["avg_tokens_per_call"],
                    "cost_per_token": doc.get("cost_per_token", 0)
                })
            
            # 목적별 비용 분석
            purpose_pipeline = [
                {"$match": match_filter},
                {
                    "$group": {
                        "_id": "$purpose",
                        "calls": {"$sum": 1},
                        "cost": {"$sum": "$total_cost"},
                        "tokens": {"$sum": "$total_tokens"}
                    }
                },
                {"$sort": {"cost": -1}}
            ]
            
            purpose_costs = []
            for doc in self.api_calls.aggregate(purpose_pipeline):
                purpose_costs.append({
                    "purpose": doc["_id"],
                    "calls": doc["calls"],
                    "cost": doc["cost"],
                    "tokens": doc["tokens"]
                })
            
            # 비용 절약 제안
            suggestions = self._generate_cost_suggestions(efficiency_data, purpose_costs)
            
            return {
                "user_id": user_id,
                "period_days": days,
                "model_efficiency": efficiency_data,
                "purpose_costs": purpose_costs,
                "cost_suggestions": suggestions
            }
            
        except Exception as e:
            logger.error(f"비용 분석 실패: {e}")
            return {"error": str(e)}

    def _generate_cost_suggestions(self, efficiency_data: List[Dict], purpose_costs: List[Dict]) -> List[str]:
        """비용 절약 제안 생성"""
        suggestions = []
        
        if len(efficiency_data) > 1:
            # 가장 비효율적인 모델과 효율적인 모델 비교
            least_efficient = max(efficiency_data, key=lambda x: x.get("cost_per_token", 0))
            most_efficient = min(efficiency_data, key=lambda x: x.get("cost_per_token", 0))
            
            if least_efficient["cost_per_token"] > most_efficient["cost_per_token"] * 2:
                suggestions.append(
                    f"{least_efficient['model']}에서 {most_efficient['model']}로 전환하면 "
                    f"토큰당 비용을 {((least_efficient['cost_per_token'] - most_efficient['cost_per_token']) / least_efficient['cost_per_token'] * 100):.1f}% 절약할 수 있습니다."
                )
        
        # 가장 비용이 높은 목적 식별
        if purpose_costs:
            highest_cost_purpose = purpose_costs[0]
            if highest_cost_purpose["cost"] > sum(p["cost"] for p in purpose_costs) * 0.5:
                suggestions.append(
                    f"'{highest_cost_purpose['purpose']}' 목적의 사용량이 전체 비용의 큰 부분을 차지합니다. "
                    "프롬프트 최적화를 검토해보세요."
                )
        
        return suggestions

    def cleanup_old_records(self, days_old: int = 90) -> int:
        """오래된 기록 정리"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()
            
            delete_result = self.api_calls.delete_many({"timestamp": {"$lt": cutoff_date}})
            deleted_count = delete_result.deleted_count
            
            logger.info(f"오래된 비용 추적 기록 {deleted_count}개가 정리되었습니다.")
            return deleted_count
            
        except Exception as e:
            logger.error(f"오래된 기록 정리 실패: {e}")
            return 0

    def export_usage_data(self, user_id: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """사용량 데이터 내보내기"""
        try:
            query = {
                "user_id": user_id,
                "timestamp": {
                    "$gte": start_date,
                    "$lte": end_date
                }
            }
            
            export_data = []
            for doc in self.api_calls.find(query).sort("timestamp", 1):
                # _id를 문자열로 변환
                doc["_id"] = str(doc["_id"])
                export_data.append(doc)
            
            logger.info(f"사용자 {user_id}의 사용량 데이터 {len(export_data)}건을 내보냈습니다.")
            return export_data
            
        except Exception as e:
            logger.error(f"사용량 데이터 내보내기 실패: {e}")
            return []

    def get_system_status(self) -> Dict[str, Any]:
        """비용 추적 시스템 상태"""
        try:
            # 총 기록 수
            total_records = self.api_calls.count_documents({})
            
            # 최근 24시간 기록 수
            yesterday = (datetime.now() - timedelta(days=1)).isoformat()
            recent_records = self.api_calls.count_documents({"timestamp": {"$gte": yesterday}})
            
            return {
                "database_ready": True,
                "mongodb_migration_complete": True,
                "total_records": total_records,
                "recent_records_24h": recent_records,
                "supported_models": list(self.model_costs.keys()),
                "default_limits": self.default_limits
            }
            
        except Exception as e:
            logger.error(f"비용 추적 시스템 상태 확인 실패: {e}")
            return {"database_ready": False, "error": str(e)}