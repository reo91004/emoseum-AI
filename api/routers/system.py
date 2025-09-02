# api/routers/system.py

# ==============================================================================
# 이 파일은 시스템 모니터링 및 헬스 체크 API 엔드포인트를 정의한다.
# 시스템 상태, 비용 통계, 메트릭 등의 정보를 제공한다.
# ==============================================================================

import logging
import psutil
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from ..database.connection import get_database, mongodb
from ..database.collections import Collections
from ..dependencies import get_current_user, get_act_therapy_system

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/system", tags=["System"])


@router.get("/status")
async def get_system_status(
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """시스템 상태 조회"""
    try:
        # 시스템 리소스
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # 데이터베이스 상태
        db_healthy = await mongodb.health_check()
        
        # 서비스 상태
        services = {
            "database": "healthy" if db_healthy else "unhealthy",
            "gpt_service": "healthy",  # 실제 헬스 체크 추가 가능
            "image_generation": "healthy",  # 실제 헬스 체크 추가 가능
        }
        
        # API 버전
        api_version = "1.0.0"
        
        return {
            "status": "operational" if all(s == "healthy" for s in services.values()) else "degraded",
            "timestamp": datetime.utcnow(),
            "api_version": api_version,
            "resources": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2)
            },
            "services": services,
            "environment": os.getenv("ENVIRONMENT", "development")
        }
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return {
            "status": "error",
            "timestamp": datetime.utcnow(),
            "error": str(e)
        }


@router.get("/health")
async def health_check(
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """헬스 체크 엔드포인트"""
    try:
        # 데이터베이스 확인
        db_healthy = await mongodb.health_check()
        
        if not db_healthy:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="데이터베이스 비정상"
            )
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="서비스 비정상"
        )


@router.get("/costs")
async def get_api_costs(
    days: int = 30,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """API 사용 비용 조회"""
    try:
        # 날짜 범위 계산
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # 쿼리 구성
        query = {}
        if current_user:
            query["user_id"] = current_user["user_id"]
        
        # 비용 집계
        pipeline = [
            {"$match": query},
            {"$unwind": "$api_calls"},
            {
                "$match": {
                    "api_calls.timestamp": {"$gte": start_date, "$lte": end_date}
                }
            },
            {
                "$group": {
                    "_id": {
                        "service": "$api_calls.service",
                        "request_type": "$api_calls.request_type"
                    },
                    "total_calls": {"$sum": 1},
                    "total_tokens": {"$sum": "$api_calls.tokens_used"},
                    "total_cost": {"$sum": "$api_calls.cost"}
                }
            }
        ]
        
        cursor = db[Collections.COST_TRACKING].aggregate(pipeline)
        
        # 결과 처리
        costs_by_service = {}
        total_cost = 0
        total_tokens = 0
        total_calls = 0
        
        async for result in cursor:
            service = result["_id"]["service"]
            request_type = result["_id"]["request_type"]
            
            if service not in costs_by_service:
                costs_by_service[service] = {
                    "total_cost": 0,
                    "total_tokens": 0,
                    "total_calls": 0,
                    "by_type": {}
                }
            
            costs_by_service[service]["total_cost"] += result["total_cost"]
            costs_by_service[service]["total_tokens"] += result["total_tokens"]
            costs_by_service[service]["total_calls"] += result["total_calls"]
            
            costs_by_service[service]["by_type"][request_type] = {
                "calls": result["total_calls"],
                "tokens": result["total_tokens"],
                "cost": result["total_cost"]
            }
            
            total_cost += result["total_cost"]
            total_tokens += result["total_tokens"]
            total_calls += result["total_calls"]
        
        # 인증된 경우 사용자 제한 조회
        limits = None
        if current_user:
            user_cost_data = await db[Collections.COST_TRACKING].find_one(
                {"user_id": current_user["user_id"]}
            )
            if user_cost_data:
                limits = user_cost_data.get("limits", {})
        
        return {
            "period": {
                "start": start_date,
                "end": end_date,
                "days": days
            },
            "total_cost": round(total_cost, 4),
            "total_tokens": total_tokens,
            "total_calls": total_calls,
            "costs_by_service": costs_by_service,
            "user_limits": limits,
            "currency": "USD"
        }
        
    except Exception as e:
        logger.error(f"Error getting API costs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API 비용 조회 실패"
        )


@router.get("/metrics")
async def get_system_metrics(
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """시스템 메트릭 조회"""
    try:
        # 사용자 메트릭
        total_users = await db[Collections.USERS].count_documents({})
        active_users = await db[Collections.USERS].count_documents({
            "last_login": {"$gte": datetime.utcnow() - timedelta(days=7)}
        })
        
        # 세션 메트릭
        total_sessions = await db[Collections.GALLERY_ITEMS].count_documents({})
        completed_sessions = await db[Collections.GALLERY_ITEMS].count_documents({
            "is_completed": True
        })
        
        # 완료율 계산
        completion_rate = (completed_sessions / total_sessions * 100) if total_sessions > 0 else 0
        
        # 평균 세션 지속 시간
        pipeline = [
            {"$match": {"is_completed": True}},
            {
                "$group": {
                    "_id": None,
                    "avg_duration": {"$avg": 1800}
                }
            }
        ]
        
        cursor = db[Collections.GALLERY_ITEMS].aggregate(pipeline)
        avg_duration_result = await cursor.to_list(length=1)
        avg_duration = avg_duration_result[0]["avg_duration"] if avg_duration_result else 0
        
        return {
            "users": {
                "total": total_users,
                "active_7d": active_users,
                "active_percentage": round((active_users / total_users * 100) if total_users > 0 else 0, 2)
            },
            "sessions": {
                "total": total_sessions,
                "completed": completed_sessions,
                "completion_rate": round(completion_rate, 2),
                "average_duration_seconds": avg_duration
            },
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="시스템 메트릭 조회 실패"
        )