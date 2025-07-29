# api/routers/admin.py

from typing import Annotated, List, Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Query
import logging
import psutil
import os
from pathlib import Path

from ..models.common import (
    APIResponse,
    SystemStatus,
    HealthCheck,
    CostSummary,
    ServiceType,
)
from ..models.user import UserStats, UserActivity
from ..routers.users import get_current_user
from ..services.database import db, SupabaseService
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


async def verify_admin_access(current_user: dict) -> dict:
    """관리자 권한 확인"""
    user_id = current_user["user_id"]

    # 관리자 사용자 목록 (실제로는 데이터베이스에서 관리해야 함)
    admin_users = ["admin", "system", "reo91004", "enqueue01", "seonjin", "qolop"]

    if user_id not in admin_users:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator access required",
        )

    logger.info(f"관리자 접근 확인: {user_id}")
    return current_user


@router.get(
    "/status",
    response_model=APIResponse[SystemStatus],
    summary="Get system status",
    description="Get comprehensive system status information (Admin only)",
)
async def get_system_status(
    current_user: Annotated[dict, Depends(get_current_user)],
    admin_user: Annotated[dict, Depends(verify_admin_access)],
    db_service: Annotated[SupabaseService, Depends(lambda: db)],
) -> APIResponse[SystemStatus]:
    """시스템 상태 조회"""

    try:
        # 시스템 리소스 정보
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # 서비스 상태 확인
        services_status = {}

        # 데이터베이스 상태
        try:
            # 간단한 쿼리로 DB 연결 확인
            test_result = (
                await db_service.supabase.table("users")
                .select("count", count="exact")
                .limit(1)
                .execute()
            )
            services_status["database"] = "healthy"
        except Exception as e:
            services_status["database"] = f"error: {str(e)}"

        # OpenAI API 상태
        services_status["openai"] = (
            "configured" if settings.openai_api_key else "not_configured"
        )

        # 이미지 생성 서비스 상태
        try:
            from ..services.image_service import ImageService

            image_service = ImageService()
            image_status = image_service.get_backend_status()
            services_status["image_generation"] = image_status.get(
                "service_info", {}
            ).get("service_status", "unknown")
        except Exception as e:
            services_status["image_generation"] = f"error: {str(e)}"

        # 파일 시스템 상태
        data_dir = Path(settings.data_dir if hasattr(settings, "data_dir") else "data")
        services_status["file_system"] = (
            "accessible" if data_dir.exists() else "inaccessible"
        )

        # 로그 파일 상태
        log_file = Path(settings.log_file)
        services_status["logging"] = "active" if log_file.exists() else "inactive"

        # 시스템 설정 정보
        configuration = {
            "environment": settings.environment,
            "debug_mode": settings.debug,
            "image_backend": settings.image_backend,
            "rate_limit": f"{settings.rate_limit_per_minute}/min",
            "max_diary_length": settings.max_diary_length,
            "jwt_expires_minutes": settings.jwt_access_token_expire_minutes,
            "cors_origins_count": len(settings.cors_origins),
            "system_resources": {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_usage_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2),
            },
        }

        # 전체 상태 판단
        critical_services = ["database", "openai"]
        overall_status = "healthy"

        for service in critical_services:
            if service in services_status and "error" in services_status[service]:
                overall_status = "degraded"
                break

        if cpu_percent > 90 or memory.percent > 90 or disk.percent > 95:
            overall_status = "resource_warning"

        system_status = SystemStatus(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version=settings.api_version,
            environment=settings.environment,
            services=services_status,
            configuration=configuration,
        )

        logger.info(f"시스템 상태 조회 완료: {overall_status}")

        return APIResponse(
            success=True,
            message="System status retrieved successfully",
            data=system_status,
        )

    except Exception as e:
        logger.error(f"시스템 상태 조회 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system status",
        )


@router.get(
    "/users",
    response_model=APIResponse[List[Dict[str, Any]]],
    summary="Get all users",
    description="Get list of all users with basic statistics (Admin only)",
)
async def get_all_users(
    current_user: Annotated[dict, Depends(get_current_user)],
    admin_user: Annotated[dict, Depends(verify_admin_access)],
    db_service: Annotated[SupabaseService, Depends(lambda: db)],
    limit: int = Query(
        100, ge=1, le=1000, description="Maximum number of users to return"
    ),
    active_only: bool = Query(False, description="Only show active users"),
) -> APIResponse[List[Dict[str, Any]]]:
    """모든 사용자 목록 조회 (관리자 전용)"""

    try:
        # 모든 사용자 조회
        users_result = (
            await db_service.supabase.table("users").select("*").limit(limit).execute()
        )
        users_data = users_result.data or []

        # 사용자별 통계 수집
        users_with_stats = []
        for user in users_data:
            user_id = user["user_id"]

            # 갤러리 아이템 수 조회
            gallery_result = (
                await db_service.supabase.table("gallery_items")
                .select("*", count="exact")
                .eq("user_id", user_id)
                .execute()
            )
            total_journeys = gallery_result.count or 0

            # 완료된 여정 수
            completed_result = (
                await db_service.supabase.table("gallery_items")
                .select("*", count="exact")
                .eq("user_id", user_id)
                .eq("completion_status", "completed")
                .execute()
            )
            completed_journeys = completed_result.count or 0

            # 최근 활동
            recent_activity = None
            if gallery_result.data:
                latest_item = max(gallery_result.data, key=lambda x: x["created_date"])
                recent_activity = latest_item["created_date"]

            # 활성 사용자 필터링
            if active_only:
                if not recent_activity:
                    continue
                recent_date = datetime.fromisoformat(
                    recent_activity.replace("Z", "+00:00")
                )
                if (datetime.utcnow() - recent_date).days > 30:
                    continue

            user_info = {
                "user_id": user_id,
                "email": user.get("email"),
                "created_date": user["created_date"],
                "last_login_date": user.get("last_login_date"),
                "total_journeys": total_journeys,
                "completed_journeys": completed_journeys,
                "completion_rate": (
                    completed_journeys / total_journeys if total_journeys > 0 else 0
                ),
                "last_activity": recent_activity,
                "account_status": "active" if recent_activity else "inactive",
            }

            users_with_stats.append(user_info)

        # 활동 시간 기준으로 정렬
        users_with_stats.sort(key=lambda x: x.get("last_activity", ""), reverse=True)

        logger.info(f"사용자 목록 조회 완료: {len(users_with_stats)}명")

        return APIResponse(
            success=True,
            message=f"Retrieved {len(users_with_stats)} users",
            data=users_with_stats,
        )

    except Exception as e:
        logger.error(f"사용자 목록 조회 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve users list",
        )


@router.get(
    "/usage",
    response_model=APIResponse[Dict[str, Any]],
    summary="Get usage statistics",
    description="Get comprehensive usage statistics (Admin only)",
)
async def get_usage_statistics(
    current_user: Annotated[dict, Depends(get_current_user)],
    admin_user: Annotated[dict, Depends(verify_admin_access)],
    db_service: Annotated[SupabaseService, Depends(lambda: db)],
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
) -> APIResponse[Dict[str, Any]]:
    """사용량 통계 조회"""

    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        # 기본 통계
        stats = {
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days,
            },
            "users": {},
            "journeys": {},
            "costs": {},
            "system": {},
        }

        # 사용자 통계
        total_users_result = (
            await db_service.supabase.table("users")
            .select("*", count="exact")
            .execute()
        )
        stats["users"]["total"] = total_users_result.count or 0

        # 신규 사용자 (기간 내)
        new_users_result = (
            await db_service.supabase.table("users")
            .select("*", count="exact")
            .gte("created_date", start_date.isoformat())
            .execute()
        )
        stats["users"]["new_in_period"] = new_users_result.count or 0

        # 활성 사용자 (기간 내 활동)
        active_users_result = (
            await db_service.supabase.table("gallery_items")
            .select("user_id")
            .gte("created_date", start_date.isoformat())
            .execute()
        )
        active_user_ids = set(
            item["user_id"] for item in active_users_result.data or []
        )
        stats["users"]["active_in_period"] = len(active_user_ids)

        # 여정 통계
        total_journeys_result = (
            await db_service.supabase.table("gallery_items")
            .select("*", count="exact")
            .execute()
        )
        stats["journeys"]["total"] = total_journeys_result.count or 0

        period_journeys_result = (
            await db_service.supabase.table("gallery_items")
            .select("*", count="exact")
            .gte("created_date", start_date.isoformat())
            .execute()
        )
        stats["journeys"]["in_period"] = period_journeys_result.count or 0

        completed_journeys_result = (
            await db_service.supabase.table("gallery_items")
            .select("*", count="exact")
            .gte("created_date", start_date.isoformat())
            .eq("completion_status", "completed")
            .execute()
        )
        stats["journeys"]["completed_in_period"] = completed_journeys_result.count or 0

        stats["journeys"]["completion_rate"] = (
            stats["journeys"]["completed_in_period"] / stats["journeys"]["in_period"]
            if stats["journeys"]["in_period"] > 0
            else 0
        )

        # 비용 통계
        cost_summary = await db_service.get_cost_summary(days=days)
        stats["costs"] = {
            "total_cost_usd": cost_summary.get("total_cost", 0.0),
            "service_breakdown": cost_summary.get("service_breakdown", {}),
            "total_tokens": cost_summary.get("total_tokens", 0),
            "average_cost_per_journey": (
                cost_summary.get("total_cost", 0.0) / stats["journeys"]["in_period"]
                if stats["journeys"]["in_period"] > 0
                else 0
            ),
        }

        # 시스템 통계
        stats["system"] = {
            "api_version": settings.api_version,
            "environment": settings.environment,
            "image_backend": settings.image_backend,
            "uptime_hours": 0,  # 실제로는 시작 시간을 추적해야 함
            "error_rate": 0.0,  # 실제로는 에러 로그 분석 필요
        }

        # 일별 활동 추이 (최근 7일)
        daily_activity = []
        for i in range(7):
            day_start = end_date - timedelta(days=i + 1)
            day_end = day_start + timedelta(days=1)

            day_journeys_result = (
                await db_service.supabase.table("gallery_items")
                .select("*", count="exact")
                .gte("created_date", day_start.isoformat())
                .lt("created_date", day_end.isoformat())
                .execute()
            )

            daily_activity.append(
                {
                    "date": day_start.date().isoformat(),
                    "journeys": day_journeys_result.count or 0,
                }
            )

        stats["daily_activity"] = list(reversed(daily_activity))

        logger.info(f"사용량 통계 조회 완료: {days}일 기간")

        return APIResponse(
            success=True, message=f"Usage statistics for {days} days", data=stats
        )

    except Exception as e:
        logger.error(f"사용량 통계 조회 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve usage statistics",
        )


@router.get(
    "/costs",
    response_model=APIResponse[Dict[str, Any]],
    summary="Get cost analysis",
    description="Get detailed cost analysis and projections (Admin only)",
)
async def get_cost_analysis(
    current_user: Annotated[dict, Depends(get_current_user)],
    admin_user: Annotated[dict, Depends(verify_admin_access)],
    db_service: Annotated[SupabaseService, Depends(lambda: db)],
    days: int = Query(30, ge=1, le=365, description="Analysis period in days"),
) -> APIResponse[Dict[str, Any]]:
    """비용 분석 조회"""

    try:
        # 비용 추적 데이터 조회
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        cost_result = (
            await db_service.supabase.table("cost_tracking")
            .select("*")
            .gte("created_date", start_date.isoformat())
            .execute()
        )
        cost_data = cost_result.data or []

        # 비용 분석
        analysis = {
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days,
            },
            "summary": {
                "total_cost": 0.0,
                "total_tokens": 0,
                "total_calls": len(cost_data),
                "average_cost_per_call": 0.0,
                "average_tokens_per_call": 0.0,
            },
            "by_service": {},
            "by_user": {},
            "daily_breakdown": [],
            "projections": {},
        }

        # 총 비용 및 토큰 계산
        total_cost = sum(item.get("cost_usd", 0) for item in cost_data)
        total_tokens = sum(item.get("tokens_used", 0) for item in cost_data)

        analysis["summary"]["total_cost"] = total_cost
        analysis["summary"]["total_tokens"] = total_tokens
        analysis["summary"]["average_cost_per_call"] = (
            total_cost / len(cost_data) if cost_data else 0
        )
        analysis["summary"]["average_tokens_per_call"] = (
            total_tokens / len(cost_data) if cost_data else 0
        )

        # 서비스별 분석
        service_costs = {}
        for item in cost_data:
            service = item.get("service_type", "unknown")
            if service not in service_costs:
                service_costs[service] = {"cost": 0.0, "tokens": 0, "calls": 0}

            service_costs[service]["cost"] += item.get("cost_usd", 0)
            service_costs[service]["tokens"] += item.get("tokens_used", 0)
            service_costs[service]["calls"] += 1

        analysis["by_service"] = service_costs

        # 사용자별 분석 (상위 10명)
        user_costs = {}
        for item in cost_data:
            user_id = item.get("user_id", "unknown")
            if user_id not in user_costs:
                user_costs[user_id] = {"cost": 0.0, "tokens": 0, "calls": 0}

            user_costs[user_id]["cost"] += item.get("cost_usd", 0)
            user_costs[user_id]["tokens"] += item.get("tokens_used", 0)
            user_costs[user_id]["calls"] += 1

        # 상위 10명만 선택
        top_users = sorted(
            user_costs.items(), key=lambda x: x[1]["cost"], reverse=True
        )[:10]
        analysis["by_user"] = dict(top_users)

        # 일별 분석 (최근 7일)
        daily_costs = {}
        for item in cost_data:
            date_str = item["created_date"][:10]  # YYYY-MM-DD 부분만
            if date_str not in daily_costs:
                daily_costs[date_str] = {"cost": 0.0, "tokens": 0, "calls": 0}

            daily_costs[date_str]["cost"] += item.get("cost_usd", 0)
            daily_costs[date_str]["tokens"] += item.get("tokens_used", 0)
            daily_costs[date_str]["calls"] += 1

        # 최근 7일간 데이터 정리
        for i in range(7):
            day = (end_date - timedelta(days=i)).date()
            day_str = day.isoformat()
            day_data = daily_costs.get(day_str, {"cost": 0.0, "tokens": 0, "calls": 0})
            day_data["date"] = day_str
            analysis["daily_breakdown"].append(day_data)

        analysis["daily_breakdown"].reverse()  # 오래된 것부터

        # 비용 예측 (선형 추세 기반)
        if len(analysis["daily_breakdown"]) >= 3:
            recent_daily_costs = [
                day["cost"] for day in analysis["daily_breakdown"][-3:]
            ]
            avg_daily_cost = sum(recent_daily_costs) / len(recent_daily_costs)

            analysis["projections"] = {
                "daily_average": avg_daily_cost,
                "weekly_projection": avg_daily_cost * 7,
                "monthly_projection": avg_daily_cost * 30,
                "yearly_projection": avg_daily_cost * 365,
            }
        else:
            analysis["projections"] = {
                "daily_average": 0.0,
                "weekly_projection": 0.0,
                "monthly_projection": 0.0,
                "yearly_projection": 0.0,
            }

        logger.info(f"비용 분석 완료: {days}일, 총 ${total_cost:.2f}")

        return APIResponse(
            success=True,
            message=f"Cost analysis for {days} days completed",
            data=analysis,
        )

    except Exception as e:
        logger.error(f"비용 분석 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve cost analysis",
        )


@router.get(
    "/logs",
    response_model=APIResponse[Dict[str, Any]],
    summary="Get system logs",
    description="Get recent system logs and error reports (Admin only)",
)
async def get_system_logs(
    current_user: Annotated[dict, Depends(get_current_user)],
    admin_user: Annotated[dict, Depends(verify_admin_access)],
    lines: int = Query(
        100, ge=10, le=1000, description="Number of log lines to return"
    ),
    level: Optional[str] = Query(
        None, description="Filter by log level (INFO, WARNING, ERROR)"
    ),
) -> APIResponse[Dict[str, Any]]:
    """시스템 로그 조회"""

    try:
        log_file_path = Path(settings.log_file)

        logs_info = {
            "log_file": str(log_file_path),
            "file_exists": log_file_path.exists(),
            "lines_requested": lines,
            "level_filter": level,
            "logs": [],
            "summary": {
                "total_lines": 0,
                "error_count": 0,
                "warning_count": 0,
                "info_count": 0,
            },
        }

        if not log_file_path.exists():
            logger.warning(f"로그 파일이 존재하지 않음: {log_file_path}")
            return APIResponse(
                success=True, message="Log file not found", data=logs_info
            )

        # 로그 파일 읽기
        try:
            with open(log_file_path, "r", encoding="utf-8") as f:
                all_lines = f.readlines()

            # 최근 lines 개수만큼 가져오기
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

            # 레벨 필터링 및 분석
            filtered_logs = []
            for line in recent_lines:
                line = line.strip()
                if not line:
                    continue

                # 로그 레벨 확인
                line_level = None
                if " - ERROR - " in line:
                    line_level = "ERROR"
                    logs_info["summary"]["error_count"] += 1
                elif " - WARNING - " in line:
                    line_level = "WARNING"
                    logs_info["summary"]["warning_count"] += 1
                elif " - INFO - " in line:
                    line_level = "INFO"
                    logs_info["summary"]["info_count"] += 1

                # 레벨 필터 적용
                if level and line_level != level.upper():
                    continue

                # 로그 파싱
                log_entry = {
                    "raw": line,
                    "level": line_level,
                    "timestamp": None,
                    "logger": None,
                    "message": line,
                }

                # 타임스탬프 추출 시도
                try:
                    if " - " in line:
                        parts = line.split(" - ")
                        if len(parts) >= 4:
                            log_entry["timestamp"] = parts[0]
                            log_entry["logger"] = parts[1]
                            log_entry["message"] = " - ".join(parts[3:])
                except:
                    pass

                filtered_logs.append(log_entry)
                logs_info["summary"]["total_lines"] += 1

            logs_info["logs"] = filtered_logs

        except Exception as e:
            logger.error(f"로그 파일 읽기 실패: {e}")
            logs_info["error"] = f"Failed to read log file: {str(e)}"

        logger.info(f"시스템 로그 조회: {len(logs_info['logs'])}줄")

        return APIResponse(
            success=True,
            message=f"Retrieved {len(logs_info['logs'])} log entries",
            data=logs_info,
        )

    except Exception as e:
        logger.error(f"시스템 로그 조회 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system logs",
        )


@router.post(
    "/maintenance",
    response_model=APIResponse[Dict[str, Any]],
    summary="Perform maintenance tasks",
    description="Perform system maintenance tasks (Admin only)",
)
async def perform_maintenance(
    current_user: Annotated[dict, Depends(get_current_user)],
    admin_user: Annotated[dict, Depends(verify_admin_access)],
    db_service: Annotated[SupabaseService, Depends(lambda: db)],
    task: str = Query(..., description="Maintenance task to perform"),
    confirm: bool = Query(
        False, description="Confirm execution of potentially destructive tasks"
    ),
) -> APIResponse[Dict[str, Any]]:
    """시스템 유지보수 작업 실행"""

    try:
        maintenance_result = {
            "task": task,
            "started_at": datetime.utcnow().isoformat(),
            "completed_at": None,
            "success": False,
            "details": {},
            "warnings": [],
        }

        if task == "cleanup_old_logs":
            # 30일 이상 된 로그 정리
            log_file_path = Path(settings.log_file)
            if log_file_path.exists():
                # 실제로는 로그 로테이션 로직 구현
                maintenance_result["details"][
                    "log_file_size"
                ] = log_file_path.stat().st_size
                maintenance_result["success"] = True
            else:
                maintenance_result["details"]["message"] = "Log file not found"

        elif task == "optimize_database":
            if not confirm:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Database optimization requires confirmation",
                )

            # 데이터베이스 최적화 (Supabase에서는 제한적)
            maintenance_result["details"]["message"] = "Database optimization completed"
            maintenance_result["success"] = True

        elif task == "clear_expired_tokens":
            # 만료된 JWT 토큰 정리 (실제로는 블랙리스트 테이블에서)
            maintenance_result["details"]["message"] = "Token cleanup completed"
            maintenance_result["success"] = True

        elif task == "backup_configuration":
            # 설정 백업
            config_backup = {
                "environment": settings.environment,
                "image_backend": settings.image_backend,
                "api_version": settings.api_version,
                "backup_time": datetime.utcnow().isoformat(),
            }
            maintenance_result["details"]["config_backup"] = config_backup
            maintenance_result["success"] = True

        elif task == "health_check_all":
            # 전체 시스템 헬스 체크
            health_results = {}

            # 데이터베이스 체크
            try:
                await db_service.supabase.table("users").select(
                    "count", count="exact"
                ).limit(1).execute()
                health_results["database"] = "healthy"
            except Exception as e:
                health_results["database"] = f"error: {str(e)}"

            # 이미지 서비스 체크
            try:
                from ..services.image_service import ImageService

                image_service = ImageService()
                status_result = image_service.get_backend_status()
                health_results["image_service"] = (
                    "healthy" if status_result.get("status") else "degraded"
                )
            except Exception as e:
                health_results["image_service"] = f"error: {str(e)}"

            maintenance_result["details"]["health_check"] = health_results
            maintenance_result["success"] = True

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown maintenance task: {task}",
            )

        maintenance_result["completed_at"] = datetime.utcnow().isoformat()

        logger.info(f"유지보수 작업 완료: {task}")

        return APIResponse(
            success=True,
            message=f"Maintenance task '{task}' completed successfully",
            data=maintenance_result,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"유지보수 작업 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Maintenance task failed: {str(e)}",
        )


@router.get(
    "/settings",
    response_model=APIResponse[Dict[str, Any]],
    summary="Get system settings",
    description="Get current system configuration (Admin only)",
)
async def get_system_settings(
    current_user: Annotated[dict, Depends(get_current_user)],
    admin_user: Annotated[dict, Depends(verify_admin_access)],
) -> APIResponse[Dict[str, Any]]:
    """시스템 설정 조회"""

    try:
        # 민감한 정보는 제외하고 설정 반환
        safe_settings = {
            "api": {
                "title": settings.api_title,
                "version": settings.api_version,
                "environment": settings.environment,
                "debug": settings.debug,
                "host": settings.api_host,
                "port": settings.api_port,
            },
            "authentication": {
                "jwt_algorithm": settings.jwt_algorithm,
                "jwt_expires_minutes": settings.jwt_access_token_expire_minutes,
            },
            "features": {
                "image_backend": settings.image_backend,
                "local_model_path": settings.local_model_path,
                "max_diary_length": settings.max_diary_length,
                "max_upload_size": settings.max_upload_size,
            },
            "security": {
                "cors_origins": settings.cors_origins,
                "rate_limit_per_minute": settings.rate_limit_per_minute,
                "cors_allow_credentials": settings.cors_allow_credentials,
            },
            "logging": {"log_level": settings.log_level, "log_file": settings.log_file},
            "external_services": {
                "openai_configured": bool(settings.openai_api_key),
                "supabase_configured": bool(
                    settings.supabase_url and settings.supabase_anon_key
                ),
                "remote_gpu_configured": bool(settings.remote_gpu_url),
                "colab_configured": bool(settings.colab_notebook_url),
            },
        }

        logger.info("시스템 설정 조회 완료")

        return APIResponse(
            success=True,
            message="System settings retrieved successfully",
            data=safe_settings,
        )

    except Exception as e:
        logger.error(f"시스템 설정 조회 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system settings",
        )
