# api/routers/gallery.py

from typing import Annotated, List, Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path
import logging

from ..models.therapy import (
    GalleryItem,
    GalleryListResponse,
    TherapyAnalytics,
    JourneyExport
)
from ..models.common import APIResponse, CompletionStatus, PaginatedResponse
from ..routers.users import get_current_user
from ..routers.therapy import convert_to_gallery_item
from ..services.database import db, SupabaseService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/gallery", tags=["gallery"])


@router.get(
    "",
    response_model=APIResponse[PaginatedResponse[GalleryItem]],
    summary="Get gallery items",
    description="Retrieve user's gallery items with pagination and filtering"
)
async def get_gallery_items(
    current_user: Annotated[dict, Depends(get_current_user)],
    db_service: Annotated[SupabaseService, Depends(lambda: db)],
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    status_filter: Optional[CompletionStatus] = Query(None, description="Filter by completion status"),
    tag_filter: Optional[str] = Query(None, description="Filter by tag"),
    date_from: Optional[datetime] = Query(None, description="Filter from date"),
    date_to: Optional[datetime] = Query(None, description="Filter to date"),
    search: Optional[str] = Query(None, min_length=2, description="Search in diary text"),
    sort_by: str = Query("created_date", description="Sort field"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$", description="Sort order")
) -> APIResponse[PaginatedResponse[GalleryItem]]:
    """갤러리 아이템 목록 조회 (페이지네이션 포함)"""
    
    try:
        user_id = current_user["user_id"]
        offset = (page - 1) * limit
        
        # 갤러리 아이템들 조회 (필터링 적용)
        gallery_items_data = await db_service.get_gallery_items(
            user_id=user_id,
            limit=limit,
            offset=offset,
            status_filter=status_filter.value if status_filter else None
        )
        
        # 추가 필터링 (클라이언트 사이드)
        filtered_items = []
        for item in gallery_items_data:
            # 날짜 필터
            item_date = datetime.fromisoformat(item["created_date"].replace("Z", "+00:00"))
            if date_from and item_date < date_from:
                continue
            if date_to and item_date > date_to:
                continue
            
            # 태그 필터
            if tag_filter and tag_filter not in item.get("guestbook_tags", []):
                continue
            
            # 검색 필터
            if search and search.lower() not in item.get("diary_text", "").lower():
                continue
            
            filtered_items.append(item)
        
        # 정렬
        reverse = (sort_order == "desc")
        if sort_by == "created_date":
            filtered_items.sort(key=lambda x: x["created_date"], reverse=reverse)
        elif sort_by == "completion_status":
            filtered_items.sort(key=lambda x: x.get("completion_status", ""), reverse=reverse)
        elif sort_by == "total_duration":
            filtered_items.sort(key=lambda x: x.get("total_duration", 0), reverse=reverse)
        
        # GalleryItem 모델들로 변환
        gallery_items = []
        for data in filtered_items:
            item = await convert_to_gallery_item(data)
            gallery_items.append(item)
        
        # 전체 개수 계산
        total_count = len(await db_service.get_gallery_items(user_id))
        total_pages = (total_count + limit - 1) // limit
        
        # 페이지네이션 응답 생성
        paginated_response = PaginatedResponse(
            items=gallery_items,
            total=total_count,
            page=page,
            limit=limit,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        )
        
        logger.info(f"갤러리 목록 조회: {user_id}, 페이지 {page}, {len(gallery_items)}개")
        
        return APIResponse(
            success=True,
            message=f"Retrieved {len(gallery_items)} gallery items",
            data=paginated_response
        )
        
    except Exception as e:
        logger.error(f"갤러리 목록 조회 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve gallery items"
        )


@router.get(
    "/{item_id}",
    response_model=APIResponse[GalleryItem],
    summary="Get gallery item",
    description="Retrieve specific gallery item details"
)
async def get_gallery_item(
    item_id: str = Path(..., description="Gallery item ID"),
    current_user: Annotated[dict, Depends(get_current_user)],
    db_service: Annotated[SupabaseService, Depends(lambda: db)]
) -> APIResponse[GalleryItem]:
    """특정 갤러리 아이템 조회"""
    
    try:
        user_id = current_user["user_id"]
        
        # 갤러리 아이템 조회
        gallery_data = await db_service.get_gallery_item(item_id)
        if not gallery_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Gallery item not found"
            )
        
        # 권한 확인
        if gallery_data["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this gallery item"
            )
        
        # GalleryItem 모델 생성
        gallery_item = await convert_to_gallery_item(gallery_data)
        
        logger.info(f"갤러리 아이템 조회: {item_id}")
        
        return APIResponse(
            success=True,
            message="Gallery item retrieved successfully",
            data=gallery_item
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"갤러리 아이템 조회 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve gallery item"
        )


@router.delete(
    "/{item_id}",
    response_model=APIResponse[dict],
    summary="Delete gallery item",
    description="Delete a gallery item and associated data"
)
async def delete_gallery_item(
    item_id: str = Path(..., description="Gallery item ID"),
    current_user: Annotated[dict, Depends(get_current_user)],
    db_service: Annotated[SupabaseService, Depends(lambda: db)]
) -> APIResponse[dict]:
    """갤러리 아이템 삭제"""
    
    try:
        user_id = current_user["user_id"]
        
        # 갤러리 아이템 존재 및 권한 확인
        gallery_data = await db_service.get_gallery_item(item_id)
        if not gallery_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Gallery item not found"
            )
        
        if gallery_data["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this gallery item"
            )
        
        # 완료된 여정만 삭제 가능
        if gallery_data.get("completion_status") not in [CompletionStatus.COMPLETED.value, CompletionStatus.ARCHIVED.value]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only completed or archived journeys can be deleted"
            )
        
        # 아이템 삭제 (실제 구현에서는 연관 데이터도 삭제해야 함)
        await db_service.delete_gallery_item(item_id)
        
        # TODO: 연관된 이미지 파일도 삭제해야 함
        # TODO: 개인화 학습 데이터 정리
        
        logger.info(f"갤러리 아이템 삭제: {item_id}")
        
        return APIResponse(
            success=True,
            message="Gallery item deleted successfully",
            data={"deleted_item_id": item_id}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"갤러리 아이템 삭제 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete gallery item"
        )


@router.patch(
    "/{item_id}/favorite",
    response_model=APIResponse[dict],
    summary="Toggle favorite status",
    description="Mark or unmark gallery item as favorite"
)
async def toggle_favorite(
    item_id: str = Path(..., description="Gallery item ID"),
    current_user: Annotated[dict, Depends(get_current_user)],
    db_service: Annotated[SupabaseService, Depends(lambda: db)]
) -> APIResponse[dict]:
    """즐겨찾기 토글"""
    
    try:
        user_id = current_user["user_id"]
        
        # 갤러리 아이템 조회
        gallery_data = await db_service.get_gallery_item(item_id)
        if not gallery_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Gallery item not found"
            )
        
        if gallery_data["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this gallery item"
            )
        
        # 즐겨찾기 상태 토글
        current_favorite = gallery_data.get("is_favorite", False)
        new_favorite = not current_favorite
        
        # 업데이트
        update_data = {
            "is_favorite": new_favorite,
            "updated_date": datetime.utcnow().isoformat()
        }
        
        await db_service.update_gallery_item(item_id, update_data)
        
        action = "added to" if new_favorite else "removed from"
        
        logger.info(f"즐겨찾기 토글: {item_id}, {action} favorites")
        
        return APIResponse(
            success=True,
            message=f"Item {action} favorites",
            data={"is_favorite": new_favorite}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"즐겨찾기 토글 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to toggle favorite status"
        )


@router.post(
    "/{item_id}/feedback",
    response_model=APIResponse[dict],
    summary="Submit user feedback",
    description="Submit rating and feedback for a gallery item"
)
async def submit_feedback(
    item_id: str = Path(..., description="Gallery item ID"),
    rating: int = Query(..., ge=1, le=5, description="Rating (1-5)"),
    feedback: Optional[str] = Query(None, max_length=1000, description="Feedback text"),
    current_user: Annotated[dict, Depends(get_current_user)],
    db_service: Annotated[SupabaseService, Depends(lambda: db)]
) -> APIResponse[dict]:
    """사용자 피드백 제출"""
    
    try:
        user_id = current_user["user_id"]
        
        # 갤러리 아이템 조회
        gallery_data = await db_service.get_gallery_item(item_id)
        if not gallery_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Gallery item not found"
            )
        
        if gallery_data["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this gallery item"
            )
        
        # 완료된 여정에만 피드백 가능
        if gallery_data.get("completion_status") != CompletionStatus.COMPLETED.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Feedback can only be submitted for completed journeys"
            )
        
        # 피드백 업데이트
        update_data = {
            "user_rating": rating,
            "user_feedback": feedback,
            "updated_date": datetime.utcnow().isoformat()
        }
        
        await db_service.update_gallery_item(item_id, update_data)
        
        # 개인화 학습을 위한 피드백 데이터 저장
        personalization_data = {
            "user_id": user_id,
            "interaction_type": "journey_feedback",
            "feedback_data": {
                "item_id": item_id,
                "rating": rating,
                "feedback_text": feedback,
                "journey_type": gallery_data.get("dominant_emotion", "unknown")
            },
            "learning_weights": {
                "satisfaction_score": rating / 5.0,
                "engagement_level": 1.0 if feedback else 0.5
            }
        }
        
        await db_service.save_personalization_data(user_id, personalization_data)
        
        logger.info(f"피드백 제출: {item_id}, 평점: {rating}")
        
        return APIResponse(
            success=True,
            message="Feedback submitted successfully",
            data={"rating": rating, "feedback_submitted": bool(feedback)}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"피드백 제출 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit feedback"
        )


@router.get(
    "/analytics",
    response_model=APIResponse[TherapyAnalytics],
    summary="Get therapy analytics",
    description="Get comprehensive analytics of user's therapy journey"
)
async def get_therapy_analytics(
    current_user: Annotated[dict, Depends(get_current_user)],
    db_service: Annotated[SupabaseService, Depends(lambda: db)],
    days: int = Query(90, ge=7, le=365, description="Analysis period in days")
) -> APIResponse[TherapyAnalytics]:
    """치료 분석 데이터 조회"""
    
    try:
        user_id = current_user["user_id"]
        
        # 분석 기간 설정
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # 갤러리 아이템들 조회
        gallery_items = await db_service.get_gallery_items(user_id)
        
        # 기간 내 아이템들 필터링
        period_items = [
            item for item in gallery_items
            if datetime.fromisoformat(item["created_date"].replace("Z", "+00:00")) >= start_date
        ]
        
        # 감정 변화 분석
        emotion_trend = {}
        vad_evolution = []
        emotion_changes = []
        
        for item in sorted(period_items, key=lambda x: x["created_date"]):
            # 감정 키워드 추이
            for emotion in item.get("emotion_keywords", []):
                if emotion not in emotion_trend:
                    emotion_trend[emotion] = []
                emotion_trend[emotion].append(item.get("emotion_intensity", 0.5))
            
            # VAD 점수 진화
            vad_data = item.get("vad_scores", {})
            if vad_data:
                from ..models.therapy import VADScores
                vad_score = VADScores(
                    valence=vad_data.get("valence", 0.0),
                    arousal=vad_data.get("arousal", 0.0),
                    dominance=vad_data.get("dominance", 0.0),
                    confidence=vad_data.get("confidence", 0.5)
                )
                vad_evolution.append(vad_score)
            
            # 주요 감정 변화
            dominant_emotion = item.get("dominant_emotion", "neutral")
            if dominant_emotion not in emotion_changes:
                emotion_changes.append(dominant_emotion)
        
        # 치료 효과 분석
        if len(vad_evolution) >= 2:
            first_valence = vad_evolution[0].valence
            last_valence = vad_evolution[-1].valence
            therapeutic_progress = last_valence - first_valence
        else:
            therapeutic_progress = 0.0
        
        # 돌파구 순간들 (valence가 크게 상승한 지점들)
        breakthrough_moments = []
        for i, item in enumerate(period_items):
            if i > 0:
                prev_valence = period_items[i-1].get("vad_scores", {}).get("valence", 0.0)
                curr_valence = item.get("vad_scores", {}).get("valence", 0.0)
                if curr_valence - prev_valence > 0.3:  # 큰 개선
                    breakthrough_moments.append(
                        datetime.fromisoformat(item["created_date"].replace("Z", "+00:00"))
                    )
        
        # 후퇴 기간들 (valence가 지속적으로 하락한 기간들)
        regression_periods = []
        regression_start = None
        for i, item in enumerate(period_items):
            curr_valence = item.get("vad_scores", {}).get("valence", 0.0)
            if i > 0:
                prev_valence = period_items[i-1].get("vad_scores", {}).get("valence", 0.0)
                if curr_valence < prev_valence - 0.1:  # 하락
                    if not regression_start:
                        regression_start = datetime.fromisoformat(period_items[i-1]["created_date"].replace("Z", "+00:00"))
                else:
                    if regression_start:
                        regression_periods.append({
                            "start": regression_start,
                            "end": datetime.fromisoformat(item["created_date"].replace("Z", "+00:00")),
                            "severity": "mild"
                        })
                        regression_start = None
        
        # 개인화 효과 분석
        personalization_data = await db_service.get_personalization_data(user_id, limit=100)
        personalization_impact = {}
        
        if personalization_data:
            total_satisfaction = 0
            count = 0
            for data in personalization_data:
                feedback = data.get("feedback_data", {})
                if "satisfaction_score" in data.get("learning_weights", {}):
                    total_satisfaction += data["learning_weights"]["satisfaction_score"]
                    count += 1
            
            if count > 0:
                personalization_impact["average_satisfaction"] = total_satisfaction / count
                personalization_impact["data_points"] = count
            else:
                personalization_impact["average_satisfaction"] = 0.5
                personalization_impact["data_points"] = 0
        else:
            personalization_impact["average_satisfaction"] = 0.5
            personalization_impact["data_points"] = 0
        
        # 선호하는 여정 패턴
        completion_times = [item.get("total_duration", 0) for item in period_items if item.get("total_duration")]
        avg_duration = sum(completion_times) / len(completion_times) if completion_times else 0
        
        preferred_patterns = []
        if avg_duration < 15:
            preferred_patterns.append("Quick sessions (under 15 minutes)")
        elif avg_duration > 30:
            preferred_patterns.append("Extended sessions (over 30 minutes)")
        else:
            preferred_patterns.append("Standard sessions (15-30 minutes)")
        
        # 다음 세션 권장사항
        latest_valence = vad_evolution[-1].valence if vad_evolution else 0.0
        if latest_valence < -0.3:
            next_recommendation = "Focus on emotional regulation and self-compassion exercises"
        elif latest_valence > 0.3:
            next_recommendation = "Continue building on positive momentum with gratitude practices"
        else:
            next_recommendation = "Explore deeper emotional patterns through reflective journaling"
        
        # 위험 평가
        risk_assessment = {
            "emotional_stability": max(0.0, min(1.0, (latest_valence + 1.0) / 2.0)),
            "engagement_level": min(1.0, len(period_items) / 10.0),  # 기대치 대비
            "progress_trend": max(-1.0, min(1.0, therapeutic_progress))
        }
        
        # 개입 제안
        intervention_suggestions = []
        if risk_assessment["emotional_stability"] < 0.3:
            intervention_suggestions.append("Consider increasing session frequency")
        if risk_assessment["engagement_level"] < 0.5:
            intervention_suggestions.append("Explore different therapeutic approaches")
        if risk_assessment["progress_trend"] < -0.2:
            intervention_suggestions.append("Review and adjust treatment goals")
        
        if not intervention_suggestions:
            intervention_suggestions.append("Continue current therapeutic approach")
        
        # TherapyAnalytics 모델 생성
        analytics = TherapyAnalytics(
            user_id=user_id,
            analysis_period=days,
            emotion_trend=emotion_trend,
            vad_evolution=vad_evolution,
            dominant_emotion_changes=emotion_changes,
            therapeutic_progress=therapeutic_progress,
            breakthrough_moments=breakthrough_moments,
            regression_periods=regression_periods,
            personalization_impact=personalization_impact,
            preferred_journey_patterns=preferred_patterns,
            next_session_recommendation=next_recommendation,
            risk_assessment=risk_assessment,
            intervention_suggestions=intervention_suggestions
        )
        
        logger.info(f"치료 분석 완료: {user_id}, {days}일 기간")
        
        return APIResponse(
            success=True,
            message=f"Analytics generated for {days} days period",
            data=analytics
        )
        
    except Exception as e:
        logger.error(f"치료 분석 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate therapy analytics"
        )


@router.post(
    "/export",
    response_model=APIResponse[dict],
    summary="Export gallery data",
    description="Export user's gallery data in specified format"
)
async def export_gallery_data(
    export_request: JourneyExport,
    current_user: Annotated[dict, Depends(get_current_user)],
    db_service: Annotated[SupabaseService, Depends(lambda: db)]
) -> APIResponse[dict]:
    """갤러리 데이터 내보내기"""
    
    try:
        user_id = current_user["user_id"]
        
        # 사용자 권한 확인
        if export_request.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Can only export your own data"
            )
        
        # 갤러리 아이템들 조회
        gallery_items = await db_service.get_gallery_items(user_id)
        
        # 날짜 범위 필터링
        if export_request.date_range:
            start_date = export_request.date_range.get("start")
            end_date = export_request.date_range.get("end")
            
            if start_date or end_date:
                filtered_items = []
                for item in gallery_items:
                    item_date = datetime.fromisoformat(item["created_date"].replace("Z", "+00:00"))
                    if start_date and item_date < start_date:
                        continue
                    if end_date and item_date > end_date:
                        continue
                    filtered_items.append(item)
                gallery_items = filtered_items
        
        # 데이터 준비
        export_data = []
        for item in gallery_items:
            item_data = {
                "id": item["id"],
                "created_date": item["created_date"],
                "diary_text": item["diary_text"] if not export_request.anonymize_data else "[ANONYMIZED]",
                "emotion_keywords": item.get("emotion_keywords", []),
                "vad_scores": item.get("vad_scores", {}),
                "completion_status": item.get("completion_status"),
                "guestbook_title": item.get("guestbook_title", ""),
                "guestbook_tags": item.get("guestbook_tags", []),
                "total_duration": item.get("total_duration", 0),
                "user_rating": item.get("user_rating")
            }
            
            if export_request.include_images:
                item_data["image_url"] = item.get("image_url", "")
                item_data["image_prompt"] = item.get("image_prompt", "")
            
            if export_request.include_analytics:
                item_data["therapeutic_effectiveness"] = item.get("therapeutic_effectiveness")
                item_data["emotion_shift_analysis"] = item.get("emotion_shift_analysis")
            
            if export_request.include_metadata:
                item_data["total_cost"] = item.get("total_cost", 0)
                item_data["updated_date"] = item.get("updated_date")
            
            export_data.append(item_data)
        
        # 포맷별 처리
        if export_request.export_format == "json":
            import json
            export_content = json.dumps(export_data, indent=2, default=str)
            content_type = "application/json"
            
        elif export_request.export_format == "csv":
            import csv
            import io
            
            output = io.StringIO()
            if export_data:
                writer = csv.DictWriter(output, fieldnames=export_data[0].keys())
                writer.writeheader()
                writer.writerows(export_data)
            export_content = output.getvalue()
            content_type = "text/csv"
            
        elif export_request.export_format == "pdf":
            # PDF 생성은 복잡하므로 여기서는 간단히 처리
            export_content = f"PDF export not fully implemented. {len(export_data)} items would be exported."
            content_type = "application/pdf"
        
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported export format"
            )
        
        # 내보내기 완료
        export_info = {
            "export_format": export_request.export_format,
            "items_count": len(export_data),
            "content_type": content_type,
            "file_size": len(export_content),
            "generated_at": datetime.utcnow().isoformat(),
            "anonymized": export_request.anonymize_data
        }
        
        logger.info(f"데이터 내보내기 완료: {user_id}, {len(export_data)}개 아이템")
        
        return APIResponse(
            success=True,
            message=f"Exported {len(export_data)} items successfully",
            data=export_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"데이터 내보내기 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export gallery data"
        )