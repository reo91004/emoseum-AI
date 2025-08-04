# api/routers/gallery.py

# ==============================================================================
# 이 파일은 갤러리 관련 API 엔드포인트를 정의한다.
# 사용자의 감정 여정 기록을 조회, 필터링, 분석, 내보내기하는 기능을 제공한다.
# ==============================================================================

import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import StreamingResponse
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId
import tempfile
import io

from ..database.connection import get_database
from ..database.collections import Collections
from ..dependencies import get_current_user
from ..models.gallery import (
    GalleryListResponse,
    GalleryItemResponse,
    GalleryAnalyticsResponse,
    GalleryExportResponse,
    GalleryFilterRequest,
    GalleryItemSummary,
    GalleryItemDetail,
    GalleryAnalytics,
    EmotionTrend,
    EmotionAnalysis,
    GeneratedImage,
    GuestbookEntry,
    DocentMessage
)
from ..models.therapy import JourneyStage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/gallery", tags=["Gallery"])


@router.get("/items", response_model=GalleryListResponse)
async def get_gallery_items(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    emotions: Optional[List[str]] = Query(None),
    completed_only: bool = Query(False),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """갤러리 아이템 목록 조회"""
    try:
        # 쿼리 구성
        query = {"user_id": current_user["user_id"]}
        
        if start_date or end_date:
            date_query = {}
            if start_date:
                date_query["$gte"] = start_date
            if end_date:
                date_query["$lte"] = end_date
            query["created_date"] = date_query
        
        if emotions:
            query["emotion_analysis.primary_emotion"] = {"$in": emotions}
        
        if completed_only:
            query["docent_message"] = {"$exists": True, "$ne": {}}
        
        # 전체 개수 조회
        total_count = await db[Collections.GALLERY_ITEMS].count_documents(query)
        
        # Get items
        cursor = db[Collections.GALLERY_ITEMS].find(query).sort("created_date", -1).skip(offset).limit(limit)
        items = []
        
        async for item in cursor:
            # 썸네일 URL 생성
            thumbnail_url = None
            if item.get("generated_image", {}).get("image_path"):
                filename = os.path.basename(item["generated_image"]["image_path"])
                thumbnail_url = f"/therapy/images/{filename}"
            
            summary = GalleryItemSummary(
                item_id=item.get("item_id", str(item["_id"])),
                session_id=item.get("item_id", str(item["_id"])),  # API compatibility
                created_date=item["created_date"],
                thumbnail_url=thumbnail_url,
                primary_emotion=item.get("emotion_analysis", {}).get("primary_emotion", "neutral"),
                is_completed=bool(item.get("docent_message"))
            )
            items.append(summary)
        
        return GalleryListResponse(
            items=items,
            total_count=total_count,
            has_more=(offset + limit) < total_count
        )
        
    except Exception as e:
        logger.error(f"Error getting gallery items: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get gallery items"
        )


@router.get("/items/{item_id}", response_model=GalleryItemResponse)
async def get_gallery_item(
    item_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """특정 갤러리 아이템 상세 조회"""
    try:
        # item_id로 먼저 찾기 (선호 방법)
        item = await db[Collections.GALLERY_ITEMS].find_one({
            "item_id": item_id,
            "user_id": current_user["user_id"]
        })
        
        # item_id로 찾지 못하면 ObjectId로 시도
        if not item:
            try:
                obj_id = ObjectId(item_id)
                item = await db[Collections.GALLERY_ITEMS].find_one({
                    "_id": obj_id,
                    "user_id": current_user["user_id"]
                })
            except:
                pass
        
        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Gallery item not found"
            )
        
        # 상세 응답 구성
        detail = GalleryItemDetail(
            item_id=item.get("item_id", str(item["_id"])),
            session_id=item.get("item_id", str(item["_id"])),  # API compatibility
            user_id=item["user_id"],
            created_date=item["created_date"],
            diary_text=item.get("diary_text", ""),
            emotion_analysis=EmotionAnalysis(**item["emotion_analysis"]) if item.get("emotion_analysis") else None,
            generated_image=GeneratedImage(**item["generated_image"]) if item.get("generated_image") else None,
            guestbook_entry=GuestbookEntry(**item["guestbook_entry"]) if item.get("guestbook_entry") else None,
            docent_message=DocentMessage(**item["docent_message"]) if item.get("docent_message") else None,
            journey_stage=item["journey_stage"],
            is_completed=item["is_completed"]
        )
        
        return GalleryItemResponse(item=detail)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting gallery item: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get gallery item"
        )


@router.get("/analytics", response_model=GalleryAnalyticsResponse)
async def get_gallery_analytics(
    days: int = Query(30, ge=1, le=365),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """갤러리 아이템의 감정 분석 데이터 조회"""
    try:
        # 날짜 범위 계산
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # 날짜 범위 내 아이템 조회
        query = {
            "user_id": current_user["user_id"],
            "created_date": {"$gte": start_date, "$lte": end_date}
        }
        
        cursor = db[Collections.GALLERY_ITEMS].find(query).sort("created_date", 1)
        
        # 데이터 분석
        total_items = 0
        completed_journeys = 0
        emotion_counts = {}
        vad_sum = {"valence": 0, "arousal": 0, "dominance": 0}
        emotion_trends = []
        
        async for item in cursor:
            total_items += 1
            if item["is_completed"]:
                completed_journeys += 1
            
            # 감정 분석
            if item.get("emotion_analysis"):
                emotion = item["emotion_analysis"].get("primary_emotion", "neutral")
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                
                vad = item["emotion_analysis"].get("vad_scores", [0, 0, 0])
                vad_sum["valence"] += vad[0]
                vad_sum["arousal"] += vad[1]
                vad_sum["dominance"] += vad[2]
                
                # 트렌드에 추가
                trend = EmotionTrend(
                    date=item["created_date"],
                    valence=vad[0],
                    arousal=vad[1],
                    dominance=vad[2],
                    primary_emotion=emotion
                )
                emotion_trends.append(trend)
        
        # 평균 계산
        if total_items > 0:
            avg_vad = {
                "valence": vad_sum["valence"] / total_items,
                "arousal": vad_sum["arousal"] / total_items,
                "dominance": vad_sum["dominance"] / total_items
            }
        else:
            avg_vad = {"valence": 0, "arousal": 0, "dominance": 0}
        
        analytics = GalleryAnalytics(
            total_items=total_items,
            completed_journeys=completed_journeys,
            emotion_trends=emotion_trends,
            most_common_emotions=emotion_counts,
            average_vad_scores=avg_vad
        )
        
        return GalleryAnalyticsResponse(
            analytics=analytics,
            period_start=start_date,
            period_end=end_date
        )
        
    except Exception as e:
        logger.error(f"Error getting gallery analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get gallery analytics"
        )


@router.get("/export", response_model=GalleryExportResponse)
async def export_gallery_data(
    format: str = Query("json", regex="^(json|csv)$"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """갤러리 데이터 내보내기"""
    try:
        # 사용자의 모든 갤러리 아이템 조회
        cursor = db[Collections.GALLERY_ITEMS].find(
            {"user_id": current_user["user_id"]}
        ).sort("created_date", -1)
        
        items = []
        async for item in cursor:
            # MongoDB 특정 필드 제거
            item.pop("_id", None)
            items.append(item)
        
        # 형식에 따른 내보내기 생성
        if format == "json":
            export_data = {
                "user_id": current_user["user_id"],
                "export_date": datetime.utcnow().isoformat(),
                "total_items": len(items),
                "items": items
            }
            
            # 파일 생성
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(export_data, f, indent=2, default=str)
                temp_path = f.name
            
            # 다운로드 URL 생성
            export_url = f"/gallery/download/{os.path.basename(temp_path)}"
            
        else:  # CSV 형식
            # CSV 형식으로 변환
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="CSV export not yet implemented"
            )
        
        return GalleryExportResponse(
            export_url=export_url,
            expires_at=datetime.utcnow() + timedelta(hours=24),
            file_format=format
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting gallery data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export gallery data"
        )


@router.get("/download/{filename}")
async def download_export(
    filename: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """내보낸 갤러리 데이터 다운로드"""
    try:
        # 프로덕션에서는 소유권 검증 및 S3/스토리지에서 제공
        file_path = os.path.join(tempfile.gettempdir(), filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Export file not found or expired"
            )
        
        # 파일 읽기
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # 정리
        os.unlink(file_path)
        
        return StreamingResponse(
            io.BytesIO(content),
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename=emoseum_export_{current_user['user_id']}.json"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading export: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download export"
        )