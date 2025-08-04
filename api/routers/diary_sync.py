# api/routers/diary_sync.py

import logging
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ..database.connection import get_database
from ..dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/diary-sync", tags=["Diary Sync"])


class DiarySyncRequest(BaseModel):
    diary_id: str
    ai_session_id: str
    diary_text: str
    emotion_keywords: list[str]
    vad_scores: list[float]
    reflection_image_path: str
    reflection_prompt: str


@router.post("/update-diary")
async def update_diary_from_ai(
    request: DiarySyncRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db = Depends(get_database)
):
    """AI 서버에서 처리된 일기 데이터를 MongoDB에 직접 저장"""
    try:
        # diaries 컬렉션에서 해당 일기 찾기
        diary_collection = db.diaries
        
        # 일기 업데이트
        update_data = {
            "ai_session_id": request.ai_session_id,
            "emotion_keywords": request.emotion_keywords,
            "vad_scores": request.vad_scores,
            "reflection_image_path": request.reflection_image_path,
            "reflection_prompt": request.reflection_prompt,
            "ai_processed_at": datetime.utcnow(),
            "status": "ai_processed"
        }
        
        result = await diary_collection.update_one(
            {"_id": request.diary_id, "user_id": current_user["user_id"]},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Diary not found or access denied"
            )
        
        return {
            "success": True,
            "diary_id": request.diary_id,
            "ai_session_id": request.ai_session_id,
            "updated_at": datetime.utcnow()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating diary from AI: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update diary"
        )