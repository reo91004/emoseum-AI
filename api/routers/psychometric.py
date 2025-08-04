# api/routers/psychometric.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from datetime import datetime
import logging

from ..database.connection import get_database
from ..database.collections import Collections

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/psychometric", tags=["psychometric"])

class PHQ9Request(BaseModel):
    user_id: str
    phq9_score: int
    scores: List[int]
    test_date: datetime

class CESDRequest(BaseModel):
    user_id: str
    cesd_score: int
    scores: List[int]
    test_date: datetime

@router.post("/phq9")
async def update_phq9_score(request: PHQ9Request):
    """PHQ-9 검사 결과 업데이트"""
    try:
        database = await get_database()
        users_collection = database[Collections.USERS]
        
        # 사용자 찾기
        user = await users_collection.find_one({"user_id": request.user_id})
        if not user:
            return {"success": False, "message": "User not found"}
        
        # psychometric_results 객체 업데이트
        existing_results = user.get("psychometric_results", {}) if isinstance(user.get("psychometric_results"), dict) else {}
        cesd_score = existing_results.get("cesd_score", 0)
        
        # coping_style 계산
        if request.phq9_score >= 15 or cesd_score >= 20:
            coping_style = "avoidant"
        elif request.phq9_score <= 5 and cesd_score <= 10:
            coping_style = "confrontational"
        else:
            coping_style = "balanced"
            
        # severity_level 계산
        if request.phq9_score >= 15 or cesd_score >= 24:
            severity_level = "severe"
        elif request.phq9_score >= 10 or cesd_score >= 16:
            severity_level = "moderate"
        else:
            severity_level = "mild"
        
        await users_collection.update_one(
            {"user_id": request.user_id},
            {
                "$set": {
                    "psychometric_results": {
                        "phq9_score": request.phq9_score,
                        "cesd_score": cesd_score,
                        "meaq_score": existing_results.get("meaq_score", 0),
                        "ciss_score": existing_results.get("ciss_score", 0),
                        "coping_style": coping_style,
                        "severity_level": severity_level,
                        "assessment_date": request.test_date
                    }
                }
            }
        )
        
        logger.info(f"PHQ-9 점수 업데이트 완료: {request.user_id} - {request.phq9_score}")
        return {"success": True, "message": "PHQ-9 score updated"}
        
    except Exception as e:
        logger.error(f"PHQ-9 업데이트 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cesd")
async def update_cesd_score(request: CESDRequest):
    """CES-D 검사 결과 업데이트"""
    try:
        database = await get_database()
        users_collection = database[Collections.USERS]
        
        # 사용자 찾기
        user = await users_collection.find_one({"user_id": request.user_id})
        if not user:
            return {"success": False, "message": "User not found"}
        
        # psychometric_results 객체 업데이트
        existing_results = user.get("psychometric_results", {}) if isinstance(user.get("psychometric_results"), dict) else {}
        phq9_score = existing_results.get("phq9_score", 0)
        
        # coping_style 계산
        if phq9_score >= 15 or request.cesd_score >= 20:
            coping_style = "avoidant"
        elif phq9_score <= 5 and request.cesd_score <= 10:
            coping_style = "confrontational"
        else:
            coping_style = "balanced"
            
        # severity_level 계산
        if phq9_score >= 15 or request.cesd_score >= 24:
            severity_level = "severe"
        elif phq9_score >= 10 or request.cesd_score >= 16:
            severity_level = "moderate"
        else:
            severity_level = "mild"
        
        await users_collection.update_one(
            {"user_id": request.user_id},
            {
                "$set": {
                    "psychometric_results": {
                        "phq9_score": phq9_score,
                        "cesd_score": request.cesd_score,
                        "meaq_score": existing_results.get("meaq_score", 0),
                        "ciss_score": existing_results.get("ciss_score", 0),
                        "coping_style": coping_style,
                        "severity_level": severity_level,
                        "assessment_date": request.test_date
                    }
                }
            }
        )
        
        logger.info(f"CES-D 점수 업데이트 완료: {request.user_id} - {request.cesd_score}")
        return {"success": True, "message": "CES-D score updated"}
        
    except Exception as e:
        logger.error(f"CES-D 업데이트 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))