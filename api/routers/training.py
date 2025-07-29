# api/routers/training.py

# ==============================================================================
# 이 파일은 Level 3 개인화 훈련 관련 API 엔드포인트를 정의한다.
# LoRA와 DRaFT+ 모델 훈련 자격 확인, 훈련 시작, 상태 확인 기능을 제공한다.
# ==============================================================================

import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from motor.motor_asyncio import AsyncIOMotorDatabase

from ..database.connection import get_database
from ..database.collections import Collections
from ..dependencies import get_current_user, get_act_therapy_system
from ..models.training import (
    TrainingEligibilityResponse,
    TrainingStartResponse,
    TrainingStatusResponse,
    StartTrainingRequest,
    TrainingType,
    TrainingStatus,
    TrainingEligibility,
    TrainingProgress
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training", tags=["Training"])

# 메모리 내 훈련 상태 (프로덕션에서는 Redis 또는 데이터베이스 사용)
training_status: Dict[str, Dict[str, Any]] = {}


@router.get("/eligibility", response_model=TrainingEligibilityResponse)
async def check_training_eligibility(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
    act_system = Depends(get_act_therapy_system)
):
    """사용자의 훈련 자격 확인"""
    try:
        # 개인화 데이터 조회
        personalization = await db[Collections.PERSONALIZATION_DATA].find_one(
            {"user_id": current_user["user_id"]}
        )
        
        if not personalization:
            # 개인화 데이터가 없으면 초기화
            personalization = {
                "user_id": current_user["user_id"],
                "training_eligibility": {
                    "lora_ready": False,
                    "draft_ready": False,
                    "positive_interactions": 0,
                    "completed_journeys": 0
                }
            }
        
        training_data = personalization.get("training_eligibility", {})
        
        # 자격 기준 확인
        positive_interactions = training_data.get("positive_interactions", 0)
        completed_journeys = training_data.get("completed_journeys", 0)
        
        # LoRA: 5개 이상의 긍정적 상호작용
        lora_ready = positive_interactions >= 5
        
        # DRaFT+: 10개 이상의 완료된 여정
        draft_ready = completed_journeys >= 10
        
        # 자격 메시지 생성
        if draft_ready:
            message = "모든 훈련 옵션을 사용할 수 있습니다!"
            recommendation = "최대 개인화를 위해 DRaFT+ 훈련을 시작하는 것을 권장합니다."
        elif lora_ready:
            message = "LoRA 개인화 훈련을 사용할 수 있습니다."
            recommendation = f"DRaFT+ 훈련을 사용하려면 {10 - completed_journeys}개의 여정을 더 완료하세요."
        else:
            message = "개인화를 사용하려면 치료 여정을 계속하세요."
            recommendation = f"LoRA 훈련을 위해 {5 - positive_interactions}개의 긍정적 상호작용이 더 필요합니다."
        
        eligibility = TrainingEligibility(
            lora_ready=lora_ready,
            draft_ready=draft_ready,
            positive_interactions=positive_interactions,
            completed_journeys=completed_journeys,
            eligibility_message=message
        )
        
        return TrainingEligibilityResponse(
            user_id=current_user["user_id"],
            eligibility=eligibility,
            recommendation=recommendation
        )
        
    except Exception as e:
        logger.error(f"Error checking training eligibility: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check training eligibility"
        )


@router.post("/lora", response_model=TrainingStartResponse)
async def start_lora_training(
    request: StartTrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
    act_system = Depends(get_act_therapy_system)
):
    """LoRA 개인화 훈련 시작"""
    try:
        # 자격 확인
        personalization = await db[Collections.PERSONALIZATION_DATA].find_one(
            {"user_id": current_user["user_id"]}
        )
        
        if not personalization or not personalization.get("training_eligibility", {}).get("lora_ready", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="아직 LoRA 훈련 자격이 없습니다"
            )
        
        # 훈련이 이미 진행 중인지 확인
        for tid, status in training_status.items():
            if (status["user_id"] == current_user["user_id"] and 
                status["training_type"] == TrainingType.LORA and 
                status["status"] == TrainingStatus.IN_PROGRESS):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="LoRA 훈련이 이미 진행 중입니다"
                )
        
        # 훈련 세션 생성
        training_id = str(uuid.uuid4())
        started_at = datetime.utcnow()
        estimated_completion = started_at + timedelta(minutes=30)  # 30분 예상
        
        training_status[training_id] = {
            "training_id": training_id,
            "user_id": current_user["user_id"],
            "training_type": TrainingType.LORA,
            "status": TrainingStatus.PENDING,
            "started_at": started_at,
            "estimated_completion": estimated_completion,
            "progress": {
                "current_step": 0,
                "total_steps": 100,
                "percentage": 0.0
            }
        }
        
        # 백그라운드에서 훈련 시작
        background_tasks.add_task(
            simulate_lora_training,
            training_id,
            current_user["user_id"],
            act_system
        )
        
        return TrainingStartResponse(
            training_id=training_id,
            training_type=TrainingType.LORA,
            status=TrainingStatus.PENDING,
            started_at=started_at,
            estimated_completion=estimated_completion
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting LoRA training: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LoRA 훈련 시작 실패"
        )


@router.post("/draft", response_model=TrainingStartResponse)
async def start_draft_training(
    request: StartTrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
    act_system = Depends(get_act_therapy_system)
):
    """DRaFT+ 강화학습 시작"""
    try:
        # 자격 확인
        personalization = await db[Collections.PERSONALIZATION_DATA].find_one(
            {"user_id": current_user["user_id"]}
        )
        
        if not personalization or not personalization.get("training_eligibility", {}).get("draft_ready", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="아직 DRaFT+ 훈련 자격이 없습니다"
            )
        
        # 훈련이 이미 진행 중인지 확인
        for tid, status in training_status.items():
            if (status["user_id"] == current_user["user_id"] and 
                status["training_type"] == TrainingType.DRAFT and 
                status["status"] == TrainingStatus.IN_PROGRESS):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="DRaFT+ 훈련이 이미 진행 중입니다"
                )
        
        # 훈련 세션 생성
        training_id = str(uuid.uuid4())
        started_at = datetime.utcnow()
        estimated_completion = started_at + timedelta(hours=1)  # 1시간 예상
        
        training_status[training_id] = {
            "training_id": training_id,
            "user_id": current_user["user_id"],
            "training_type": TrainingType.DRAFT,
            "status": TrainingStatus.PENDING,
            "started_at": started_at,
            "estimated_completion": estimated_completion,
            "progress": {
                "current_step": 0,
                "total_steps": 200,
                "percentage": 0.0
            }
        }
        
        # 백그라운드에서 훈련 시작
        background_tasks.add_task(
            simulate_draft_training,
            training_id,
            current_user["user_id"],
            act_system
        )
        
        return TrainingStartResponse(
            training_id=training_id,
            training_type=TrainingType.DRAFT,
            status=TrainingStatus.PENDING,
            started_at=started_at,
            estimated_completion=estimated_completion
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting DRaFT+ training: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="DRaFT+ 훈련 시작 실패"
        )


@router.get("/status/{training_id}", response_model=TrainingStatusResponse)
async def get_training_status(
    training_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """훈련 상태 조회"""
    try:
        if training_id not in training_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="훈련 세션을 찾을 수 없음"
            )
        
        status = training_status[training_id]
        
        # 소유권 확인
        if status["user_id"] != current_user["user_id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="접근 거부"
            )
        
        return TrainingStatusResponse(
            training_id=status["training_id"],
            training_type=status["training_type"],
            status=status["status"],
            progress=TrainingProgress(**status["progress"]) if status.get("progress") else None,
            started_at=status["started_at"],
            completed_at=status.get("completed_at"),
            error_message=status.get("error_message"),
            result_metrics=status.get("result_metrics")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="훈련 상태 조회 실패"
        )


# 백그라운드 태스크 시뮬레이터 (프로덕션에서는 실제 훈련 로직 사용)
async def simulate_lora_training(training_id: str, user_id: str, act_system):
    """LoRA 훈련 진행 시뮬레이션"""
    try:
        logger.info(f"Starting LoRA training simulation for user {user_id}")
        training_status[training_id]["status"] = TrainingStatus.IN_PROGRESS
        
        # 훈련 단계 시뮬레이션
        for step in range(100):
            await asyncio.sleep(1)  # 처리 시간 시뮬레이션
            
            training_status[training_id]["progress"] = {
                "current_step": step + 1,
                "total_steps": 100,
                "percentage": (step + 1) / 100 * 100,
                "estimated_time_remaining": (100 - step - 1) * 1
            }
        
        # 훈련 완료
        training_status[training_id]["status"] = TrainingStatus.COMPLETED
        training_status[training_id]["completed_at"] = datetime.utcnow()
        training_status[training_id]["result_metrics"] = {
            "model_improvement": 0.15,
            "personalization_score": 0.82,
            "training_loss": 0.023
        }
        
        logger.info(f"LoRA training completed for user {user_id}")
        
    except Exception as e:
        logger.error(f"LoRA training failed: {e}")
        training_status[training_id]["status"] = TrainingStatus.FAILED
        training_status[training_id]["error_message"] = str(e)


async def simulate_draft_training(training_id: str, user_id: str, act_system):
    """DRaFT+ 훈련 진행 시뮬레이션"""
    try:
        logger.info(f"Starting DRaFT+ training simulation for user {user_id}")
        training_status[training_id]["status"] = TrainingStatus.IN_PROGRESS
        
        # 훈련 단계 시뮬레이션
        for step in range(200):
            await asyncio.sleep(1)  # 처리 시간 시뮬레이션
            
            training_status[training_id]["progress"] = {
                "current_step": step + 1,
                "total_steps": 200,
                "percentage": (step + 1) / 200 * 100,
                "estimated_time_remaining": (200 - step - 1) * 1
            }
        
        # 훈련 완료
        training_status[training_id]["status"] = TrainingStatus.COMPLETED
        training_status[training_id]["completed_at"] = datetime.utcnow()
        training_status[training_id]["result_metrics"] = {
            "policy_improvement": 0.23,
            "reward_optimization": 0.89,
            "convergence_score": 0.91
        }
        
        logger.info(f"DRaFT+ training completed for user {user_id}")
        
    except Exception as e:
        logger.error(f"DRaFT+ training failed: {e}")
        training_status[training_id]["status"] = TrainingStatus.FAILED
        training_status[training_id]["error_message"] = str(e)


# Import asyncio for background tasks
import asyncio