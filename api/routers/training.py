# api/routers/training.py

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

# In-memory training status (in production, use Redis or database)
training_status: Dict[str, Dict[str, Any]] = {}


@router.get("/eligibility", response_model=TrainingEligibilityResponse)
async def check_training_eligibility(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
    act_system = Depends(get_act_therapy_system)
):
    """Check if user is eligible for advanced training"""
    try:
        # Get personalization data
        personalization = await db[Collections.PERSONALIZATION_DATA].find_one(
            {"user_id": current_user["user_id"]}
        )
        
        if not personalization:
            # Initialize personalization data if not exists
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
        
        # Check eligibility criteria
        positive_interactions = training_data.get("positive_interactions", 0)
        completed_journeys = training_data.get("completed_journeys", 0)
        
        # LoRA: 5+ positive interactions
        lora_ready = positive_interactions >= 5
        
        # DRaFT+: 10+ completed journeys
        draft_ready = completed_journeys >= 10
        
        # Generate eligibility message
        if draft_ready:
            message = "You are eligible for all advanced training options!"
            recommendation = "We recommend starting with DRaFT+ training for maximum personalization."
        elif lora_ready:
            message = "You are eligible for LoRA personalization training."
            recommendation = f"Complete {10 - completed_journeys} more journeys to unlock DRaFT+ training."
        else:
            message = "Continue your therapy journey to unlock advanced personalization."
            recommendation = f"You need {5 - positive_interactions} more positive interactions for LoRA training."
        
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
    """Start LoRA personalization training"""
    try:
        # Check eligibility
        personalization = await db[Collections.PERSONALIZATION_DATA].find_one(
            {"user_id": current_user["user_id"]}
        )
        
        if not personalization or not personalization.get("training_eligibility", {}).get("lora_ready", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not eligible for LoRA training yet"
            )
        
        # Check if training already in progress
        for tid, status in training_status.items():
            if (status["user_id"] == current_user["user_id"] and 
                status["training_type"] == TrainingType.LORA and 
                status["status"] == TrainingStatus.IN_PROGRESS):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="LoRA training already in progress"
                )
        
        # Create training session
        training_id = str(uuid.uuid4())
        started_at = datetime.utcnow()
        estimated_completion = started_at + timedelta(minutes=30)  # Estimate 30 minutes
        
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
        
        # Start training in background
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
            detail="Failed to start LoRA training"
        )


@router.post("/draft", response_model=TrainingStartResponse)
async def start_draft_training(
    request: StartTrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
    act_system = Depends(get_act_therapy_system)
):
    """Start DRaFT+ reinforcement learning"""
    try:
        # Check eligibility
        personalization = await db[Collections.PERSONALIZATION_DATA].find_one(
            {"user_id": current_user["user_id"]}
        )
        
        if not personalization or not personalization.get("training_eligibility", {}).get("draft_ready", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not eligible for DRaFT+ training yet"
            )
        
        # Check if training already in progress
        for tid, status in training_status.items():
            if (status["user_id"] == current_user["user_id"] and 
                status["training_type"] == TrainingType.DRAFT and 
                status["status"] == TrainingStatus.IN_PROGRESS):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="DRaFT+ training already in progress"
                )
        
        # Create training session
        training_id = str(uuid.uuid4())
        started_at = datetime.utcnow()
        estimated_completion = started_at + timedelta(hours=1)  # Estimate 1 hour
        
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
        
        # Start training in background
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
            detail="Failed to start DRaFT+ training"
        )


@router.get("/status/{training_id}", response_model=TrainingStatusResponse)
async def get_training_status(
    training_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get training status"""
    try:
        if training_id not in training_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Training session not found"
            )
        
        status = training_status[training_id]
        
        # Verify ownership
        if status["user_id"] != current_user["user_id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
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
            detail="Failed to get training status"
        )


# Background task simulators (in production, use actual training logic)
async def simulate_lora_training(training_id: str, user_id: str, act_system):
    """Simulate LoRA training progress"""
    try:
        logger.info(f"Starting LoRA training simulation for user {user_id}")
        training_status[training_id]["status"] = TrainingStatus.IN_PROGRESS
        
        # Simulate training steps
        for step in range(100):
            await asyncio.sleep(1)  # Simulate processing time
            
            training_status[training_id]["progress"] = {
                "current_step": step + 1,
                "total_steps": 100,
                "percentage": (step + 1) / 100 * 100,
                "estimated_time_remaining": (100 - step - 1) * 1
            }
        
        # Complete training
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
    """Simulate DRaFT+ training progress"""
    try:
        logger.info(f"Starting DRaFT+ training simulation for user {user_id}")
        training_status[training_id]["status"] = TrainingStatus.IN_PROGRESS
        
        # Simulate training steps
        for step in range(200):
            await asyncio.sleep(1)  # Simulate processing time
            
            training_status[training_id]["progress"] = {
                "current_step": step + 1,
                "total_steps": 200,
                "percentage": (step + 1) / 200 * 100,
                "estimated_time_remaining": (200 - step - 1) * 1
            }
        
        # Complete training
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