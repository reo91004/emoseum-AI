# api/routers/auth.py

import logging
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from ..database.connection import get_database
from ..database.collections import Collections
from ..dependencies import create_access_token, get_act_therapy_system
from ..models.user import UserRegistrationRequest, UserProfileResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=dict)
async def register(
    request: UserRegistrationRequest,
    act_system = Depends(get_act_therapy_system)
):
    """Register a new user"""
    try:
        # Check if user already exists (MongoDB only through ACT system)
        existing_user = act_system.user_manager.get_user(request.user_id)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User already exists"
            )
        
        # Create user through ACT system (MongoDB) - 생성 시 자동으로 화풍 가져옴
        result = act_system.onboard_new_user(request.user_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user"
            )
        
        # Generate access token
        access_token = create_access_token(data={"sub": request.user_id})
        
        logger.info(f"User registered successfully: {request.user_id}")
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user_id": request.user_id,
            "message": "User registered successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=dict)
async def login(
    request: UserRegistrationRequest,
    act_system = Depends(get_act_therapy_system)
):
    """Login existing user"""
    try:
        # Check if user exists (MongoDB only through ACT system)
        user = act_system.user_manager.get_user(request.user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # 로그인 시 화풍 업데이트
        act_system.user_manager.update_user_art_style(request.user_id)
        
        # Generate access token
        access_token = create_access_token(data={"sub": request.user_id})
        
        logger.info(f"User logged in successfully: {request.user_id}")
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user_id": request.user_id,
            "message": "Login successful"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.delete("/logout", response_model=dict)
async def logout():
    """Logout user (client-side token removal)"""
    return {
        "message": "Logout successful. Please remove the token from client storage."
    }


@router.post("/update-style", response_model=dict)
async def update_user_style(
    request: dict,
    act_system = Depends(get_act_therapy_system)
):
    """화풍 선호도 검사 완료 시 AI 서버 화풍 업데이트"""
    try:
        user_id = request.get("user_id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="user_id is required"
            )
        
        # 화풍 업데이트
        success = act_system.user_manager.update_user_art_style(user_id)
        
        if success:
            logger.info(f"User art style updated: {user_id}")
            return {"message": "Art style updated successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update art style"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Style update error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Style update failed"
        )