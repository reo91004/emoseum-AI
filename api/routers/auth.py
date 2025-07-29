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
        
        # Create user through ACT system (MongoDB)
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