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
    db: AsyncIOMotorDatabase = Depends(get_database),
    act_system = Depends(get_act_therapy_system)
):
    """Register a new user"""
    try:
        # Check if user already exists in MongoDB
        existing_user = await db[Collections.USERS].find_one({"user_id": request.user_id})
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User already exists"
            )
        
        # Also check in ACT system (SQLite)
        existing_in_act = act_system.user_manager.get_user(request.user_id)
        if existing_in_act:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User already exists"
            )
        
        # Create user through ACT system (SQLite)
        result = act_system.onboard_new_user(request.user_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user"
            )
        
        # Also create user in MongoDB
        user_doc = {
            "user_id": request.user_id,
            "created_date": datetime.utcnow(),
            "psychometric_results": None,
            "visual_preferences": {
                "preferred_styles": [],
                "color_preferences": [],
                "complexity_level": "medium",
                "art_movements": []
            },
            "personalization_level": 1,
            "settings": {
                "language": "en",
                "notifications": True
            }
        }
        
        await db[Collections.USERS].insert_one(user_doc)
        
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
    db: AsyncIOMotorDatabase = Depends(get_database),
    act_system = Depends(get_act_therapy_system)
):
    """Login existing user"""
    try:
        # Check if user exists in MongoDB
        user = await db[Collections.USERS].find_one({"user_id": request.user_id})
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Also verify user exists in ACT system
        act_user = act_system.user_manager.get_user(request.user_id)
        if not act_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found in therapy system"
            )
        
        # Generate access token
        access_token = create_access_token(data={"sub": request.user_id})
        
        # Update last login in MongoDB
        await db[Collections.USERS].update_one(
            {"user_id": request.user_id},
            {"$set": {"last_login": datetime.utcnow()}}
        )
        
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