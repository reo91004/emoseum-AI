# api/dependencies.py

import os
import logging
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# from jose import JWTError, jwt
from jose import jwt
from jose.exceptions import JWTError
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorDatabase

from .database.connection import get_database
from .database.collections import Collections
from src.core.act_therapy_system import ACTTherapySystem

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))


# Global ACT Therapy System instance
act_therapy_system: Optional[ACTTherapySystem] = None


async def initialize_act_therapy_system() -> None:
    """Initialize ACT Therapy System during startup"""
    global act_therapy_system
    if act_therapy_system is None:
        logger.info("Initializing ACT Therapy System...")
        act_therapy_system = ACTTherapySystem()
        logger.info("ACT Therapy System initialized successfully")


def get_act_therapy_system() -> ACTTherapySystem:
    """Get the global ACT Therapy System instance"""
    global act_therapy_system
    if act_therapy_system is None:
        raise RuntimeError("ACT Therapy System not initialized. Call initialize_act_therapy_system() first.")
    return act_therapy_system


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    act_system: ACTTherapySystem = Depends(get_act_therapy_system)
) -> Dict[str, Any]:
    """Get current user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    # Get user from ACT system (MongoDB)
    user = act_system.user_manager.get_user(user_id)
    if user is None:
        raise credentials_exception
    
    # Convert User object to dict for API compatibility
    return {
        "user_id": user.user_id,
        "created_date": user.created_date,
        "last_updated": user.last_updated,
        "psychometric_results": [
            {
                "phq9_score": r.phq9_score,
                "cesd_score": r.cesd_score,
                "meaq_score": r.meaq_score,
                "ciss_score": r.ciss_score,
                "coping_style": r.coping_style,
                "severity_level": r.severity_level,
                "test_date": r.test_date
            } for r in user.psychometric_results
        ],
        "visual_preferences": {
            "art_style": user.visual_preferences.art_style,
            "color_tone": user.visual_preferences.color_tone,
            "complexity": user.visual_preferences.complexity,
            "brightness": user.visual_preferences.brightness,
            "saturation": user.visual_preferences.saturation,
            "style_weights": user.visual_preferences.style_weights
        }
    }


async def get_optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    act_system: ACTTherapySystem = Depends(get_act_therapy_system)
) -> Optional[Dict[str, Any]]:
    """Get current user from JWT token if provided, otherwise return None"""
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials, act_system)
    except HTTPException:
        return None


class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self, calls: int = 10, period: int = 60):
        self.calls = calls
        self.period = period
        self.records: Dict[str, list] = {}
    
    def __call__(self, user: Dict[str, Any] = Depends(get_current_user)):
        user_id = user["user_id"]
        now = datetime.utcnow()
        
        if user_id not in self.records:
            self.records[user_id] = []
        
        # Clean old records
        self.records[user_id] = [
            timestamp for timestamp in self.records[user_id]
            if now - timestamp < timedelta(seconds=self.period)
        ]
        
        if len(self.records[user_id]) >= self.calls:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        self.records[user_id].append(now)
        return user
