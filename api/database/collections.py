# api/database/collections.py

from datetime import datetime
from typing import Dict, List, Optional, Any
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
import logging

## add
from typing import TypedDict

logger = logging.getLogger(__name__)


class Collections:
    """MongoDB collections schemas and indexes"""
    
    # Collection names
    USERS = "users"
    GALLERY_ITEMS = "gallery_items"
    PERSONALIZATION_DATA = "personalization_data"
    COST_TRACKING = "cost_tracking"
    
    @classmethod
    async def create_indexes(cls, database: AsyncIOMotorDatabase) -> None:
        """Create all required indexes for collections"""
        logger.info("Creating MongoDB indexes...")
        
        # Users collection indexes
        users_collection = database[cls.USERS]
        await users_collection.create_index("user_id", unique=True)
        await users_collection.create_index("created_date")
        await users_collection.create_index("psychometric_results.coping_style")
        
        # Gallery items collection indexes
        gallery_collection = database[cls.GALLERY_ITEMS]
        await gallery_collection.create_index("user_id")
        await gallery_collection.create_index("item_id", unique=True)
        await gallery_collection.create_index("created_date")
        await gallery_collection.create_index([("user_id", 1), ("created_date", -1)])
        await gallery_collection.create_index("journey_stage")
        await gallery_collection.create_index("is_completed")
        
        # Personalization data collection indexes
        personalization_collection = database[cls.PERSONALIZATION_DATA]
        await personalization_collection.create_index("user_id", unique=True)
        await personalization_collection.create_index("training_eligibility.lora_ready")
        await personalization_collection.create_index("training_eligibility.draft_ready")
        
        # Cost tracking collection indexes
        cost_collection = database[cls.COST_TRACKING]
        await cost_collection.create_index("user_id")
        await cost_collection.create_index("api_calls.timestamp")
        await cost_collection.create_index([("user_id", 1), ("api_calls.timestamp", -1)])
        
        logger.info("MongoDB indexes created successfully")


# Collection schemas for reference
USER_SCHEMA = {
    "_id": ObjectId,
    "user_id": str,  # Unique identifier
    "created_date": datetime,
    "psychometric_results": {
        "phq9_score": int,
        "cesd_score": int,
        "meaq_score": int,
        "ciss_score": int,
        "coping_style": str,  # task_oriented, emotion_oriented, avoidance_oriented
        "severity_level": str,  # mild, moderate, severe
        "assessment_date": datetime
    },
    "visual_preferences": {
        "preferred_styles": List[str],
        "color_preferences": List[str],
        "complexity_level": str,
        "art_movements": List[str]
    },
    "personalization_level": int,  # 1, 2, or 3
    "settings": {
        "language": str,
        "notifications": bool
    }
}

GALLERY_ITEM_SCHEMA = {
    "_id": ObjectId,
    "user_id": str,
    "item_id": str,  # Unique gallery item identifier (user-timestamp format)
    "created_date": datetime,
    "diary_text": str,
    "emotion_analysis": {
        "keywords": List[str],
        "vad_scores": List[float],  # [valence, arousal, dominance]
        "primary_emotion": str,
        "intensity": float
    },
    "generated_image": {
        "image_path": str,
        "prompt_used": str,
        "generation_metadata": {
            "service_used": str,  # local_gpu, external_gpu, colab
            "generation_time": float,
            "model_version": str
        }
    },
    "guestbook_entry": {
        "title": str,
        "tags": List[str],
        "reflection": str
    },
    "curator_message": {
        "message": str,
        "message_type": str,
        "personalization_data": Dict[str, Any]
    },
    "journey_stage": str,  # the_moment, reflection, defusion, closure
    "is_completed": bool
}

class Interaction(TypedDict):
    timestamp: datetime
    interaction_type: str
    response_rating: int

PERSONALIZATION_DATA_SCHEMA = {
    "_id": ObjectId,
    "user_id": str,
    "interaction_history": List[Interaction],
    # "interaction_history": List[{
    #     "timestamp": datetime,
    #     "interaction_type": str,
    #     "response_rating": int,
    #     "engagement_score": float
    # }],
    "preference_weights": {
        "visual_elements": Dict[str, float],
        "message_tone": Dict[str, float],
        "content_style": Dict[str, float]
    },
    "training_eligibility": {
        "lora_ready": bool,
        "draft_ready": bool,
        "positive_interactions": int,
        "completed_journeys": int
    }
}

##
class ApiCall(TypedDict):
    timestamp: datetime
    service: str
    tokens_used: int
    cost: float
    request_type: str
##

COST_TRACKING_SCHEMA = {
    "_id": ObjectId,
    "user_id": str,
    "api_calls": List[ApiCall],
    # "api_calls": List[{
    #     "timestamp": datetime,
    #     "service": str,
    #     "tokens_used": int,
    #     "cost": float,
    #     "request_type": str
    # }],
    "monthly_usage": {
        "total_tokens": int,
        "total_cost": float,
        "call_count": int
    },
    "limits": {
        "monthly_token_limit": int,
        "daily_call_limit": int
    }
}