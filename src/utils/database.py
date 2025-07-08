"""
Emoseum 감정 치료 시스템을 위한 데이터베이스 유틸리티 함수.

이 모듈은 다음을 포함한 모든 데이터베이스 관련 기능을 포함한다:
- SQLite 데이터베이스 초기화
- 사용자 프로필 관리
- 감정 기록 추적
- 피드백 기록 관리
- LoRA 어댑터 저장/복원
"""

import sqlite3
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import torch

# Setup logging
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database management utility class for Emoseum system"""

    def __init__(self, db_path: str = "data/user_profiles.db"):
        """
        Initialize database manager.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Emotion history table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS emotion_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    input_text TEXT,
                    valence REAL,
                    arousal REAL,
                    dominance REAL,
                    confidence REAL,
                    generated_prompt TEXT,
                    image_path TEXT
                )
                """
            )

            # Feedback history table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    emotion_id INTEGER,
                    timestamp TEXT NOT NULL,
                    feedback_score REAL,
                    feedback_type TEXT,
                    comments TEXT,
                    FOREIGN KEY (emotion_id) REFERENCES emotion_history (id)
                )
                """
            )

            # User profiles table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    preference_weights TEXT,
                    therapeutic_progress TEXT,
                    learning_metadata TEXT,
                    last_updated TEXT
                )
                """
            )

            conn.commit()
            logger.info("✅ Database initialized successfully")

        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise
        finally:
            conn.close()

    def load_user_profile(
        self, user_id: str
    ) -> Tuple[List[Dict], List[Dict], Dict, Dict, Dict]:
        """
        Load user profile data from database.

        Args:
            user_id: User identifier

        Returns:
            Tuple of (emotion_history, feedback_history, preference_weights,
                     therapeutic_progress, learning_metadata)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        emotion_history = []
        feedback_history = []
        preference_weights = {}
        therapeutic_progress = {}
        learning_metadata = {}

        try:
            # Load emotion history (last 50 records)
            cursor.execute(
                """
                SELECT * FROM emotion_history 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 50
                """,
                (user_id,),
            )

            emotion_rows = cursor.fetchall()
            for row in emotion_rows:
                emotion_history.append(
                    {
                        "id": row[0],
                        "timestamp": row[2],
                        "input_text": row[3],
                        "valence": row[4],
                        "arousal": row[5],
                        "dominance": row[6],
                        "confidence": row[7],
                        "generated_prompt": row[8],
                        "image_path": row[9],
                    }
                )

            # Load feedback history
            cursor.execute(
                """
                SELECT * FROM feedback_history 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 50
                """,
                (user_id,),
            )

            feedback_rows = cursor.fetchall()
            for row in feedback_rows:
                feedback_history.append(
                    {
                        "id": row[0],
                        "emotion_id": row[2],
                        "timestamp": row[3],
                        "feedback_score": row[4],
                        "feedback_type": row[5],
                        "comments": row[6],
                    }
                )

            # Load user profile settings
            cursor.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,))
            profile_row = cursor.fetchone()

            if profile_row:
                preference_weights = json.loads(profile_row[1])
                therapeutic_progress = json.loads(profile_row[2])
                learning_metadata = json.loads(profile_row[3])

            logger.info(
                f"✅ User {user_id} profile loaded: "
                f"emotions {len(emotion_history)}, feedback {len(feedback_history)}"
            )

        except Exception as e:
            logger.error(f"❌ Profile loading failed: {e}")
            raise
        finally:
            conn.close()

        return (
            emotion_history,
            feedback_history,
            preference_weights,
            therapeutic_progress,
            learning_metadata,
        )

    def save_user_profile(
        self,
        user_id: str,
        preference_weights: Dict,
        therapeutic_progress: Dict,
        learning_metadata: Dict,
    ):
        """
        Save user profile data to database.

        Args:
            user_id: User identifier
            preference_weights: User preference weights
            therapeutic_progress: Therapeutic progress metrics
            learning_metadata: Learning metadata
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            timestamp = datetime.now().isoformat()
            cursor.execute(
                """
                INSERT OR REPLACE INTO user_profiles 
                (user_id, preference_weights, therapeutic_progress, learning_metadata, last_updated)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    json.dumps(preference_weights),
                    json.dumps(therapeutic_progress),
                    json.dumps(learning_metadata),
                    timestamp,
                ),
            )

            conn.commit()
            logger.info(f"✅ User {user_id} profile saved successfully")

        except Exception as e:
            logger.error(f"❌ Profile saving failed: {e}")
            raise
        finally:
            conn.close()

    def add_emotion_record(
        self,
        user_id: str,
        input_text: str,
        valence: float,
        arousal: float,
        dominance: float,
        confidence: float,
        generated_prompt: str,
        image_path: str = None,
    ) -> int:
        """
        Add emotion record to database.

        Args:
            user_id: User identifier
            input_text: Original input text
            valence: Emotion valence (-1.0 to 1.0)
            arousal: Emotion arousal (-1.0 to 1.0)
            dominance: Emotion dominance (-1.0 to 1.0)
            confidence: Prediction confidence (0.0 to 1.0)
            generated_prompt: Generated image prompt
            image_path: Path to generated image

        Returns:
            ID of the inserted emotion record
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            timestamp = datetime.now().isoformat()
            cursor.execute(
                """
                INSERT INTO emotion_history 
                (user_id, timestamp, input_text, valence, arousal, dominance, confidence, generated_prompt, image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    timestamp,
                    input_text,
                    valence,
                    arousal,
                    dominance,
                    confidence,
                    generated_prompt,
                    image_path,
                ),
            )

            emotion_id = cursor.lastrowid
            conn.commit()
            logger.info(f"✅ Emotion record added: ID {emotion_id}")
            return emotion_id

        except Exception as e:
            logger.error(f"❌ Emotion record addition failed: {e}")
            return -1
        finally:
            conn.close()

    def add_feedback_record(
        self,
        user_id: str,
        emotion_id: int,
        feedback_score: float,
        feedback_type: str = "rating",
        comments: str = None,
    ) -> bool:
        """
        Add feedback record to database.

        Args:
            user_id: User identifier
            emotion_id: Related emotion record ID
            feedback_score: Feedback score (typically 1-5)
            feedback_type: Type of feedback (rating, comment, etc.)
            comments: Optional feedback comments

        Returns:
            True if successful, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            timestamp = datetime.now().isoformat()
            cursor.execute(
                """
                INSERT INTO feedback_history 
                (user_id, emotion_id, timestamp, feedback_score, feedback_type, comments)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    emotion_id,
                    timestamp,
                    feedback_score,
                    feedback_type,
                    comments,
                ),
            )

            conn.commit()
            logger.info(
                f"✅ Feedback added: emotion ID {emotion_id}, score {feedback_score}"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Feedback addition failed: {e}")
            return False
        finally:
            conn.close()

    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """
        Get user statistics from database.

        Args:
            user_id: User identifier

        Returns:
            Dictionary containing user statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Get emotion count
            cursor.execute(
                "SELECT COUNT(*) FROM emotion_history WHERE user_id = ?", (user_id,)
            )
            emotion_count = cursor.fetchone()[0]

            # Get feedback count
            cursor.execute(
                "SELECT COUNT(*) FROM feedback_history WHERE user_id = ?", (user_id,)
            )
            feedback_count = cursor.fetchone()[0]

            # Get average feedback score
            cursor.execute(
                "SELECT AVG(feedback_score) FROM feedback_history WHERE user_id = ?",
                (user_id,),
            )
            avg_feedback = cursor.fetchone()[0] or 0.0

            # Get last activity
            cursor.execute(
                "SELECT MAX(timestamp) FROM emotion_history WHERE user_id = ?",
                (user_id,),
            )
            last_activity = cursor.fetchone()[0]

            return {
                "emotion_count": emotion_count,
                "feedback_count": feedback_count,
                "average_feedback": avg_feedback,
                "last_activity": last_activity,
            }

        except Exception as e:
            logger.error(f"❌ Statistics retrieval failed: {e}")
            return {}
        finally:
            conn.close()


class LoRAManager:
    """LoRA adapter persistence manager"""

    def __init__(self, lora_dir: str = "data/user_loras", device: str = "cpu"):
        """
        Initialize LoRA manager.

        Args:
            lora_dir: Directory to store LoRA files
            device: Device for tensor operations
        """
        self.lora_dir = Path(lora_dir)
        self.lora_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

    def save_user_lora(self, user_id: str, model_state_dict: Dict[str, torch.Tensor]):
        """
        Save user LoRA adapter to disk.

        Args:
            user_id: User identifier
            model_state_dict: Model state dictionary to save
        """
        try:
            user_lora_path = self.lora_dir / f"{user_id}_lora.pt"
            torch.save(model_state_dict, user_lora_path)
            logger.info(f"✅ User {user_id} LoRA saved: {user_lora_path}")
        except Exception as e:
            logger.error(f"❌ LoRA saving failed: {e}")
            raise

    def load_user_lora(self, user_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Load user LoRA adapter from disk.

        Args:
            user_id: User identifier

        Returns:
            Model state dictionary if found, None otherwise
        """
        try:
            user_lora_path = self.lora_dir / f"{user_id}_lora.pt"
            if user_lora_path.exists():
                state_dict = torch.load(user_lora_path, map_location=self.device)
                logger.info(f"✅ User {user_id} LoRA loaded: {user_lora_path}")
                return state_dict
        except Exception as e:
            logger.error(f"❌ LoRA loading failed: {e}")
        return None

    def user_lora_exists(self, user_id: str) -> bool:
        """
        Check if user LoRA adapter exists.

        Args:
            user_id: User identifier

        Returns:
            True if LoRA exists, False otherwise
        """
        user_lora_path = self.lora_dir / f"{user_id}_lora.pt"
        return user_lora_path.exists()

    def delete_user_lora(self, user_id: str) -> bool:
        """
        Delete user LoRA adapter.

        Args:
            user_id: User identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            user_lora_path = self.lora_dir / f"{user_id}_lora.pt"
            if user_lora_path.exists():
                user_lora_path.unlink()
                logger.info(f"✅ User {user_id} LoRA deleted")
                return True
        except Exception as e:
            logger.error(f"❌ LoRA deletion failed: {e}")
        return False

    def get_user_lora_info(self, user_id: str) -> Dict[str, Any]:
        """
        Get user LoRA adapter information.

        Args:
            user_id: User identifier

        Returns:
            Dictionary containing LoRA information
        """
        user_lora_path = self.lora_dir / f"{user_id}_lora.pt"

        info = {
            "user_id": user_id,
            "path": str(user_lora_path),
            "exists": user_lora_path.exists(),
            "size": 0,
            "modified": None,
        }

        if user_lora_path.exists():
            try:
                stat = user_lora_path.stat()
                info["size"] = stat.st_size
                info["modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
            except Exception as e:
                logger.error(f"❌ Failed to get LoRA info: {e}")

        return info

    def list_user_loras(self) -> List[str]:
        """
        List all available user LoRA adapters.

        Returns:
            List of user IDs with LoRA adapters
        """
        try:
            user_ids = []
            for lora_file in self.lora_dir.glob("*_lora.pt"):
                user_id = lora_file.stem.replace("_lora", "")
                user_ids.append(user_id)
            return user_ids
        except Exception as e:
            logger.error(f"❌ Failed to list LoRAs: {e}")
            return []


def create_database_backup(db_path: str, backup_path: str) -> bool:
    """
    Create a backup of the database.

    Args:
        db_path: Source database path
        backup_path: Destination backup path

    Returns:
        True if successful, False otherwise
    """
    try:
        import shutil

        shutil.copy2(db_path, backup_path)
        logger.info(f"✅ Database backup created: {backup_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Database backup failed: {e}")
        return False


def cleanup_old_records(db_path: str, days_to_keep: int = 30) -> bool:
    """
    Clean up old records from database.

    Args:
        db_path: Database path
        days_to_keep: Number of days to keep records

    Returns:
        True if successful, False otherwise
    """
    try:
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cutoff_str = cutoff_date.isoformat()

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Delete old emotion records
        cursor.execute("DELETE FROM emotion_history WHERE timestamp < ?", (cutoff_str,))
        emotion_deleted = cursor.rowcount

        # Delete old feedback records
        cursor.execute(
            "DELETE FROM feedback_history WHERE timestamp < ?", (cutoff_str,)
        )
        feedback_deleted = cursor.rowcount

        conn.commit()
        conn.close()

        logger.info(
            f"✅ Cleanup completed: {emotion_deleted} emotion records, "
            f"{feedback_deleted} feedback records deleted"
        )
        return True

    except Exception as e:
        logger.error(f"❌ Database cleanup failed: {e}")
        return False
