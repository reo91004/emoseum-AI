# tools/backup_sqlite_data.py

import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def backup_users_db(db_path: str) -> List[Dict[str, Any]]:
    """Backup users database to JSON format"""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get all users
        cursor.execute("SELECT * FROM users")
        users = [dict(row) for row in cursor.fetchall()]
        
        # Get psychometric results
        cursor.execute("SELECT * FROM psychometric_results")
        psychometric_results = [dict(row) for row in cursor.fetchall()]
        
        # Get user preferences
        cursor.execute("SELECT * FROM user_preferences")
        user_preferences = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            "users": users,
            "psychometric_results": psychometric_results,
            "user_preferences": user_preferences
        }
        
    except Exception as e:
        logger.error(f"Failed to backup users database: {e}")
        return {"users": [], "psychometric_results": [], "user_preferences": []}


def backup_gallery_db(db_path: str) -> List[Dict[str, Any]]:
    """Backup gallery database to JSON format"""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get all gallery items
        cursor.execute("SELECT * FROM gallery_items")
        gallery_items = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return {"gallery_items": gallery_items}
        
    except Exception as e:
        logger.error(f"Failed to backup gallery database: {e}")
        return {"gallery_items": []}


def backup_cost_tracking_db(db_path: str) -> List[Dict[str, Any]]:
    """Backup cost tracking database to JSON format"""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get all cost records
        cursor.execute("SELECT * FROM cost_records")
        cost_records = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return {"cost_records": cost_records}
        
    except Exception as e:
        logger.error(f"Failed to backup cost tracking database: {e}")
        return {"cost_records": []}


def main():
    """Main backup function"""
    data_dir = Path("data")
    backup_dir = Path("data/backup")
    backup_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Backup users database
    users_db_path = data_dir / "users.db"
    if users_db_path.exists():
        logger.info("Backing up users database...")
        users_data = backup_users_db(str(users_db_path))
        
        backup_file = backup_dir / f"users_backup_{timestamp}.json"
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(users_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Users database backed up to: {backup_file}")
        print(f"Users: {len(users_data.get('users', []))} records")
        print(f"Psychometric results: {len(users_data.get('psychometric_results', []))} records")
        print(f"User preferences: {len(users_data.get('user_preferences', []))} records")
    
    # Backup gallery database
    gallery_db_path = data_dir / "gallery.db"
    if gallery_db_path.exists():
        logger.info("Backing up gallery database...")
        gallery_data = backup_gallery_db(str(gallery_db_path))
        
        backup_file = backup_dir / f"gallery_backup_{timestamp}.json"
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(gallery_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Gallery database backed up to: {backup_file}")
        print(f"Gallery items: {len(gallery_data.get('gallery_items', []))} records")
    
    # Backup cost tracking database
    cost_db_path = data_dir / "cost_tracking.db"
    if cost_db_path.exists():
        logger.info("Backing up cost tracking database...")
        cost_data = backup_cost_tracking_db(str(cost_db_path))
        
        backup_file = backup_dir / f"cost_tracking_backup_{timestamp}.json"
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(cost_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Cost tracking database backed up to: {backup_file}")
        print(f"Cost records: {len(cost_data.get('cost_records', []))} records")
    
    print(f"\nAll SQLite databases backed up successfully!")
    print(f"Backup directory: {backup_dir.absolute()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()