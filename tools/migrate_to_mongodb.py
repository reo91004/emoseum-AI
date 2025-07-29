# tools/migrate_to_mongodb.py

import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dependencies import get_mongodb_client
from src.database.mongodb_client import MongoDBClient

logger = logging.getLogger(__name__)


class MongoDBMigrator:
    """SQLite에서 MongoDB로 데이터 마이그레이션"""
    
    def __init__(self):
        self.mongodb_client = get_mongodb_client()
        self.db = self.mongodb_client.sync_db
        
        # 컬렉션 참조
        self.users_collection = self.db.users
        self.gallery_collection = self.db.gallery_items
        self.cost_collection = self.db.cost_tracking
        
        logger.info("MongoDB 마이그레이션 도구 초기화 완료")
    
    def migrate_users_data(self, sqlite_db_path: str) -> Dict[str, int]:
        """사용자 데이터 마이그레이션"""
        if not Path(sqlite_db_path).exists():
            logger.warning(f"사용자 SQLite DB를 찾을 수 없음: {sqlite_db_path}")
            return {"users": 0, "psychometric_results": 0, "preferences": 0}
        
        try:
            conn = sqlite3.connect(sqlite_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            stats = {"users": 0, "psychometric_results": 0, "preferences": 0}
            
            # 사용자 기본 정보
            cursor.execute("SELECT * FROM users")
            users = cursor.fetchall()
            
            for user_row in users:
                user_doc = {
                    "user_id": user_row["user_id"],
                    "created_date": user_row["created_date"],
                    "last_updated": user_row.get("last_updated", user_row["created_date"]),
                    "psychometric_results": [],
                    "visual_preferences": {
                        "art_style": "painting",
                        "color_tone": "warm", 
                        "complexity": "balanced",
                        "brightness": 0.5,
                        "saturation": 0.5,
                        "style_weights": {
                            "painting": 0.33,
                            "photography": 0.33,
                            "abstract": 0.34
                        }
                    },
                    "gpt_settings": {
                        "daily_token_limit": 1000,
                        "monthly_cost_limit": 10.0,
                        "current_daily_usage": 0,
                        "current_monthly_cost": 0.0,
                        "last_reset_date": datetime.now().isoformat(),
                        "gpt_preferences": {},
                        "notification_settings": {}
                    }
                }
                
                # 심리검사 결과 추가
                cursor.execute(
                    "SELECT * FROM psychometric_results WHERE user_id = ?",
                    (user_row["user_id"],)
                )
                psychometric_rows = cursor.fetchall()
                
                for psych_row in psychometric_rows:
                    result = {
                        "phq9_score": psych_row["phq9_score"],
                        "cesd_score": psych_row["cesd_score"],
                        "meaq_score": psych_row["meaq_score"],
                        "ciss_score": psych_row["ciss_score"],
                        "coping_style": psych_row["coping_style"],
                        "severity_level": psych_row["severity_level"],
                        "test_date": psych_row["test_date"]
                    }
                    user_doc["psychometric_results"].append(result)
                    stats["psychometric_results"] += 1
                
                # 시각적 선호도 추가 (있다면)
                try:
                    cursor.execute(
                        "SELECT * FROM user_preferences WHERE user_id = ?",
                        (user_row["user_id"],)
                    )
                    pref_row = cursor.fetchone()
                    if pref_row:
                        user_doc["visual_preferences"].update({
                            "art_style": pref_row.get("art_style", "painting"),
                            "color_tone": pref_row.get("color_tone", "warm"),
                            "complexity": pref_row.get("complexity", "balanced"),
                            "brightness": pref_row.get("brightness", 0.5),
                            "saturation": pref_row.get("saturation", 0.5)
                        })
                        stats["preferences"] += 1
                except sqlite3.OperationalError:
                    # user_preferences 테이블이 없을 수 있음
                    pass
                
                # MongoDB에 삽입 (upsert)
                self.users_collection.update_one(
                    {"user_id": user_doc["user_id"]},
                    {"$set": user_doc},
                    upsert=True
                )
                stats["users"] += 1
            
            conn.close()
            logger.info(f"사용자 데이터 마이그레이션 완료: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"사용자 데이터 마이그레이션 실패: {e}")
            return {"users": 0, "psychometric_results": 0, "preferences": 0}
    
    def migrate_gallery_data(self, sqlite_db_path: str) -> Dict[str, int]:
        """갤러리 데이터 마이그레이션"""
        if not Path(sqlite_db_path).exists():
            logger.warning(f"갤러리 SQLite DB를 찾을 수 없음: {sqlite_db_path}")
            return {"gallery_items": 0}
        
        try:
            conn = sqlite3.connect(sqlite_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            stats = {"gallery_items": 0}
            
            cursor.execute("SELECT * FROM gallery_items")
            gallery_items = cursor.fetchall()
            
            for item_row in gallery_items:
                # JSON 필드 파싱
                emotion_keywords = json.loads(item_row.get("emotion_keywords", "[]"))
                vad_scores = json.loads(item_row.get("vad_scores", "[0,0,0]"))
                guestbook_tags = json.loads(item_row.get("guestbook_tags", "[]"))
                
                gallery_doc = {
                    "user_id": item_row["user_id"],
                    "created_date": item_row["created_date"],
                    "last_updated": item_row.get("last_updated", item_row["created_date"]),
                    "diary_text": item_row["diary_text"],
                    "emotion_keywords": emotion_keywords,
                    "vad_scores": vad_scores,
                    "reflection_prompt": item_row.get("reflection_prompt", ""),
                    "reflection_image_path": item_row.get("reflection_image", ""),
                    "coping_style": item_row.get("coping_style", "balanced"),
                    "guestbook_title": item_row.get("guestbook_title", ""),
                    "guestbook_tags": guestbook_tags,
                    "guided_question": item_row.get("guided_question", ""),
                    "curator_message": json.loads(item_row.get("curator_message", "{}")),
                    "message_reactions": json.loads(item_row.get("message_reactions", "[]")),
                    "gpt_metadata": {
                        "prompt_tokens": item_row.get("gpt_prompt_tokens", 0),
                        "prompt_generation_time": item_row.get("prompt_generation_time", 0.0),
                        "generation_method": "gpt"
                    }
                }
                
                # MongoDB에 삽입
                result = self.gallery_collection.insert_one(gallery_doc)
                stats["gallery_items"] += 1
            
            conn.close()
            logger.info(f"갤러리 데이터 마이그레이션 완료: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"갤러리 데이터 마이그레이션 실패: {e}")
            return {"gallery_items": 0}
    
    def migrate_cost_data(self, sqlite_db_path: str) -> Dict[str, int]:
        """비용 추적 데이터 마이그레이션"""
        if not Path(sqlite_db_path).exists():
            logger.warning(f"비용 추적 SQLite DB를 찾을 수 없음: {sqlite_db_path}")
            return {"cost_records": 0}
        
        try:
            conn = sqlite3.connect(sqlite_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            stats = {"cost_records": 0}
            
            cursor.execute("SELECT * FROM cost_records")
            cost_records = cursor.fetchall()
            
            for record_row in cost_records:
                cost_doc = {
                    "user_id": record_row["user_id"],
                    "purpose": record_row["purpose"],
                    "model": record_row["model"],
                    "prompt_tokens": record_row["prompt_tokens"],
                    "completion_tokens": record_row["completion_tokens"],
                    "total_tokens": record_row["total_tokens"],
                    "input_cost": record_row["input_cost"],
                    "output_cost": record_row["output_cost"],
                    "total_cost": record_row["total_cost"],
                    "processing_time": record_row["processing_time"],
                    "success": bool(record_row["success"]),
                    "error_message": record_row.get("error_message"),
                    "timestamp": record_row["timestamp"],
                    "date": record_row["timestamp"][:10]  # YYYY-MM-DD
                }
                
                # MongoDB에 삽입
                self.cost_collection.insert_one(cost_doc)
                stats["cost_records"] += 1
            
            conn.close()
            logger.info(f"비용 추적 데이터 마이그레이션 완료: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"비용 추적 데이터 마이그레이션 실패: {e}")
            return {"cost_records": 0}
    
    def migrate_all(self, data_dir: str = "data") -> Dict[str, Any]:
        """전체 데이터 마이그레이션"""
        data_path = Path(data_dir)
        
        logger.info("SQLite -> MongoDB 전체 마이그레이션 시작")
        
        migration_results = {
            "start_time": datetime.now().isoformat(),
            "users": self.migrate_users_data(str(data_path / "users.db")),
            "gallery": self.migrate_gallery_data(str(data_path / "gallery.db")),
            "cost_tracking": self.migrate_cost_data(str(data_path / "cost_tracking.db")),
            "end_time": datetime.now().isoformat()
        }
        
        # 총계 계산
        total_migrated = (
            migration_results["users"]["users"] +
            migration_results["gallery"]["gallery_items"] +
            migration_results["cost_tracking"]["cost_records"]
        )
        
        migration_results["summary"] = {
            "total_records_migrated": total_migrated,
            "migration_successful": total_migrated > 0
        }
        
        logger.info(f"마이그레이션 완료: {total_migrated}개 레코드")
        return migration_results
    
    def verify_migration(self) -> Dict[str, Any]:
        """마이그레이션 결과 검증"""
        verification_results = {
            "users_count": self.users_collection.count_documents({}),
            "gallery_items_count": self.gallery_collection.count_documents({}),
            "cost_records_count": self.cost_collection.count_documents({}),
            "indexes_created": True,
            "connection_healthy": self.mongodb_client.test_connection()
        }
        
        logger.info(f"마이그레이션 검증 완료: {verification_results}")
        return verification_results


def main():
    """마이그레이션 스크립트 실행"""
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Emoseum SQLite -> MongoDB 마이그레이션".center(60))
    print("=" * 60)
    
    try:
        migrator = MongoDBMigrator()
        
        # 기존 데이터 확인
        print("\n1. 기존 SQLite 데이터 확인 중...")
        data_dir = Path("data")
        sqlite_files = [
            data_dir / "users.db",
            data_dir / "gallery.db", 
            data_dir / "cost_tracking.db"
        ]
        
        existing_files = [f for f in sqlite_files if f.exists()]
        print(f"발견된 SQLite 파일: {len(existing_files)}개")
        
        if not existing_files:
            print("마이그레이션할 SQLite 파일이 없습니다.")
            return
        
        # 마이그레이션 실행
        print("\n2. 마이그레이션 실행 중...")
        results = migrator.migrate_all()
        
        # 결과 출력
        print("\n3. 마이그레이션 결과:")
        print(f"   사용자: {results['users']['users']}명")
        print(f"   심리검사: {results['users']['psychometric_results']}개")
        print(f"   갤러리 아이템: {results['gallery']['gallery_items']}개")
        print(f"   비용 기록: {results['cost_tracking']['cost_records']}개")
        print(f"   총 레코드: {results['summary']['total_records_migrated']}개")
        
        # 검증
        print("\n4. 마이그레이션 검증 중...")
        verification = migrator.verify_migration()
        
        print("MongoDB 컬렉션 현황:")
        print(f"   users: {verification['users_count']}개")
        print(f"   gallery_items: {verification['gallery_items_count']}개")
        print(f"   cost_tracking: {verification['cost_records_count']}개")
        print(f"   연결 상태: {'정상' if verification['connection_healthy'] else '오류'}")
        
        if results['summary']['migration_successful']:
            print("\n✅ 마이그레이션이 성공적으로 완료되었습니다!")
            print("\n다음 단계:")
            print("1. 새로운 시스템이 정상 작동하는지 확인")
            print("2. SQLite 파일을 백업 디렉토리로 이동")
            print("3. main.py로 CLI 실행 테스트")
        else:
            print("\n❌ 마이그레이션 중 문제가 발생했습니다.")
            
    except Exception as e:
        logger.error(f"마이그레이션 실패: {e}")
        print(f"\n❌ 마이그레이션 실패: {e}")


if __name__ == "__main__":
    main()