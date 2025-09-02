import os
import requests
import logging
from api.database.connection import get_database
from bson.json_util import dumps

logger = logging.getLogger(__name__)

def sync_ai_json_to_server(gallery_item_id):
    """AI JSON 데이터를 Emoseum-server에 동기화"""
    try:
        db = get_database()
        gallery_item_doc = db["gallery_items"].find_one({"item_id": gallery_item_id})
        
        if gallery_item_doc:
            gallery_item_json = dumps(gallery_item_doc)
            ai_json_update = {"ai_json": gallery_item_json}
            
            print(f"[DEBUG] 전송할 AI JSON 데이터 크기: {len(gallery_item_json)} bytes")
            print(f"[DEBUG] AI JSON 데이터 샘플: {gallery_item_json[:200]}...")
            
            emoseum_server_url = os.getenv("EMOSEUM_SERVER_URL", "http://localhost:3000")
            response = requests.patch(
                f"{emoseum_server_url}/ai-sync/gallery-item/{gallery_item_id}",
                json=ai_json_update,
                timeout=10
            )
            
            print(f"[DEBUG] AI JSON 업데이트 응답: {response.status_code}")
            print(f"[DEBUG] 응답 내용: {response.text}")
            
            if response.status_code == 200:
                logger.info(f"AI JSON 동기화 성공: {gallery_item_id}")
                return True
            else:
                logger.warning(f"AI JSON 동기화 실패: {response.status_code}")
                return False
        
    except Exception as e:
        logger.warning(f"AI JSON 동기화 오류: {e}")
        return False