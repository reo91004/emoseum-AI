from ..database.connection import get_database
from bson.json_util import dumps

def sync_ai_json_to_server(gallery_item_id, emoseum_server_url):
    try:
        db = get_database()
        gallery_item_doc = db["gallery_items"].find_one({"item_id": gallery_item_id})
        
        if gallery_item_doc:
            # bson.json_util.dumps로 ObjectId 포함 JSON 직렬화
            gallery_item_json = dumps(gallery_item_doc)
            
            ai_json_update = {"ai_json": gallery_item_json}
            
            import requests
            response = requests.patch(
                f"{emoseum_server_url}/ai-sync/gallery-item/{gallery_item_id}",
                json=ai_json_update,
                timeout=10
            )
            
            return response.status_code == 200
    except Exception as e:
        print(f"AI JSON 동기화 실패: {e}")
        return False