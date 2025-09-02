import os
import requests
import logging

logger = logging.getLogger(__name__)

def sync_docent_data_to_server(gallery_item_id, docent_message, gallery_manager):
    """도슨트 메시지와 추가 데이터를 Emoseum-server에 동기화"""
    try:
        # 갤러리 아이템에서 추가 데이터 가져오기
        gallery_item = gallery_manager.get_gallery_item(gallery_item_id)
        
        # 도슨트 메시지 내용 파싱
        message_content = ""
        if isinstance(docent_message, dict):
            if "content" in docent_message:
                content = docent_message["content"]
                if isinstance(content, dict):
                    parts = []
                    for key in ["opening", "recognition", "personal_note", "guidance", "closing"]:
                        if content.get(key):
                            parts.append(content[key])
                    message_content = " ".join(parts)
                else:
                    message_content = str(content)
            else:
                message_content = str(docent_message)
        else:
            message_content = str(docent_message)
        
        update_data = {
            "docent_message": {
                "message": message_content,
                "message_type": "encouragement",
                "personalization_data": docent_message.get("metadata", {}) if isinstance(docent_message, dict) else {}
            },
            "journey_stage": "closure",
            "is_completed": True,
            "normalized_all": gallery_item.normalized_all if gallery_item else {},
            "emotion_categories": gallery_item.emotion_categories if gallery_item else {},
            "artwork_description": gallery_item.artwork_description if gallery_item else ""
        }
        
        print(f"[DEBUG] 전송할 update_data: {update_data}")
        
        emoseum_server_url = os.getenv("EMOSEUM_SERVER_URL", "http://localhost:3000")
        response = requests.patch(
            f"{emoseum_server_url}/ai-sync/gallery-item/{gallery_item_id}",
            json=update_data,
            timeout=10
        )
        
        print(f"[DEBUG] 도슨트 업데이트 응답: {response.status_code}")
        print(f"[DEBUG] 응답 내용: {response.text}")
        
        if response.status_code == 200:
            logger.info(f"도슨트 메시지 동기화 성공: {gallery_item_id}")
            return True
        else:
            logger.warning(f"도슨트 메시지 동기화 실패: {response.status_code}")
            return False
        
    except Exception as e:
        logger.warning(f"도슨트 메시지 동기화 오류: {e}")
        return False