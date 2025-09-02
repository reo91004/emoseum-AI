import requests
import os

def update_emoseum_diary(diary_id, image_path, keywords, primary_emotion):
    """AI 처리 완료 후 Emoseum-server DB 업데이트"""
    webhook_url = f"{os.getenv('EMOSEUM_SERVER_URL', 'http://localhost:3000')}/webhook/diary-update"
    
    payload = {
        "diary_id": diary_id,
        "image_path": image_path,
        "keywords": keywords,
        "primary_emotion": primary_emotion
    }
    
    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        if response.status_code == 200:
            print(f"Emoseum DB 업데이트 성공: {diary_id}")
        else:
            print(f"Emoseum DB 업데이트 실패: {response.text}")
    except Exception as e:
        print(f"Webhook 전송 실패: {e}")