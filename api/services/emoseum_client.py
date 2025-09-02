# api/services/emoseum_client.py

import httpx
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class EmoseumServerClient:
    """Emoseum 중앙 서버와 통신하는 클라이언트"""
    
    def __init__(self, base_url: str = None):
        import os
        self.base_url = base_url or os.getenv('EMOSEUM_SERVER_URL', 'http://localhost:3000')
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def update_diary_from_ai(
        self,
        diary_id: str,
        keywords: list,
        image_path: str,
        reflection_prompt: str = ""
    ) -> Dict[str, Any]:
        """AI 처리 결과를 중앙 서버에 전송"""
        try:
            payload = {
                "diary_id": diary_id,
                "keywords": keywords,
                "imagePath": image_path,
                "reflection_prompt": reflection_prompt
            }
            
            logger.info(f"Sending to {self.base_url}/api/diary/updateFromAISession with payload: {payload}")
            
            response = await self.client.post(
                f"{self.base_url}/diary/updateFromAISession",
                json=payload
            )
            
            logger.info(f"Response status: {response.status_code}, body: {response.text}")
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to update diary: {response.status_code} - {response.text}")
                return {"success": False, "error": response.text}
                
        except Exception as e:
            logger.error(f"Error calling emoseum server: {e}")
            return {"success": False, "error": str(e)}
    
    async def close(self):
        """클라이언트 연결 종료"""
        await self.client.aclose()


# 싱글톤 인스턴스
emoseum_client = EmoseumServerClient("http://51.20.51.115:3000")