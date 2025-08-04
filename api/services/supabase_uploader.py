import os
import logging
from supabase import create_client, Client
from typing import Optional

logger = logging.getLogger(__name__)

class SupabaseUploader:
    def __init__(self):
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
        
        self.client: Client = create_client(self.supabase_url, self.supabase_key)
    
    def upload_image(self, image_path: str, filename: str) -> Optional[str]:
        """Upload image to Supabase and return public URL"""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Upload to Supabase storage
            result = self.client.storage.from_('emoseum-images').upload(
                f'generated/{filename}',
                image_data,
                file_options={'content-type': 'image/png', 'upsert': 'true'}
            )
            
            if result:
                # Get public URL
                public_url = self.client.storage.from_('emoseum-images').get_public_url(f'generated/{filename}')
                logger.info(f"Image uploaded successfully: {public_url}")
                return public_url
            
        except Exception as e:
            logger.error(f"Failed to upload image to Supabase: {e}")
            return None

# Global instance
supabase_uploader = SupabaseUploader()