# src/image_manager.py
import os
from pathlib import Path

class ImageManager:
    """Manages local image retrieval to prevent unnecessary API calls."""
    
    def __init__(self, image_base_path='../data/city_images'):
        self.base_path = Path(image_base_path)
    
    def get_city_images(self, city_name, limit=5):
        """Get a list of image dictionaries for a given city."""
        city_folder = self.base_path / city_name.replace(' ', '_')
        
        if not city_folder.exists():
            return []
        
        images = []
        for img_file in sorted(city_folder.glob('*.jpg'))[:limit]:
            images.append({
                'path': str(img_file),
                'filename': img_file.name,
                'category': img_file.stem.split('_')[0]
            })
        
        return images