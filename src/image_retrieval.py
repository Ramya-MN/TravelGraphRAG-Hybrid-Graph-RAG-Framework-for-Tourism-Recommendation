# src/image_retrieval.py
from image_manager import ImageManager

class ImageAwareRanker:
    """Layer 2 & 3: Re-ranks recommendations based on visual evidence."""
    
    def __init__(self):
        self.image_manager = ImageManager()
    
    def score_image_availability(self, city_name, ideal_images=3):
        """Scores based on image availability (Returns 0.0 to 1.0)."""
        images = self.image_manager.get_city_images(city_name)
        num_images = len(images)
        
        if num_images == 0: return 0.0 
        return min(1.0, num_images / ideal_images)
    
    def rerank_recommendations(self, recommendations, image_weight=0.15):
        """
        Takes list of (city, text_score) tuples.
        Re-ranks based on image availability.
        """
        reranked = []
        for city, text_score in recommendations:
            image_score = self.score_image_availability(city)
            
            # Combine Text and Visual Scores
            final_score = ((1 - image_weight) * text_score) + (image_weight * image_score)
            
            # Layer 2 Filtering: Only keep cities with at least SOME images
            if image_score > 0:  
                reranked.append((city, round(final_score, 3), image_score))
        
        # Sort by the new Hybrid score
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked
    
    def prepare_final_output(self, reranked_recommendations, images_per_city=3):
        """Layer 3: Prepares the final payload for the user interface."""
        results = []
        for city, final_score, image_score in reranked_recommendations:
            images = self.image_manager.get_city_images(city, limit=images_per_city)
            
            results.append({
                'city': city,
                'confidence_score': final_score,
                'image_availability_score': image_score,
                'images': images,
                'num_images': len(images)
            })
        return results