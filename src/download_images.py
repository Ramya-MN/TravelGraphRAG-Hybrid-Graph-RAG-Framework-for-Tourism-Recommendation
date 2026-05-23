# src/download_images.py
import pandas as pd
import requests
import os
import time

UNSPLASH_API_KEY = "3bd9AEp5DzOQGsBaOEELvruGwLHHrMMd0EiPvwVdz-o"  # Replace this!
BASE_DIR = "C:/Users/Win/Documents/grag2/data/city_images"

def setup_all_images():
    df = pd.read_csv('C:/Users/Win/Documents/grag2/data/europe_cities.csv')
    os.makedirs(BASE_DIR, exist_ok=True)
    
    print(f"Starting image download for {len(df)} cities...")
    
    for idx, row in df.iterrows():
        city = row['city']
        country = row['country']
        print(f"[{idx+1}/{len(df)}] Processing {city}, {country}...")
        
        city_folder = os.path.join(BASE_DIR, city.replace(' ', '_'))
        os.makedirs(city_folder, exist_ok=True)
        
        # Unsplash API Call
        search_query = f"{city} {country} tourism"
        unsplash_url = "https://api.unsplash.com/search/photos"
        params = {
            'query': search_query,
            'per_page': 5, # Getting 5 images per city
            'client_id': UNSPLASH_API_KEY,
            'orientation': 'landscape'
        }
        
        try:
            response = requests.get(unsplash_url, params=params)
            if response.status_code == 200:
                results = response.json().get('results', [])
                for i, photo in enumerate(results):
                    image_url = photo['urls']['regular']
                    img_data = requests.get(image_url).content
                    filepath = os.path.join(city_folder, f"tourism_{i+1}.jpg")
                    
                    with open(filepath, 'wb') as f:
                        f.write(img_data)
                print(f"  ✓ Saved {len(results)} images for {city}")
            else:
                print(f"  ⚠️ API Error for {city}: {response.text}")
        except Exception as e:
            print(f"  ❌ Error downloading images for {city}: {e}")
            
        # Respect Unsplash Free Tier rate limits!
        time.sleep(2)

if __name__ == '__main__':
    setup_all_images()