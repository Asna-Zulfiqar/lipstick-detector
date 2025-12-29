import requests
import cv2
import time
from pathlib import Path

# Config
DATA_DIR = Path("data/raw/")
DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_TO_DOWNLOAD = 200
API_KEY = "gwwCcA4RAAWSE0mvRLcmUGbxReov8QZXSGxRnmMXz1iW7Im7pfypUORw"


def download_lipstick_images():
    print("Downloading lipstick images from Pexels...")

    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": API_KEY}

    downloaded = 0
    page = 1

    while downloaded < IMAGES_TO_DOWNLOAD:
        params = {
            "query": "lipstick",
            "per_page": 80,
            "page": page,
            "orientation": "square"
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)

            if response.status_code == 429:
                print("Rate limited. Waiting 60 seconds...")
                time.sleep(60)
                continue

            data = response.json()
            photos = data.get('photos', [])

            if not photos:
                print("No more photos found.")
                break

            for photo in photos:
                if downloaded >= IMAGES_TO_DOWNLOAD:
                    break

                img_url = photo.get('src', {}).get('large')
                if img_url:
                    try:
                        img_data = requests.get(img_url, timeout=10).content
                        filename = f"lipstick_pexels_{downloaded:04d}.jpg"
                        path = DATA_DIR / filename

                        with open(path, "wb") as f:
                            f.write(img_data)

                        # Quick validation
                        img = cv2.imread(str(path))
                        if img is not None and img.shape[0] > 200 and img.shape[1] > 200:
                            downloaded += 1
                            print(f"✅ Downloaded {filename} ({downloaded}/{IMAGES_TO_DOWNLOAD})")
                        else:
                            path.unlink()

                        time.sleep(0.2)
                    except Exception as e:
                        print(f"Error: {e}")
                        continue

            page += 1
            time.sleep(1)

        except Exception as e:
            print(f"Request error: {e}")
            break

    print(f"\n✅ Downloaded {downloaded} images to {DATA_DIR}")


if __name__ == "__main__":
    download_lipstick_images()