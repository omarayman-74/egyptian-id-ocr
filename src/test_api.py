import requests
import time

url = "http://localhost:8000/analyze"
img_path = "moraaa.png"

print(f"Testing API at {url} with {img_path}")

try:
    with open(img_path, 'rb') as f:
        files = {'file': ('image.png', f, 'image/png')}
        resp = requests.post(url, files=files)
        
    print(f"Status Code: {resp.status_code}")
    print("Response JSON:")
    print(resp.json())

except Exception as e:
    print(f"Test Failed: {e}")
