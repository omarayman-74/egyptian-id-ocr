import json
import os
import datetime
import requests

LOG_FILE = "ocr_results.jsonl"
API_URL = "https://vso-dev-customer.nuca-mycluster-eu-de-1-cx-5fc3035946e1f798c7284cb63267e8d1-0000.eu-de.containers.appdomain.cloud/api/ocr/extract_verified"

def log_result(source, image_path, result_data):
    """
    Logs the OCR result into a JSON Lines file.
    
    :param source: String, e.g., 'dual_pipeline' or 'api'
    :param image_path: Path to the image that was processed
    :param result_data: Dictionary containing the extracted data
    """
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "source": source,
        "image_path": os.path.abspath(image_path),
        "result": result_data
    }
    
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    print(f"[Logger] Saved {source} result to {LOG_FILE}")


def run_api_and_log(image_path):
    """
    Calls the external API and logs the result automatically.
    
    :param image_path: Path to the image
    :return: The JSON response from the API
    """
    print(f"Calling API for image: {image_path}...")
    try:
        with open(image_path, 'rb') as f:
            # Assuming the API expects a form-data field named 'file' or 'image'
            # Usually FastAPI docs with that name structure expect 'file'
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            response = requests.post(API_URL, files=files)
            
        if response.status_code == 200:
            result = response.json()
            log_result("api", image_path, result)
            return result
        else:
            print(f"API Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"Failed to call API: {e}")
        return None
