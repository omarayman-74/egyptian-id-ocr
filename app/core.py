import cv2
import numpy as np
import pytesseract
from rembg import remove
from PIL import Image
from easyocr import easyocr
import io

from .utils import (
    clean_name, choose_address, to_western_digits, 
    arabic_words, extract_birthdate_from_id, clean_id,
    remove_cross_line_duplicates, ARABIC_DIGITS, PUN
)

# Configuration
pytesseract.pytesseract.tesseract_cmd = r'D:\ocr\tesseract.exe'

# Initialize EasyOCR Reader once
reader = easyocr.Reader(['ar', 'ar'])

def preprocess_for_ocr_light(img: np.ndarray) -> np.ndarray:
    if img is None:
        return img
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    gray = cv2.medianBlur(gray, 3)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def process_image(image_bytes: bytes) -> dict:
    data = {
        "first name": "0",
        "seconed name": "0",
        "address": "0",
        "id": "0",
        "birthdate": "0",
        "error": 0
    }
    
    try:
        # Load image
        input_image = Image.open(io.BytesIO(image_bytes))
        
        # Remove background - returns RGBA
        output_image = remove(input_image)
        
        # Convert to OpenCV format (BGR)
        img_np = np.array(output_image)
        if img_np.shape[2] == 4:
            # Convert RGBA to BGR
            img = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Preprocessing mainly derived from notebook
        blurred = cv2.blur(img, (5, 5))
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(blurred, -1, kernel)
        canny = cv2.Canny(sharpened, 50, 200)
        
        pts = np.argwhere(canny > 0)
        if pts.size == 0:
            cropped = img
        else:
            y1, x1 = pts.min(axis=0)
            y2, x2 = pts.max(axis=0)
            cropped = img[y1:y2, x1:x2]
            
        w, h, c = cropped.shape
        o = int(w / 2)
        i = int(h / 2.5)
        n = int(h / 6)
        
        # Region of Interest for text
        cr = cropped[n - 13 : i + 7, o:]
        
        # Read text using Tesseract
        text = pytesseract.image_to_string(cr, lang='ara', config='--psm 11 --oem 3')
        splited = text.split('\n')
        
        # Read text using EasyOCR
        d_text_region = reader.readtext(cr, detail=0, text_threshold=0.18, width_ths=0.9, low_text=0.17)
        
        state = 0
        
        # Heuristic 1: Perfect 8 lines
        if len(text.split('\n')) == 8:
            state = 1
            firstname = splited[0] if len(splited) > 0 else "0"
            secondname = splited[2] if len(splited) > 2 else "0"
            
            address_tesseract = splited[4] + " " + splited[6] if len(splited) > 6 else ""
            address_easyocr = ' '.join(d_text_region[2:]) if len(d_text_region) > 2 else ' '.join(d_text_region)
            
            address = choose_address(address_tesseract, address_easyocr)
            # Remove cross-line duplicates
            address = remove_cross_line_duplicates(address)
            
            data["first name"] = firstname
            data["seconed name"] = secondname
            data["address"] = address
            
            # Extract ID
            cropped_id_region = cropped[i+10:, o+10:]
            if len(cropped_id_region.shape) == 3:
                cropped_id_region = cv2.cvtColor(cropped_id_region, cv2.COLOR_BGR2GRAY)
            
            gauss = cv2.GaussianBlur(cropped_id_region, (7, 7), 0)
            unsharp_image = cv2.addWeighted(cropped_id_region, 2, gauss, -1, 0)
            
            o_id = reader.readtext(unsharp_image, detail=0, text_threshold=0.27, width_ths=0.8, low_text=0.008)
            
            if len(o_id) == 1:
                data["id"] = o_id[0]
            elif len(o_id) == 0:
                data["id"] = "0"
            elif len(o_id) > 1:
                data["id"] = max(o_id, key=len)

        else:
            # Heuristic 2: Fallback
            state = 2
            
            cropped_id_region = cropped[i+10:, o+10:]
            if len(cropped_id_region.shape) == 3:
                cropped_id_region = cv2.cvtColor(cropped_id_region, cv2.COLOR_BGR2GRAY)
                
            gauss = cv2.GaussianBlur(cropped_id_region, (7, 7), 0)
            unsharp_image = cv2.addWeighted(cropped_id_region, 2, gauss, -1, 0)
            
            d = d_text_region
            t_lines = [x.strip() for x in splited if x.strip()]
            address_tesseract = ' '.join(t_lines[2:]) if len(t_lines) > 2 else ''
            
            if d:
                data["first name"] = d[0] if len(d) > 0 else "0"
                data["seconed name"] = d[1] if len(d) > 1 else "0"
                address_easyocr_loop = ' '.join(d[2:]) if len(d) > 2 else ' '.join(d)
                data["address"] = choose_address(address_tesseract, address_easyocr_loop)
                # Remove cross-line duplicates
                data["address"] = remove_cross_line_duplicates(data["address"])
                
            o_id = reader.readtext(unsharp_image, detail=0, text_threshold=0.27, width_ths=0.8, low_text=0.008)
            
            if not o_id and not d:
                state = 4
                data["error"] = 1
            else:
                if len(o_id) == 1:
                    data["id"] = o_id[0]
                elif len(o_id) == 0:
                    data["id"] = "0"
                elif len(o_id) > 1:
                    data["id"] = max(o_id, key=len)

        if state == 4:
            data["error"] = 1
            
        # Post-processing / Cleaning
        
        # Clean Names
        for char in data["first name"]:
            if char in ARABIC_DIGITS or char in PUN:
                data["first name"] = data["first name"].replace(char, "")
        for char in data["seconed name"]:
            if char in ARABIC_DIGITS or char in PUN:
                data["seconed name"] = data["seconed name"].replace(char, "")
                
        # Address Cleanup
        if isinstance(data["address"], str):
             data["address"] = data["address"].replace("[", "").replace("]", "").replace("'", "")

        # Name length checks
        f_parts = data["first name"].split()
        s_parts = data["seconed name"].split()
        if len(f_parts) > 3:
            data["error"] = 1
        if len(s_parts) <= 1:
            data["error"] = 1

        # ID Cleaning
        data["id"] = clean_id(data["id"])
        
        data["birthdate"] = extract_birthdate_from_id(data["id"])
        
        return data

    except Exception as e:
        print(f"Processing Error: {e}")
        return {
            "id": 0,
            "error": 1,
            "birthdate": "0",
            "message": str(e)
        }
