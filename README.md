# Egyptian National ID OCR Pipeline

An end-to-end OCR pipeline for extracting personal information from Egyptian national ID cards. The system combines **YOLO object detection** with **traditional OCR** (Tesseract + EasyOCR) and uses a smart merge strategy to produce the most accurate results.

## Features

- **Auto-rotation** - Automatically corrects orientation (0/90/180/270 degrees) and detects upside-down cards using green-strip color histogram matching
- **Background removal** - Uses rembg to isolate the ID card from any background
- **Fake ID rejection** - Dual-reference verification gate compares test images against real and fake reference IDs using SSIM
- **Dual OCR pipelines** - Runs two independent extraction methods:
  - **YOLO Pipeline** - Uses YOLOv8 to detect and crop individual fields (name, address, NID, etc.)
  - **Full System Pipeline** - Uses fixed-ratio pixel cuts based on known ID card dimensions
- **Smart merge** - Scores results from both pipelines per field and picks the best, with bonuses for valid patterns (e.g., 14-digit NID, Arabic-only names)
- **Back-of-ID support** - Extracts expiration date from the back of the card
- **Result logging** - Saves OCR results to JSONL for analysis

## Files

| File | Description |
|------|-------------|
| `unified_pipeline.ipynb` | Main notebook - runs the complete pipeline end-to-end |
| `rotation_app.py` | Auto-rotation module (rembg + green-strip detection + orientation correction) |
| `ocr_helpers.py` | Shared OCR utilities (address parsing, name cleaning, birthdate extraction) |
| `ocr_enhancements.py` | National ID validation and enhancement functions |
| `result_logger.py` | JSONL result logger with timestamps |

## Pipeline Flow

```
Load References -> Select Front -> Auto-Rotate -> Verify (reject fakes)
    -> Dual OCR (YOLO + Full System) -> Smart Merge -> Back OCR (optional)
    -> Final Result with source tracking
```

## Requirements

- Python 3.11+
- OpenCV (cv2)
- NumPy
- Tesseract OCR (pytesseract)
- EasyOCR
- Ultralytics YOLO
- rembg
- scikit-image
- Pillow

## Usage

1. Open `unified_pipeline.ipynb` in Jupyter Notebook or VS Code
2. Run all cells
3. Follow the prompts to:
   - Select real reference ID images (1-3)
   - Optionally select fake reference images
   - Select the front of the test ID
   - Optionally select the back of the test ID
4. The pipeline will auto-rotate, verify, extract, and display the merged OCR results

## Extracted Fields

- First Name (Arabic)
- Last Name (Arabic)
- Address (Arabic)
- National ID Number (14 digits)
- Date of Birth
- Governorate
- Expiration Date (from back)
