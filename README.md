# Egyptian National ID — Unified OCR Pipeline

A complete, end-to-end pipeline for extracting personal information from **Egyptian national ID cards**. It combines two independent OCR approaches — **YOLO-based field detection** and **fixed-ratio pixel cuts** — then intelligently merges their outputs to produce the most accurate result.

---

## How It Works

The pipeline runs in **7 sequential steps** inside a single Jupyter notebook:

### Step 1–2: Load Reference Images
Loads (or caches) real and fake ID reference images used for verification. References are saved between runs so you only select them once.

### Step 3: Select the Front of the Test ID
A file picker dialog lets you choose the front-side image to process.

### Step 4: Auto-Rotate
Corrects the card's orientation automatically:
- Removes the background using `rembg` to isolate the card
- Tries 0, 90, 180, 270 degree rotations and picks the one matching the reference aspect ratio
- Detects upside-down cards via **green-strip histogram similarity** (the green header bar on Egyptian IDs) and flips if needed

### Step 5: Verification Gate
Compares the test image against real and fake references using **SSIM + color histograms** across 5 zones. Rejects images that look more like fakes or score below the threshold.

### Step 6: Dual OCR + Smart Merge
Runs **two OCR pipelines independently** on the verified, rotated image:

| Pipeline | Method | OCR Engine |
|----------|--------|------------|
| **YOLO Pipeline** | YOLOv8 detects and crops each field (name, address, NID, etc.) | EasyOCR |
| **Full System Pipeline** | Fixed-ratio pixel cuts based on known ID dimensions | Tesseract + EasyOCR |

Then a **Smart Merger** scores each field from both pipelines and picks the best:
- Length and character-validity scoring
- Bonus points for 14-digit NIDs, valid date formats, Arabic-only names
- Cross-checks NID governorate code against the extracted address

### Step 7: Back-of-ID OCR (Optional)
If you provide the back of the ID, it extracts the **expiration date**.

---

## Extracted Fields

| Field | Example |
|-------|---------|
| First Name | محمد |
| Last Name | أحمد عبد الله |
| Address | القاهرة - مصر الجديدة |
| National ID | 29901011234567 |
| Date of Birth | 1999-01-01 |
| Governorate | القاهرة |
| Expiration Date | 2030-01-01 |

---

## Project Structure

| File | Role |
|------|------|
| `unified_pipeline.ipynb` | Main notebook — all 7 steps in one place |
| `rotation_app.py` | Auto-rotation module (background removal + orientation + flip detection) |
| `ocr_helpers.py` | Shared utilities: address parsing, name cleaning, birthdate extraction, Arabic/English number conversion |
| `ocr_enhancements.py` | National ID validation logic |
| `result_logger.py` | Saves results to JSONL with timestamps |

---

## Requirements

- **Python 3.11+**
- OpenCV (`cv2`)
- NumPy
- `pytesseract` + Tesseract OCR installed
- `easyocr`
- `ultralytics` (YOLO)
- `rembg`
- `scikit-image`
- Pillow

---

## Quick Start

1. Open `unified_pipeline.ipynb` in Jupyter or VS Code
2. Run all cells
3. Follow the file-picker prompts:
   - Select 1–3 **real** reference ID images
   - Optionally select **fake** reference images
   - Select the **front** of the test ID
   - Optionally select the **back**
4. Results are printed and logged to `ocr_results.jsonl`
