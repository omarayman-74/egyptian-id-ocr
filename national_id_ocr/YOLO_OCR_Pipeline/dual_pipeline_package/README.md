# Dual Pipeline Package

This folder contains **all the code files needed** to run the `dual_pipeline_merge.ipynb` notebook.

## Contents

| File | Purpose |
| :--- | :--- |
| `dual_pipeline_merge.ipynb` | The main notebook — runs both pipelines and smart-merges the results |
| `ocr_helpers.py` | Shared helper functions: address parsing, name cleaning, birthdate extraction |
| `ocr_enhancements.py` | National ID validation (14-digit check, OCR digit fix) |
| `rotation_app.py` | Auto-rotation and background removal for ID card images |
| `result_logger.py` | Logs every OCR result to `ocr_results.jsonl` |
| `detect_odjects.pt` | YOLO model weights for detecting ID card fields |
| `requirements.txt` | Python pip dependencies |

## External Dependencies (not included)

These must be installed on the machine separately:

1. **Tesseract OCR** — The notebook expects it at `D:\ocr\tesseract\tesseract.exe` with Arabic language data at `D:\ocr\tessdata`. Update the path in the notebook if yours is different.

## How to Run

1. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Open `dual_pipeline_merge.ipynb` in Jupyter / VS Code.

3. Run all cells. A file dialog will ask you to select an ID card image.

4. Results are saved to `ocr_results.jsonl` automatically.
