# Egyptian ID Card OCR

OCR system for extracting data from Egyptian national ID cards using Tesseract and EasyOCR.

## Features

✅ **Dual OCR engines** (Tesseract + EasyOCR) for better accuracy  
✅ **Extracts**: First name, second name, address, ID number, birthdate  
✅ **Address parsing** with multiple marker types (م, ق, ك)  
✅ **Birthdate extraction** from Egyptian ID format (CYYMMDD)  
✅ **Arabic digit conversion** for proper display  
✅ **Validation rules** for data quality

## Requirements

- Python 3.8+
- Tesseract OCR installed at `D:\ocr\tesseract.exe`
- Arabic language data files in `tessdata/`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Jupyter Notebook (Recommended)

1. Open `ocr.ipynb` in Jupyter or VS Code
2. Run Cell 1 to install dependencies (first time only)
3. Run Cell 2 to start OCR processing
4. Select an ID card image when prompted
5. Results will be printed as a dictionary

### Output Format

```python
{
    "first name": "أحمد",
    "seconed name": "محمد علي",
    "address": "ابوخليفه مركز القنطره غرب ك ١٤",
    "id": "29001234567890",
    "birthdate": "1990-01-23",
    "error": 0
}
```

## Project Structure

```
ocr/
├── ocr.ipynb              # Main Jupyter notebook
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .gitignore            # Git ignore rules
├── tesseract.exe         # Tesseract OCR binary
├── tessdata/             # Tesseract language files
│   └── ara.traineddata   # Arabic language data
├── doc/                  # Documentation
├── data/                 # Input/output data
│   ├── input/           # Sample ID card images
│   └── output/          # Processed results
└── src/                 # Modular code (future)
    └── (empty - for future refactoring)
```

## How It Works

1. **Background Removal**: Uses rembg to isolate the ID card
2. **Image Preprocessing**: Blur, sharpen, edge detection, cropping
3. **Dual OCR**: Both Tesseract and EasyOCR process the image
4. **Smart Extraction**: Voting system picks best results
5. **Address Parsing**: Detects locality and markers (م, ق, ك)
6. **Validation**: Checks data format and quality

## Accuracy Tips

- Use high-quality images (300+ DPI)
- Ensure good lighting and minimal shadows
- Keep card flat and aligned
- Avoid reflections on the card surface

## Future Improvements

See recommendations in the notebook for:
- Multi-PSM voting system
- Advanced preprocessing
- Reference data validation
- Confidence-based selection

## License

See `doc/LICENSE`

## Author

Omar
