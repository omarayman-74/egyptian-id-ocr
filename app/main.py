from fastapi import FastAPI, UploadFile, File, HTTPException
from .core import process_image
import uvicorn

app = FastAPI(title="OCR Service", description="API for ID OCR processing")

@app.get("/")
async def root():
    return {"message": "OCR Service is Running"}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        result = process_image(contents)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
