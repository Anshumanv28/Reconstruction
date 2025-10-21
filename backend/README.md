## Backend API (FastAPI)

Run locally:

```bash
# From project root directory (C:\work\github\Reconstruction)
pip install -r backend/requirements.txt
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

**Or from backend directory:**
```bash
# From backend directory
cd backend
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints:
- GET `/health` - Health check
- GET `/models` - List available models
- POST `/infer` - Base64 image inference (JSON response)
- POST `/infer-file` - Raw file upload with JSON response (PNG, JPG, PDF)
- POST `/infer-file-with-files` - Raw file upload returning ZIP with annotated image + JSON files
- POST `/infer-file-json` - Raw file upload returning only raw JSON data

Request body example for `/infer` (base64):

```json
{
  "inputs": [
    { "image_id": "page1", "image_b64": "data:image/png;base64,...." }
  ],
  "layout_model": "docling_layout_v1",
  "ocr_model": "tesseract_default",
  "params": { "layout": {}, "ocr": {} },
  "return_visualization": true
}
```

Request example for `/infer-file` (multipart form data):

```
file: [upload PNG/JPG/PDF file]
layout_model: docling_layout_v1
ocr_model: tesseract_default
return_visualization: true
```

Request example for `/infer-file-with-files` (multipart form data):

```
file: [upload PNG/JPG/PDF file]
layout_model: docling_layout_v1
ocr_model: tesseract_default
```

**Returns:** ZIP file containing:
- `annotated_[filename]`: Image with visual annotations overlaid
- `data_[filename].json`: Raw JSON data with layout boxes and OCR tokens

Request example for `/infer-file-json` (multipart form data):

```
file: [upload PNG/JPG/PDF file]
layout_model: docling_layout_v1
ocr_model: tesseract_default
```

**Returns:** Raw JSON data only (no visualization)


