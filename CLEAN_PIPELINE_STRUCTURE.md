# Clean OCR Pipeline Structure

## Overview
This document describes the cleaned and organized OCR pipeline structure with exactly 2 files per stage for easy testing and enhancement.

## Directory Structure

```
OCR_pipe_test/
├── pipe_input/                    # Input images (6 test images)
├── intermediate_outputs/          # All pipeline outputs
│   ├── ocr_outputs/              # OCR text extraction results
│   ├── layout_outputs/           # Layout detection results  
│   ├── tableformer_outputs/      # Table structure analysis
│   └── batch_integrated_visualizations/  # Final PDF outputs
├── tesseract_OCR/                # OCR Stage
│   ├── ocr_pipeline.py          # BATCH: Process all images
│   └── batch_ocr_processor.py   # INDIVIDUAL: Process single image
├── docling-ibm-models/batch_processing/  # Layout/Table Stage
│   ├── batch_pipeline.py        # BATCH: Process all images
│   └── run_tableformer.py       # INDIVIDUAL: Process single image
└── reconstruction/               # Reconstruction Stage
    ├── batch_integrated_visualization.py  # BATCH: Process all images
    └── integrated_visualization.py       # INDIVIDUAL: Process single image
```

## Stage 1: OCR Text Extraction

### Batch Processing
```bash
cd tesseract_OCR
python ocr_pipeline.py
```
- Processes all images in `../pipe_input/`
- Outputs to `../intermediate_outputs/ocr_outputs/`
- Generates JSON results + text summaries

### Individual Testing
```bash
cd tesseract_OCR
python batch_ocr_processor.py --input-dir ../pipe_input --output-dir ../intermediate_outputs/ocr_outputs
```
- Process specific images for testing
- Easy to modify and test enhancements

## Stage 2: Layout & Table Detection

### Batch Processing
```bash
cd docling-ibm-models/batch_processing
python batch_pipeline.py
```
- Processes all images in `../../pipe_input/`
- Outputs to `../../intermediate_outputs/layout_outputs/` and `../../intermediate_outputs/tableformer_outputs/`
- Generates layout predictions + table analysis

### Individual Testing
```bash
cd docling-ibm-models/batch_processing
python run_tableformer.py --input-dir ../../pipe_input --layout-dir ../../intermediate_outputs/layout_outputs --table-dir ../../intermediate_outputs/tableformer_outputs
```
- Process specific images for testing
- Easy to modify and test enhancements

## Stage 3: Document Reconstruction

### Batch Processing
```bash
cd reconstruction
python batch_integrated_visualization.py
```
- Processes all images using data from `../intermediate_outputs/`
- Outputs to `intermediate_outputs/batch_integrated_visualizations/`
- Generates final PDF visualizations

### Individual Testing
```bash
cd reconstruction
python integrated_visualization.py --layout-file ../intermediate_outputs/layout_outputs/IMAGE_layout_predictions.json --tableformer-file ../intermediate_outputs/tableformer_outputs/IMAGE_tableformer_results.json --ocr-file ../intermediate_outputs/ocr_outputs/IMAGE_ocr_results.json --output output.pdf
```
- Process specific image for testing
- Easy to modify and test enhancements

## Master Pipeline Runners

### Complete Pipeline (All Images)
```bash
# Python version (recommended)
python run_complete_pipeline.py

# Windows batch version
run_complete_pipeline.bat
```

### Single File Pipeline (Individual Testing)
```bash
# Python version (recommended)
python run_single_file_pipeline.py <image_name>

# Windows batch version
run_single_file_pipeline.bat <image_name>
```

### Examples
```bash
# Process all images
python run_complete_pipeline.py

# Process single image for testing
python run_single_file_pipeline.py page_with_table.png
python run_single_file_pipeline.py 1zM9MmA6dM2_2dbMHiJKd_m-FRdPpNTR3lJUT_P1QuiE.png

# Skip specific stages
python run_complete_pipeline.py --skip-ocr
python run_single_file_pipeline.py page_with_table.png --skip-layout
```

## Quick Testing Workflow

### Option 1: Master Scripts (Recommended)
1. **Single File Testing**: `python run_single_file_pipeline.py page_with_table.png`
2. **Full Pipeline**: `python run_complete_pipeline.py`

### Option 2: Individual Stage Testing
1. **Test OCR Enhancement**: Modify `tesseract_OCR/batch_ocr_processor.py` → run individual test
2. **Test Layout Enhancement**: Modify `docling-ibm-models/batch_processing/run_tableformer.py` → run individual test  
3. **Test Reconstruction Enhancement**: Modify `reconstruction/integrated_visualization.py` → run individual test

## Benefits

- ✅ **Clean Structure**: Only 2 files per stage
- ✅ **Fast Testing**: Individual files for quick iteration
- ✅ **Batch Processing**: Full pipeline for production
- ✅ **Easy Enhancement**: Clear separation of concerns
- ✅ **Standardized I/O**: All stages use same input/output directories

## Dependencies

- **OCR**: `pytesseract`, `PIL`, `opencv-python`
- **Layout/Table**: `transformers`, `torch`, `huggingface-hub`
- **Reconstruction**: `reportlab`, `PIL`, `numpy`

All dependencies are available in the `myenv` virtual environment.
