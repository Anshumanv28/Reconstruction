# Standardized OCR Pipeline Structure

This document describes the standardized input/output structure for the OCR processing pipeline, ensuring both Docling and Tesseract models work with the same data source.

## 📁 Directory Structure

```
OCR_pipe_test/
├── pipe_input/                          # 🎯 STANDARDIZED INPUT SOURCE
│   ├── ADS.2007.page_123.png
│   ├── PHM.2013.page_30.png
│   ├── page_with_table.png
│   ├── page_with_list.png
│   ├── empty_iocr.png
│   └── 1zM9MmA6dM2_2dbMHiJKd_m-FRdPpNTR3lJUT_P1QuiE.png
│
├── intermediate_outputs/                 # 📊 ORGANIZED INTERMEDIATE RESULTS
│   ├── layout_outputs/                  # Docling layout detection results
│   ├── tableformer_outputs/             # Docling table structure analysis
│   ├── ocr_outputs/                     # Tesseract OCR results
│   └── complete_pipeline_output/        # Complete pipeline results
│
├── pipe_output/                         # 🎯 FINAL PROCESSED RESULTS
│
├── docling-ibm-models/                  # Docling AI models
│   └── batch_processing/
│       ├── batch_pipeline.py            # ✅ Updated to use pipe_input
│       ├── complete_pipeline.py         # ✅ Updated to use pipe_input
│       └── test_batch.py                # ✅ Updated paths
│
├── tesseract_OCR/                       # Tesseract OCR processing
│   ├── ocr_pipeline.py                  # ✅ Updated to use pipe_input
│   ├── batch_ocr_processor.py           # ✅ Updated to use pipe_input
│   └── test_batch_ocr.py
│
└── reconstruction/                      # Document reconstruction
    ├── improved_fresh_reconstruction.py
    └── reconstruction_output/
```

## 🎯 Standardized Input Source

**All models now use `pipe_input/` as the common input directory:**

- **Docling Models**: `../../pipe_input` (from batch_processing folder)
- **Tesseract OCR**: `../pipe_input` (from tesseract_OCR folder)
- **Reconstruction**: Uses outputs from both pipelines

## 📊 Organized Output Structure

### Docling Pipeline Outputs
- **Layout Detection**: `intermediate_outputs/layout_outputs/`
  - `{image}_layout_annotated.png` - Annotated images with layout elements
  - `{image}_layout_predictions.json` - All detected elements (tables, figures, text)
  - `{image}_table_coordinates.json` - Table bounding boxes only

- **Table Structure Analysis**: `intermediate_outputs/tableformer_outputs/`
  - `{image}_tableformer_analysis.png` - Annotated images with table structure
  - `{image}_tableformer_results.json` - Detailed table cell analysis

### Tesseract OCR Outputs
- **OCR Results**: `intermediate_outputs/ocr_outputs/`
  - `{image}_ocr_results.json` - Detailed OCR results with confidence scores
  - `{image}_ocr_summary.txt` - Human-readable text summary
  - `batch_ocr_results_{timestamp}.json` - Batch processing results
  - `batch_ocr_summary_{timestamp}.txt` - Batch summary report

### Complete Pipeline Outputs
- **Complete Results**: `intermediate_outputs/complete_pipeline_output/`
  - Combined results from layout detection and table analysis

## 🚀 Usage Commands

### Docling Pipeline
```bash
# Navigate to docling batch processing
cd docling-ibm-models/batch_processing

# Test setup
python test_batch.py

# Process all images from pipe_input
python batch_pipeline.py

# Process single image
python complete_pipeline.py ../../pipe_input/ADS.2007.page_123.png

# With custom parameters
python batch_pipeline.py --input-dir ../../pipe_input --layout-dir ../../intermediate_outputs/layout_outputs --table-dir ../../intermediate_outputs/tableformer_outputs
```

### Tesseract OCR Pipeline
```bash
# Navigate to tesseract OCR
cd tesseract_OCR

# Process all images from pipe_input
python batch_ocr_processor.py

# Process with custom parameters
python batch_ocr_processor.py --input ../pipe_input --output ../intermediate_outputs/ocr_outputs --workers 4

# Single image processing
python ocr_pipeline.py
```

### Complete Workflow
```bash
# 1. Run Docling layout detection and table analysis
cd docling-ibm-models/batch_processing
python batch_pipeline.py

# 2. Run Tesseract OCR processing
cd ../../tesseract_OCR
python batch_ocr_processor.py

# 3. Run document reconstruction (uses outputs from both)
cd ../reconstruction
python improved_fresh_reconstruction.py
```

## 🔧 Configuration Changes Made

### Docling Pipeline Updates
- **batch_pipeline.py**: 
  - Input: `input_images` → `../../pipe_input`
  - Layout output: `layout_output` → `../../intermediate_outputs/layout_outputs`
  - Table output: `tableformer_output` → `../../intermediate_outputs/tableformer_outputs`

- **complete_pipeline.py**:
  - Output: `pipeline_output` → `../../intermediate_outputs/complete_pipeline_output`

- **test_batch.py**:
  - Updated all path references to use new standardized structure

### Tesseract OCR Updates
- **ocr_pipeline.py**:
  - Input: `input` → `../pipe_input`
  - Output: `output` → `../intermediate_outputs/ocr_outputs`

- **batch_ocr_processor.py**:
  - Input: `input` → `../pipe_input`
  - Output: `output` → `../intermediate_outputs/ocr_outputs`

## ✅ Benefits of Standardization

1. **Single Source of Truth**: All models use `pipe_input/` as the common input
2. **Organized Outputs**: Clear separation of intermediate results by processing stage
3. **Easy Integration**: Both pipelines can be run independently or together
4. **Consistent Paths**: Relative paths work from any processing directory
5. **Scalable Structure**: Easy to add new processing stages or models

## 🧪 Testing the Standardized Setup

### Test Docling Setup
```bash
cd docling-ibm-models/batch_processing
python test_batch.py
```

### Test Tesseract Setup
```bash
cd tesseract_OCR
python test_batch_ocr.py
```

### Verify Input Images
```bash
# Check that pipe_input contains the expected images
ls pipe_input/
```

## 📋 Next Steps

1. **Add Images**: Place new test images in `pipe_input/`
2. **Run Pipelines**: Execute both Docling and Tesseract processing
3. **Check Results**: Review outputs in respective `intermediate_outputs/` folders
4. **Reconstruct**: Use reconstruction module to create final documents
5. **Integrate**: Combine results for your specific use case

## 🔄 Pipeline Integration

The standardized structure enables seamless integration between:

- **Layout Detection** → **Table Analysis** → **OCR Processing** → **Document Reconstruction**

Each stage produces outputs that can be consumed by the next stage, creating a complete document processing pipeline.
