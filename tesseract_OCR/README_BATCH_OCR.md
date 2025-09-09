# Batch OCR Processing System

This directory contains an enhanced batch processing system for OCR operations using Tesseract OCR. The system can process multiple document images in parallel or sequentially, with comprehensive error handling, progress tracking, and detailed reporting.

## 📁 Directory Structure

```
tesseract_OCR/
├── input/                          # Input images directory
├── output/                         # Output results directory
├── ocr_pipeline.py                 # Core OCR pipeline
├── batch_ocr_processor.py          # Enhanced batch processor
├── test_batch_ocr.py              # Test script
└── README_BATCH_OCR.md            # This documentation
```

## 🚀 Quick Start

### 1. Basic Usage

```bash
# Process all images in the input directory
python batch_ocr_processor.py

# Process with custom input/output directories
python batch_ocr_processor.py --input /path/to/images --output /path/to/results

# Use sequential processing (slower but more stable)
python batch_ocr_processor.py --sequential

# Use more parallel workers for faster processing
python batch_ocr_processor.py --workers 8
```

### 2. Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--input` | `-i` | Input directory containing images | `input` |
| `--output` | `-o` | Output directory for results | `output` |
| `--workers` | `-w` | Number of parallel workers | `4` |
| `--sequential` | `-s` | Use sequential processing | `False` |
| `--log-level` | `-l` | Logging level (DEBUG/INFO/WARNING/ERROR) | `INFO` |
| `--formats` | `-f` | Supported image formats | `.png .jpg .jpeg .tiff .bmp .gif` |

### 3. Example Commands

```bash
# High-performance processing with 8 workers
python batch_ocr_processor.py --workers 8 --log-level INFO

# Debug mode with detailed logging
python batch_ocr_processor.py --log-level DEBUG

# Process only specific image formats
python batch_ocr_processor.py --formats .png .jpg

# Sequential processing for debugging
python batch_ocr_processor.py --sequential --log-level DEBUG
```

## 📊 Output Files

The batch processor generates several types of output files:

### Individual Results
- `{filename}_ocr_results.json` - Detailed OCR results for each image
- `{filename}_ocr_summary.txt` - Human-readable text summary

### Batch Results
- `batch_ocr_results_{timestamp}.json` - Comprehensive batch processing results
- `batch_ocr_summary_{timestamp}.txt` - Human-readable batch summary
- `batch_ocr_{timestamp}.log` - Detailed processing log

### Output Structure
```json
{
  "batch_processing_info": {
    "total_files": 10,
    "processed_files": 9,
    "failed_files": 1,
    "success_rate": 90.0,
    "processing_duration": "0:02:15.123456",
    "max_workers": 4
  },
  "processing_errors": [
    {
      "file": "corrupted_image.jpg",
      "error": "Error processing corrupted_image.jpg: Invalid image format",
      "timestamp": "2025-01-09T12:30:45.123456"
    }
  ],
  "individual_results": [...]
}
```

## 🔧 Configuration

### Supported Image Formats
- PNG (`.png`)
- JPEG (`.jpg`, `.jpeg`)
- TIFF (`.tiff`)
- BMP (`.bmp`)
- GIF (`.gif`)

### OCR Configuration
The OCR pipeline uses the following default settings:
- **PSM**: 6 (Assume a single uniform block of text)
- **OEM**: 3 (Default OCR Engine Mode)
- **Language**: English (`eng`)
- **Minimum Confidence**: 30%

### Performance Tuning

#### Parallel Processing
- **Recommended workers**: 2-4 for most systems
- **High-end systems**: 6-8 workers
- **Memory considerations**: Each worker loads the full image into memory

#### Sequential Processing
- Use when debugging or when system resources are limited
- More stable but significantly slower
- Better for processing very large images

## 🧪 Testing

### Run Test Suite
```bash
python test_batch_ocr.py
```

The test suite will:
1. Test parallel batch processing
2. Test sequential processing
3. Generate comprehensive reports
4. Validate output files

### Manual Testing
1. Add test images to the `input/` directory
2. Run the batch processor
3. Check output files in the `output/` directory
4. Review logs for any issues

## 📈 Performance Metrics

### Typical Performance (on modern hardware)
- **Small images** (1-2 MB): 2-5 seconds per image
- **Medium images** (5-10 MB): 5-15 seconds per image
- **Large images** (10+ MB): 15-30 seconds per image

### Parallel Processing Benefits
- **2 workers**: ~1.5x speedup
- **4 workers**: ~2.5x speedup
- **8 workers**: ~3-4x speedup (diminishing returns)

## 🐛 Troubleshooting

### Common Issues

#### 1. Tesseract Not Found
```
Error: Tesseract not found: pytesseract module not available
```
**Solution**: Ensure Tesseract OCR is installed and in PATH, or modify the script to include the system Python path.

#### 2. Memory Issues
```
Error: Out of memory during processing
```
**Solution**: Reduce the number of parallel workers or use sequential processing.

#### 3. Image Format Issues
```
Error: Unsupported image format
```
**Solution**: Convert images to supported formats or add the format to the supported_formats list.

#### 4. Permission Issues
```
Error: Permission denied when writing to output directory
```
**Solution**: Ensure write permissions for the output directory.

### Debug Mode
Enable debug logging for detailed troubleshooting:
```bash
python batch_ocr_processor.py --log-level DEBUG
```

## 🔄 Integration with Docling Pipeline

The batch OCR processor can be integrated with the Docling document processing pipeline:

1. **Input**: Use images processed by Docling's layout detection
2. **Processing**: Extract text using OCR
3. **Output**: Generate text data for document reconstruction
4. **Integration**: Combine with Docling's table structure recognition

### Example Integration Workflow
```bash
# 1. Process documents with Docling
python ../batch_pipeline.py

# 2. Extract text with OCR
python batch_ocr_processor.py --input ../layout_output --output ocr_results

# 3. Combine results for reconstruction
python ../reconstruction/improved_fresh_reconstruction.py
```

## 📝 Logging

The system provides comprehensive logging at multiple levels:

- **DEBUG**: Detailed processing information
- **INFO**: General processing status
- **WARNING**: Non-fatal issues
- **ERROR**: Processing failures

Log files are automatically created with timestamps and include:
- Processing start/end times
- Individual file processing status
- Error details and stack traces
- Performance metrics

## 🚀 Advanced Usage

### Custom OCR Configuration
Modify the `OCRPipeline` class in `ocr_pipeline.py` to customize:
- OCR engine parameters
- Image preprocessing
- Text post-processing
- Output formats

### Batch Processing API
Use the `BatchOCRProcessor` class programmatically:

```python
from batch_ocr_processor import BatchOCRProcessor

# Initialize processor
processor = BatchOCRProcessor(
    input_dir="my_images",
    output_dir="my_results",
    max_workers=6,
    log_level="INFO"
)

# Run processing
success = processor.run_batch_processing(parallel=True)

# Access results
if success:
    print(f"Processed {processor.stats['processed_files']} files")
```

## 📋 Requirements

- Python 3.7+
- Tesseract OCR 5.0+
- pytesseract
- Pillow (PIL)
- OpenCV
- NumPy

## 🤝 Contributing

To contribute to the batch OCR processing system:

1. Add new features to `batch_ocr_processor.py`
2. Update tests in `test_batch_ocr.py`
3. Update documentation in `README_BATCH_OCR.md`
4. Test with various image types and sizes
5. Ensure backward compatibility

## 📄 License

This batch OCR processing system is part of the Docling IBM models project and follows the same licensing terms.
