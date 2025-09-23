@echo off
echo ========================================
echo OCR Pipeline Master Runner
echo ========================================
echo.

echo Stage 1: OCR Text Extraction...
cd tesseract_OCR
python ocr_pipeline.py
if errorlevel 1 (
    echo OCR stage failed!
    pause
    exit /b 1
)
cd ..

echo.
echo Stage 2: Layout ^& Table Detection...
cd docling-ibm-models\batch_processing
python batch_pipeline.py
if errorlevel 1 (
    echo Layout/Table stage failed!
    pause
    exit /b 1
)
cd ..\..

echo.
echo Stage 3: Document Reconstruction...
cd reconstruction
python batch_integrated_visualization.py
if errorlevel 1 (
    echo Reconstruction stage failed!
    pause
    exit /b 1
)
cd ..

echo.
echo ========================================
echo Pipeline Complete!
echo ========================================
echo.
echo Output locations:
echo   OCR results: intermediate_outputs\ocr_outputs\
echo   Layout results: intermediate_outputs\layout_outputs\
echo   Table results: intermediate_outputs\tableformer_outputs\
echo   Final PDFs: intermediate_outputs\batch_integrated_visualizations\
echo.
pause
