@echo off
if "%~1"=="" (
    echo Usage: run_single_file_pipeline.bat ^<image_name^>
    echo.
    echo Examples:
    echo   run_single_file_pipeline.bat page_with_table.png
    echo   run_single_file_pipeline.bat 1zM9MmA6dM2_2dbMHiJKd_m-FRdPpNTR3lJUT_P1QuiE.png
    echo.
    echo Available images:
    for %%f in (pipe_input\*.png) do echo   • %%~nxf
    pause
    exit /b 1
)

set IMAGE_NAME=%~1
set BASE_NAME=%~n1

echo ========================================
echo Single File OCR Pipeline Runner
echo ========================================
echo Target image: %IMAGE_NAME%
echo.

echo Stage 1: OCR Text Extraction...
cd tesseract_OCR
python batch_ocr_processor.py --input-dir ../pipe_input --output-dir ../intermediate_outputs/ocr_outputs --image %IMAGE_NAME%
if errorlevel 1 (
    echo OCR stage failed!
    pause
    exit /b 1
)
cd ..

echo.
echo Stage 2: Layout ^& Table Detection...
cd docling-ibm-models\batch_processing
python run_tableformer.py --input-dir ../../pipe_input --layout-dir ../../intermediate_outputs/layout_outputs --table-dir ../../intermediate_outputs/tableformer_outputs --image %IMAGE_NAME%
if errorlevel 1 (
    echo Layout/Table stage failed!
    pause
    exit /b 1
)
cd ..\..

echo.
echo Stage 3: Document Reconstruction...
cd reconstruction
if exist "..\intermediate_outputs\tableformer_outputs\%BASE_NAME%_tableformer_results.json" (
    python integrated_visualization.py --layout-file ../intermediate_outputs/layout_outputs/%BASE_NAME%_layout_predictions.json --tableformer-file ../intermediate_outputs/tableformer_outputs/%BASE_NAME%_tableformer_results.json --ocr-file ../intermediate_outputs/ocr_outputs/%BASE_NAME%_ocr_results.json --output %BASE_NAME%_test_reconstruction.pdf
) else (
    echo Warning: No table data found. Running reconstruction without table data.
    python integrated_visualization.py --layout-file ../intermediate_outputs/layout_outputs/%BASE_NAME%_layout_predictions.json --ocr-file ../intermediate_outputs/ocr_outputs/%BASE_NAME%_ocr_results.json --output %BASE_NAME%_test_reconstruction.pdf
)
if errorlevel 1 (
    echo Reconstruction stage failed!
    pause
    exit /b 1
)
cd ..

echo.
echo ========================================
echo Single File Pipeline Complete!
echo ========================================
echo.
echo Processed image: %IMAGE_NAME%
echo.
echo Output files:
echo   OCR results: intermediate_outputs\ocr_outputs\%BASE_NAME%_ocr_results.json
echo   Layout results: intermediate_outputs\layout_outputs\%BASE_NAME%_layout_predictions.json
if exist "intermediate_outputs\tableformer_outputs\%BASE_NAME%_tableformer_results.json" (
    echo   Table results: intermediate_outputs\tableformer_outputs\%BASE_NAME%_tableformer_results.json
)
echo   Final PDF: reconstruction\%BASE_NAME%_test_reconstruction.pdf
echo.
pause
