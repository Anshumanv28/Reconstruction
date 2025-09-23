#!/usr/bin/env python3
"""
Single File Pipeline Runner
==========================

This script runs the complete OCR pipeline on a single image for testing:
1. OCR Text Extraction (single image)
2. Layout & Table Detection (single image)
3. Document Reconstruction (single image)

Usage:
    python run_single_file_pipeline.py <image_name> [--skip-ocr] [--skip-layout] [--skip-reconstruction]

Examples:
    python run_single_file_pipeline.py 1zM9MmA6dM2_2dbMHiJKd_m-FRdPpNTR3lJUT_P1QuiE.png
    python run_single_file_pipeline.py page_with_table.png --skip-ocr
    python run_single_file_pipeline.py ADS.2007.page_123.png --skip-layout --skip-reconstruction

Options:
    --skip-ocr           Skip OCR text extraction stage
    --skip-layout        Skip layout and table detection stage
    --skip-reconstruction Skip document reconstruction stage
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path
import time

def run_command(command, description, cwd=None):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    print(f"Working directory: {cwd or os.getcwd()}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=cwd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ {description} completed successfully!")
        print(f"⏱️  Duration: {duration:.1f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed!")
        print(f"Exit code: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ {description} failed with error: {e}")
        return False

def check_environment_and_image(image_name):
    """Check if we're in the right environment and image exists."""
    print("🔍 Checking environment and image...")
    
    # Check if we're in the right directory
    if not Path("tesseract_OCR").exists():
        print("❌ Error: tesseract_OCR directory not found. Please run from OCR_pipe_test root directory.")
        return False
    
    if not Path("docling-ibm-models/batch_processing").exists():
        print("❌ Error: docling-ibm-models/batch_processing directory not found.")
        return False
        
    if not Path("reconstruction").exists():
        print("❌ Error: reconstruction directory not found.")
        return False
    
    # Check if input directory exists
    if not Path("pipe_input").exists():
        print("❌ Error: pipe_input directory not found.")
        return False
    
    # Check if the specific image exists
    image_path = Path("pipe_input") / image_name
    if not image_path.exists():
        print(f"❌ Error: Image '{image_name}' not found in pipe_input directory.")
        print("Available images:")
        for img in Path("pipe_input").glob("*.png"):
            print(f"   • {img.name}")
        return False
    
    print(f"✅ Environment check passed!")
    print(f"✅ Image '{image_name}' found!")
    return True

def get_image_base_name(image_name):
    """Get base name without extension for file naming."""
    return Path(image_name).stem

def main():
    parser = argparse.ArgumentParser(description="Run OCR pipeline on a single image")
    parser.add_argument("image_name", help="Name of the image file (e.g., 'page_with_table.png')")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip OCR text extraction stage")
    parser.add_argument("--skip-layout", action="store_true", help="Skip layout and table detection stage")
    parser.add_argument("--skip-reconstruction", action="store_true", help="Skip document reconstruction stage")
    
    args = parser.parse_args()
    
    print("🎯 Single File OCR Pipeline Runner")
    print("=" * 60)
    print(f"📁 Target image: {args.image_name}")
    
    # Check environment and image
    if not check_environment_and_image(args.image_name):
        sys.exit(1)
    
    # Get base name for file operations
    base_name = get_image_base_name(args.image_name)
    
    # Track overall progress
    total_start_time = time.time()
    stages_completed = 0
    total_stages = 3
    
    # Stage 1: OCR Text Extraction (single image)
    if not args.skip_ocr:
        # Create a temporary directory with just the target image
        temp_input_dir = f"temp_input_{base_name}"
        temp_output_dir = f"temp_output_{base_name}"
        
        # Create temp directories
        os.makedirs(temp_input_dir, exist_ok=True)
        os.makedirs(temp_output_dir, exist_ok=True)
        
        # Copy the target image to temp input directory
        import shutil
        shutil.copy2(f"../pipe_input/{args.image_name}", f"{temp_input_dir}/{args.image_name}")
        
        success = run_command(
            f"python batch_ocr_processor.py --input {temp_input_dir} --output {temp_output_dir}",
            f"Stage 1: OCR Text Extraction - {args.image_name}",
            cwd="tesseract_OCR"
        )
        
        if success:
            # Move results to the correct location
            import glob
            for file in glob.glob(f"{temp_output_dir}/*"):
                shutil.move(file, f"../intermediate_outputs/ocr_outputs/")
            
            # Clean up temp directories
            shutil.rmtree(temp_input_dir)
            shutil.rmtree(temp_output_dir)
            stages_completed += 1
        else:
            print(f"\n❌ Pipeline failed at OCR stage for {args.image_name}. Stopping.")
            sys.exit(1)
    else:
        print(f"\n⏭️  Skipping OCR Text Extraction stage for {args.image_name}")
    
    # Stage 2: Layout & Table Detection (single image)
    if not args.skip_layout:
        # Create a temporary directory with just the target image
        temp_input_dir = f"../../temp_input_{base_name}"
        os.makedirs(temp_input_dir, exist_ok=True)
        
        # Copy the target image to temp input directory
        import shutil
        shutil.copy2(f"../../pipe_input/{args.image_name}", f"{temp_input_dir}/{args.image_name}")
        
        success = run_command(
            f"python batch_pipeline.py --input-dir {temp_input_dir} --layout-dir ../../intermediate_outputs/layout_outputs --table-dir ../../intermediate_outputs/tableformer_outputs",
            f"Stage 2: Layout & Table Detection - {args.image_name}",
            cwd="docling-ibm-models/batch_processing"
        )
        
        if success:
            # Clean up temp directory
            shutil.rmtree(temp_input_dir)
            stages_completed += 1
        else:
            # Clean up temp directory
            shutil.rmtree(temp_input_dir)
            print(f"\n❌ Pipeline failed at Layout/Table stage for {args.image_name}. Stopping.")
            sys.exit(1)
    else:
        print(f"\n⏭️  Skipping Layout & Table Detection stage for {args.image_name}")
    
    # Stage 3: Document Reconstruction (single image)
    if not args.skip_reconstruction:
        # Check if required files exist
        layout_file = f"../intermediate_outputs/layout_outputs/{base_name}_layout_predictions.json"
        tableformer_file = f"../intermediate_outputs/tableformer_outputs/{base_name}_tableformer_results.json"
        ocr_file = f"../intermediate_outputs/ocr_outputs/{base_name}_ocr_results.json"
        
        # Check if tableformer file exists (some images might not have tables)
        if not Path(tableformer_file).exists():
            print(f"⚠️  Warning: No table data found for {args.image_name}. Running reconstruction without table data.")
            tableformer_file = None
        
        # Build reconstruction command
        reconstruction_cmd = f"python integrated_visualization.py --layout-file {layout_file} --ocr-file {ocr_file} --output {base_name}_test_reconstruction.pdf"
        if tableformer_file:
            reconstruction_cmd += f" --tableformer-file {tableformer_file}"
        
        success = run_command(
            reconstruction_cmd,
            f"Stage 3: Document Reconstruction - {args.image_name}",
            cwd="reconstruction"
        )
        if success:
            stages_completed += 1
        else:
            print(f"\n❌ Pipeline failed at Reconstruction stage for {args.image_name}.")
            sys.exit(1)
    else:
        print(f"\n⏭️  Skipping Document Reconstruction stage for {args.image_name}")
    
    # Final summary
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print(f"\n{'='*60}")
    print("🎉 SINGLE FILE PIPELINE COMPLETE!")
    print(f"{'='*60}")
    print(f"📁 Processed image: {args.image_name}")
    print(f"✅ Stages completed: {stages_completed}/{total_stages}")
    print(f"⏱️  Total duration: {total_duration:.1f} seconds")
    
    if stages_completed == total_stages:
        print(f"\n📁 Output files for {base_name}:")
        print(f"   • OCR results: intermediate_outputs/ocr_outputs/{base_name}_ocr_results.json")
        print(f"   • Layout results: intermediate_outputs/layout_outputs/{base_name}_layout_predictions.json")
        if Path(f"intermediate_outputs/tableformer_outputs/{base_name}_tableformer_results.json").exists():
            print(f"   • Table results: intermediate_outputs/tableformer_outputs/{base_name}_tableformer_results.json")
        print(f"   • Final PDF: reconstruction/{base_name}_test_reconstruction.pdf")
        print(f"\n🎯 Image '{args.image_name}' processed successfully!")
    else:
        print(f"\n⚠️  Pipeline completed with {total_stages - stages_completed} stages skipped.")

if __name__ == "__main__":
    main()
