#!/usr/bin/env python3
"""
Complete OCR Pipeline Runner

This script runs the entire OCR pipeline:
1. Docling batch processing (layout detection + table analysis)
2. Tesseract OCR processing
3. Document reconstruction

All using the standardized input/output structure.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import time

def run_command(command, cwd=None, description=""):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"Command: {command}")
    print(f"Working directory: {cwd or 'current'}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=cwd, 
            capture_output=False,  # Show output in real-time
            text=True,
            check=True
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ {description} completed successfully!")
        print(f"⏱️  Duration: {duration:.2f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n❌ {description} failed!")
        print(f"⏱️  Duration: {duration:.2f} seconds")
        print(f"Exit code: {e.returncode}")
        return False

def check_prerequisites():
    """Check if all required directories and files exist."""
    print("🔍 Checking prerequisites...")
    
    # Check input directory
    if not Path("pipe_input").exists():
        print("❌ pipe_input directory not found!")
        return False
    
    input_files = list(Path("pipe_input").glob("*.png")) + list(Path("pipe_input").glob("*.jpg"))
    if not input_files:
        print("❌ No image files found in pipe_input directory!")
        return False
    
    print(f"✅ Found {len(input_files)} input images")
    
    # Check required directories
    required_dirs = [
        "docling-ibm-models/batch_processing",
        "tesseract_OCR",
        "reconstruction"
    ]
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            print(f"❌ Required directory not found: {dir_path}")
            return False
    
    print("✅ All prerequisites met!")
    return True

def run_docling_pipeline():
    """Run the Docling batch processing pipeline."""
    return run_command(
        "python batch_pipeline.py",
        cwd="docling-ibm-models/batch_processing",
        description="Docling Layout Detection & Table Analysis"
    )

def run_tesseract_pipeline():
    """Run the Tesseract OCR pipeline."""
    return run_command(
        "python batch_ocr_processor.py",
        cwd="tesseract_OCR",
        description="Tesseract OCR Processing"
    )

def run_reconstruction():
    """Run the document reconstruction."""
    return run_command(
        "python improved_fresh_reconstruction.py --mode pipeline",
        cwd="reconstruction",
        description="Document Reconstruction"
    )

def main():
    """Main pipeline runner."""
    parser = argparse.ArgumentParser(description="Complete OCR Pipeline Runner")
    parser.add_argument("--skip-docling", action="store_true",
                       help="Skip Docling processing (use existing intermediate outputs)")
    parser.add_argument("--skip-tesseract", action="store_true",
                       help="Skip Tesseract processing")
    parser.add_argument("--skip-reconstruction", action="store_true",
                       help="Skip document reconstruction")
    parser.add_argument("--steps", nargs="+", 
                       choices=["docling", "tesseract", "reconstruction"],
                       help="Run only specific steps")
    
    args = parser.parse_args()
    
    print("🚀 Starting Complete OCR Pipeline")
    print("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites check failed. Exiting.")
        sys.exit(1)
    
    # Track overall success
    overall_success = True
    start_time = time.time()
    
    # Run pipeline steps
    steps_to_run = []
    
    if args.steps:
        # Run only specified steps
        steps_to_run = args.steps
    else:
        # Run all steps unless skipped
        if not args.skip_docling:
            steps_to_run.append("docling")
        if not args.skip_tesseract:
            steps_to_run.append("tesseract")
        if not args.skip_reconstruction:
            steps_to_run.append("reconstruction")
    
    print(f"\n📋 Pipeline steps to run: {', '.join(steps_to_run)}")
    
    # Execute steps
    for step in steps_to_run:
        if step == "docling":
            success = run_docling_pipeline()
        elif step == "tesseract":
            success = run_tesseract_pipeline()
        elif step == "reconstruction":
            success = run_reconstruction()
        
        if not success:
            overall_success = False
            print(f"\n❌ Pipeline failed at step: {step}")
            break
    
    # Final summary
    end_time = time.time()
    total_duration = end_time - start_time
    
    print(f"\n{'='*60}")
    if overall_success:
        print("🎉 COMPLETE PIPELINE SUCCESS!")
        print("📁 Check the following output directories:")
        print("   • intermediate_outputs/layout_outputs/ - Layout detection results")
        print("   • intermediate_outputs/tableformer_outputs/ - Table analysis results")
        print("   • intermediate_outputs/ocr_outputs/ - OCR results")
        print("   • pipe_output/ - Final reconstructed documents")
    else:
        print("❌ PIPELINE FAILED!")
        print("Check the error messages above for details.")
    
    print(f"⏱️  Total pipeline duration: {total_duration:.2f} seconds")
    print(f"{'='*60}")
    
    return 0 if overall_success else 1

if __name__ == "__main__":
    sys.exit(main())
