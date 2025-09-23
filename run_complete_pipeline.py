#!/usr/bin/env python3
"""
Master Pipeline Runner
=====================

This script runs the complete OCR pipeline in sequence:
1. OCR Text Extraction
2. Layout & Table Detection  
3. Document Reconstruction

Usage:
    python run_complete_pipeline.py [--skip-ocr] [--skip-layout] [--skip-reconstruction]

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

def check_environment():
    """Check if we're in the right environment and directories exist."""
    print("🔍 Checking environment...")
    
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
    
    print("✅ Environment check passed!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Run complete OCR pipeline")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip OCR text extraction stage")
    parser.add_argument("--skip-layout", action="store_true", help="Skip layout and table detection stage")
    parser.add_argument("--skip-reconstruction", action="store_true", help="Skip document reconstruction stage")
    
    args = parser.parse_args()
    
    print("🎯 OCR Pipeline Master Runner")
    print("=" * 60)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Track overall progress
    total_start_time = time.time()
    stages_completed = 0
    total_stages = 3
    
    # Stage 1: OCR Text Extraction
    if not args.skip_ocr:
        success = run_command(
            "python ocr_pipeline.py",
            "Stage 1: OCR Text Extraction",
            cwd="tesseract_OCR"
        )
        if success:
            stages_completed += 1
        else:
            print("\n❌ Pipeline failed at OCR stage. Stopping.")
            sys.exit(1)
    else:
        print("\n⏭️  Skipping OCR Text Extraction stage")
    
    # Stage 2: Layout & Table Detection
    if not args.skip_layout:
        success = run_command(
            "python batch_pipeline.py",
            "Stage 2: Layout & Table Detection",
            cwd="docling-ibm-models/batch_processing"
        )
        if success:
            stages_completed += 1
        else:
            print("\n❌ Pipeline failed at Layout/Table stage. Stopping.")
            sys.exit(1)
    else:
        print("\n⏭️  Skipping Layout & Table Detection stage")
    
    # Stage 3: Document Reconstruction
    if not args.skip_reconstruction:
        success = run_command(
            "python batch_integrated_visualization.py",
            "Stage 3: Document Reconstruction",
            cwd="reconstruction"
        )
        if success:
            stages_completed += 1
        else:
            print("\n❌ Pipeline failed at Reconstruction stage.")
            sys.exit(1)
    else:
        print("\n⏭️  Skipping Document Reconstruction stage")
    
    # Final summary
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print(f"\n{'='*60}")
    print("🎉 PIPELINE COMPLETE!")
    print(f"{'='*60}")
    print(f"✅ Stages completed: {stages_completed}/{total_stages}")
    print(f"⏱️  Total duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    
    if stages_completed == total_stages:
        print("\n📁 Output locations:")
        print("   • OCR results: intermediate_outputs/ocr_outputs/")
        print("   • Layout results: intermediate_outputs/layout_outputs/")
        print("   • Table results: intermediate_outputs/tableformer_outputs/")
        print("   • Final PDFs: intermediate_outputs/batch_integrated_visualizations/")
        print("\n🎯 All 6 images processed successfully!")
    else:
        print(f"\n⚠️  Pipeline completed with {total_stages - stages_completed} stages skipped.")

if __name__ == "__main__":
    main()