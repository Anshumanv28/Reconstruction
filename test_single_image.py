#!/usr/bin/env python3
"""
Simple Single Image Test Script
==============================

This script tests the pipeline on a single image by temporarily moving other images
and then running the batch pipeline.

Usage:
    python test_single_image.py <image_name>

Example:
    python test_single_image.py 1zM9MmA6dM2_2dbMHiJKd_m-FRdPpNTR3lJUT_P1QuiE.png
"""

import os
import shutil
import sys
import subprocess
from pathlib import Path

def test_single_image(image_name):
    """Test pipeline on a single image."""
    
    print(f"🎯 Testing pipeline on: {image_name}")
    
    # Check if image exists
    image_path = Path("pipe_input") / image_name
    if not image_path.exists():
        print(f"❌ Image '{image_name}' not found in pipe_input directory.")
        print("Available images:")
        for img in Path("pipe_input").glob("*.png"):
            print(f"   • {img.name}")
        return False
    
    # Create backup directory
    backup_dir = Path("pipe_input_backup")
    backup_dir.mkdir(exist_ok=True)
    
    # Move all other images to backup
    moved_images = []
    for img in Path("pipe_input").glob("*.png"):
        if img.name != image_name:
            shutil.move(str(img), str(backup_dir / img.name))
            moved_images.append(img.name)
    
    print(f"📁 Moved {len(moved_images)} other images to backup")
    
    try:
        # Run the complete pipeline
        print("\n🚀 Running complete pipeline...")
        result = subprocess.run(["python", "run_complete_pipeline.py"], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print(f"\n✅ Pipeline completed successfully for {image_name}!")
            print("\n📁 Output files:")
            base_name = Path(image_name).stem
            
            # Check for output files
            ocr_file = f"intermediate_outputs/ocr_outputs/{base_name}_ocr_results.json"
            layout_file = f"intermediate_outputs/layout_outputs/{base_name}_layout_predictions.json"
            table_file = f"intermediate_outputs/tableformer_outputs/{base_name}_tableformer_results.json"
            pdf_file = f"intermediate_outputs/batch_integrated_visualizations/{base_name}_integrated_visualization.pdf"
            
            if Path(ocr_file).exists():
                print(f"   ✅ OCR: {ocr_file}")
            if Path(layout_file).exists():
                print(f"   ✅ Layout: {layout_file}")
            if Path(table_file).exists():
                print(f"   ✅ Table: {table_file}")
            if Path(pdf_file).exists():
                print(f"   ✅ PDF: {pdf_file}")
            
            return True
        else:
            print(f"\n❌ Pipeline failed for {image_name}")
            return False
            
    finally:
        # Restore all images
        print(f"\n🔄 Restoring {len(moved_images)} images...")
        for img_name in moved_images:
            shutil.move(str(backup_dir / img_name), str(Path("pipe_input") / img_name))
        
        # Remove backup directory
        backup_dir.rmdir()
        print("✅ Images restored")

def main():
    if len(sys.argv) != 2:
        print("Usage: python test_single_image.py <image_name>")
        print("\nAvailable images:")
        for img in Path("pipe_input").glob("*.png"):
            print(f"   • {img.name}")
        sys.exit(1)
    
    image_name = sys.argv[1]
    success = test_single_image(image_name)
    
    if success:
        print(f"\n🎉 Test completed successfully for {image_name}!")
    else:
        print(f"\n❌ Test failed for {image_name}")
        sys.exit(1)

if __name__ == "__main__":
    main()
