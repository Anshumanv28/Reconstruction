#!/usr/bin/env python3
"""
Test script for Batch OCR Processor

This script demonstrates how to use the batch OCR processor
with different configurations and options.
"""

import os
import sys
from pathlib import Path
from batch_ocr_processor import BatchOCRProcessor

def test_batch_processing():
    """Test the batch OCR processing functionality."""
    
    print("🔍 Testing Batch OCR Processor")
    print("=" * 50)
    
    # Initialize processor
    processor = BatchOCRProcessor(
        input_dir="input",
        output_dir="output",
        max_workers=2,  # Use fewer workers for testing
        log_level="INFO"
    )
    
    # Check if input directory has images
    image_files = processor.get_image_files()
    
    if not image_files:
        print("❌ No images found in input directory")
        print("📁 Please add some image files to the 'input' directory")
        return False
    
    print(f"📸 Found {len(image_files)} images to process:")
    for img_file in image_files:
        print(f"   - {img_file.name}")
    
    print("\n🚀 Starting batch processing...")
    
    # Run batch processing
    success = processor.run_batch_processing(parallel=True)
    
    if success:
        print("\n✅ Batch processing completed successfully!")
        print("📊 Check the output directory for results and logs")
        return True
    else:
        print("\n❌ Batch processing failed!")
        return False

def test_sequential_processing():
    """Test sequential processing mode."""
    
    print("\n🔍 Testing Sequential OCR Processing")
    print("=" * 50)
    
    # Initialize processor
    processor = BatchOCRProcessor(
        input_dir="input",
        output_dir="output_sequential",
        max_workers=1,
        log_level="INFO"
    )
    
    # Run sequential processing
    success = processor.run_batch_processing(parallel=False)
    
    if success:
        print("\n✅ Sequential processing completed successfully!")
        return True
    else:
        print("\n❌ Sequential processing failed!")
        return False

def main():
    """Main test function."""
    
    print("🧪 Batch OCR Processor Test Suite")
    print("=" * 60)
    
    # Test 1: Parallel batch processing
    print("\n1️⃣ Testing Parallel Batch Processing")
    parallel_success = test_batch_processing()
    
    # Test 2: Sequential processing
    print("\n2️⃣ Testing Sequential Processing")
    sequential_success = test_sequential_processing()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 30)
    print(f"Parallel Processing: {'✅ PASS' if parallel_success else '❌ FAIL'}")
    print(f"Sequential Processing: {'✅ PASS' if sequential_success else '❌ FAIL'}")
    
    if parallel_success and sequential_success:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️ Some tests failed. Check the logs for details.")
        return 1

if __name__ == "__main__":
    exit(main())
