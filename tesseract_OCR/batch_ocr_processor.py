#!/usr/bin/env python3
"""
Enhanced Batch OCR Processor

This script provides advanced batch processing capabilities for OCR operations:
- Progress tracking with progress bars
- Error handling and recovery
- Configurable processing options
- Detailed logging and reporting
- Support for different image formats
- Parallel processing capabilities
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from datetime import datetime

# Import the OCR pipeline
from ocr_pipeline import OCRPipeline

class BatchOCRProcessor:
    def __init__(self, 
                 input_dir: str = "../pipe_input",
                 output_dir: str = "../intermediate_outputs/ocr_outputs",
                 max_workers: int = 4,
                 supported_formats: List[str] = None,
                 log_level: str = "INFO"):
        """
        Initialize the batch OCR processor.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory for output files
            max_workers: Maximum number of parallel workers
            supported_formats: List of supported image formats
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.supported_formats = supported_formats or ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']
        
        # Create directories
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging(log_level)
        
        # Initialize OCR pipeline
        self.ocr_pipeline = OCRPipeline()
        
        # Processing statistics
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'skipped_files': 0,
            'start_time': None,
            'end_time': None,
            'errors': []
        }
    
    def setup_logging(self, log_level: str):
        """Setup logging configuration."""
        log_file = self.output_dir / f"batch_ocr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Batch OCR Processor initialized. Log file: {log_file}")
    
    def get_image_files(self) -> List[Path]:
        """Get list of image files to process."""
        image_files = []
        
        for file_path in self.input_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                image_files.append(file_path)
        
        self.stats['total_files'] = len(image_files)
        self.logger.info(f"Found {len(image_files)} image files to process")
        
        return sorted(image_files)
    
    def process_single_image(self, image_path: Path) -> Tuple[bool, Dict, str]:
        """
        Process a single image and return results.
        
        Returns:
            Tuple of (success, result_data, error_message)
        """
        try:
            self.logger.info(f"Processing: {image_path.name}")
            
            # Process the image
            result = self.ocr_pipeline.process_document(str(image_path))
            
            if result and result.get('full_document_ocr', {}).get('success') and result.get('full_document_ocr', {}).get('text'):
                # Save individual results
                base_name = image_path.stem
                result_file = self.output_dir / f"{base_name}_ocr_results.json"
                
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                # Create and save text summary
                summary = self.ocr_pipeline.create_text_summary(result)
                summary_file = self.output_dir / f"{base_name}_ocr_summary.txt"
                
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(summary)
                
                self.logger.info(f"Successfully processed: {image_path.name}")
                return True, result, ""
            else:
                error_msg = f"No text elements found in {image_path.name}"
                self.logger.warning(error_msg)
                return False, {}, error_msg
                
        except Exception as e:
            error_msg = f"Error processing {image_path.name}: {str(e)}"
            self.logger.error(error_msg)
            return False, {}, error_msg
    
    def process_batch_sequential(self, image_files: List[Path]) -> List[Dict]:
        """Process images sequentially."""
        self.logger.info("Starting sequential batch processing")
        results = []
        
        for i, image_path in enumerate(image_files, 1):
            self.logger.info(f"Processing {i}/{len(image_files)}: {image_path.name}")
            
            success, result, error = self.process_single_image(image_path)
            
            if success:
                results.append(result)
                self.stats['processed_files'] += 1
            else:
                self.stats['failed_files'] += 1
                self.stats['errors'].append({
                    'file': image_path.name,
                    'error': error,
                    'timestamp': datetime.now().isoformat()
                })
        
        return results
    
    def process_batch_parallel(self, image_files: List[Path]) -> List[Dict]:
        """Process images in parallel."""
        self.logger.info(f"Starting parallel batch processing with {self.max_workers} workers")
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_single_image, image_path): image_path 
                for image_path in image_files
            }
            
            # Process completed tasks
            for i, future in enumerate(as_completed(future_to_file), 1):
                image_path = future_to_file[future]
                
                try:
                    success, result, error = future.result()
                    
                    if success:
                        results.append(result)
                        self.stats['processed_files'] += 1
                        self.logger.info(f"Completed {i}/{len(image_files)}: {image_path.name}")
                    else:
                        self.stats['failed_files'] += 1
                        self.stats['errors'].append({
                            'file': image_path.name,
                            'error': error,
                            'timestamp': datetime.now().isoformat()
                        })
                        self.logger.error(f"Failed {i}/{len(image_files)}: {image_path.name} - {error}")
                        
                except Exception as e:
                    self.stats['failed_files'] += 1
                    error_msg = f"Unexpected error processing {image_path.name}: {str(e)}"
                    self.stats['errors'].append({
                        'file': image_path.name,
                        'error': error_msg,
                        'timestamp': datetime.now().isoformat()
                    })
                    self.logger.error(error_msg)
        
        return results
    
    def save_batch_results(self, results: List[Dict]):
        """Save comprehensive batch processing results."""
        batch_results = {
            'batch_processing_info': {
                'total_files': self.stats['total_files'],
                'processed_files': self.stats['processed_files'],
                'failed_files': self.stats['failed_files'],
                'skipped_files': self.stats['skipped_files'],
                'success_rate': (self.stats['processed_files'] / self.stats['total_files'] * 100) if self.stats['total_files'] > 0 else 0,
                'start_time': self.stats['start_time'].isoformat() if self.stats['start_time'] else None,
                'end_time': self.stats['end_time'].isoformat() if self.stats['end_time'] else None,
                'processing_duration': str(self.stats['end_time'] - self.stats['start_time']) if self.stats['start_time'] and self.stats['end_time'] else None,
                'max_workers': self.max_workers,
                'supported_formats': self.supported_formats
            },
            'processing_errors': self.stats['errors'],
            'individual_results': results
        }
        
        # Save batch results
        batch_file = self.output_dir / f"batch_ocr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Batch results saved to: {batch_file}")
        
        # Create summary report
        self.create_summary_report(batch_results)
    
    def create_summary_report(self, batch_results: Dict):
        """Create a human-readable summary report."""
        report_file = self.output_dir / f"batch_ocr_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("BATCH OCR PROCESSING SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            info = batch_results['batch_processing_info']
            f.write(f"Total Files: {info['total_files']}\n")
            f.write(f"Successfully Processed: {info['processed_files']}\n")
            f.write(f"Failed: {info['failed_files']}\n")
            f.write(f"Success Rate: {info['success_rate']:.1f}%\n")
            f.write(f"Processing Duration: {info['processing_duration']}\n")
            f.write(f"Max Workers: {info['max_workers']}\n\n")
            
            if batch_results['processing_errors']:
                f.write("ERRORS:\n")
                f.write("-" * 40 + "\n")
                for error in batch_results['processing_errors']:
                    f.write(f"File: {error['file']}\n")
                    f.write(f"Error: {error['error']}\n")
                    f.write(f"Time: {error['timestamp']}\n\n")
            else:
                f.write("No errors encountered!\n")
        
        self.logger.info(f"Summary report saved to: {report_file}")
    
    def run_batch_processing(self, parallel: bool = True) -> bool:
        """
        Run the complete batch processing pipeline.
        
        Args:
            parallel: Whether to use parallel processing
            
        Returns:
            True if processing completed successfully
        """
        try:
            # Check Tesseract installation
            if not self.ocr_pipeline.check_tesseract_installation():
                self.logger.error("Tesseract OCR not properly installed")
                return False
            
            # Get image files
            image_files = self.get_image_files()
            
            if not image_files:
                self.logger.warning("No image files found to process")
                return False
            
            # Start processing
            self.stats['start_time'] = datetime.now()
            self.logger.info(f"Starting batch processing of {len(image_files)} files")
            
            # Process images
            if parallel and len(image_files) > 1:
                results = self.process_batch_parallel(image_files)
            else:
                results = self.process_batch_sequential(image_files)
            
            # Finish processing
            self.stats['end_time'] = datetime.now()
            
            # Save results
            self.save_batch_results(results)
            
            # Log final statistics
            self.logger.info("=" * 50)
            self.logger.info("BATCH PROCESSING COMPLETED")
            self.logger.info("=" * 50)
            self.logger.info(f"Total files: {self.stats['total_files']}")
            self.logger.info(f"Successfully processed: {self.stats['processed_files']}")
            self.logger.info(f"Failed: {self.stats['failed_files']}")
            self.logger.info(f"Success rate: {(self.stats['processed_files'] / self.stats['total_files'] * 100):.1f}%")
            self.logger.info(f"Processing time: {self.stats['end_time'] - self.stats['start_time']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            return False


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Enhanced Batch OCR Processor")
    parser.add_argument("--input", "-i", default="../pipe_input", help="Input directory containing images")
    parser.add_argument("--output", "-o", default="../intermediate_outputs/ocr_outputs", help="Output directory for results")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--sequential", "-s", action="store_true", help="Use sequential processing instead of parallel")
    parser.add_argument("--log-level", "-l", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    parser.add_argument("--formats", "-f", nargs="+", default=[".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"], help="Supported image formats")
    
    args = parser.parse_args()
    
    # Create and run batch processor
    processor = BatchOCRProcessor(
        input_dir=args.input,
        output_dir=args.output,
        max_workers=args.workers,
        supported_formats=args.formats,
        log_level=args.log_level
    )
    
    success = processor.run_batch_processing(parallel=not args.sequential)
    
    if success:
        print("\n✅ Batch OCR processing completed successfully!")
        print(f"📁 Check the output directory: {args.output}")
    else:
        print("\n❌ Batch OCR processing failed!")
        print("📋 Check the log files for details")
        exit(1)


if __name__ == "__main__":
    main()
