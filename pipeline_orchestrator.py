#!/usr/bin/env python3
"""
OCR Pipeline Orchestrator

4-Stage Pipeline:
1. Docling Layout & Table Detection
2. Tesseract OCR Character Extraction  
3. Grid Reconstruction (Median-based)
4. Complete File Reconstruction

Each stage produces:
- Visualization output (PNG/PDF)
- Coordinate JSON (structured data)

Test Modes:
- Single stage with single input
- All stages with single input
- Whole pipeline with single input
- Whole pipeline with multiple inputs
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
import time

class PipelineOrchestrator:
    """Orchestrates the 4-stage OCR pipeline"""
    
    def __init__(self, intermediate_dir: str = "intermediate_outputs"):
        self.intermediate_dir = Path(intermediate_dir)
        self.stages = {
            1: "layout_table_detection",
            2: "ocr_extraction", 
            3: "grid_reconstruction",
            4: "file_reconstruction"
        }
        
        # Create intermediate directory structure
        self._create_directory_structure()
    
    def _create_directory_structure(self):
        """Create the intermediate outputs directory structure"""
        subdirs = [
            "stage1_layout_table",
            "stage2_ocr_extraction", 
            "stage3_grid_reconstruction",
            "stage4_file_reconstruction"
        ]
        
        for subdir in subdirs:
            (self.intermediate_dir / subdir).mkdir(parents=True, exist_ok=True)
            (self.intermediate_dir / subdir / "visualizations").mkdir(exist_ok=True)
            (self.intermediate_dir / subdir / "coordinates").mkdir(exist_ok=True)
    
    def run_stage1_layout_table_detection(self, input_file: str, output_prefix: str) -> Dict:
        """Run Stage 1: Docling Layout & Table Detection"""
        print(f"\n{'='*60}")
        print(f"STAGE 1: LAYOUT & TABLE DETECTION")
        print(f"{'='*60}")
        print(f"Input: {input_file}")
        print(f"Output prefix: {output_prefix}")
        
        # Import and run the layout/table detection
        try:
            from pipeline_stages.docling_layout_table import LayoutTableDetector
            
            detector = LayoutTableDetector()
            result = detector.process_image(input_file, output_prefix, self.intermediate_dir)
            
            print(f"Stage 1 completed successfully")
            print(f"   Layout elements: {result.get('layout_count', 0)}")
            print(f"   Tables detected: {result.get('table_count', 0)}")
            print(f"   Visualization: {result.get('visualization_path', 'N/A')}")
            print(f"   Coordinates: {result.get('coordinates_path', 'N/A')}")
            
            return result
            
        except Exception as e:
            print(f"Stage 1 failed: {e}")
            return {"success": False, "error": str(e)}
    
    def run_stage2_ocr_extraction(self, input_file: str, output_prefix: str) -> Dict:
        """Run Stage 2: Tesseract OCR Character Extraction"""
        print(f"\n{'='*60}")
        print(f"STAGE 2: OCR CHARACTER EXTRACTION")
        print(f"{'='*60}")
        print(f"Input: {input_file}")
        print(f"Output prefix: {output_prefix}")
        
        try:
            from pipeline_stages.tesseract_ocr_extraction import TesseractOCRExtractor
            
            extractor = TesseractOCRExtractor()
            result = extractor.process_image(input_file, output_prefix, self.intermediate_dir)
            
            print(f"SUCCESS Stage 2 completed successfully")
            print(f"   📝 Text blocks: {result.get('text_block_count', 0)}")
            print(f"   VIZ  Visualization: {result.get('visualization_path', 'N/A')}")
            print(f"   FILES Coordinates: {result.get('coordinates_path', 'N/A')}")
            
            return result
            
        except Exception as e:
            print(f"ERROR Stage 2 failed: {e}")
            return {"success": False, "error": str(e)}
    
    def run_stage3_grid_reconstruction(self, input_file: str, output_prefix: str) -> Dict:
        """Run Stage 3: Grid Reconstruction (Median-based)"""
        print(f"\n{'='*60}")
        print(f"STAGE 3: GRID RECONSTRUCTION")
        print(f"{'='*60}")
        print(f"Input: {input_file}")
        print(f"Output prefix: {output_prefix}")
        
        try:
            from pipeline_stages.grid_reconstruction import GridReconstructor
            
            reconstructor = GridReconstructor()
            result = reconstructor.process_image(input_file, output_prefix, self.intermediate_dir)
            
            print(f"SUCCESS Stage 3 completed successfully")
            print(f"   INFO Tables processed: {result.get('table_count', 0)}")
            print(f"   TABLES Grid cells: {result.get('total_cells', 0)}")
            print(f"   VIZ  Visualization: {result.get('visualization_path', 'N/A')}")
            print(f"   FILES Coordinates: {result.get('coordinates_path', 'N/A')}")
            
            return result
            
        except Exception as e:
            print(f"ERROR Stage 3 failed: {e}")
            return {"success": False, "error": str(e)}
    
    def run_stage4_file_reconstruction(self, input_file: str, output_prefix: str) -> Dict:
        """Run Stage 4: Complete File Reconstruction"""
        print(f"\n{'='*60}")
        print(f"STAGE 4: COMPLETE FILE RECONSTRUCTION")
        print(f"{'='*60}")
        print(f"Input: {input_file}")
        print(f"Output prefix: {output_prefix}")
        
        try:
            from pipeline_stages.file_reconstruction import FileReconstructor
            
            reconstructor = FileReconstructor()
            result = reconstructor.process_image(input_file, output_prefix, self.intermediate_dir)
            
            print(f"SUCCESS Stage 4 completed successfully")
            print(f"   FILES Output formats: {result.get('output_formats', [])}")
            print(f"   VIZ  Visualization: {result.get('visualization_path', 'N/A')}")
            print(f"   FILES Coordinates: {result.get('coordinates_path', 'N/A')}")
            
            return result
            
        except Exception as e:
            print(f"ERROR Stage 4 failed: {e}")
            return {"success": False, "error": str(e)}
    
    def run_single_stage(self, stage: int, input_path: str) -> Dict:
        """Run a single stage of the pipeline on a single file or directory of files"""
        if stage not in self.stages:
            return {"success": False, "error": f"Invalid stage: {stage}"}
        
        input_path_obj = Path(input_path)
        if not input_path_obj.exists():
            return {"success": False, "error": f"Input path not found: {input_path}"}
        
        # Check if input is a directory
        if input_path_obj.is_dir():
            return self._run_single_stage_on_directory(stage, input_path)
        else:
            return self._run_single_stage_on_file(stage, input_path)
    
    def _run_single_stage_on_file(self, stage: int, input_file: str) -> Dict:
        """Run a single stage on a single file"""
        input_path = Path(input_file)
        output_prefix = input_path.stem
        
        print(f"RUNNING STAGE {stage}: {self.stages[stage].upper()}")
        print(f"Input file: {input_file}")
        
        start_time = time.time()
        
        if stage == 1:
            result = self.run_stage1_layout_table_detection(input_file, output_prefix)
        elif stage == 2:
            result = self.run_stage2_ocr_extraction(input_file, output_prefix)
        elif stage == 3:
            result = self.run_stage3_grid_reconstruction(input_file, output_prefix)
        elif stage == 4:
            result = self.run_stage4_file_reconstruction(input_file, output_prefix)
        
        end_time = time.time()
        result["execution_time"] = end_time - start_time
        
        return result
    
    def _run_single_stage_on_directory(self, stage: int, input_dir: str) -> Dict:
        """Run a single stage on all files in a directory"""
        print(f"\n{'='*80}")
        print(f"RUNNING STAGE {stage} ON DIRECTORY: {self.stages[stage].upper()}")
        print(f"{'='*80}")
        print(f"Input directory: {input_dir}")
        
        input_path = Path(input_dir)
        
        # Find all image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
        image_files = []
        
        for file_path in input_path.iterdir():
            if file_path.suffix.lower() in image_extensions:
                image_files.append(str(file_path))
        
        if not image_files:
            return {"success": False, "error": "No image files found in input directory"}
        
        print(f"Found {len(image_files)} image files to process")
        
        results = {}
        overall_start_time = time.time()
        successful = 0
        failed = 0
        
        for i, image_file in enumerate(image_files, 1):
            print(f"\n--- Processing {i}/{len(image_files)}: {Path(image_file).name} ---")
            
            file_result = self._run_single_stage_on_file(stage, image_file)
            results[Path(image_file).name] = file_result
            
            if file_result.get("success", False):
                successful += 1
            else:
                failed += 1
        
        overall_end_time = time.time()
        
        return {
            "success": failed == 0,
            "total_files": len(image_files),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(image_files) * 100,
            "total_execution_time": overall_end_time - overall_start_time,
            "file_results": results
        }
    
    def run_all_stages_single_input(self, input_file: str) -> Dict:
        """Run all stages with a single input file"""
        print(f"\n{'='*80}")
        print(f"RUNNING ALL STAGES WITH SINGLE INPUT")
        print(f"{'='*80}")
        print(f"Input file: {input_file}")
        
        input_path = Path(input_file)
        if not input_path.exists():
            return {"success": False, "error": f"Input file not found: {input_file}"}
        
        output_prefix = input_path.stem
        results = {}
        overall_start_time = time.time()
        
        # Run each stage
        for stage_num in range(1, 5):
            stage_result = self.run_single_stage(stage_num, input_file)
            results[f"stage_{stage_num}"] = stage_result
            
            if not stage_result.get("success", False):
                print(f"ERROR Pipeline failed at stage {stage_num}")
                break
        
        overall_end_time = time.time()
        
        # Check if all stages succeeded
        all_success = all(result.get("success", False) for result in results.values())
        
        return {
            "success": all_success,
            "stages": results,
            "total_execution_time": overall_end_time - overall_start_time,
            "input_file": input_file,
            "output_prefix": output_prefix
        }
    
    def run_whole_pipeline_single_input(self, input_file: str) -> Dict:
        """Run the complete pipeline with a single input file"""
        print(f"\n{'='*80}")
        print(f"RUNNING COMPLETE PIPELINE WITH SINGLE INPUT")
        print(f"{'='*80}")
        print(f"Input file: {input_file}")
        
        # This is the same as run_all_stages_single_input for now
        # but could include additional pipeline-level processing
        return self.run_all_stages_single_input(input_file)
    
    def run_whole_pipeline_multi_input(self, input_dir: str) -> Dict:
        """Run the complete pipeline with multiple input files"""
        print(f"\n{'='*80}")
        print(f"RUNNING COMPLETE PIPELINE WITH MULTIPLE INPUTS")
        print(f"{'='*80}")
        print(f"Input directory: {input_dir}")
        
        input_path = Path(input_dir)
        if not input_path.exists():
            return {"success": False, "error": f"Input directory not found: {input_dir}"}
        
        # Find all image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
        image_files = []
        
        for file_path in input_path.iterdir():
            if file_path.suffix.lower() in image_extensions:
                image_files.append(str(file_path))
        
        if not image_files:
            return {"success": False, "error": "No image files found in input directory"}
        
        print(f"Found {len(image_files)} image files to process")
        
        results = {}
        overall_start_time = time.time()
        successful = 0
        failed = 0
        
        for i, image_file in enumerate(image_files, 1):
            print(f"\n--- Processing {i}/{len(image_files)}: {Path(image_file).name} ---")
            
            file_result = self.run_whole_pipeline_single_input(image_file)
            results[Path(image_file).name] = file_result
            
            if file_result.get("success", False):
                successful += 1
            else:
                failed += 1
        
        overall_end_time = time.time()
        
        return {
            "success": failed == 0,
            "total_files": len(image_files),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(image_files) * 100,
            "total_execution_time": overall_end_time - overall_start_time,
            "file_results": results
        }

def main():
    """Main function for pipeline orchestrator"""
    parser = argparse.ArgumentParser(description="OCR Pipeline Orchestrator")
    
    # Test mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--single-stage", type=int, choices=[1,2,3,4], 
                          help="Run single stage (1-4) on file or directory")
    mode_group.add_argument("--all-stages", action="store_true",
                          help="Run all stages with single input")
    mode_group.add_argument("--whole-pipeline-single", action="store_true",
                          help="Run whole pipeline with single input")
    mode_group.add_argument("--whole-pipeline-multi", action="store_true",
                          help="Run whole pipeline with multiple inputs")
    
    # Input specification
    parser.add_argument("--input", "-i", 
                      help="Input file or directory (single-stage handles both, all-stages/whole-pipeline-single expects file, whole-pipeline-multi expects directory)")
    parser.add_argument("--intermediate-dir", default="intermediate_outputs",
                      help="Intermediate outputs directory")
    
    args = parser.parse_args()
    
    # Validate input requirements
    if args.whole_pipeline_multi:
        # For multi-input mode, use default input directory if not specified
        if not args.input:
            args.input = "pipe_input"
    else:
        # For all other modes, input is required
        if not args.input:
            parser.error("--input is required for single stage, all stages, and whole pipeline single modes")
    
    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(args.intermediate_dir)
    
    # Run based on mode
    if args.single_stage:
        result = orchestrator.run_single_stage(args.single_stage, args.input)
    elif args.all_stages:
        result = orchestrator.run_all_stages_single_input(args.input)
    elif args.whole_pipeline_single:
        result = orchestrator.run_whole_pipeline_single_input(args.input)
    elif args.whole_pipeline_multi:
        result = orchestrator.run_whole_pipeline_multi_input(args.input)
    
    # Print final results
    print(f"\n{'='*80}")
    print(f"PIPELINE EXECUTION COMPLETE")
    print(f"{'='*80}")
    
    if result.get("success", False):
        print("SUCCESS Pipeline completed successfully!")
    else:
        print("ERROR Pipeline failed!")
        if "error" in result:
            print(f"Error: {result['error']}")
    
    if "total_execution_time" in result:
        print(f"Total execution time: {result['total_execution_time']:.2f} seconds")
    
    # Save results summary
    results_file = orchestrator.intermediate_dir / "pipeline_results.json"
    with open(results_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()
