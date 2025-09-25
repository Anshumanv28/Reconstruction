#!/usr/bin/env python3
"""
Pipeline Runner - Easy Access to All Pipeline Modes

This script provides convenient access to all pipeline execution modes:
1. Single stage with single input
2. All stages with single input  
3. Single stage with multiple inputs
4. Complete pipeline with multiple inputs

Usage Examples:
    python run_pipeline.py --help                           # Show all options
    python run_pipeline.py --stage 2 --input file.png      # Single stage, single input
    python run_pipeline.py --all-stages --input file.png   # All stages, single input
    python run_pipeline.py --stage 2 --multi               # Single stage, multi input
    python run_pipeline.py --pipeline --multi              # Complete pipeline, multi input
"""

import argparse
import sys
import subprocess
from pathlib import Path
from typing import List, Optional

class PipelineRunner:
    """Convenient wrapper for running the OCR pipeline in different modes"""
    
    def __init__(self):
        self.pipeline_script = "pipeline_orchestrator.py"
        self.input_dir = "pipe_input"
        
    def run_command(self, args: List[str]) -> bool:
        """Run a command and return success status"""
        try:
            print(f"🚀 Running: python {self.pipeline_script} {' '.join(args)}")
            print("=" * 80)
            
            result = subprocess.run(
                [sys.executable, self.pipeline_script] + args,
                check=True,
                capture_output=False
            )
            
            print("=" * 80)
            print("✅ Command completed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print("=" * 80)
            print(f"❌ Command failed with exit code {e.returncode}")
            return False
        except FileNotFoundError:
            print(f"❌ Error: {self.pipeline_script} not found!")
            return False
    
    def list_available_files(self) -> List[str]:
        """List available input files"""
        input_path = Path(self.input_dir)
        if not input_path.exists():
            return []
        
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']:
            image_files.extend(input_path.glob(ext))
        
        return [f.name for f in image_files]
    
    def run_single_stage_single_input(self, stage: int, input_file: str) -> bool:
        """Run a single stage with a single input file"""
        if not Path(input_file).exists():
            print(f"❌ Error: Input file '{input_file}' not found!")
            return False
        
        args = ["--single-stage", str(stage), "--input", input_file]
        return self.run_command(args)
    
    def run_all_stages_single_input(self, input_file: str) -> bool:
        """Run all stages with a single input file"""
        if not Path(input_file).exists():
            print(f"❌ Error: Input file '{input_file}' not found!")
            return False
        
        args = ["--all-stages", "--input", input_file]
        return self.run_command(args)
    
    def run_single_stage_multi_input(self, stage: int) -> bool:
        """Run a single stage with multiple input files"""
        available_files = self.list_available_files()
        if not available_files:
            print(f"❌ Error: No image files found in '{self.input_dir}' directory!")
            return False
        
        print(f"📁 Found {len(available_files)} files in {self.input_dir}:")
        for i, file in enumerate(available_files, 1):
            print(f"   {i}. {file}")
        
        print(f"\n🔄 Running Stage {stage} on {len(available_files)} files...")
        
        success_count = 0
        for i, file in enumerate(available_files, 1):
            input_path = Path(self.input_dir) / file
            print(f"\n--- Processing {i}/{len(available_files)}: {file} ---")
            
            args = ["--single-stage", str(stage), "--input", str(input_path)]
            if self.run_command(args):
                success_count += 1
        
        print(f"\n📊 Results: {success_count}/{len(available_files)} files processed successfully")
        return success_count == len(available_files)
    
    def run_pipeline_multi_input(self) -> bool:
        """Run complete pipeline with multiple input files"""
        available_files = self.list_available_files()
        if not available_files:
            print(f"❌ Error: No image files found in '{self.input_dir}' directory!")
            return False
        
        print(f"📁 Found {len(available_files)} files in {self.input_dir}:")
        for i, file in enumerate(available_files, 1):
            print(f"   {i}. {file}")
        
        args = ["--whole-pipeline-multi"]
        return self.run_command(args)
    
    def show_stage_info(self):
        """Show information about each pipeline stage"""
        print("\n📋 Pipeline Stages:")
        print("   1. Layout & Table Detection    - Detects document structure and tables")
        print("   2. OCR Character Extraction    - Extracts text using Tesseract OCR")
        print("   3. Grid Reconstruction         - Creates median-based table grids")
        print("   4. Complete File Reconstruction - Generates final HTML/Markdown/JSON")
        print()
    
    def show_usage_examples(self):
        """Show usage examples"""
        print("\n💡 Usage Examples:")
        print("   # Run Stage 2 (OCR) on a single file:")
        print("   python run_pipeline.py --stage 2 --input pipe_input/meeting.png")
        print()
        print("   # Run all stages on a single file:")
        print("   python run_pipeline.py --all-stages --input pipe_input/meeting.png")
        print()
        print("   # Run Stage 3 (Grid) on all files in pipe_input:")
        print("   python run_pipeline.py --stage 3 --multi")
        print()
        print("   # Run complete pipeline on all files:")
        print("   python run_pipeline.py --pipeline --multi")
        print()

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Easy Pipeline Runner - Access all pipeline modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --stage 2 --input file.png          # Single stage, single input
  %(prog)s --all-stages --input file.png       # All stages, single input  
  %(prog)s --stage 2 --multi                   # Single stage, multi input
  %(prog)s --pipeline --multi                  # Complete pipeline, multi input
  %(prog)s --info                               # Show stage information
  %(prog)s --examples                           # Show usage examples
        """
    )
    
    # Execution modes (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--stage", 
        type=int, 
        choices=[1, 2, 3, 4],
        help="Run a single stage (1-4)"
    )
    mode_group.add_argument(
        "--all-stages", 
        action="store_true",
        help="Run all stages sequentially"
    )
    mode_group.add_argument(
        "--pipeline", 
        action="store_true",
        help="Run complete pipeline (same as --all-stages)"
    )
    mode_group.add_argument(
        "--info", 
        action="store_true",
        help="Show pipeline stage information"
    )
    mode_group.add_argument(
        "--examples", 
        action="store_true",
        help="Show usage examples"
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input", 
        type=str,
        help="Single input file path"
    )
    input_group.add_argument(
        "--multi", 
        action="store_true",
        help="Process all files in pipe_input directory"
    )
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = PipelineRunner()
    
    # Handle info and examples
    if args.info:
        runner.show_stage_info()
        return
    
    if args.examples:
        runner.show_usage_examples()
        return
    
    # Validate arguments for execution modes
    if args.stage or args.all_stages or args.pipeline:
        if not args.input and not args.multi:
            print("❌ Error: Must specify either --input <file> or --multi")
            print("   Use --help for more information")
            return
    
    # Execute based on mode
    success = False
    
    if args.stage:
        if args.input:
            success = runner.run_single_stage_single_input(args.stage, args.input)
        elif args.multi:
            success = runner.run_single_stage_multi_input(args.stage)
    
    elif args.all_stages or args.pipeline:
        if args.input:
            success = runner.run_all_stages_single_input(args.input)
        elif args.multi:
            success = runner.run_pipeline_multi_input()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
