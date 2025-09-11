#!/usr/bin/env python3
"""
Document Reconstruction Pipeline

This module orchestrates the three-stage document reconstruction process:
1. Table Reconstruction (using TableFormer output)
2. Layout Reconstruction (using Layout output) 
3. OCR Integration (combining OCR text with both structures)
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Import our reconstruction modules
from table_reconstruction import TableReconstructor
from layout_reconstruction import LayoutReconstructor
from ocr_integration import OCRIntegrator


class DocumentReconstructionPipeline:
    """Orchestrates the complete document reconstruction pipeline."""
    
    def __init__(self, 
                 background_color: Tuple[int, int, int] = (255, 255, 255),
                 text_color: Tuple[int, int, int] = (0, 0, 0),
                 grid_size: int = 20):
        """Initialize the reconstruction pipeline."""
        self.background_color = background_color
        self.text_color = text_color
        self.grid_size = grid_size
        
        # Initialize the three reconstruction modules
        self.table_reconstructor = TableReconstructor(background_color, text_color, grid_size)
        self.layout_reconstructor = LayoutReconstructor(background_color, text_color, grid_size)
        self.ocr_integrator = OCRIntegrator(background_color, text_color, grid_size)
    
    def load_specific_image_data(self, base_name: str, intermediate_dir: str = "../intermediate_outputs") -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Load layout, tableformer, and OCR data for a specific image."""
        intermediate_path = Path(intermediate_dir)
        
        # Load layout data
        layout_file = intermediate_path / "layout_outputs" / f"{base_name}_layout_predictions.json"
        if not layout_file.exists():
            raise FileNotFoundError(f"Layout file not found: {layout_file}")
        
        layout_elements = self.layout_reconstructor.load_layout_data(str(layout_file))
        print(f"Loaded {len(layout_elements)} layout elements from {layout_file}")
        
        # Load tableformer data
        tableformer_file = intermediate_path / "tableformer_outputs" / f"{base_name}_tableformer_results.json"
        if not tableformer_file.exists():
            raise FileNotFoundError(f"TableFormer file not found: {tableformer_file}")
        
        tables = self.table_reconstructor.load_tableformer_data(str(tableformer_file))
        print(f"Loaded {len(tables)} tables from {tableformer_file}")
        
        # Load OCR data
        ocr_file = intermediate_path / "ocr_outputs" / f"{base_name}_ocr_results.json"
        if not ocr_file.exists():
            raise FileNotFoundError(f"OCR file not found: {ocr_file}")
        
        ocr_blocks = self.ocr_integrator.load_ocr_data(str(ocr_file))
        print(f"Loaded {len(ocr_blocks)} OCR blocks from {ocr_file}")
        
        return layout_elements, tables, ocr_blocks
    
    def run_stage1_table_reconstruction(self, tables: List[Dict], output_dir: str, base_name: str) -> str:
        """Run Stage 1: Table Reconstruction."""
        print("\n=== STAGE 1: TABLE RECONSTRUCTION ===")
        
        # Reconstruct tables
        table_canvas = self.table_reconstructor.reconstruct_tables(tables)
        
        if table_canvas:
            # Save table reconstruction
            table_output = Path(output_dir) / f"{base_name}_stage1_tables.pdf"
            self.table_reconstructor.save_table_reconstruction(table_canvas, str(table_output))
            print(f"Stage 1 completed: Table reconstruction saved to {table_output}")
            return str(table_output)
        else:
            print("Stage 1 failed: No tables found")
            return None
    
    def run_stage2_layout_reconstruction(self, layout_elements: List[Dict], output_dir: str, base_name: str) -> str:
        """Run Stage 2: Layout Reconstruction."""
        print("\n=== STAGE 2: LAYOUT RECONSTRUCTION ===")
        
        # Reconstruct layout
        layout_canvas = self.layout_reconstructor.reconstruct_layout(layout_elements)
        
        if layout_canvas:
            # Save layout reconstruction
            layout_output = Path(output_dir) / f"{base_name}_stage2_layout.pdf"
            self.layout_reconstructor.save_layout_reconstruction(layout_canvas, str(layout_output))
            print(f"Stage 2 completed: Layout reconstruction saved to {layout_output}")
            return str(layout_output)
        else:
            print("Stage 2 failed: No layout elements found")
            return None
    
    def run_stage3_ocr_integration(self, layout_elements: List[Dict], tables: List[Dict], ocr_blocks: List[Dict], output_dir: str, base_name: str) -> str:
        """Run Stage 3: OCR Integration."""
        print("\n=== STAGE 3: OCR INTEGRATION ===")
        
        # Integrate OCR with structures
        integrated_canvas = self.ocr_integrator.integrate_ocr_with_structures(layout_elements, tables, ocr_blocks)
        
        if integrated_canvas:
            # Save integrated reconstruction
            integrated_output = Path(output_dir) / f"{base_name}_stage3_integrated.pdf"
            self.ocr_integrator.save_integrated_reconstruction(integrated_canvas, str(integrated_output))
            print(f"Stage 3 completed: OCR integration saved to {integrated_output}")
            return str(integrated_output)
        else:
            print("Stage 3 failed: No OCR blocks found")
            return None
    
    def run_complete_pipeline(self, base_name: str, intermediate_dir: str = "../intermediate_outputs", output_dir: str = "../pipe_output") -> Dict[str, str]:
        """Run the complete three-stage reconstruction pipeline."""
        print(f"=== DOCUMENT RECONSTRUCTION PIPELINE: {base_name} ===")
        print(f"Loading from: {intermediate_dir}")
        print(f"Output to: {output_dir}")
        
        # Load all data
        layout_elements, tables, ocr_blocks = self.load_specific_image_data(base_name, intermediate_dir)
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Stage 1: Table Reconstruction
        table_output = self.run_stage1_table_reconstruction(tables, output_dir, base_name)
        if table_output:
            results['stage1_tables'] = table_output
        
        # Stage 2: Layout Reconstruction
        layout_output = self.run_stage2_layout_reconstruction(layout_elements, output_dir, base_name)
        if layout_output:
            results['stage2_layout'] = layout_output
        
        # Stage 3: OCR Integration
        integrated_output = self.run_stage3_ocr_integration(layout_elements, tables, ocr_blocks, output_dir, base_name)
        if integrated_output:
            results['stage3_integrated'] = integrated_output
        
        # Summary
        print(f"\n=== PIPELINE COMPLETED ===")
        print(f"Results:")
        for stage, output_path in results.items():
            print(f"  {stage}: {output_path}")
        
        return results


def main():
    """Main function for the reconstruction pipeline."""
    parser = argparse.ArgumentParser(description="Run complete document reconstruction pipeline")
    parser.add_argument("--image", required=True, help="Image name (without extension)")
    parser.add_argument("--intermediate-dir", default="../intermediate_outputs", help="Intermediate outputs directory")
    parser.add_argument("--output-dir", default="../pipe_output", help="Output directory")
    parser.add_argument("--stage", choices=['1', '2', '3', 'all'], default='all', help="Which stage to run (1=tables, 2=layout, 3=integration, all=complete pipeline)")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DocumentReconstructionPipeline()
    
    if args.stage == 'all':
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(args.image, args.intermediate_dir, args.output_dir)
        
        if results:
            print(f"\nPipeline completed successfully!")
            print(f"Final integrated result: {results.get('stage3_integrated', 'Not available')}")
        else:
            print(f"\nPipeline failed - no results generated")
    
    else:
        # Run specific stage
        layout_elements, tables, ocr_blocks = pipeline.load_specific_image_data(args.image, args.intermediate_dir)
        
        if args.stage == '1':
            pipeline.run_stage1_table_reconstruction(tables, args.output_dir, args.image)
        elif args.stage == '2':
            pipeline.run_stage2_layout_reconstruction(layout_elements, args.output_dir, args.image)
        elif args.stage == '3':
            pipeline.run_stage3_ocr_integration(layout_elements, tables, ocr_blocks, args.output_dir, args.image)


if __name__ == "__main__":
    main()
