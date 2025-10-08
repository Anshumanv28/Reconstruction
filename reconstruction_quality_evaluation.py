#!/usr/bin/env python3
"""
Reconstruction Quality Evaluation Pipeline.
Tests if our reconstruction preserves table structure by running TableFormer on reconstructed images.
"""

import json
import jsonlines
import ast
from pathlib import Path
from PIL import Image
from enhanced_dimension_preserving_reconstructor_v3 import EnhancedDimensionPreservingReconstructorV3
from tableformer_baseline_evaluation import TableFormerBaselineEvaluator
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

class ReconstructionQualityEvaluator:
    """Evaluates reconstruction quality by comparing TableFormer outputs."""
    
    def __init__(self, data_dir: str = "data/pubtabnet_test"):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.annotations_dir = self.data_dir / "annotations"
        
        # Initialize components
        self.reconstructor = EnhancedDimensionPreservingReconstructorV3()
        self.tableformer_evaluator = TableFormerBaselineEvaluator(str(self.data_dir))
        
        # Output directories
        self.output_dir = Path("outputs/reconstruction_evaluation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reconstructed_dir = self.output_dir / "reconstructed_images"
        self.reconstructed_dir.mkdir(exist_ok=True)
        self.comparison_dir = self.output_dir / "comparison_images"
        self.comparison_dir.mkdir(exist_ok=True)
    
    def parse_pubtabnet_html(self, html_str: str) -> Dict:
        """Parse PubTabNet HTML structure."""
        try:
            html_data = ast.literal_eval(html_str)
            return html_data
        except Exception as e:
            print(f"Error parsing HTML: {e}")
            return None
    
    def evaluate_reconstruction_quality(self, sample: Dict) -> Dict:
        """Evaluate reconstruction quality for a single sample."""
        filename = sample['filename']
        ground_truth_html = sample['html']
        
        print(f"  Evaluating reconstruction for {filename}")
        
        # Load original image
        original_image_path = self.images_dir / filename
        if not original_image_path.exists():
            return {"error": f"Original image not found: {filename}"}
        
        original_image = Image.open(original_image_path).convert('RGB')
        
        # Step 1: Get TableFormer detections on original image
        print(f"    Step 1: TableFormer on original image")
        original_detections = self.tableformer_evaluator.extract_tableformer_detections(original_image)
        
        # Step 2: Reconstruct table using our logic
        print(f"    Step 2: Reconstructing table")
        try:
            # Create sample dict for the reconstructor
            reconstruction_sample = {
                'filename': filename,
                'html_table': sample.get('html_table', ''),
                'html': sample.get('html', '')
            }
            reconstruction_result = self.reconstructor.reconstruct_with_advanced_validation(reconstruction_sample)
            
            if not reconstruction_result.get("success", False):
                return {"error": f"Reconstruction failed: {reconstruction_result.get('error', 'Unknown error')}"}
            
            # The reconstruction method doesn't return the image directly, we need to load it
            # The image is saved by the reconstruction method, so we load it from the saved location
            reconstructed_path = Path("outputs/enhanced_reconstruction_v3") / f"{filename.replace('.png', '')}_reconstructed.png"
            if reconstructed_path.exists():
                reconstructed_image = Image.open(reconstructed_path).convert('RGB')
            else:
                return {"error": f"Reconstructed image not found: {reconstructed_path}"}
            
        except Exception as e:
            return {"error": f"Reconstruction exception: {e}"}
        
        # Step 3: Get TableFormer detections on reconstructed image
        print(f"    Step 3: TableFormer on reconstructed image")
        reconstructed_detections = self.tableformer_evaluator.extract_tableformer_detections(reconstructed_image)
        
        # Step 4: Compare detections
        print(f"    Step 4: Comparing detections")
        comparison_metrics = self._compare_detections(original_detections, reconstructed_detections)
        
        # Step 5: Save images for visual inspection
        self._save_evaluation_images(
            original_image, reconstructed_image, filename,
            original_detections, reconstructed_detections
        )
        
        # Parse ground truth for reference
        gt_data = self.parse_pubtabnet_html(ground_truth_html)
        gt_cells = gt_data.get('cells', []) if gt_data else []
        gt_structure = gt_data.get('structure', {}) if gt_data else {}
        gt_tokens = gt_structure.get('tokens', []) if gt_structure else []
        expected_rows = gt_tokens.count('<tr>')
        expected_cols = max(gt_tokens.count('<td>'), gt_tokens.count('<th>')) // expected_rows if expected_rows > 0 else 0
        
        return {
            "filename": filename,
            "original_detections": original_detections,
            "reconstructed_detections": reconstructed_detections,
            "comparison_metrics": comparison_metrics,
            "ground_truth": {
                "cells": len(gt_cells),
                "rows": expected_rows,
                "cols": expected_cols
            },
            "reconstruction_success": True
        }
    
    def _compare_detections(self, original_detections: Dict, reconstructed_detections: Dict) -> Dict:
        """Compare TableFormer detections between original and reconstructed images."""
        metrics = {}
        
        # Compare detection counts
        for detection_type in ["table", "table row", "table column", "table column header", "table spanning cell"]:
            original_count = len(original_detections.get(detection_type, []))
            reconstructed_count = len(reconstructed_detections.get(detection_type, []))
            
            metrics[f"{detection_type}_original"] = original_count
            metrics[f"{detection_type}_reconstructed"] = reconstructed_count
            metrics[f"{detection_type}_preservation"] = reconstructed_count / original_count if original_count > 0 else 0
        
        # Calculate overall preservation score
        preservation_scores = []
        for detection_type in ["table row", "table column", "table spanning cell"]:
            if f"{detection_type}_preservation" in metrics:
                preservation_scores.append(metrics[f"{detection_type}_preservation"])
        
        metrics["overall_preservation"] = np.mean(preservation_scores) if preservation_scores else 0
        
        # Calculate structure preservation
        original_rows = len(original_detections.get("table row", []))
        original_cols = len(original_detections.get("table column", []))
        reconstructed_rows = len(reconstructed_detections.get("table row", []))
        reconstructed_cols = len(reconstructed_detections.get("table column", []))
        
        row_preservation = reconstructed_rows / original_rows if original_rows > 0 else 0
        col_preservation = reconstructed_cols / original_cols if original_cols > 0 else 0
        
        metrics["structure_preservation"] = (row_preservation + col_preservation) / 2
        metrics["row_preservation"] = row_preservation
        metrics["col_preservation"] = col_preservation
        
        return metrics
    
    def _save_evaluation_images(self, original_image: Image.Image, reconstructed_image: Image.Image, 
                              filename: str, original_detections: Dict, reconstructed_detections: Dict):
        """Save images for visual inspection."""
        base_name = filename.replace('.png', '')
        
        # Save reconstructed image
        reconstructed_path = self.reconstructed_dir / f"{base_name}_reconstructed.png"
        reconstructed_image.save(reconstructed_path)
        
        # Create comparison image
        comparison_image = self._create_comparison_image(
            original_image, reconstructed_image, original_detections, reconstructed_detections
        )
        comparison_path = self.comparison_dir / f"{base_name}_comparison.png"
        comparison_image.save(comparison_path)
    
    def _create_comparison_image(self, original_image: Image.Image, reconstructed_image: Image.Image,
                               original_detections: Dict, reconstructed_detections: Dict) -> Image.Image:
        """Create side-by-side comparison image with detection overlays."""
        # Resize images to same height
        target_height = min(original_image.height, reconstructed_image.height)
        original_resized = original_image.resize((int(original_image.width * target_height / original_image.height), target_height))
        reconstructed_resized = reconstructed_image.resize((int(reconstructed_image.width * target_height / reconstructed_image.height), target_height))
        
        # Create side-by-side image
        total_width = original_resized.width + reconstructed_resized.width
        comparison = Image.new('RGB', (total_width, target_height), 'white')
        comparison.paste(original_resized, (0, 0))
        comparison.paste(reconstructed_resized, (original_resized.width, 0))
        
        return comparison
    
    def evaluate_reconstruction_pipeline(self, max_samples: int = 10) -> Dict:
        """Evaluate reconstruction quality on multiple samples."""
        print(f"Evaluating reconstruction quality on {max_samples} samples...")
        
        # Load test samples
        val_file = self.annotations_dir / "pubtabnet_val_test.jsonl"
        if not val_file.exists():
            print(f"Validation file not found: {val_file}")
            return {}
        
        samples = []
        with jsonlines.open(val_file, 'r') as reader:
            for i, sample in enumerate(reader):
                if len(samples) >= max_samples:
                    break
                samples.append(sample)
        
        print(f"Loaded {len(samples)} test samples")
        
        # Evaluate each sample
        results = []
        for i, sample in enumerate(samples):
            print(f"Sample {i+1}/{len(samples)}: {sample['filename']}")
            
            try:
                result = self.evaluate_reconstruction_quality(sample)
                results.append(result)
                
                if "error" not in result:
                    metrics = result["comparison_metrics"]
                    print(f"  Structure preservation: {metrics['structure_preservation']:.3f}")
                    print(f"  Overall preservation: {metrics['overall_preservation']:.3f}")
                    print(f"  Row preservation: {metrics['row_preservation']:.3f}")
                    print(f"  Column preservation: {metrics['col_preservation']:.3f}")
                else:
                    print(f"  Error: {result['error']}")
                    
            except Exception as e:
                print(f"  Exception: {e}")
                continue
        
        # Calculate overall metrics
        valid_results = [r for r in results if "error" not in r]
        if not valid_results:
            print("No valid results to analyze")
            return {}
        
        # Aggregate metrics
        avg_structure_preservation = np.mean([r["comparison_metrics"]["structure_preservation"] for r in valid_results])
        avg_overall_preservation = np.mean([r["comparison_metrics"]["overall_preservation"] for r in valid_results])
        avg_row_preservation = np.mean([r["comparison_metrics"]["row_preservation"] for r in valid_results])
        avg_col_preservation = np.mean([r["comparison_metrics"]["col_preservation"] for r in valid_results])
        
        # Quality distribution
        structure_preservations = [r["comparison_metrics"]["structure_preservation"] for r in valid_results]
        excellent_preservation = sum(1 for p in structure_preservations if p >= 0.95)
        good_preservation = sum(1 for p in structure_preservations if 0.8 <= p < 0.95)
        poor_preservation = sum(1 for p in structure_preservations if p < 0.8)
        
        overall_results = {
            "total_samples": len(results),
            "valid_samples": len(valid_results),
            "success_rate": len(valid_results) / len(results),
            "avg_structure_preservation": avg_structure_preservation,
            "avg_overall_preservation": avg_overall_preservation,
            "avg_row_preservation": avg_row_preservation,
            "avg_col_preservation": avg_col_preservation,
            "preservation_quality": {
                "excellent": excellent_preservation,
                "good": good_preservation,
                "poor": poor_preservation
            },
            "individual_results": valid_results
        }
        
        # Print summary
        print(f"\nRECONSTRUCTION QUALITY EVALUATION RESULTS:")
        print(f"  Total samples: {overall_results['total_samples']}")
        print(f"  Valid samples: {overall_results['valid_samples']}")
        print(f"  Success rate: {overall_results['success_rate']:.1%}")
        print(f"  Average structure preservation: {overall_results['avg_structure_preservation']:.3f}")
        print(f"  Average overall preservation: {overall_results['avg_overall_preservation']:.3f}")
        print(f"  Average row preservation: {overall_results['avg_row_preservation']:.3f}")
        print(f"  Average column preservation: {overall_results['avg_col_preservation']:.3f}")
        print(f"  Preservation quality distribution:")
        print(f"    Excellent (≥0.95): {excellent_preservation}")
        print(f"    Good (0.8-0.95): {good_preservation}")
        print(f"    Poor (<0.8): {poor_preservation}")
        
        # Quality assessment
        if avg_structure_preservation >= 0.9:
            print(f"\n✅ EXCELLENT: Reconstruction preserves structure very well!")
        elif avg_structure_preservation >= 0.8:
            print(f"\n✅ GOOD: Reconstruction preserves structure well")
        elif avg_structure_preservation >= 0.6:
            print(f"\n⚠️  FAIR: Reconstruction has some structure preservation issues")
        else:
            print(f"\n❌ POOR: Reconstruction fails to preserve structure")
        
        return overall_results
    
    def save_results(self, results: Dict):
        """Save evaluation results."""
        # Save detailed results
        with open(self.output_dir / "reconstruction_quality_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {self.output_dir}")

def main():
    """Run reconstruction quality evaluation."""
    evaluator = ReconstructionQualityEvaluator()
    results = evaluator.evaluate_reconstruction_pipeline(max_samples=5)
    
    if results:
        evaluator.save_results(results)

if __name__ == "__main__":
    main()
