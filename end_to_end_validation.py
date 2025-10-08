#!/usr/bin/env python3
"""
End-to-End Validation Pipeline.
Complete validation: Original → TableFormer → Reconstruction → TableFormer → Ground Truth Comparison.
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

class EndToEndValidator:
    """Complete end-to-end validation pipeline."""
    
    def __init__(self, data_dir: str = "data/pubtabnet_test"):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.annotations_dir = self.data_dir / "annotations"
        
        # Initialize components
        self.reconstructor = EnhancedDimensionPreservingReconstructorV3()
        self.tableformer_evaluator = TableFormerBaselineEvaluator(str(self.data_dir))
        
        # Output directories
        self.output_dir = Path("outputs/end_to_end_evaluation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_dir = self.output_dir / "pipeline_images"
        self.pipeline_dir.mkdir(exist_ok=True)
    
    def parse_pubtabnet_html(self, html_str: str) -> Dict:
        """Parse PubTabNet HTML structure."""
        try:
            html_data = ast.literal_eval(html_str)
            return html_data
        except Exception as e:
            print(f"Error parsing HTML: {e}")
            return None
    
    def run_end_to_end_validation(self, sample: Dict) -> Dict:
        """Run complete end-to-end validation pipeline."""
        filename = sample['filename']
        ground_truth_html = sample['html']
        
        print(f"  Running end-to-end validation for {filename}")
        
        # Load original image
        original_image_path = self.images_dir / filename
        if not original_image_path.exists():
            return {"error": f"Original image not found: {filename}"}
        
        original_image = Image.open(original_image_path).convert('RGB')
        
        # Parse ground truth
        gt_data = self.parse_pubtabnet_html(ground_truth_html)
        if not gt_data:
            return {"error": f"Could not parse ground truth HTML for {filename}"}
        
        gt_cells = gt_data.get('cells', [])
        gt_structure = gt_data.get('structure', {})
        gt_tokens = gt_structure.get('tokens', [])
        expected_rows = gt_tokens.count('<tr>')
        expected_cols = max(gt_tokens.count('<td>'), gt_tokens.count('<th>')) // expected_rows if expected_rows > 0 else 0
        
        # Step 1: TableFormer on original image
        print(f"    Step 1: TableFormer on original image")
        original_detections = self.tableformer_evaluator.extract_tableformer_detections(original_image)
        original_metrics = self._calculate_detection_metrics(original_detections, gt_data)
        
        # Step 2: Reconstruct table
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
        
        # Step 3: TableFormer on reconstructed image
        print(f"    Step 3: TableFormer on reconstructed image")
        reconstructed_detections = self.tableformer_evaluator.extract_tableformer_detections(reconstructed_image)
        reconstructed_metrics = self._calculate_detection_metrics(reconstructed_detections, gt_data)
        
        # Step 4: Calculate degradation
        print(f"    Step 4: Calculating performance degradation")
        degradation_metrics = self._calculate_degradation(original_metrics, reconstructed_metrics)
        
        # Step 5: Save pipeline visualization
        self._save_pipeline_visualization(
            original_image, reconstructed_image, filename,
            original_detections, reconstructed_detections
        )
        
        return {
            "filename": filename,
            "ground_truth": {
                "cells": len(gt_cells),
                "rows": expected_rows,
                "cols": expected_cols
            },
            "original_metrics": original_metrics,
            "reconstructed_metrics": reconstructed_metrics,
            "degradation_metrics": degradation_metrics,
            "pipeline_success": True
        }
    
    def _calculate_detection_metrics(self, detections: Dict, gt_data: Dict) -> Dict:
        """Calculate detection metrics against ground truth."""
        if not gt_data:
            return {"error": "No ground truth data"}
        
        gt_cells = gt_data.get('cells', [])
        gt_structure = gt_data.get('structure', {})
        gt_tokens = gt_structure.get('tokens', [])
        expected_rows = gt_tokens.count('<tr>')
        expected_cols = max(gt_tokens.count('<td>'), gt_tokens.count('<th>')) // expected_rows if expected_rows > 0 else 0
        
        # Count detections
        detected_rows = len(detections.get('table row', []))
        detected_cols = len(detections.get('table column', []))
        detected_cells = len(detections.get('table spanning cell', [])) + \
                        len(detections.get('table column header', [])) + \
                        len(detections.get('table projected row header', []))
        
        # Calculate accuracies
        row_accuracy = min(detected_rows, expected_rows) / max(detected_rows, expected_rows) if max(detected_rows, expected_rows) > 0 else 0
        col_accuracy = min(detected_cols, expected_cols) / max(detected_cols, expected_cols) if max(detected_cols, expected_cols) > 0 else 0
        structure_accuracy = (row_accuracy + col_accuracy) / 2
        cell_detection_ratio = detected_cells / len(gt_cells) if gt_cells else 0
        
        return {
            "detected_rows": detected_rows,
            "detected_cols": detected_cols,
            "detected_cells": detected_cells,
            "expected_rows": expected_rows,
            "expected_cols": expected_cols,
            "expected_cells": len(gt_cells),
            "row_accuracy": row_accuracy,
            "col_accuracy": col_accuracy,
            "structure_accuracy": structure_accuracy,
            "cell_detection_ratio": cell_detection_ratio
        }
    
    def _calculate_degradation(self, original_metrics: Dict, reconstructed_metrics: Dict) -> Dict:
        """Calculate performance degradation from original to reconstructed."""
        degradation = {}
        
        for metric in ["row_accuracy", "col_accuracy", "structure_accuracy", "cell_detection_ratio"]:
            if metric in original_metrics and metric in reconstructed_metrics:
                original_val = original_metrics[metric]
                reconstructed_val = reconstructed_metrics[metric]
                
                if original_val > 0:
                    degradation[metric] = (original_val - reconstructed_val) / original_val
                else:
                    degradation[metric] = 0
        
        # Calculate overall degradation
        degradation_values = [v for v in degradation.values() if v is not None]
        degradation["overall_degradation"] = np.mean(degradation_values) if degradation_values else 0
        
        return degradation
    
    def _save_pipeline_visualization(self, original_image: Image.Image, reconstructed_image: Image.Image,
                                   filename: str, original_detections: Dict, reconstructed_detections: Dict):
        """Save pipeline visualization."""
        base_name = filename.replace('.png', '')
        
        # Create pipeline visualization
        pipeline_image = self._create_pipeline_image(
            original_image, reconstructed_image, original_detections, reconstructed_detections
        )
        pipeline_path = self.pipeline_dir / f"{base_name}_pipeline.png"
        pipeline_image.save(pipeline_path)
    
    def _create_pipeline_image(self, original_image: Image.Image, reconstructed_image: Image.Image,
                             original_detections: Dict, reconstructed_detections: Dict) -> Image.Image:
        """Create pipeline visualization showing original → reconstructed."""
        # Resize images to same height
        target_height = min(original_image.height, reconstructed_image.height)
        original_resized = original_image.resize((int(original_image.width * target_height / original_image.height), target_height))
        reconstructed_resized = reconstructed_image.resize((int(reconstructed_image.width * target_height / reconstructed_image.height), target_height))
        
        # Create side-by-side image with arrow
        total_width = original_resized.width + reconstructed_resized.width + 50  # Space for arrow
        pipeline = Image.new('RGB', (total_width, target_height), 'white')
        pipeline.paste(original_resized, (0, 0))
        pipeline.paste(reconstructed_resized, (original_resized.width + 50, 0))
        
        return pipeline
    
    def run_complete_validation(self, max_samples: int = 10) -> Dict:
        """Run complete end-to-end validation on multiple samples."""
        print(f"Running end-to-end validation on {max_samples} samples...")
        
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
        
        # Run validation on each sample
        results = []
        for i, sample in enumerate(samples):
            print(f"Sample {i+1}/{len(samples)}: {sample['filename']}")
            
            try:
                result = self.run_end_to_end_validation(sample)
                results.append(result)
                
                if "error" not in result:
                    degradation = result["degradation_metrics"]
                    print(f"  Overall degradation: {degradation['overall_degradation']:.3f}")
                    print(f"  Structure degradation: {degradation.get('structure_accuracy', 0):.3f}")
                    print(f"  Cell detection degradation: {degradation.get('cell_detection_ratio', 0):.3f}")
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
        
        # Aggregate degradation metrics
        degradation_values = [r["degradation_metrics"]["overall_degradation"] for r in valid_results]
        avg_degradation = np.mean(degradation_values)
        max_degradation = np.max(degradation_values)
        min_degradation = np.min(degradation_values)
        
        # Quality distribution
        excellent_preservation = sum(1 for d in degradation_values if d <= 0.05)  # <5% degradation
        good_preservation = sum(1 for d in degradation_values if 0.05 < d <= 0.10)  # 5-10% degradation
        fair_preservation = sum(1 for d in degradation_values if 0.10 < d <= 0.20)  # 10-20% degradation
        poor_preservation = sum(1 for d in degradation_values if d > 0.20)  # >20% degradation
        
        overall_results = {
            "total_samples": len(results),
            "valid_samples": len(valid_results),
            "success_rate": len(valid_results) / len(results),
            "avg_degradation": avg_degradation,
            "max_degradation": max_degradation,
            "min_degradation": min_degradation,
            "degradation_distribution": {
                "excellent": excellent_preservation,
                "good": good_preservation,
                "fair": fair_preservation,
                "poor": poor_preservation
            },
            "individual_results": valid_results
        }
        
        # Print summary
        print(f"\nEND-TO-END VALIDATION RESULTS:")
        print(f"  Total samples: {overall_results['total_samples']}")
        print(f"  Valid samples: {overall_results['valid_samples']}")
        print(f"  Success rate: {overall_results['success_rate']:.1%}")
        print(f"  Average degradation: {overall_results['avg_degradation']:.1%}")
        print(f"  Max degradation: {overall_results['max_degradation']:.1%}")
        print(f"  Min degradation: {overall_results['min_degradation']:.1%}")
        print(f"  Degradation distribution:")
        print(f"    Excellent (≤5%): {excellent_preservation}")
        print(f"    Good (5-10%): {good_preservation}")
        print(f"    Fair (10-20%): {fair_preservation}")
        print(f"    Poor (>20%): {poor_preservation}")
        
        # Final assessment
        if avg_degradation <= 0.05:
            print(f"\n🎉 EXCELLENT: Reconstruction pipeline performs exceptionally well!")
            print(f"   Average degradation is only {avg_degradation:.1%}")
        elif avg_degradation <= 0.10:
            print(f"\n✅ GOOD: Reconstruction pipeline performs well")
            print(f"   Average degradation is {avg_degradation:.1%}")
        elif avg_degradation <= 0.20:
            print(f"\n⚠️  FAIR: Reconstruction pipeline has moderate issues")
            print(f"   Average degradation is {avg_degradation:.1%}")
        else:
            print(f"\n❌ POOR: Reconstruction pipeline needs significant improvement")
            print(f"   Average degradation is {avg_degradation:.1%}")
        
        return overall_results
    
    def save_results(self, results: Dict):
        """Save validation results."""
        # Save detailed results
        with open(self.output_dir / "end_to_end_validation_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {self.output_dir}")

def main():
    """Run end-to-end validation."""
    validator = EndToEndValidator()
    results = validator.run_complete_validation(max_samples=5)
    
    if results:
        validator.save_results(results)

if __name__ == "__main__":
    main()
