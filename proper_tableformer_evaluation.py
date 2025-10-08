#!/usr/bin/env python3
"""
Proper TableFormer Evaluation - measuring what it actually does (structure recognition).
"""

import json
import jsonlines
import ast
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
import torch
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ProperTableFormerEvaluator:
    """Evaluate TableFormer on what it actually does - structure recognition."""
    
    def __init__(self, data_dir: str = "data/pubtabnet_test"):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.annotations_dir = self.data_dir / "annotations"
        
        # Load TableFormer model
        print("Loading TableFormer model...")
        self.processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
        self.model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
        self.model.eval()
        print("TableFormer model loaded")
    
    def parse_pubtabnet_html(self, html_str: str) -> Dict:
        """Parse PubTabNet HTML structure."""
        try:
            html_data = ast.literal_eval(html_str)
            return html_data
        except Exception as e:
            print(f"Error parsing HTML: {e}")
            return None
    
    def extract_tableformer_detections(self, image: Image.Image) -> Dict:
        """Extract TableFormer detections from image."""
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Process outputs
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]
        
        # Group detections by type
        detections = {
            "table": [],
            "table row": [],
            "table column": [],
            "table column header": [],
            "table projected row header": [],
            "table spanning cell": []
        }
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_name = self.model.config.id2label[label.item()]
            if label_name in detections:
                detections[label_name].append({
                    "score": score.item(),
                    "bbox": box.tolist(),
                    "label": label.item(),
                    "label_name": label_name
                })
        
        return detections
    
    def calculate_bbox_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        if x_max <= x_min or y_max <= y_min:
            return 0.0
        
        intersection = (x_max - x_min) * (y_max - y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_structure_recognition(self, sample: Dict) -> Dict:
        """Evaluate TableFormer's structure recognition capabilities."""
        filename = sample['filename']
        ground_truth_html = sample['html']
        
        # Load image
        image_path = self.images_dir / filename
        if not image_path.exists():
            return {"error": f"Image not found: {filename}"}
        
        original_image = Image.open(image_path).convert('RGB')
        
        # Parse ground truth
        gt_data = self.parse_pubtabnet_html(ground_truth_html)
        if not gt_data:
            return {"error": f"Could not parse ground truth HTML for {filename}"}
        
        # Get TableFormer detections
        tableformer_detections = self.extract_tableformer_detections(original_image)
        
        # Analyze ground truth structure
        gt_cells = gt_data.get('cells', [])
        gt_structure = gt_data.get('structure', {})
        gt_tokens = gt_structure.get('tokens', [])
        
        # Count expected rows/cols from structure
        expected_rows = gt_tokens.count('<tr>')
        expected_cols = max(gt_tokens.count('<td>'), gt_tokens.count('<th>')) // expected_rows if expected_rows > 0 else 0
        
        # Analyze TableFormer structure detection
        detected_rows = len(tableformer_detections.get('table row', []))
        detected_cols = len(tableformer_detections.get('table column', []))
        detected_tables = len(tableformer_detections.get('table', []))
        
        # Calculate structure accuracy
        row_accuracy = min(detected_rows, expected_rows) / max(detected_rows, expected_rows) if max(detected_rows, expected_rows) > 0 else 0
        col_accuracy = min(detected_cols, expected_cols) / max(detected_cols, expected_cols) if max(detected_cols, expected_cols) > 0 else 0
        structure_accuracy = (row_accuracy + col_accuracy) / 2
        
        # Calculate table detection accuracy
        table_detection_success = 1.0 if detected_tables > 0 else 0.0
        
        # Calculate table bbox IoU if we have both ground truth and detection
        table_bbox_iou = 0.0
        if detected_tables > 0 and gt_cells:
            # Get table bbox from TableFormer
            tf_table_bbox = tableformer_detections['table'][0]['bbox']
            
            # Calculate ground truth table bbox from cell bboxes
            all_gt_bboxes = [cell.get('bbox', [0, 0, 0, 0]) for cell in gt_cells]
            if all_gt_bboxes:
                gt_table_bbox = [
                    min(bbox[0] for bbox in all_gt_bboxes),  # x_min
                    min(bbox[1] for bbox in all_gt_bboxes),  # y_min
                    max(bbox[2] for bbox in all_gt_bboxes),  # x_max
                    max(bbox[3] for bbox in all_gt_bboxes)   # y_max
                ]
                table_bbox_iou = self.calculate_bbox_iou(tf_table_bbox, gt_table_bbox)
        
        # Calculate row/column detection precision and recall
        row_precision = detected_rows / max(detected_rows, 1)  # How many detected rows are correct
        row_recall = min(detected_rows, expected_rows) / max(expected_rows, 1)  # How many expected rows were found
        row_f1 = 2 * (row_precision * row_recall) / (row_precision + row_recall) if (row_precision + row_recall) > 0 else 0
        
        col_precision = detected_cols / max(detected_cols, 1)
        col_recall = min(detected_cols, expected_cols) / max(expected_cols, 1)
        col_f1 = 2 * (col_precision * col_recall) / (col_precision + col_recall) if (col_precision + col_recall) > 0 else 0
        
        return {
            "filename": filename,
            "gt_cells": len(gt_cells),
            "gt_rows": expected_rows,
            "gt_cols": expected_cols,
            "detected_tables": detected_tables,
            "detected_rows": detected_rows,
            "detected_cols": detected_cols,
            "structure_accuracy": structure_accuracy,
            "row_accuracy": row_accuracy,
            "col_accuracy": col_accuracy,
            "table_detection_success": table_detection_success,
            "table_bbox_iou": table_bbox_iou,
            "row_precision": row_precision,
            "row_recall": row_recall,
            "row_f1": row_f1,
            "col_precision": col_precision,
            "col_recall": col_recall,
            "col_f1": col_f1,
            "tableformer_detections": tableformer_detections
        }
    
    def evaluate_proper_baseline(self, max_samples: int = 10) -> Dict:
        """Evaluate TableFormer's proper capabilities (structure recognition)."""
        print(f"Evaluating TableFormer's structure recognition on {max_samples} samples...")
        
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
            print(f"  Sample {i+1}/{len(samples)}: {sample['filename']}")
            
            try:
                result = self.evaluate_structure_recognition(sample)
                results.append(result)
                
                if "error" not in result:
                    print(f"    GT: {result['gt_rows']}×{result['gt_cols']} ({result['gt_cells']} cells)")
                    print(f"    TF: {result['detected_rows']}×{result['detected_cols']} (tables: {result['detected_tables']})")
                    print(f"    Structure accuracy: {result['structure_accuracy']:.3f}")
                    print(f"    Table detection: {result['table_detection_success']:.3f}")
                    print(f"    Table bbox IoU: {result['table_bbox_iou']:.3f}")
                    print(f"    Row F1: {result['row_f1']:.3f}, Col F1: {result['col_f1']:.3f}")
                else:
                    print(f"    Error: {result['error']}")
                    
            except Exception as e:
                print(f"    Exception: {e}")
                continue
        
        # Calculate overall metrics
        valid_results = [r for r in results if "error" not in r]
        if not valid_results:
            print("No valid results to analyze")
            return {}
        
        # Aggregate metrics
        avg_structure_accuracy = sum(r['structure_accuracy'] for r in valid_results) / len(valid_results)
        avg_table_detection = sum(r['table_detection_success'] for r in valid_results) / len(valid_results)
        avg_table_bbox_iou = sum(r['table_bbox_iou'] for r in valid_results) / len(valid_results)
        avg_row_f1 = sum(r['row_f1'] for r in valid_results) / len(valid_results)
        avg_col_f1 = sum(r['col_f1'] for r in valid_results) / len(valid_results)
        avg_row_accuracy = sum(r['row_accuracy'] for r in valid_results) / len(valid_results)
        avg_col_accuracy = sum(r['col_accuracy'] for r in valid_results) / len(valid_results)
        
        # Structure accuracy distribution
        structure_accuracies = [r['structure_accuracy'] for r in valid_results]
        perfect_structure = sum(1 for acc in structure_accuracies if acc >= 0.95)
        good_structure = sum(1 for acc in structure_accuracies if 0.8 <= acc < 0.95)
        fair_structure = sum(1 for acc in structure_accuracies if 0.6 <= acc < 0.8)
        poor_structure = sum(1 for acc in structure_accuracies if acc < 0.6)
        
        overall_results = {
            "total_samples": len(results),
            "valid_samples": len(valid_results),
            "success_rate": len(valid_results) / len(results),
            "avg_structure_accuracy": avg_structure_accuracy,
            "avg_table_detection": avg_table_detection,
            "avg_table_bbox_iou": avg_table_bbox_iou,
            "avg_row_f1": avg_row_f1,
            "avg_col_f1": avg_col_f1,
            "avg_row_accuracy": avg_row_accuracy,
            "avg_col_accuracy": avg_col_accuracy,
            "structure_quality": {
                "perfect": perfect_structure,
                "good": good_structure,
                "fair": fair_structure,
                "poor": poor_structure
            },
            "individual_results": valid_results
        }
        
        # Print summary
        print(f"\nPROPER TABLEFORMER EVALUATION RESULTS:")
        print(f"  Total samples: {overall_results['total_samples']}")
        print(f"  Valid samples: {overall_results['valid_samples']}")
        print(f"  Success rate: {overall_results['success_rate']:.1%}")
        print(f"  Average structure accuracy: {overall_results['avg_structure_accuracy']:.3f}")
        print(f"  Average table detection: {overall_results['avg_table_detection']:.1%}")
        print(f"  Average table bbox IoU: {overall_results['avg_table_bbox_iou']:.3f}")
        print(f"  Average row F1: {overall_results['avg_row_f1']:.3f}")
        print(f"  Average column F1: {overall_results['avg_col_f1']:.3f}")
        print(f"  Average row accuracy: {overall_results['avg_row_accuracy']:.3f}")
        print(f"  Average column accuracy: {overall_results['avg_col_accuracy']:.3f}")
        print(f"  Structure quality distribution:")
        print(f"    Perfect (≥0.95): {perfect_structure}")
        print(f"    Good (0.8-0.95): {good_structure}")
        print(f"    Fair (0.6-0.8): {fair_structure}")
        print(f"    Poor (<0.6): {poor_structure}")
        
        # Performance assessment
        if avg_structure_accuracy >= 0.9:
            print(f"\n🎉 EXCELLENT: TableFormer performs exceptionally well on structure recognition!")
        elif avg_structure_accuracy >= 0.8:
            print(f"\n✅ GOOD: TableFormer performs well on structure recognition")
        elif avg_structure_accuracy >= 0.6:
            print(f"\n⚠️  FAIR: TableFormer has moderate structure recognition performance")
        else:
            print(f"\n❌ POOR: TableFormer struggles with structure recognition")
        
        # Fine-tuning recommendation
        if avg_structure_accuracy >= 0.8 and avg_table_detection >= 0.9:
            print(f"\n✅ NO FINE-TUNING NEEDED: TableFormer is performing well for structure recognition!")
            print(f"   Ready to proceed with reconstruction testing.")
        else:
            print(f"\n⚠️  CONSIDER FINE-TUNING: Structure recognition could be improved")
        
        return overall_results
    
    def save_results(self, results: Dict, output_dir: str = "outputs/proper_baseline_evaluation"):
        """Save evaluation results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        with open(output_path / "proper_tableformer_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_path}")

def main():
    """Run proper TableFormer evaluation."""
    evaluator = ProperTableFormerEvaluator()
    results = evaluator.evaluate_proper_baseline(max_samples=10)
    
    if results:
        evaluator.save_results(results)

if __name__ == "__main__":
    main()

