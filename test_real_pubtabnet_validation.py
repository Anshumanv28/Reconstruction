#!/usr/bin/env python3
"""
Test reconstruction against REAL PubTabNet ground truth data.
"""

import json
import jsonlines
from pathlib import Path
from PIL import Image
from enhanced_dimension_preserving_reconstructor_v3 import EnhancedDimensionPreservingReconstructorV3
import ast

def parse_pubtabnet_html(html_str):
    """Parse PubTabNet HTML structure."""
    try:
        # Parse the HTML structure
        html_data = ast.literal_eval(html_str)
        return html_data
    except:
        return None

def calculate_bbox_iou(bbox1, bbox2):
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

def validate_against_ground_truth(reconstructor, sample):
    """Validate reconstruction against PubTabNet ground truth."""
    filename = sample['filename']
    ground_truth_html = sample['html']
    
    # Load image
    image_path = reconstructor.images_dir / filename
    if not image_path.exists():
        return {"error": f"Image not found: {filename}"}
    
    original_image = Image.open(image_path).convert('RGB')
    
    # Parse ground truth
    gt_data = parse_pubtabnet_html(ground_truth_html)
    if not gt_data:
        return {"error": f"Could not parse ground truth HTML for {filename}"}
    
    # Get TableFormer detections
    tableformer_detections = reconstructor._extract_tableformer_detections(original_image)
    
    # Parse our HTML structure
    html_structure = reconstructor._parse_html_structure(sample.get('html_table', ''))
    
    # Create our structure
    our_structure = reconstructor._create_clustered_structure(
        html_structure, tableformer_detections, original_image.size
    )
    
    # Compare with ground truth
    gt_cells = gt_data.get('cells', [])
    our_cells = our_structure.cells
    
    print(f"  Ground truth cells: {len(gt_cells)}")
    print(f"  Our detected cells: {len(our_cells)}")
    
    # Calculate cell-level IoU
    cell_ious = []
    matched_cells = 0
    
    for gt_cell in gt_cells:
        gt_bbox = gt_cell.get('bbox', [0, 0, 0, 0])
        gt_content = ''.join(gt_cell.get('tokens', []))
        
        best_iou = 0.0
        best_match = None
        
        for our_cell in our_cells:
            # We don't have bbox in our cells, so we'll estimate position
            # This is a limitation - we need to add bbox tracking
            if our_cell.content and gt_content:
                # Simple content matching for now
                if our_cell.content.strip() == gt_content.strip():
                    best_iou = 1.0
                    best_match = our_cell
                    break
        
        if best_iou > 0.5:  # Threshold for matching
            matched_cells += 1
            cell_ious.append(best_iou)
    
    # Calculate metrics
    cell_precision = matched_cells / len(our_cells) if our_cells else 0
    cell_recall = matched_cells / len(gt_cells) if gt_cells else 0
    cell_f1 = 2 * (cell_precision * cell_recall) / (cell_precision + cell_recall) if (cell_precision + cell_recall) > 0 else 0
    avg_iou = sum(cell_ious) / len(cell_ious) if cell_ious else 0
    
    # Structure comparison
    gt_structure = gt_data.get('structure', {})
    gt_tokens = gt_structure.get('tokens', [])
    
    # Count expected rows/cols from structure
    expected_rows = gt_tokens.count('<tr>')
    expected_cols = max(gt_tokens.count('<td>'), gt_tokens.count('<th>')) // expected_rows if expected_rows > 0 else 0
    
    structure_accuracy = 0.0
    if expected_rows > 0 and expected_cols > 0:
        row_accuracy = min(our_structure.rows, expected_rows) / max(our_structure.rows, expected_rows)
        col_accuracy = min(our_structure.cols, expected_cols) / max(our_structure.cols, expected_cols)
        structure_accuracy = (row_accuracy + col_accuracy) / 2
    
    return {
        "filename": filename,
        "gt_cells": len(gt_cells),
        "our_cells": len(our_cells),
        "matched_cells": matched_cells,
        "cell_precision": cell_precision,
        "cell_recall": cell_recall,
        "cell_f1": cell_f1,
        "avg_iou": avg_iou,
        "expected_rows": expected_rows,
        "expected_cols": expected_cols,
        "our_rows": our_structure.rows,
        "our_cols": our_structure.cols,
        "structure_accuracy": structure_accuracy
    }

def main():
    """Test against real PubTabNet data."""
    print("Testing Against REAL PubTabNet Ground Truth")
    print("=" * 50)
    
    # Initialize reconstructor
    reconstructor = EnhancedDimensionPreservingReconstructorV3()
    
    # Load test samples
    val_file = reconstructor.annotations_dir / "pubtabnet_val_test.jsonl"
    if not val_file.exists():
        print(f"Validation file not found: {val_file}")
        return
    
    samples = []
    with jsonlines.open(val_file, 'r') as reader:
        for i, sample in enumerate(reader):
            if len(samples) >= 5:  # Test with 5 samples
                break
            samples.append(sample)
    
    print(f"Testing {len(samples)} samples against ground truth...")
    
    results = []
    for i, sample in enumerate(samples):
        print(f"\nSample {i+1}/{len(samples)}: {sample['filename']}")
        
        try:
            result = validate_against_ground_truth(reconstructor, sample)
            results.append(result)
            
            if "error" not in result:
                print(f"  GT cells: {result['gt_cells']}, Our cells: {result['our_cells']}")
                print(f"  Matched: {result['matched_cells']}")
                print(f"  Precision: {result['cell_precision']:.3f}")
                print(f"  Recall: {result['cell_recall']:.3f}")
                print(f"  F1: {result['cell_f1']:.3f}")
                print(f"  Structure: {result['expected_rows']}x{result['expected_cols']} vs {result['our_rows']}x{result['our_cols']}")
                print(f"  Structure accuracy: {result['structure_accuracy']:.3f}")
            else:
                print(f"  Error: {result['error']}")
                
        except Exception as e:
            print(f"  Exception: {e}")
            continue
    
    # Calculate overall metrics
    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        avg_precision = sum(r['cell_precision'] for r in valid_results) / len(valid_results)
        avg_recall = sum(r['cell_recall'] for r in valid_results) / len(valid_results)
        avg_f1 = sum(r['cell_f1'] for r in valid_results) / len(valid_results)
        avg_structure = sum(r['structure_accuracy'] for r in valid_results) / len(valid_results)
        
        print(f"\nOVERALL RESULTS:")
        print(f"  Average Precision: {avg_precision:.3f}")
        print(f"  Average Recall: {avg_recall:.3f}")
        print(f"  Average F1: {avg_f1:.3f}")
        print(f"  Average Structure Accuracy: {avg_structure:.3f}")
        print(f"  Valid samples: {len(valid_results)}/{len(results)}")

if __name__ == "__main__":
    main()
