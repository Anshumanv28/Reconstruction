"""
Evaluation Metrics for Table Reconstruction

Quantitative metrics to evaluate table reconstruction quality
against PubTabNet ground truth annotations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from html_parser import TableStructure, CellInfo


@dataclass
class EvaluationResult:
    """Results of table reconstruction evaluation"""
    cell_iou_score: float
    structural_similarity: float
    content_matching_score: float
    spanning_accuracy: float
    overall_score: float
    detailed_metrics: Dict
    errors: List[str]


class TableReconstructionEvaluator:
    """Evaluator for table reconstruction quality"""
    
    def __init__(self):
        self.iou_threshold = 0.5  # Minimum IoU for cell matching
        self.content_similarity_threshold = 0.8  # Minimum content similarity
    
    def evaluate_reconstruction(self, 
                              predicted_structure: TableStructure, 
                              ground_truth: TableStructure) -> EvaluationResult:
        """
        Evaluate predicted table structure against ground truth
        
        Args:
            predicted_structure: Reconstructed table structure
            ground_truth: PubTabNet ground truth structure
            
        Returns:
            EvaluationResult with detailed metrics
        """
        # Calculate individual metrics
        cell_iou_score = self._calculate_cell_iou(predicted_structure, ground_truth)
        structural_similarity = self._calculate_structural_similarity(predicted_structure, ground_truth)
        content_matching_score = self._calculate_content_matching(predicted_structure, ground_truth)
        spanning_accuracy = self._calculate_spanning_accuracy(predicted_structure, ground_truth)
        
        # Calculate overall score (weighted average)
        overall_score = (
            cell_iou_score * 0.3 +
            structural_similarity * 0.25 +
            content_matching_score * 0.25 +
            spanning_accuracy * 0.2
        )
        
        # Identify errors
        errors = self._identify_errors(predicted_structure, ground_truth)
        
        # Detailed metrics
        detailed_metrics = {
            'cell_count_match': len(predicted_structure.cells) == len(ground_truth.cells),
            'row_count_match': predicted_structure.rows == ground_truth.rows,
            'col_count_match': predicted_structure.cols == ground_truth.cols,
            'spanning_cell_count_match': self._count_spanning_cells(predicted_structure) == self._count_spanning_cells(ground_truth),
            'bbox_overlap_ratio': self._calculate_bbox_overlap_ratio(predicted_structure, ground_truth)
        }
        
        return EvaluationResult(
            cell_iou_score=cell_iou_score,
            structural_similarity=structural_similarity,
            content_matching_score=content_matching_score,
            spanning_accuracy=spanning_accuracy,
            overall_score=overall_score,
            detailed_metrics=detailed_metrics,
            errors=errors
        )
    
    def _calculate_cell_iou(self, predicted: TableStructure, ground_truth: TableStructure) -> float:
        """Calculate cell-level Intersection over Union"""
        if not predicted.cells or not ground_truth.cells:
            return 0.0
        
        total_iou = 0.0
        matched_cells = 0
        
        for pred_cell in predicted.cells:
            best_iou = 0.0
            best_match = None
            
            for gt_cell in ground_truth.cells:
                iou = self._calculate_bbox_iou(pred_cell.bbox, gt_cell.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match = gt_cell
            
            if best_iou >= self.iou_threshold:
                total_iou += best_iou
                matched_cells += 1
        
        return total_iou / len(predicted.cells) if predicted.cells else 0.0
    
    def _calculate_bbox_iou(self, bbox1: Optional[List[float]], bbox2: Optional[List[float]]) -> float:
        """Calculate IoU between two bounding boxes"""
        if not bbox1 or not bbox2 or len(bbox1) != 4 or len(bbox2) != 4:
            return 0.0
        
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x1_i >= x2_i or y1_i >= y2_i:
            return 0.0
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def _calculate_structural_similarity(self, predicted: TableStructure, ground_truth: TableStructure) -> float:
        """Calculate structural similarity score"""
        # Row and column count similarity
        row_similarity = 1.0 - abs(predicted.rows - ground_truth.rows) / max(predicted.rows, ground_truth.rows, 1)
        col_similarity = 1.0 - abs(predicted.cols - ground_truth.cols) / max(predicted.cols, ground_truth.cols, 1)
        
        # Cell count similarity
        cell_count_similarity = 1.0 - abs(len(predicted.cells) - len(ground_truth.cells)) / max(len(predicted.cells), len(ground_truth.cells), 1)
        
        # Spanning cell count similarity
        pred_spanning = self._count_spanning_cells(predicted)
        gt_spanning = self._count_spanning_cells(ground_truth)
        spanning_similarity = 1.0 - abs(pred_spanning - gt_spanning) / max(pred_spanning, gt_spanning, 1)
        
        return (row_similarity + col_similarity + cell_count_similarity + spanning_similarity) / 4.0
    
    def _calculate_content_matching(self, predicted: TableStructure, ground_truth: TableStructure) -> float:
        """Calculate content matching score using text similarity"""
        if not predicted.cells or not ground_truth.cells:
            return 0.0
        
        total_similarity = 0.0
        matched_content = 0
        
        for pred_cell in predicted.cells:
            best_similarity = 0.0
            
            for gt_cell in ground_truth.cells:
                similarity = self._calculate_text_similarity(pred_cell.content, gt_cell.content)
                if similarity > best_similarity:
                    best_similarity = similarity
            
            if best_similarity >= self.content_similarity_threshold:
                total_similarity += best_similarity
                matched_content += 1
        
        return total_similarity / len(predicted.cells) if predicted.cells else 0.0
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using simple character overlap"""
        if not text1 or not text2:
            return 1.0 if not text1 and not text2 else 0.0
        
        # Simple character-based similarity
        set1 = set(text1.lower().strip())
        set2 = set(text2.lower().strip())
        
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_spanning_accuracy(self, predicted: TableStructure, ground_truth: TableStructure) -> float:
        """Calculate accuracy of spanning cell detection"""
        pred_spanning = self._get_spanning_cells(predicted)
        gt_spanning = self._get_spanning_cells(ground_truth)
        
        if not gt_spanning:
            return 1.0 if not pred_spanning else 0.0
        
        correct_spanning = 0
        for pred_span in pred_spanning:
            for gt_span in gt_spanning:
                if (pred_span.rowspan == gt_span.rowspan and 
                    pred_span.colspan == gt_span.colspan and
                    self._calculate_text_similarity(pred_span.content, gt_span.content) > 0.8):
                    correct_spanning += 1
                    break
        
        return correct_spanning / len(gt_spanning) if gt_spanning else 0.0
    
    def _count_spanning_cells(self, structure: TableStructure) -> int:
        """Count cells that span multiple rows or columns"""
        return sum(1 for cell in structure.cells if cell.rowspan > 1 or cell.colspan > 1)
    
    def _get_spanning_cells(self, structure: TableStructure) -> List[CellInfo]:
        """Get cells that span multiple rows or columns"""
        return [cell for cell in structure.cells if cell.rowspan > 1 or cell.colspan > 1]
    
    def _calculate_bbox_overlap_ratio(self, predicted: TableStructure, ground_truth: TableStructure) -> float:
        """Calculate overall bounding box overlap ratio"""
        if not predicted.table_bbox or not ground_truth.table_bbox:
            return 0.0
        
        return self._calculate_bbox_iou(predicted.table_bbox, ground_truth.table_bbox)
    
    def _identify_errors(self, predicted: TableStructure, ground_truth: TableStructure) -> List[str]:
        """Identify specific reconstruction errors"""
        errors = []
        
        # Dimension errors
        if predicted.rows != ground_truth.rows:
            errors.append(f"Row count mismatch: predicted {predicted.rows}, expected {ground_truth.rows}")
        
        if predicted.cols != ground_truth.cols:
            errors.append(f"Column count mismatch: predicted {predicted.cols}, expected {ground_truth.cols}")
        
        # Cell count errors
        if len(predicted.cells) != len(ground_truth.cells):
            errors.append(f"Cell count mismatch: predicted {len(predicted.cells)}, expected {len(ground_truth.cells)}")
        
        # Spanning cell errors
        pred_spanning = self._count_spanning_cells(predicted)
        gt_spanning = self._count_spanning_cells(ground_truth)
        if pred_spanning != gt_spanning:
            errors.append(f"Spanning cell count mismatch: predicted {pred_spanning}, expected {gt_spanning}")
        
        # Content errors
        content_errors = 0
        for pred_cell in predicted.cells:
            best_similarity = max(
                self._calculate_text_similarity(pred_cell.content, gt_cell.content)
                for gt_cell in ground_truth.cells
            )
            if best_similarity < 0.5:
                content_errors += 1
        
        if content_errors > len(predicted.cells) * 0.1:  # More than 10% content errors
            errors.append(f"High content error rate: {content_errors}/{len(predicted.cells)} cells")
        
        return errors


def main():
    """Test the evaluation metrics"""
    # Create sample table structures for testing
    from html_parser import CellInfo, TableStructure
    
    # Sample ground truth
    gt_cells = [
        CellInfo(0, 0, 1, 2, "Header", is_header=True),
        CellInfo(0, 2, 1, 1, "Col3", is_header=True),
        CellInfo(1, 0, 2, 1, "Cell1"),
        CellInfo(1, 1, 1, 1, "Cell2"),
        CellInfo(1, 2, 1, 1, "Cell3"),
        CellInfo(2, 1, 1, 1, "Cell4"),
        CellInfo(2, 2, 1, 1, "Cell5")
    ]
    
    ground_truth = TableStructure(rows=3, cols=3, cells=gt_cells)
    
    # Sample prediction (with some errors)
    pred_cells = [
        CellInfo(0, 0, 1, 2, "Header", is_header=True),
        CellInfo(0, 2, 1, 1, "Column3", is_header=True),  # Slightly different content
        CellInfo(1, 0, 2, 1, "Cell1"),
        CellInfo(1, 1, 1, 1, "Cell2"),
        CellInfo(1, 2, 1, 1, "Cell3"),
        CellInfo(2, 1, 1, 1, "Cell4"),
        CellInfo(2, 2, 1, 1, "Cell5")
    ]
    
    predicted = TableStructure(rows=3, cols=3, cells=pred_cells)
    
    # Evaluate
    evaluator = TableReconstructionEvaluator()
    result = evaluator.evaluate_reconstruction(predicted, ground_truth)
    
    print("Evaluation Results:")
    print(f"Cell IoU Score: {result.cell_iou_score:.3f}")
    print(f"Structural Similarity: {result.structural_similarity:.3f}")
    print(f"Content Matching: {result.content_matching_score:.3f}")
    print(f"Spanning Accuracy: {result.spanning_accuracy:.3f}")
    print(f"Overall Score: {result.overall_score:.3f}")
    print(f"Errors: {result.errors}")


if __name__ == "__main__":
    main()
