"""
Perfect Table Reconstruction Logic

Core reconstruction algorithms for creating optimal table structures
using PubTabNet annotations as the gold standard.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import json


@dataclass
class CellInfo:
    """Information about a table cell"""
    row: int
    col: int
    rowspan: int
    colspan: int
    content: str
    bbox: Optional[List[float]] = None  # [x1, y1, x2, y2]
    is_header: bool = False
    confidence: float = 1.0


@dataclass
class TableStructure:
    """Complete table structure information"""
    rows: int
    cols: int
    cells: List[CellInfo]
    table_bbox: Optional[List[float]] = None
    html_structure: str = ""


@dataclass
class ReconstructionConfig:
    """Configuration for table reconstruction"""
    min_cell_size: Tuple[int, int] = (10, 10)  # Minimum cell width, height
    max_cell_size: Tuple[int, int] = (1000, 100)  # Maximum cell width, height
    overlap_threshold: float = 0.3  # Minimum overlap for cell merging
    confidence_threshold: float = 0.5  # Minimum confidence for cell inclusion
    spanning_detection_threshold: float = 1.5  # Size ratio threshold for spanning detection
    grid_tolerance: int = 5  # Pixel tolerance for grid alignment


class PerfectTableReconstructor:
    """
    Perfect table reconstruction using PubTabNet ground truth as reference
    
    This reconstructor creates optimal table structures that can be used
    as a benchmark for model evaluation.
    """
    
    def __init__(self, config: ReconstructionConfig = None):
        self.config = config or ReconstructionConfig()
    
    def reconstruct_from_model_output(self, 
                                    model_cells: List[Dict], 
                                    ocr_text_blocks: List[Dict],
                                    reference_structure: TableStructure = None) -> TableStructure:
        """
        Reconstruct table from model output using hybrid approach
        
        Args:
            model_cells: Cell detections from TableFormer/Table Transformer
            ocr_text_blocks: OCR text blocks
            reference_structure: Optional reference structure for guidance
            
        Returns:
            Reconstructed table structure
        """
        # Extract and normalize cell information
        normalized_cells = self._normalize_model_cells(model_cells)
        
        # Create hybrid grid using both TableFormer cells and OCR text positions
        hybrid_grid_cells = self._create_hybrid_grid_from_detections(normalized_cells, ocr_text_blocks)
        
        # Detect spanning cells in the hybrid grid
        spanning_analysis = self._detect_spanning_cells_in_hybrid_grid(hybrid_grid_cells, normalized_cells, ocr_text_blocks)
        
        # Merge cells based on spanning analysis
        final_cells = self._merge_hybrid_cells_intelligently(hybrid_grid_cells, spanning_analysis)
        
        # Calculate final dimensions
        final_rows, final_cols = self._calculate_final_dimensions(final_cells)
        
        return TableStructure(
            rows=final_rows,
            cols=final_cols,
            cells=final_cells
        )
    
    def _normalize_model_cells(self, model_cells: List[Dict]) -> List[Dict]:
        """Normalize model cell detections to standard format"""
        normalized = []
        
        for i, cell in enumerate(model_cells):
            # Extract bounding box
            bbox = cell.get('bbox', [])
            if len(bbox) < 4:
                continue
            
            # Calculate dimensions
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            # Filter by size constraints
            if (width < self.config.min_cell_size[0] or 
                height < self.config.min_cell_size[1] or
                width > self.config.max_cell_size[0] or 
                height > self.config.max_cell_size[1]):
                continue
            
            normalized_cell = {
                'id': i,
                'bbox': bbox,
                'width': width,
                'height': height,
                'center_x': (bbox[0] + bbox[2]) / 2,
                'center_y': (bbox[1] + bbox[3]) / 2,
                'confidence': cell.get('confidence', 1.0),
                'content': cell.get('content', ''),
                'is_header': cell.get('is_header', False)
            }
            normalized.append(normalized_cell)
        
        return normalized
    
    def _create_hybrid_grid_from_detections(self, table_cells: List[Dict], text_blocks: List[Dict]) -> List[Dict]:
        """Create hybrid grid by averaging positions from TableFormer cells and OCR text"""
        # Extract all detected elements (cells + text)
        all_elements = []
        
        # Add TableFormer cells
        for cell in table_cells:
            all_elements.append({
                'type': 'tableformer_cell',
                'bbox': cell['bbox'],
                'center_x': cell['center_x'],
                'center_y': cell['center_y'],
                'width': cell['width'],
                'height': cell['height'],
                'confidence': cell['confidence'],
                'content': cell['content'],
                'is_header': cell['is_header']
            })
        
        # Add OCR text blocks
        for text_block in text_blocks:
            bbox = text_block.get('bbox', [])
            if len(bbox) >= 4:
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                
                all_elements.append({
                    'type': 'ocr_text',
                    'bbox': bbox,
                    'center_x': center_x,
                    'center_y': center_y,
                    'width': width,
                    'height': height,
                    'confidence': text_block.get('confidence', 0.5),
                    'content': text_block.get('text', ''),
                    'is_header': False
                })
        
        # Cluster elements by position to create grid structure
        grid_positions = self._cluster_elements_by_position(all_elements)
        
        # Create hybrid grid cells
        hybrid_cells = []
        cell_id = 0
        
        for row_idx, row_positions in enumerate(grid_positions):
            for col_idx, position in enumerate(row_positions):
                if position:
                    # Calculate average position from all elements at this grid position
                    avg_bbox = self._calculate_average_bbox(position['elements'])
                    
                    hybrid_cells.append({
                        'cell_id': cell_id,
                        'row': row_idx,
                        'col': col_idx,
                        'absolute_bbox': avg_bbox,
                        'width': avg_bbox[2] - avg_bbox[0],
                        'height': avg_bbox[3] - avg_bbox[1],
                        'center_x': (avg_bbox[0] + avg_bbox[2]) / 2,
                        'center_y': (avg_bbox[1] + avg_bbox[3]) / 2,
                        'elements': position['elements'],
                        'tableformer_elements': [e for e in position['elements'] if e['type'] == 'tableformer_cell'],
                        'ocr_elements': [e for e in position['elements'] if e['type'] == 'ocr_text'],
                        'is_hybrid': True
                    })
                    cell_id += 1
        
        return hybrid_cells
    
    def _cluster_elements_by_position(self, all_elements: List[Dict]) -> List[List[Optional[Dict]]]:
        """Cluster elements by position to create grid structure"""
        if not all_elements:
            return []
        
        # Sort elements by position
        sorted_elements = sorted(all_elements, key=lambda e: (e['center_y'], e['center_x']))
        
        # Cluster into rows using y-position
        y_tolerance = 25  # Pixels tolerance for row clustering
        row_clusters = []
        used_indices = set()
        
        for i, element in enumerate(sorted_elements):
            if i in used_indices:
                continue
            
            row_cluster = [element]
            used_indices.add(i)
            element_y = element['center_y']
            
            # Find elements at similar y-positions
            for j, other_element in enumerate(sorted_elements):
                if j in used_indices:
                    continue
                
                other_y = other_element['center_y']
                if abs(element_y - other_y) <= y_tolerance:
                    row_cluster.append(other_element)
                    used_indices.add(j)
            
            # Sort row cluster by x-position
            row_cluster_sorted = sorted(row_cluster, key=lambda e: e['center_x'])
            row_clusters.append(row_cluster_sorted)
        
        # Sort rows by y-position
        row_clusters_sorted = sorted(row_clusters, key=lambda rc: sum(e['center_y'] for e in rc) / len(rc))
        
        # Create grid positions
        max_cols = max(len(row) for row in row_clusters_sorted) if row_clusters_sorted else 0
        grid_positions = []
        
        for row_idx, row_elements in enumerate(row_clusters_sorted):
            row_positions = []
            for col_idx in range(max_cols):
                if col_idx < len(row_elements):
                    row_positions.append({
                        'elements': [row_elements[col_idx]],
                        'row': row_idx,
                        'col': col_idx
                    })
                else:
                    row_positions.append(None)
            grid_positions.append(row_positions)
        
        return grid_positions
    
    def _calculate_average_bbox(self, elements: List[Dict]) -> List[float]:
        """Calculate weighted average bounding box from multiple elements"""
        if not elements:
            return [0, 0, 0, 0]
        
        # Calculate weighted average based on element type and confidence
        total_weight = 0
        weighted_x1 = 0
        weighted_y1 = 0
        weighted_x2 = 0
        weighted_y2 = 0
        
        for element in elements:
            bbox = element['bbox']
            weight = 1.0
            
            # Weight TableFormer cells higher
            if element['type'] == 'tableformer_cell':
                weight = 1.5
            elif element['type'] == 'ocr_text':
                # Weight by confidence
                weight = element.get('confidence', 50) / 100.0
            
            total_weight += weight
            weighted_x1 += bbox[0] * weight
            weighted_y1 += bbox[1] * weight
            weighted_x2 += bbox[2] * weight
            weighted_y2 += bbox[3] * weight
        
        if total_weight == 0:
            return [0, 0, 0, 0]
        
        return [
            weighted_x1 / total_weight,
            weighted_y1 / total_weight,
            weighted_x2 / total_weight,
            weighted_y2 / total_weight
        ]
    
    def _detect_spanning_cells_in_hybrid_grid(self, hybrid_cells: List[Dict], table_cells: List[Dict], text_blocks: List[Dict]) -> Dict:
        """Detect spanning cells in the hybrid grid"""
        spanning_candidates = []
        
        # Analyze each hybrid cell for spanning potential
        for cell in hybrid_cells:
            # Check if this cell contains multiple TableFormer cells or large text blocks
            tf_elements = cell.get('tableformer_elements', [])
            ocr_elements = cell.get('ocr_elements', [])
            
            # Calculate cell size
            cell_width = cell['width']
            cell_height = cell['height']
            
            # Check if cell is significantly larger than typical
            avg_width = sum(c['width'] for c in hybrid_cells) / len(hybrid_cells) if hybrid_cells else 0
            avg_height = sum(c['height'] for c in hybrid_cells) / len(hybrid_cells) if hybrid_cells else 0
            
            width_ratio = cell_width / avg_width if avg_width > 0 else 1
            height_ratio = cell_height / avg_height if avg_height > 0 else 1
            
            # Check for spanning indicators
            has_multiple_tf_cells = len(tf_elements) > 1
            has_large_text = any(e.get('width', 0) > avg_width * 1.5 for e in ocr_elements)
            is_large_cell = width_ratio > 1.8 or height_ratio > 1.3
            
            if has_multiple_tf_cells or has_large_text or is_large_cell:
                # Determine span type
                span_type = "column_span" if width_ratio > height_ratio else "row_span"
                if width_ratio > 1.5 and height_ratio > 1.2:
                    span_type = "both_span"
                
                confidence = min(0.9, (width_ratio - 1) * 0.3 + (height_ratio - 1) * 0.3 + 0.3)
                if has_multiple_tf_cells:
                    confidence += 0.2
                if has_large_text:
                    confidence += 0.1
                
                spanning_candidates.append({
                    'hybrid_cell': cell,
                    'span_type': span_type,
                    'confidence': confidence,
                    'indicators': {
                        'multiple_tf_cells': has_multiple_tf_cells,
                        'large_text': has_large_text,
                        'large_cell': is_large_cell,
                        'width_ratio': width_ratio,
                        'height_ratio': height_ratio
                    }
                })
        
        # Group spanning candidates
        merged_groups = self._group_hybrid_spanning_candidates(spanning_candidates)
        
        spanning_analysis = {
            'spanning_candidates': spanning_candidates,
            'merged_groups': merged_groups,
            'total_spanning_cells': len(merged_groups)
        }
        
        return spanning_analysis
    
    def _group_hybrid_spanning_candidates(self, spanning_candidates: List[Dict]) -> List[Dict]:
        """Group hybrid spanning candidates"""
        if not spanning_candidates:
            return []
        
        merged_groups = []
        used_cells = set()
        
        for candidate in spanning_candidates:
            if candidate['confidence'] < 0.3:  # Skip low confidence candidates
                continue
            
            cell_id = candidate['hybrid_cell']['cell_id']
            if cell_id in used_cells:
                continue
            
            # Create spanning group
            spanning_group = {
                'hybrid_cell': candidate['hybrid_cell'],
                'span_type': candidate['span_type'],
                'confidence': candidate['confidence'],
                'indicators': candidate['indicators'],
                'group_id': len(merged_groups)
            }
            
            merged_groups.append(spanning_group)
            used_cells.add(cell_id)
        
        return merged_groups
    
    def _merge_hybrid_cells_intelligently(self, hybrid_cells: List[Dict], spanning_analysis: Dict) -> List[CellInfo]:
        """Intelligently merge hybrid cells based on spanning analysis"""
        spanning_candidates = spanning_analysis.get('spanning_candidates', [])
        
        # Convert grid cells to CellInfo objects
        final_cells = []
        
        for cell in hybrid_cells:
            # Check if this cell should be spanning
            spanning_info = None
            for candidate in spanning_candidates:
                if candidate['hybrid_cell']['cell_id'] == cell['cell_id']:
                    spanning_info = candidate
                    break
            
            if spanning_info and spanning_info['confidence'] > self.config.confidence_threshold:
                # Create spanning cell
                rowspan = 2 if spanning_info['span_type'] in ['row_span', 'both_span'] else 1
                colspan = 2 if spanning_info['span_type'] in ['column_span', 'both_span'] else 1
            else:
                rowspan = 1
                colspan = 1
            
            cell_info = CellInfo(
                row=cell['row'],
                col=cell['col'],
                rowspan=rowspan,
                colspan=colspan,
                content=cell.get('content', ''),
                bbox=cell['absolute_bbox'],
                is_header=cell.get('is_header', False),
                confidence=cell.get('confidence', 1.0)
            )
            final_cells.append(cell_info)
        
        return final_cells
    
    def _calculate_final_dimensions(self, cells: List[CellInfo]) -> Tuple[int, int]:
        """Calculate final table dimensions"""
        if not cells:
            return 0, 0
        
        max_row = max(cell.row + cell.rowspan for cell in cells)
        max_col = max(cell.col + cell.colspan for cell in cells)
        
        return max_row, max_col


def main():
    """Test the perfect reconstructor"""
    # Create sample data
    model_cells = [
        {'bbox': [100, 50, 300, 100], 'content': 'Header', 'confidence': 0.9, 'is_header': True},
        {'bbox': [300, 50, 400, 100], 'content': 'Column 3', 'confidence': 0.8, 'is_header': True},
        {'bbox': [100, 100, 200, 200], 'content': 'Cell 1', 'confidence': 0.9, 'is_header': False},
        {'bbox': [200, 100, 300, 150], 'content': 'Cell 2', 'confidence': 0.8, 'is_header': False},
        {'bbox': [300, 100, 400, 150], 'content': 'Cell 3', 'confidence': 0.8, 'is_header': False},
        {'bbox': [200, 150, 300, 200], 'content': 'Cell 4', 'confidence': 0.8, 'is_header': False},
        {'bbox': [300, 150, 400, 200], 'content': 'Cell 5', 'confidence': 0.8, 'is_header': False}
    ]
    
    ocr_text_blocks = [
        {'bbox': [100, 50, 300, 100], 'text': 'Header', 'confidence': 0.9},
        {'bbox': [300, 50, 400, 100], 'text': 'Column 3', 'confidence': 0.8},
        {'bbox': [100, 100, 200, 200], 'text': 'Cell 1', 'confidence': 0.9},
        {'bbox': [200, 100, 300, 150], 'text': 'Cell 2', 'confidence': 0.8},
        {'bbox': [300, 100, 400, 150], 'text': 'Cell 3', 'confidence': 0.8},
        {'bbox': [200, 150, 300, 200], 'text': 'Cell 4', 'confidence': 0.8},
        {'bbox': [300, 150, 400, 200], 'text': 'Cell 5', 'confidence': 0.8}
    ]
    
    # Test reconstruction
    reconstructor = PerfectTableReconstructor()
    reconstructed = reconstructor.reconstruct_from_model_output(model_cells, ocr_text_blocks)
    
    print("Reconstruction Results:")
    print(f"Dimensions: {reconstructed.rows} rows × {reconstructed.cols} columns")
    print(f"Total cells: {len(reconstructed.cells)}")
    
    for cell in reconstructed.cells:
        print(f"Cell R{cell.row}C{cell.col}: {cell.content} (span: {cell.rowspan}×{cell.colspan})")


if __name__ == "__main__":
    main()
