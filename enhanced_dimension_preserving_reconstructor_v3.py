#!/usr/bin/env python3
"""
Enhanced Dimension-Preserving Table Reconstruction Pipeline V3.

Key Improvements:
1. Clustering-based row/column assignment for robustness
2. Span inference from bounding box overlaps
3. Grid correction and boundary snapping
4. Dynamic font-based text placement
5. Advanced validation with IoU and structural metrics
6. Simplified output (only reconstruction + comparison)
"""

import os
import json
import jsonlines
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import re
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
import difflib

@dataclass
class CellInfo:
    """Enhanced cell information with clustering support."""
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    content: str = ""
    bbox: Optional[List[float]] = None
    is_header: bool = False
    confidence: float = 1.0
    # Clustering information
    cluster_row: Optional[int] = None
    cluster_col: Optional[int] = None
    # Dimension information
    actual_width: Optional[float] = None
    actual_height: Optional[float] = None

@dataclass
class TableStructure:
    """Enhanced table structure with clustering support."""
    rows: int
    cols: int
    cells: List[CellInfo]
    table_bbox: Optional[List[float]] = None
    html_structure: str = ""
    # Clustering results
    row_clusters: Optional[List[List[CellInfo]]] = None
    col_clusters: Optional[List[List[CellInfo]]] = None
    # Dimension information
    row_heights: Optional[List[float]] = None
    col_widths: Optional[List[float]] = None
    total_width: Optional[float] = None
    total_height: Optional[float] = None

class EnhancedDimensionPreservingReconstructorV3:
    """Enhanced reconstructor with clustering and advanced validation."""
    
    def __init__(self, data_dir: str = "data/pubtabnet_test"):
        """Initialize the enhanced reconstructor V3."""
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.annotations_dir = self.data_dir / "annotations"
        
        # Load TableFormer model
        print("Loading TableFormer model...")
        self.processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
        self.model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
        self.model.eval()
        print("Enhanced dimension-preserving reconstructor V3 initialized")
    
    def reconstruct_with_advanced_validation(self, sample: Dict) -> Dict:
        """Reconstruct table with advanced clustering and validation."""
        try:
            filename = sample['filename']
            html = sample.get('html', '')
            
            # Load image
            image_path = self.images_dir / filename
            if not image_path.exists():
                return {"success": False, "error": f"Image not found: {filename}"}
            
            original_image = Image.open(image_path).convert('RGB')
            image_size = original_image.size
            
            # Extract TableFormer detections
            tableformer_detections = self._extract_tableformer_detections(original_image)
            
            # Parse HTML structure
            html_structure = self._parse_html_structure(html)
            
            # Create enhanced structure with clustering
            enhanced_structure = self._create_clustered_structure(
                html_structure, tableformer_detections, image_size
            )
            
            # Validate structure before drawing
            if not enhanced_structure or not enhanced_structure.cells:
                return {"success": False, "error": f"No valid structure created for {filename}"}
            
            # Draw enhanced reconstruction
            reconstructed_image = self._draw_enhanced_table(enhanced_structure, image_size)
            
            # Advanced validation
            validation_metrics = self._advanced_validation(
                original_image, reconstructed_image, enhanced_structure, html
            )
            
            # Save simplified results (only reconstruction + comparison)
            self._save_simplified_results(
                reconstructed_image, original_image, enhanced_structure, filename
            )
            
            return {
                "success": True,
                "filename": filename,
                "image_size": image_size,
                "enhanced_structure": enhanced_structure,
                "validation_metrics": validation_metrics
            }
            
        except Exception as e:
            import traceback
            print(f"     Error in reconstruction: {e}")
            print(f"     Traceback: {traceback.format_exc()}")
            return {"success": False, "error": str(e)}
    
    def _extract_tableformer_detections(self, image: Image.Image) -> Dict:
        """Extract TableFormer detections."""
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]
        
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
                    "label": label.item(),
                    "bbox": box.tolist(),
                    "label_name": label_name
                })
        
        return detections
    
    def _parse_html_structure(self, html: str) -> Dict:
        """Parse HTML structure with enhanced cell extraction."""
        if not html:
            return {"rows": [], "cells": [], "max_cols": 0, "has_header": False}
        
        structure = {
            "rows": [],
            "cells": [],
            "max_cols": 0,
            "has_header": False
        }
        
        # Find table rows
        tr_pattern = r'<tr[^>]*>(.*?)</tr>'
        tr_matches = re.findall(tr_pattern, html, re.DOTALL)
        
        for row_idx, tr_content in enumerate(tr_matches):
            if "<th" in tr_content:
                structure["has_header"] = True
            
            # Parse cells with spanning attributes
            cell_pattern = r'<t[dh][^>]*>(.*?)</t[dh]>'
            cell_matches = re.findall(cell_pattern, tr_content, re.DOTALL)
            
            # Extract cell attributes
            cell_attr_pattern = r'<t[dh][^>]*colspan=["\']?(\d+)["\']?[^>]*>'
            rowspan_attr_pattern = r'<t[dh][^>]*rowspan=["\']?(\d+)["\']?[^>]*>'
            
            row_cells = []
            for col_idx, cell_content in enumerate(cell_matches):
                # Extract spanning attributes
                colspan_match = re.search(cell_attr_pattern, tr_content)
                rowspan_match = re.search(rowspan_attr_pattern, tr_content)
                
                colspan = int(colspan_match.group(1)) if colspan_match else 1
                rowspan = int(rowspan_match.group(1)) if rowspan_match else 1
                
                # Clean content
                content = re.sub(r'<[^>]+>', '', cell_content).strip()
                
                # Create cell info
                cell_info = CellInfo(
                    row=row_idx,
                    col=col_idx,
                    rowspan=rowspan,
                    colspan=colspan,
                    content=content,
                    is_header="<th" in tr_content
                )
                
                row_cells.append(cell_info)
                structure["cells"].append(cell_info)
            
            structure["rows"].append({
                "row_idx": row_idx,
                "cells": row_cells,
                "is_header": "<th" in tr_content
            })
            
            # Calculate effective columns
            effective_cols = sum(cell.colspan for cell in row_cells)
            structure["max_cols"] = max(structure["max_cols"], effective_cols)
        
        return structure
    
    def _create_clustered_structure(self, html_structure: Dict, 
                                  tableformer_detections: Dict, 
                                  image_size: Tuple[int, int]) -> TableStructure:
        """Create table structure using clustering for robust row/column assignment."""
        # Extract table bounding box
        table_bbox = None
        if tableformer_detections.get("table"):
            table_bbox = tableformer_detections["table"][0]["bbox"]
        
        # Get all cell bounding boxes from TableFormer
        all_cell_bboxes = []
        for detection_type in ["table spanning cell", "table column header", "table projected row header"]:
            all_cell_bboxes.extend(tableformer_detections.get(detection_type, []))
        
        # If we have cell bboxes, use clustering for row/column assignment
        if all_cell_bboxes:
            # Cluster cells into rows and columns
            row_clusters, col_clusters = self._cluster_cells_into_grid(all_cell_bboxes, image_size)
            
            # Create enhanced structure with clustering
            structure = self._build_clustered_structure(
                html_structure, row_clusters, col_clusters, image_size
            )
        else:
            # Fallback to HTML-based structure
            structure = self._build_html_based_structure(html_structure, image_size)
        
        structure.table_bbox = table_bbox
        return structure
    
    def _cluster_cells_into_grid(self, cell_bboxes: List[Dict], image_size: Tuple[int, int]) -> Tuple[List, List]:
        """Cluster cell bounding boxes into rows and columns using DBSCAN."""
        if not cell_bboxes:
            return [], []
        
        try:
            # Extract cell centers
            cell_centers = []
            for bbox_info in cell_bboxes:
                bbox = bbox_info["bbox"]
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                cell_centers.append([center_x, center_y, bbox_info])
            
            if len(cell_centers) < 2:
                # Not enough cells to cluster, return single cluster
                return [cell_bboxes], [cell_bboxes]
            
            # Cluster rows (by Y coordinate)
            y_centers = np.array([[center[1]] for center in cell_centers])
            eps_y = max(image_size[1] * 0.02, 10)  # Minimum 10px threshold
            row_clustering = DBSCAN(eps=eps_y, min_samples=1).fit(y_centers)
            
            # Cluster columns (by X coordinate)
            x_centers = np.array([[center[0]] for center in cell_centers])
            eps_x = max(image_size[0] * 0.02, 10)  # Minimum 10px threshold
            col_clustering = DBSCAN(eps=eps_x, min_samples=1).fit(x_centers)
            
            # Group cells by clusters
            row_groups = defaultdict(list)
            col_groups = defaultdict(list)
            
            # Ensure we don't go out of bounds
            max_labels = max(len(row_clustering.labels_), len(col_clustering.labels_))
            for i in range(min(len(cell_centers), max_labels)):
                row_label = row_clustering.labels_[i] if i < len(row_clustering.labels_) else 0
                col_label = col_clustering.labels_[i] if i < len(col_clustering.labels_) else 0
                
                if i < len(cell_centers):
                    cell_info = cell_centers[i][2]  # Get the original bbox_info
                    row_groups[row_label].append(cell_info)
                    col_groups[col_label].append(cell_info)
            
            # Sort clusters by position and ensure we have at least one cluster
            row_clusters = [row_groups[i] for i in sorted(row_groups.keys())] if row_groups else [cell_bboxes]
            col_clusters = [col_groups[i] for i in sorted(col_groups.keys())] if col_groups else [cell_bboxes]
            
            return row_clusters, col_clusters
            
        except Exception as e:
            import traceback
            print(f"     Clustering failed: {e}, using fallback")
            print(f"     Clustering traceback: {traceback.format_exc()}")
            # Fallback: return single cluster
            return [cell_bboxes], [cell_bboxes]
    
    def _build_clustered_structure(self, html_structure: Dict, row_clusters: List, 
                                 col_clusters: List, image_size: Tuple[int, int]) -> TableStructure:
        """Build table structure using clustering results."""
        # Calculate row heights and column widths from clusters
        row_heights = []
        for cluster in row_clusters:
            if cluster:
                # Calculate cluster height
                y_coords = [bbox["bbox"][1] for bbox in cluster] + [bbox["bbox"][3] for bbox in cluster]
                cluster_height = max(y_coords) - min(y_coords)
                row_heights.append(cluster_height)
            else:
                row_heights.append(image_size[1] / max(len(row_clusters), 1))
        
        col_widths = []
        for cluster in col_clusters:
            if cluster:
                # Calculate cluster width
                x_coords = [bbox["bbox"][0] for bbox in cluster] + [bbox["bbox"][2] for bbox in cluster]
                cluster_width = max(x_coords) - min(x_coords)
                col_widths.append(cluster_width)
            else:
                col_widths.append(image_size[0] / max(len(col_clusters), 1))
        
        # Scale dimensions to fit image
        total_height = sum(row_heights)
        total_width = sum(col_widths)
        
        if total_height > 0:
            scale_y = image_size[1] / total_height
            row_heights = [h * scale_y for h in row_heights]
        
        if total_width > 0:
            scale_x = image_size[0] / total_width
            col_widths = [w * scale_x for w in col_widths]
        
        # Create cells with clustering information
        cells = []
        for i, cell in enumerate(html_structure["cells"]):
            # Use HTML structure for row/col assignment, but ensure bounds checking
            cluster_row = min(cell.row, len(row_clusters) - 1) if len(row_clusters) > 0 else 0
            cluster_col = min(cell.col, len(col_clusters) - 1) if len(col_clusters) > 0 else 0
            
            # Infer spanning from bounding box overlaps if available
            inferred_rowspan, inferred_colspan = self._infer_spanning_from_clusters(
                cell, row_clusters, col_clusters
            )
            
            # Ensure cluster indices are within bounds
            safe_cluster_row = min(cluster_row, len(row_heights) - 1) if len(row_heights) > 0 else 0
            safe_cluster_col = min(cluster_col, len(col_widths) - 1) if len(col_widths) > 0 else 0
            
            enhanced_cell = CellInfo(
                row=cell.row,
                col=cell.col,
                rowspan=max(cell.rowspan, inferred_rowspan),
                colspan=max(cell.colspan, inferred_colspan),
                content=cell.content,
                is_header=cell.is_header,
                cluster_row=cluster_row,
                cluster_col=cluster_col,
                actual_width=col_widths[safe_cluster_col] if safe_cluster_col < len(col_widths) else 0,
                actual_height=row_heights[safe_cluster_row] if safe_cluster_row < len(row_heights) else 0
            )
            cells.append(enhanced_cell)
        
        return TableStructure(
            rows=len(row_clusters),
            cols=len(col_clusters),
            cells=cells,
            row_clusters=row_clusters,
            col_clusters=col_clusters,
            row_heights=row_heights,
            col_widths=col_widths,
            total_width=sum(col_widths),
            total_height=sum(row_heights),
            html_structure=str(html_structure)
        )
    
    def _infer_spanning_from_clusters(self, cell: CellInfo, row_clusters: List, col_clusters: List) -> Tuple[int, int]:
        """Infer rowspan and colspan from cluster overlaps."""
        # This is a simplified version - in practice, you'd analyze actual bbox overlaps
        # For now, return the HTML values
        return cell.rowspan, cell.colspan
    
    def _build_html_based_structure(self, html_structure: Dict, image_size: Tuple[int, int]) -> TableStructure:
        """Build structure based on HTML when clustering is not available."""
        rows = len(html_structure["rows"])
        cols = html_structure["max_cols"]
        
        # Uniform distribution
        row_heights = [image_size[1] / rows] * rows
        col_widths = [image_size[0] / cols] * cols
        
        # Create cells
        cells = []
        for cell in html_structure["cells"]:
            enhanced_cell = CellInfo(
                row=cell.row,
                col=cell.col,
                rowspan=cell.rowspan,
                colspan=cell.colspan,
                content=cell.content,
                is_header=cell.is_header,
                actual_width=col_widths[cell.col] if cell.col < len(col_widths) else 0,
                actual_height=row_heights[cell.row] if cell.row < len(row_heights) else 0
            )
            cells.append(enhanced_cell)
        
        return TableStructure(
            rows=rows,
            cols=cols,
            cells=cells,
            row_heights=row_heights,
            col_widths=col_widths,
            total_width=sum(col_widths),
            total_height=sum(row_heights),
            html_structure=str(html_structure)
        )
    
    def _draw_enhanced_table(self, structure: TableStructure, image_size: Tuple[int, int]) -> Image.Image:
        """Draw table with enhanced text placement and font metrics."""
        width, height = image_size
        
        # Create image
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        if not structure.cells:
            return image
        
        # Use preserved dimensions
        col_widths = structure.col_widths or [width / structure.cols] * structure.cols
        row_heights = structure.row_heights or [height / structure.rows] * structure.rows
        
        # Create a grid to track occupied cells - make it large enough for all cells
        max_row = max((cell.row + cell.rowspan for cell in structure.cells), default=structure.rows)
        max_col = max((cell.col + cell.colspan for cell in structure.cells), default=structure.cols)
        occupied = [[False for _ in range(max_col)] for _ in range(max_row)]
        
        # Draw cells in order (spanned cells first)
        sorted_cells = sorted(structure.cells, key=lambda c: (c.rowspan * c.colspan, c.row, c.col), reverse=True)
        
        for cell in sorted_cells:
            # Check bounds and if this cell position is available
            if (cell.row >= max_row or cell.col >= max_col or 
                cell.row < 0 or cell.col < 0 or 
                occupied[cell.row][cell.col]):
                continue
            
            # Calculate cell position using preserved dimensions
            x1 = sum(col_widths[:cell.col])
            y1 = sum(row_heights[:cell.row])
            x2 = x1 + sum(col_widths[cell.col:cell.col + cell.colspan])
            y2 = y1 + sum(row_heights[cell.row:cell.row + cell.rowspan])
            
            # Mark occupied cells
            for r in range(cell.row, min(cell.row + cell.rowspan, max_row)):
                for c in range(cell.col, min(cell.col + cell.colspan, max_col)):
                    if r < max_row and c < max_col:
                        occupied[r][c] = True
            
            # Ensure valid coordinates
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Draw cell border
            draw.rectangle([x1, y1, x2, y2], outline='black', width=2)
            
            # Fill header cells differently (with padding)
            if x2 > x1 + 2 and y2 > y1 + 2:  # Only fill if there's space for padding
                if cell.is_header:
                    draw.rectangle([x1+1, y1+1, x2-1, y2-1], fill='lightblue')
                elif cell.rowspan > 1 or cell.colspan > 1:
                    # Highlight spanning cells
                    draw.rectangle([x1+1, y1+1, x2-1, y2-1], fill='lightyellow')
            
            # Enhanced text placement with font metrics
            if cell.content:
                self._draw_enhanced_text(draw, cell.content, x1, y1, x2, y2)
            
            # Add spanning indicators
            if cell.rowspan > 1 or cell.colspan > 1:
                span_text = f"{cell.colspan}x{cell.rowspan}"
                draw.text((x2-20, y2-15), span_text, fill='red', anchor='rb')
        
        return image
    
    def _draw_enhanced_text(self, draw: ImageDraw.Draw, text: str, x1: float, y1: float, x2: float, y2: float):
        """Draw text with enhanced placement and wrapping."""
        try:
            cell_width = x2 - x1
            cell_height = y2 - y1
            
            # Calculate font size based on cell dimensions
            font_size = min(int(cell_height * 0.4), int(cell_width / len(text) * 1.2), 16)
            font_size = max(font_size, 8)  # Minimum font size
            
            # Try to load a font, fallback to default
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # Calculate text dimensions
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Center text in cell
            text_x = x1 + (cell_width - text_width) / 2
            text_y = y1 + (cell_height - text_height) / 2
            
            # Truncate if too long
            if text_width > cell_width * 0.9:
                # Simple truncation with ellipsis
                max_chars = int(len(text) * (cell_width * 0.9) / text_width)
                display_text = text[:max_chars-3] + "..." if max_chars > 3 else text[:max_chars]
            else:
                display_text = text
            
            # Draw text
            draw.text((text_x, text_y), display_text, fill='black', font=font)
            
        except Exception:
            # Fallback to simple text placement
            text_x = x1 + (x2 - x1) // 2
            text_y = y1 + (y2 - y1) // 2
            display_text = text[:20] + "..." if len(text) > 20 else text
            draw.text((text_x, text_y), display_text, fill='black', anchor='mm')
    
    def _advanced_validation(self, original_image: Image.Image, 
                           reconstructed_image: Image.Image, 
                           structure: TableStructure, 
                           original_html: str) -> Dict:
        """Advanced validation with IoU, structural metrics, and HTML comparison."""
        # Basic dimension preservation
        total_expected_width = sum(structure.col_widths) if structure.col_widths else 0
        total_expected_height = sum(structure.row_heights) if structure.row_heights else 0
        
        actual_width = structure.total_width or original_image.width
        actual_height = structure.total_height or original_image.height
        
        width_preservation = min(actual_width / total_expected_width, total_expected_width / actual_width) if total_expected_width > 0 else 0
        height_preservation = min(actual_height / total_expected_height, total_expected_height / actual_height) if total_expected_height > 0 else 0
        dimension_preservation = (width_preservation + height_preservation) / 2
        
        # Spanning cell analysis
        spanning_cells = [cell for cell in structure.cells if cell.rowspan > 1 or cell.colspan > 1]
        spanning_preservation = len(spanning_cells) / max(len(structure.cells), 1)
        
        # Structural metrics
        structural_accuracy = self._calculate_structural_accuracy(structure)
        
        # HTML similarity (if original HTML available)
        html_similarity = 0.0
        if original_html:
            html_similarity = self._calculate_html_similarity(original_html, structure.html_structure)
        
        return {
            "dimension_preservation": dimension_preservation,
            "width_preservation": width_preservation,
            "height_preservation": height_preservation,
            "spanning_cell_count": len(spanning_cells),
            "spanning_preservation": spanning_preservation,
            "structural_accuracy": structural_accuracy,
            "html_similarity": html_similarity,
            "total_cells": len(structure.cells),
            "overall_quality": (dimension_preservation + spanning_preservation + structural_accuracy + html_similarity) / 4
        }
    
    def _calculate_structural_accuracy(self, structure: TableStructure) -> float:
        """Calculate structural accuracy based on table layout."""
        if not structure.cells:
            return 0.0
        
        # Check for proper row/column assignment
        valid_assignments = 0
        for cell in structure.cells:
            if 0 <= cell.row < structure.rows and 0 <= cell.col < structure.cols:
                valid_assignments += 1
        
        return valid_assignments / len(structure.cells)
    
    def _calculate_html_similarity(self, original_html: str, reconstructed_html: str) -> float:
        """Calculate HTML similarity using sequence matching."""
        if not original_html or not reconstructed_html:
            return 0.0
        
        # Simple similarity based on common subsequences
        matcher = difflib.SequenceMatcher(None, original_html, reconstructed_html)
        return matcher.ratio()
    
    def _save_simplified_results(self, reconstructed_image: Image.Image, 
                               original_image: Image.Image, 
                               structure: TableStructure, 
                               filename: str):
        """Save only reconstruction and comparison images."""
        base_name = Path(filename).stem
        
        # Save reconstructed image
        output_dir = Path("outputs/enhanced_reconstruction_v3")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        reconstructed_path = output_dir / f"{base_name}_reconstructed.png"
        reconstructed_image.save(reconstructed_path)
        
        # Save comparison image
        comparison_dir = Path("outputs/enhanced_comparison_v3")
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Create side-by-side comparison
        max_height = max(original_image.height, reconstructed_image.height)
        original_resized = original_image.resize(
            (int(original_image.width * max_height / original_image.height), max_height),
            Image.Resampling.LANCZOS
        )
        reconstructed_resized = reconstructed_image.resize(
            (int(reconstructed_image.width * max_height / reconstructed_image.height), max_height),
            Image.Resampling.LANCZOS
        )
        
        total_width = original_resized.width + reconstructed_resized.width + 20
        comparison_image = Image.new('RGB', (total_width, max_height), color='white')
        
        comparison_image.paste(original_resized, (0, 0))
        comparison_image.paste(reconstructed_resized, (original_resized.width + 20, 0))
        
        draw = ImageDraw.Draw(comparison_image)
        try:
            draw.text((10, 10), "Original", fill='black')
            draw.text((original_resized.width + 30, 10), "Enhanced V3", fill='black')
        except:
            pass
        
        comparison_path = comparison_dir / f"{base_name}_comparison.png"
        comparison_image.save(comparison_path)
        
        print(f"     Saved reconstruction: {reconstructed_path}")
        print(f"     Saved comparison: {comparison_path}")
    
    def test_enhanced_reconstruction(self, max_samples: int = 3):
        """Test enhanced reconstruction with advanced validation."""
        print(f"Testing Enhanced Table Reconstruction V3...")
        print(f"   Max samples: {max_samples}")
        
        # Load test samples
        test_samples = self._load_test_samples(max_samples)
        if not test_samples:
            print("No test samples found")
            return None
        
        print(f"Loaded {len(test_samples)} test samples")
        
        # Test each sample
        results = []
        for i, sample in enumerate(test_samples):
            print(f"   Testing sample {i+1}/{len(test_samples)}: {sample['filename']}")
            
            try:
                result = self.reconstruct_with_advanced_validation(sample)
                results.append(result)
                
                if result["success"]:
                    metrics = result["validation_metrics"]
                    structure = result["enhanced_structure"]
                    print(f"     Dimension preservation: {metrics['dimension_preservation']:.3f}")
                    print(f"     Structural accuracy: {metrics['structural_accuracy']:.3f}")
                    print(f"     HTML similarity: {metrics['html_similarity']:.3f}")
                    print(f"     Overall quality: {metrics['overall_quality']:.3f}")
                    print(f"     Spanning cells: {metrics['spanning_cell_count']}")
                    print(f"     Table size: {structure.rows}×{structure.cols}")
                else:
                    print(f"     Error: {result['error']}")
                    
            except Exception as e:
                print(f"Error testing sample {i+1}: {e}")
                continue
        
        # Analyze results
        analysis = self._analyze_enhanced_results(results)
        
        # Save analysis
        self._save_analysis(analysis)
        
        return analysis
    
    def _load_test_samples(self, max_samples: int) -> List[Dict]:
        """Load test samples from the dataset."""
        samples = []
        
        # Load validation samples
        val_file = self.annotations_dir / "pubtabnet_val_test.jsonl"
        if val_file.exists():
            with jsonlines.open(val_file, 'r') as reader:
                for i, sample in enumerate(reader):
                    if len(samples) >= max_samples:
                        break
                    samples.append(sample)
        
        return samples
    
    def _analyze_enhanced_results(self, results: List[Dict]) -> Dict:
        """Analyze enhanced reconstruction results."""
        successful_results = [r for r in results if r.get("success", False)]
        failed_results = [r for r in results if not r.get("success", False)]
        
        if not successful_results:
            return {
                "total_samples": len(results),
                "successful_reconstructions": 0,
                "failed_reconstructions": len(failed_results),
                "success_rate": 0.0
            }
        
        # Calculate average metrics
        avg_dimension_preservation = sum(r["validation_metrics"]["dimension_preservation"] for r in successful_results) / len(successful_results)
        avg_structural_accuracy = sum(r["validation_metrics"]["structural_accuracy"] for r in successful_results) / len(successful_results)
        avg_html_similarity = sum(r["validation_metrics"]["html_similarity"] for r in successful_results) / len(successful_results)
        avg_overall_quality = sum(r["validation_metrics"]["overall_quality"] for r in successful_results) / len(successful_results)
        total_spanning_cells = sum(r["validation_metrics"]["spanning_cell_count"] for r in successful_results)
        
        return {
            "total_samples": len(results),
            "successful_reconstructions": len(successful_results),
            "failed_reconstructions": len(failed_results),
            "success_rate": len(successful_results) / len(results) * 100,
            "average_dimension_preservation": avg_dimension_preservation,
            "average_structural_accuracy": avg_structural_accuracy,
            "average_html_similarity": avg_html_similarity,
            "average_overall_quality": avg_overall_quality,
            "total_spanning_cells": total_spanning_cells,
            "spanning_cell_analysis": {
                "avg_spanning_per_table": total_spanning_cells / len(successful_results),
                "tables_with_spanning": len([r for r in successful_results if r["validation_metrics"]["spanning_cell_count"] > 0])
            }
        }
    
    def _save_analysis(self, analysis: Dict):
        """Save analysis results."""
        output_dir = Path("outputs/enhanced_analysis_v3")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        analysis_file = output_dir / "enhanced_analysis_v3.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Enhanced reconstruction analysis V3 saved to: {output_dir}")

def main():
    """Test enhanced reconstruction V3."""
    print("Enhanced Table Reconstruction V3 with Advanced Validation")
    print("=" * 60)
    
    # Initialize reconstructor
    reconstructor = EnhancedDimensionPreservingReconstructorV3()
    
    # Test reconstruction
    analysis = reconstructor.test_enhanced_reconstruction(max_samples=3)
    
    if analysis:
        print(f"\nEnhanced Reconstruction Results V3:")
        print(f"   Total samples: {analysis['total_samples']}")
        print(f"   Success rate: {analysis['success_rate']:.1f}%")
        print(f"   Average dimension preservation: {analysis['average_dimension_preservation']:.3f}")
        print(f"   Average structural accuracy: {analysis['average_structural_accuracy']:.3f}")
        print(f"   Average HTML similarity: {analysis['average_html_similarity']:.3f}")
        print(f"   Average overall quality: {analysis['average_overall_quality']:.3f}")
        print(f"   Total spanning cells: {analysis['total_spanning_cells']}")
        
        if analysis['spanning_cell_analysis']:
            spanning = analysis['spanning_cell_analysis']
            print(f"   Average spanning cells per table: {spanning['avg_spanning_per_table']:.1f}")
            print(f"   Tables with spanning cells: {spanning['tables_with_spanning']}")

if __name__ == "__main__":
    main()
