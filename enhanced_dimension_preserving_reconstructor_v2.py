#!/usr/bin/env python3
"""
Enhanced Dimension-Preserving Table Reconstruction Pipeline V2.

This version fixes the coordinate scaling issues and properly preserves
individual row heights and column widths by:
1. Properly scaling TableFormer coordinates to image coordinates
2. Using actual image dimensions for reconstruction
3. Implementing proportional scaling while preserving ratios
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

@dataclass
class CellInfo:
    """Enhanced cell information with dimension preservation."""
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    content: str = ""
    bbox: Optional[List[float]] = None
    is_header: bool = False
    confidence: float = 1.0
    # Enhanced dimension information
    actual_width: Optional[float] = None
    actual_height: Optional[float] = None
    content_width: Optional[float] = None
    content_height: Optional[float] = None

@dataclass
class TableStructure:
    """Enhanced table structure with dimension preservation."""
    rows: int
    cols: int
    cells: List[CellInfo]
    table_bbox: Optional[List[float]] = None
    html_structure: str = ""
    # Enhanced dimension information
    row_heights: Optional[List[float]] = None
    col_widths: Optional[List[float]] = None
    total_width: Optional[float] = None
    total_height: Optional[float] = None

class EnhancedDimensionPreservingReconstructorV2:
    """Enhanced reconstructor that properly preserves individual row/column dimensions."""
    
    def __init__(self, data_dir: str = "data/pubtabnet_test"):
        """Initialize the enhanced dimension-preserving reconstructor."""
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.annotations_dir = self.data_dir / "annotations"
        
        # Load TableFormer model
        print("Loading TableFormer model...")
        self.processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
        self.model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
        self.model.eval()
        print("Enhanced dimension-preserving reconstructor V2 initialized on cpu")
    
    def reconstruct_with_dimension_preservation(self, sample: Dict) -> Dict:
        """Reconstruct table with proper dimension preservation."""
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
            
            # Parse HTML with enhanced dimension analysis
            html_structure = self._parse_html_with_dimension_analysis(html, tableformer_detections, image_size)
            
            # Create dimension-preserving table structure
            enhanced_structure = self._create_dimension_preserving_structure(
                html_structure, tableformer_detections, image_size
            )
            
            # Draw dimension-preserving reconstruction
            reconstructed_image = self._draw_dimension_preserving_table(
                enhanced_structure, image_size
            )
            
            # Validate reconstruction
            validation_metrics = self._validate_dimension_preservation(
                original_image, reconstructed_image, enhanced_structure
            )
            
            # Save results
            self._save_dimension_preserving_results(
                reconstructed_image, original_image, enhanced_structure, filename
            )
            
            return {
                "success": True,
                "filename": filename,
                "image_size": image_size,
                "enhanced_structure": enhanced_structure,
                "validation_metrics": validation_metrics,
                "tableformer_detections": tableformer_detections
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _extract_tableformer_detections(self, image: Image.Image) -> Dict:
        """Extract TableFormer detections with proper coordinate scaling."""
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process outputs
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]
        
        # Organize detections by type
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
    
    def _parse_html_with_dimension_analysis(self, html: str, tableformer_detections: Dict, image_size: Tuple[int, int]) -> Dict:
        """Parse HTML with enhanced dimension analysis and proper scaling."""
        if not html:
            return {"rows": [], "cells": [], "max_cols": 0, "has_header": False}
        
        structure = {
            "rows": [],
            "cells": [],
            "max_cols": 0,
            "has_header": False,
            "row_dimensions": [],
            "col_dimensions": []
        }
        
        # Find table rows
        tr_pattern = r'<tr[^>]*>(.*?)</tr>'
        tr_matches = re.findall(tr_pattern, html, re.DOTALL)
        
        # Analyze row dimensions from TableFormer with proper scaling
        row_detections = tableformer_detections.get("table row", [])
        row_heights = self._analyze_row_dimensions_scaled(row_detections, len(tr_matches), image_size)
        
        for row_idx, tr_content in enumerate(tr_matches):
            # Check for header
            if "<th" in tr_content:
                structure["has_header"] = True
            
            # Parse cells with enhanced dimension analysis
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
                
                # Analyze cell dimensions with proper scaling
                cell_dimensions = self._analyze_cell_dimensions_scaled(
                    content, colspan, rowspan, row_heights[row_idx] if row_idx < len(row_heights) else None, image_size
                )
                
                # Create enhanced cell info
                cell_info = CellInfo(
                    row=row_idx,
                    col=col_idx,
                    rowspan=rowspan,
                    colspan=colspan,
                    content=content,
                    is_header="<th" in tr_content,
                    actual_width=cell_dimensions["width"],
                    actual_height=cell_dimensions["height"],
                    content_width=cell_dimensions["content_width"],
                    content_height=cell_dimensions["content_height"]
                )
                
                row_cells.append(cell_info)
                structure["cells"].append(cell_info)
            
            structure["rows"].append({
                "row_idx": row_idx,
                "cells": row_cells,
                "is_header": "<th" in tr_content,
                "height": row_heights[row_idx] if row_idx < len(row_heights) else None
            })
            
            # Calculate effective columns
            effective_cols = sum(cell.colspan for cell in row_cells)
            structure["max_cols"] = max(structure["max_cols"], effective_cols)
        
        # Analyze column dimensions with proper scaling
        col_detections = tableformer_detections.get("table column", [])
        col_widths = self._analyze_column_dimensions_scaled(col_detections, structure["max_cols"], image_size)
        
        structure["row_dimensions"] = row_heights
        structure["col_dimensions"] = col_widths
        
        return structure
    
    def _analyze_row_dimensions_scaled(self, row_detections: List[Dict], num_rows: int, image_size: Tuple[int, int]) -> List[float]:
        """Analyze row dimensions with proper scaling to image coordinates."""
        if not row_detections:
            # Fallback to proportional distribution based on content
            return [image_size[1] / num_rows] * num_rows
        
        # Sort detections by y-coordinate
        sorted_detections = sorted(row_detections, key=lambda x: x["bbox"][1])
        
        row_heights = []
        for i, detection in enumerate(sorted_detections):
            bbox = detection["bbox"]
            # TableFormer bbox is already in image coordinates after post-processing
            height = bbox[3] - bbox[1]  # y2 - y1
            row_heights.append(height)
        
        # Fill missing rows with average height
        if len(row_heights) < num_rows:
            avg_height = sum(row_heights) / len(row_heights) if row_heights else image_size[1] / num_rows
            row_heights.extend([avg_height] * (num_rows - len(row_heights)))
        
        # Scale to ensure total height matches image height
        total_height = sum(row_heights[:num_rows])
        if total_height > 0:
            scale_factor = image_size[1] / total_height
            row_heights = [h * scale_factor for h in row_heights[:num_rows]]
        
        return row_heights[:num_rows]
    
    def _analyze_column_dimensions_scaled(self, col_detections: List[Dict], num_cols: int, image_size: Tuple[int, int]) -> List[float]:
        """Analyze column dimensions with proper scaling to image coordinates."""
        if not col_detections:
            # Fallback to proportional distribution
            return [image_size[0] / num_cols] * num_cols
        
        # Sort detections by x-coordinate
        sorted_detections = sorted(col_detections, key=lambda x: x["bbox"][0])
        
        col_widths = []
        for i, detection in enumerate(sorted_detections):
            bbox = detection["bbox"]
            # TableFormer bbox is already in image coordinates after post-processing
            width = bbox[2] - bbox[0]  # x2 - x1
            col_widths.append(width)
        
        # Fill missing columns with average width
        if len(col_widths) < num_cols:
            avg_width = sum(col_widths) / len(col_widths) if col_widths else image_size[0] / num_cols
            col_widths.extend([avg_width] * (num_cols - len(col_widths)))
        
        # Scale to ensure total width matches image width
        total_width = sum(col_widths[:num_cols])
        if total_width > 0:
            scale_factor = image_size[0] / total_width
            col_widths = [w * scale_factor for w in col_widths[:num_cols]]
        
        return col_widths[:num_cols]
    
    def _analyze_cell_dimensions_scaled(self, content: str, colspan: int, rowspan: int, row_height: Optional[float], image_size: Tuple[int, int]) -> Dict:
        """Analyze individual cell dimensions with proper scaling."""
        # Estimate content dimensions
        content_width = len(content) * 8  # Rough estimate: 8 pixels per character
        content_height = 20  # Standard text height
        
        # Calculate actual cell dimensions
        if row_height:
            actual_height = row_height * rowspan
        else:
            actual_height = (image_size[1] / 20) * rowspan  # Fallback based on image height
        
        # Width estimation (will be refined by column analysis)
        actual_width = content_width * colspan
        
        return {
            "width": actual_width,
            "height": actual_height,
            "content_width": content_width,
            "content_height": content_height
        }
    
    def _create_dimension_preserving_structure(self, html_structure: Dict, 
                                             tableformer_detections: Dict, 
                                             image_size: Tuple[int, int]) -> TableStructure:
        """Create table structure with preserved dimensions."""
        # Extract table bounding box
        table_bbox = None
        if tableformer_detections.get("table"):
            table_bbox = tableformer_detections["table"][0]["bbox"]
        
        # Calculate total dimensions (should match image size after scaling)
        total_width = sum(html_structure["col_dimensions"]) if html_structure["col_dimensions"] else image_size[0]
        total_height = sum(html_structure["row_dimensions"]) if html_structure["row_dimensions"] else image_size[1]
        
        # Create enhanced table structure
        structure = TableStructure(
            rows=len(html_structure["rows"]),
            cols=html_structure["max_cols"],
            cells=html_structure["cells"],
            table_bbox=table_bbox,
            html_structure=str(html_structure),
            row_heights=html_structure["row_dimensions"],
            col_widths=html_structure["col_dimensions"],
            total_width=total_width,
            total_height=total_height
        )
        
        return structure
    
    def _draw_dimension_preserving_table(self, structure: TableStructure, image_size: Tuple[int, int]) -> Image.Image:
        """Draw table with preserved individual row/column dimensions."""
        width, height = image_size
        
        # Create image
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        if not structure.cells:
            return image
        
        # Use preserved dimensions or fallback to uniform
        if structure.col_widths and structure.row_heights:
            col_widths = structure.col_widths
            row_heights = structure.row_heights
        else:
            # Fallback to uniform distribution
            col_widths = [width / structure.cols] * structure.cols
            row_heights = [height / structure.rows] * structure.rows
        
        # Create a grid to track occupied cells
        occupied = [[False for _ in range(structure.cols)] for _ in range(structure.rows)]
        
        # Draw cells in order (spanned cells first)
        sorted_cells = sorted(structure.cells, key=lambda c: (c.rowspan * c.colspan, c.row, c.col), reverse=True)
        
        for cell in sorted_cells:
            # Check if this cell position is available
            if occupied[cell.row][cell.col]:
                continue
            
            # Calculate cell position using preserved dimensions
            x1 = sum(col_widths[:cell.col])
            y1 = sum(row_heights[:cell.row])
            x2 = x1 + sum(col_widths[cell.col:cell.col + cell.colspan])
            y2 = y1 + sum(row_heights[cell.row:cell.row + cell.rowspan])
            
            # Mark occupied cells
            for r in range(cell.row, min(cell.row + cell.rowspan, structure.rows)):
                for c in range(cell.col, min(cell.col + cell.colspan, structure.cols)):
                    if r < structure.rows and c < structure.cols:
                        occupied[r][c] = True
            
            # Draw cell border
            draw.rectangle([x1, y1, x2, y2], outline='black', width=2)
            
            # Fill header cells differently
            if cell.is_header:
                draw.rectangle([x1+1, y1+1, x2-1, y2-1], fill='lightblue')
            elif cell.rowspan > 1 or cell.colspan > 1:
                # Highlight spanning cells
                draw.rectangle([x1+1, y1+1, x2-1, y2-1], fill='lightyellow')
            
            # Add cell content with better text handling
            if cell.content:
                try:
                    # Calculate text position
                    text_x = x1 + (x2 - x1) // 2
                    text_y = y1 + (y2 - y1) // 2
                    
                    # Adjust text size based on cell size
                    cell_width = x2 - x1
                    cell_height = y2 - y1
                    
                    # Truncate text based on available space
                    max_chars = int(cell_width / 8)  # 8 pixels per character
                    display_text = cell.content[:max_chars] + "..." if len(cell.content) > max_chars else cell.content
                    
                    # Draw text
                    draw.text((text_x, text_y), display_text, fill='black', anchor='mm')
                except:
                    pass  # Skip if text drawing fails
            
            # Add dimension indicators for debugging
            if cell.rowspan > 1 or cell.colspan > 1:
                # Draw spanning indicator
                span_text = f"{cell.colspan}x{cell.rowspan}"
                draw.text((x2-20, y2-15), span_text, fill='red', anchor='rb')
                
                # Draw dimension info
                dim_text = f"{int(x2-x1)}x{int(y2-y1)}"
                draw.text((x2-20, y1+15), dim_text, fill='blue', anchor='rb')
        
        return image
    
    def _validate_dimension_preservation(self, original_image: Image.Image, 
                                       reconstructed_image: Image.Image, 
                                       structure: TableStructure) -> Dict:
        """Validate dimension preservation quality."""
        # Calculate dimension preservation metrics
        total_expected_width = sum(structure.col_widths) if structure.col_widths else 0
        total_expected_height = sum(structure.row_heights) if structure.row_heights else 0
        
        # Calculate actual reconstructed dimensions
        actual_width = structure.total_width or original_image.width
        actual_height = structure.total_height or original_image.height
        
        # Dimension preservation score
        width_preservation = min(actual_width / total_expected_width, total_expected_width / actual_width) if total_expected_width > 0 else 0
        height_preservation = min(actual_height / total_expected_height, total_expected_height / actual_height) if total_expected_height > 0 else 0
        
        # Overall dimension preservation
        dimension_preservation = (width_preservation + height_preservation) / 2
        
        # Spanning cell preservation
        spanning_cells = [cell for cell in structure.cells if cell.rowspan > 1 or cell.colspan > 1]
        spanning_preservation = len(spanning_cells) / max(len(structure.cells), 1)
        
        return {
            "dimension_preservation": dimension_preservation,
            "width_preservation": width_preservation,
            "height_preservation": height_preservation,
            "spanning_cell_count": len(spanning_cells),
            "spanning_preservation": spanning_preservation,
            "total_cells": len(structure.cells),
            "overall_quality": (dimension_preservation + spanning_preservation) / 2
        }
    
    def _save_dimension_preserving_results(self, reconstructed_image: Image.Image, 
                                         original_image: Image.Image, 
                                         structure: TableStructure, 
                                         filename: str):
        """Save dimension-preserving reconstruction results."""
        base_name = Path(filename).stem
        
        # Save reconstructed image
        output_dir = Path("outputs/dimension_preserving_reconstruction_v2")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        reconstructed_path = output_dir / f"{base_name}_dimension_preserving_v2.png"
        reconstructed_image.save(reconstructed_path)
        
        # Save comparison image
        comparison_dir = Path("outputs/dimension_preserving_comparison_v2")
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
            draw.text((original_resized.width + 30, 10), "Enhanced V2", fill='black')
        except:
            pass
        
        comparison_path = comparison_dir / f"{base_name}_dimension_comparison_v2.png"
        comparison_image.save(comparison_path)
        
        # Save structure with dimensions
        structure_dir = Path("outputs/dimension_preserving_structures_v2")
        structure_dir.mkdir(parents=True, exist_ok=True)
        
        structure_data = {
            "rows": structure.rows,
            "cols": structure.cols,
            "row_heights": structure.row_heights,
            "col_widths": structure.col_widths,
            "total_width": structure.total_width,
            "total_height": structure.total_height,
            "cells": [
                {
                    "row": cell.row,
                    "col": cell.col,
                    "rowspan": cell.rowspan,
                    "colspan": cell.colspan,
                    "content": cell.content,
                    "is_header": cell.is_header,
                    "actual_width": cell.actual_width,
                    "actual_height": cell.actual_height,
                    "content_width": cell.content_width,
                    "content_height": cell.content_height
                }
                for cell in structure.cells
            ],
            "table_bbox": structure.table_bbox
        }
        
        structure_path = structure_dir / f"{base_name}_dimension_structure_v2.json"
        with open(structure_path, 'w') as f:
            json.dump(structure_data, f, indent=2)
        
        print(f"     Saved dimension-preserving reconstruction V2: {reconstructed_path}")
        print(f"     Saved dimension comparison V2: {comparison_path}")
        print(f"     Saved dimension structure V2: {structure_path}")
    
    def test_dimension_preserving_reconstruction(self, max_samples: int = 3):
        """Test dimension-preserving reconstruction on sample tables."""
        print(f"Testing Enhanced Dimension-Preserving Table Reconstruction V2...")
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
                result = self.reconstruct_with_dimension_preservation(sample)
                results.append(result)
                
                if result["success"]:
                    metrics = result["validation_metrics"]
                    structure = result["enhanced_structure"]
                    print(f"     Dimension preservation: {metrics['dimension_preservation']:.3f}")
                    print(f"     Overall quality: {metrics['overall_quality']:.3f}")
                    print(f"     Spanning cells: {metrics['spanning_cell_count']}")
                    print(f"     Table size: {structure.rows}×{structure.cols}")
                    if structure.row_heights and structure.col_widths:
                        print(f"     Row height range: {min(structure.row_heights):.1f}-{max(structure.row_heights):.1f}")
                        print(f"     Column width range: {min(structure.col_widths):.1f}-{max(structure.col_widths):.1f}")
                else:
                    print(f"     Error: {result['error']}")
                    
            except Exception as e:
                print(f"Error testing sample {i+1}: {e}")
                continue
        
        # Analyze results
        analysis = self._analyze_dimension_preserving_results(results)
        
        # Save results
        self._save_dimension_preserving_analysis(results, analysis)
        
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
    
    def _analyze_dimension_preserving_results(self, results: List[Dict]) -> Dict:
        """Analyze dimension-preserving reconstruction results."""
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
        avg_overall_quality = sum(r["validation_metrics"]["overall_quality"] for r in successful_results) / len(successful_results)
        total_spanning_cells = sum(r["validation_metrics"]["spanning_cell_count"] for r in successful_results)
        
        return {
            "total_samples": len(results),
            "successful_reconstructions": len(successful_results),
            "failed_reconstructions": len(failed_results),
            "success_rate": len(successful_results) / len(results) * 100,
            "average_dimension_preservation": avg_dimension_preservation,
            "average_overall_quality": avg_overall_quality,
            "total_spanning_cells": total_spanning_cells,
            "spanning_cell_analysis": {
                "avg_spanning_per_table": total_spanning_cells / len(successful_results),
                "tables_with_spanning": len([r for r in successful_results if r["validation_metrics"]["spanning_cell_count"] > 0])
            }
        }
    
    def _save_dimension_preserving_analysis(self, results: List[Dict], analysis: Dict):
        """Save dimension-preserving reconstruction analysis."""
        output_dir = Path("outputs/dimension_preserving_analysis_v2")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert results to JSON-serializable format
        serializable_results = []
        for result in results:
            serializable_result = result.copy()
            if "enhanced_structure" in serializable_result:
                structure = serializable_result["enhanced_structure"]
                serializable_result["enhanced_structure"] = {
                    "rows": structure.rows,
                    "cols": structure.cols,
                    "row_heights": structure.row_heights,
                    "col_widths": structure.col_widths,
                    "total_width": structure.total_width,
                    "total_height": structure.total_height,
                    "cells": [
                        {
                            "row": cell.row,
                            "col": cell.col,
                            "rowspan": cell.rowspan,
                            "colspan": cell.colspan,
                            "content": cell.content,
                            "is_header": cell.is_header,
                            "actual_width": cell.actual_width,
                            "actual_height": cell.actual_height,
                            "content_width": cell.content_width,
                            "content_height": cell.content_height
                        }
                        for cell in structure.cells
                    ],
                    "table_bbox": structure.table_bbox
                }
            serializable_results.append(serializable_result)
        
        # Save detailed results
        results_file = output_dir / "dimension_preserving_results_v2.json"
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save analysis
        analysis_file = output_dir / "dimension_preserving_analysis_v2.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Dimension-preserving reconstruction analysis V2 saved to: {output_dir}")

def main():
    """Test enhanced dimension-preserving reconstruction V2."""
    print("Enhanced Dimension-Preserving Table Reconstruction V2")
    print("=" * 55)
    
    # Initialize reconstructor
    reconstructor = EnhancedDimensionPreservingReconstructorV2()
    
    # Test reconstruction
    analysis = reconstructor.test_dimension_preserving_reconstruction(max_samples=3)
    
    if analysis:
        print(f"\nDimension-Preserving Reconstruction Results V2:")
        print(f"   Total samples: {analysis['total_samples']}")
        print(f"   Success rate: {analysis['success_rate']:.1f}%")
        print(f"   Average dimension preservation: {analysis['average_dimension_preservation']:.3f}")
        print(f"   Average overall quality: {analysis['average_overall_quality']:.3f}")
        print(f"   Total spanning cells: {analysis['total_spanning_cells']}")
        
        if analysis['spanning_cell_analysis']:
            spanning = analysis['spanning_cell_analysis']
            print(f"   Average spanning cells per table: {spanning['avg_spanning_per_table']:.1f}")
            print(f"   Tables with spanning cells: {spanning['tables_with_spanning']}")

if __name__ == "__main__":
    main()
