#!/usr/bin/env python3
"""
Stage 3: Grid Reconstruction (Visual Structure Reconstruction)

This module handles:
- Visual table structure reconstruction from actual table appearance
- Logical cell merging based on visual patterns
- Header hierarchy detection
- Creates visualization output (PNG)
- Creates coordinate JSON with structured data
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from datetime import datetime
from collections import defaultdict


class GridReconstructor:
    """Handles grid reconstruction by understanding visual table structure"""
    
    def __init__(self):
        self.stage_name = "grid_reconstruction"
    
    def process_image(self, input_file: str, output_prefix: str, intermediate_dir: Path) -> Dict:
        """Process a single image for grid reconstruction"""
        print(f"🔍 Processing: {input_file}")
        
        try:
            # Load previous stage data
            layout_data = self._load_stage1_data(output_prefix, intermediate_dir)
            ocr_data = self._load_stage2_data(output_prefix, intermediate_dir)
            
            # Run grid reconstruction
            grid_result = self._run_grid_reconstruction(layout_data, ocr_data)
            
            # Create visualization
            visualization_path = self._create_visualization(
                input_file, output_prefix, intermediate_dir, grid_result
            )
            
            # Create coordinate JSON
            coordinates_path = self._create_coordinates_json(
                output_prefix, intermediate_dir, grid_result
            )
            
            return {
                "success": True,
                "table_count": len(grid_result.get("tables", [])),
                "total_cells": sum(len(table.get("grid_cells", [])) for table in grid_result.get("tables", [])),
                "visualization_path": str(visualization_path),
                "coordinates_path": str(coordinates_path),
                "grid_result": grid_result
            }
            
        except Exception as e:
            print(f"❌ Error in grid reconstruction: {e}")
            return {"success": False, "error": str(e)}
    
    def _load_stage1_data(self, output_prefix: str, intermediate_dir: Path) -> Dict:
        """Load Stage 1 data (layout and table detection) from separate directories"""
        # Load layout data from stage1/layout_outputs
        stage1_dir = intermediate_dir / "stage1_layout_table"
        layout_dir = stage1_dir / "layout_outputs"
        layout_file = layout_dir / f"{output_prefix}_layout_coordinates.json"
        
        # Load table data from stage1/tableformer_outputs
        tableformer_dir = stage1_dir / "tableformer_outputs"
        tableformer_file = tableformer_dir / f"{output_prefix}_tableformer_coordinates.json"
        
        layout_data = {"layout_elements": []}
        table_data = {"tableformer_results": []}
        
        if layout_file.exists():
            with open(layout_file, 'r') as f:
                layout_data = json.load(f)
            print(f"    ✅ Loaded layout data: {len(layout_data.get('layout_elements', []))} elements")
        else:
            print(f"    ⚠️  Layout data not found: {layout_file}")
        
        if tableformer_file.exists():
            with open(tableformer_file, 'r') as f:
                table_data = json.load(f)
            print(f"    ✅ Loaded table data: {len(table_data.get('tableformer_results', []))} tables")
        else:
            print(f"    ⚠️  Table data not found: {tableformer_file}")
        
        # Combine the data
        combined_data = {
            "layout_elements": layout_data.get("layout_elements", []),
            "tableformer_results": table_data.get("tableformer_results", []),
            "total_tables": table_data.get("summary", {}).get("total_tables", 0)
        }
        
        return combined_data
    
    def _load_stage2_data(self, output_prefix: str, intermediate_dir: Path) -> Dict:
        """Load Stage 2 data (OCR extraction)"""
        stage2_dir = intermediate_dir / "stage2_ocr_extraction" / "coordinates"
        
        # Try to load OCR data
        ocr_file = stage2_dir / f"{output_prefix}_stage2_coordinates.json"
        if ocr_file.exists():
            with open(ocr_file, 'r') as f:
                return json.load(f)
        else:
            print(f"    ⚠️  Stage 2 data not found: {ocr_file}")
            return {"text_blocks": []}
    
    def _run_grid_reconstruction(self, layout_data: Dict, ocr_data: Dict) -> Dict:
        """NEW: Visual structure reconstruction approach"""
        print("  📐 Running visual structure reconstruction...")
        
        tableformer_results = layout_data.get("tableformer_results", [])
        text_blocks = ocr_data.get("text_blocks", [])
        
        if not tableformer_results:
            print("    ⚠️  No TableFormer results found")
            return {"tables": [], "total_tables": 0, "total_cells": 0}
        
        print(f"    Found {len(tableformer_results)} table(s) to reconstruct")
        
        reconstructed_tables = []
        
        for i, tf_result in enumerate(tableformer_results):
            print(f"    Processing table {i+1}/{len(tableformer_results)}")
            
            # Extract all cell detections
            cell_detections = self._extract_all_cell_detections(tf_result)
            if not cell_detections:
                print(f"      ⚠️  No cell detections found for table {i+1}")
                continue
            
            print(f"      Found {len(cell_detections)} cell detections")
            
            # Reconstruct visual structure
            visual_structure = self._reconstruct_visual_structure(cell_detections, text_blocks)
            if not visual_structure:
                print(f"      ⚠️  Could not reconstruct visual structure for table {i+1}")
                continue
            
            print(f"      Reconstructed visual structure: {visual_structure['logical_rows']} rows × {visual_structure['logical_cols']} columns")
            print(f"      Created {len(visual_structure['logical_cells'])} logical cells")
            
            # Build final table
            table_data = self._build_visual_table(visual_structure, i + 1)
            if table_data:
                reconstructed_tables.append(table_data)
                
                spanning_count = sum(1 for cell in table_data["grid_cells"] 
                                   if cell.get("row_span", 1) > 1 or cell.get("col_span", 1) > 1)
                header_count = sum(1 for cell in table_data["grid_cells"] 
                                 if cell.get("is_header", False))
                print(f"      ✅ Visual Table {i+1}: {spanning_count} spanning cells, {header_count} headers")
        
        return {
            "tables": reconstructed_tables,
            "processing_timestamp": datetime.now().isoformat(),
            "total_tables": len(reconstructed_tables),
            "total_cells": sum(len(table["grid_cells"]) for table in reconstructed_tables)
        }
    
    def _extract_all_cell_detections(self, tf_result: Dict) -> List[Dict]:
        """Extract all cell detections from TableFormer result"""
        detections = []
        
        # Method 1: Enhanced structure
        enhanced_structure = tf_result.get("enhanced_table_structure", [])
        if enhanced_structure:
            print("      Using enhanced_table_structure")
            for table_struct in enhanced_structure:
                for row in table_struct.get("rows", []):
                    for cell in row.get("cells", []):
                        detections.append({
                            "bbox": cell.get("bbox", [0, 0, 10, 10]),
                            "row_id": cell.get("row", 0),
                            "col_id": cell.get("col", 0),
                            "row_span": cell.get("rowspan", 1),
                            "col_span": cell.get("colspan", 1),
                            "is_header": cell.get("is_header", False),
                            "cell_type": cell.get("cell_type", "cell"),
                            "text": cell.get("text", ""),
                            "source": "enhanced"
                        })
            return detections
        
        # Method 2: TableCells
        if "tablecells" in tf_result:
            print("      Using tablecells")
            for cell_data in tf_result["tablecells"]:
                detections.append({
                    "bbox": cell_data.get("bbox", [0, 0, 10, 10]),
                    "row_id": cell_data.get("rowid", 0),
                    "col_id": cell_data.get("columnid", 0),
                    "row_span": cell_data.get("rowspanval", cell_data.get("rowspan", 1)),
                    "col_span": cell_data.get("colspanval", cell_data.get("colspan", 1)),
                    "is_header": cell_data.get("celltype", "cell") in ["columnheader", "header"],
                    "cell_type": cell_data.get("celltype", "cell"),
                    "text": "",
                    "source": "tablecells"
                })
            return detections
        
        # Method 3: Basic detections
        predict_details = tf_result.get("predict_details", {})
        cell_bboxes = predict_details.get("prediction_bboxes_page", [])
        
        if cell_bboxes:
            print("      Using prediction_bboxes_page")
            for j, bbox in enumerate(cell_bboxes):
                if len(bbox) >= 4:
                    detections.append({
                        "bbox": bbox[:4],
                        "row_id": 0,  # Will be determined by visual analysis
                        "col_id": 0,
                        "row_span": 1,
                        "col_span": 1,
                        "is_header": False,
                        "cell_type": "cell",
                        "text": "",
                        "source": "basic"
                    })
        
        return detections
    
    def _reconstruct_visual_structure(self, detections: List[Dict], text_blocks: List[Dict]) -> Optional[Dict]:
        """Reconstruct visual table structure from detections"""
        if not detections:
            return None
        
        # Calculate table bounds
        table_bbox = self._calculate_table_bounds_from_detections(detections)
        
        # Perform visual clustering
        visual_clusters = self._perform_visual_clustering(detections, table_bbox)
        
        # Create logical grid
        logical_grid = self._create_logical_grid_from_clusters(visual_clusters, table_bbox)
        
        # Detect headers and spanning
        logical_grid = self._detect_headers_and_spanning(logical_grid, text_blocks)
        
        return logical_grid
    
    def _calculate_table_bounds_from_detections(self, detections: List[Dict]) -> List[float]:
        """Calculate overall table bounds"""
        min_x = min(d["bbox"][0] for d in detections)
        min_y = min(d["bbox"][1] for d in detections)
        max_x = max(d["bbox"][2] for d in detections)
        max_y = max(d["bbox"][3] for d in detections)
        
        return [min_x, min_y, max_x, max_y]
    
    def _perform_visual_clustering(self, detections: List[Dict], table_bbox: List[float]) -> Dict:
        """Cluster detections into visual rows and columns"""
        # Sort by position
        sorted_detections = sorted(detections, key=lambda d: (d["bbox"][1], d["bbox"][0]))
        
        # Group into visual rows
        visual_rows = []
        row_tolerance = 15  # Pixels
        used_indices = set()
        
        for i, detection in enumerate(sorted_detections):
            if i in used_indices:
                continue
            
            row_group = [detection]
            used_indices.add(i)
            detection_y = (detection["bbox"][1] + detection["bbox"][3]) / 2
            
            # Find detections at similar Y positions
            for j, other_detection in enumerate(sorted_detections):
                if j in used_indices:
                    continue
                
                other_y = (other_detection["bbox"][1] + other_detection["bbox"][3]) / 2
                if abs(detection_y - other_y) <= row_tolerance:
                    row_group.append(other_detection)
                    used_indices.add(j)
            
            # Sort row by X position
            row_group.sort(key=lambda d: d["bbox"][0])
            visual_rows.append(row_group)
        
        # Sort rows by Y position
        visual_rows.sort(key=lambda row: min(d["bbox"][1] for d in row))
        
        print(f"        Visual clustering: {len(visual_rows)} visual rows")
        for i, row in enumerate(visual_rows):
            print(f"          Row {i}: {len(row)} detections")
        
        return {
            "table_bbox": table_bbox,
            "visual_rows": visual_rows,
            "num_visual_rows": len(visual_rows),
            "max_visual_cols": max(len(row) for row in visual_rows) if visual_rows else 0
        }
    
    def _create_logical_grid_from_clusters(self, visual_clusters: Dict, table_bbox: List[float]) -> Dict:
        """Create logical grid from visual clusters"""
        visual_rows = visual_clusters["visual_rows"]
        
        # Create logical cells by merging visually adjacent detections
        logical_cells = []
        
        for row_idx, row_detections in enumerate(visual_rows):
            logical_row_cells = self._merge_row_detections(row_detections, row_idx)
            logical_cells.extend(logical_row_cells)
        
        # Calculate grid dimensions
        if logical_cells:
            logical_rows = max(cell["logical_row"] for cell in logical_cells) + 1
            logical_cols = max(cell["logical_col"] + cell["col_span"] - 1 for cell in logical_cells) + 1
        else:
            logical_rows = logical_cols = 0
        
        return {
            "table_bbox": table_bbox,
            "logical_cells": logical_cells,
            "logical_rows": logical_rows,
            "logical_cols": logical_cols,
            "visual_clusters": visual_clusters
        }
    
    def _merge_row_detections(self, row_detections: List[Dict], row_idx: int) -> List[Dict]:
        """Merge adjacent detections in a row into logical cells"""
        if not row_detections:
            return []
        
        # Sort by X position
        sorted_detections = sorted(row_detections, key=lambda d: d["bbox"][0])
        
        logical_cells = []
        col_idx = 0
        
        i = 0
        while i < len(sorted_detections):
            # Start a new logical cell
            current_detection = sorted_detections[i]
            merge_candidates = [current_detection]
            
            # Look for adjacent detections to merge
            j = i + 1
            while j < len(sorted_detections):
                next_detection = sorted_detections[j]
                
                # Check if detections should be merged (close X positions, similar heights)
                if self._should_merge_detections(current_detection, next_detection):
                    merge_candidates.append(next_detection)
                    j += 1
                else:
                    break
            
            # Create logical cell from merged candidates
            logical_cell = self._create_logical_cell(merge_candidates, row_idx, col_idx)
            logical_cells.append(logical_cell)
            
            col_idx += logical_cell["col_span"]
            i = j
        
        return logical_cells
    
    def _should_merge_detections(self, det1: Dict, det2: Dict) -> bool:
        """Determine if two detections should be merged into one logical cell"""
        bbox1, bbox2 = det1["bbox"], det2["bbox"]
        
        # Calculate gap between detections
        gap = bbox2[0] - bbox1[2]
        
        # Calculate heights
        height1 = bbox1[3] - bbox1[1]
        height2 = bbox2[3] - bbox2[1]
        
        # Merge if:
        # 1. Small gap (< 10 pixels)
        # 2. Similar heights (within 30% difference)
        # 3. Overlapping Y ranges
        
        small_gap = gap < 10
        similar_heights = abs(height1 - height2) / max(height1, height2) < 0.3
        
        # Check Y overlap
        y_overlap = not (bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1])
        
        should_merge = small_gap and similar_heights and y_overlap
        
        if should_merge:
            print(f"            Merging detections: gap={gap:.1f}, heights={height1:.1f},{height2:.1f}")
        
        return should_merge
    
    def _create_logical_cell(self, detections: List[Dict], row_idx: int, col_idx: int) -> Dict:
        """Create a logical cell from merged detections"""
        if not detections:
            return {}
        
        # Calculate merged bounding box
        min_x = min(d["bbox"][0] for d in detections)
        min_y = min(d["bbox"][1] for d in detections)
        max_x = max(d["bbox"][2] for d in detections)
        max_y = max(d["bbox"][3] for d in detections)
        
        merged_bbox = [min_x, min_y, max_x, max_y]
        
        # Determine logical spans
        col_span = len(detections)  # Number of merged detections becomes column span
        
        # Use the maximum span values from individual detections
        max_row_span = max(d.get("row_span", 1) for d in detections)
        
        # Combine text and properties
        is_header = any(d.get("is_header", False) for d in detections)
        cell_types = [d.get("cell_type", "cell") for d in detections]
        combined_cell_type = "header" if is_header else "cell"
        
        # Combine text from detections
        texts = [d.get("text", "") for d in detections if d.get("text", "").strip()]
        combined_text = " ".join(texts)
        
        return {
            "logical_row": row_idx,
            "logical_col": col_idx,
            "row_span": max_row_span,
            "col_span": col_span,
            "bbox": merged_bbox,
            "is_header": is_header,
            "cell_type": combined_cell_type,
            "text": combined_text,
            "merged_detections": detections,
            "width": max_x - min_x,
            "height": max_y - min_y
        }
    
    def _detect_headers_and_spanning(self, logical_grid: Dict, text_blocks: List[Dict]) -> Dict:
        """Detect headers and refine spanning based on text content"""
        logical_cells = logical_grid.get("logical_cells", [])
        
        if not logical_cells:
            return logical_grid
        
        # Enhanced header detection
        for cell in logical_cells:
            # First row is likely header
            if cell["logical_row"] == 0:
                cell["is_header"] = True
                cell["cell_type"] = "header"
            
            # Cells with specific patterns are headers
            text = cell.get("text", "").lower()
            if any(pattern in text for pattern in ["december", "total", "header", "date", "time"]):
                cell["is_header"] = True
                cell["cell_type"] = "header"
        
        # Map OCR text to logical cells
        for cell in logical_cells:
            cell_text_blocks = self._find_text_blocks_for_cell(cell, text_blocks)
            if cell_text_blocks:
                # Combine OCR text with existing text
                ocr_text = " ".join(block["text"] for block in cell_text_blocks)
                if cell.get("text", "").strip():
                    cell["text"] = f"{cell['text']} {ocr_text}"
                else:
                    cell["text"] = ocr_text
        
        return logical_grid
    
    def _find_text_blocks_for_cell(self, cell: Dict, text_blocks: List[Dict]) -> List[Dict]:
        """Find text blocks that belong to a logical cell"""
        cell_bbox = cell["bbox"]
        matching_blocks = []
        
        for text_block in text_blocks:
            text_bbox = text_block["bbox"]
            
            # Calculate overlap
            overlap_x1 = max(cell_bbox[0], text_bbox[0])
            overlap_y1 = max(cell_bbox[1], text_bbox[1])
            overlap_x2 = min(cell_bbox[2], text_bbox[2])
            overlap_y2 = min(cell_bbox[3], text_bbox[3])
            
            if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                text_area = (text_bbox[2] - text_bbox[0]) * (text_bbox[3] - text_bbox[1])
                
                # If significant overlap, assign to cell
                if text_area > 0 and overlap_area / text_area > 0.3:
                    matching_blocks.append(text_block)
        
        return matching_blocks
    
    def _build_visual_table(self, visual_structure: Dict, table_id: int) -> Dict:
        """Build final table structure from visual analysis"""
        logical_cells = visual_structure.get("logical_cells", [])
        table_bbox = visual_structure.get("table_bbox", [0, 0, 100, 100])
        logical_rows = visual_structure.get("logical_rows", 0)
        logical_cols = visual_structure.get("logical_cols", 0)
        
        # Create final grid cells
        grid_cells = []
        for i, logical_cell in enumerate(logical_cells):
            grid_cell = {
                'cell_id': i,
                'row': logical_cell["logical_row"],
                'col': logical_cell["logical_col"],
                'row_span': logical_cell.get("row_span", 1),
                'col_span': logical_cell.get("col_span", 1),
                'relative_bbox': [0, 0, logical_cell["width"], logical_cell["height"]],  # Simplified
                'absolute_bbox': logical_cell["bbox"],
                'width': logical_cell["width"],
                'height': logical_cell["height"],
                'is_header': logical_cell.get("is_header", False),
                'cell_type': logical_cell.get("cell_type", "cell"),
                'text': logical_cell.get("text", ""),
                'confidence': 1.0,
                'is_median_based': False
            }
            grid_cells.append(grid_cell)
            
            # Debug output
            if grid_cell['row_span'] > 1 or grid_cell['col_span'] > 1:
                print(f"        VISUAL SPANNING: R{grid_cell['row']}C{grid_cell['col']} = {grid_cell['row_span']}×{grid_cell['col_span']}")
        
        # Create cell content mapping
        cell_content = {}
        for cell in grid_cells:
            cell_content[cell['cell_id']] = {
                'cell': cell,
                'text_blocks': [],
                'text': cell.get('text', '')
            }
        
        return {
            "table_id": table_id,
            "table_bbox": table_bbox,
            "grid_structure": {
                "rows": logical_rows,
                "cols": logical_cols,
                "cell_groups": {"rows": [], "cols": []}
            },
            "median_dimensions": {
                "row_heights": [],
                "column_widths": []
            },
            "scaled_dimensions": {
                "row_heights": [],
                "column_widths": [],
                "scaling_factors": {"width_scale": 1.0, "height_scale": 1.0}
            },
            "grid_cells": grid_cells,
            "cell_content": cell_content,
            "original_cells": logical_cells,
            "tableformer_result": {},
            "has_spanning": any(c.get("row_span", 1) > 1 or c.get("col_span", 1) > 1 for c in grid_cells),
            "has_headers": any(c.get("is_header", False) for c in grid_cells),
            "visual_structure": visual_structure
        }
    
    def _create_visualization(self, input_file: str, output_prefix: str, intermediate_dir: Path, 
                            grid_result: Dict) -> Path:
        """Create visual structure visualization"""
        print("  🎨 Creating visual structure visualization...")
        
        # Create output paths
        stage_dir = intermediate_dir / "stage3_grid_reconstruction"
        viz_dir = stage_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = viz_dir / f"{output_prefix}_grid_reconstruction.png"
        
        try:
            # Load original image
            original_img = Image.open(input_file)
            img_width, img_height = original_img.size
            
            # Create visualization canvas
            canvas = Image.new('RGB', (img_width, img_height), 'white')
            canvas.paste(original_img)
            draw = ImageDraw.Draw(canvas)
            
            # Try to load font
            try:
                font = ImageFont.truetype("arial.ttf", 16)
                small_font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Draw visual structure
            tables = grid_result.get("tables", [])
            for table in tables:
                self._draw_visual_table(draw, table, font, small_font)
            
            # Save visualization
            canvas.save(output_path)
            print(f"    ✅ Visual structure visualization saved: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"    ❌ Error creating visualization: {e}")
            placeholder = Image.new('RGB', (800, 600), 'lightgray')
            placeholder.save(output_path)
            return output_path
    
    def _draw_visual_table(self, draw: ImageDraw.Draw, table: Dict, font, small_font):
        """Draw table with visual structure representation"""
        grid_cells = table.get("grid_cells", [])
        
        # Visual structure colors
        colors = {
            'header': '#B3E5FC',      # Light blue for headers
            'spanning': '#FFECB3',    # Light amber for spanning
            'regular': '#F3E5F5'      # Light purple for regular
        }
        
        for cell in grid_cells:
            row, col = cell['row'], cell['col']
            row_span, col_span = cell.get('row_span', 1), cell.get('col_span', 1)
            is_header = cell.get('is_header', False)
            bbox = cell['absolute_bbox']
            
            x1, y1, x2, y2 = bbox
            
            # Visual styling
            if is_header:
                fill_color = colors['header']
                outline_color = 'darkblue'
                outline_width = 3
            elif row_span > 1 or col_span > 1:
                fill_color = colors['spanning']
                outline_color = 'darkorange'
                outline_width = 3
            else:
                fill_color = colors['regular']
                outline_color = 'purple'
                outline_width = 2
            
            # Draw visual cell
            draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=outline_width, fill=fill_color)
            
            # Visual cell label
            if row_span > 1 or col_span > 1:
                label = f"VISUAL R{row}C{col} ({row_span}×{col_span})"
            else:
                label = f"VISUAL R{row}C{col}"
            
            if is_header:
                label = f"HDR {label}"
            
            # Draw label with strong background
            text_bbox = draw.textbbox((x1+5, y1+5), label, font=font)
            draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], 
                         fill='white', outline='black', width=1)
            draw.text((x1+5, y1+5), label, fill='black', font=font)
            
            # Draw visual content
            text = cell.get('text', '')
            if text.strip():
                # Enhanced text rendering
                available_width = max(100, x2 - x1 - 20)
                chars_per_line = max(8, int(available_width / 10))
                
                words = text.split()
                lines = []
                current_line = ""
                
                for word in words:
                    if len(current_line + " " + word) <= chars_per_line:
                        current_line += (" " + word) if current_line else word
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                
                if current_line:
                    lines.append(current_line)
                
                # Draw text with background
                y_offset = 30
                for line in lines[:3]:
                    if y1 + y_offset + 20 < y2:
                        text_bbox = draw.textbbox((x1+5, y1+y_offset), line, font=small_font)
                        draw.rectangle([text_bbox[0]-1, text_bbox[1]-1, text_bbox[2]+1, text_bbox[3]+1], 
                                     fill='lightyellow', outline=None)
                        draw.text((x1+5, y1+y_offset), line, fill='darkgreen', font=small_font)
                        y_offset += 18
    
    def _create_coordinates_json(self, output_prefix: str, intermediate_dir: Path, 
                               grid_result: Dict) -> Path:
        """Create coordinates JSON with visual structure info"""
        print("  📄 Creating visual structure coordinates JSON...")
        
        # Create output paths
        stage_dir = intermediate_dir / "stage3_grid_reconstruction"
        coordinates_dir = stage_dir / "coordinates"
        coordinates_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = coordinates_dir / f"{output_prefix}_stage3_coordinates.json"
        
        # Create visual structure data
        coordinates_data = {
            "stage": 3,
            "stage_name": "grid_reconstruction",
            "reconstruction_type": "visual_structure_aware",
            "processing_timestamp": grid_result.get("processing_timestamp", datetime.now().isoformat()),
            "tables": grid_result.get("tables", []),
            "summary": {
                "total_tables": grid_result.get("total_tables", 0),
                "total_visual_cells": grid_result.get("total_cells", 0),
                "tables_with_content": sum(1 for table in grid_result.get("tables", []) 
                                         if any(cell_content.get('text', '').strip() 
                                              for cell_content in table.get("cell_content", {}).values())),
                "tables_with_spanning": sum(1 for table in grid_result.get("tables", []) 
                                          if table.get("has_spanning", False)),
                "tables_with_headers": sum(1 for table in grid_result.get("tables", []) 
                                         if table.get("has_headers", False)),
                "total_spanning_cells": sum(
                    len([c for c in table.get("grid_cells", []) 
                         if c.get("row_span", 1) > 1 or c.get("col_span", 1) > 1])
                    for table in grid_result.get("tables", [])
                ),
                "total_header_cells": sum(
                    len([c for c in table.get("grid_cells", []) 
                         if c.get("is_header", False)])
                    for table in grid_result.get("tables", [])
                ),
                "visual_structure_summary": [
                    {
                        "table_id": table.get("table_id"),
                        "visual_dimensions": f"{table.get('grid_structure', {}).get('rows', 0)}×{table.get('grid_structure', {}).get('cols', 0)}",
                        "visual_cells": len(table.get("grid_cells", [])),
                        "merged_from_detections": sum(
                            len(c.get("merged_detections", []))
                            for c in table.get("original_cells", [])
                        )
                    }
                    for table in grid_result.get("tables", [])
                ]
            }
        }
        
        # Save coordinates JSON
        with open(output_path, 'w') as f:
            json.dump(coordinates_data, f, indent=2)
        
        print(f"    ✅ Visual structure coordinates JSON saved: {output_path}")
        return output_path


def main():
    """Main function for testing Visual Structure-Aware Stage 3"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage 3: Visual Structure-Aware Grid Reconstruction")
    parser.add_argument("--input", "-i", required=True, help="Input image file")
    parser.add_argument("--output-prefix", "-o", help="Output prefix (default: input filename)")
    parser.add_argument("--intermediate-dir", default="intermediate_outputs", help="Intermediate outputs directory")
    
    args = parser.parse_args()
    
    # Set default output prefix
    if not args.output_prefix:
        args.output_prefix = Path(args.input).stem
    
    # Initialize visual reconstructor
    reconstructor = GridReconstructor()
    
    # Process image
    result = reconstructor.process_image(args.input, args.output_prefix, Path(args.intermediate_dir))
    
    # Print visual results
    if result.get("success", False):
        print(f"\n✅ Visual Structure-Aware Stage 3 completed successfully!")
        print(f"   Tables processed: {result.get('table_count', 0)}")
        print(f"   Visual cells: {result.get('total_cells', 0)}")
        print(f"   Visualization: {result.get('visualization_path', 'N/A')}")
        print(f"   Coordinates: {result.get('coordinates_path', 'N/A')}")
    else:
        print(f"\n❌ Visual Structure-Aware Stage 3 failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
