#!/usr/bin/env python3
"""
Stage 3: Grid Reconstruction (Median-based)

This module handles:
- Grid structure analysis using median-based positioning
- Cell dimension calculation and scaling
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

class GridReconstructor:
    """Handles grid reconstruction using median-based positioning"""
    
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
        """Run grid reconstruction using median-based positioning"""
        print("  📐 Running grid reconstruction...")
        
        # Extract data from new structure
        tableformer_results = layout_data.get("tableformer_results", [])
        text_blocks = ocr_data.get("text_blocks", [])
        
        if not tableformer_results:
            print("    ⚠️  No TableFormer results found")
            return {"tables": [], "total_tables": 0, "total_cells": 0}
        
        print(f"    Found {len(tableformer_results)} table(s) to reconstruct")
        
        reconstructed_tables = []
        
        for i, tf_result in enumerate(tableformer_results):
            print(f"    Processing table {i+1}/{len(tableformer_results)}")
            
            # Extract table information from TableFormer result
            predict_details = tf_result.get("predict_details", {})
            table_bbox = predict_details.get("table_bbox", [0, 0, 100, 100])
            cell_bboxes = predict_details.get("prediction_bboxes_page", [])
            
            if not cell_bboxes:
                print(f"      ⚠️  No cell bboxes found for table {i+1}")
                continue
            
            # Convert TableFormer cell bboxes to our format
            table_cells = []
            for j, cell_bbox in enumerate(cell_bboxes):
                if len(cell_bbox) >= 4:
                    cell = {
                        "cell_id": j,
                        "bbox": cell_bbox[:4],  # [x1, y1, x2, y2]
                        "row_span": 1,
                        "col_span": 1
                    }
                    table_cells.append(cell)
            
            if not table_cells:
                print(f"      ⚠️  No valid cells found for table {i+1}")
                continue
            
            # Analyze grid structure
            grid_structure = self._analyze_grid_structure(table_cells, table_bbox)
            
            # Calculate median dimensions
            median_dimensions = self._calculate_median_dimensions(table_cells, grid_structure)
            
            # Scale dimensions to fit table
            scaled_dimensions = self._scale_dimensions_to_fit_table(
                median_dimensions, table_bbox
            )
            
            # Create grid cells
            grid_cells = self._create_grid_cells(
                table_cells, grid_structure, scaled_dimensions, table_bbox
            )
            
            # Map OCR text to grid cells
            cell_content = self._map_ocr_to_grid_cells(grid_cells, text_blocks, table_bbox)
            
            reconstructed_table = {
                "table_id": i + 1,
                "table_bbox": table_bbox,
                "grid_structure": grid_structure,
                "median_dimensions": median_dimensions,
                "scaled_dimensions": scaled_dimensions,
                "grid_cells": grid_cells,
                "cell_content": cell_content,
                "original_cells": table_cells,
                "tableformer_result": tf_result
            }
            
            reconstructed_tables.append(reconstructed_table)
            print(f"      ✅ Table {i+1}: {grid_structure['rows']} rows × {grid_structure['cols']} columns, {len(grid_cells)} cells")
        
        return {
            "tables": reconstructed_tables,
            "processing_timestamp": datetime.now().isoformat(),
            "total_tables": len(reconstructed_tables),
            "total_cells": sum(len(table["grid_cells"]) for table in reconstructed_tables)
        }
    
    def _analyze_grid_structure(self, table_cells: List[Dict], table_bbox: List[float]) -> Dict:
        """Analyze cell positions to determine grid structure"""
        if not table_cells:
            return {"rows": 0, "cols": 0}
        
        # Convert cells to relative coordinates
        table_x1, table_y1, table_x2, table_y2 = table_bbox
        relative_cells = []
        
        for cell in table_cells:
            if 'bbox' in cell:
                x1, y1, x2, y2 = cell['bbox']
                rel_x1 = x1 - table_x1
                rel_y1 = y1 - table_y1
                rel_x2 = x2 - table_x1
                rel_y2 = y2 - table_y1
                
                relative_cells.append({
                    'bbox': [rel_x1, rel_y1, rel_x2, rel_y2],
                    'width': rel_x2 - rel_x1,
                    'height': rel_y2 - rel_y1,
                    'center_x': (rel_x1 + rel_x2) / 2,
                    'center_y': (rel_y1 + rel_y2) / 2
                })
        
        # Group cells by rows and columns
        rows, cols = self._group_cells_by_position(relative_cells)
        
        return {
            "rows": len(rows),
            "cols": len(cols),
            "cell_groups": {"rows": rows, "cols": cols}
        }
    
    def _group_cells_by_position(self, relative_cells: List[Dict]) -> Tuple[List, List]:
        """Group cells into rows and columns based on position"""
        if not relative_cells:
            return [], []
        
        # Sort cells by position
        sorted_cells = sorted(relative_cells, key=lambda c: (c['center_y'], c['center_x']))
        
        # Group into rows
        y_tolerance = 15
        row_groups = []
        used_indices = set()
        
        for i, cell in enumerate(sorted_cells):
            if i in used_indices:
                continue
            
            row_group = [cell]
            used_indices.add(i)
            cell_y = cell['center_y']
            
            # Find cells at similar y-positions
            for j, other_cell in enumerate(sorted_cells):
                if j in used_indices:
                    continue
                
                other_y = other_cell['center_y']
                if abs(cell_y - other_y) <= y_tolerance:
                    row_group.append(other_cell)
                    used_indices.add(j)
            
            # Sort row group by x-position
            row_group_sorted = sorted(row_group, key=lambda c: c['center_x'])
            row_groups.append(row_group_sorted)
        
        # Sort rows by y-position
        row_groups_sorted = sorted(row_groups, key=lambda rg: sum(c['center_y'] for c in rg) / len(rg))
        
        # Create columns
        max_cols = max(len(row) for row in row_groups_sorted) if row_groups_sorted else 0
        cols = []
        
        for col_idx in range(max_cols):
            col_cells = []
            for row in row_groups_sorted:
                if col_idx < len(row):
                    col_cells.append(row[col_idx])
            cols.append(col_cells)
        
        return row_groups_sorted, cols
    
    def _calculate_median_dimensions(self, table_cells: List[Dict], grid_structure: Dict) -> Dict:
        """Calculate median dimensions for rows and columns"""
        cell_groups = grid_structure.get("cell_groups", {})
        rows = cell_groups.get("rows", [])
        cols = cell_groups.get("cols", [])
        
        # Calculate row heights
        row_heights = []
        for row in rows:
            if row:
                max_height = max(cell['height'] for cell in row)
                row_heights.append(max_height)
        
        # Calculate column widths
        column_widths = []
        for col in cols:
            if col:
                max_width = max(cell['width'] for cell in col)
                column_widths.append(max_width)
        
        return {
            "row_heights": row_heights,
            "column_widths": column_widths
        }
    
    def _scale_dimensions_to_fit_table(self, median_dimensions: Dict, table_bbox: List[float]) -> Dict:
        """Scale dimensions to fit table boundaries"""
        table_x1, table_y1, table_x2, table_y2 = table_bbox
        table_width = table_x2 - table_x1
        table_height = table_y2 - table_y1
        
        raw_column_widths = median_dimensions.get("column_widths", [])
        raw_row_heights = median_dimensions.get("row_heights", [])
        
        if not raw_column_widths or not raw_row_heights:
            return {
                "column_widths": [],
                "row_heights": [],
                "scaling_factors": {"width_scale": 1.0, "height_scale": 1.0}
            }
        
        # Calculate scaling factors
        current_width = sum(raw_column_widths)
        current_height = sum(raw_row_heights)
        
        width_scale = table_width / current_width if current_width > 0 else 1.0
        height_scale = table_height / current_height if current_height > 0 else 1.0
        
        # Scale dimensions
        scaled_column_widths = [w * width_scale for w in raw_column_widths]
        scaled_row_heights = [h * height_scale for h in raw_row_heights]
        
        return {
            "column_widths": scaled_column_widths,
            "row_heights": scaled_row_heights,
            "scaling_factors": {
                "width_scale": width_scale,
                "height_scale": height_scale
            }
        }
    
    def _create_grid_cells(self, table_cells: List[Dict], grid_structure: Dict, 
                          scaled_dimensions: Dict, table_bbox: List[float]) -> List[Dict]:
        """Create grid cells based on scaled dimensions"""
        table_x1, table_y1, table_x2, table_y2 = table_bbox
        cell_groups = grid_structure.get("cell_groups", {})
        rows = cell_groups.get("rows", [])
        cols = cell_groups.get("cols", [])
        
        row_heights = scaled_dimensions.get("row_heights", [])
        column_widths = scaled_dimensions.get("column_widths", [])
        
        grid_cells = []
        cell_id = 0
        
        current_y = 0
        for row_idx, row in enumerate(rows):
            current_x = 0
            for col_idx, col in enumerate(cols):
                if row_idx < len(row_heights) and col_idx < len(column_widths):
                    width = column_widths[col_idx]
                    height = row_heights[row_idx]
                    
                    # Create cell
                    rel_x1 = current_x
                    rel_y1 = current_y
                    rel_x2 = current_x + width
                    rel_y2 = current_y + height
                    
                    # Convert to absolute coordinates
                    abs_x1 = rel_x1 + table_x1
                    abs_y1 = rel_y1 + table_y1
                    abs_x2 = rel_x2 + table_x1
                    abs_y2 = rel_y2 + table_y1
                    
                    grid_cells.append({
                        'cell_id': cell_id,
                        'row': row_idx,
                        'col': col_idx,
                        'relative_bbox': [rel_x1, rel_y1, rel_x2, rel_y2],
                        'absolute_bbox': [abs_x1, abs_y1, abs_x2, abs_y2],
                        'width': width,
                        'height': height,
                        'is_median_based': True
                    })
                    cell_id += 1
                    
                    current_x += width
            current_y += height
        
        return grid_cells
    
    def _map_ocr_to_grid_cells(self, grid_cells: List[Dict], text_blocks: List[Dict], table_bbox: List[float]) -> Dict:
        """Map OCR text blocks to grid cells"""
        cell_content = {}
        
        for cell in grid_cells:
            cell_id = cell['cell_id']
            cell_content[cell_id] = {
                'cell': cell,
                'text_blocks': [],
                'text': ''
            }
        
        # Map text blocks to cells
        for text_block in text_blocks:
            best_cell = self._find_best_cell_for_text(text_block, grid_cells)
            if best_cell:
                cell_id = best_cell['cell_id']
                cell_content[cell_id]['text_blocks'].append(text_block)
        
        # Combine text within each cell
        for cell_id, content in cell_content.items():
            if content['text_blocks']:
                sorted_blocks = sorted(content['text_blocks'], 
                                     key=lambda tb: (tb['bbox'][1], tb['bbox'][0]))
                text_parts = [block['text'] for block in sorted_blocks]
                content['text'] = ' '.join(text_parts)
        
        return cell_content
    
    def _find_best_cell_for_text(self, text_block: Dict, grid_cells: List[Dict]) -> Optional[Dict]:
        """Find the best grid cell for a text block"""
        text_bbox = text_block['bbox']
        text_center_x = (text_bbox[0] + text_bbox[2]) / 2
        text_center_y = (text_bbox[1] + text_bbox[3]) / 2
        
        best_cell = None
        best_overlap_score = 0
        
        for cell in grid_cells:
            cell_bbox = cell['absolute_bbox']
            
            # Check if text center is in cell
            if (cell_bbox[0] <= text_center_x <= cell_bbox[2] and 
                cell_bbox[1] <= text_center_y <= cell_bbox[3]):
                
                # Calculate overlap area
                overlap_x1 = max(text_bbox[0], cell_bbox[0])
                overlap_y1 = max(text_bbox[1], cell_bbox[1])
                overlap_x2 = min(text_bbox[2], cell_bbox[2])
                overlap_y2 = min(text_bbox[3], cell_bbox[3])
                
                if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                    text_area = (text_bbox[2] - text_bbox[0]) * (text_bbox[3] - text_bbox[1])
                    overlap_score = overlap_area / text_area if text_area > 0 else 0
                    
                    if overlap_score > best_overlap_score:
                        best_overlap_score = overlap_score
                        best_cell = cell
        
        return best_cell
    
    def _create_visualization(self, input_file: str, output_prefix: str, intermediate_dir: Path, 
                            grid_result: Dict) -> Path:
        """Create visualization of grid reconstruction results"""
        print("  🎨 Creating grid reconstruction visualization...")
        
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
                font = ImageFont.truetype("arial.ttf", 10)
                small_font = ImageFont.truetype("arial.ttf", 8)
            except:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Draw grid reconstruction
            tables = grid_result.get("tables", [])
            for table in tables:
                self._draw_reconstructed_table(draw, table, font, small_font)
            
            # Save visualization
            canvas.save(output_path)
            print(f"    ✅ Visualization saved: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"    ❌ Error creating visualization: {e}")
            # Create a simple placeholder image
            placeholder = Image.new('RGB', (800, 600), 'lightgray')
            placeholder.save(output_path)
            return output_path
    
    def _draw_reconstructed_table(self, draw: ImageDraw.Draw, table: Dict, font, small_font):
        """Draw a reconstructed table"""
        grid_cells = table.get("grid_cells", [])
        cell_content = table.get("cell_content", {})
        
        # Colors for different rows
        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink', 'lightgray']
        
        for cell in grid_cells:
            row = cell['row']
            col = cell['col']
            bbox = cell['absolute_bbox']
            
            x1, y1, x2, y2 = bbox
            
            # Draw cell
            color = colors[row % len(colors)]
            draw.rectangle([x1, y1, x2, y2], outline='black', width=1, fill=color)
            
            # Draw cell label
            cell_label = f"R{row}C{col}"
            draw.text((x1+2, y1+2), cell_label, fill='black', font=font)
            
            # Draw text content if available
            cell_id = cell['cell_id']
            if cell_id in cell_content:
                text = cell_content[cell_id]['text']
                if text.strip():
                    # Truncate text if too long
                    display_text = text[:20] + "..." if len(text) > 20 else text
                    draw.text((x1+2, y1+15), display_text, fill='black', font=small_font)
    
    def _create_coordinates_json(self, output_prefix: str, intermediate_dir: Path, 
                               grid_result: Dict) -> Path:
        """Create structured coordinate JSON"""
        print("  📄 Creating coordinates JSON...")
        
        # Create output paths
        stage_dir = intermediate_dir / "stage3_grid_reconstruction"
        coordinates_dir = stage_dir / "coordinates"
        coordinates_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = coordinates_dir / f"{output_prefix}_stage3_coordinates.json"
        
        # Create structured data
        coordinates_data = {
            "stage": 3,
            "stage_name": "grid_reconstruction",
            "processing_timestamp": grid_result.get("processing_timestamp", datetime.now().isoformat()),
            "tables": grid_result.get("tables", []),
            "summary": {
                "total_tables": grid_result.get("total_tables", 0),
                "total_cells": grid_result.get("total_cells", 0),
                "tables_with_content": sum(1 for table in grid_result.get("tables", []) 
                                         if any(cell_content.get('text', '').strip() 
                                              for cell_content in table.get("cell_content", {}).values()))
            }
        }
        
        # Save coordinates JSON
        with open(output_path, 'w') as f:
            json.dump(coordinates_data, f, indent=2)
        
        print(f"    ✅ Coordinates JSON saved: {output_path}")
        return output_path

def main():
    """Main function for testing Stage 3"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage 3: Grid Reconstruction")
    parser.add_argument("--input", "-i", required=True, help="Input image file")
    parser.add_argument("--output-prefix", "-o", help="Output prefix (default: input filename)")
    parser.add_argument("--intermediate-dir", default="intermediate_outputs", help="Intermediate outputs directory")
    
    args = parser.parse_args()
    
    # Set default output prefix
    if not args.output_prefix:
        args.output_prefix = Path(args.input).stem
    
    # Initialize reconstructor
    reconstructor = GridReconstructor()
    
    # Process image
    result = reconstructor.process_image(args.input, args.output_prefix, Path(args.intermediate_dir))
    
    # Print results
    if result.get("success", False):
        print(f"\n✅ Stage 3 completed successfully!")
        print(f"   Tables processed: {result.get('table_count', 0)}")
        print(f"   Total cells: {result.get('total_cells', 0)}")
        print(f"   Visualization: {result.get('visualization_path', 'N/A')}")
        print(f"   Coordinates: {result.get('coordinates_path', 'N/A')}")
    else:
        print(f"\n❌ Stage 3 failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()