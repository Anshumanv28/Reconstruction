#!/usr/bin/env python3
"""
Visualize TableFormer Grid Analysis Results
Creates visual overlays showing detected grid structures on original images
"""

import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def visualize_tableformer_grid(image_name="1zM9MmA6dM2_2dbMHiJKd_m-FRdPpNTR3lJUT_P1QuiE"):
    """Visualize TableFormer grid analysis results"""
    
    print("=" * 60)
    print("VISUALIZING TABLEFORMER GRID ANALYSIS")
    print("=" * 60)
    
    # File paths
    tableformer_results_path = f"intermediate_outputs/tableformer_outputs/{image_name}_tableformer_results.json"
    original_image_path = f"pipe_input/{image_name}.png"
    output_path = f"intermediate_outputs/former_grid_outputs/{image_name}_tableformer_grid_lines_only.png"
    json_output_path = f"intermediate_outputs/former_grid_outputs/{image_name}_tableformer_grid_data.json"
    
    # Check if files exist
    if not Path(tableformer_results_path).exists():
        print(f"❌ TableFormer results not found: {tableformer_results_path}")
        return
    
    if not Path(original_image_path).exists():
        print(f"❌ Original image not found: {original_image_path}")
        return
    
    # Load tableformer results (contains both cell detections and grid analysis)
    with open(tableformer_results_path, 'r') as f:
        tableformer_data = json.load(f)
    
    # Extract grid data from tableformer results
    grid_data = {"estimated_grids": tableformer_data.get('estimated_grids', [])}
    
    # Load original image
    img = Image.open(original_image_path)
    draw = ImageDraw.Draw(img)
    
    # Try to load fonts
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        small_font = ImageFont.truetype("arial.ttf", 10)
        tiny_font = ImageFont.truetype("arial.ttf", 8)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
        tiny_font = ImageFont.load_default()
    
    print(f"📄 Processing: {image_name}")
    print(f"📊 Total tables detected: {len(grid_data['estimated_grids'])}")
    
    # Colors for different rows
    row_colors = [
        'lightblue', 'lightgreen', 'lightyellow', 'lightpink', 
        'lightgray', 'lightcyan', 'lightcoral', 'lightsteelblue', 
        'lightseagreen', 'lightgoldenrodyellow'
    ]
    
    # Process each table
    for i, table_info in enumerate(grid_data['estimated_grids']):
        table_id = i + 1
        table_bbox = table_info['table_bbox']
        grid_cells = table_info['grid_cells']
        estimated_rows = table_info['estimated_rows']
        estimated_cols = table_info['estimated_cols']
        
        print(f"\n🔧 TABLE {table_id}:")
        print(f"   📦 Bbox: {[f'{x:.1f}' for x in table_bbox]}")
        
        # Calculate table dimensions
        x1, y1, x2, y2 = table_bbox
        table_width = x2 - x1
        table_height = y2 - y1
        
        # Determine actual grid structure from detected cells
        if i < len(tableformer_results):
            tf_table = tableformer_results[i]
            predict_details = tf_table.get('predict_details', {})
            tf_table_cells = predict_details.get('table_cells', [])
            
            if tf_table_cells:
                actual_rows = len(set(cell.get('row_id', 0) for cell in tf_table_cells))
                actual_cols = len(set(cell.get('column_id', 0) for cell in tf_table_cells))
                print(f"   📊 Grid Analysis: {estimated_rows} rows × {estimated_cols} columns = {len(grid_cells)} cells")
                print(f"   📊 Detected Cells: {actual_rows} rows × {actual_cols} columns = {len(tf_table_cells)} cells")
                print(f"   📏 Dimensions: {table_width:.1f} × {table_height:.1f}")
                
                # Use detected cells structure for display
                estimated_rows = actual_rows
                estimated_cols = actual_cols
            else:
                print(f"   📊 Grid: {estimated_rows} rows × {estimated_cols} columns = {len(grid_cells)} cells")
                print(f"   📏 Dimensions: {table_width:.1f} × {table_height:.1f}")
        else:
            print(f"   📊 Grid: {estimated_rows} rows × {estimated_cols} columns = {len(grid_cells)} cells")
            print(f"   📏 Dimensions: {table_width:.1f} × {table_height:.1f}")
        
        # Draw table boundary (thick red outline)
        draw.rectangle([x1, y1, x2, y2], outline='red', width=4)
        
        # Draw table title
        title_text = f"Table {table_id}: {estimated_rows}×{estimated_cols} Grid"
        draw.text((x1, y1-25), title_text, fill='red', font=font)
        
        # Skip drawing grid cells - only show grid lines
        
        # Draw grid lines if available
        grid_lines = table_info.get('grid_lines', {})
        horizontal_lines = grid_lines.get('horizontal', [])
        vertical_lines = grid_lines.get('vertical', [])
        
        if horizontal_lines or vertical_lines:
            print(f"   📏 Grid lines: {len(horizontal_lines)} horizontal, {len(vertical_lines)} vertical")
            
            # Draw horizontal lines (relative to table)
            for line in horizontal_lines:
                y = y1 + line['y']
                x1_line = x1 + line['x1']
                x2_line = x1 + line['x2']
                draw.line([x1_line, y, x2_line, y], fill='blue', width=1)
            
            # Draw vertical lines (relative to table)
            for line in vertical_lines:
                x = x1 + line['x']
                y1_line = y1 + line['y1']
                y2_line = y1 + line['y2']
                draw.line([x, y1_line, x, y2_line], fill='green', width=1)
    
    # Add legend
    legend_y = 20
    draw.text((20, legend_y), "TableFormer Grid Lines Only", fill='black', font=font)
    draw.text((20, legend_y + 25), "Red rectangles = Table boundaries", fill='red', font=small_font)
    draw.text((20, legend_y + 40), "Blue lines = Horizontal grid lines", fill='blue', font=small_font)
    draw.text((20, legend_y + 55), "Green lines = Vertical grid lines", fill='green', font=small_font)
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save the visualization
    img.save(output_path)
    print(f"\n✅ Grid analysis visualization saved: {output_path}")
    
    # Extract tableformer results for detailed cell info
    tableformer_results = tableformer_data.get('tableformer_results', [])
    
    # Create and save JSON output with processed data
    json_data = {
        "image_name": image_name,
        "total_tables": len(grid_data['estimated_grids']),
        "processing_timestamp": "N/A",
        "tableformer_grid_analysis": []
    }
    
    for i, table_info in enumerate(grid_data['estimated_grids']):
        table_id = i + 1
        table_bbox = table_info['table_bbox']
        grid_cells = table_info['grid_cells']
        estimated_rows = table_info['estimated_rows']
        estimated_cols = table_info['estimated_cols']
        grid_lines = table_info.get('grid_lines', {})
        
        # Calculate table dimensions
        x1, y1, x2, y2 = table_bbox
        table_width = x2 - x1
        table_height = y2 - y1
        
        # Get detailed cell information from tableformer results
        detailed_cells = []
        if i < len(tableformer_results):
            tf_table = tableformer_results[i]
            predict_details = tf_table.get('predict_details', {})
            tf_cells = predict_details.get('prediction_bboxes_page', [])
            tf_table_cells = predict_details.get('table_cells', [])
            
            # Convert TF cells to relative coordinates
            for j, tf_cell_bbox in enumerate(tf_cells):
                rel_x1 = tf_cell_bbox[0] - x1
                rel_y1 = tf_cell_bbox[1] - y1
                rel_x2 = tf_cell_bbox[2] - x1
                rel_y2 = tf_cell_bbox[3] - y1
                
                detailed_cell = {
                    "cell_id": j,
                    "absolute_bbox": tf_cell_bbox,
                    "relative_bbox": [rel_x1, rel_y1, rel_x2, rel_y2],
                    "width": rel_x2 - rel_x1,
                    "height": rel_y2 - rel_y1,
                    "center_x": (rel_x1 + rel_x2) / 2,
                    "center_y": (rel_y1 + rel_y2) / 2,
                    "source": "tableformer_detection"
                }
                
                # Add table cell info if available
                if j < len(tf_table_cells):
                    tf_table_cell = tf_table_cells[j]
                    detailed_cell.update({
                        "row_id": tf_table_cell.get('row_id'),
                        "column_id": tf_table_cell.get('column_id'),
                        "cell_class": tf_table_cell.get('cell_class'),
                        "label": tf_table_cell.get('label'),
                        "multicol_tag": tf_table_cell.get('multicol_tag', "")
                    })
                
                detailed_cells.append(detailed_cell)
        
        # Process grid cells for JSON output (from grid analysis)
        processed_cells = []
        for cell in grid_cells:
            bbox = cell['bbox']
            cell_width = bbox[2] - bbox[0]
            cell_height = bbox[3] - bbox[1]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            processed_cell = {
                "cell_id": cell['cell_id'],
                "row": cell['row'],
                "col": cell['col'],
                "absolute_bbox": bbox,
                "relative_bbox": [
                    bbox[0] - x1,  # rel_x1
                    bbox[1] - y1,  # rel_y1
                    bbox[2] - x1,  # rel_x2
                    bbox[3] - y1   # rel_y2
                ],
                "width": cell_width,
                "height": cell_height,
                "center_x": center_x,
                "center_y": center_y,
                "label": cell.get('label', 'fcel'),
                "confidence": cell.get('confidence', 0.3),
                "is_inferred": cell.get('is_inferred', True)
            }
            processed_cells.append(processed_cell)
        
        table_data = {
            "table_id": table_id,
            "table_bbox": table_bbox,
            "table_dimensions": {
                "width": table_width,
                "height": table_height
            },
            "grid_structure": {
                "rows": estimated_rows,
                "columns": estimated_cols,
                "total_grid_cells": len(grid_cells),
                "total_detected_cells": len(detailed_cells)
            },
            "grid_cells": processed_cells,
            "detected_cells": detailed_cells,
            "grid_lines": {
                "horizontal_count": len(grid_lines.get('horizontal', [])),
                "vertical_count": len(grid_lines.get('vertical', [])),
                "horizontal_lines": grid_lines.get('horizontal', []),
                "vertical_lines": grid_lines.get('vertical', [])
            },
            "confidence": table_info.get('confidence', 0.9),
            "method": table_info.get('method', 'cells'),
            "data_sources": {
                "grid_structure": "grid_analysis.json",
                "cell_detections": "tableformer_results.json"
            }
        }
        
        json_data["tableformer_grid_analysis"].append(table_data)
    
    # Save JSON output
    with open(json_output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"✅ TableFormer grid data saved: {json_output_path}")
    
    # Print summary
    print(f"\n📊 SUMMARY:")
    for i, table_info in enumerate(grid_data['estimated_grids']):
        table_id = i + 1
        estimated_rows = table_info['estimated_rows']
        estimated_cols = table_info['estimated_cols']
        grid_cells = table_info['grid_cells']
        print(f"   Table {table_id}: {estimated_rows}×{estimated_cols} = {len(grid_cells)} cells")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        image_name = sys.argv[1]
    else:
        image_name = "1zM9MmA6dM2_2dbMHiJKd_m-FRdPpNTR3lJUT_P1QuiE"
    
    visualize_tableformer_grid(image_name)
    
    print("\n" + "=" * 60)
    print("✅ VISUALIZATION COMPLETE!")
    print("📁 Check 'intermediate_outputs/former_grid_outputs/' for the visualization")
    print("=" * 60)

if __name__ == "__main__":
    main()
