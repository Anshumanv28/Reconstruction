#!/usr/bin/env python3
"""
Create corrected scaled average-based grid with proper row count
"""

import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_corrected_scaled_grid():
    """Create scaled average-based grid with correct row count"""
    
    print("=" * 60)
    print("CREATING CORRECTED SCALED AVERAGE-BASED GRID")
    print("=" * 60)
    
    # Read TableFormer results
    tableformer_path = "intermediate_outputs/tableformer_outputs/1zM9MmA6dM2_2dbMHiJKd_m-FRdPpNTR3lJUT_P1QuiE_tableformer_results.json"
    
    if not Path(tableformer_path).exists():
        print(f"❌ TableFormer results not found: {tableformer_path}")
        return
    
    with open(tableformer_path, 'r') as f:
        tableformer_data = json.load(f)
    
    # Extract image name from file path
    image_name = Path(tableformer_path).stem.replace('_tableformer_results', '')
    print(f"📄 Processing: {image_name}")
    
    results = tableformer_data.get('tableformer_results', [])
    print(f"📊 Total tables detected: {len(results)}")
    
    # Create output directory
    output_dir = Path("intermediate_outputs/grid_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, result in enumerate(results):
        table_id = i + 1
        print(f"\n🔧 PROCESSING TABLE {table_id}:")
        
        # Extract table bbox and cells from predict_details
        predict_details = result.get('predict_details', {})
        table_bbox = predict_details.get('table_bbox', [])
        cells = predict_details.get('prediction_bboxes_page', [])
        
        print(f"   📦 Table bbox: {table_bbox}")
        print(f"   📋 Cells detected by TableFormer: {len(cells)}")
        
        if not cells:
            print(f"   ⚠️  No cells to process for Table {table_id}")
            continue
        
        # Calculate table dimensions
        table_x1, table_y1, table_x2, table_y2 = table_bbox
        table_width = table_x2 - table_x1
        table_height = table_y2 - table_y1
        
        print(f"   📏 Table dimensions: {table_width:.1f} x {table_height:.1f}")
        
        # Convert cells to relative coordinates and analyze structure
        relative_cells = []
        for cell in cells:
            x1, y1, x2, y2 = cell
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
        
        # Analyze cell positions to determine grid structure with 5% tolerance
        grid_structure = analyze_cell_positions_for_averages(relative_cells, table_width, table_height)
        
        print(f"   🎯 Detected grid structure: {grid_structure['rows']} rows × {grid_structure['cols']} columns")
        
        # Calculate average dimensions per row and column
        avg_dimensions = calculate_average_dimensions(relative_cells, grid_structure)
        
        print(f"   📐 Intra-column average widths: {[f'{w:.1f}' for w in avg_dimensions['column_widths']]}")
        print(f"   📐 Intra-row average heights: {[f'{h:.1f}' for h in avg_dimensions['row_heights']]}")
        
        # Scale dimensions to fit table boundaries
        scaled_dimensions = scale_dimensions_to_fit_table(avg_dimensions, table_width, table_height)
        
        print(f"   📐 Scaled column widths: {[f'{w:.1f}' for w in scaled_dimensions['column_widths']]}")
        print(f"   📐 Scaled row heights: {[f'{h:.1f}' for h in scaled_dimensions['row_heights']]}")
        print(f"   📏 Total scaled width: {sum(scaled_dimensions['column_widths']):.1f} (target: {table_width:.1f})")
        print(f"   📏 Total scaled height: {sum(scaled_dimensions['row_heights']):.1f} (target: {table_height:.1f})")
        
        # Create scaled grid based on averages
        grid_cells = create_scaled_average_grid_cells(relative_cells, grid_structure, scaled_dimensions, table_bbox)
        
        print(f"   ✅ Created {len(grid_cells)} scaled grid cells")
        
        # Save detailed grid data
        grid_data = {
            'table_id': table_id,
            'table_bbox': table_bbox,
            'detected_rows': grid_structure['rows'],
            'detected_cols': grid_structure['cols'],
            'total_grid_cells': len(grid_cells),
            'original_detected_cells': len(cells),
            'intra_column_average_widths': avg_dimensions['column_widths'],
            'intra_row_average_heights': avg_dimensions['row_heights'],
            'scaled_column_widths': scaled_dimensions['column_widths'],
            'scaled_row_heights': scaled_dimensions['row_heights'],
            'scaling_factors': scaled_dimensions['scaling_factors'],
            'grid_cells': grid_cells,
            'original_cells': cells
        }
        
        # Save reconstruction data
        recon_path = output_dir / f"table_{table_id}_corrected_scaled_grid_data.json"
        with open(recon_path, 'w') as f:
            json.dump(grid_data, f, indent=2)
        
        print(f"   💾 Corrected scaled grid data saved: {recon_path}")
        
        # Create visual reconstruction with corrected scaled grid
        create_corrected_scaled_visualization(grid_data, output_dir)

def analyze_cell_positions_for_averages(relative_cells, table_width, table_height):
    """Analyze cell positions to determine grid structure for averaging with 5% tolerance"""
    
    # Group cells by rows and columns based on their positions with 5% tolerance
    cell_groups = group_cells_by_position(relative_cells, table_width, table_height)
    
    return {
        'rows': len(cell_groups['rows']),
        'cols': len(cell_groups['cols']),
        'cell_groups': cell_groups
    }

def group_cells_by_position(relative_cells, table_width, table_height):
    """Group cells by their row and column positions with 5% tolerance"""
    
    # Sort cells by y position (rows)
    cells_by_y = sorted(relative_cells, key=lambda c: c['center_y'])
    
    # Group cells into rows with 5% tolerance
    rows = []
    current_row = [cells_by_y[0]]
    
    for cell in cells_by_y[1:]:
        # If cell is close to current row (within 5% of table height)
        if abs(cell['center_y'] - current_row[0]['center_y']) < table_height * 0.05:
            current_row.append(cell)
        else:
            rows.append(current_row)
            current_row = [cell]
    
    rows.append(current_row)
    
    # Sort cells by x position (columns)
    cells_by_x = sorted(relative_cells, key=lambda c: c['center_x'])
    
    # Group cells into columns with 5% tolerance
    cols = []
    current_col = [cells_by_x[0]]
    
    for cell in cells_by_x[1:]:
        # If cell is close to current column (within 5% of table width)
        if abs(cell['center_x'] - current_col[0]['center_x']) < table_width * 0.05:
            current_col.append(cell)
        else:
            cols.append(current_col)
            current_col = [cell]
    
    cols.append(current_col)
    
    return {
        'rows': rows,
        'cols': cols
    }

def calculate_average_dimensions(relative_cells, grid_structure):
    """Calculate average dimensions for each row and column (intra-row/column averaging)"""
    
    cell_groups = grid_structure['cell_groups']
    rows = cell_groups['rows']
    cols = cell_groups['cols']
    
    # Calculate average height FOR EACH ROW (intra-row averaging)
    row_heights = []
    for row_idx, row_cells in enumerate(rows):
        heights = [cell['height'] for cell in row_cells]
        avg_height = sum(heights) / len(heights)
        row_heights.append(avg_height)
        print(f"   📏 Row {row_idx}: heights {[f'{h:.1f}' for h in heights]} → avg {avg_height:.1f}")
    
    # Calculate average width FOR EACH COLUMN (intra-column averaging)
    column_widths = []
    for col_idx, col_cells in enumerate(cols):
        widths = [cell['width'] for cell in col_cells]
        avg_width = sum(widths) / len(widths)
        column_widths.append(avg_width)
        print(f"   📏 Col {col_idx}: widths {[f'{w:.1f}' for w in widths]} → avg {avg_width:.1f}")
    
    return {
        'row_heights': row_heights,
        'column_widths': column_widths
    }

def scale_dimensions_to_fit_table(avg_dimensions, table_width, table_height):
    """Scale average dimensions to fit the exact table boundaries"""
    
    raw_column_widths = avg_dimensions['column_widths']
    raw_row_heights = avg_dimensions['row_heights']
    
    # Calculate current totals
    current_width = sum(raw_column_widths)
    current_height = sum(raw_row_heights)
    
    # Calculate scaling factors
    width_scale = table_width / current_width
    height_scale = table_height / current_height
    
    print(f"   🔧 Scaling factors: width={width_scale:.3f}, height={height_scale:.3f}")
    
    # Scale dimensions
    scaled_column_widths = [w * width_scale for w in raw_column_widths]
    scaled_row_heights = [h * height_scale for h in raw_row_heights]
    
    return {
        'column_widths': scaled_column_widths,
        'row_heights': scaled_row_heights,
        'scaling_factors': {
            'width_scale': width_scale,
            'height_scale': height_scale
        }
    }

def create_scaled_average_grid_cells(relative_cells, grid_structure, scaled_dimensions, table_bbox):
    """Create grid cells based on scaled average dimensions"""
    
    table_x1, table_y1, table_x2, table_y2 = table_bbox
    cell_groups = grid_structure['cell_groups']
    rows = cell_groups['rows']
    cols = cell_groups['cols']
    row_heights = scaled_dimensions['row_heights']
    column_widths = scaled_dimensions['column_widths']
    
    # Create grid cells
    grid_cells = []
    cell_id = 0
    
    current_y = 0
    for row_idx, row_cells in enumerate(rows):
        current_x = 0
        for col_idx, col_cells in enumerate(cols):
            # Use scaled average dimensions
            width = column_widths[col_idx]
            height = row_heights[row_idx]
            
            # Create cell with scaled average dimensions
            rel_x1 = current_x
            rel_y1 = current_y
            rel_x2 = current_x + width
            rel_y2 = current_y + height
            
            # Convert back to absolute coordinates
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
                'is_corrected_scaled_average_based': True
            })
            cell_id += 1
            
            current_x += width
        current_y += height
    
    return grid_cells

def create_corrected_scaled_visualization(grid_data, output_dir):
    """Create visual reconstruction using corrected scaled average-based grid"""
    
    table_id = grid_data['table_id']
    table_bbox = grid_data['table_bbox']
    grid_cells = grid_data['grid_cells']
    original_cells = grid_data['original_cells']
    column_widths = grid_data['scaled_column_widths']
    row_heights = grid_data['scaled_row_heights']
    scaling_factors = grid_data['scaling_factors']
    
    print(f"   🎨 Creating visual reconstruction with corrected scaled grid...")
    
    # Calculate dimensions
    table_x1, table_y1, table_x2, table_y2 = table_bbox
    table_width = table_x2 - table_x1
    table_height = table_y2 - table_y1
    
    # Create image
    img_width = max(1200, int(table_width) + 300)
    img_height = max(900, int(table_height) + 300)
    
    img = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Try to load font
    try:
        font = ImageFont.truetype("arial.ttf", 10)
        title_font = ImageFont.truetype("arial.ttf", 16)
        small_font = ImageFont.truetype("arial.ttf", 8)
    except:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Draw title
    title = f"Table {table_id} - Corrected Scaled Grid (9 Rows)"
    draw.text((20, 20), title, fill='black', font=title_font)
    
    # Draw info
    info_text = f"Grid: {len(row_heights)}×{len(column_widths)} = {len(grid_cells)} cells | Detected: {len(original_cells)} cells"
    draw.text((20, 50), info_text, fill='gray', font=font)
    
    # Draw scaling info
    scale_text = f"Scaling: width×{scaling_factors['width_scale']:.3f}, height×{scaling_factors['height_scale']:.3f}"
    draw.text((20, 70), scale_text, fill='blue', font=small_font)
    
    # Draw scaled dimensions
    width_text = f"Scaled column widths: {[f'{w:.0f}' for w in column_widths]}"
    height_text = f"Scaled row heights: {[f'{h:.0f}' for h in row_heights]}"
    draw.text((20, 85), width_text, fill='blue', font=small_font)
    draw.text((20, 100), height_text, fill='green', font=small_font)
    
    # Calculate offset to center the table
    offset_x = (img_width - table_width) // 2
    offset_y = 130
    
    # Draw table boundary
    table_rect = [offset_x, offset_y, offset_x + table_width, offset_y + table_height]
    draw.rectangle(table_rect, outline='black', width=3)
    
    # Draw corrected scaled average-based grid cells
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink', 'lightgray', 'lightcyan', 'lightcoral', 'lightsteelblue', 'lightseagreen']
    
    for cell in grid_cells:
        row = cell['row']
        col = cell['col']
        rel_bbox = cell['relative_bbox']
        width = cell['width']
        height = cell['height']
        
        x1, y1, x2, y2 = rel_bbox
        
        # Draw cell with scaled average dimensions
        cell_rect = [offset_x + x1, offset_y + y1, offset_x + x2, offset_y + y2]
        color = colors[row % len(colors)]
        draw.rectangle(cell_rect, outline='black', width=1, fill=color)
        
        # Draw cell label
        cell_label = f"R{row}C{col}"
        text_x = offset_x + x1 + 2
        text_y = offset_y + y1 + 2
        
        # Check if text fits
        if width > 25 and height > 12:
            draw.text((text_x, text_y), cell_label, fill='black', font=font)
        else:
            draw.text((text_x, text_y), str(cell['cell_id']), fill='black', font=small_font)
        
        # Draw dimensions
        if width > 40 and height > 20:
            dim_text = f"{width:.0f}×{height:.0f}"
            draw.text((text_x, text_y + 15), dim_text, fill='gray', font=small_font)
    
    # Draw original detected cells in red
    for i, cell in enumerate(original_cells):
        x1, y1, x2, y2 = cell
        # Convert to relative coordinates
        rel_x1 = x1 - table_x1
        rel_y1 = y1 - table_y1
        rel_x2 = x2 - table_x1
        rel_y2 = y2 - table_y1
        
        # Draw detected cell
        detected_rect = [
            offset_x + rel_x1, offset_y + rel_y1,
            offset_x + rel_x2, offset_y + rel_y2
        ]
        draw.rectangle(detected_rect, outline='red', width=2)
        draw.text((offset_x + rel_x1 + 2, offset_y + rel_y1 + 2), str(i+1), fill='red', font=small_font)
    
    # Draw legend
    legend_y = offset_y + table_height + 20
    draw.text((offset_x, legend_y), "Legend:", fill='black', font=font)
    draw.text((offset_x, legend_y + 20), "Colored cells = Corrected scaled grid (9 rows, fits table boundaries)", fill='black', font=font)
    draw.text((offset_x, legend_y + 40), "Red outlines = Original detected cells", fill='red', font=font)
    draw.text((offset_x, legend_y + 60), "Numbers = Scaled dimensions (width×height)", fill='gray', font=font)
    
    # Save image
    output_path = output_dir / f"table_{table_id}_corrected_scaled_grid_reconstruction.png"
    img.save(output_path)
    print(f"   🖼️  Corrected scaled grid visualization saved: {output_path}")

def main():
    """Main function"""
    print("🚀 CREATING CORRECTED SCALED AVERAGE-BASED GRID")
    print("=" * 60)
    
    create_corrected_scaled_grid()
    
    print("\n" + "=" * 60)
    print("✅ CORRECTED SCALED AVERAGE-BASED GRID CREATED!")
    print("📁 Check 'intermediate_outputs/grid_output/' for outputs")
    print("=" * 60)

if __name__ == "__main__":
    main()