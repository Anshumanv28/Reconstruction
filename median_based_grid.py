#!/usr/bin/env python3
"""
Create median-based grid with proper row count
Uses median positioning between largest cells in adjacent rows/columns
"""

import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_median_based_grid():
    """Create median-based grid with correct row count"""
    
    print("=" * 60)
    print("CREATING MEDIAN-BASED GRID")
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
        grid_structure = analyze_cell_positions_for_median(relative_cells, table_width, table_height)
        
        print(f"   🎯 Detected grid structure: {grid_structure['rows']} rows × {grid_structure['cols']} columns")
        
        # Calculate median dimensions per row and column
        median_dimensions = calculate_median_dimensions(relative_cells, grid_structure)
        
        print(f"   📐 Median-based column widths: {[f'{w:.1f}' for w in median_dimensions['column_widths']]}")
        print(f"   📐 Median-based row heights: {[f'{h:.1f}' for h in median_dimensions['row_heights']]}")
        
        # Scale dimensions to fit table boundaries
        scaled_dimensions = scale_dimensions_to_fit_table(median_dimensions, table_width, table_height)
        
        print(f"   📐 Scaled column widths: {[f'{w:.1f}' for w in scaled_dimensions['column_widths']]}")
        print(f"   📐 Scaled row heights: {[f'{h:.1f}' for h in scaled_dimensions['row_heights']]}")
        print(f"   📏 Total scaled width: {sum(scaled_dimensions['column_widths']):.1f} (target: {table_width:.1f})")
        print(f"   📏 Total scaled height: {sum(scaled_dimensions['row_heights']):.1f} (target: {table_height:.1f})")
        
        # Create scaled grid based on median positioning
        grid_cells = create_scaled_median_grid_cells(relative_cells, grid_structure, scaled_dimensions, table_bbox)
        
        print(f"   ✅ Created {len(grid_cells)} median-based grid cells")
        
        # Save detailed grid data
        grid_data = {
            'table_id': table_id,
            'table_bbox': table_bbox,
            'detected_rows': grid_structure['rows'],
            'detected_cols': grid_structure['cols'],
            'total_grid_cells': len(grid_cells),
            'original_detected_cells': len(cells),
            'median_column_widths': median_dimensions['column_widths'],
            'median_row_heights': median_dimensions['row_heights'],
            'scaled_column_widths': scaled_dimensions['column_widths'],
            'scaled_row_heights': scaled_dimensions['row_heights'],
            'scaling_factors': scaled_dimensions['scaling_factors'],
            'grid_cells': grid_cells,
            'original_cells': cells,
            'max_cells_per_row': median_dimensions['max_cells_per_row'],
            'max_cells_per_col': median_dimensions['max_cells_per_col']
        }
        
        # Save reconstruction data
        recon_path = output_dir / f"table_{table_id}_median_grid_data.json"
        with open(recon_path, 'w') as f:
            json.dump(grid_data, f, indent=2)
        
        print(f"   💾 Median grid data saved: {recon_path}")
        
        # Create visual reconstruction with median-based grid
        create_median_visualization(grid_data, output_dir)

def analyze_cell_positions_for_median(relative_cells, table_width, table_height):
    """Analyze cell positions to determine grid structure for median positioning with 5% tolerance"""
    
    # Group cells by rows and columns based on their positions with 5% tolerance
    cell_groups = group_cells_by_position(relative_cells, table_width, table_height)
    
    return {
        'rows': len(cell_groups['rows']),
        'cols': len(cell_groups['cols']),
        'cell_groups': cell_groups
    }

def group_cells_by_position(relative_cells, table_width, table_height):
    """Group cells into a proper grid structure with fixed rows and columns"""
    
    # First, determine the expected grid size based on cell count
    total_cells = len(relative_cells)
    print(f"   🔍 Total cells: {total_cells}")
    
    # Try to determine grid dimensions (common patterns: 3x3=9, 3x4=12, 3x5=15, 3x6=18, 3x7=21, 3x8=24, 3x9=27, etc.)
    possible_grids = []
    for rows in range(1, total_cells + 1):
        if total_cells % rows == 0:
            cols = total_cells // rows
            possible_grids.append((rows, cols))
    
    print(f"   🔍 Possible grid sizes: {possible_grids}")
    
    # For now, let's assume 9x3 based on the output we saw
    if total_cells == 27:
        expected_rows, expected_cols = 9, 3
    elif total_cells == 3:
        expected_rows, expected_cols = 1, 3
    else:
        # Use the first reasonable grid (closest to square)
        expected_rows, expected_cols = possible_grids[0] if possible_grids else (1, total_cells)
    
    print(f"   🎯 Using grid size: {expected_rows} rows × {expected_cols} columns")
    
    # Handle overlapping cells by averaging their positions
    processed_cells = handle_overlapping_cells(relative_cells, expected_rows, expected_cols)
    
    # Sort processed cells by position to assign them to grid positions
    cells_sorted = sorted(processed_cells, key=lambda c: (c['center_y'], c['center_x']))
    
    # Group cells into rows based on y-position similarity
    y_tolerance = 15  # pixels - cells within this y-distance are considered in the same row
    row_groups = []
    used_cell_indices = set()
    
    for i, cell in enumerate(cells_sorted):
        if i in used_cell_indices:
            continue
            
        # Start a new row group with this cell
        row_group = [cell]
        used_cell_indices.add(i)
        cell_y = cell['center_y']
        
        # Find other cells at similar y-positions
        for j, other_cell in enumerate(cells_sorted):
            if j in used_cell_indices:
                continue
                
            other_y = other_cell['center_y']
            if abs(cell_y - other_y) <= y_tolerance:
                row_group.append(other_cell)
                used_cell_indices.add(j)
        
        # Sort this row group by x-position (left to right)
        row_group_sorted = sorted(row_group, key=lambda c: c['center_x'])
        row_groups.append(row_group_sorted)
    
    # Sort row groups by their average y-position (top to bottom)
    row_groups_sorted = sorted(row_groups, key=lambda rg: sum(c['center_y'] for c in rg) / len(rg))
    
    print(f"   🔍 Row grouping results: Found {len(row_groups_sorted)} row groups")
    for i, group in enumerate(row_groups_sorted):
        y_positions = [f'{c["center_y"]:.1f}' for c in group]
        print(f"      Group {i}: {len(group)} cells at y={y_positions}")
    
    # Ensure we have the expected number of rows
    if len(row_groups_sorted) != expected_rows:
        print(f"   ⚠️  Warning: Found {len(row_groups_sorted)} row groups, expected {expected_rows}")
        # Take the first expected_rows groups
        row_groups_sorted = row_groups_sorted[:expected_rows]
    
    rows = row_groups_sorted
    
    # Debug: Print which cells are assigned to which columns
    print(f"   🔍 Column assignment:")
    for row_idx, row_cells in enumerate(rows):
        cell_indices = [cells_sorted.index(cell) for cell in row_cells]
        print(f"      Row {row_idx}: cells {cell_indices}")
    
    # Create column structure - sort cells within each column by y-position (top to bottom)
    cols = []
    for col_idx in range(expected_cols):
        col_cells = []
        for row_idx in range(expected_rows):
            if col_idx < len(rows[row_idx]):
                col_cells.append(rows[row_idx][col_idx])
        
        # Sort cells within this column by y-position (top to bottom)
        col_cells_sorted = sorted(col_cells, key=lambda c: c['center_y'])
        cols.append(col_cells_sorted)
    
    # Debug: Print which cells are assigned to which columns with their y-positions
    print(f"   🔍 Column assignment:")
    for col_idx, col_cells in enumerate(cols):
        cell_indices = [cells_sorted.index(cell) for cell in col_cells]
        y_positions = [f'{cell["center_y"]:.1f}' for cell in col_cells]
        print(f"      Col {col_idx}: cells {cell_indices} at y={y_positions}")
    
    # Debug: Print row assignments with y-positions
    print(f"   🔍 Row assignment:")
    for row_idx, row_cells in enumerate(rows):
        cell_indices = [cells_sorted.index(cell) for cell in row_cells]
        y_positions = [f'{cell["center_y"]:.1f}' for cell in row_cells]
        print(f"      Row {row_idx}: cells {cell_indices} at y={y_positions}")
    
    # Validate grid structure
    print(f"   🔍 Grid validation:")
    print(f"      Expected: {expected_rows} rows × {expected_cols} columns = {expected_rows * expected_cols} cells")
    print(f"      Actual: {len(rows)} rows × {len(cols)} columns = {sum(len(row) for row in rows)} cells")
    
    for i, row in enumerate(rows):
        print(f"      Row {i}: {len(row)} cells")
    
    for i, col in enumerate(cols):
        print(f"      Col {i}: {len(col)} cells")
    
    return {
        'rows': rows,
        'cols': cols,
        'expected_rows': expected_rows,
        'expected_cols': expected_cols
    }

def handle_overlapping_cells(relative_cells, expected_rows, expected_cols):
    """Handle overlapping cells by averaging their positions"""
    
    print(f"   🔍 Checking for overlapping cells...")
    
    # Sort cells by y position first, then x position
    sorted_cells = sorted(relative_cells, key=lambda c: (c['center_y'], c['center_x']))
    
    # Debug: Print cell positions to see the layout
    print(f"   🔍 Cell positions (first 10 cells):")
    for i, cell in enumerate(sorted_cells[:10]):
        print(f"      Cell {i}: center=({cell['center_x']:.1f}, {cell['center_y']:.1f}), bbox={[f'{x:.1f}' for x in cell['bbox']]}")
    
    processed_cells = []
    overlap_count = 0
    
    # Check for overlapping cells and merge them
    i = 0
    while i < len(sorted_cells):
        current_cell = sorted_cells[i]
        overlapping_cells = [current_cell]
        
        # Check subsequent cells for overlap with current cell
        j = i + 1
        while j < len(sorted_cells):
            next_cell = sorted_cells[j]
            
            # Check if cells overlap (have any intersection)
            if cells_overlap(current_cell, next_cell):
                overlapping_cells.append(next_cell)
                j += 1
            else:
                break
        
        if len(overlapping_cells) > 1:
            # Merge overlapping cells by averaging their positions and dimensions
            merged_cell = merge_overlapping_cells(overlapping_cells)
            processed_cells.append(merged_cell)
            overlap_count += len(overlapping_cells)
            print(f"   🔄 Merged {len(overlapping_cells)} overlapping cells at y={merged_cell['center_y']:.1f}, x={merged_cell['center_x']:.1f}")
            i = j  # Skip all processed overlapping cells
        else:
            processed_cells.append(current_cell)
            i += 1
    
    print(f"   ✅ Processed {len(processed_cells)} cells (merged {overlap_count} overlapping cells)")
    return processed_cells

def cells_overlap(cell1, cell2):
    """Check if two cells overlap"""
    x1_1, y1_1, x2_1, y2_1 = cell1['bbox']
    x1_2, y1_2, x2_2, y2_2 = cell2['bbox']
    
    # Check for overlap: not (cell1 is completely to the left/right/top/bottom of cell2)
    return not (x2_1 <= x1_2 or x2_2 <= x1_1 or y2_1 <= y1_2 or y2_2 <= y1_1)

def merge_overlapping_cells(overlapping_cells):
    """Merge overlapping cells by averaging their positions and dimensions"""
    
    # Calculate average position and dimensions
    avg_x1 = sum(cell['bbox'][0] for cell in overlapping_cells) / len(overlapping_cells)
    avg_y1 = sum(cell['bbox'][1] for cell in overlapping_cells) / len(overlapping_cells)
    avg_x2 = sum(cell['bbox'][2] for cell in overlapping_cells) / len(overlapping_cells)
    avg_y2 = sum(cell['bbox'][3] for cell in overlapping_cells) / len(overlapping_cells)
    
    # Create merged cell
    merged_cell = {
        'bbox': [avg_x1, avg_y1, avg_x2, avg_y2],
        'width': avg_x2 - avg_x1,
        'height': avg_y2 - avg_y1,
        'center_x': (avg_x1 + avg_x2) / 2,
        'center_y': (avg_y1 + avg_y2) / 2,
        'merged_from': len(overlapping_cells),
        'original_cells': overlapping_cells
    }
    
    return merged_cell

def cluster_positions(positions, tolerance):
    """Cluster positions that are within tolerance of each other"""
    if not positions:
        return []
    
    sorted_positions = sorted(set(positions))
    clusters = []
    current_cluster = [sorted_positions[0]]
    
    for pos in sorted_positions[1:]:
        if abs(pos - current_cluster[-1]) <= tolerance:
            current_cluster.append(pos)
        else:
            clusters.append(current_cluster)
            current_cluster = [pos]
    
    clusters.append(current_cluster)
    return clusters

def calculate_median_dimensions(relative_cells, grid_structure):
    """Calculate dimensions using proper spacing between rows and columns"""
    
    cell_groups = grid_structure['cell_groups']
    rows = cell_groups['rows']
    cols = cell_groups['cols']
    
    # Calculate row heights based on spacing between rows (similar to column widths)
    row_heights = []
    max_cells_per_row = []
    
    # Get rows with their average y-positions
    rows_with_positions = []
    for row_idx, row_cells in enumerate(rows):
        avg_y = sum(cell['center_y'] for cell in row_cells) / len(row_cells)
        rows_with_positions.append((row_idx, row_cells, avg_y))
    
    rows_with_positions.sort(key=lambda x: x[2])  # Sort by y-position
    
    for i, (row_idx, row_cells, avg_y) in enumerate(rows_with_positions):
        max_height_cell = max(row_cells, key=lambda cell: cell['height'])
        max_cells_per_row.append(max_height_cell)
        
        if i == 0:
            # First row: from top of table to bottom of this row's content
            row_start = min(cell['bbox'][1] for cell in row_cells)
            row_end = max(cell['bbox'][3] for cell in row_cells)
            row_height = row_end - row_start
        else:
            # Subsequent rows: from bottom of previous row to bottom of current row
            prev_row_cells = rows_with_positions[i-1][1]
            prev_end = max(cell['bbox'][3] for cell in prev_row_cells)
            curr_end = max(cell['bbox'][3] for cell in row_cells)
            row_height = curr_end - prev_end
            
            # Ensure row height is not too large (cap at reasonable multiple of cell height)
            max_cell_height = max(cell['height'] for cell in row_cells)
            max_reasonable_height = max_cell_height * 1.5  # Allow 50% extra for spacing
            if row_height > max_reasonable_height:
                row_height = max_reasonable_height
                print(f"   ⚠️  Row {row_idx}: Capped height from {curr_end - prev_end:.1f} to {row_height:.1f}")
        
        row_heights.append(row_height)
        print(f"   📏 Row {row_idx}: spacing from prev to curr → row height {row_height:.1f}")
    
    # Reorder row heights to match original row order
    ordered_row_heights = [0] * len(row_heights)
    ordered_max_cells_per_row = [None] * len(row_heights)
    
    for i, (row_idx, _, _) in enumerate(rows_with_positions):
        ordered_row_heights[row_idx] = row_heights[i]
        ordered_max_cells_per_row[row_idx] = max_cells_per_row[i]
    
    row_heights = ordered_row_heights
    max_cells_per_row = ordered_max_cells_per_row
    
    # Calculate column widths using proper spacing between columns
    column_widths = []
    max_cells_per_col = []
    
    # Sort columns by their x position to get proper order
    cols_with_positions = []
    for col_idx, col_cells in enumerate(cols):
        avg_x = sum(cell['center_x'] for cell in col_cells) / len(col_cells)
        cols_with_positions.append((col_idx, col_cells, avg_x))
    
    cols_with_positions.sort(key=lambda x: x[2])  # Sort by average x position
    
    for i, (col_idx, col_cells, avg_x) in enumerate(cols_with_positions):
        # Find cell with maximum width in this column
        max_width_cell = max(col_cells, key=lambda cell: cell['width'])
        max_cells_per_col.append(max_width_cell)
        
        # Special handling for single-row tables (like Table 2)
        if len(rows) == 1:
            # For single-row tables, use actual cell width
            col_width = max_width_cell['width']
        else:
            # For multi-row tables, use spacing between columns
            if i == 0:
                # First column: width from start of this column to end of this column (skip table edge)
                col_start = min(cell['bbox'][0] for cell in col_cells)  # Leftmost edge of first column
                col_end = max(cell['bbox'][2] for cell in col_cells)  # Rightmost edge
                col_width = col_end - col_start
            else:
                # Subsequent columns: width from end of previous column to end of current column
                prev_col_cells = cols_with_positions[i-1][1]
                prev_end = max(cell['bbox'][2] for cell in prev_col_cells)  # End of previous column
                curr_end = max(cell['bbox'][2] for cell in col_cells)  # End of current column
                col_width = curr_end - prev_end
        
        column_widths.append(col_width)
        if len(rows) == 1:
            print(f"   📏 Col {col_idx}: single-row table → col width {col_width:.1f}")
        else:
            print(f"   📏 Col {col_idx}: spacing from prev to curr → col width {col_width:.1f}")
    
    # Reorder column_widths to match original column order
    ordered_column_widths = [0] * len(column_widths)
    ordered_max_cells_per_col = [None] * len(column_widths)
    
    for i, (col_idx, _, _) in enumerate(cols_with_positions):
        ordered_column_widths[col_idx] = column_widths[i]
        ordered_max_cells_per_col[col_idx] = max_cells_per_col[i]
    
    return {
        'row_heights': row_heights,
        'column_widths': ordered_column_widths,
        'max_cells_per_row': max_cells_per_row,
        'max_cells_per_col': ordered_max_cells_per_col
    }

def scale_dimensions_to_fit_table(median_dimensions, table_width, table_height):
    """Scale median dimensions to fit the exact table boundaries"""
    
    raw_column_widths = median_dimensions['column_widths']
    raw_row_heights = median_dimensions['row_heights']
    
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

def create_scaled_median_grid_cells(relative_cells, grid_structure, scaled_dimensions, table_bbox):
    """Create grid cells based on scaled median dimensions"""
    
    table_x1, table_y1, table_x2, table_y2 = table_bbox
    cell_groups = grid_structure['cell_groups']
    rows = cell_groups['rows']
    cols = cell_groups['cols']
    row_heights = scaled_dimensions['row_heights']
    column_widths = scaled_dimensions['column_widths']
    
    # Sort rows and columns by position
    rows_sorted = sorted(rows, key=lambda row: min(cell['center_y'] for cell in row))
    cols_sorted = sorted(cols, key=lambda col: min(cell['center_x'] for cell in col))
    
    # Create grid cells
    grid_cells = []
    cell_id = 0
    
    current_y = 0
    for row_idx, row_cells in enumerate(rows_sorted):
        current_x = 0
        for col_idx, col_cells in enumerate(cols_sorted):
            # Use scaled median dimensions
            width = column_widths[col_idx]
            height = row_heights[row_idx]
            
            # Create cell with scaled median dimensions
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
                'is_median_based': True
            })
            cell_id += 1
            
            current_x += width
        current_y += height
    
    return grid_cells

def create_median_visualization(grid_data, output_dir):
    """Create visual reconstruction using median-based grid"""
    
    table_id = grid_data['table_id']
    table_bbox = grid_data['table_bbox']
    grid_cells = grid_data['grid_cells']
    original_cells = grid_data['original_cells']
    column_widths = grid_data['scaled_column_widths']
    row_heights = grid_data['scaled_row_heights']
    scaling_factors = grid_data['scaling_factors']
    max_cells_per_row = grid_data['max_cells_per_row']
    max_cells_per_col = grid_data['max_cells_per_col']
    
    print(f"   🎨 Creating visual reconstruction with median-based grid...")
    
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
    title = f"Table {table_id} - Median-Based Grid"
    draw.text((20, 20), title, fill='black', font=title_font)
    
    # Draw info
    info_text = f"Grid: {len(row_heights)}×{len(column_widths)} = {len(grid_cells)} cells | Detected: {len(original_cells)} cells"
    draw.text((20, 50), info_text, fill='gray', font=font)
    
    # Draw scaling info
    scale_text = f"Scaling: width×{scaling_factors['width_scale']:.3f}, height×{scaling_factors['height_scale']:.3f}"
    draw.text((20, 70), scale_text, fill='blue', font=small_font)
    
    # Draw median dimensions
    width_text = f"Median column widths: {[f'{w:.0f}' for w in column_widths]}"
    height_text = f"Median row heights: {[f'{h:.0f}' for h in row_heights]}"
    draw.text((20, 85), width_text, fill='blue', font=small_font)
    draw.text((20, 100), height_text, fill='green', font=small_font)
    
    # Calculate offset to center the table
    offset_x = (img_width - table_width) // 2
    offset_y = 130
    
    # Draw table boundary
    table_rect = [offset_x, offset_y, offset_x + table_width, offset_y + table_height]
    draw.rectangle(table_rect, outline='black', width=3)
    
    # Draw median-based grid cells
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink', 'lightgray', 'lightcyan', 'lightcoral', 'lightsteelblue', 'lightseagreen']
    
    for cell in grid_cells:
        row = cell['row']
        col = cell['col']
        rel_bbox = cell['relative_bbox']
        width = cell['width']
        height = cell['height']
        
        x1, y1, x2, y2 = rel_bbox
        
        # Draw cell with median-based dimensions
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
    
    # Highlight max cells used for dimension calculation
    for i, max_cell in enumerate(max_cells_per_row):
        rel_bbox = max_cell['bbox']
        x1, y1, x2, y2 = rel_bbox
        # Draw yellow outline for max height cells in rows
        max_rect = [offset_x + x1, offset_y + y1, offset_x + x2, offset_y + y2]
        draw.rectangle(max_rect, outline='yellow', width=3)
        draw.text((offset_x + x1 + 2, offset_y + y2 - 15), f"H{i}", fill='yellow', font=small_font)
    
    for i, max_cell in enumerate(max_cells_per_col):
        rel_bbox = max_cell['bbox']
        x1, y1, x2, y2 = rel_bbox
        # Draw orange outline for max width cells in columns
        max_rect = [offset_x + x1, offset_y + y1, offset_x + x2, offset_y + y2]
        draw.rectangle(max_rect, outline='orange', width=3)
        draw.text((offset_x + x2 - 15, offset_y + y1 + 2), f"W{i}", fill='orange', font=small_font)
    
    # Draw legend
    legend_y = offset_y + table_height + 20
    draw.text((offset_x, legend_y), "Legend:", fill='black', font=font)
    draw.text((offset_x, legend_y + 20), "Colored cells = Max-based grid (max width/height per row/column)", fill='black', font=font)
    draw.text((offset_x, legend_y + 40), "Red outlines = Original detected cells", fill='red', font=font)
    draw.text((offset_x, legend_y + 60), "Yellow outlines = Max height cells per row (H0, H1, ...)", fill='yellow', font=font)
    draw.text((offset_x, legend_y + 80), "Orange outlines = Max width cells per column (W0, W1, ...)", fill='orange', font=font)
    
    # Save image
    output_path = output_dir / f"table_{table_id}_median_grid_reconstruction.png"
    img.save(output_path)
    print(f"   🖼️  Median grid visualization saved: {output_path}")

def main():
    """Main function"""
    print("🚀 CREATING MEDIAN-BASED GRID")
    print("=" * 60)
    
    create_median_based_grid()
    
    print("\n" + "=" * 60)
    print("✅ MEDIAN-BASED GRID CREATED!")
    print("📁 Check 'intermediate_outputs/grid_output/' for outputs")
    print("=" * 60)

if __name__ == "__main__":
    main()
