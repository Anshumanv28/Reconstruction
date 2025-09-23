#!/usr/bin/env python3
"""
Simple TableFormer reconstruction that matches coordinates and estimates missing data.
No grid structure changes, just coordinate matching and gap filling.
"""

import json
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

def simple_reconstruction():
    """Simple reconstruction that matches coordinates and estimates missing data"""
    
    print("=" * 60)
    print("SIMPLE TABLEFORMER RECONSTRUCTION")
    print("=" * 60)
    
    # Load the TableFormer grid data
    json_path = "intermediate_outputs/former_grid_outputs/1zM9MmA6dM2_2dbMHiJKd_m-FRdPpNTR3lJUT_P1QuiE_tableformer_grid_data.json"
    original_image_path = "pipe_input/1zM9MmA6dM2_2dbMHiJKd_m-FRdPpNTR3lJUT_P1QuiE.png"
    
    if not Path(json_path).exists():
        print(f"❌ Grid data not found: {json_path}")
        return
    
    if not Path(original_image_path).exists():
        print(f"❌ Original image not found: {original_image_path}")
        return
    
    # Load data
    with open(json_path, 'r') as f:
        grid_data = json.load(f)
    
    # Load original image
    original_img = Image.open(original_image_path).convert("RGB")
    draw = ImageDraw.Draw(original_img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 12)
        small_font = ImageFont.truetype("arial.ttf", 8)
    except IOError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Process each table
    for table_info in grid_data['tableformer_grid_analysis']:
        table_id = table_info['table_id']
        table_bbox = table_info['table_bbox']
        grid_cells = table_info['grid_cells']
        detected_cells = table_info['detected_cells']
        
        print(f"\n🔧 TABLE {table_id}:")
        print(f"   📊 Grid cells: {len(grid_cells)}")
        print(f"   📊 Detected cells: {len(detected_cells)}")
        
        x1, y1, x2, y2 = table_bbox
        
        # Draw table boundary
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        
        # Draw grid structure cells (from estimated_grids)
        print(f"   🎯 Drawing {len(grid_cells)} grid structure cells:")
        for i, cell in enumerate(grid_cells):
            # Use the absolute_bbox from processed grid_cells
            bbox = cell['absolute_bbox']
            # Draw grid cell
            draw.rectangle(bbox, outline='blue', width=2)
            # Add label
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            draw.text((center_x-10, center_y-5), f"G{i}", fill='blue', font=small_font)
            print(f"      Grid Cell {i}: {[f'{val:.1f}' for val in bbox]}")
        
        # Draw detected cells (from individual detections)
        print(f"   🎯 Drawing {len(detected_cells)} detected cells:")
        for i, cell in enumerate(detected_cells):
            bbox = cell['absolute_bbox']
            # Draw detected cell
            draw.rectangle(bbox, outline='green', width=1)
            # Add label
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            draw.text((center_x-10, center_y-5), f"D{i}", fill='green', font=small_font)
            print(f"      Detected Cell {i}: {[f'{val:.1f}' for val in bbox]}")
        
        # Add table title
        title_text = f"Table {table_id} - Simple Reconstruction"
        draw.text((x1, y1-25), title_text, fill='red', font=font)
    
    # Add legend
    legend_y = 20
    draw.text((20, legend_y), "Simple TableFormer Reconstruction", fill='black', font=font)
    draw.text((20, legend_y + 20), "Red = Table boundaries", fill='red', font=small_font)
    draw.text((20, legend_y + 35), "Blue = Grid structure cells", fill='blue', font=small_font)
    draw.text((20, legend_y + 50), "Green = Detected individual cells", fill='green', font=small_font)
    
    # Save the reconstruction
    output_path = "intermediate_outputs/former_grid_outputs/simple_tableformer_reconstruction.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    original_img.save(output_path)
    
    print(f"\n✅ Simple reconstruction saved: {output_path}")
    print("\n" + "=" * 60)
    print("✅ RECONSTRUCTION COMPLETE!")
    print("📁 Check 'intermediate_outputs/former_grid_outputs/' for the result")
    print("=" * 60)

if __name__ == "__main__":
    simple_reconstruction()
