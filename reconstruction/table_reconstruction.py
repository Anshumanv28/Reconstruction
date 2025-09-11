#!/usr/bin/env python3
"""
Table Reconstruction Module

This module handles the reconstruction of tables using TableFormer output.
It focuses purely on drawing table structures without OCR text integration.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set
from PIL import Image, ImageDraw, ImageFont
import numpy as np


class TableReconstructor:
    """Handles table reconstruction using TableFormer output."""
    
    def __init__(self, 
                 background_color: Tuple[int, int, int] = (255, 255, 255),
                 text_color: Tuple[int, int, int] = (0, 0, 0),
                 grid_size: int = 20):
        """Initialize the table reconstructor."""
        self.background_color = background_color
        self.text_color = text_color
        self.grid_size = grid_size
        
    def load_tableformer_data(self, tableformer_file: str) -> List[Dict]:
        """Load TableFormer results from JSON file."""
        with open(tableformer_file, 'r') as f:
            data = json.load(f)
        
        # Handle the actual TableFormer data structure
        if isinstance(data, list) and len(data) > 0:
            # Data is a list of table results
            tables = []
            for i, table_data in enumerate(data):
                if 'predict_details' in table_data:
                    # Extract table information from predict_details
                    predict_details = table_data['predict_details']
                    table_bbox = predict_details.get('table_bbox', [0, 0, 100, 100])
                    prediction_bboxes = predict_details.get('prediction_bboxes_page', [])
                    
                    # Create table structure
                    table = {
                        'bbox': table_bbox,
                        'cells': []
                    }
                    
                    # Convert prediction bboxes to cells
                    for j, cell_bbox in enumerate(prediction_bboxes):
                        cell = {
                            'bbox': cell_bbox,
                            'row': j // 3,  # Approximate row (assuming 3 columns)
                            'col': j % 3,   # Approximate column
                            'cell_id': j
                        }
                        table['cells'].append(cell)
                    
                    tables.append(table)
                    print(f"  Table {i+1}: {len(table['cells'])} cells from {len(prediction_bboxes)} prediction bboxes")
            
            return tables
        
        # Fallback to original logic
        if 'tables' in data:
            return data['tables']
        elif 'tableformer_results' in data:
            return data['tableformer_results']
        else:
            return data  # Assume data is already the table list
    
    def calculate_table_dimensions(self, tables: List[Dict]) -> Tuple[int, int]:
        """Calculate overall document dimensions based on table bounding boxes."""
        if not tables:
            return 1000, 1000  # Default size
        
        max_width = 0
        max_height = 0
        
        for table in tables:
            if 'bbox' in table:
                x1, y1, x2, y2 = table['bbox']
                max_width = max(max_width, int(x2))
                max_height = max(max_height, int(y2))
            elif 'cells' in table:
                for cell in table['cells']:
                    if 'bbox' in cell:
                        x1, y1, x2, y2 = cell['bbox']
                        max_width = max(max_width, int(x2))
                        max_height = max(max_height, int(y2))
        
        # Add padding
        return max_width + 100, max_height + 100
    
    def reconstruct_tables(self, tables: List[Dict]) -> Image.Image:
        """Reconstruct tables by drawing their structure."""
        if not tables:
            print("No tables found in TableFormer data")
            return None
        
        # Calculate document dimensions
        width, height = self.calculate_table_dimensions(tables)
        print(f"Table document dimensions: {width} x {height}")
        
        # Create canvas
        canvas = Image.new('RGB', (width, height), self.background_color)
        draw = ImageDraw.Draw(canvas)
        
        # Draw grid
        self._draw_grid(draw, width, height)
        
        # Process each table
        for i, table in enumerate(tables):
            print(f"Processing table {i+1}/{len(tables)}")
            self._draw_table_structure(draw, table, i)
        
        return canvas
    
    def _draw_grid(self, draw: ImageDraw.Draw, width: int, height: int) -> None:
        """Draw a grid overlay to help visualize table structure."""
        grid_size = 50  # Grid spacing in pixels
        
        # Draw vertical lines
        for x in range(0, width, grid_size):
            draw.line([(x, 0), (x, height)], fill=(220, 220, 220), width=1)
        
        # Draw horizontal lines
        for y in range(0, height, grid_size):
            draw.line([(0, y), (width, y)], fill=(220, 220, 220), width=1)
        
        print(f"  Drew grid with {grid_size}px spacing")
    
    def _draw_table_structure(self, draw: ImageDraw.Draw, table: Dict, table_index: int) -> None:
        """Draw a single table structure."""
        if 'cells' not in table:
            print(f"  Table {table_index + 1}: No cells found")
            return
        
        cells = table['cells']
        print(f"  Table {table_index + 1}: Drawing {len(cells)} cells")
        
        # Draw table border
        if 'bbox' in table:
            x1, y1, x2, y2 = table['bbox']
            draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 0), width=2)
            print(f"    Table border: ({x1}, {y1}) to ({x2}, {y2})")
        
        # Draw each cell
        for cell_index, cell in enumerate(cells):
            self._draw_cell_structure(draw, cell, cell_index)
    
    def _draw_cell_structure(self, draw: ImageDraw.Draw, cell: Dict, cell_index: int) -> None:
        """Draw a single cell structure."""
        if 'bbox' not in cell:
            return
        
        x1, y1, x2, y2 = cell['bbox']
        # Convert to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw cell border
        draw.rectangle([x1, y1, x2, y2], outline=(100, 100, 100), width=1)
        
        # Add cell information as placeholder text
        cell_info = f"Cell {cell_index + 1}"
        if 'row' in cell and 'col' in cell:
            cell_info = f"R{cell['row']}C{cell['col']}"
        
        # Try to get font
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            try:
                font = ImageFont.truetype("calibri.ttf", 12)
            except:
                font = ImageFont.load_default()
        
        # Calculate text position (center of cell) - using textbbox for newer PIL versions
        try:
            # Newer PIL versions
            bbox = draw.textbbox((0, 0), cell_info, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            # Older PIL versions
            text_width, text_height = draw.textsize(cell_info, font=font)
        
        text_x = x1 + (x2 - x1 - text_width) // 2
        text_y = y1 + (y2 - y1 - text_height) // 2
        
        # Draw placeholder text
        draw.text((text_x, text_y), cell_info, fill=self.text_color, font=font)
        
        print(f"    Cell {cell_index + 1}: ({x1}, {y1}) to ({x2}, {y2}) - {cell_info}")
    
    def save_table_reconstruction(self, canvas: Image.Image, output_path: str) -> None:
        """Save the table reconstruction."""
        if canvas:
            canvas.save(output_path)
            print(f"Table reconstruction saved to: {output_path}")
        else:
            print("No canvas to save")


def main():
    """Main function for table reconstruction."""
    parser = argparse.ArgumentParser(description="Reconstruct tables using TableFormer output")
    parser.add_argument("--tableformer-file", required=True, help="Path to TableFormer results JSON")
    parser.add_argument("--output", "-o", required=True, help="Output PDF path")
    parser.add_argument("--image-name", help="Image name for debugging")
    
    args = parser.parse_args()
    
    print(f"=== Table Reconstruction ===")
    print(f"TableFormer file: {args.tableformer_file}")
    print(f"Output: {args.output}")
    
    # Initialize reconstructor
    reconstructor = TableReconstructor()
    
    # Load TableFormer data
    tables = reconstructor.load_tableformer_data(args.tableformer_file)
    print(f"Loaded {len(tables)} tables")
    
    # Reconstruct tables
    canvas = reconstructor.reconstruct_tables(tables)
    
    # Save result
    if canvas:
        reconstructor.save_table_reconstruction(canvas, args.output)
        print("Table reconstruction completed successfully!")
    else:
        print("Table reconstruction failed - no tables found")


if __name__ == "__main__":
    main()
