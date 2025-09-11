#!/usr/bin/env python3
"""
OCR Integration Module

This module handles the integration of OCR text with reconstructed layout and table structures.
It takes the structural elements and populates them with actual OCR text.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set
from PIL import Image, ImageDraw, ImageFont
import numpy as np


class OCRIntegrator:
    """Handles OCR text integration with reconstructed elements."""
    
    def __init__(self, 
                 background_color: Tuple[int, int, int] = (255, 255, 255),
                 text_color: Tuple[int, int, int] = (0, 0, 0),
                 grid_size: int = 20):
        """Initialize the OCR integrator."""
        self.background_color = background_color
        self.text_color = text_color
        self.grid_size = grid_size
        
    def load_ocr_data(self, ocr_file: str) -> List[Dict]:
        """Load OCR results from JSON file."""
        with open(ocr_file, 'r') as f:
            data = json.load(f)
        
        # Extract OCR text blocks - adjust path based on actual structure
        if 'text_blocks' in data:
            return data['text_blocks']
        elif 'full_document_ocr' in data and 'text_blocks' in data['full_document_ocr']:
            return data['full_document_ocr']['text_blocks']
        else:
            return data  # Assume data is already the text blocks list
    
    def load_layout_data(self, layout_file: str) -> List[Dict]:
        """Load layout predictions from JSON file."""
        with open(layout_file, 'r') as f:
            data = json.load(f)
        
        # Extract layout data - adjust path based on actual structure
        if 'layout_elements' in data:
            return data['layout_elements']
        elif 'predictions' in data:
            return data['predictions']
        else:
            return data  # Assume data is already the layout list
    
    def load_tableformer_data(self, tableformer_file: str) -> List[Dict]:
        """Load TableFormer results from JSON file."""
        with open(tableformer_file, 'r') as f:
            data = json.load(f)
        
        # Extract table data - adjust path based on actual structure
        if 'tables' in data:
            return data['tables']
        elif 'tableformer_results' in data:
            return data['tableformer_results']
        else:
            return data  # Assume data is already the table list
    
    def calculate_document_dimensions(self, layout_elements: List[Dict], tables: List[Dict], ocr_blocks: List[Dict]) -> Tuple[int, int]:
        """Calculate overall document dimensions based on all data sources."""
        max_width = 0
        max_height = 0
        
        # Check layout elements
        for element in layout_elements:
            if 'bbox' in element:
                x1, y1, x2, y2 = element['bbox']
                max_width = max(max_width, int(x2))
                max_height = max(max_height, int(y2))
            elif all(key in element for key in ['l', 't', 'r', 'b']):
                l, t, r, b = element['l'], element['t'], element['r'], element['b']
                max_width = max(max_width, int(r))
                max_height = max(max_height, int(b))
        
        # Check tables
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
        
        # Check OCR blocks
        for block in ocr_blocks:
            if 'bbox' in block:
                x1, y1, x2, y2 = block['bbox']
                max_width = max(max_width, int(x2))
                max_height = max(max_height, int(y2))
        
        # Add padding
        return max_width + 100, max_height + 100
    
    def integrate_ocr_with_structures(self, layout_elements: List[Dict], tables: List[Dict], ocr_blocks: List[Dict]) -> Image.Image:
        """Integrate OCR text with layout and table structures."""
        if not ocr_blocks:
            print("No OCR blocks found")
            return None
        
        # Calculate document dimensions
        width, height = self.calculate_document_dimensions(layout_elements, tables, ocr_blocks)
        print(f"Integrated document dimensions: {width} x {height}")
        
        # Create canvas
        canvas = Image.new('RGB', (width, height), self.background_color)
        draw = ImageDraw.Draw(canvas)
        
        # Draw grid
        self._draw_grid(draw, width, height)
        
        # Sort OCR blocks by position (top to bottom, left to right)
        sorted_ocr_blocks = sorted(ocr_blocks, key=lambda x: (x['bbox'][1], x['bbox'][0]))
        
        print(f"Processing {len(sorted_ocr_blocks)} OCR blocks")
        
        # Process each OCR block
        for i, ocr_block in enumerate(sorted_ocr_blocks):
            self._process_ocr_block(draw, ocr_block, layout_elements, tables, i)
        
        return canvas
    
    def _draw_grid(self, draw: ImageDraw.Draw, width: int, height: int) -> None:
        """Draw a grid overlay to help visualize OCR text positioning."""
        grid_size = 50  # Grid spacing in pixels
        
        # Draw vertical lines
        for x in range(0, width, grid_size):
            draw.line([(x, 0), (x, height)], fill=(220, 220, 220), width=1)
        
        # Draw horizontal lines
        for y in range(0, height, grid_size):
            draw.line([(0, y), (width, y)], fill=(220, 220, 220), width=1)
        
        print(f"  Drew grid with {grid_size}px spacing")
    
    def _process_ocr_block(self, draw: ImageDraw.Draw, ocr_block: Dict, layout_elements: List[Dict], tables: List[Dict], block_index: int) -> None:
        """Process a single OCR block and integrate it with structures."""
        x1, y1, x2, y2 = ocr_block['bbox']
        text = ocr_block.get('text', '').strip()
        
        if not text:
            return
        
        print(f"  OCR Block {block_index + 1}: '{text}' at ({x1}, {y1}) to ({x2}, {y2})")
        
        # Check if this OCR block overlaps with any layout element
        layout_element = self._find_overlapping_layout_element(x1, y1, x2, y2, layout_elements)
        
        # Check if this OCR block overlaps with any table
        table_element = self._find_overlapping_table(x1, y1, x2, y2, tables)
        
        # Render based on what it overlaps with
        if table_element:
            self._render_text_in_table(draw, x1, y1, x2, y2, text, table_element)
        elif layout_element:
            self._render_text_in_layout(draw, x1, y1, x2, y2, text, layout_element)
        else:
            self._render_plain_text(draw, x1, y1, x2, y2, text)
    
    def _find_overlapping_layout_element(self, x1: int, y1: int, x2: int, y2: int, layout_elements: List[Dict]) -> Dict:
        """Find layout element that overlaps with the given bounding box."""
        for element in layout_elements:
            if 'bbox' in element:
                ex1, ey1, ex2, ey2 = element['bbox']
            elif all(key in element for key in ['l', 't', 'r', 'b']):
                ex1, ey1, ex2, ey2 = element['l'], element['t'], element['r'], element['b']
            else:
                continue
            
            # Check for overlap
            if not (x2 < ex1 or x1 > ex2 or y2 < ey1 or y1 > ey2):
                return element
        
        return None
    
    def _find_overlapping_table(self, x1: int, y1: int, x2: int, y2: int, tables: List[Dict]) -> Dict:
        """Find table that overlaps with the given bounding box."""
        for table in tables:
            if 'bbox' in table:
                tx1, ty1, tx2, ty2 = table['bbox']
            elif 'cells' in table and table['cells']:
                # Calculate table bounds from cells
                cell_bboxes = [cell['bbox'] for cell in table['cells'] if 'bbox' in cell]
                if not cell_bboxes:
                    continue
                tx1 = min(bbox[0] for bbox in cell_bboxes)
                ty1 = min(bbox[1] for bbox in cell_bboxes)
                tx2 = max(bbox[2] for bbox in cell_bboxes)
                ty2 = max(bbox[3] for bbox in cell_bboxes)
            else:
                continue
            
            # Check for overlap
            if not (x2 < tx1 or x1 > tx2 or y2 < ty1 or y1 > ty2):
                return table
        
        return None
    
    def _render_text_in_table(self, draw: ImageDraw.Draw, x1: int, y1: int, x2: int, y2: int, text: str, table: Dict) -> None:
        """Render text within a table structure."""
        # Draw table border if not already drawn
        if 'bbox' in table:
            tx1, ty1, tx2, ty2 = table['bbox']
            draw.rectangle([tx1, ty1, tx2, ty2], outline=(0, 0, 0), width=2)
        
        # Render text with table styling
        self._draw_text_with_styling(draw, x1, y1, x2, y2, text, font_size=14, color=(0, 0, 0))
        print(f"    Rendered in table: '{text}'")
    
    def _render_text_in_layout(self, draw: ImageDraw.Draw, x1: int, y1: int, x2: int, y2: int, text: str, layout_element: Dict) -> None:
        """Render text within a layout structure."""
        label = layout_element.get('label', 'unknown')
        
        # Draw layout border
        if 'bbox' in layout_element:
            lx1, ly1, lx2, ly2 = layout_element['bbox']
        else:
            lx1, ly1, lx2, ly2 = layout_element['l'], layout_element['t'], layout_element['r'], layout_element['b']
        
        # Choose border color based on layout type
        border_color = (100, 100, 100)  # Default
        if label == 'section-header':
            border_color = (0, 100, 0)
        elif label == 'key-value':
            border_color = (100, 0, 100)
        elif label == 'list-item':
            border_color = (0, 0, 100)
        
        draw.rectangle([lx1, ly1, lx2, ly2], outline=border_color, width=2)
        
        # Render text with layout styling
        font_size = 16
        if label == 'section-header':
            font_size = 18
        
        self._draw_text_with_styling(draw, x1, y1, x2, y2, text, font_size=font_size, color=self.text_color)
        print(f"    Rendered in {label}: '{text}'")
    
    def _render_plain_text(self, draw: ImageDraw.Draw, x1: int, y1: int, x2: int, y2: int, text: str) -> None:
        """Render plain text without any structure."""
        self._draw_text_with_styling(draw, x1, y1, x2, y2, text, font_size=16, color=self.text_color)
        print(f"    Rendered as plain text: '{text}'")
    
    def _draw_text_with_styling(self, draw: ImageDraw.Draw, x1: int, y1: int, x2: int, y2: int, text: str, font_size: int, color: Tuple[int, int, int]) -> None:
        """Draw text with specified styling."""
        # Try to get font
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("calibri.ttf", font_size)
            except:
                font = ImageFont.load_default()
        
        # Calculate text position
        text_width, text_height = draw.textsize(text, font=font)
        text_x = x1 + (x2 - x1 - text_width) // 2
        text_y = y1 + (y2 - y1 - text_height) // 2
        
        # Ensure text fits within bounds
        if text_x < x1:
            text_x = x1
        if text_y < y1:
            text_y = y1
        
        # Draw text
        draw.text((text_x, text_y), text, fill=color, font=font)
    
    def save_integrated_reconstruction(self, canvas: Image.Image, output_path: str) -> None:
        """Save the integrated reconstruction."""
        if canvas:
            canvas.save(output_path)
            print(f"Integrated reconstruction saved to: {output_path}")
        else:
            print("No canvas to save")


def main():
    """Main function for OCR integration."""
    parser = argparse.ArgumentParser(description="Integrate OCR text with reconstructed structures")
    parser.add_argument("--layout-file", required=True, help="Path to Layout predictions JSON")
    parser.add_argument("--tableformer-file", required=True, help="Path to TableFormer results JSON")
    parser.add_argument("--ocr-file", required=True, help="Path to OCR results JSON")
    parser.add_argument("--output", "-o", required=True, help="Output PDF path")
    parser.add_argument("--image-name", help="Image name for debugging")
    
    args = parser.parse_args()
    
    print(f"=== OCR Integration ===")
    print(f"Layout file: {args.layout_file}")
    print(f"TableFormer file: {args.tableformer_file}")
    print(f"OCR file: {args.ocr_file}")
    print(f"Output: {args.output}")
    
    # Initialize integrator
    integrator = OCRIntegrator()
    
    # Load data
    layout_elements = integrator.load_layout_data(args.layout_file)
    tables = integrator.load_tableformer_data(args.tableformer_file)
    ocr_blocks = integrator.load_ocr_data(args.ocr_file)
    
    print(f"Loaded {len(layout_elements)} layout elements, {len(tables)} tables, {len(ocr_blocks)} OCR blocks")
    
    # Integrate OCR with structures
    canvas = integrator.integrate_ocr_with_structures(layout_elements, tables, ocr_blocks)
    
    # Save result
    if canvas:
        integrator.save_integrated_reconstruction(canvas, args.output)
        print("OCR integration completed successfully!")
    else:
        print("OCR integration failed - no OCR blocks found")


if __name__ == "__main__":
    main()
