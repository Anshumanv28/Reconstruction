#!/usr/bin/env python3
"""
OCR Visualization Module

This module creates a simple visualization of OCR text blocks with grid overlay.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw, ImageFont


class OCRVisualizer:
    """Handles OCR text visualization with grid overlay."""
    
    def __init__(self, 
                 background_color: Tuple[int, int, int] = (255, 255, 255),
                 text_color: Tuple[int, int, int] = (0, 0, 0),
                 grid_size: int = 50):
        """Initialize the OCR visualizer."""
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
    
    def calculate_document_dimensions(self, ocr_blocks: List[Dict]) -> Tuple[int, int]:
        """Calculate overall document dimensions based on OCR bounding boxes."""
        if not ocr_blocks:
            return 1000, 1000  # Default size
        
        max_width = 0
        max_height = 0
        
        for block in ocr_blocks:
            if 'bbox' in block:
                x1, y1, x2, y2 = block['bbox']
                max_width = max(max_width, int(x2))
                max_height = max(max_height, int(y2))
        
        # Add padding
        return max_width + 100, max_height + 100
    
    def visualize_ocr_text(self, ocr_blocks: List[Dict]) -> Image.Image:
        """Create visualization of OCR text blocks."""
        if not ocr_blocks:
            print("No OCR blocks found")
            return None
        
        # Calculate document dimensions
        width, height = self.calculate_document_dimensions(ocr_blocks)
        print(f"OCR document dimensions: {width} x {height}")
        
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
            self._draw_ocr_block(draw, ocr_block, i)
        
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
    
    def _draw_ocr_block(self, draw: ImageDraw.Draw, ocr_block: Dict, block_index: int) -> None:
        """Draw a single OCR text block."""
        x1, y1, x2, y2 = ocr_block['bbox']
        text = ocr_block.get('text', '').strip()
        
        if not text:
            return
        
        # Convert to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 255), width=1)
        
        # Try to get font
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            try:
                font = ImageFont.truetype("calibri.ttf", 12)
            except:
                font = ImageFont.load_default()
        
        # Calculate text position (center of box)
        try:
            # Newer PIL versions
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            # Older PIL versions
            text_width, text_height = draw.textsize(text, font=font)
        
        text_x = x1 + (x2 - x1 - text_width) // 2
        text_y = y1 + (y2 - y1 - text_height) // 2
        
        # Ensure text fits within bounds
        if text_x < x1:
            text_x = x1
        if text_y < y1:
            text_y = y1
        
        # Draw text
        draw.text((text_x, text_y), text, fill=self.text_color, font=font)
        
        print(f"  OCR Block {block_index + 1}: '{text}' at ({x1}, {y1}) to ({x2}, {y2})")
    
    def save_ocr_visualization(self, canvas: Image.Image, output_path: str) -> None:
        """Save the OCR visualization."""
        if canvas:
            canvas.save(output_path)
            print(f"OCR visualization saved to: {output_path}")
        else:
            print("No canvas to save")


def main():
    """Main function for OCR visualization."""
    parser = argparse.ArgumentParser(description="Visualize OCR text blocks with grid overlay")
    parser.add_argument("--ocr-file", required=True, help="Path to OCR results JSON")
    parser.add_argument("--output", "-o", required=True, help="Output PDF path")
    parser.add_argument("--image-name", help="Image name for debugging")
    
    args = parser.parse_args()
    
    print(f"=== OCR Visualization ===")
    print(f"OCR file: {args.ocr_file}")
    print(f"Output: {args.output}")
    
    # Initialize visualizer
    visualizer = OCRVisualizer()
    
    # Load OCR data
    ocr_blocks = visualizer.load_ocr_data(args.ocr_file)
    print(f"Loaded {len(ocr_blocks)} OCR blocks")
    
    # Create visualization
    canvas = visualizer.visualize_ocr_text(ocr_blocks)
    
    # Save result
    if canvas:
        visualizer.save_ocr_visualization(canvas, args.output)
        print("OCR visualization completed successfully!")
    else:
        print("OCR visualization failed - no OCR blocks found")


if __name__ == "__main__":
    main()
