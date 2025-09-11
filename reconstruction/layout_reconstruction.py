#!/usr/bin/env python3
"""
Layout Reconstruction Module

This module handles the reconstruction of overall document layout using Layout output.
It focuses on drawing layout structures (headers, sections, lists, etc.) without OCR text integration.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set
from PIL import Image, ImageDraw, ImageFont
import numpy as np


class LayoutReconstructor:
    """Handles layout reconstruction using Layout output."""
    
    def __init__(self, 
                 background_color: Tuple[int, int, int] = (255, 255, 255),
                 text_color: Tuple[int, int, int] = (0, 0, 0),
                 grid_size: int = 20):
        """Initialize the layout reconstructor."""
        self.background_color = background_color
        self.text_color = text_color
        self.grid_size = grid_size
        
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
    
    def calculate_layout_dimensions(self, layout_elements: List[Dict]) -> Tuple[int, int]:
        """Calculate overall document dimensions based on layout bounding boxes."""
        if not layout_elements:
            return 1000, 1000  # Default size
        
        max_width = 0
        max_height = 0
        
        for element in layout_elements:
            if 'bbox' in element:
                x1, y1, x2, y2 = element['bbox']
                max_width = max(max_width, int(x2))
                max_height = max(max_height, int(y2))
            elif all(key in element for key in ['l', 't', 'r', 'b']):
                l, t, r, b = element['l'], element['t'], element['r'], element['b']
                max_width = max(max_width, int(r))
                max_height = max(max_height, int(b))
        
        # Add padding
        return max_width + 100, max_height + 100
    
    def reconstruct_layout(self, layout_elements: List[Dict]) -> Image.Image:
        """Reconstruct layout by drawing structural elements."""
        if not layout_elements:
            print("No layout elements found in Layout data")
            return None
        
        # Calculate document dimensions
        width, height = self.calculate_layout_dimensions(layout_elements)
        print(f"Layout document dimensions: {width} x {height}")
        
        # Create canvas
        canvas = Image.new('RGB', (width, height), self.background_color)
        draw = ImageDraw.Draw(canvas)
        
        # Draw grid
        self._draw_grid(draw, width, height)
        
        # Filter out overlapping elements to avoid visual confusion
        filtered_elements = self._filter_overlapping_elements(layout_elements)
        print(f"Filtered {len(layout_elements)} elements down to {len(filtered_elements)} non-overlapping elements")
        
        # Sort elements by position (top to bottom, left to right)
        sorted_elements = sorted(filtered_elements, key=lambda x: (x.get('t', 0), x.get('l', 0)))
        
        # Process each layout element
        for i, element in enumerate(sorted_elements):
            print(f"Processing layout element {i+1}/{len(sorted_elements)}: {element.get('label', 'unknown')}")
            self._draw_layout_element(draw, element, i)
        
        return canvas
    
    def reconstruct_section_headers_only(self, layout_elements: List[Dict]) -> Image.Image:
        """Reconstruct layout showing only section headers."""
        if not layout_elements:
            print("No layout elements found in Layout data")
            return None
        
        # Calculate document dimensions
        width, height = self.calculate_layout_dimensions(layout_elements)
        print(f"Section headers document dimensions: {width} x {height}")
        
        # Create canvas
        canvas = Image.new('RGB', (width, height), self.background_color)
        draw = ImageDraw.Draw(canvas)
        
        # Draw grid
        self._draw_grid(draw, width, height)
        
        # Filter elements to only section headers
        section_headers = [elem for elem in layout_elements if elem.get('label', '').lower() == 'section-header']
        print(f"Found {len(section_headers)} section headers")
        
        # Sort elements by position (top to bottom, left to right)
        sorted_elements = sorted(section_headers, key=lambda x: (x.get('t', 0), x.get('l', 0)))
        
        # Process each section header
        for i, element in enumerate(sorted_elements):
            print(f"Processing section header {i+1}/{len(sorted_elements)}")
            self._draw_layout_element(draw, element, i)
        
        return canvas
    
    def reconstruct_text_and_others(self, layout_elements: List[Dict]) -> Image.Image:
        """Reconstruct layout showing text and other elements (excluding section headers)."""
        if not layout_elements:
            print("No layout elements found in Layout data")
            return None
        
        # Calculate document dimensions
        width, height = self.calculate_layout_dimensions(layout_elements)
        print(f"Text and others document dimensions: {width} x {height}")
        
        # Create canvas
        canvas = Image.new('RGB', (width, height), self.background_color)
        draw = ImageDraw.Draw(canvas)
        
        # Draw grid
        self._draw_grid(draw, width, height)
        
        # Filter elements to exclude section headers
        other_elements = [elem for elem in layout_elements if elem.get('label', '').lower() != 'section-header']
        print(f"Found {len(other_elements)} non-section-header elements")
        
        # Filter out overlapping elements
        filtered_elements = self._filter_overlapping_elements(other_elements)
        print(f"Filtered {len(other_elements)} elements down to {len(filtered_elements)} non-overlapping elements")
        
        # Sort elements by position (top to bottom, left to right)
        sorted_elements = sorted(filtered_elements, key=lambda x: (x.get('t', 0), x.get('l', 0)))
        
        # Process each element
        for i, element in enumerate(sorted_elements):
            print(f"Processing element {i+1}/{len(sorted_elements)}: {element.get('label', 'unknown')}")
            self._draw_layout_element(draw, element, i)
        
        return canvas
    
    def _filter_overlapping_elements(self, layout_elements: List[Dict]) -> List[Dict]:
        """Filter out overlapping elements, keeping only the highest priority one in each group."""
        # Define element priority (higher number = higher priority)
        element_priority = {
            'section-header': 5,
            'table': 4,
            'key-value': 3,
            'list-item': 2,
            'text': 1,
            'form': 1,
            'unknown': 0
        }
        
        filtered_elements = []
        processed_coordinates = set()
        
        # Sort elements by priority (highest first), then by position
        sorted_elements = sorted(layout_elements, 
                               key=lambda x: (element_priority.get(x.get('label', 'unknown'), 0), 
                                            x.get('t', 0), x.get('l', 0)), 
                               reverse=True)
        
        for element in sorted_elements:
            # Get bounding box coordinates
            if 'bbox' in element:
                x1, y1, x2, y2 = element['bbox']
            elif all(key in element for key in ['l', 't', 'r', 'b']):
                x1, y1, x2, y2 = element['l'], element['t'], element['r'], element['b']
            else:
                continue
            
            # Convert to integers and create coordinate key
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            coord_key = (x1, y1, x2, y2)
            
            # Check if this coordinate has already been processed
            if coord_key not in processed_coordinates:
                filtered_elements.append(element)
                processed_coordinates.add(coord_key)
                print(f"  Kept {element.get('label', 'unknown')} at ({x1}, {y1}) to ({x2}, {y2})")
            else:
                print(f"  Skipped overlapping {element.get('label', 'unknown')} at ({x1}, {y1}) to ({x2}, {y2})")
        
        return filtered_elements
    
    def _draw_grid(self, draw: ImageDraw.Draw, width: int, height: int) -> None:
        """Draw a grid overlay to help visualize layout."""
        grid_size = 50  # Grid spacing in pixels
        
        # Draw vertical lines
        for x in range(0, width, grid_size):
            draw.line([(x, 0), (x, height)], fill=(220, 220, 220), width=1)
        
        # Draw horizontal lines
        for y in range(0, height, grid_size):
            draw.line([(0, y), (width, y)], fill=(220, 220, 220), width=1)
        
        print(f"  Drew grid with {grid_size}px spacing")
    
    def _draw_layout_element(self, draw: ImageDraw.Draw, element: Dict, element_index: int) -> None:
        """Draw a single layout element."""
        # Get bounding box coordinates
        if 'bbox' in element:
            x1, y1, x2, y2 = element['bbox']
        elif all(key in element for key in ['l', 't', 'r', 'b']):
            x1, y1, x2, y2 = element['l'], element['t'], element['r'], element['b']
        else:
            print(f"  Element {element_index + 1}: No valid bounding box found")
            return
        
        # Convert to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        label = element.get('label', 'unknown')
        print(f"  Element {element_index + 1}: {label} at ({x1}, {y1}) to ({x2}, {y2})")
        
        # Draw based on element type
        if label == 'table':
            self._draw_table_layout(draw, x1, y1, x2, y2, element_index)
        elif label == 'section-header':
            self._draw_header_layout(draw, x1, y1, x2, y2, element_index)
        elif label == 'key-value':
            self._draw_keyvalue_layout(draw, x1, y1, x2, y2, element_index)
        elif label == 'list-item':
            self._draw_listitem_layout(draw, x1, y1, x2, y2, element_index)
        else:
            self._draw_generic_layout(draw, x1, y1, x2, y2, element_index, label)
    
    def _draw_table_layout(self, draw: ImageDraw.Draw, x1: int, y1: int, x2: int, y2: int, index: int) -> None:
        """Draw table layout structure."""
        # Draw table border
        draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 0), width=2)
        
        # Add table label
        self._draw_placeholder_text(draw, x1, y1, x2, y2, f"Table {index + 1}")
    
    def _draw_header_layout(self, draw: ImageDraw.Draw, x1: int, y1: int, x2: int, y2: int, index: int) -> None:
        """Draw header layout structure."""
        # Draw header border
        draw.rectangle([x1, y1, x2, y2], outline=(0, 100, 0), width=2)
        
        # Add header label
        self._draw_placeholder_text(draw, x1, y1, x2, y2, f"Header {index + 1}")
    
    def _draw_keyvalue_layout(self, draw: ImageDraw.Draw, x1: int, y1: int, x2: int, y2: int, index: int) -> None:
        """Draw key-value layout structure."""
        # Draw key-value border
        draw.rectangle([x1, y1, x2, y2], outline=(100, 0, 100), width=2)
        
        # Add key-value label
        self._draw_placeholder_text(draw, x1, y1, x2, y2, f"Key-Value {index + 1}")
    
    def _draw_listitem_layout(self, draw: ImageDraw.Draw, x1: int, y1: int, x2: int, y2: int, index: int) -> None:
        """Draw list item layout structure."""
        # Draw list item border
        draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 100), width=2)
        
        # Add list item label
        self._draw_placeholder_text(draw, x1, y1, x2, y2, f"List Item {index + 1}")
    
    def _draw_generic_layout(self, draw: ImageDraw.Draw, x1: int, y1: int, x2: int, y2: int, index: int, label: str) -> None:
        """Draw generic layout structure."""
        # Draw generic border
        draw.rectangle([x1, y1, x2, y2], outline=(100, 100, 100), width=1)
        
        # Add generic label
        self._draw_placeholder_text(draw, x1, y1, x2, y2, f"{label} {index + 1}")
    
    def _draw_placeholder_text(self, draw: ImageDraw.Draw, x1: int, y1: int, x2: int, y2: int, text: str) -> None:
        """Draw placeholder text in the center of a bounding box."""
        # Try to get font
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            try:
                font = ImageFont.truetype("calibri.ttf", 14)
            except:
                font = ImageFont.load_default()
        
        # Calculate text position (center of box) - using textbbox for newer PIL versions
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
        
        # Draw placeholder text
        draw.text((text_x, text_y), text, fill=self.text_color, font=font)
    
    def save_layout_reconstruction(self, canvas: Image.Image, output_path: str) -> None:
        """Save the layout reconstruction."""
        if canvas:
            # Ensure the output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            canvas.save(output_path)
            print(f"Layout reconstruction saved to: {output_path}")
        else:
            print("No canvas to save")


def main():
    """Main function for layout reconstruction."""
    parser = argparse.ArgumentParser(description="Reconstruct layout using Layout output")
    parser.add_argument("--layout-file", required=True, help="Path to Layout predictions JSON")
    parser.add_argument("--output", "-o", required=True, help="Output PDF path")
    parser.add_argument("--mode", choices=['all', 'headers', 'others'], default='all', 
                       help="Reconstruction mode: all=complete layout, headers=section headers only, others=text and other elements")
    parser.add_argument("--image-name", help="Image name for debugging")
    
    args = parser.parse_args()
    
    print(f"=== Layout Reconstruction ===")
    print(f"Layout file: {args.layout_file}")
    print(f"Output: {args.output}")
    print(f"Mode: {args.mode}")
    
    # Initialize reconstructor
    reconstructor = LayoutReconstructor()
    
    # Load layout data
    layout_elements = reconstructor.load_layout_data(args.layout_file)
    print(f"Loaded {len(layout_elements)} layout elements")
    
    # Reconstruct based on mode
    if args.mode == 'headers':
        canvas = reconstructor.reconstruct_section_headers_only(layout_elements)
    elif args.mode == 'others':
        canvas = reconstructor.reconstruct_text_and_others(layout_elements)
    else:  # 'all'
        canvas = reconstructor.reconstruct_layout(layout_elements)
    
    # Save result
    if canvas:
        reconstructor.save_layout_reconstruction(canvas, args.output)
        print("Layout reconstruction completed successfully!")
    else:
        print("Layout reconstruction failed - no layout elements found")


if __name__ == "__main__":
    main()
