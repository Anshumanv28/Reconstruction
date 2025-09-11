#!/usr/bin/env python3
"""
Batch Integrated Visualization Pipeline

This module processes multiple images and creates integrated visualizations
combining layout, table, and OCR data for each image.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont


class BatchIntegratedVisualizer:
    """Handles batch processing of integrated visualizations."""
    
    def __init__(self, 
                 background_color: Tuple[int, int, int] = (255, 255, 255),
                 text_color: Tuple[int, int, int] = (0, 0, 0),
                 grid_size: int = 50):
        """Initialize the batch visualizer."""
        self.background_color = background_color
        self.text_color = text_color
        self.grid_size = grid_size
        
    def get_available_images(self, input_dir: str) -> List[str]:
        """Get list of available images from input directory."""
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"Input directory does not exist: {input_dir}")
            return []
        
        # Look for common image extensions
        image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
        images = []
        
        for file_path in input_path.iterdir():
            if file_path.suffix.lower() in image_extensions:
                # Get base name without extension
                base_name = file_path.stem
                images.append(base_name)
        
        print(f"Found {len(images)} images in {input_dir}")
        return sorted(images)
    
    def check_data_availability(self, base_name: str, intermediate_dir: str) -> Dict[str, bool]:
        """Check which data files are available for a given image."""
        intermediate_path = Path(intermediate_dir)
        
        # Check for required files
        layout_file = intermediate_path / "layout_outputs" / f"{base_name}_layout_predictions.json"
        tableformer_file = intermediate_path / "tableformer_outputs" / f"{base_name}_tableformer_results.json"
        ocr_file = intermediate_path / "ocr_outputs" / f"{base_name}_ocr_results.json"
        
        availability = {
            'layout': layout_file.exists(),
            'tableformer': tableformer_file.exists(),
            'ocr': ocr_file.exists(),
            'layout_file': str(layout_file) if layout_file.exists() else None,
            'tableformer_file': str(tableformer_file) if tableformer_file.exists() else None,
            'ocr_file': str(ocr_file) if ocr_file.exists() else None
        }
        
        return availability
    
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
            
            return tables
        
        # Fallback to original logic
        if 'tables' in data:
            return data['tables']
        elif 'tableformer_results' in data:
            return data['tableformer_results']
        else:
            return data  # Assume data is already the table list
    
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
    
    def create_integrated_visualization(self, layout_elements: List[Dict], tables: List[Dict], ocr_blocks: List[Dict]) -> Image.Image:
        """Create integrated visualization combining all data sources."""
        # Calculate document dimensions
        width, height = self.calculate_document_dimensions(layout_elements, tables, ocr_blocks)
        
        # Create canvas
        canvas = Image.new('RGB', (width, height), self.background_color)
        draw = ImageDraw.Draw(canvas)
        
        # Draw grid
        self._draw_grid(draw, width, height)
        
        # 1. Draw layout structure (borders only, no text)
        self._draw_layout_structure(draw, layout_elements)
        
        # 2. Draw table structure (cell borders only, no text)
        self._draw_table_structure(draw, tables)
        
        # 3. Draw OCR text (actual text content)
        self._draw_ocr_text(draw, ocr_blocks)
        
        return canvas
    
    def _draw_grid(self, draw: ImageDraw.Draw, width: int, height: int) -> None:
        """Draw a grid overlay to help visualize positioning."""
        grid_size = 50  # Grid spacing in pixels
        
        # Draw vertical lines
        for x in range(0, width, grid_size):
            draw.line([(x, 0), (x, height)], fill=(220, 220, 220), width=1)
        
        # Draw horizontal lines
        for y in range(0, height, grid_size):
            draw.line([(0, y), (width, y)], fill=(220, 220, 220), width=1)
    
    def _draw_layout_structure(self, draw: ImageDraw.Draw, layout_elements: List[Dict]) -> None:
        """Draw layout structure borders only (no placeholder text)."""
        # Filter out overlapping elements to avoid visual confusion
        filtered_elements = self._filter_overlapping_elements(layout_elements)
        
        # Sort elements by position (top to bottom, left to right)
        sorted_elements = sorted(filtered_elements, key=lambda x: (x.get('t', 0), x.get('l', 0)))
        
        # Draw each layout element border
        for element in sorted_elements:
            self._draw_layout_border(draw, element)
    
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
        
        return filtered_elements
    
    def _draw_layout_border(self, draw: ImageDraw.Draw, element: Dict) -> None:
        """Draw layout element border only (no text)."""
        # Get bounding box coordinates
        if 'bbox' in element:
            x1, y1, x2, y2 = element['bbox']
        elif all(key in element for key in ['l', 't', 'r', 'b']):
            x1, y1, x2, y2 = element['l'], element['t'], element['r'], element['b']
        else:
            return
        
        # Convert to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        label = element.get('label', 'unknown')
        
        # Choose border color based on layout type
        border_color = (100, 100, 100)  # Default
        border_width = 1
        
        if label.lower() == 'section-header':
            border_color = (0, 100, 0)  # Green
            border_width = 2
        elif label.lower() == 'table':
            border_color = (0, 0, 0)    # Black
            border_width = 2
        elif label.lower() == 'key-value':
            border_color = (100, 0, 100)  # Purple
            border_width = 2
        elif label.lower() == 'list-item':
            border_color = (0, 0, 100)  # Blue
            border_width = 2
        
        # Draw border
        draw.rectangle([x1, y1, x2, y2], outline=border_color, width=border_width)
    
    def _draw_table_structure(self, draw: ImageDraw.Draw, tables: List[Dict]) -> None:
        """Draw table structure borders only (no placeholder text)."""
        for table in tables:
            # Draw table border
            if 'bbox' in table:
                x1, y1, x2, y2 = table['bbox']
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 0), width=2)
            
            # Draw cell borders
            if 'cells' in table:
                for cell in table['cells']:
                    if 'bbox' in cell:
                        x1, y1, x2, y2 = cell['bbox']
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        draw.rectangle([x1, y1, x2, y2], outline=(100, 100, 100), width=1)
    
    def _draw_ocr_text(self, draw: ImageDraw.Draw, ocr_blocks: List[Dict]) -> None:
        """Draw OCR text content."""
        # Sort OCR blocks by position (top to bottom, left to right)
        sorted_ocr_blocks = sorted(ocr_blocks, key=lambda x: (x['bbox'][1], x['bbox'][0]))
        
        # Process each OCR block
        for ocr_block in sorted_ocr_blocks:
            self._draw_single_ocr_text(draw, ocr_block)
    
    def _draw_single_ocr_text(self, draw: ImageDraw.Draw, ocr_block: Dict) -> None:
        """Draw a single OCR text block."""
        x1, y1, x2, y2 = ocr_block['bbox']
        text = ocr_block.get('text', '').strip()
        
        if not text:
            return
        
        # Convert to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Try to get font
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            try:
                font = ImageFont.truetype("calibri.ttf", 14)
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
    
    def process_single_image(self, base_name: str, intermediate_dir: str, output_dir: str) -> bool:
        """Process a single image and create integrated visualization."""
        print(f"\n=== Processing: {base_name} ===")
        
        # Check data availability
        availability = self.check_data_availability(base_name, intermediate_dir)
        
        if not availability['layout']:
            print(f"  ❌ Layout data not found for {base_name}")
            return False
        
        if not availability['ocr']:
            print(f"  ❌ OCR data not found for {base_name}")
            return False
        
        # TableFormer data is optional (documents without tables won't have it)
        if availability['tableformer']:
            print(f"  ✅ Layout, OCR, and TableFormer data found for {base_name}")
        else:
            print(f"  ✅ Layout and OCR data found for {base_name} (no tables detected)")
        
        try:
            # Load required data
            layout_elements = self.load_layout_data(availability['layout_file'])
            ocr_blocks = self.load_ocr_data(availability['ocr_file'])
            
            # Load optional TableFormer data
            tables = []
            if availability['tableformer']:
                tables = self.load_tableformer_data(availability['tableformer_file'])
                print(f"  📊 Loaded {len(layout_elements)} layout elements, {len(tables)} tables, {len(ocr_blocks)} OCR blocks")
            else:
                print(f"  📊 Loaded {len(layout_elements)} layout elements, 0 tables, {len(ocr_blocks)} OCR blocks")
            
            # Create integrated visualization
            canvas = self.create_integrated_visualization(layout_elements, tables, ocr_blocks)
            
            # Save result
            output_path = Path(output_dir) / f"{base_name}_integrated_visualization.pdf"
            canvas.save(str(output_path))
            
            print(f"  ✅ Integrated visualization saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"  ❌ Error processing {base_name}: {str(e)}")
            return False
    
    def run_batch_processing(self, input_dir: str, intermediate_dir: str, output_dir: str) -> Dict[str, bool]:
        """Run batch processing for all available images."""
        print(f"=== Batch Integrated Visualization Pipeline ===")
        print(f"Input directory: {input_dir}")
        print(f"Intermediate directory: {intermediate_dir}")
        print(f"Output directory: {output_dir}")
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get available images
        images = self.get_available_images(input_dir)
        if not images:
            print("No images found to process")
            return {}
        
        # Process each image
        results = {}
        successful = 0
        failed = 0
        
        for base_name in images:
            success = self.process_single_image(base_name, intermediate_dir, output_dir)
            results[base_name] = success
            
            if success:
                successful += 1
            else:
                failed += 1
        
        # Summary
        print(f"\n=== Batch Processing Complete ===")
        print(f"Total images: {len(images)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {successful/len(images)*100:.1f}%")
        
        return results


def main():
    """Main function for batch integrated visualization."""
    parser = argparse.ArgumentParser(description="Batch process integrated visualizations")
    parser.add_argument("--input-dir", default="../pipe_input", help="Input directory with images")
    parser.add_argument("--intermediate-dir", default="../intermediate_outputs", help="Intermediate outputs directory")
    parser.add_argument("--output-dir", default="intermediate_outputs/batch_integrated_visualizations", help="Output directory for integrated visualizations")
    parser.add_argument("--image", help="Process specific image (base name without extension)")
    
    args = parser.parse_args()
    
    # Initialize batch processor
    processor = BatchIntegratedVisualizer()
    
    if args.image:
        # Process single image
        success = processor.process_single_image(args.image, args.intermediate_dir, args.output_dir)
        if success:
            print(f"✅ Successfully processed {args.image}")
        else:
            print(f"❌ Failed to process {args.image}")
    else:
        # Process all images
        results = processor.run_batch_processing(args.input_dir, args.intermediate_dir, args.output_dir)


if __name__ == "__main__":
    main()
