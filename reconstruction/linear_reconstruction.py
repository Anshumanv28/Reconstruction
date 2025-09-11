#!/usr/bin/env python3
"""
Linear Reconstruction Approach for OCR Pipeline

This module implements a linear scanning approach for document reconstruction:
1. Create a standard grid across the document
2. Scan from top-left to bottom-right
3. When we hit a coordinate (either from layout or OCR), stop and look for the corresponding coordinate in the other data source
4. Render the word and continue scanning
"""

import os
import json
import argparse
from typing import List, Dict, Tuple, Set
from PIL import Image, ImageDraw, ImageFont


class LinearReconstructor:
    """Linear scanning approach for document reconstruction."""
    
    def __init__(self):
        """Initialize the linear reconstructor."""
        # Document styling
        self.background_color = (255, 255, 255)  # White background
        self.text_color = (0, 0, 0)  # Black text
        
        # Grid settings
        self.grid_size = 20  # pixels - scan every 20 pixels
        
    def load_specific_image_data(self, intermediate_dir: str, image_name: str) -> Tuple[Dict, Dict, Dict, str]:
        """Load data for a specific image from pipeline outputs."""
        layout_path = os.path.join(intermediate_dir, "layout_outputs")
        tableformer_path = os.path.join(intermediate_dir, "tableformer_outputs")
        ocr_path = os.path.join(intermediate_dir, "ocr_outputs")
        
        # Construct expected filenames
        base_name = image_name.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
        layout_file = os.path.join(layout_path, f"{base_name}_layout_predictions.json")
        tableformer_file = os.path.join(tableformer_path, f"{base_name}_tableformer_results.json")
        ocr_file = os.path.join(ocr_path, f"{base_name}_ocr_results.json")
        
        # Check if files exist
        if not os.path.exists(layout_file):
            raise FileNotFoundError(f"Layout file not found: {layout_file}")
        if not os.path.exists(tableformer_file):
            raise FileNotFoundError(f"Tableformer file not found: {tableformer_file}")
        if not os.path.exists(ocr_file):
            raise FileNotFoundError(f"OCR file not found: {ocr_file}")
        
        print(f"Loading layout data from: {layout_file}")
        print(f"Loading tableformer data from: {tableformer_file}")
        print(f"Loading OCR data from: {ocr_file}")
        
        # Load the data
        with open(layout_file, 'r', encoding='utf-8') as f:
            layout_data = json.load(f)
        
        with open(tableformer_file, 'r', encoding='utf-8') as f:
            tableformer_data = json.load(f)
        
        with open(ocr_file, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
        
        return layout_data, tableformer_data, ocr_data, base_name
    
    def calculate_document_dimensions(self, layout_data: List[Dict], tableformer_data: List[Dict]) -> Tuple[int, int]:
        """Calculate document dimensions from layout and tableformer data."""
        max_width = 0
        max_height = 0
        
        # Check layout data
        for element in layout_data:
            max_width = max(max_width, element['r'])
            max_height = max(max_height, element['b'])
        
        # Check tableformer data
        for table_result in tableformer_data:
            if 'predict_details' in table_result:
                table_bbox = table_result['predict_details'].get('table_bbox', [])
                if len(table_bbox) >= 4:
                    max_width = max(max_width, table_bbox[2])
                    max_height = max(max_height, table_bbox[3])
        
        # Add some padding
        return int(max_width + 50), int(max_height + 50)
    
    def reconstruct_with_linear_scanning(self, layout_data: List[Dict], tableformer_data: List[Dict], 
                                       ocr_data: Dict, doc_width: int, doc_height: int, 
                                       output_dir: str, base_name: str, translate: bool = False) -> None:
        """Reconstruct document using true linear 3-pointer approach: 2 read pointers + 1 write pointer."""
        
        # Extract OCR text blocks
        ocr_text_blocks = []
        if 'full_document_ocr' in ocr_data and 'text_blocks' in ocr_data['full_document_ocr']:
            for block in ocr_data['full_document_ocr']['text_blocks']:
                if 'bbox' in block and 'text' in block:
                    ocr_text_blocks.append({
                        'bbox': block['bbox'],
                        'text': block['text'],
                        'confidence': block.get('confidence', 1.0)
                    })
        
        print(f"Found {len(ocr_text_blocks)} OCR text blocks")
        print(f"Found {len(layout_data)} layout elements")
        print(f"Found {len(tableformer_data)} tableformer elements")
        
        # Debug: Show first few OCR blocks
        print("First 5 OCR blocks:")
        for i, block in enumerate(ocr_text_blocks[:5]):
            print(f"  {i+1}: '{block['text']}' at {block['bbox']}")
        
        # Debug: Check for empty or very short text blocks
        empty_blocks = [block for block in ocr_text_blocks if not block['text'].strip() or len(block['text'].strip()) < 2]
        if empty_blocks:
            print(f"WARNING: Found {len(empty_blocks)} empty or very short OCR blocks")
            for i, block in enumerate(empty_blocks[:3]):
                print(f"  Empty block {i+1}: '{block['text']}' at {block['bbox']}")
        
        # Debug: Check for specific missing words
        missing_words = ["Name", "Team", "Date:", "Time:", "Purpose", "Recognition", "Needed"]
        for word in missing_words:
            found_blocks = [block for block in ocr_text_blocks if word in block['text']]
            if found_blocks:
                print(f"FOUND '{word}': {len(found_blocks)} blocks")
                for block in found_blocks:
                    print(f"  '{block['text']}' at {block['bbox']}")
            else:
                print(f"MISSING '{word}': Not found in OCR blocks")
        
        # Debug: Show the first few OCR blocks that will be processed
        print("First 10 OCR blocks to be processed:")
        for i, block in enumerate(ocr_text_blocks[:10]):
            print(f"  {i+1}: '{block['text']}' at {block['bbox']}")
        
        # Create fresh document
        image = Image.new('RGB', (doc_width, doc_height), self.background_color)
        draw = ImageDraw.Draw(image)
        
        # Track what we've already drawn to avoid duplicates
        drawn_coordinates = set()
        
        # Counters for debugging
        processed_layout = 0
        processed_ocr = 0
        processed_overlap = 0
        
        print("Starting true linear 3-pointer scan: 2 read pointers + 1 write pointer...")
        
        # Sort layout and OCR data by position (top-left to bottom-right)
        layout_sorted = sorted(layout_data, key=lambda elem: (elem['t'], elem['l']))
        ocr_sorted = sorted(ocr_text_blocks, key=lambda block: (block['bbox'][1], block['bbox'][0]))
        
        # Initialize pointers
        layout_ptr = 0
        ocr_ptr = 0
        
        print(f"Processing with 2 read pointers moving together...")
        
        # Linear scan: move both read pointers together
        while layout_ptr < len(layout_sorted) or ocr_ptr < len(ocr_sorted):
            # Get current elements from both pointers
            current_layout = layout_sorted[layout_ptr] if layout_ptr < len(layout_sorted) else None
            current_ocr = ocr_sorted[ocr_ptr] if ocr_ptr < len(ocr_sorted) else None
            
            # Determine which pointer to advance based on position
            if current_layout is None:
                # Only OCR left, advance OCR pointer
                self._process_ocr_element(draw, current_ocr, drawn_coordinates)
                processed_ocr += 1
                ocr_ptr += 1
            elif current_ocr is None:
                # Only layout left, advance layout pointer
                self._process_layout_element(draw, current_layout, drawn_coordinates)
                processed_layout += 1
                layout_ptr += 1
            else:
                # Both available, compare positions
                layout_y, layout_x = current_layout['t'], current_layout['l']
                ocr_y, ocr_x = current_ocr['bbox'][1], current_ocr['bbox'][0]
                
                if (layout_y < ocr_y) or (layout_y == ocr_y and layout_x < ocr_x):
                    # Layout element comes first
                    self._process_layout_element(draw, current_layout, drawn_coordinates)
                    processed_layout += 1
                    layout_ptr += 1
                elif (ocr_y < layout_y) or (ocr_y == layout_y and ocr_x < layout_x):
                    # OCR element comes first
                    self._process_ocr_element(draw, current_ocr, drawn_coordinates)
                    processed_ocr += 1
                    ocr_ptr += 1
                else:
                    # Same position - check for overlap and process both
                    if self._elements_overlap(current_layout, current_ocr):
                        # They overlap - render with both data
                        self._process_overlapping_elements(draw, current_layout, current_ocr, drawn_coordinates)
                        processed_overlap += 1
                        layout_ptr += 1
                        ocr_ptr += 1
                    else:
                        # Same position but no overlap - process layout first
                        self._process_layout_element(draw, current_layout, drawn_coordinates)
                        processed_layout += 1
                        layout_ptr += 1
        
        # Generate output filename
        suffix = "_translated" if translate else ""
        output_filename = f"{base_name}_linear_reconstructed{suffix}.pdf"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"Saving linear reconstructed document...")
        print(f"Processing summary:")
        print(f"  Layout elements processed: {processed_layout}")
        print(f"  OCR elements processed: {processed_ocr}")
        print(f"  Overlapping elements processed: {processed_overlap}")
        print(f"  Total elements processed: {processed_layout + processed_ocr + processed_overlap}")
        
        # Convert to PDF
        image.save(output_path, 'PDF', resolution=300.0)
        
        print(f"Linear reconstructed document saved as: {output_path}")
    
    def _elements_overlap(self, layout_element: Dict, ocr_element: Dict) -> bool:
        """Check if layout and OCR elements overlap with tolerance."""
        l, t, r, b = layout_element['l'], layout_element['t'], layout_element['r'], layout_element['b']
        ox1, oy1, ox2, oy2 = ocr_element['bbox']
        
        # Add tolerance for coordinate differences between models
        tolerance = 20  # pixels
        
        # Check for overlap with tolerance
        return not (r + tolerance < ox1 or ox2 < l - tolerance or b + tolerance < oy1 or oy2 < t - tolerance)
    
    def _process_layout_element(self, draw: ImageDraw.Draw, layout_element: Dict, drawn_coordinates: Set[Tuple[int, int]]) -> None:
        """Process a layout element (no overlapping OCR) - render layout structure."""
        x, y = layout_element['l'], layout_element['t']
        
        # Check if already drawn
        if self._is_coordinate_drawn(x, y, drawn_coordinates):
            return
        
        # Render layout structure with appropriate styling
        self._render_layout_structure(draw, layout_element, drawn_coordinates)
        print(f"  Processed Layout {layout_element['label']} at ({x}, {y}) - rendered structure")
    
    def _process_ocr_element(self, draw: ImageDraw.Draw, ocr_element: Dict, drawn_coordinates: Set[Tuple[int, int]]) -> None:
        """Process an OCR element (no overlapping layout)."""
        x, y = ocr_element['bbox'][0], ocr_element['bbox'][1]
        
        # Check if already drawn
        if self._is_coordinate_drawn(x, y, drawn_coordinates):
            # Debug: Check if this is one of our missing words
            if ocr_element['text'] in ["Name", "Team", "Date:", "Time:", "Purpose", "Recognition", "Needed"]:
                print(f"  SKIPPED '{ocr_element['text']}' at ({x}, {y}) - already drawn")
            return
        
        # Render as plain text
        self._render_plain_text_at_coordinate(draw, x, y, ocr_element['text'], drawn_coordinates)
        print(f"  Processed OCR at ({x}, {y}): '{ocr_element['text'][:30]}{'...' if len(ocr_element['text']) > 30 else ''}'")
    
    def _process_overlapping_elements(self, draw: ImageDraw.Draw, layout_element: Dict, ocr_element: Dict, drawn_coordinates: Set[Tuple[int, int]]) -> None:
        """Process overlapping layout and OCR elements."""
        x, y = layout_element['l'], layout_element['t']
        
        # Check if already drawn
        if self._is_coordinate_drawn(x, y, drawn_coordinates):
            # Debug: Check if this is one of our missing words
            if ocr_element['text'] in ["Name", "Team", "Date:", "Time:", "Purpose", "Recognition", "Needed"]:
                print(f"  SKIPPED OVERLAP '{ocr_element['text']}' at ({x}, {y}) - already drawn")
            return
        
        # Render OCR text with layout styling
        self._render_text_at_coordinate(draw, x, y, ocr_element['text'], layout_element, drawn_coordinates)
        print(f"  Processed Overlap {layout_element['label']} + OCR at ({x}, {y}): '{ocr_element['text'][:30]}{'...' if len(ocr_element['text']) > 30 else ''}'")
    
    def _render_layout_structure(self, draw: ImageDraw.Draw, layout_element: Dict, drawn_coordinates: Set[Tuple[int, int]]) -> None:
        """Render layout structure with appropriate styling."""
        x, y = layout_element['l'], layout_element['t']
        bbox = [layout_element['l'], layout_element['t'], layout_element['r'], layout_element['b']]
        
        element_type = layout_element['label']
        
        if element_type == 'Table':
            # Draw table structure
            self._draw_table_structure(draw, bbox)
        elif element_type in ['Title', 'Section-header']:
            # Draw header structure
            self._draw_header_structure(draw, bbox, element_type)
        elif element_type == 'Key-Value Region':
            # Draw key-value structure
            self._draw_keyvalue_structure(draw, bbox)
        elif element_type == 'List-item':
            # Draw list item structure
            self._draw_listitem_structure(draw, bbox)
        else:
            # Draw generic layout structure
            self._draw_generic_structure(draw, bbox, element_type)
        
        # Mark this area as drawn
        drawn_coordinates.add((x, y))
    
    def _draw_table_structure(self, draw: ImageDraw.Draw, bbox: List[float]) -> None:
        """Draw table structure with borders."""
        x1, y1, x2, y2 = bbox
        
        # Draw table background
        draw.rectangle([x1, y1, x2, y2], fill=(250, 250, 250), outline=(0, 0, 0), width=2)
        
        # Draw some internal grid lines (simplified)
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        draw.line([mid_x, y1, mid_x, y2], fill=(200, 200, 200), width=1)
        draw.line([x1, mid_y, x2, mid_y], fill=(200, 200, 200), width=1)
    
    def _draw_header_structure(self, draw: ImageDraw.Draw, bbox: List[float], element_type: str) -> None:
        """Draw header structure."""
        x1, y1, x2, y2 = bbox
        
        # Draw header background
        bg_color = (240, 240, 240) if element_type == 'Section-header' else (220, 220, 220)
        draw.rectangle([x1, y1, x2, y2], fill=bg_color, outline=(0, 0, 0), width=1)
        
        # Draw underline for headers
        draw.line([x1, y2-2, x2, y2-2], fill=(0, 0, 0), width=2)
    
    def _draw_keyvalue_structure(self, draw: ImageDraw.Draw, bbox: List[float]) -> None:
        """Draw key-value structure."""
        x1, y1, x2, y2 = bbox
        
        # Draw subtle background
        draw.rectangle([x1, y1, x2, y2], fill=(245, 245, 245), outline=(180, 180, 180), width=1)
    
    def _draw_listitem_structure(self, draw: ImageDraw.Draw, bbox: List[float]) -> None:
        """Draw list item structure."""
        x1, y1, x2, y2 = bbox
        
        # Draw bullet point
        bullet_size = 4
        bullet_y = y1 + (y2 - y1) / 2
        draw.ellipse([x1, bullet_y - bullet_size, x1 + bullet_size*2, bullet_y + bullet_size], fill=(0, 0, 0))
    
    def _draw_generic_structure(self, draw: ImageDraw.Draw, bbox: List[float], element_type: str) -> None:
        """Draw generic layout structure."""
        x1, y1, x2, y2 = bbox
        
        # Draw subtle border
        draw.rectangle([x1, y1, x2, y2], outline=(200, 200, 200), width=1)
    
    def _is_coordinate_drawn(self, x: int, y: int, drawn_coordinates: Set[Tuple[int, int]]) -> bool:
        """Check if we've already drawn something near this coordinate."""
        # Use a very small tolerance to avoid skipping nearby but distinct words
        # This allows OCR text to render even if layout elements are nearby
        tolerance = 2  # Very small tolerance - only skip if almost exactly the same position
        for drawn_x, drawn_y in drawn_coordinates:
            if abs(x - drawn_x) < tolerance and abs(y - drawn_y) < tolerance:
                return True
        return False
    
    def _find_layout_element_at_coordinate(self, x: int, y: int, layout_data: List[Dict]) -> Dict:
        """Find layout element that contains this coordinate."""
        for element in layout_data:
            l, t, r, b = element['l'], element['t'], element['r'], element['b']
            if l <= x <= r and t <= y <= b:
                return element
        return None
    
    def _find_ocr_text_at_coordinate(self, x: int, y: int, ocr_text_blocks: List[Dict]) -> Dict:
        """Find OCR text that contains this coordinate."""
        for block in ocr_text_blocks:
            bbox = block['bbox']
            ox1, oy1, ox2, oy2 = bbox
            if ox1 <= x <= ox2 and oy1 <= y <= oy2:
                return block
        return None
    
    def _find_tableformer_element_at_coordinate(self, x: int, y: int, tableformer_data: List[Dict]) -> Dict:
        """Find tableformer element that contains this coordinate."""
        for table_result in tableformer_data:
            if 'predict_details' in table_result:
                table_bbox = table_result['predict_details'].get('table_bbox', [])
                if len(table_bbox) >= 4:
                    tx1, ty1, tx2, ty2 = table_bbox
                    if tx1 <= x <= tx2 and ty1 <= y <= ty2:
                        return table_result
        return None
    
    def _render_text_at_coordinate(self, draw: ImageDraw.Draw, x: int, y: int, text: str, 
                                 layout_element: Dict, drawn_coordinates: Set[Tuple[int, int]]) -> None:
        """Render text at the given coordinate using layout element styling."""
        bbox = [layout_element['l'], layout_element['t'], layout_element['r'], layout_element['b']]
        
        # Add styling based on element type
        bg_color = None
        font_size = 16  # Default font size (increased by 2 more)
        
        if layout_element['label'] in ['Title', 'Section-header']:
            bg_color = (240, 240, 240)  # Light background for headers
            font_size = 18  # Larger font for headers (increased by 2 more)
        elif layout_element['label'] == 'Table':
            font_size = 14  # Smaller font for table content (increased by 2 more)
        
        self.draw_text_element(draw, bbox, text, bg_color=bg_color, font_size=font_size)
        
        # Mark this area as drawn
        drawn_coordinates.add((x, y))
        print(f"  Rendered {layout_element['label']} at ({x}, {y}): '{text[:30]}{'...' if len(text) > 30 else ''}'")
        print(f"    DEBUG: Added coordinate ({x}, {y}) to drawn_coordinates (total: {len(drawn_coordinates)})")
    
    def _render_plain_text_at_coordinate(self, draw: ImageDraw.Draw, x: int, y: int, 
                                       text: str, drawn_coordinates: Set[Tuple[int, int]]) -> None:
        """Render plain text at the given coordinate with better sizing."""
        # Create a more appropriate bounding box around the text
        text_width = len(text) * 12  # Estimate width based on character count (increased for larger font)
        bbox = [x, y, x + text_width, y + 26]  # Better sized box for plain text (increased height)
        
        self.draw_text_element(draw, bbox, text, font_size=16)  # Fixed font size for OCR text (increased by 2 more)
        
        # Mark this area as drawn
        drawn_coordinates.add((x, y))
        print(f"  Rendered plain text at ({x}, {y}): '{text[:30]}{'...' if len(text) > 30 else ''}'")
        print(f"    DEBUG: Added coordinate ({x}, {y}) to drawn_coordinates (total: {len(drawn_coordinates)})")
    
    def _generate_generic_text_for_element(self, layout_element: Dict) -> str:
        """Generate generic text for layout element when no OCR text is found."""
        element_type = layout_element['label']
        
        if element_type == 'Title':
            return "Document Title"
        elif element_type == 'Section-header':
            return "Section Header"
        elif element_type == 'Text':
            return "Text content"
        elif element_type == 'Table':
            return "Table content"
        else:
            return f"{element_type} content"
    
    def _render_table_at_coordinate(self, draw: ImageDraw.Draw, x: int, y: int, 
                                  tableformer_data: Dict, drawn_coordinates: Set[Tuple[int, int]]) -> None:
        """Render table at the given coordinate."""
        if 'predict_details' in tableformer_data:
            table_bbox = tableformer_data['predict_details'].get('table_bbox', [])
            if len(table_bbox) >= 4:
                tx1, ty1, tx2, ty2 = table_bbox
                
                # Draw table background
                draw.rectangle([tx1, ty1, tx2, ty2], fill=(250, 250, 250), outline=(0, 0, 0))
                
                # Draw table content
                table_text = "Table content"
                self.draw_text_element(draw, table_bbox, table_text)
                
                # Mark this area as drawn
                drawn_coordinates.add((x, y))
                print(f"  Rendered Table at ({x}, {y}): '{table_text}'")
    
    def draw_text_element(self, draw: ImageDraw.Draw, bbox: List[float], text: str, bg_color: Tuple[int, int, int] = None, font_size: int = None) -> None:
        """Draw text element with proper styling."""
        x1, y1, x2, y2 = bbox
        
        # Ensure minimum size for visibility
        if x2 - x1 < 20:
            x2 = x1 + 20
        if y2 - y1 < 20:
            y2 = y1 + 20
        
        # Draw background if specified
        if bg_color:
            draw.rectangle([x1, y1, x2, y2], fill=bg_color)
        
        # Use provided font size or calculate based on bounding box
        if font_size is None:
            font_size = max(12, min(24, int((y2 - y1) * 0.6)))  # Increased minimum font size
        
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                # Try other common fonts
                font = ImageFont.truetype("calibri.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("times.ttf", font_size)
                except:
                    font = ImageFont.load_default()
        
        # Ensure text is not empty and has reasonable length
        if text and len(text.strip()) > 0:
            # Draw text with better positioning
            text_x = x1 + 3
            text_y = y1 + 3
            
            # Draw text with black color for better visibility
            draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
            
            # Debug: Print when text is actually drawn
            print(f"    DEBUG: Drew text '{text[:20]}{'...' if len(text) > 20 else ''}' at ({text_x}, {text_y}) with font size {font_size}")
        else:
            print(f"    DEBUG: Skipped empty text at ({x1}, {y1})")
    
    def reconstruct_single_image(self, image_name: str, intermediate_dir: str = "../intermediate_outputs", 
                                output_dir: str = "../pipe_output", translate: bool = False) -> None:
        """Reconstruct a single image using linear scanning approach."""
        print(f"=== Linear Reconstruction: {image_name} ===")
        print(f"Loading from: {intermediate_dir}")
        print(f"Output to: {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Load data for specific image
            layout_data, tableformer_data, ocr_data, base_name = self.load_specific_image_data(intermediate_dir, image_name)
            
            # Calculate document dimensions
            doc_width, doc_height = self.calculate_document_dimensions(layout_data, tableformer_data)
            print(f"Document dimensions: {doc_width} x {doc_height}")
            
            # Use linear scanning approach
            self.reconstruct_with_linear_scanning(layout_data, tableformer_data, ocr_data, 
                                                doc_width, doc_height, output_dir, base_name, translate)
            
        except Exception as e:
            print(f"Error during linear reconstruction: {e}")
            raise


def main():
    """Main function for linear reconstruction."""
    parser = argparse.ArgumentParser(description="Linear Reconstruction for OCR Pipeline")
    parser.add_argument("--image", "-i", required=True, help="Image name to reconstruct")
    parser.add_argument("--intermediate-dir", "-d", default="../intermediate_outputs", 
                       help="Intermediate outputs directory")
    parser.add_argument("--output-dir", "-o", default="../pipe_output", 
                       help="Output directory")
    parser.add_argument("--translate", "-t", action="store_true", 
                       help="Enable translation")
    
    args = parser.parse_args()
    
    # Create reconstructor and run
    reconstructor = LinearReconstructor()
    reconstructor.reconstruct_single_image(
        args.image, 
        args.intermediate_dir, 
        args.output_dir, 
        args.translate
    )


if __name__ == "__main__":
    main()
