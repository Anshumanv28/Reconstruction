"""
Improved Fresh Document Reconstruction

This script creates a clean document reconstruction with:
- No "Translated:" prefix
- Overlap detection and prevention
- Better text positioning
- Improved element spacing
"""

import os
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageEnhance
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import re

class ImprovedFreshReconstructor:
    def __init__(self, font_path: Optional[str] = None):
        """
        Initialize the improved fresh reconstructor.
        
        Args:
            font_path: Path to font file. If None, uses default system font.
        """
        self.font_path = font_path or self._get_default_font()
        self.min_confidence = 0.5
        
        # Document styling
        self.background_color = (255, 255, 255)  # White background
        self.text_color = (0, 0, 0)  # Black text
        self.table_border_color = (0, 0, 0)  # Black borders
        self.table_bg_color = (248, 248, 248)  # Light gray table background
        self.header_bg_color = (220, 220, 220)  # Darker gray for headers
        
        # Spacing and overlap prevention
        self.min_element_spacing = 10  # Minimum spacing between elements
        self.element_padding = 5  # Padding around each element
        
    def _get_default_font(self) -> str:
        """Get default font path, trying common system fonts."""
        font_paths = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf", 
            "C:/Windows/Fonts/times.ttf",
            os.path.join(os.path.dirname(__file__), "SakalBharati.ttf")
        ]
        
        for path in font_paths:
            if os.path.exists(path):
                return path
        return None
    
    def load_docling_outputs(self, layout_predictions_path: str, 
                           tableformer_results_path: str) -> Tuple[Dict, Dict]:
        """Load and parse Docling pipeline outputs."""
        with open(layout_predictions_path, 'r', encoding='utf-8') as f:
            layout_data = json.load(f)
            
        with open(tableformer_results_path, 'r', encoding='utf-8') as f:
            tableformer_data = json.load(f)
            
        return layout_data, tableformer_data
    
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
        layout_data, tableformer_data = self.load_docling_outputs(layout_file, tableformer_file)
        
        # Load OCR data
        with open(ocr_file, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
        
        return layout_data, tableformer_data, ocr_data, base_name
    
    def detect_overlaps(self, elements: List[Dict]) -> List[Dict]:
        """Detect and resolve overlapping elements."""
        non_overlapping_elements = []
        
        for i, element in enumerate(elements):
            bbox1 = element['bounding_box']
            x1, y1, x2, y2 = bbox1
            
            # Check for overlaps with already processed elements
            has_overlap = False
            for j, existing_element in enumerate(non_overlapping_elements):
                bbox2 = existing_element['bounding_box']
                ex1, ey1, ex2, ey2 = bbox2
                
                # Check if bounding boxes overlap
                if not (x2 < ex1 or x1 > ex2 or y2 < ey1 or y1 > ey2):
                    has_overlap = True
                    break
            
            if not has_overlap:
                non_overlapping_elements.append(element)
            else:
                # Skip overlapping element or adjust position
                print(f"Skipping overlapping element: {element['label']} at {bbox1}")
        
        return non_overlapping_elements
    
    def get_document_dimensions(self, layout_data: List[Dict], 
                              tableformer_data: List[Dict]) -> Tuple[int, int]:
        """Calculate document dimensions based on all elements."""
        max_width = 0
        max_height = 0
        
        # Check layout elements
        for element in layout_data:
            if element.get('confidence', 0) >= self.min_confidence:
                max_width = max(max_width, element.get('r', 0))
                max_height = max(max_height, element.get('b', 0))
        
        # Check table elements
        for table_result in tableformer_data:
            if not table_result.get('predict_details'):
                continue
            predict_details = table_result['predict_details']
            table_bbox = predict_details.get('table_bbox', [])
            if table_bbox:
                max_width = max(max_width, table_bbox[2])
                max_height = max(max_height, table_bbox[3])
        
        # Add padding for better appearance
        return int(max_width + 100), int(max_height + 100)
    
    def extract_ocr_text_from_layout(self, layout_data: List[Dict], ocr_data: Dict) -> List[Dict]:
        """Extract text from layout predictions using actual OCR data."""
        enhanced_elements = []
        
        # Extract OCR text blocks for matching
        ocr_text_blocks = []
        
        # Check different possible OCR data structures
        text_blocks_source = None
        if 'text_blocks' in ocr_data:
            text_blocks_source = ocr_data['text_blocks']
        elif 'full_document_ocr' in ocr_data and 'text_blocks' in ocr_data['full_document_ocr']:
            text_blocks_source = ocr_data['full_document_ocr']['text_blocks']
        
        if text_blocks_source:
            for block in text_blocks_source:
                if 'bbox' in block and 'text' in block:
                    ocr_text_blocks.append({
                        'bbox': block['bbox'],
                        'text': block['text'],
                        'confidence': block.get('confidence', 1.0)
                    })
        
        print(f"Debug: Found {len(ocr_text_blocks)} OCR text blocks")
        if ocr_text_blocks:
            print(f"Debug: First few OCR blocks: {[b['text'] for b in ocr_text_blocks[:5]]}")
        
        for element in layout_data:
            if element.get('confidence', 0) < self.min_confidence:
                continue
            
            bbox = [element['l'], element['t'], element['r'], element['b']]
            
            # Find matching OCR text for this layout element
            ocr_text = self._find_matching_ocr_text(bbox, ocr_text_blocks)
            
            # Debug: Show text matching results for first few elements
            if len(enhanced_elements) < 3:
                print(f"Debug: {element['label']} bbox={bbox} -> text='{ocr_text[:30]}{'...' if len(ocr_text) > 30 else ''}'")
                # Show coordinate comparison for debugging
                if ocr_text_blocks:
                    print(f"  Layout bbox: {bbox}")
                    print(f"  First few OCR blocks: {[(b['bbox'], b['text']) for b in ocr_text_blocks[:3]]}")
            
            enhanced_element = {
                'label': element['label'],
                'bounding_box': bbox,
                'confidence': element['confidence'],
                'ocr_text': ocr_text,
                'fresh_text': ocr_text  # Use OCR text as fresh text
            }
            enhanced_elements.append(enhanced_element)
            
        return enhanced_elements
    
    def _find_matching_ocr_text(self, layout_bbox: List[float], ocr_blocks: List[Dict]) -> str:
        """Find and aggregate OCR text that matches the layout element bounding box with strict layout boundaries."""
        x1, y1, x2, y2 = layout_bbox
        
        # Conservative tolerance for coordinate differences between models
        tolerance = 5  # pixels tolerance for coordinate matching
        
        # Find all OCR words that overlap with or are contained within the layout element
        matching_words = []
        
        for ocr_block in ocr_blocks:
            ocr_bbox = ocr_block['bbox']
            ox1, oy1, ox2, oy2 = ocr_bbox
            
            # Calculate OCR word center
            ocr_center_x = (ox1 + ox2) / 2
            ocr_center_y = (oy1 + oy2) / 2
            
            # STRICT matching: OCR word must be primarily within layout bbox
            # Strategy 1: OCR word center within layout bbox (with small tolerance)
            if (x1 - tolerance <= ocr_center_x <= x2 + tolerance and 
                y1 - tolerance <= ocr_center_y <= y2 + tolerance):
                
                # Calculate overlap to ensure meaningful overlap
                overlap_x = max(0, min(x2, ox2) - max(x1, ox1))
                overlap_y = max(0, min(y2, oy2) - max(y1, oy1))
                overlap_area = overlap_x * overlap_y
                
                # Only include if there's meaningful overlap
                if overlap_area > 0:
                    layout_area = (x2 - x1) * (y2 - y1)
                    ocr_area = (ox2 - ox1) * (oy2 - oy1)
                    
                    # Require significant overlap with layout area
                    layout_overlap_ratio = overlap_area / layout_area
                    ocr_overlap_ratio = overlap_area / ocr_area
                    
                    # Include if OCR word has significant overlap with layout element
                    if layout_overlap_ratio > 0.1 or ocr_overlap_ratio > 0.5:
                        matching_words.append({
                            'text': ocr_block['text'],
                            'bbox': ocr_bbox,
                            'overlap_area': overlap_area,
                            'layout_overlap_ratio': layout_overlap_ratio,
                            'ocr_overlap_ratio': ocr_overlap_ratio,
                            'center_x': ocr_center_x,
                            'center_y': ocr_center_y,
                            'confidence': ocr_block.get('confidence', 1.0)
                        })
        
        if not matching_words:
            return ""
        
        # Sort words by layout overlap ratio first, then by position
        matching_words.sort(key=lambda w: (-w['layout_overlap_ratio'], w['center_y'], w['center_x']))
        
        # Group words into lines based on vertical proximity with conservative tolerance
        lines = []
        current_line = []
        current_y = None
        line_tolerance = 15  # pixels - conservative for line grouping
        
        for word in matching_words:
            if current_y is None or abs(word['center_y'] - current_y) <= line_tolerance:
                current_line.append(word)
                # Update current_y to be the average of the line
                current_y = sum(w['center_y'] for w in current_line) / len(current_line)
            else:
                if current_line:
                    lines.append(current_line)
                current_line = [word]
                current_y = word['center_y']
        
        if current_line:
            lines.append(current_line)
        
        # Combine lines into final text with proper spacing
        result_text = []
        for line in lines:
            # Sort words within each line by horizontal position
            line.sort(key=lambda w: w['center_x'])
            
            # Join words with appropriate spacing
            line_words = [word['text'] for word in line]
            
            # Handle special cases for better text flow
            line_text = ' '.join(line_words)
            
            # Clean up common OCR artifacts
            line_text = line_text.replace('  ', ' ')  # Remove double spaces
            line_text = line_text.replace('|', ' ')   # Replace pipe characters with spaces
            
            result_text.append(line_text.strip())
        
        final_text = '\n'.join(result_text).strip()
        
        # Debug: Show what text was found for this layout element
        if len(matching_words) > 0:
            print(f"    Found {len(matching_words)} OCR words: '{final_text[:50]}{'...' if len(final_text) > 50 else ''}'")
        
        return final_text
    
    def _generate_fresh_content(self, element: Dict, bbox: List[float]) -> str:
        """Generate fresh, realistic content based on element type and size."""
        element_type = element.get('label', 'Text')
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Skip very small elements to avoid clutter
        if width < 30 or height < 15:
            return ""
        
        # Generate content based on element type
        if element_type == 'Title':
            return "Document Title"
        elif element_type == 'Section-header':
            return "Section Header"
        elif element_type == 'Table':
            return ""  # Tables are handled separately
        elif element_type == 'Figure':
            return "[Figure]"
        elif element_type == 'Caption':
            return "Figure caption text"
        elif element_type == 'Footer':
            return "Page footer information"
        elif element_type == 'Header':
            return "Page header information"
        else:  # Text, Paragraph, etc.
            # Generate text based on size
            if width < 100:
                return "Short text"
            elif width < 200:
                return "This is a medium length text block that contains several words."
            elif width < 400:
                return "This is a longer paragraph that contains multiple sentences and provides more detailed information about the topic being discussed."
            else:
                return "This is a very long paragraph that contains multiple sentences and provides comprehensive information about the topic. It includes detailed explanations, examples, and supporting details that help the reader understand the content better."
    
    def process_tableformer_results(self, tableformer_data: List[Dict]) -> List[Dict]:
        """Process tableformer results with fresh table content."""
        processed_tables = []
        
        for table_result in tableformer_data:
            if not table_result.get('predict_details'):
                continue
                
            predict_details = table_result['predict_details']
            table_bbox = predict_details.get('table_bbox', [])
            table_cells = predict_details.get('table_cells', [])
            
            if not table_bbox or not table_cells:
                continue
                
            # Process table cells with fresh content
            processed_cells = []
            for cell in table_cells:
                cell_bbox = cell.get('bbox', [])
                if not cell_bbox:
                    continue
                
                # Generate fresh cell content
                fresh_text = self._generate_fresh_cell_content(cell)
                
                processed_cell = {
                    'bbox': cell_bbox,
                    'row_id': cell.get('row_id', 0),
                    'column_id': cell.get('column_id', 0),
                    'cell_class': cell.get('cell_class', 2),
                    'label': cell.get('label', 'fcel'),
                    'fresh_text': fresh_text,
                    'translated_text': fresh_text  # No "Translated:" prefix
                }
                processed_cells.append(processed_cell)
            
            processed_table = {
                'table_bbox': table_bbox,
                'table_cells': processed_cells,
                'num_rows': max([cell.get('row_id', 0) for cell in processed_cells], default=0) + 1,
                'num_cols': max([cell.get('column_id', 0) for cell in processed_cells], default=0) + 1
            }
            processed_tables.append(processed_table)
            
        return processed_tables
    
    def _generate_fresh_cell_content(self, cell: Dict) -> str:
        """Generate fresh content for table cells."""
        row_id = cell.get('row_id', 0)
        column_id = cell.get('column_id', 0)
        cell_class = cell.get('cell_class', 2)
        
        # Generate content based on cell position and type
        if row_id == 0:  # Header row
            if column_id == 0:
                return "Header 1"
            elif column_id == 1:
                return "Header 2"
            elif column_id == 2:
                return "Header 3"
            else:
                return f"Header {column_id + 1}"
        else:  # Data rows
            if column_id == 0:
                return f"Item {row_id}"
            elif column_id == 1:
                return f"Description {row_id}"
            elif column_id == 2:
                return f"Value {row_id}"
            else:
                return f"Data {row_id},{column_id}"
    
    def get_optimal_font_size(self, draw: ImageDraw.Draw, bbox: List[float], 
                            text: str, max_font_size: int = 50) -> Tuple[int, str]:
        """Calculate optimal font size for text to fit in bounding box."""
        x1, y1, x2, y2 = bbox
        box_width = x2 - x1 - (2 * self.element_padding)
        box_height = y2 - y1 - (2 * self.element_padding)
        
        if box_width <= 0 or box_height <= 0:
            return 8, ""
        
        min_size, max_size = 8, min(max_font_size, int(box_height * 0.8))
        
        while min_size < max_size:
            mid_size = (min_size + max_size + 1) // 2
            
            try:
                font = ImageFont.truetype(self.font_path, mid_size)
            except:
                font = ImageFont.load_default()
            
            wrapped_text = self._wrap_text(draw, text, font, box_width)
            bbox = draw.textbbox((0, 0), wrapped_text, font=font)
            text_height = bbox[3] - bbox[1]
            
            if text_height <= box_height:
                min_size = mid_size
            else:
                max_size = mid_size - 1
        
        try:
            final_font = ImageFont.truetype(self.font_path, min_size)
        except:
            final_font = ImageFont.load_default()
            
        final_wrapped = self._wrap_text(draw, text, final_font, box_width)
        
        return min_size, final_wrapped
    
    def _wrap_text(self, draw: ImageDraw.Draw, text: str, font: ImageFont, 
                  max_width: float) -> str:
        """Wrap text to fit within max_width."""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            text_width = bbox[2] - bbox[0]
            
            if text_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)
        
        if current_line:
            lines.append(' '.join(current_line))
            
        return '\n'.join(lines)
    
    def draw_text_element(self, draw: ImageDraw.Draw, bbox: List[float], 
                         text: str, font_size: int = None, 
                         bg_color: Tuple[int, int, int] = None) -> None:
        """Draw text element with proper positioning and background."""
        if not text.strip():
            return
            
        x1, y1, x2, y2 = bbox
        
        # Add padding to bounding box
        x1 += self.element_padding
        y1 += self.element_padding
        x2 -= self.element_padding
        y2 -= self.element_padding
        
        if x2 <= x1 or y2 <= y1:
            return  # Skip if bounding box is too small
        
        # Draw background if specified
        if bg_color:
            draw.rectangle([x1, y1, x2, y2], fill=bg_color)
        
        if font_size is None:
            font_size, wrapped_text = self.get_optimal_font_size(draw, bbox, text)
        else:
            try:
                font = ImageFont.truetype(self.font_path, font_size)
            except:
                font = ImageFont.load_default()
            wrapped_text = self._wrap_text(draw, text, font, x2 - x1)
        
        try:
            font = ImageFont.truetype(self.font_path, font_size)
        except:
            font = ImageFont.load_default()
        
        # Calculate text position (centered in bounding box)
        text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Center the text
        text_x = x1 + (x2 - x1 - text_width) // 2
        text_y = y1 + (y2 - y1 - text_height) // 2
        
        # Draw text
        draw.text((text_x, text_y), wrapped_text, font=font, fill=self.text_color)
    
    def _find_ocr_text_for_table(self, table_bbox: List[float], layout_elements: List[Dict]) -> str:
        """Find OCR text that matches a table's bounding box."""
        tx1, ty1, tx2, ty2 = table_bbox
        
        for element in layout_elements:
            if element['label'] == 'Table':
                bbox = element['bounding_box']
                ex1, ey1, ex2, ey2 = bbox
                
                # Check if this table element matches the table bbox
                if (abs(tx1 - ex1) < 10 and abs(ty1 - ey1) < 10 and 
                    abs(tx2 - ex2) < 10 and abs(ty2 - ey2) < 10):
                    return element.get('fresh_text', '')
        
        return ""
    
    def draw_table_systematic_approach(self, draw: ImageDraw.Draw, table_data: Dict, 
                                     ocr_text: str, ocr_data: Dict) -> None:
        """Draw table using systematic approach: TableFormer structure + intelligent OCR text mapping."""
        table_bbox = table_data['table_bbox']
        table_cells = table_data['table_cells']
        
        if not table_bbox or not table_cells:
            return
        
        x1, y1, x2, y2 = table_bbox
        
        # Draw table background
        draw.rectangle([x1, y1, x2, y2], fill=self.table_bg_color)
        
        # Draw table border
        draw.rectangle([x1, y1, x2, y2], outline=self.table_border_color, width=2)
        
        # Extract OCR text blocks for precise mapping
        ocr_text_blocks = []
        if 'full_document_ocr' in ocr_data and 'text_blocks' in ocr_data['full_document_ocr']:
            for block in ocr_data['full_document_ocr']['text_blocks']:
                if 'bbox' in block and 'text' in block:
                    ocr_text_blocks.append({
                        'bbox': block['bbox'],
                        'text': block['text'],
                        'confidence': block.get('confidence', 1.0)
                    })
        
        print(f"    Table has {len(table_cells)} cells, {len(ocr_text_blocks)} OCR blocks available")
        
        # Group cells by type for systematic mapping
        header_cells = []
        data_cells = []
        
        for cell in table_cells:
            if cell.get('label') == 'ched':  # Column header
                header_cells.append(cell)
            elif cell.get('label') == 'fcel':  # Field cell
                data_cells.append(cell)
        
        print(f"    Found {len(header_cells)} header cells, {len(data_cells)} data cells")
        
        # Draw header cells with OCR text
        for cell in header_cells:
            self._draw_cell_with_ocr_mapping(draw, cell, ocr_text_blocks, is_header=True)
        
        # Draw data cells with OCR text
        for cell in data_cells:
            self._draw_cell_with_ocr_mapping(draw, cell, ocr_text_blocks, is_header=False)
    
    def _draw_cell_with_ocr_mapping(self, draw: ImageDraw.Draw, cell: Dict, 
                                   ocr_text_blocks: List[Dict], is_header: bool = False) -> None:
        """Draw a single cell with OCR text mapping based on cell position and type."""
        cell_bbox = cell['bbox']
        cx1, cy1, cx2, cy2 = cell_bbox
        
        # Determine cell background color
        if is_header:
            bg_color = self.header_bg_color
        else:
            bg_color = self.table_bg_color
        
        # Draw cell background
        draw.rectangle(cell_bbox, fill=bg_color)
        
        # Draw cell border
        draw.rectangle(cell_bbox, outline=self.table_border_color, width=1)
        
        # Find OCR text that overlaps with this cell
        cell_text = self._find_ocr_text_for_cell(cell_bbox, ocr_text_blocks)
        
        if cell_text.strip():
            # Use smaller font for table cells
            font_size = 10 if is_header else 9
            self.draw_text_element(draw, cell_bbox, cell_text, font_size=font_size, bg_color=None)
            print(f"      Cell {cell.get('row_id', 0)},{cell.get('column_id', 0)} ({cell.get('label', 'unknown')}): '{cell_text[:20]}{'...' if len(cell_text) > 20 else ''}'")
        else:
            # Fallback to more meaningful text based on cell type and position
            row_id = cell.get('row_id', 0)
            col_id = cell.get('column_id', 0)
            
            if is_header:
                # Header fallbacks based on common table headers
                header_names = ["Time", "Topic", "Facilitator", "Description", "Status", "Notes"]
                if col_id < len(header_names):
                    fallback_text = header_names[col_id]
                else:
                    fallback_text = f"Column {col_id + 1}"
            else:
                # Data cell fallbacks based on position
                if row_id == 1:  # First data row
                    data_examples = ["", "Discussion Item", "Facilitator Name"]
                    if col_id < len(data_examples):
                        fallback_text = data_examples[col_id] if data_examples[col_id] else ""
                    else:
                        fallback_text = ""
                else:
                    # Generic data cell
                    fallback_text = ""
            
            # Only draw if there's meaningful fallback text
            if fallback_text.strip():
                font_size = 10 if is_header else 9
                self.draw_text_element(draw, cell_bbox, fallback_text, font_size=font_size, bg_color=None)
                print(f"      Cell {row_id},{col_id} ({cell.get('label', 'unknown')}): '{fallback_text}' (fallback)")
            else:
                print(f"      Cell {row_id},{col_id} ({cell.get('label', 'unknown')}): (empty - no OCR text found)")
    
    def _find_ocr_text_for_cell(self, cell_bbox: List[float], ocr_text_blocks: List[Dict]) -> str:
        """Find OCR text that best matches a specific cell's bounding box with intelligent word grouping."""
        cx1, cy1, cx2, cy2 = cell_bbox
        
        # More generous tolerance for table cells to capture related words
        tolerance = 15  # pixels tolerance for cell matching
        
        # Find OCR words that overlap with this cell
        matching_words = []
        
        for ocr_block in ocr_text_blocks:
            ocr_bbox = ocr_block['bbox']
            ox1, oy1, ox2, oy2 = ocr_bbox
            
            # Calculate OCR word center
            ocr_center_x = (ox1 + ox2) / 2
            ocr_center_y = (oy1 + oy2) / 2
            
            # Calculate overlap
            overlap_x = max(0, min(cx2, ox2) - max(cx1, ox1))
            overlap_y = max(0, min(cy2, oy2) - max(cy1, oy1))
            overlap_area = overlap_x * overlap_y
            
            # Multiple strategies for matching words to cells
            matches = False
            
            # Strategy 1: OCR word center within cell bbox (with tolerance)
            if (cx1 - tolerance <= ocr_center_x <= cx2 + tolerance and 
                cy1 - tolerance <= ocr_center_y <= cy2 + tolerance):
                matches = True
            
            # Strategy 2: OCR word has significant overlap with cell
            if not matches and overlap_area > 0:
                cell_area = (cx2 - cx1) * (cy2 - cy1)
                ocr_area = (ox2 - ox1) * (oy2 - oy1)
                
                cell_overlap_ratio = overlap_area / cell_area
                ocr_overlap_ratio = overlap_area / ocr_area
                
                # Include if significant overlap with either cell or OCR area
                if cell_overlap_ratio > 0.1 or ocr_overlap_ratio > 0.3:
                    matches = True
            
            # Strategy 3: OCR word is close to cell edges (for edge cases)
            if not matches:
                distance_to_cell = min(
                    abs(ox1 - cx1), abs(ox2 - cx2), abs(oy1 - cy1), abs(oy2 - cy2)
                )
                if distance_to_cell <= tolerance:
                    matches = True
            
            if matches:
                cell_area = (cx2 - cx1) * (cy2 - cy1)
                ocr_area = (ox2 - ox1) * (oy2 - oy1)
                cell_overlap_ratio = overlap_area / cell_area if cell_area > 0 else 0
                ocr_overlap_ratio = overlap_area / ocr_area if ocr_area > 0 else 0
                
                matching_words.append({
                    'text': ocr_block['text'],
                    'bbox': ocr_bbox,
                    'cell_overlap_ratio': cell_overlap_ratio,
                    'ocr_overlap_ratio': ocr_overlap_ratio,
                    'center_x': ocr_center_x,
                    'center_y': ocr_center_y,
                    'confidence': ocr_block.get('confidence', 1.0)
                })
        
        if not matching_words:
            return ""
        
        # Sort by cell overlap ratio first, then by position
        matching_words.sort(key=lambda w: (-w['cell_overlap_ratio'], w['center_y'], w['center_x']))
        
        # Group words by proximity and combine intelligently
        if len(matching_words) == 1:
            return matching_words[0]['text']
        
        # Group words that are close to each other (more generous for table cells)
        word_groups = []
        current_group = [matching_words[0]]
        
        for word in matching_words[1:]:
            # Check if this word is close to the current group
            last_word = current_group[-1]
            distance = ((word['center_x'] - last_word['center_x'])**2 + 
                       (word['center_y'] - last_word['center_y'])**2)**0.5
            
            # If close enough (within 50 pixels for table cells), add to current group
            if distance < 50:
                current_group.append(word)
            else:
                # Start new group
                word_groups.append(current_group)
                current_group = [word]
        
        # Add the last group
        word_groups.append(current_group)
        
        # Take the best group (highest overlap scores)
        best_group = max(word_groups, key=lambda group: 
                        sum(w['cell_overlap_ratio'] for w in group))
        
        # Sort words within the group by position
        best_group.sort(key=lambda w: (w['center_y'], w['center_x']))
        
        # Combine words intelligently
        result_words = []
        for word in best_group:
            text = word['text'].strip()
            if text and text not in result_words:  # Avoid duplicates
                result_words.append(text)
        
        final_text = ' '.join(result_words)
        
        # Clean up common OCR artifacts for table cells
        final_text = final_text.replace('  ', ' ')  # Remove double spaces
        final_text = final_text.replace('|', ' ')   # Replace pipe characters with spaces
        
        return final_text
    
    def draw_table_with_ocr_cells(self, draw: ImageDraw.Draw, table_data: Dict, 
                                 ocr_text: str) -> None:
        """Draw table with proper cell structure but using OCR text."""
        table_bbox = table_data['table_bbox']
        table_cells = table_data['table_cells']
        
        if not table_bbox or not table_cells:
            return
        
        x1, y1, x2, y2 = table_bbox
        
        # Draw table background
        draw.rectangle([x1, y1, x2, y2], fill=self.table_bg_color)
        
        # Draw table border
        draw.rectangle([x1, y1, x2, y2], outline=self.table_border_color, width=2)
        
        # Split OCR text into words for cell distribution
        ocr_words = ocr_text.strip().split()
        word_index = 0
        
        # Draw cells and distribute OCR text
        for cell in table_cells:
            cell_bbox = cell['bbox']
            
            # Determine cell background color
            row_id = cell.get('row_id', 0)
            if row_id == 0:  # Header row
                bg_color = self.header_bg_color
            else:
                bg_color = self.table_bg_color
            
            # Draw cell background
            draw.rectangle(cell_bbox, fill=bg_color)
            
            # Draw cell border
            draw.rectangle(cell_bbox, outline=self.table_border_color, width=1)
            
            # Assign OCR text to cell
            if word_index < len(ocr_words):
                # Take 1-3 words for each cell depending on cell size
                cell_width = cell_bbox[2] - cell_bbox[0]
                if cell_width > 200:  # Large cell
                    words_to_take = min(3, len(ocr_words) - word_index)
                elif cell_width > 100:  # Medium cell
                    words_to_take = min(2, len(ocr_words) - word_index)
                else:  # Small cell
                    words_to_take = min(1, len(ocr_words) - word_index)
                
                cell_text = ' '.join(ocr_words[word_index:word_index + words_to_take])
                word_index += words_to_take
                
                # Draw cell text
                if cell_text.strip():
                    self.draw_text_element(draw, cell_bbox, cell_text, bg_color=None)
    
    def draw_table_with_ocr_text(self, draw: ImageDraw.Draw, bbox: List[float], 
                                ocr_text: str) -> None:
        """Draw table with proper formatting using OCR text."""
        x1, y1, x2, y2 = bbox
        
        # Debug: Show the OCR text being used
        print(f"    Table OCR text: '{ocr_text[:100]}{'...' if len(ocr_text) > 100 else ''}'")
        
        # Draw table background
        draw.rectangle([x1, y1, x2, y2], fill=self.table_bg_color)
        
        # Draw table border
        draw.rectangle([x1, y1, x2, y2], outline=self.table_border_color, width=2)
        
        # Split OCR text into lines and create a simple table structure
        lines = ocr_text.strip().split('\n')
        if not lines:
            return
        
        # Calculate cell dimensions
        table_width = x2 - x1
        table_height = y2 - y1
        num_rows = len(lines)
        
        # Use smaller font for table content - more intelligent sizing
        if num_rows > 0:
            font_size = max(8, min(14, int(table_height / (num_rows * 1.5))))
        else:
            font_size = 10
        
        print(f"    Table dimensions: {table_width}x{table_height}, {num_rows} rows, font_size: {font_size}")
        
        # Draw each line as a table row
        for i, line in enumerate(lines):
            if not line.strip():
                continue
                
            # Calculate row position
            row_y1 = y1 + (i * table_height / num_rows)
            row_y2 = y1 + ((i + 1) * table_height / num_rows)
            
            # Alternate row background colors
            if i % 2 == 0:
                row_bg_color = self.table_bg_color
            else:
                row_bg_color = (240, 240, 240)  # Slightly darker for alternating rows
            
            # Draw row background
            draw.rectangle([x1, row_y1, x2, row_y2], fill=row_bg_color)
            
            # Draw row border
            draw.rectangle([x1, row_y1, x2, row_y2], outline=self.table_border_color, width=1)
            
            # Draw text in the row
            if line.strip():
                # Use smaller font for table text
                self.draw_text_element(draw, [x1, row_y1, x2, row_y2], line.strip(), 
                                     font_size=font_size, bg_color=None)
    
    def draw_table(self, image: Image.Image, draw: ImageDraw.Draw, 
                   table_data: Dict, translate: bool = False) -> None:
        """Draw table with proper cell formatting and fresh content."""
        table_bbox = table_data['table_bbox']
        table_cells = table_data['table_cells']
        
        if not table_bbox or not table_cells:
            return
        
        x1, y1, x2, y2 = table_bbox
        
        # Draw table background
        draw.rectangle([x1, y1, x2, y2], fill=self.table_bg_color)
        
        # Draw table border
        draw.rectangle([x1, y1, x2, y2], outline=self.table_border_color, width=2)
        
        # Draw cells and text
        for cell in table_cells:
            cell_bbox = cell['bbox']
            text = cell.get('translated_text' if translate else 'fresh_text', '')
            
            # Determine cell background color
            row_id = cell.get('row_id', 0)
            if row_id == 0:  # Header row
                bg_color = self.header_bg_color
            else:
                bg_color = self.table_bg_color
            
            # Draw cell background
            draw.rectangle(cell_bbox, fill=bg_color)
            
            # Draw cell border
            draw.rectangle(cell_bbox, outline=self.table_border_color, width=1)
            
            # Draw cell text
            if text.strip():
                self.draw_text_element(draw, cell_bbox, text, bg_color=None)
    
    def save_intermediate_data(self, layout_elements: List[Dict], tables: List[Dict], 
                             doc_width: int, doc_height: int, translate: bool = False) -> None:
        """Save intermediate data for debugging and tracking."""
        intermediate_dir = "intermediate_outputs"
        os.makedirs(intermediate_dir, exist_ok=True)
        
        # Save processed layout elements
        layout_output = {
            "document_dimensions": {
                "width": doc_width,
                "height": doc_height
            },
            "processing_info": {
                "translate_mode": translate,
                "min_confidence": self.min_confidence,
                "element_padding": self.element_padding,
                "min_element_spacing": self.min_element_spacing
            },
            "layout_elements": layout_elements,
            "total_elements": len(layout_elements)
        }
        
        layout_file = f"{intermediate_dir}/processed_layout_elements.json"
        with open(layout_file, 'w', encoding='utf-8') as f:
            json.dump(layout_output, f, indent=2, ensure_ascii=False)
        print(f"Saved processed layout elements to: {layout_file}")
        
        # Save processed table data
        tables_output = {
            "processing_info": {
                "translate_mode": translate,
                "min_confidence": self.min_confidence
            },
            "tables": tables,
            "total_tables": len(tables)
        }
        
        tables_file = f"{intermediate_dir}/processed_table_data.json"
        with open(tables_file, 'w', encoding='utf-8') as f:
            json.dump(tables_output, f, indent=2, ensure_ascii=False)
        print(f"Saved processed table data to: {tables_file}")
        
        # Save reconstruction summary
        summary = {
            "reconstruction_info": {
                "timestamp": str(pd.Timestamp.now()) if 'pd' in globals() else "N/A",
                "translate_mode": translate,
                "document_dimensions": {"width": doc_width, "height": doc_height},
                "styling": {
                    "background_color": self.background_color,
                    "text_color": self.text_color,
                    "table_bg_color": self.table_bg_color,
                    "header_bg_color": self.header_bg_color,
                    "table_border_color": self.table_border_color
                }
            },
            "element_counts": {
                "total_layout_elements": len(layout_elements),
                "total_tables": len(tables),
                "elements_by_type": {}
            },
            "files_generated": {
                "layout_elements": layout_file,
                "table_data": tables_file
            }
        }
        
        # Count elements by type
        for element in layout_elements:
            element_type = element.get('label', 'Unknown')
            summary["element_counts"]["elements_by_type"][element_type] = \
                summary["element_counts"]["elements_by_type"].get(element_type, 0) + 1
        
        summary_file = f"{intermediate_dir}/reconstruction_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved reconstruction summary to: {summary_file}")
        
        # Save detailed element breakdown
        self.save_detailed_breakdown(layout_elements, tables, intermediate_dir)

    def save_detailed_breakdown(self, layout_elements: List[Dict], tables: List[Dict], 
                              intermediate_dir: str) -> None:
        """Save detailed element-by-element breakdown for analysis."""
        
        # Create detailed layout elements breakdown
        detailed_elements = []
        for i, element in enumerate(layout_elements):
            bbox = element['bounding_box']
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            detailed_element = {
                "element_id": i + 1,
                "label": element['label'],
                "confidence": element['confidence'],
                "bounding_box": {
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "width": width, "height": height
                },
                "content": {
                    "fresh_text": element.get('fresh_text', ''),
                    "translated_text": element.get('translated_text', ''),
                    "text_length": len(element.get('fresh_text', ''))
                },
                "styling": {
                    "will_have_background": element['label'] in ['Title', 'Section-header'],
                    "background_color": (240, 240, 240) if element['label'] in ['Title', 'Section-header'] else None
                }
            }
            detailed_elements.append(detailed_element)
        
        elements_file = f"{intermediate_dir}/detailed_layout_elements.json"
        with open(elements_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_elements, f, indent=2, ensure_ascii=False)
        print(f"Saved detailed layout elements to: {elements_file}")
        
        # Create detailed table breakdown
        detailed_tables = []
        for i, table in enumerate(tables):
            table_bbox = table['table_bbox']
            x1, y1, x2, y2 = table_bbox
            width = x2 - x1
            height = y2 - y1
            
            detailed_table = {
                "table_id": i + 1,
                "table_bbox": {
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "width": width, "height": height
                },
                "structure": {
                    "num_rows": table['num_rows'],
                    "num_cols": table['num_cols'],
                    "total_cells": len(table['table_cells'])
                },
                "cells": []
            }
            
            for j, cell in enumerate(table['table_cells']):
                cell_bbox = cell['bbox']
                cx1, cy1, cx2, cy2 = cell_bbox
                cell_width = cx2 - cx1
                cell_height = cy2 - cy1
                
                detailed_cell = {
                    "cell_id": j + 1,
                    "position": {
                        "row_id": cell['row_id'],
                        "column_id": cell['column_id'],
                        "cell_class": cell['cell_class']
                    },
                    "bounding_box": {
                        "x1": cx1, "y1": cy1, "x2": cx2, "y2": cy2,
                        "width": cell_width, "height": cell_height
                    },
                    "content": {
                        "fresh_text": cell.get('fresh_text', ''),
                        "translated_text": cell.get('translated_text', ''),
                        "text_length": len(cell.get('fresh_text', ''))
                    },
                    "styling": {
                        "is_header": cell['row_id'] == 0,
                        "background_color": "header_bg_color" if cell['row_id'] == 0 else "table_bg_color"
                    }
                }
                detailed_table["cells"].append(detailed_cell)
            
            detailed_tables.append(detailed_table)
        
        tables_file = f"{intermediate_dir}/detailed_table_breakdown.json"
        with open(tables_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_tables, f, indent=2, ensure_ascii=False)
        print(f"Saved detailed table breakdown to: {tables_file}")

    def reconstruct_document(self, layout_predictions_path: str, 
                           tableformer_results_path: str, 
                           image_path: str, 
                           output_path: str, 
                           translate: bool = False) -> None:
        """Main reconstruction function that creates a fresh document."""
        print("Loading Docling outputs...")
        layout_data, tableformer_data = self.load_docling_outputs(
            layout_predictions_path, tableformer_results_path
        )
        
        # Save raw input data for reference
        intermediate_dir = "intermediate_outputs"
        os.makedirs(intermediate_dir, exist_ok=True)
        
        raw_inputs = {
            "layout_predictions": layout_data,
            "tableformer_results": tableformer_data,
            "input_files": {
                "layout_predictions_path": layout_predictions_path,
                "tableformer_results_path": tableformer_results_path,
                "image_path": image_path
            }
        }
        
        raw_inputs_file = f"{intermediate_dir}/raw_input_data.json"
        with open(raw_inputs_file, 'w', encoding='utf-8') as f:
            json.dump(raw_inputs, f, indent=2, ensure_ascii=False)
        print(f"Saved raw input data to: {raw_inputs_file}")
        
        print("Calculating document dimensions...")
        doc_width, doc_height = self.get_document_dimensions(layout_data, tableformer_data)
        print(f"Document dimensions: {doc_width} x {doc_height}")
        
        print("Processing layout elements...")
        layout_elements = self.extract_ocr_text_from_layout(layout_data)
        
        print("Detecting and resolving overlaps...")
        layout_elements = self.detect_overlaps(layout_elements)
        print(f"Processing {len(layout_elements)} non-overlapping elements")
        
        print("Processing table data...")
        tables = self.process_tableformer_results(tableformer_data)
        
        print("Saving intermediate data...")
        self.save_intermediate_data(layout_elements, tables, doc_width, doc_height, translate)
        
        print("Creating fresh document...")
        # Create a fresh, clean document
        image = Image.new('RGB', (doc_width, doc_height), self.background_color)
        draw = ImageDraw.Draw(image)
        
        print("Drawing elements...")
        
        # Draw non-table elements
        for element in layout_elements:
            if element['label'] != 'Table':
                bbox = element['bounding_box']
                text = element.get('translated_text' if translate else 'fresh_text', '')
                if text.strip():
                    # Add some styling based on element type
                    bg_color = None
                    if element['label'] in ['Title', 'Section-header']:
                        bg_color = (240, 240, 240)  # Light background for headers
                    
                    self.draw_text_element(draw, bbox, text, bg_color=bg_color)
        
        # Draw tables
        for table in tables:
            self.draw_table(image, draw, table, translate)
        
        print("Saving improved fresh reconstructed document...")
        image.save(output_path, 'PDF', resolution=100.0)
        print(f"Improved fresh document saved as: {output_path}")
    
    def reconstruct_single_image(self, image_name: str, intermediate_dir: str = "../intermediate_outputs", 
                                output_dir: str = "../pipe_output", translate: bool = False) -> None:
        """Reconstruct document for a single specific image using pipeline data."""
        print(f"=== Single Image Reconstruction: {image_name} ===")
        print(f"Loading from: {intermediate_dir}")
        print(f"Output to: {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Load data for specific image
            layout_data, tableformer_data, ocr_data, base_name = self.load_specific_image_data(intermediate_dir, image_name)
            
            # Save raw input data for reference
            intermediate_debug_dir = os.path.join(intermediate_dir, "reconstruction_debug")
            os.makedirs(intermediate_debug_dir, exist_ok=True)
            
            raw_inputs = {
                "layout_predictions": layout_data,
                "tableformer_results": tableformer_data,
                "ocr_data": ocr_data,
                "input_files": {
                    "intermediate_dir": intermediate_dir,
                    "output_dir": output_dir,
                    "image_name": image_name,
                    "base_name": base_name
                }
            }
            
            raw_inputs_file = os.path.join(intermediate_debug_dir, f"{base_name}_single_reconstruction_input.json")
            with open(raw_inputs_file, 'w', encoding='utf-8') as f:
                json.dump(raw_inputs, f, indent=2, ensure_ascii=False)
            print(f"Saved raw input data to: {raw_inputs_file}")
            
            print("Calculating document dimensions...")
            doc_width, doc_height = self.get_document_dimensions(layout_data, tableformer_data)
            print(f"Document dimensions: {doc_width} x {doc_height}")
            
            print("Processing layout elements with OCR data...")
            layout_elements = self.extract_ocr_text_from_layout(layout_data, ocr_data)
            
            # Debug: Show what text was extracted
            print("Debug - Text extraction results:")
            for i, element in enumerate(layout_elements[:5]):  # Show first 5 elements
                text = element.get('fresh_text', '')
                print(f"  Element {i+1} ({element['label']}): '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            print("Detecting and resolving overlaps...")
            layout_elements = self.detect_overlaps(layout_elements)
            print(f"Processing {len(layout_elements)} non-overlapping elements")
            
            # Debug: Show elements that will be drawn
            elements_with_text = [e for e in layout_elements if e.get('fresh_text', '').strip()]
            print(f"Elements with text to draw: {len(elements_with_text)}")
            for i, element in enumerate(elements_with_text[:3]):  # Show first 3 with text
                print(f"  Will draw {element['label']}: '{element.get('fresh_text', '')[:30]}...'")
            
            print("Processing table data...")
            tables = self.process_tableformer_results(tableformer_data)
            
            print("Creating fresh document...")
            # Create a fresh, clean document
            image = Image.new('RGB', (doc_width, doc_height), self.background_color)
            draw = ImageDraw.Draw(image)
            
            print("Drawing elements...")
            
            # Draw non-table elements
            drawn_count = 0
            for element in layout_elements:
                if element['label'] != 'Table':
                    bbox = element['bounding_box']
                    text = element.get('fresh_text', '')  # Use OCR text
                    if text.strip():
                        # Add some styling based on element type
                        bg_color = None
                        if element['label'] in ['Title', 'Section-header']:
                            bg_color = (240, 240, 240)  # Light background for headers
                        
                        print(f"  Drawing {element['label']} at {bbox}: '{text[:30]}{'...' if len(text) > 30 else ''}'")
                        self.draw_text_element(draw, bbox, text, bg_color=bg_color)
                        drawn_count += 1
                    else:
                        print(f"  Skipping {element['label']} (no text)")
            
            print(f"Drew {drawn_count} non-table elements")
            
            # Draw tables using systematic approach: TableFormer structure + OCR text mapping
            for table in tables:
                # Find matching OCR text for this table
                table_bbox = table['table_bbox']
                matching_ocr_text = self._find_ocr_text_for_table(table_bbox, layout_elements)
                if matching_ocr_text:
                    print(f"  Drawing Table with systematic OCR mapping: '{matching_ocr_text[:30]}{'...' if len(matching_ocr_text) > 30 else ''}'")
                    self.draw_table_systematic_approach(draw, table, matching_ocr_text, ocr_data)
                else:
                    print(f"  Drawing Table with default structure")
                    self.draw_table(image, draw, table, translate)
            
            # Generate output filename
            suffix = "_translated" if translate else ""
            output_filename = f"{base_name}_single_reconstructed{suffix}.pdf"
            output_path = os.path.join(output_dir, output_filename)
            
            print(f"Saving single image reconstructed document...")
            image.save(output_path, 'PDF', resolution=100.0)
            print(f"Single image reconstructed document saved as: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"Error during single image reconstruction: {e}")
            raise


def main():
    """Main function for single image reconstruction."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Single Image Document Reconstruction")
    parser.add_argument("--image", default="1zM9MmA6dM2_2dbMHiJKd_m-FRdPpNTR3lJUT_P1QuiE.png",
                       help="Image filename to reconstruct (default: 1zM9MmA6dM2_2dbMHiJKd_m-FRdPpNTR3lJUT_P1QuiE.png)")
    parser.add_argument("--intermediate-dir", default="../intermediate_outputs",
                       help="Intermediate outputs directory")
    parser.add_argument("--output-dir", default="../pipe_output",
                       help="Output directory")
    parser.add_argument("--translate", action="store_true",
                       help="Enable translation mode")
    
    args = parser.parse_args()
    
    reconstructor = ImprovedFreshReconstructor()
    
    try:
        # Reconstruct without translation
        print("=== Creating Single Image Document (no translation) ===")
        reconstructor.reconstruct_single_image(
            args.image, args.intermediate_dir, args.output_dir, translate=False
        )
        
        # Reconstruct with translation
        print("\n=== Creating Single Image Document (with translation) ===")
        reconstructor.reconstruct_single_image(
            args.image, args.intermediate_dir, args.output_dir, translate=True
        )
        
        print(f"\nSingle image reconstruction complete for: {args.image}")
        
    except Exception as e:
        print(f"Single image reconstruction failed: {e}")
        print("Make sure you have run the docling batch pipeline and tesseract OCR first to generate intermediate outputs.")
        print(f"Looking for files with base name: {args.image.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')}")


if __name__ == "__main__":
    main()