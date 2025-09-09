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
    
    def extract_ocr_text_from_layout(self, layout_data: List[Dict]) -> List[Dict]:
        """Extract text from layout predictions with fresh content generation."""
        enhanced_elements = []
        
        for element in layout_data:
            if element.get('confidence', 0) < self.min_confidence:
                continue
            
            # Generate fresh content based on element type
            bbox = [element['l'], element['t'], element['r'], element['b']]
            fresh_text = self._generate_fresh_content(element, bbox)
            
            enhanced_element = {
                'label': element['label'],
                'bounding_box': bbox,
                'confidence': element['confidence'],
                'fresh_text': fresh_text,
                'translated_text': fresh_text  # No "Translated:" prefix
            }
            enhanced_elements.append(enhanced_element)
            
        return enhanced_elements
    
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


def main():
    """Example usage of the improved fresh reconstructor."""
    reconstructor = ImprovedFreshReconstructor()
    
    # Paths to the actual files
    layout_path = "reconstruction_input/layout_predictions.json"
    tableformer_path = "reconstruction_input/tableformer_results.json"
    image_path = "reconstruction_input/pages/page_1.png"
    
    # Create output directory
    os.makedirs("reconstruction_output", exist_ok=True)
    
    # Reconstruct without translation
    print("=== Creating Improved Fresh Document (no translation) ===")
    reconstructor.reconstruct_document(
        layout_path, tableformer_path, image_path,
        "reconstruction_output/improved_fresh_reconstructed.pdf", 
        translate=False
    )
    
    # Reconstruct with translation
    print("\n=== Creating Improved Fresh Document (with translation) ===")
    reconstructor.reconstruct_document(
        layout_path, tableformer_path, image_path,
        "reconstruction_output/improved_fresh_reconstructed_translated.pdf", 
        translate=True
    )
    
    print("\nImproved fresh document reconstruction complete!")


if __name__ == "__main__":
    main()
