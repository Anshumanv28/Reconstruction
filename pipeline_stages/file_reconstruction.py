#!/usr/bin/env python3
"""
Stage 4: Complete File Reconstruction

This module handles:
- Final file reconstruction combining all previous stages
- HTML, Markdown, and JSON output generation
- Creates visualization output (PNG)
- Creates coordinate JSON with structured data
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

class FileReconstructor:
    """Handles complete file reconstruction combining all pipeline stages"""
    
    def __init__(self):
        self.stage_name = "file_reconstruction"
    
    def process_image(self, input_file: str, output_prefix: str, intermediate_dir: Path) -> Dict:
        """Process a single image for complete file reconstruction"""
        print(f"🔍 Processing: {input_file}")
        
        try:
            # Load all previous stage data
            stage1_data = self._load_stage1_data(output_prefix, intermediate_dir)
            stage2_data = self._load_stage2_data(output_prefix, intermediate_dir)
            stage3_data = self._load_stage3_data(output_prefix, intermediate_dir)
            
            # Run file reconstruction
            reconstruction_result = self._run_file_reconstruction(stage1_data, stage2_data, stage3_data)
            
            # Create visualization
            visualization_path = self._create_visualization(
                input_file, output_prefix, intermediate_dir, reconstruction_result
            )
            
            # Create coordinate JSON
            coordinates_path = self._create_coordinates_json(
                output_prefix, intermediate_dir, reconstruction_result
            )
            
            # Generate output files
            output_files = self._generate_output_files(
                output_prefix, intermediate_dir, reconstruction_result
            )
            
            return {
                "success": True,
                "output_formats": list(output_files.keys()),
                "visualization_path": str(visualization_path),
                "coordinates_path": str(coordinates_path),
                "output_files": output_files,
                "reconstruction_result": reconstruction_result
            }
            
        except Exception as e:
            print(f"❌ Error in file reconstruction: {e}")
            return {"success": False, "error": str(e)}
    
    def _load_stage1_data(self, output_prefix: str, intermediate_dir: Path) -> Dict:
        """Load Stage 1 data (layout and table detection)"""
        stage1_dir = intermediate_dir / "stage1_layout_table" / "coordinates"
        stage1_file = stage1_dir / f"{output_prefix}_stage1_coordinates.json"
        
        if stage1_file.exists():
            with open(stage1_file, 'r') as f:
                return json.load(f)
        else:
            print(f"    ⚠️  Stage 1 data not found: {stage1_file}")
            return {"layout_elements": [], "tables": []}
    
    def _load_stage2_data(self, output_prefix: str, intermediate_dir: Path) -> Dict:
        """Load Stage 2 data (OCR extraction)"""
        stage2_dir = intermediate_dir / "stage2_ocr_extraction" / "coordinates"
        stage2_file = stage2_dir / f"{output_prefix}_stage2_coordinates.json"
        
        if stage2_file.exists():
            with open(stage2_file, 'r') as f:
                return json.load(f)
        else:
            print(f"    ⚠️  Stage 2 data not found: {stage2_file}")
            return {"text_blocks": []}
    
    def _load_stage3_data(self, output_prefix: str, intermediate_dir: Path) -> Dict:
        """Load Stage 3 data (grid reconstruction)"""
        stage3_dir = intermediate_dir / "stage3_grid_reconstruction" / "coordinates"
        stage3_file = stage3_dir / f"{output_prefix}_stage3_coordinates.json"
        
        if stage3_file.exists():
            with open(stage3_file, 'r') as f:
                return json.load(f)
        else:
            print(f"    ⚠️  Stage 3 data not found: {stage3_file}")
            return {"tables": []}
    
    def _run_file_reconstruction(self, stage1_data: Dict, stage2_data: Dict, stage3_data: Dict) -> Dict:
        """Run complete file reconstruction"""
        print("  📄 Running complete file reconstruction...")
        
        # Extract data from all stages
        layout_elements = stage1_data.get("layout_elements", [])
        tables = stage3_data.get("tables", [])
        text_blocks = stage2_data.get("text_blocks", [])
        
        # Create document structure
        document_structure = self._create_document_structure(layout_elements, tables, text_blocks)
        
        # Generate content for each format
        html_content = self._generate_html_content(document_structure)
        markdown_content = self._generate_markdown_content(document_structure)
        json_content = self._generate_json_content(document_structure)
        
        print(f"    ✅ File reconstruction completed")
        print(f"       Layout elements: {len(layout_elements)}")
        print(f"       Tables: {len(tables)}")
        print(f"       Text blocks: {len(text_blocks)}")
        
        return {
            "document_structure": document_structure,
            "html_content": html_content,
            "markdown_content": markdown_content,
            "json_content": json_content,
            "processing_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_layout_elements": len(layout_elements),
                "total_tables": len(tables),
                "total_text_blocks": len(text_blocks),
                "total_cells": sum(len(table.get("grid_cells", [])) for table in tables)
            }
        }
    
    def _create_document_structure(self, layout_elements: List[Dict], tables: List[Dict], text_blocks: List[Dict]) -> Dict:
        """Create structured document representation"""
        document = {
            "metadata": {
                "creation_timestamp": datetime.now().isoformat(),
                "total_elements": len(layout_elements) + len(tables),
                "total_text_blocks": len(text_blocks)
            },
            "layout_elements": layout_elements,
            "tables": tables,
            "text_blocks": text_blocks,
            "content_sections": []
        }
        
        # Organize content by sections
        sections = self._organize_content_by_sections(layout_elements, tables, text_blocks)
        document["content_sections"] = sections
        
        return document
    
    def _organize_content_by_sections(self, layout_elements: List[Dict], tables: List[Dict], text_blocks: List[Dict]) -> List[Dict]:
        """Organize content into logical sections"""
        sections = []
        
        # Group elements by position (top to bottom)
        all_elements = []
        
        # Add layout elements
        for element in layout_elements:
            if 'bbox' in element:
                x1, y1, x2, y2 = element['bbox']
            elif all(key in element for key in ['l', 't', 'r', 'b']):
                x1, y1, x2, y2 = element['l'], element['t'], element['r'], element['b']
            else:
                continue
            
            all_elements.append({
                'type': 'layout',
                'element': element,
                'y_position': y1
            })
        
        # Add tables
        for table in tables:
            if 'table_bbox' in table:
                x1, y1, x2, y2 = table['table_bbox']
            elif 'bbox' in table:
                x1, y1, x2, y2 = table['bbox']
            else:
                continue
            
            all_elements.append({
                'type': 'table',
                'element': table,
                'y_position': y1
            })
        
        # Sort by y-position
        all_elements.sort(key=lambda x: x['y_position'])
        
        # Create sections
        current_section = {
            'section_id': 0,
            'elements': [],
            'y_start': 0,
            'y_end': 0
        }
        
        for element_data in all_elements:
            if not current_section['elements']:
                current_section['y_start'] = element_data['y_position']
            
            current_section['elements'].append(element_data)
            current_section['y_end'] = element_data['y_position']
        
        if current_section['elements']:
            sections.append(current_section)
        
        return sections
    
    def _generate_html_content(self, document_structure: Dict) -> str:
        """Generate HTML content"""
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "  <title>Document Reconstruction</title>",
            "  <style>",
            "    body { font-family: Arial, sans-serif; margin: 20px; }",
            "    table { border-collapse: collapse; margin: 10px 0; }",
            "    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "    th { background-color: #f2f2f2; }",
            "    .section { margin: 20px 0; }",
            "    .layout-element { margin: 10px 0; padding: 5px; border-left: 3px solid #ccc; }",
            "  </style>",
            "</head>",
            "<body>",
            "  <h1>Document Reconstruction</h1>",
            f"  <p>Generated on: {document_structure['metadata']['creation_timestamp']}</p>",
            f"  <p>Total elements: {document_structure['metadata']['total_elements']}</p>",
            f"  <p>Total text blocks: {document_structure['metadata']['total_text_blocks']}</p>",
            ""
        ]
        
        # Add content sections
        for section in document_structure['content_sections']:
            html_parts.append(f"  <div class='section'>")
            html_parts.append(f"    <h2>Section {section['section_id'] + 1}</h2>")
            
            for element_data in section['elements']:
                if element_data['type'] == 'table':
                    html_parts.extend(self._generate_table_html(element_data['element']))
                elif element_data['type'] == 'layout':
                    html_parts.extend(self._generate_layout_element_html(element_data['element']))
            
            html_parts.append(f"  </div>")
        
        html_parts.extend([
            "</body>",
            "</html>"
        ])
        
        return "\n".join(html_parts)
    
    def _generate_table_html(self, table: Dict) -> List[str]:
        """Generate HTML for a table"""
        html_parts = []
        
        grid_cells = table.get("grid_cells", [])
        cell_content = table.get("cell_content", {})
        
        if not grid_cells:
            return html_parts
        
        # Determine table dimensions
        max_row = max(cell['row'] for cell in grid_cells) if grid_cells else 0
        max_col = max(cell['col'] for cell in grid_cells) if grid_cells else 0
        
        html_parts.append("    <table>")
        
        for row in range(max_row + 1):
            html_parts.append("      <tr>")
            for col in range(max_col + 1):
                # Find cell for this row/col
                cell = None
                for grid_cell in grid_cells:
                    if grid_cell['row'] == row and grid_cell['col'] == col:
                        cell = grid_cell
                        break
                
                if cell:
                    cell_id = cell['cell_id']
                    # Try both string and integer keys for cell_id
                    text = ""
                    if str(cell_id) in cell_content:
                        text = cell_content[str(cell_id)].get('text', '')
                    elif cell_id in cell_content:
                        text = cell_content[cell_id].get('text', '')
                    
                    if not text.strip():
                        text = f"R{row}C{col}"
                else:
                    text = ""
                
                html_parts.append(f"        <td>{text}</td>")
            
            html_parts.append("      </tr>")
        
        html_parts.append("    </table>")
        
        return html_parts
    
    def _generate_layout_element_html(self, element: Dict) -> List[str]:
        """Generate HTML for a layout element"""
        label = element.get('label', 'unknown')
        html_parts = [
            f"    <div class='layout-element'>",
            f"      <strong>{label.title()}:</strong>",
            f"    </div>"
        ]
        return html_parts
    
    def _generate_markdown_content(self, document_structure: Dict) -> str:
        """Generate Markdown content"""
        md_parts = [
            "# Document Reconstruction",
            "",
            f"**Generated on:** {document_structure['metadata']['creation_timestamp']}",
            f"**Total elements:** {document_structure['metadata']['total_elements']}",
            f"**Total text blocks:** {document_structure['metadata']['total_text_blocks']}",
            ""
        ]
        
        # Add content sections
        for section in document_structure['content_sections']:
            md_parts.append(f"## Section {section['section_id'] + 1}")
            md_parts.append("")
            
            for element_data in section['elements']:
                if element_data['type'] == 'table':
                    md_parts.extend(self._generate_table_markdown(element_data['element']))
                elif element_data['type'] == 'layout':
                    md_parts.extend(self._generate_layout_element_markdown(element_data['element']))
            
            md_parts.append("")
        
        return "\n".join(md_parts)
    
    def _generate_table_markdown(self, table: Dict) -> List[str]:
        """Generate Markdown for a table"""
        md_parts = []
        
        grid_cells = table.get("grid_cells", [])
        cell_content = table.get("cell_content", {})
        
        if not grid_cells:
            return md_parts
        
        # Determine table dimensions
        max_row = max(cell['row'] for cell in grid_cells) if grid_cells else 0
        max_col = max(cell['col'] for cell in grid_cells) if grid_cells else 0
        
        # Create header row
        header = "| " + " | ".join([f"Column {i+1}" for i in range(max_col + 1)]) + " |"
        separator = "| " + " | ".join(["---" for _ in range(max_col + 1)]) + " |"
        
        md_parts.append(header)
        md_parts.append(separator)
        
        # Create data rows
        for row in range(max_row + 1):
            row_data = []
            for col in range(max_col + 1):
                # Find cell for this row/col
                cell = None
                for grid_cell in grid_cells:
                    if grid_cell['row'] == row and grid_cell['col'] == col:
                        cell = grid_cell
                        break
                
                if cell:
                    cell_id = cell['cell_id']
                    # Try both string and integer keys for cell_id
                    text = ""
                    if str(cell_id) in cell_content:
                        text = cell_content[str(cell_id)].get('text', '')
                    elif cell_id in cell_content:
                        text = cell_content[cell_id].get('text', '')
                    
                    if not text.strip():
                        text = f"R{row}C{col}"
                else:
                    text = ""
                
                # Escape pipe characters
                text = text.replace('|', '\\|')
                row_data.append(text)
            
            md_parts.append("| " + " | ".join(row_data) + " |")
        
        md_parts.append("")
        
        return md_parts
    
    def _generate_layout_element_markdown(self, element: Dict) -> List[str]:
        """Generate Markdown for a layout element"""
        label = element.get('label', 'unknown')
        md_parts = [
            f"**{label.title()}:**",
            ""
        ]
        return md_parts
    
    def _generate_json_content(self, document_structure: Dict) -> Dict:
        """Generate JSON content"""
        return document_structure
    
    def _generate_output_files(self, output_prefix: str, intermediate_dir: Path, reconstruction_result: Dict) -> Dict:
        """Generate output files in various formats"""
        print("  📁 Generating output files...")
        
        # Create output directory
        stage_dir = intermediate_dir / "stage4_file_reconstruction"
        output_dir = stage_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_files = {}
        
        # Generate HTML file
        html_file = output_dir / f"{output_prefix}_reconstruction.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(reconstruction_result['html_content'])
        output_files['html'] = str(html_file)
        
        # Generate Markdown file
        md_file = output_dir / f"{output_prefix}_reconstruction.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(reconstruction_result['markdown_content'])
        output_files['markdown'] = str(md_file)
        
        # Generate JSON file
        json_file = output_dir / f"{output_prefix}_reconstruction.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(reconstruction_result['json_content'], f, indent=2, ensure_ascii=False)
        output_files['json'] = str(json_file)
        
        print(f"    ✅ Generated {len(output_files)} output files")
        
        return output_files
    
    def _create_visualization(self, input_file: str, output_prefix: str, intermediate_dir: Path, 
                            reconstruction_result: Dict) -> Path:
        """Create visualization of file reconstruction results"""
        print("  🎨 Creating file reconstruction visualization...")
        
        # Create output paths
        stage_dir = intermediate_dir / "stage4_file_reconstruction"
        viz_dir = stage_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = viz_dir / f"{output_prefix}_file_reconstruction.png"
        
        try:
            # Load original image
            original_img = Image.open(input_file)
            img_width, img_height = original_img.size
            
            # Create visualization canvas
            canvas = Image.new('RGB', (img_width, img_height), 'white')
            canvas.paste(original_img)
            draw = ImageDraw.Draw(canvas)
            
            # Try to load font
            try:
                font = ImageFont.truetype("arial.ttf", 20)
                small_font = ImageFont.truetype("arial.ttf", 18)
                print(f"    📝 Using Arial font: 20pt")
            except:
                try:
                    # Try common Windows fonts
                    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
                    small_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
                    print(f"    📝 Using Windows Arial font: 20pt")
                except:
                    # Use default font but make it larger
                    font = ImageFont.load_default()
                    small_font = ImageFont.load_default()
                    print(f"    📝 Using default font (may be small)")
            
            # Draw final reconstruction
            document_structure = reconstruction_result.get("document_structure", {})
            self._draw_final_reconstruction(draw, document_structure, font, small_font)
            
            # Save visualization
            canvas.save(output_path)
            print(f"    ✅ Visualization saved: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"    ❌ Error creating visualization: {e}")
            # Create a simple placeholder image
            placeholder = Image.new('RGB', (800, 600), 'lightgray')
            placeholder.save(output_path)
            return output_path
    
    def _draw_final_reconstruction(self, draw: ImageDraw.Draw, document_structure: Dict, font, small_font):
        """Draw the final reconstruction"""
        # Draw layout elements
        layout_elements = document_structure.get("layout_elements", [])
        for element in layout_elements:
            self._draw_layout_element_final(draw, element, font)
        
        # Draw tables
        tables = document_structure.get("tables", [])
        for table in tables:
            self._draw_table_final(draw, table, font)
        
        # Draw ALL OCR text blocks (like integrated_visualization.py does)
        text_blocks = document_structure.get("text_blocks", [])
        self._draw_all_ocr_text(draw, text_blocks, font)
    
    def _draw_layout_element_final(self, draw: ImageDraw.Draw, element: Dict, font):
        """Draw layout element in final reconstruction"""
        # Get bounding box
        if 'bbox' in element:
            x1, y1, x2, y2 = element['bbox']
        elif all(key in element for key in ['l', 't', 'r', 'b']):
            x1, y1, x2, y2 = element['l'], element['t'], element['r'], element['b']
        else:
            return
        
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = element.get('label', 'unknown')
        
        # Draw with final styling
        draw.rectangle([x1, y1, x2, y2], outline=(0, 128, 0), width=2)
        draw.text((x1+5, y1+5), f"FINAL: {label}", fill=(0, 128, 0), font=font)
    
    def _draw_table_final(self, draw: ImageDraw.Draw, table: Dict, font):
        """Draw table in final reconstruction"""
        grid_cells = table.get("grid_cells", [])
        cell_content = table.get("cell_content", {})
        
        # Draw final table with content
        for cell in grid_cells:
            bbox = cell['absolute_bbox']
            x1, y1, x2, y2 = bbox
            
            # Draw cell with final styling
            draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 255), width=2, fill=(240, 248, 255))
            
            # Draw content
            cell_id = cell['cell_id']
            text = ""
            if str(cell_id) in cell_content:
                text = cell_content[str(cell_id)].get('text', '')
            elif cell_id in cell_content:
                text = cell_content[cell_id].get('text', '')
            
            if text.strip():
                # Truncate text if too long (allow more characters)
                display_text = text[:25] + "..." if len(text) > 25 else text
                # Use black text for better visibility
                draw.text((x1+5, y1+5), display_text, fill=(0, 0, 0), font=font)
    
    def _draw_all_ocr_text(self, draw: ImageDraw.Draw, text_blocks: List[Dict], font):
        """Draw all OCR text blocks (like integrated_visualization.py does)"""
        if not text_blocks:
            return
        
        print(f"    📝 Drawing {len(text_blocks)} OCR text blocks...")
        
        # Sort OCR blocks by position (top to bottom, left to right)
        sorted_text_blocks = sorted(text_blocks, key=lambda x: (x['bbox'][1], x['bbox'][0]))
        
        # Process each OCR block
        for i, text_block in enumerate(sorted_text_blocks):
            self._draw_single_ocr_text_final(draw, text_block, font, i)
    
    def _draw_single_ocr_text_final(self, draw: ImageDraw.Draw, text_block: Dict, font, block_index: int):
        """Draw a single OCR text block in final reconstruction"""
        x1, y1, x2, y2 = text_block['bbox']
        text = text_block.get('text', '').strip()
        
        if not text:
            return
        
        # Convert to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
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
        
        # Draw text with a subtle background for visibility
        # Draw a light background rectangle
        draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 200), outline=(200, 200, 200), width=1)
        
        # Draw the text
        draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
        
        # Debug output for first few blocks
        if block_index < 10 or block_index % 50 == 0:
            print(f"      OCR {block_index + 1}: '{text[:30]}...' at ({x1}, {y1}) to ({x2}, {y2})")
    
    def _create_coordinates_json(self, output_prefix: str, intermediate_dir: Path, 
                               reconstruction_result: Dict) -> Path:
        """Create structured coordinate JSON"""
        print("  📄 Creating coordinates JSON...")
        
        # Create output paths
        stage_dir = intermediate_dir / "stage4_file_reconstruction"
        coordinates_dir = stage_dir / "coordinates"
        coordinates_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = coordinates_dir / f"{output_prefix}_stage4_coordinates.json"
        
        # Create structured data
        coordinates_data = {
            "stage": 4,
            "stage_name": "file_reconstruction",
            "processing_timestamp": reconstruction_result.get("processing_timestamp", datetime.now().isoformat()),
            "document_structure": reconstruction_result.get("document_structure", {}),
            "summary": reconstruction_result.get("summary", {}),
            "output_files": reconstruction_result.get("output_files", {})
        }
        
        # Save coordinates JSON
        with open(output_path, 'w') as f:
            json.dump(coordinates_data, f, indent=2)
        
        print(f"    ✅ Coordinates JSON saved: {output_path}")
        return output_path

def main():
    """Main function for testing Stage 4"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage 4: Complete File Reconstruction")
    parser.add_argument("--input", "-i", required=True, help="Input image file")
    parser.add_argument("--output-prefix", "-o", help="Output prefix (default: input filename)")
    parser.add_argument("--intermediate-dir", default="intermediate_outputs", help="Intermediate outputs directory")
    
    args = parser.parse_args()
    
    # Set default output prefix
    if not args.output_prefix:
        args.output_prefix = Path(args.input).stem
    
    # Initialize reconstructor
    reconstructor = FileReconstructor()
    
    # Process image
    result = reconstructor.process_image(args.input, args.output_prefix, Path(args.intermediate_dir))
    
    # Print results
    if result.get("success", False):
        print(f"\n✅ Stage 4 completed successfully!")
        print(f"   Output formats: {result.get('output_formats', [])}")
        print(f"   Visualization: {result.get('visualization_path', 'N/A')}")
        print(f"   Coordinates: {result.get('coordinates_path', 'N/A')}")
        print(f"   Output files: {list(result.get('output_files', {}).keys())}")
    else:
        print(f"\n❌ Stage 4 failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()
