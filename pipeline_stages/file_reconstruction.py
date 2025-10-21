#!/usr/bin/env python3
"""
Stage 4: Complete File Reconstruction

This module combines:
- Grid structure from Stage 3 (table cells with row/col positions)
- OCR text from Stage 2 (actual text content)
- Layout from Stage 1 (document structure)

Output: HTML, Markdown, JSON files with fully reconstructed tables
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

class FileReconstructor:
    """Combines OCR text with grid structure to reconstruct documents"""
    
    def __init__(self):
        self.stage_name = "file_reconstruction"
        print("🔧 Initialized FileReconstructor")
    
    def process_image(self, input_file: str, output_prefix: str, intermediate_dir: Path) -> Dict:
        """Main reconstruction process"""
        print(f"📄 Processing: {input_file}")
        
        try:
            # Step 1: Load data from all previous stages
            print("  📥 Loading Stage 1 (Layout)...")
            stage1_data = self._load_stage1_data(output_prefix, intermediate_dir)
            
            print("  📥 Loading Stage 2 (OCR)...")
            stage2_data = self._load_stage2_data(output_prefix, intermediate_dir)
            
            print("  📥 Loading Stage 3 (Grid Structure)...")
            stage3_data = self._load_stage3_data(output_prefix, intermediate_dir)
            
            # Step 2: Build complete document structure
            print("  🔗 Building complete document structure...")
            document_structure = self._build_document_structure(stage1_data, stage2_data, stage3_data)
            
            # Step 3: Generate outputs
            print("  📝 Generating HTML...")
            html_content = self._generate_html(document_structure)
            
            print("  📝 Generating Markdown...")
            markdown_content = self._generate_markdown(document_structure)
            
            print("  📝 Generating JSON...")
            json_content = self._generate_json(document_structure)
            
            # Step 4: Save output files
            output_files = self._save_outputs(
                output_prefix, intermediate_dir,
                html_content, markdown_content, json_content
            )
            
            # Step 5: Create visualization
            print("  🎨 Creating visualization...")
            viz_path = self._create_visualization(
                input_file, output_prefix, intermediate_dir, document_structure
            )
            
            # Step 6: Save coordinates JSON
            coords_path = self._save_coordinates(
                output_prefix, intermediate_dir, document_structure
            )
            
            print("  ✅ Document reconstruction complete!")
            
            return {
                "success": True,
                "document_structure": document_structure,
                "output_files": output_files,
                "visualization_path": str(viz_path),
                "coordinates_path": str(coords_path),
                "summary": {
                    "total_elements": len(document_structure.get("elements", [])),
                    "tables": document_structure.get("metadata", {}).get("total_tables", 0),
                    "text_blocks": document_structure.get("metadata", {}).get("total_text_blocks", 0)
                }
            }
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    # ==================== DATA LOADING ====================
    
    def _load_stage1_data(self, output_prefix: str, intermediate_dir: Path) -> Dict:
        """Load Stage 1: Layout and table detection"""
        stage1_dir = intermediate_dir / "stage1_layout_table" / "coordinates"
        stage1_file = stage1_dir / f"{output_prefix}_stage1_coordinates.json"
        
        if stage1_file.exists():
            with open(stage1_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        print(f"    ⚠️  Stage 1 data not found")
        return {"layout_elements": [], "tables": []}
    
    def _load_stage2_data(self, output_prefix: str, intermediate_dir: Path) -> Dict:
        """Load Stage 2: OCR text extraction"""
        stage2_dir = intermediate_dir / "stage2_ocr_extraction" / "coordinates"
        stage2_file = stage2_dir / f"{output_prefix}_stage2_coordinates.json"
        
        if stage2_file.exists():
            with open(stage2_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        print(f"    ⚠️  Stage 2 data not found")
        return {"text_blocks": []}
    
    def _load_stage3_data(self, output_prefix: str, intermediate_dir: Path) -> Dict:
        """Load Stage 3: Grid structure"""
        stage3_dir = intermediate_dir / "stage3_grid_reconstruction" / "coordinates"
        stage3_file = stage3_dir / f"{output_prefix}_stage3_coordinates.json"
        
        if stage3_file.exists():
            with open(stage3_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        print(f"    ⚠️  Stage 3 data not found")
        return {"tables": []}
    
    # ==================== DOCUMENT STRUCTURE ====================
    
    def _build_document_structure(self, stage1_data: Dict, stage2_data: Dict, stage3_data: Dict) -> Dict:
        """Build complete document structure with tables AND text"""
        
        # Get all components
        layout_elements = stage1_data.get("layout_elements", [])
        all_text_blocks = stage2_data.get("text_blocks", [])
        tables_data = stage3_data.get("tables", [])
        
        print(f"    📊 Found {len(layout_elements)} layout elements, {len(all_text_blocks)} OCR blocks, {len(tables_data)} tables")
        
        # Step 1: Categorize text blocks by granularity
        table_text_blocks = [b for b in all_text_blocks if b.get("region_type") == "table"]
        paragraph_blocks = [b for b in all_text_blocks if b.get("granularity") == "paragraph"]
        other_blocks = [b for b in all_text_blocks if b.get("granularity") not in ["word", "paragraph"]]
        
        print(f"    📝 OCR breakdown: {len(table_text_blocks)} table words, {len(paragraph_blocks)} paragraphs, {len(other_blocks)} other")
        
        # Step 2: Enrich tables with word-level OCR
        enriched_tables = []
        for table in tables_data:
            enriched_table = self._enrich_table_with_word_ocr(table, table_text_blocks)
            enriched_tables.append(enriched_table)
        
        # Step 3: Filter out text blocks that overlap with tables
        # Get all table bounding boxes
        table_bboxes = [t.get("table_bbox", []) for t in enriched_tables if t.get("table_bbox")]
        
        # Filter paragraphs that don't overlap with tables
        non_overlapping_paragraphs = []
        for para_block in paragraph_blocks:
            para_bbox = para_block.get("bbox", [])
            if not self._overlaps_any_table(para_bbox, table_bboxes):
                non_overlapping_paragraphs.append(para_block)
        
        # Filter other blocks that don't overlap with tables
        non_overlapping_others = []
        for other_block in other_blocks:
            other_bbox = other_block.get("bbox", [])
            if not self._overlaps_any_table(other_bbox, table_bboxes):
                non_overlapping_others.append(other_block)
        
        print(f"    🔍 Filtered out overlaps: {len(paragraph_blocks) - len(non_overlapping_paragraphs)} paragraphs, {len(other_blocks) - len(non_overlapping_others)} other blocks")
        
        # Step 4: Combine all elements and sort by position
        all_elements = []
        
        # Add tables
        for table in enriched_tables:
            table_bbox = table.get("table_bbox", [0, 0, 0, 0])
            all_elements.append({
                "type": "table",
                "y_position": table_bbox[1] if len(table_bbox) > 1 else 0,
                "x_position": table_bbox[0] if len(table_bbox) > 0 else 0,
                "content": table
            })
        
        # Add non-overlapping paragraph blocks
        for para_block in non_overlapping_paragraphs:
            para_bbox = para_block.get("bbox", [0, 0, 0, 0])
            all_elements.append({
                "type": "paragraph",
                "y_position": para_bbox[1] if len(para_bbox) > 1 else 0,
                "x_position": para_bbox[0] if len(para_bbox) > 0 else 0,
                "content": para_block
            })
        
        # Add non-overlapping other blocks
        for other_block in non_overlapping_others:
            other_bbox = other_block.get("bbox", [0, 0, 0, 0])
            region_type = other_block.get("region_type", "text")
            all_elements.append({
                "type": region_type,
                "y_position": other_bbox[1] if len(other_bbox) > 1 else 0,
                "x_position": other_bbox[0] if len(other_bbox) > 0 else 0,
                "content": other_block
            })
        
        # Sort by vertical position (top to bottom), then horizontal
        all_elements.sort(key=lambda x: (x["y_position"], x["x_position"]))
        
        print(f"    📄 Built document with {len(all_elements)} elements")
        
        return {
            "metadata": {
                "generated": datetime.now().isoformat(),
                "total_elements": len(all_elements),
                "total_tables": len(enriched_tables),
                "total_paragraphs": len(non_overlapping_paragraphs),
                "total_other": len(non_overlapping_others)
            },
            "elements": all_elements,
            "tables": enriched_tables,
            "paragraphs": non_overlapping_paragraphs,
            "other_blocks": non_overlapping_others
        }
    
    def _overlaps_any_table(self, bbox: List[float], table_bboxes: List[List[float]]) -> bool:
        """Check if a bounding box overlaps with any table"""
        for table_bbox in table_bboxes:
            if len(bbox) < 4 or len(table_bbox) < 4:
                continue
            
            # Check for ANY overlap (even 1% means it's inside a table region)
            overlap = self._calculate_overlap(bbox, table_bbox)
            if overlap > 0.01:  # Even 1% overlap means it's part of table
                return True
            
            # Also check if text is completely contained in table
            if self._is_contained(bbox, table_bbox):
                return True
        
        return False
    
    def _is_contained(self, inner_bbox: List[float], outer_bbox: List[float]) -> bool:
        """Check if inner bbox is completely contained within outer bbox"""
        if len(inner_bbox) < 4 or len(outer_bbox) < 4:
            return False
        
        x1_i, y1_i, x2_i, y2_i = inner_bbox
        x1_o, y1_o, x2_o, y2_o = outer_bbox
        
        return (x1_i >= x1_o and y1_i >= y1_o and 
                x2_i <= x2_o and y2_i <= y2_o)
    
    # ==================== OCR MATCHING ====================
    
    def _enrich_table_with_word_ocr(self, table: Dict, table_text_blocks: List[Dict]) -> Dict:
        """Use text from Stage 3 cells (already populated) or match word-level OCR if needed"""
        grid_cells = table.get("grid_cells", [])
        enriched_cells = []
        cells_with_text = 0
        
        for cell in grid_cells:
            # Check if cell already has text from Stage 3
            existing_text = cell.get("text", "").strip()
            
            if existing_text:
                # Cell already has text from Stage 3 - use it!
                enriched_cell = cell.copy()
                cells_with_text += 1
            else:
                # No text in Stage 3, try to match OCR
                cell_bbox = cell.get("absolute_bbox") or cell.get("bbox", [0, 0, 0, 0])
                cell_text = self._find_text_in_cell(cell_bbox, table_text_blocks)
                
                enriched_cell = cell.copy()
                enriched_cell["text"] = cell_text
                
                if cell_text.strip():
                    cells_with_text += 1
            
            enriched_cells.append(enriched_cell)
        
        # Create enriched table
        enriched_table = table.copy()
        enriched_table["grid_cells"] = enriched_cells
        enriched_table["cells_with_text"] = cells_with_text
        
        return enriched_table
    
    def _find_text_in_cell(self, cell_bbox: List[float], text_blocks: List[Dict]) -> str:
        """Find all OCR text that belongs to a cell"""
        texts = []
        
        for text_block in text_blocks:
            text_bbox = text_block.get("bbox", [])
            if not text_bbox or len(text_bbox) < 4:
                continue
            
            # Check if text block overlaps with cell
            overlap = self._calculate_overlap(cell_bbox, text_bbox)
            
            if overlap > 0.3:  # 30% overlap threshold
                text = text_block.get("text", "").strip()
                if text:
                    texts.append(text)
        
        return " ".join(texts)
    
    def _calculate_overlap(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU (Intersection over Union)"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    # ==================== OUTPUT GENERATION ====================
    
    def _generate_html(self, document_structure: Dict) -> str:
        """Generate HTML document"""
        metadata = document_structure.get("metadata", {})
        elements = document_structure.get("elements", [])
        
        html_parts = [
            '<!DOCTYPE html>',
            '<html>',
            '<head>',
            '  <meta charset="UTF-8">',
            '  <title>Reconstructed Document</title>',
            '  <style>',
            '    body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }',
            '    table { border-collapse: collapse; margin: 20px 0; width: 100%; table-layout: fixed; }',
            '    th, td { ',
            '      border: 1px solid #ddd; ',
            '      padding: 8px; ',
            '      text-align: center; ',
            '      word-wrap: break-word; ',
            '      overflow-wrap: break-word; ',
            '      vertical-align: middle;',
            '      max-width: 300px;',
            '    }',
            '    th { background-color: #f5f5f5; font-weight: bold; }',
            '    h1 { color: #333; }',
            '    h2 { color: #666; margin-top: 30px; }',
            '    .text-block { margin: 10px 0; padding: 8px; background-color: #f9f9f9; border-left: 3px solid #007acc; }',
            '    .metadata { color: #666; font-size: 0.9em; margin-bottom: 20px; }',
            '  </style>',
            '</head>',
            '<body>',
            '  <h1>Reconstructed Document</h1>',
            f'  <div class="metadata">',
            f'    <p><em>Generated: {metadata.get("generated", "")}</em></p>',
            f'    <p>Total Elements: {metadata.get("total_elements", 0)} | ',
            f'Tables: {metadata.get("total_tables", 0)} | ',
            f'Text Blocks: {metadata.get("total_text_blocks", 0)}</p>',
            f'  </div>',
        ]
        
        table_counter = 1
        
        for element in elements:
            element_type = element.get("type")
            content = element.get("content", {})
            
            if element_type == "table":
                # Render table
                grid_structure = content.get("grid_structure", {})
                rows = grid_structure.get("rows", 0)
                cols = grid_structure.get("cols", 0)
                
                html_parts.append(f'  <h2>Table {table_counter}</h2>')
                html_parts.append(f'  <p><em>{rows} rows × {cols} columns</em></p>')
                
                table_html = self._generate_html_table(content)
                html_parts.append(table_html)
                
                table_counter += 1
                
            elif element_type == "paragraph":
                # Render paragraph (full text block)
                text = content.get("text", "").strip()
                if text:
                    html_parts.append(f'  <p>{text}</p>')
                    
            elif element_type == "section-header":
                # Render as header
                text = content.get("text", "").strip()
                if text:
                    html_parts.append(f'  <h3>{text}</h3>')
                    
            elif element_type in ["title", "heading"]:
                # Render as main header
                text = content.get("text", "").strip()
                if text:
                    html_parts.append(f'  <h2>{text}</h2>')
                    
            else:
                # Render other text blocks
                text = content.get("text", "").strip()
                if text:
                    html_parts.append(f'  <div class="text-block">{text}</div>')
        
        html_parts.extend([
            '</body>',
            '</html>'
        ])
        
        return '\n'.join(html_parts)
    
    def _generate_html_table(self, table: Dict) -> str:
        """Generate HTML for a single table"""
        grid_cells = table.get("grid_cells", [])
        grid_structure = table.get("grid_structure", {})
        rows = grid_structure.get("rows", 0)
        cols = grid_structure.get("cols", 0)
        
        if rows == 0 or cols == 0:
            return '  <p><em>Empty table</em></p>'
        
        # Create grid matrix
        grid_matrix = {}
        occupied = set()
        
        for cell in grid_cells:
            row = cell.get("row", 0)
            col = cell.get("col", 0)
            row_span = cell.get("row_span", 1)
            col_span = cell.get("col_span", 1)
            
            grid_matrix[(row, col)] = cell
            
            # Mark occupied cells
            for r in range(row, min(row + row_span, rows)):
                for c in range(col, min(col + col_span, cols)):
                    if r != row or c != col:
                        occupied.add((r, c))
        
        # Generate HTML
        html_lines = ['  <table>']
        
        for row in range(rows):
            html_lines.append('    <tr>')
            
            for col in range(cols):
                if (row, col) in occupied:
                    continue  # Skip cells covered by rowspan/colspan
                
                if (row, col) in grid_matrix:
                    cell = grid_matrix[(row, col)]
                    text = cell.get("text", "").strip()
                    row_span = cell.get("row_span", 1)
                    col_span = cell.get("col_span", 1)
                    is_header = cell.get("is_header", False) or row == 0
                    
                    tag = "th" if is_header else "td"
                    attrs = []
                    
                    if row_span > 1:
                        attrs.append(f'rowspan="{row_span}"')
                    if col_span > 1:
                        attrs.append(f'colspan="{col_span}"')
                    
                    # Add styling for better text rendering
                    style = 'word-wrap: break-word; overflow: hidden; max-width: 300px;'
                    if is_header:
                        style += ' font-weight: bold;'
                    
                    attrs.append(f'style="{style}"')
                    attrs_str = " " + " ".join(attrs) if attrs else ""
                    html_lines.append(f'      <{tag}{attrs_str}>{text or "&nbsp;"}</{tag}>')
                else:
                    html_lines.append('      <td>&nbsp;</td>')
            
            html_lines.append('    </tr>')
        
        html_lines.append('  </table>')
        return '\n'.join(html_lines)
    
    def _generate_markdown(self, document_structure: Dict) -> str:
        """Generate Markdown document"""
        metadata = document_structure.get("metadata", {})
        elements = document_structure.get("elements", [])
        
        md_lines = [
            '# Reconstructed Document',
            '',
            f'*Generated: {metadata.get("generated", "")}*',
            '',
            f'**Total Elements:** {metadata.get("total_elements", 0)} | ',
            f'**Tables:** {metadata.get("total_tables", 0)} | ',
            f'**Text Blocks:** {metadata.get("total_text_blocks", 0)}',
            '',
            '---',
            '',
        ]
        
        table_counter = 1
        
        for element in elements:
            element_type = element.get("type")
            content = element.get("content", {})
            
            if element_type == "table":
                # Render table
                grid_structure = content.get("grid_structure", {})
                rows = grid_structure.get("rows", 0)
                cols = grid_structure.get("cols", 0)
                
                md_lines.append(f'## Table {table_counter}')
                md_lines.append(f'*{rows} rows × {cols} columns*')
                md_lines.append('')
                
                table_md = self._generate_markdown_table(content)
                md_lines.extend(table_md)
                md_lines.append('')
                
                table_counter += 1
                
            elif element_type == "paragraph":
                # Render paragraph (full text)
                text = content.get("text", "").strip()
                if text:
                    md_lines.append(text)
                    md_lines.append('')
                    
            elif element_type == "section-header":
                # Render as header
                text = content.get("text", "").strip()
                if text:
                    md_lines.append(f'### {text}')
                    md_lines.append('')
                    
            elif element_type in ["title", "heading"]:
                # Render as main header
                text = content.get("text", "").strip()
                if text:
                    md_lines.append(f'## {text}')
                    md_lines.append('')
                    
            else:
                # Render other text blocks
                text = content.get("text", "").strip()
                if text:
                    md_lines.append(text)
                    md_lines.append('')
        
        return '\n'.join(md_lines)
    
    def _generate_markdown_table(self, table: Dict) -> List[str]:
        """Generate Markdown for a single table"""
        grid_cells = table.get("grid_cells", [])
        grid_structure = table.get("grid_structure", {})
        rows = grid_structure.get("rows", 0)
        cols = grid_structure.get("cols", 0)
        
        if rows == 0 or cols == 0:
            return ['*Empty table*']
        
        # Create simple matrix (Markdown doesn't support rowspan/colspan well)
        md_matrix = [["" for _ in range(cols)] for _ in range(rows)]
        
        for cell in grid_cells:
            row = cell.get("row", 0)
            col = cell.get("col", 0)
            text = cell.get("text", "").strip()
            row_span = cell.get("row_span", 1)
            col_span = cell.get("col_span", 1)
            
            # Clean text for markdown
            text = text.replace("|", "\\|").replace("\n", " ")
            
            # Add span info if needed
            if row_span > 1 or col_span > 1:
                text += f" [span:{row_span}×{col_span}]"
            
            if 0 <= row < rows and 0 <= col < cols:
                md_matrix[row][col] = text
        
        # Generate markdown
        md_lines = []
        
        if rows > 0:
            # Header row
            header = "| " + " | ".join(md_matrix[0]) + " |"
            md_lines.append(header)
            
            # Separator
            separator = "| " + " | ".join(["---"] * cols) + " |"
            md_lines.append(separator)
            
            # Data rows
            for row_idx in range(1, rows):
                row_text = "| " + " | ".join(md_matrix[row_idx]) + " |"
                md_lines.append(row_text)
        
        return md_lines
    
    def _generate_json(self, document_structure: Dict) -> Dict:
        """Generate JSON document"""
        metadata = document_structure.get("metadata", {})
        elements = document_structure.get("elements", [])
        
        json_elements = []
        
        for element in elements:
            element_type = element.get("type")
            content = element.get("content", {})
            
            if element_type == "table":
                json_elements.append({
                    "type": "table",
                    "table_id": content.get("table_id", ""),
                    "dimensions": {
                        "rows": content.get("grid_structure", {}).get("rows", 0),
                        "cols": content.get("grid_structure", {}).get("cols", 0)
                    },
                    "cells": [
                        {
                            "row": cell.get("row"),
                            "col": cell.get("col"),
                            "text": cell.get("text", ""),
                            "rowspan": cell.get("row_span", 1),
                            "colspan": cell.get("col_span", 1),
                            "is_header": cell.get("is_header", False),
                            "bbox": cell.get("absolute_bbox", cell.get("bbox", []))
                        }
                        for cell in content.get("grid_cells", [])
                    ]
                })
            elif element_type == "paragraph":
                json_elements.append({
                    "type": "paragraph",
                    "text": content.get("text", ""),
                    "bbox": content.get("bbox", []),
                    "confidence": content.get("confidence", 0),
                    "granularity": "paragraph"
                })
            else:
                # Headers, captions, etc.
                json_elements.append({
                    "type": element_type,
                    "text": content.get("text", ""),
                    "bbox": content.get("bbox", []),
                    "confidence": content.get("confidence", 0),
                    "granularity": content.get("granularity", "single")
                })
        
        return {
            "metadata": metadata,
            "elements": json_elements
        }
    
    # ==================== FILE OUTPUT ====================
    
    def _save_outputs(self, output_prefix: str, intermediate_dir: Path,
                     html_content: str, markdown_content: str, json_content: Dict) -> Dict:
        """Save all output files"""
        output_dir = intermediate_dir / "stage4_file_reconstruction" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_files = {}
        
        # HTML
        html_file = output_dir / f"{output_prefix}_reconstruction.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        output_files['html'] = str(html_file)
        
        # Markdown
        md_file = output_dir / f"{output_prefix}_reconstruction.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        output_files['markdown'] = str(md_file)
        
        # JSON
        json_file = output_dir / f"{output_prefix}_reconstruction.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_content, f, indent=2, ensure_ascii=False)
        output_files['json'] = str(json_file)
        
        print(f"    💾 Saved {len(output_files)} output files")
        return output_files
    
    def _save_coordinates(self, output_prefix: str, intermediate_dir: Path,
                         document_structure: Dict) -> Path:
        """Save coordinates JSON"""
        coords_dir = intermediate_dir / "stage4_file_reconstruction" / "coordinates"
        coords_dir.mkdir(parents=True, exist_ok=True)
        
        coords_file = coords_dir / f"{output_prefix}_stage4_coordinates.json"
        
        coords_data = {
            "stage": 4,
            "stage_name": "file_reconstruction",
            "processing_timestamp": datetime.now().isoformat(),
            "document_structure": document_structure
        }
        
        with open(coords_file, 'w', encoding='utf-8') as f:
            json.dump(coords_data, f, indent=2, ensure_ascii=False)
        
        return coords_file
    
    # ==================== VISUALIZATION ====================
    
    def _create_visualization(self, input_file: str, output_prefix: str,
                            intermediate_dir: Path, document_structure: Dict) -> Path:
        """Create visualization of complete document reconstruction"""
        viz_dir = intermediate_dir / "stage4_file_reconstruction" / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        viz_file = viz_dir / f"{output_prefix}_reconstruction.png"
        
        try:
            # Load original image to get dimensions
            original_img = Image.open(input_file).convert('RGB')
            img_width, img_height = original_img.size
            
            # Create a BLANK WHITE CANVAS for clean reconstruction
            img = Image.new('RGB', (img_width, img_height), 'white')
            draw = ImageDraw.Draw(img)
            
            print(f"    📐 Canvas size: {img_width} × {img_height} (blank)")
            
            # Try to load font
            try:
                font = ImageFont.truetype("arial.ttf", 16)
                small_font = ImageFont.truetype("arial.ttf", 12)
            except:
                try:
                    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
                    small_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 12)
                except:
                    font = ImageFont.load_default()
                    small_font = ImageFont.load_default()
            
            # Draw all elements on blank canvas
            elements = document_structure.get("elements", [])
            
            for element in elements:
                element_type = element.get("type")
                content = element.get("content", {})
                
                if element_type == "table":
                    self._draw_table_on_image(draw, content, font)
                elif element_type == "paragraph":
                    self._draw_paragraph_on_image(draw, content, font)
                elif element_type in ["section-header", "title", "heading"]:
                    self._draw_header_on_image(draw, content, font)
                else:
                    self._draw_text_on_image(draw, content, small_font)
            
            img.save(viz_file)
            print(f"    🎨 Visualization saved: {viz_file}")
            
        except Exception as e:
            print(f"    ⚠️  Visualization error: {e}")
            import traceback
            traceback.print_exc()
            # Create placeholder
            placeholder = Image.new('RGB', (800, 600), 'lightgray')
            placeholder.save(viz_file)
        
        return viz_file
    
    def _draw_table_on_image(self, draw: ImageDraw.Draw, table: Dict, font):
        """Draw table with intelligent grid lines that respect cell spanning"""
        grid_cells = table.get("grid_cells", [])
        grid_structure = table.get("grid_structure", {})
        rows = grid_structure.get("rows", 0)
        cols = grid_structure.get("cols", 0)
        
        if rows == 0 or cols == 0:
            return
        
        # Step 1: Create cell matrix and track spanning
        cell_matrix = {}
        for cell in grid_cells:
            row = cell.get("row", 0)
            col = cell.get("col", 0)
            cell_matrix[(row, col)] = cell
        
        # Step 2: Draw cell backgrounds and fills first
        for cell in grid_cells:
            bbox = cell.get("absolute_bbox") or cell.get("bbox")
            if not bbox:
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            is_header = cell.get("is_header", False) or cell.get("row", 0) == 0
            
            # Fill color
            fill_color = (230, 240, 255) if not is_header else (220, 230, 250)
            draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=None)
        
        # Step 3: Draw intelligent grid lines
        self._draw_smart_grid_lines(draw, grid_cells, cell_matrix, rows, cols)
        
        # Step 4: Draw text in cells
        for cell in grid_cells:
            bbox = cell.get("absolute_bbox") or cell.get("bbox")
            text = cell.get("text", "").strip()
            is_header = cell.get("is_header", False) or cell.get("row", 0) == 0
            
            if text and bbox:
                self._draw_text_in_cell(draw, text, bbox, is_header)
    
    def _draw_smart_grid_lines(self, draw: ImageDraw.Draw, grid_cells: List[Dict], 
                               cell_matrix: Dict, rows: int, cols: int):
        """Draw grid lines that intelligently avoid spanning cells"""
        
        line_color = (0, 100, 200)
        line_width = 2
        
        # Collect all unique horizontal and vertical line positions
        horizontal_lines = set()  # y-coordinates
        vertical_lines = set()    # x-coordinates
        
        for cell in grid_cells:
            bbox = cell.get("absolute_bbox") or cell.get("bbox")
            if not bbox:
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            horizontal_lines.add(y1)
            horizontal_lines.add(y2)
            vertical_lines.add(x1)
            vertical_lines.add(x2)
        
        # Convert to sorted lists
        h_lines = sorted(horizontal_lines)
        v_lines = sorted(vertical_lines)
        
        # Draw horizontal lines (row separators)
        for y in h_lines:
            # For each horizontal line, draw segments that don't cross spanning cells
            segments = self._get_line_segments_horizontal(y, v_lines, grid_cells, cell_matrix, rows, cols)
            for x_start, x_end in segments:
                draw.line([(x_start, y), (x_end, y)], fill=line_color, width=line_width)
        
        # Draw vertical lines (column separators)
        for x in v_lines:
            # For each vertical line, draw segments that don't cross spanning cells
            segments = self._get_line_segments_vertical(x, h_lines, grid_cells, cell_matrix, rows, cols)
            for y_start, y_end in segments:
                draw.line([(x, y_start), (x, y_end)], fill=line_color, width=line_width)
    
    def _get_line_segments_horizontal(self, y: int, v_lines: List[int], 
                                      grid_cells: List[Dict], cell_matrix: Dict, 
                                      rows: int, cols: int) -> List[Tuple[int, int]]:
        """Get horizontal line segments avoiding cells where y is in the middle of a rowspan"""
        if not v_lines:
            return []
        
        # Find cells that have this y as a boundary
        cells_at_y = []
        for cell in grid_cells:
            bbox = cell.get("absolute_bbox") or cell.get("bbox")
            if not bbox:
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # This y-coordinate is a boundary of this cell
            if y == y1 or y == y2:
                cells_at_y.append({
                    'cell': cell,
                    'x1': x1,
                    'x2': x2,
                    'y1': y1,
                    'y2': y2,
                    'is_top': (y == y1),
                    'is_bottom': (y == y2)
                })
        
        if not cells_at_y:
            return []
        
        # Sort by x position
        cells_at_y.sort(key=lambda c: c['x1'])
        
        # Build line segments from cell boundaries
        segments = []
        for cell_info in cells_at_y:
            x1, x2 = cell_info['x1'], cell_info['x2']
            segments.append((x1, x2))
        
        # Merge overlapping segments
        merged_segments = self._merge_segments(segments)
        
        return merged_segments
    
    def _get_line_segments_vertical(self, x: int, h_lines: List[int], 
                                    grid_cells: List[Dict], cell_matrix: Dict, 
                                    rows: int, cols: int) -> List[Tuple[int, int]]:
        """Get vertical line segments avoiding cells where x is in the middle of a colspan"""
        if not h_lines:
            return []
        
        # Find cells that have this x as a boundary
        cells_at_x = []
        for cell in grid_cells:
            bbox = cell.get("absolute_bbox") or cell.get("bbox")
            if not bbox:
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # This x-coordinate is a boundary of this cell
            if x == x1 or x == x2:
                cells_at_x.append({
                    'cell': cell,
                    'x1': x1,
                    'x2': x2,
                    'y1': y1,
                    'y2': y2,
                    'is_left': (x == x1),
                    'is_right': (x == x2)
                })
        
        if not cells_at_x:
            return []
        
        # Sort by y position
        cells_at_x.sort(key=lambda c: c['y1'])
        
        # Build line segments from cell boundaries
        segments = []
        for cell_info in cells_at_x:
            y1, y2 = cell_info['y1'], cell_info['y2']
            segments.append((y1, y2))
        
        # Merge overlapping segments
        merged_segments = self._merge_segments(segments)
        
        return merged_segments
    
    def _merge_segments(self, segments: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merge overlapping or adjacent segments"""
        if not segments:
            return []
        
        # Sort segments by start position
        sorted_segments = sorted(segments)
        merged = [sorted_segments[0]]
        
        for current in sorted_segments[1:]:
            last = merged[-1]
            
            # If current overlaps or is adjacent to last, merge them
            if current[0] <= last[1]:
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)
        
        return merged
    
    def _draw_text_in_cell(self, draw: ImageDraw.Draw, text: str, bbox: List[float], is_header: bool = False):
        """Draw text strictly within cell boundaries with adaptive font sizing"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Cell dimensions
        cell_width = x2 - x1
        cell_height = y2 - y1
        
        # Padding inside cell (5px on each side)
        padding = 5
        available_width = cell_width - (2 * padding)
        available_height = cell_height - (2 * padding)
        
        if available_width <= 0 or available_height <= 0:
            return
        
        # Find optimal font size that fits text in cell
        font_size = self._find_optimal_font_size(
            draw, text, available_width, available_height, is_header
        )
        
        # Load font at optimal size
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
        
        # Word wrap text to fit width
        wrapped_lines = self._wrap_text_to_width(draw, text, available_width, font)
        
        # Calculate total text height
        line_height = font_size + 2
        total_text_height = len(wrapped_lines) * line_height
        
        # If text is too tall, truncate
        if total_text_height > available_height:
            max_lines = max(1, int(available_height / line_height))
            wrapped_lines = wrapped_lines[:max_lines]
            if len(wrapped_lines) > 0:
                wrapped_lines[-1] = wrapped_lines[-1][:max(0, len(wrapped_lines[-1])-3)] + "..."
        
        # Center text vertically in cell
        text_y = y1 + padding + (available_height - len(wrapped_lines) * line_height) // 2
        
        # Draw each line (centered horizontally)
        for line in wrapped_lines:
            # Get text width for horizontal centering
            try:
                text_bbox = draw.textbbox((0, 0), line, font=font)
                text_width = text_bbox[2] - text_bbox[0]
            except:
                text_width = len(line) * (font_size * 0.6)
            
            # Center horizontally in cell
            text_x = x1 + padding + (available_width - text_width) // 2
            
            # Ensure text doesn't go outside cell bounds
            text_x = max(x1 + padding, text_x)
            text_y = max(y1 + padding, text_y)
            
            # Draw text
            draw.text((text_x, text_y), line, fill=(0, 0, 0), font=font)
            text_y += line_height
    
    def _find_optimal_font_size(self, draw: ImageDraw.Draw, text: str, 
                                max_width: int, max_height: int, is_header: bool) -> int:
        """Find the largest font size that fits text in given dimensions"""
        # Start with larger font for headers
        if is_header:
            font_sizes = [20, 18, 16, 14, 12, 10, 8]
        else:
            font_sizes = [16, 14, 12, 10, 8, 6]
        
        for size in font_sizes:
            try:
                test_font = ImageFont.truetype("arial.ttf", size)
            except:
                try:
                    test_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
                except:
                    test_font = ImageFont.load_default()
                    return 10
            
            # Test if text fits
            wrapped = self._wrap_text_to_width(draw, text, max_width, test_font)
            line_height = size + 2
            total_height = len(wrapped) * line_height
            
            if total_height <= max_height:
                return size
        
        # Fallback to smallest
        return 6
    
    def _wrap_text_to_width(self, draw: ImageDraw.Draw, text: str, 
                           max_width: int, font) -> List[str]:
        """Wrap text to fit within given width"""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            
            try:
                bbox = draw.textbbox((0, 0), test_line, font=font)
                text_width = bbox[2] - bbox[0]
            except:
                # Fallback estimate
                text_width = len(test_line) * (font.size * 0.6 if hasattr(font, 'size') else 6)
            
            if text_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # Single word is too long, truncate it
                    lines.append(word[:max(1, max_width // 6)] + "...")
                    current_line = []
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines if lines else [text]
    
    def _draw_paragraph_on_image(self, draw: ImageDraw.Draw, paragraph: Dict, font):
        """Draw paragraph block on image"""
        bbox = paragraph.get("bbox", [])
        if not bbox or len(bbox) < 4:
            return
        
        x1, y1, x2, y2 = map(int, bbox)
        text = paragraph.get("text", "").strip()
        
        if not text:
            return
        
        # Color code: green for paragraphs
        border_color = (0, 180, 80)
        fill_color = (240, 255, 240)
        
        # Draw paragraph region with fill
        draw.rectangle([x1, y1, x2, y2], outline=border_color, fill=fill_color, width=2)
        
        # Draw text with same adaptive sizing as table cells
        self._draw_text_in_cell(draw, text, bbox, is_header=False)
    
    def _draw_header_on_image(self, draw: ImageDraw.Draw, header: Dict, font):
        """Draw header/title on image"""
        bbox = header.get("bbox", [])
        if not bbox or len(bbox) < 4:
            return
        
        x1, y1, x2, y2 = map(int, bbox)
        text = header.get("text", "").strip()
        
        if not text:
            return
        
        # Color code: magenta for headers
        border_color = (200, 0, 200)
        fill_color = (255, 240, 255)
        
        # Draw header region with fill
        draw.rectangle([x1, y1, x2, y2], outline=border_color, fill=fill_color, width=2)
        
        # Draw text with adaptive sizing (headers can be larger)
        self._draw_text_in_cell(draw, text, bbox, is_header=True)
    
    def _draw_text_on_image(self, draw: ImageDraw.Draw, text_block: Dict, font):
        """Draw other text block on image"""
        bbox = text_block.get("bbox", [])
        if not bbox or len(bbox) < 4:
            return
        
        x1, y1, x2, y2 = map(int, bbox)
        text = text_block.get("text", "").strip()
        
        if not text:
            return
        
        # Color code: gray for other text
        border_color = (128, 128, 128)
        fill_color = (245, 245, 245)
        
        # Draw text block with fill
        draw.rectangle([x1, y1, x2, y2], outline=border_color, fill=fill_color, width=1)
        
        # Draw text with adaptive sizing
        self._draw_text_in_cell(draw, text, bbox, is_header=False)


def main():
    """Main function for testing Stage 4"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage 4: Complete File Reconstruction")
    parser.add_argument("--input", "-i", required=True, help="Input image file")
    parser.add_argument("--output-prefix", "-o", help="Output prefix (default: input filename)")
    parser.add_argument("--intermediate-dir", default="intermediate_outputs", help="Intermediate outputs directory")
    
    args = parser.parse_args()
    
    if not args.output_prefix:
        args.output_prefix = Path(args.input).stem
    
    # Initialize reconstructor
    reconstructor = FileReconstructor()
    
    # Process image
    result = reconstructor.process_image(
        args.input,
        args.output_prefix,
        Path(args.intermediate_dir)
    )
    
    # Print results
    if result.get("success", False):
        print(f"\n✅ Stage 4 completed successfully!")
        print(f"   Tables: {result.get('total_tables', 0)}")
        print(f"   Output files: {result.get('output_files', {})}")
        print(f"   Visualization: {result.get('visualization_path', 'N/A')}")
    else:
        print(f"\n❌ Stage 4 failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
