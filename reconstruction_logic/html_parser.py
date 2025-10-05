"""
HTML Parser for PubTabNet Ground Truth Extraction

Parses HTML annotations from PubTabNet to extract cell coordinates,
spans, and content structure.
"""

import re
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from bs4 import BeautifulSoup


@dataclass
class CellInfo:
    """Information about a table cell"""
    row: int
    col: int
    rowspan: int
    colspan: int
    content: str
    bbox: Optional[List[float]] = None  # [x1, y1, x2, y2]
    is_header: bool = False
    confidence: float = 1.0


@dataclass
class TableStructure:
    """Complete table structure information"""
    rows: int
    cols: int
    cells: List[CellInfo]
    table_bbox: Optional[List[float]] = None
    html_structure: str = ""


class PubTabNetHTMLParser:
    """Parser for PubTabNet HTML annotations"""
    
    def __init__(self):
        self.cell_id_counter = 0
    
    def parse_html_annotation(self, html_string: str, image_width: int = None, image_height: int = None) -> TableStructure:
        """
        Parse HTML annotation from PubTabNet
        
        Args:
            html_string: HTML string containing table structure
            image_width: Width of the source image (for coordinate normalization)
            image_height: Height of the source image (for coordinate normalization)
            
        Returns:
            TableStructure object with parsed information
        """
        try:
            # Parse HTML using BeautifulSoup
            soup = BeautifulSoup(html_string, 'html.parser')
            
            # Find the table element
            table = soup.find('table')
            if not table:
                raise ValueError("No table found in HTML annotation")
            
            # Extract table structure
            cells = []
            max_row = 0
            max_col = 0
            
            # Process table rows
            rows = table.find_all('tr')
            for row_idx, row in enumerate(rows):
                cells_in_row = row.find_all(['td', 'th'])
                col_idx = 0
                
                for cell in cells_in_row:
                    # Skip cells that are already occupied by rowspan/colspan
                    while self._is_cell_occupied(cells, row_idx, col_idx):
                        col_idx += 1
                    
                    # Extract cell information
                    cell_info = self._extract_cell_info(cell, row_idx, col_idx)
                    cells.append(cell_info)
                    
                    # Update column index
                    col_idx += cell_info.colspan
                    max_col = max(max_col, col_idx)
                
                max_row = max(max_row, row_idx + 1)
            
            # Calculate table bounding box if image dimensions provided
            table_bbox = None
            if image_width and image_height and cells:
                table_bbox = self._calculate_table_bbox(cells, image_width, image_height)
            
            return TableStructure(
                rows=max_row,
                cols=max_col,
                cells=cells,
                table_bbox=table_bbox,
                html_structure=html_string
            )
            
        except Exception as e:
            raise ValueError(f"Error parsing HTML annotation: {str(e)}")
    
    def _extract_cell_info(self, cell_element, row: int, col: int) -> CellInfo:
        """Extract information from a single cell element"""
        # Get rowspan and colspan
        rowspan = int(cell_element.get('rowspan', 1))
        colspan = int(cell_element.get('colspan', 1))
        
        # Extract content
        content = cell_element.get_text(strip=True)
        
        # Determine if it's a header cell
        is_header = cell_element.name == 'th'
        
        return CellInfo(
            row=row,
            col=col,
            rowspan=rowspan,
            colspan=colspan,
            content=content,
            is_header=is_header
        )
    
    def _is_cell_occupied(self, cells: List[CellInfo], row: int, col: int) -> bool:
        """Check if a cell position is already occupied by a spanning cell"""
        for cell in cells:
            if (cell.row <= row < cell.row + cell.rowspan and 
                cell.col <= col < cell.col + cell.colspan):
                return True
        return False
    
    def _calculate_table_bbox(self, cells: List[CellInfo], image_width: int, image_height: int) -> List[float]:
        """Calculate table bounding box from cell positions"""
        if not cells:
            return [0, 0, image_width, image_height]
        
        # For now, return the full image dimensions
        # In a real implementation, you'd calculate based on cell positions
        return [0, 0, image_width, image_height]
    
    def validate_table_structure(self, table_structure: TableStructure) -> Dict[str, bool]:
        """Validate the parsed table structure"""
        validation_results = {
            'has_cells': len(table_structure.cells) > 0,
            'valid_dimensions': table_structure.rows > 0 and table_structure.cols > 0,
            'no_overlapping_cells': self._check_no_overlapping_cells(table_structure.cells),
            'consistent_spans': self._check_consistent_spans(table_structure.cells),
            'has_content': any(cell.content.strip() for cell in table_structure.cells)
        }
        
        return validation_results
    
    def _check_no_overlapping_cells(self, cells: List[CellInfo]) -> bool:
        """Check that no cells overlap"""
        for i, cell1 in enumerate(cells):
            for j, cell2 in enumerate(cells):
                if i >= j:
                    continue
                
                # Check for overlap
                if (cell1.row < cell2.row + cell2.rowspan and
                    cell1.row + cell1.rowspan > cell2.row and
                    cell1.col < cell2.col + cell2.colspan and
                    cell1.col + cell1.colspan > cell2.col):
                    return False
        
        return True
    
    def _check_consistent_spans(self, cells: List[CellInfo]) -> bool:
        """Check that row and column spans are consistent"""
        for cell in cells:
            if cell.rowspan <= 0 or cell.colspan <= 0:
                return False
        return True


def main():
    """Test the HTML parser with sample data"""
    parser = PubTabNetHTMLParser()
    
    # Sample HTML annotation
    sample_html = """
    <table>
        <tr>
            <th colspan="2">Header</th>
            <th>Column 3</th>
        </tr>
        <tr>
            <td rowspan="2">Cell 1</td>
            <td>Cell 2</td>
            <td>Cell 3</td>
        </tr>
        <tr>
            <td>Cell 4</td>
            <td>Cell 5</td>
        </tr>
    </table>
    """
    
    # Parse the HTML
    table_structure = parser.parse_html_annotation(sample_html)
    
    print(f"Table structure: {table_structure.rows} rows × {table_structure.cols} columns")
    print(f"Total cells: {len(table_structure.cells)}")
    
    # Validate structure
    validation = parser.validate_table_structure(table_structure)
    print(f"Validation results: {validation}")
    
    # Print cell details
    for cell in table_structure.cells:
        print(f"Cell R{cell.row}C{cell.col}: {cell.content} (span: {cell.rowspan}×{cell.colspan})")


if __name__ == "__main__":
    main()
